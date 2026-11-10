"""Phase 4: Drift-Aware Model Adaptation Under Limited Label Budgets.

Implements and evaluates four adaptation strategies under temporal distribution shift:
- Strategy A: Static Baseline (Control) — No retraining, frozen historical model.
- Strategy B: Periodic Retraining — Retrains monthly with accumulated budgeted labels.
- Strategy C: Incremental Online Learning — Updates SGDClassifier weights via partial_fit.
- Strategy D: Drift-Triggered Retraining — Retrains selectively only when PSI exceeds drift thresholds.

Evaluates performance across constrained label budgets (0%, 1%, 5%, 10%, 100%) and multiple random seeds.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.experiments.drift_analysis import calculate_psi
from src.features.url_features import FEATURE_COLUMNS
from src.models.evaluate import compute_metrics, save_predictions
from src.models.train_baseline import _estimator, _pipeline
from src.utils.config import append_event, load_config, path_of
from src.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)

REFERENCE_MONTHS = ["2024-08", "2024-09", "2024-10"]
STREAM_MONTHS = ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11"]

IN_DIST_F1 = {
    "xgboost": 0.90648,
    "random_forest": 0.90421,
    "logistic_regression": 0.83434,
    "sgd_classifier": 0.83386,
}

STATIC_TEMPORAL_F1 = {
    "xgboost": 0.83133,
    "random_forest": 0.82042,
    "logistic_regression": 0.77623,
    "sgd_classifier": 0.76959,
}


def sample_budget(month_df: pd.DataFrame, budget_frac: float, seed: int) -> pd.DataFrame:
    """Sample a fraction of monthly streaming data stratified by label."""
    if budget_frac <= 0.0:
        return month_df.iloc[0:0].copy()
    if budget_frac >= 1.0:
        return month_df.copy()

    sampled_indices = []
    for _, grp in month_df.groupby("label"):
        n_sample = max(1, int(np.round(len(grp) * budget_frac)))
        n_sample = min(n_sample, len(grp))
        sampled_indices.extend(grp.sample(n=n_sample, random_state=seed).index)

    return month_df.loc[sampled_indices].copy()


def run_adaptation_experiments(config: dict | None = None) -> dict:
    config = config or load_config()
    seed = int(config["seed"])
    set_seed(seed)
    results_dir = path_of(config, "results_dir")
    processed_dir = path_of(config, "processed_dir")

    out_base = results_dir / "adaptation"
    metrics_dir = out_base / "metrics"
    preds_dir = out_base / "predictions"
    graphs_dir = out_base / "graphs"
    for d in [metrics_dir, preds_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Phase 4: Drift-Aware Model Adaptation Experiments")
    append_event(config, "phase4_adaptation", "Started Phase 4 model adaptation experiments")

    df = pd.read_parquet(processed_dir / "phreshphish.parquet")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["month"] = df["date"].dt.strftime("%Y-%m")

    hist_df = df.loc[df["month"].isin(REFERENCE_MONTHS)].reset_index(drop=True)
    stream_df = df.loc[df["month"].isin(STREAM_MONTHS)].reset_index(drop=True)

    x_hist = hist_df[FEATURE_COLUMNS]
    y_hist = hist_df["label"].astype(int)

    LOGGER.info(
        "Historical Reference: N=%d (phish rate %.3f); Streaming Stream: N=%d (phish rate %.3f)",
        len(hist_df), y_hist.mean(), len(stream_df), stream_df["label"].mean()
    )

    models_config = config["models"]
    seeds = [42, 43, 44]
    budgets = [0.01, 0.05, 0.10, 1.00]

    all_results = []
    monthly_traces = []

    # -------------------------------------------------------------------------
    # 1. Strategy A: Static Baseline (Control, budget=0.0)
    # -------------------------------------------------------------------------
    LOGGER.info("--- Executing Strategy A: Static Control ---")
    for model_name in ["xgboost", "random_forest", "logistic_regression", "sgd_classifier"]:
        params = dict(models_config[model_name])
        scale = model_name in {"logistic_regression", "sgd_classifier"}
        est = _estimator(model_name, params, y_hist, seed)
        pipe = _pipeline(est, FEATURE_COLUMNS, [], scale, 0)
        pipe.fit(x_hist, y_hist)

        # Sequential monthly evaluation without retraining
        stream_preds = []
        stream_probs = []
        stream_trues = []

        for m in STREAM_MONTHS:
            m_df = stream_df.loc[stream_df["month"] == m]
            m_x = m_df[FEATURE_COLUMNS]
            m_y = m_df["label"].astype(int)

            prob = pipe.predict_proba(m_x)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m_metrics = compute_metrics(m_y, pred, prob)

            stream_preds.extend(pred)
            stream_probs.extend(prob)
            stream_trues.extend(m_y)

            monthly_traces.append({
                "strategy": "static",
                "model": model_name,
                "budget": 0.0,
                "seed": seed,
                "month": m,
                "f1": m_metrics["f1"],
                "accuracy": m_metrics["accuracy"],
                "false_positive_rate": m_metrics["false_positive_rate"],
                "retrained": False,
            })

        cum_metrics = compute_metrics(stream_trues, stream_preds, stream_probs)
        in_f1 = IN_DIST_F1[model_name]
        st_f1 = cum_metrics["f1"]
        rec = {
            "strategy": "static",
            "model": model_name,
            "budget": 0.0,
            "seed": seed,
            "n_updates": 0,
            "labels_consumed": 0,
            "adaptation_seconds": 0.0,
            "accuracy": cum_metrics["accuracy"],
            "precision": cum_metrics["precision"],
            "recall": cum_metrics["recall"],
            "f1": cum_metrics["f1"],
            "roc_auc": cum_metrics["roc_auc"],
            "false_positive_rate": cum_metrics["false_positive_rate"],
            "recovery_pct": 0.0,
            "label_efficiency": 0.0,
        }
        all_results.append(rec)
        save_predictions(
            preds_dir / f"static__{model_name}__b0.0__s{seed}.parquet",
            stream_trues, stream_preds, stream_probs
        )

    # -------------------------------------------------------------------------
    # 2. Strategy B: Periodic Retraining (Monthly Batch)
    # -------------------------------------------------------------------------
    LOGGER.info("--- Executing Strategy B: Periodic Monthly Retraining ---")
    for model_name in ["xgboost", "random_forest", "logistic_regression", "sgd_classifier"]:
        params = dict(models_config[model_name])
        scale = model_name in {"logistic_regression", "sgd_classifier"}

        for budget in budgets:
            eval_seeds = [seed] if budget == 1.0 else seeds
            for s in eval_seeds:
                # Initialize model with historical data
                accum_df = hist_df.copy()
                est = _estimator(model_name, params, accum_df["label"].astype(int), s)
                current_pipe = _pipeline(est, FEATURE_COLUMNS, [], scale, 0)
                current_pipe.fit(accum_df[FEATURE_COLUMNS], accum_df["label"].astype(int))

                stream_preds = []
                stream_probs = []
                stream_trues = []
                total_adapt_time = 0.0
                total_labels = 0
                n_updates = 0

                for m_idx, m in enumerate(STREAM_MONTHS):
                    m_df = stream_df.loc[stream_df["month"] == m]
                    m_x = m_df[FEATURE_COLUMNS]
                    m_y = m_df["label"].astype(int)

                    # 1. Forward evaluate on month m using model trained on prior data
                    prob = current_pipe.predict_proba(m_x)[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    m_metrics = compute_metrics(m_y, pred, prob)

                    stream_preds.extend(pred)
                    stream_probs.extend(prob)
                    stream_trues.extend(m_y)

                    # 2. Sample budgeted data from month m
                    budget_m = sample_budget(m_df, budget, s)
                    total_labels += len(budget_m)

                    # 3. Retrain model for future months (if not the last month)
                    retrained = False
                    if m_idx < len(STREAM_MONTHS) - 1:
                        accum_df = pd.concat([accum_df, budget_m], ignore_index=True)
                        t0 = time.perf_counter()
                        est_next = _estimator(model_name, params, accum_df["label"].astype(int), s)
                        current_pipe = _pipeline(est_next, FEATURE_COLUMNS, [], scale, 0)
                        current_pipe.fit(accum_df[FEATURE_COLUMNS], accum_df["label"].astype(int))
                        total_adapt_time += (time.perf_counter() - t0)
                        n_updates += 1
                        retrained = True

                    monthly_traces.append({
                        "strategy": "periodic",
                        "model": model_name,
                        "budget": budget,
                        "seed": s,
                        "month": m,
                        "f1": m_metrics["f1"],
                        "accuracy": m_metrics["accuracy"],
                        "false_positive_rate": m_metrics["false_positive_rate"],
                        "retrained": retrained,
                    })

                cum_metrics = compute_metrics(stream_trues, stream_preds, stream_probs)
                in_f1 = IN_DIST_F1[model_name]
                base_st_f1 = STATIC_TEMPORAL_F1[model_name]
                recovery = ((cum_metrics["f1"] - base_st_f1) / max(1e-5, (in_f1 - base_st_f1))) * 100.0

                rec = {
                    "strategy": "periodic",
                    "model": model_name,
                    "budget": budget,
                    "seed": s,
                    "n_updates": n_updates,
                    "labels_consumed": total_labels,
                    "adaptation_seconds": total_adapt_time,
                    "accuracy": cum_metrics["accuracy"],
                    "precision": cum_metrics["precision"],
                    "recall": cum_metrics["recall"],
                    "f1": cum_metrics["f1"],
                    "roc_auc": cum_metrics["roc_auc"],
                    "false_positive_rate": cum_metrics["false_positive_rate"],
                    "recovery_pct": recovery,
                    "label_efficiency": (cum_metrics["f1"] - base_st_f1) / max(1, total_labels) * 1000.0,
                }
                all_results.append(rec)
                if s == seed:
                    save_predictions(
                        preds_dir / f"periodic__{model_name}__b{budget}__s{s}.parquet",
                        stream_trues, stream_preds, stream_probs
                    )

    # -------------------------------------------------------------------------
    # 3. Strategy C: Online Incremental Learning (SGDClassifier)
    # -------------------------------------------------------------------------
    LOGGER.info("--- Executing Strategy C: Online Incremental Learning (SGD) ---")
    for budget in budgets:
        eval_seeds = [seed] if budget == 1.0 else seeds
        for s in eval_seeds:
            # Scaler fit strictly on historical data
            scaler = StandardScaler()
            x_hist_sc = scaler.fit_transform(x_hist)
            sgd = SGDClassifier(loss="log_loss", penalty="l2", random_state=s, max_iter=1000)
            sgd.fit(x_hist_sc, y_hist)

            stream_preds = []
            stream_probs = []
            stream_trues = []
            total_adapt_time = 0.0
            total_labels = 0
            n_updates = 0

            for m_idx, m in enumerate(STREAM_MONTHS):
                m_df = stream_df.loc[stream_df["month"] == m]
                m_x = scaler.transform(m_df[FEATURE_COLUMNS])
                m_y = m_df["label"].astype(int)

                prob = sgd.predict_proba(m_x)[:, 1]
                pred = (prob >= 0.5).astype(int)
                m_metrics = compute_metrics(m_y, pred, prob)

                stream_preds.extend(pred)
                stream_probs.extend(prob)
                stream_trues.extend(m_y)

                # Budgeted update
                budget_m = sample_budget(m_df, budget, s)
                total_labels += len(budget_m)

                retrained = False
                if m_idx < len(STREAM_MONTHS) - 1 and len(budget_m) > 0:
                    b_x = scaler.transform(budget_m[FEATURE_COLUMNS])
                    b_y = budget_m["label"].astype(int).to_numpy()
                    t0 = time.perf_counter()
                    sgd.partial_fit(b_x, b_y, classes=[0, 1])
                    total_adapt_time += (time.perf_counter() - t0)
                    n_updates += 1
                    retrained = True

                monthly_traces.append({
                    "strategy": "online_incremental",
                    "model": "sgd_classifier",
                    "budget": budget,
                    "seed": s,
                    "month": m,
                    "f1": m_metrics["f1"],
                    "accuracy": m_metrics["accuracy"],
                    "false_positive_rate": m_metrics["false_positive_rate"],
                    "retrained": retrained,
                })

            cum_metrics = compute_metrics(stream_trues, stream_preds, stream_probs)
            in_f1 = IN_DIST_F1["sgd_classifier"]
            base_st_f1 = STATIC_TEMPORAL_F1["sgd_classifier"]
            recovery = ((cum_metrics["f1"] - base_st_f1) / max(1e-5, (in_f1 - base_st_f1))) * 100.0

            rec = {
                "strategy": "online_incremental",
                "model": "sgd_classifier",
                "budget": budget,
                "seed": s,
                "n_updates": n_updates,
                "labels_consumed": total_labels,
                "adaptation_seconds": total_adapt_time,
                "accuracy": cum_metrics["accuracy"],
                "precision": cum_metrics["precision"],
                "recall": cum_metrics["recall"],
                "f1": cum_metrics["f1"],
                "roc_auc": cum_metrics["roc_auc"],
                "false_positive_rate": cum_metrics["false_positive_rate"],
                "recovery_pct": recovery,
                "label_efficiency": (cum_metrics["f1"] - base_st_f1) / max(1, total_labels) * 1000.0,
            }
            all_results.append(rec)
            if s == seed:
                save_predictions(
                    preds_dir / f"online_incremental__sgd_classifier__b{budget}__s{s}.parquet",
                    stream_trues, stream_preds, stream_probs
                )

    # -------------------------------------------------------------------------
    # 4. Strategy D: Drift-Triggered Retraining
    # -------------------------------------------------------------------------
    LOGGER.info("--- Executing Strategy D: Drift-Triggered Retraining ---")
    # Drift rule: Retrain only when mean PSI >= 0.10 or max feature PSI >= 0.25
    for model_name in ["xgboost", "random_forest", "logistic_regression"]:
        params = dict(models_config[model_name])
        scale = model_name in {"logistic_regression", "sgd_classifier"}

        for budget in budgets:
            eval_seeds = [seed] if budget == 1.0 else seeds
            for s in eval_seeds:
                accum_df = hist_df.copy()
                unconsumed_budget = []
                est = _estimator(model_name, params, accum_df["label"].astype(int), s)
                current_pipe = _pipeline(est, FEATURE_COLUMNS, [], scale, 0)
                current_pipe.fit(accum_df[FEATURE_COLUMNS], accum_df["label"].astype(int))

                stream_preds = []
                stream_probs = []
                stream_trues = []
                total_adapt_time = 0.0
                total_labels = 0
                n_updates = 0

                for m_idx, m in enumerate(STREAM_MONTHS):
                    m_df = stream_df.loc[stream_df["month"] == m]
                    m_x = m_df[FEATURE_COLUMNS]
                    m_y = m_df["label"].astype(int)

                    prob = current_pipe.predict_proba(m_x)[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    m_metrics = compute_metrics(m_y, pred, prob)

                    stream_preds.extend(pred)
                    stream_probs.extend(prob)
                    stream_trues.extend(m_y)

                    # Sample budget for month m
                    budget_m = sample_budget(m_df, budget, s)
                    total_labels += len(budget_m)
                    unconsumed_budget.append(budget_m)

                    # Compute drift vs historical reference
                    psis = [
                        calculate_psi(hist_df[col].to_numpy(dtype=float), m_df[col].to_numpy(dtype=float))
                        for col in FEATURE_COLUMNS
                    ]
                    mean_psi = float(np.mean(psis))
                    max_psi = float(np.max(psis))
                    is_drift_triggered = (mean_psi >= 0.10) or (max_psi >= 0.25)

                    retrained = False
                    if m_idx < len(STREAM_MONTHS) - 1 and is_drift_triggered:
                        # Append all accumulated unconsumed budget batches
                        for b_batch in unconsumed_budget:
                            accum_df = pd.concat([accum_df, b_batch], ignore_index=True)
                        unconsumed_budget = []

                        t0 = time.perf_counter()
                        est_next = _estimator(model_name, params, accum_df["label"].astype(int), s)
                        current_pipe = _pipeline(est_next, FEATURE_COLUMNS, [], scale, 0)
                        current_pipe.fit(accum_df[FEATURE_COLUMNS], accum_df["label"].astype(int))
                        total_adapt_time += (time.perf_counter() - t0)
                        n_updates += 1
                        retrained = True

                    monthly_traces.append({
                        "strategy": "drift_triggered",
                        "model": model_name,
                        "budget": budget,
                        "seed": s,
                        "month": m,
                        "f1": m_metrics["f1"],
                        "accuracy": m_metrics["accuracy"],
                        "false_positive_rate": m_metrics["false_positive_rate"],
                        "retrained": retrained,
                        "mean_psi": mean_psi,
                        "max_psi": max_psi,
                    })

                cum_metrics = compute_metrics(stream_trues, stream_preds, stream_probs)
                in_f1 = IN_DIST_F1[model_name]
                base_st_f1 = STATIC_TEMPORAL_F1[model_name]
                recovery = ((cum_metrics["f1"] - base_st_f1) / max(1e-5, (in_f1 - base_st_f1))) * 100.0

                rec = {
                    "strategy": "drift_triggered",
                    "model": model_name,
                    "budget": budget,
                    "seed": s,
                    "n_updates": n_updates,
                    "labels_consumed": total_labels,
                    "adaptation_seconds": total_adapt_time,
                    "accuracy": cum_metrics["accuracy"],
                    "precision": cum_metrics["precision"],
                    "recall": cum_metrics["recall"],
                    "f1": cum_metrics["f1"],
                    "roc_auc": cum_metrics["roc_auc"],
                    "false_positive_rate": cum_metrics["false_positive_rate"],
                    "recovery_pct": recovery,
                    "label_efficiency": (cum_metrics["f1"] - base_st_f1) / max(1, total_labels) * 1000.0,
                }
                all_results.append(rec)
                if s == seed:
                    save_predictions(
                        preds_dir / f"drift_triggered__{model_name}__b{budget}__s{s}.parquet",
                        stream_trues, stream_preds, stream_probs
                    )

    # -------------------------------------------------------------------------
    # 5. Compile and Summarize Results
    # -------------------------------------------------------------------------
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(metrics_dir / "adaptation_comparison.csv", index=False)

    traces_df = pd.DataFrame(monthly_traces)
    traces_df.to_csv(metrics_dir / "adaptation_monthly_trajectories.csv", index=False)

    # Aggregate over seeds to report mean +/- std
    agg_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "recovery_pct", "n_updates", "labels_consumed", "adaptation_seconds"]
    summary_grouped = (
        res_df.groupby(["strategy", "model", "budget"])
        .agg({c: ["mean", "std"] for c in agg_cols})
        .reset_index()
    )
    # Flatten column multi-index
    summary_grouped.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in summary_grouped.columns
    ]
    summary_grouped.to_csv(metrics_dir / "adaptation_budget_summary.csv", index=False)

    payload = {
        "reference_period": f"{REFERENCE_MONTHS[0]} to {REFERENCE_MONTHS[-1]}",
        "streaming_period": f"{STREAM_MONTHS[0]} to {STREAM_MONTHS[-1]}",
        "total_stream_samples": len(stream_df),
        "total_experiments_evaluated": len(res_df),
        "summary": summary_grouped.to_dict(orient="records"),
    }
    (metrics_dir / "adaptation_summary.json").write_text(json.dumps(payload, indent=2))

    # -------------------------------------------------------------------------
    # 6. Generate Visualizations
    # -------------------------------------------------------------------------
    LOGGER.info("Generating Phase 4 Visualizations...")
    _plot_budget_recovery_curves(graphs_dir, res_df)
    _plot_adaptation_strategy_comparison(graphs_dir, res_df)
    _plot_drift_triggered_vs_periodic(graphs_dir, res_df)
    _plot_monthly_trajectories(graphs_dir, traces_df)

    append_event(config, "phase4_adaptation", "Completed Phase 4 model adaptation experiments", total_runs=len(res_df))
    LOGGER.info("Phase 4 Adaptation Experiments Completed Successfully (Total Runs: %d)", len(res_df))
    return payload


def _plot_budget_recovery_curves(graphs_dir: Path, df: pd.DataFrame) -> None:
    """Plot F1 score and Recovery % as a function of label budget for XGBoost and other models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Filter for XGBoost across strategies
    xgb_df = df[df["model"] == "xgboost"]
    for strat, color, marker in [("periodic", "#2b5c8f", "o"), ("drift_triggered", "#d95f02", "s")]:
        sub = xgb_df[xgb_df["strategy"] == strat].groupby("budget").agg({"f1": "mean", "recovery_pct": "mean"}).reset_index()
        # Add budget 0 from static
        stat_f1 = df[(df["model"] == "xgboost") & (df["strategy"] == "static")]["f1"].mean()
        budget_vals = [0.0] + list(sub["budget"])
        f1_vals = [stat_f1] + list(sub["f1"])
        rec_vals = [0.0] + list(sub["recovery_pct"])

        axes[0].plot(budget_vals, f1_vals, marker=marker, color=color, linewidth=2, label=f"XGBoost ({strat.replace('_', ' ').title()})")
        axes[1].plot(budget_vals, rec_vals, marker=marker, color=color, linewidth=2, label=f"XGBoost ({strat.replace('_', ' ').title()})")

    # Also plot Online SGD
    sgd_df = df[df["strategy"] == "online_incremental"].groupby("budget").agg({"f1": "mean", "recovery_pct": "mean"}).reset_index()
    stat_sgd_f1 = df[(df["model"] == "sgd_classifier") & (df["strategy"] == "static")]["f1"].mean()
    axes[0].plot([0.0] + list(sgd_df["budget"]), [stat_sgd_f1] + list(sgd_df["f1"]), marker="^", color="#7570b3", linestyle="--", label="SGD (Online Incremental)")
    axes[1].plot([0.0] + list(sgd_df["budget"]), [0.0] + list(sgd_df["recovery_pct"]), marker="^", color="#7570b3", linestyle="--", label="SGD (Online Incremental)")

    # In-distribution benchmark line for XGBoost
    axes[0].axhline(IN_DIST_F1["xgboost"], color="black", linestyle=":", label="In-Dist Random Split (0.9065)")

    axes[0].set_title("F1-Score vs. Label Budget (0% to 100%)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Label Budget Fraction", fontsize=10)
    axes[0].set_ylabel("F1-Score", fontsize=10)
    axes[0].set_ylim(0.75, 0.93)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_title("Performance Recovery % vs. Label Budget", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Label Budget Fraction", fontsize=10)
    axes[1].set_ylabel("Recovery %", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(graphs_dir / "budget_recovery_curves.png", dpi=130)
    plt.close(fig)


def _plot_adaptation_strategy_comparison(graphs_dir: Path, df: pd.DataFrame) -> None:
    """Compare strategies at a realistic 5% label budget."""
    sub = df[(df["budget"] == 0.05) | (df["strategy"] == "static")].copy()
    grouped = sub.groupby(["model", "strategy"]).agg({"f1": "mean", "false_positive_rate": "mean", "recovery_pct": "mean"}).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # F1 comparison
    pivot_f1 = grouped.pivot(index="model", columns="strategy", values="f1")
    cols_order = [c for c in ["static", "online_incremental", "drift_triggered", "periodic"] if c in pivot_f1.columns]
    pivot_f1[cols_order].plot(kind="bar", ax=axes[0], colormap="tab10", width=0.75)
    axes[0].set_title("Model F1 at 5% Label Budget Across Strategies", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("F1-Score", fontsize=10)
    axes[0].set_ylim(0.70, 0.92)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(fontsize=8, title="Strategy")

    # FPR comparison
    pivot_fpr = grouped.pivot(index="model", columns="strategy", values="false_positive_rate")
    pivot_fpr[cols_order].plot(kind="bar", ax=axes[1], colormap="tab10", width=0.75)
    axes[1].set_title("False Positive Rate at 5% Label Budget", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("FPR", fontsize=10)
    axes[1].set_ylim(0.0, 0.22)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].legend(fontsize=8, title="Strategy")

    fig.tight_layout()
    fig.savefig(graphs_dir / "adaptation_strategy_comparison.png", dpi=130)
    plt.close(fig)


def _plot_drift_triggered_vs_periodic(graphs_dir: Path, df: pd.DataFrame) -> None:
    """Plot retrain efficiency: Compute Updates vs F1 Recovery for Periodic vs Drift-Triggered."""
    sub = df[df["strategy"].isin(["periodic", "drift_triggered"]) & (df["model"] == "xgboost")]
    grouped = sub.groupby(["strategy", "budget"]).agg({
        "f1": "mean",
        "n_updates": "mean",
        "adaptation_seconds": "mean",
        "recovery_pct": "mean"
    }).reset_index()

    fig, axis = plt.subplots(figsize=(7.5, 5))
    for strat, color, marker in [("periodic", "#1f77b4", "o"), ("drift_triggered", "#2ca02c", "s")]:
        data = grouped[grouped["strategy"] == strat]
        axis.scatter(data["adaptation_seconds"], data["f1"], color=color, s=120, label=strat.replace("_", " ").title())
        for _, r in data.iterrows():
            axis.annotate(
                f"Budget {int(r['budget']*100)}%\n({int(r['n_updates'])} updates)",
                (r["adaptation_seconds"], r["f1"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8
            )

    axis.set_title("Compute vs Accuracy Tradeoff: Drift-Triggered vs Periodic (XGBoost)", fontsize=11, fontweight="bold")
    axis.set_xlabel("Total Adaptation Retraining Time (Seconds)", fontsize=10)
    axis.set_ylabel("Evaluation Stream F1-Score", fontsize=10)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(graphs_dir / "drift_triggered_vs_periodic.png", dpi=130)
    plt.close(fig)


def _plot_monthly_trajectories(graphs_dir: Path, traces_df: pd.DataFrame) -> None:
    """Plot monthly trajectory over the streaming window for 5% budget."""
    sub = traces_df[(traces_df["budget"] == 0.05) | (traces_df["strategy"] == "static")]
    sub_xgb = sub[sub["model"] == "xgboost"].groupby(["strategy", "month"]).agg({"f1": "mean"}).reset_index()

    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    for strat, color, marker in [("static", "#e41a1c", "x"), ("periodic", "#377eb8", "o"), ("drift_triggered", "#4daf4a", "s")]:
        data = sub_xgb[sub_xgb["strategy"] == strat]
        axis.plot(data["month"], data["f1"], marker=marker, color=color, linewidth=2, label=strat.replace("_", " ").title())

    axis.set_title("Monthly Trajectory: XGBoost F1 (5% Label Budget vs Static)", fontsize=11, fontweight="bold")
    axes = axis
    axes.set_xlabel("Evaluation Month (2025)", fontsize=10)
    axes.set_ylabel("Monthly F1-Score", fontsize=10)
    axes.set_ylim(0.65, 0.95)
    axes.grid(True, linestyle="--", alpha=0.5)
    axes.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(graphs_dir / "adaptation_monthly_trajectories.png", dpi=130)
    plt.close(fig)
