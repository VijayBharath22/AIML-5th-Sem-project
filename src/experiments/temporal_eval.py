"""Phase 3 Module 1: Chronological Temporal Evaluation on PhreshPhish.

Evaluates baseline models (Logistic Regression, Random Forest, XGBoost, SGDClassifier)
trained strictly on historical data (2024-08 to 2024-10) against future data (2025-06 to 2025-11),
including month-by-month trajectory tracking.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.url_features import FEATURE_COLUMNS
from src.models.evaluate import compute_metrics, plot_roc_curves, save_confusion_artifacts, save_predictions
from src.models.train_baseline import _estimator, _pipeline
from src.utils.config import append_event, load_config, path_of
from src.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)

TRAIN_MONTHS = ["2024-08", "2024-09", "2024-10"]
TEST_MONTHS = ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11"]


def run_temporal_evaluation(config: dict | None = None) -> dict:
    config = config or load_config()
    set_seed(int(config["seed"]))
    seed = int(config["seed"])
    results_dir = path_of(config, "results_dir")
    processed_dir = path_of(config, "processed_dir")

    out_base = results_dir / "temporal"
    metrics_dir = out_base / "metrics"
    preds_dir = out_base / "predictions"
    graphs_dir = out_base / "graphs"
    monthly_dir = out_base / "monthly_analysis"
    for d in [metrics_dir, preds_dir, graphs_dir, monthly_dir]:
        d.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Phase 3 Temporal Evaluation")
    append_event(config, "phase3_temporal", "Started temporal distribution shift evaluation")

    df = pd.read_parquet(processed_dir / "phreshphish.parquet")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["month"] = df["date"].dt.strftime("%Y-%m")

    train_df = df.loc[df["month"].isin(TRAIN_MONTHS)].reset_index(drop=True)
    test_df = df.loc[df["month"].isin(TEST_MONTHS)].reset_index(drop=True)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"].astype(int)
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int)

    LOGGER.info(
        "Chronological Split: Historical Train N=%d (phish rate %.3f), Future Test N=%d (phish rate %.3f)",
        len(train_df), y_train.mean(), len(test_df), y_test.mean()
    )

    models_config = config["models"]
    curves = {}
    aggregate_results = []
    monthly_trajectory = []

    for model_name, params in models_config.items():
        key = f"temporal__{model_name}"
        LOGGER.info("Training temporal baseline %s", model_name)
        scale = model_name in {"logistic_regression", "sgd_classifier"}
        estimator = _estimator(model_name, dict(params), y_train, seed)
        pipeline = _pipeline(estimator, FEATURE_COLUMNS, [], scale, 0)

        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        train_time = time.perf_counter() - t0

        # Predict on future test window
        t0 = time.perf_counter()
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        predict_time = time.perf_counter() - t0

        metrics = compute_metrics(y_test, predictions, probabilities)
        metrics["train_seconds"] = train_time
        metrics["predict_seconds"] = predict_time
        metrics["n_train"] = int(len(train_df))
        metrics["n_test"] = int(len(test_df))

        # Save predictions and confusion matrix
        save_predictions(preds_dir / f"{key}.parquet", y_test, predictions, probabilities)
        save_confusion_artifacts(graphs_dir, key, metrics["confusion_matrix"])
        curves[model_name] = (y_test.to_numpy(), probabilities)

        rec = {
            "model": model_name,
            "train_months": TRAIN_MONTHS,
            "test_months": TEST_MONTHS,
            "n_train": len(train_df),
            "n_test": len(test_df),
            **metrics,
        }
        (metrics_dir / f"{key}.json").write_text(json.dumps(rec, indent=2))
        aggregate_results.append(rec)

        # Monthly trajectory tracking
        for m in TEST_MONTHS:
            m_sub = test_df.loc[test_df["month"] == m]
            m_x = m_sub[FEATURE_COLUMNS]
            m_y = m_sub["label"].astype(int)
            m_prob = pipeline.predict_proba(m_x)[:, 1]
            m_pred = (m_prob >= 0.5).astype(int)
            m_metrics = compute_metrics(m_y, m_pred, m_prob)

            m_key = f"{model_name}__{m}"
            save_confusion_artifacts(monthly_dir, m_key, m_metrics["confusion_matrix"])

            monthly_trajectory.append({
                "model": model_name,
                "month": m,
                "n_samples": len(m_sub),
                "phish_rate": float(m_y.mean()),
                "accuracy": m_metrics["accuracy"],
                "precision": m_metrics["precision"],
                "recall": m_metrics["recall"],
                "f1": m_metrics["f1"],
                "roc_auc": m_metrics["roc_auc"],
                "false_positive_rate": m_metrics["false_positive_rate"],
            })

    # Summary tables
    agg_df = pd.DataFrame(aggregate_results)
    agg_df.to_csv(metrics_dir / "temporal_aggregate_metrics.csv", index=False)

    month_df = pd.DataFrame(monthly_trajectory)
    month_df.to_csv(metrics_dir / "temporal_monthly_metrics.csv", index=False)

    # Plot ROC curves
    plot_roc_curves(graphs_dir, "temporal_future_eval", curves)

    # Plot Degradation Bar Chart vs Phase 2 Baselines
    _plot_temporal_degradation(results_dir, graphs_dir, agg_df)

    # Plot Monthly Trajectory
    _plot_monthly_curves(graphs_dir, month_df)

    summary = {
        "train_window": f"{TRAIN_MONTHS[0]} to {TRAIN_MONTHS[-1]}",
        "test_window": f"{TEST_MONTHS[0]} to {TEST_MONTHS[-1]}",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "models": aggregate_results,
        "monthly": monthly_trajectory,
    }
    (metrics_dir / "temporal_summary.json").write_text(json.dumps(summary, indent=2))
    append_event(config, "phase3_temporal", "Completed temporal distribution shift evaluation", n_models=len(aggregate_results))
    return summary


def _plot_temporal_degradation(results_dir: Path, graphs_dir: Path, temporal_df: pd.DataFrame) -> None:
    base_csv = results_dir / "metrics" / "comparison.csv"
    if not base_csv.exists():
        return
    base_df = pd.read_csv(base_csv)
    base_phresh = base_df[base_df["dataset"] == "phreshphish"]

    rows = []
    for model in ["logistic_regression", "random_forest", "xgboost", "sgd_classifier"]:
        r_f1 = base_phresh[(base_phresh["split"] == "split_random") & (base_phresh["model"] == model)]["f1"].values[0]
        d_f1 = base_phresh[(base_phresh["split"] == "split_domain") & (base_phresh["model"] == model)]["f1"].values[0]
        t_f1 = temporal_df[temporal_df["model"] == model]["f1"].values[0]
        rows.append({
            "model": model,
            "Random Split (In-Dist)": r_f1,
            "Domain-Disjoint Split": d_f1,
            "Temporal Shift (8-12 mo Gap)": t_f1,
        })
    plot_df = pd.DataFrame(rows).set_index("model")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    plot_df.plot(kind="bar", ax=axis, colormap="viridis")
    axis.set_ylim(0.5, 1.0)
    axis.set_ylabel("F1-Score")
    axis.set_title("Performance Degradation: In-Distribution vs. Domain Shift vs. Temporal Shift")
    axis.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(graphs_dir / "temporal_vs_baseline_comparison.png", dpi=120)
    plt.close(fig)


def _plot_monthly_curves(graphs_dir: Path, month_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, group in month_df.groupby("model"):
        axes[0].plot(group["month"], group["f1"], marker="o", label=model)
        axes[1].plot(group["month"], group["roc_auc"], marker="s", label=model)

    axes[0].set_title("Monthly Trajectory: F1-Score Over Time")
    axes[0].set_xlabel("Evaluation Month (2025)")
    axes[0].set_ylabel("F1-Score")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("Monthly Trajectory: ROC-AUC Over Time")
    axes[1].set_xlabel("Evaluation Month (2025)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0.7, 1.0)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(graphs_dir / "temporal_monthly_trajectory.png", dpi=120)
    plt.close(fig)

