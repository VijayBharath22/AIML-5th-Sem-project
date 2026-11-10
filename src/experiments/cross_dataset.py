"""Phase 3 Module 2: Cross-Dataset Distribution Shift Evaluation.

Evaluates bidirectional transfer generalizability between PhreshPhish and PhiUSIIL
using an identical 15-dimensional lexical URL feature space.
Direction A: Train on PhreshPhish -> Test on PhiUSIIL.
Direction B: Train on PhiUSIIL -> Test on PhreshPhish.
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

from src.data.preprocess import spec_keywords, spec_shorteners
from src.features.url_features import FEATURE_COLUMNS, extract_url_features
from src.models.evaluate import compute_metrics, plot_roc_curves, save_confusion_artifacts, save_predictions
from src.models.train_baseline import _estimator, _pipeline
from src.utils.config import append_event, load_config, path_of
from src.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)


def extract_or_load_phiusiil_url_features(config: dict) -> pd.DataFrame:
    processed_dir = path_of(config, "processed_dir")
    cached_path = processed_dir / "phiusiil_url_features.parquet"
    if cached_path.exists():
        LOGGER.info("Loading cached PhiUSIIL 15-feature URL table")
        return pd.read_parquet(cached_path)

    LOGGER.info("Extracting 15 lexical URL features from PhiUSIIL raw URLs...")
    df_phi = pd.read_parquet(processed_dir / "phiusiil.parquet")
    keywords = spec_keywords(config)
    shorteners = spec_shorteners(config)

    extracted = [extract_url_features(url, keywords, shorteners) for url in df_phi["URL"]]
    feature_df = pd.DataFrame(extracted)[FEATURE_COLUMNS]
    feature_df["label"] = df_phi["label"].astype(int)
    feature_df["registrable_domain"] = df_phi["registrable_domain"].astype(str)
    feature_df["URL"] = df_phi["URL"].astype(str)

    feature_df.to_parquet(cached_path, index=False)
    LOGGER.info("Saved PhiUSIIL URL features to %s (N=%d)", cached_path, len(feature_df))
    return feature_df


def run_cross_dataset_evaluation(config: dict | None = None) -> dict:
    config = config or load_config()
    set_seed(int(config["seed"]))
    seed = int(config["seed"])
    results_dir = path_of(config, "results_dir")
    processed_dir = path_of(config, "processed_dir")

    out_base = results_dir / "cross_dataset"
    metrics_dir = out_base / "metrics"
    preds_dir = out_base / "predictions"
    graphs_dir = out_base / "graphs"
    for d in [metrics_dir, preds_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Phase 3 Cross-Dataset Evaluation")
    append_event(config, "phase3_cross_dataset", "Started cross-dataset transfer evaluation")

    # Load both datasets with identical 15 lexical features
    df_phresh = pd.read_parquet(processed_dir / "phreshphish.parquet")
    df_phi = extract_or_load_phiusiil_url_features(config)

    x_phresh = df_phresh[FEATURE_COLUMNS]
    y_phresh = df_phresh["label"].astype(int)

    x_phi = df_phi[FEATURE_COLUMNS]
    y_phi = df_phi["label"].astype(int)

    # Check overlaps
    url_overlap = len(set(df_phi["URL"]) & set(df_phresh["url"]))
    dom_overlap = len(set(df_phi["registrable_domain"]) & set(df_phresh["registrable_domain"]))
    overlap_info = {
        "phiusiil_samples": len(df_phi),
        "phreshphish_samples": len(df_phresh),
        "exact_url_overlap": url_overlap,
        "registrable_domain_overlap": dom_overlap,
        "phiusiil_domain_overlap_pct": float(dom_overlap / df_phi["registrable_domain"].nunique() * 100),
        "phreshphish_domain_overlap_pct": float(dom_overlap / df_phresh["registrable_domain"].nunique() * 100),
    }
    (metrics_dir / "dataset_overlap.json").write_text(json.dumps(overlap_info, indent=2))

    models_config = config["models"]
    results = []
    curves_a = {}
    curves_b = {}

    for model_name, params in models_config.items():
        scale = model_name in {"logistic_regression", "sgd_classifier"}

        # -------------------------------------------------------------
        # Direction A: Train on PhreshPhish -> Test on PhiUSIIL
        # -------------------------------------------------------------
        est_a = _estimator(model_name, dict(params), y_phresh, seed)
        pipe_a = _pipeline(est_a, FEATURE_COLUMNS, [], scale, 0)

        t0 = time.perf_counter()
        pipe_a.fit(x_phresh, y_phresh)
        train_time_a = time.perf_counter() - t0

        t0 = time.perf_counter()
        prob_a = pipe_a.predict_proba(x_phi)[:, 1]
        pred_a = (prob_a >= 0.5).astype(int)
        pred_time_a = time.perf_counter() - t0

        metrics_a = compute_metrics(y_phi, pred_a, prob_a)
        metrics_a["train_seconds"] = train_time_a
        metrics_a["predict_seconds"] = pred_time_a

        key_a = f"train_phreshphish__test_phiusiil__{model_name}"
        save_predictions(preds_dir / f"{key_a}.parquet", y_phi, pred_a, prob_a)
        save_confusion_artifacts(graphs_dir, key_a, metrics_a["confusion_matrix"])
        curves_a[model_name] = (y_phi.to_numpy(), prob_a)

        results.append({
            "direction": "PhreshPhish -> PhiUSIIL",
            "train_dataset": "phreshphish",
            "test_dataset": "phiusiil",
            "model": model_name,
            **metrics_a,
        })

        # -------------------------------------------------------------
        # Direction B: Train on PhiUSIIL -> Test on PhreshPhish
        # -------------------------------------------------------------
        est_b = _estimator(model_name, dict(params), y_phi, seed)
        pipe_b = _pipeline(est_b, FEATURE_COLUMNS, [], scale, 0)

        t0 = time.perf_counter()
        pipe_b.fit(x_phi, y_phi)
        train_time_b = time.perf_counter() - t0

        t0 = time.perf_counter()
        prob_b = pipe_b.predict_proba(x_phresh)[:, 1]
        pred_b = (prob_b >= 0.5).astype(int)
        pred_time_b = time.perf_counter() - t0

        metrics_b = compute_metrics(y_phresh, pred_b, prob_b)
        metrics_b["train_seconds"] = train_time_b
        metrics_b["predict_seconds"] = pred_time_b

        key_b = f"train_phiusiil__test_phreshphish__{model_name}"
        save_predictions(preds_dir / f"{key_b}.parquet", y_phresh, pred_b, prob_b)
        save_confusion_artifacts(graphs_dir, key_b, metrics_b["confusion_matrix"])
        curves_b[model_name] = (y_phresh.to_numpy(), prob_b)

        results.append({
            "direction": "PhiUSIIL -> PhreshPhish",
            "train_dataset": "phiusiil",
            "test_dataset": "phreshphish",
            "model": model_name,
            **metrics_b,
        })

    # Summary table
    res_df = pd.DataFrame(results)
    res_df.to_csv(metrics_dir / "cross_dataset_metrics.csv", index=False)

    summary_payload = {
        "experiment": "cross_dataset_transfer",
        "overlap": overlap_info,
        "results": results,
    }
    (metrics_dir / "cross_dataset_summary.json").write_text(json.dumps(summary_payload, indent=2))

    # ROC plots
    plot_roc_curves(graphs_dir, "cross_dataset__train_phresh_test_phi", curves_a)
    plot_roc_curves(graphs_dir, "cross_dataset__train_phi_test_phresh", curves_b)

    # Cross-dataset vs within-dataset comparison chart
    _plot_cross_dataset_comparison(results_dir, graphs_dir, res_df)

    append_event(config, "phase3_cross_dataset", "Completed cross-dataset transfer evaluation", directions=2, models=len(models_config))
    return summary_payload


def _plot_cross_dataset_comparison(results_dir: Path, graphs_dir: Path, cross_df: pd.DataFrame) -> None:
    # Within-dataset URL-only baseline from Phase 2 comparison
    base_csv = results_dir / "metrics" / "comparison.csv"
    if not base_csv.exists():
        return
    base_df = pd.read_csv(base_csv)

    rows = []
    for model in ["logistic_regression", "random_forest", "xgboost", "sgd_classifier"]:
        phresh_in_dist = base_df[(base_df["dataset"] == "phreshphish") & (base_df["split"] == "split_random") & (base_df["model"] == model)]["f1"].values[0]
        a_f1 = cross_df[(cross_df["direction"] == "PhreshPhish -> PhiUSIIL") & (cross_df["model"] == model)]["f1"].values[0]
        b_f1 = cross_df[(cross_df["direction"] == "PhiUSIIL -> PhreshPhish") & (cross_df["model"] == model)]["f1"].values[0]

        rows.append({
            "model": model,
            "In-Dist PhreshPhish": phresh_in_dist,
            "Transfer: Phresh -> Phi": a_f1,
            "Transfer: Phi -> Phresh": b_f1,
        })

    plot_df = pd.DataFrame(rows).set_index("model")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    plot_df.plot(kind="bar", ax=axis, colormap="coolwarm")
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("F1-Score")
    axis.set_title("Cross-Dataset Generalization Collapse vs. In-Distribution Baseline")
    axis.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(graphs_dir / "cross_dataset_transfer_comparison.png", dpi=120)
    plt.close(fig)

