"""Phase 3 Module 3: Statistical Drift Detection and Analysis.

Computes Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests
with Bonferroni multiple-comparison correction across all streaming months.
Distinguishes covariate shift from label prevalence changes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.features.url_features import FEATURE_COLUMNS
from src.utils.config import append_event, load_config, path_of
from src.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)

REFERENCE_MONTHS = ["2024-08", "2024-09", "2024-10"]
STREAM_MONTHS = ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11"]


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / max(len(expected), 1)
    act_pct = act_counts / max(len(actual), 1)

    eps = 1e-4
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    exp_pct = exp_pct / exp_pct.sum()
    act_pct = act_pct / act_pct.sum()

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def run_drift_analysis(config: dict | None = None) -> dict:
    config = config or load_config()
    set_seed(int(config["seed"]))
    results_dir = path_of(config, "results_dir")
    processed_dir = path_of(config, "processed_dir")

    out_base = results_dir / "drift"
    metrics_dir = out_base / "metrics"
    graphs_dir = out_base / "graphs"
    for d in [metrics_dir, graphs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Phase 3 Statistical Drift Detection")
    append_event(config, "phase3_drift", "Started statistical drift analysis")

    df = pd.read_parquet(processed_dir / "phreshphish.parquet")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["month"] = df["date"].dt.strftime("%Y-%m")

    ref_df = df.loc[df["month"].isin(REFERENCE_MONTHS)].reset_index(drop=True)
    LOGGER.info("Reference baseline sample: N=%d across %s", len(ref_df), REFERENCE_MONTHS)

    # Bonferroni alpha for 15 simultaneous feature tests
    bonferroni_alpha = 0.05 / len(FEATURE_COLUMNS)

    detailed_records = []
    monthly_summaries = []
    heatmap_psi = []

    for target_month in STREAM_MONTHS:
        cur_df = df.loc[df["month"] == target_month].reset_index(drop=True)
        row_psi = {"month": target_month}
        ks_stats = []

        for col in FEATURE_COLUMNS:
            ref_vals = ref_df[col].to_numpy(dtype=float)
            cur_vals = cur_df[col].to_numpy(dtype=float)

            psi_val = calculate_psi(ref_vals, cur_vals)
            row_psi[col] = psi_val

            ks_res = stats.ks_2samp(ref_vals, cur_vals)
            ks_stat = float(ks_res.statistic)
            ks_p = float(ks_res.pvalue)
            ks_stats.append((col, ks_stat, ks_p))

            # Interpretation based on established standards
            if psi_val >= 0.25:
                interp = "Significant Drift"
            elif psi_val >= 0.10:
                interp = "Moderate Drift"
            else:
                interp = "Stable"

            detailed_records.append({
                "month": target_month,
                "feature": col,
                "ref_mean": float(np.mean(ref_vals)),
                "ref_std": float(np.std(ref_vals)),
                "cur_mean": float(np.mean(cur_vals)),
                "cur_std": float(np.std(cur_vals)),
                "psi": psi_val,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_p,
                "is_ks_significant_bonferroni": bool(ks_p < bonferroni_alpha),
                "interpretation": interp,
            })

        heatmap_psi.append(row_psi)

        mean_psi = float(np.mean([row_psi[c] for c in FEATURE_COLUMNS]))
        mean_ks = float(np.mean([s[1] for s in ks_stats]))
        max_psi_feature = max(FEATURE_COLUMNS, key=lambda c: row_psi[c])
        max_ks_feature = max(FEATURE_COLUMNS, key=lambda c: [s[1] for s in ks_stats if s[0] == c][0])

        monthly_summaries.append({
            "month": target_month,
            "n_samples": len(cur_df),
            "phish_rate": float(cur_df["label"].mean()),
            "mean_psi": mean_psi,
            "max_psi": row_psi[max_psi_feature],
            "max_psi_feature": max_psi_feature,
            "mean_ks": mean_ks,
            "max_ks_feature": max_ks_feature,
            "drift_status": "Severe" if mean_psi >= 0.25 else "Moderate" if mean_psi >= 0.10 else "Stable",
        })

    # Save CSVs
    det_df = pd.DataFrame(detailed_records)
    det_df.to_csv(metrics_dir / "drift_detailed_features.csv", index=False)

    sum_df = pd.DataFrame(monthly_summaries)
    sum_df.to_csv(metrics_dir / "drift_monthly_summary.csv", index=False)

    psi_mat = pd.DataFrame(heatmap_psi)
    psi_mat.to_csv(metrics_dir / "drift_feature_psi_matrix.csv", index=False)

    payload = {
        "reference_window": f"{REFERENCE_MONTHS[0]} to {REFERENCE_MONTHS[-1]}",
        "reference_samples": len(ref_df),
        "bonferroni_threshold": bonferroni_alpha,
        "monthly_summary": monthly_summaries,
    }
    (metrics_dir / "drift_summary.json").write_text(json.dumps(payload, indent=2))

    # Visualizations
    _plot_drift_heatmap(graphs_dir, psi_mat)
    _plot_drift_trajectory(graphs_dir, sum_df)
    _plot_drift_vs_f1(results_dir, graphs_dir, sum_df)

    append_event(config, "phase3_drift", "Completed statistical drift detection", months=len(STREAM_MONTHS))
    return payload


def _plot_drift_heatmap(graphs_dir: Path, psi_df: pd.DataFrame) -> None:
    plot_data = psi_df.set_index("month").T
    fig, axis = plt.subplots(figsize=(9.5, 6.5))
    im = axis.imshow(plot_data.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.4)
    axis.set_xticks(range(len(plot_data.columns)))
    axis.set_xticklabels(plot_data.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(plot_data.index)))
    axis.set_yticklabels(plot_data.index)
    axis.set_title("Feature-Level Population Stability Index (PSI) Across Streaming Months")
    cbar = fig.colorbar(im, ax=axis)
    cbar.set_label("PSI (>0.1 moderate, >0.25 severe)")
    fig.tight_layout()
    fig.savefig(graphs_dir / "drift_feature_heatmap.png", dpi=120)
    plt.close(fig)


def _plot_drift_trajectory(graphs_dir: Path, sum_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(sum_df["month"], sum_df["mean_psi"], marker="o", color="#d95f02", linewidth=2.2, label="Mean PSI")
    axes[0].axhline(0.10, color="orange", linestyle="--", label="Moderate Drift (0.10)")
    axes[0].axhline(0.25, color="red", linestyle="--", label="Severe Drift (0.25)")
    axes[0].set_title("Population Stability Index (PSI) Over Time")
    axes[0].set_xlabel("Evaluation Month (2025)")
    axes[0].set_ylabel("Mean PSI")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(sum_df["month"], sum_df["mean_ks"], marker="s", color="#7570b3", linewidth=2.2, label="Mean KS Distance")
    axes[1].set_title("Two-Sample Kolmogorov-Smirnov Distance Over Time")
    axes[1].set_xlabel("Evaluation Month (2025)")
    axes[1].set_ylabel("Mean KS Statistic")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(graphs_dir / "drift_monthly_trajectory.png", dpi=120)
    plt.close(fig)


def _plot_drift_vs_f1(results_dir: Path, graphs_dir: Path, sum_df: pd.DataFrame) -> None:
    temp_csv = results_dir / "temporal" / "metrics" / "temporal_monthly_metrics.csv"
    if not temp_csv.exists():
        return
    t_df = pd.read_csv(temp_csv)
    avg_f1 = t_df.groupby("month")["f1"].mean().reset_index()
    merged = pd.merge(sum_df, avg_f1, on="month")

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    color = "#d95f02"
    ax1.set_xlabel("Streaming Month (2025)")
    ax1.set_ylabel("Mean PSI Drift", color=color)
    l1 = ax1.plot(merged["month"], merged["mean_psi"], marker="o", color=color, linewidth=2.5, label="Mean PSI")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    color = "#2b83ba"
    ax2.set_ylabel("Average Model F1-Score", color=color)
    l2 = ax2.plot(merged["month"], merged["f1"], marker="s", color=color, linewidth=2.5, linestyle="--", label="Model F1")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", fontsize=9)
    ax1.set_title("Drift vs. Performance: Inverse Correlation Between PSI and Model F1")
    fig.tight_layout()
    fig.savefig(graphs_dir / "drift_vs_f1_correlation.png", dpi=120)
    plt.close(fig)

