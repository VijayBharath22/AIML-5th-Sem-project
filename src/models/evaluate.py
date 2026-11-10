"""Metrics, prediction persistence, and plots for baseline models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "support_positive": int((y_true == 1).sum()),
        "support_negative": int((y_true == 0).sum()),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix.astype(int).tolist(),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
    return metrics


def save_predictions(path: Path, y_true, y_pred, y_prob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"y_true": np.asarray(y_true).astype(int), "y_pred": np.asarray(y_pred).astype(int)})
    if y_prob is not None:
        frame["y_prob"] = np.asarray(y_prob, dtype=float)
    frame.to_parquet(path, index=False)


def save_confusion_artifacts(results_dir: Path, key: str, matrix: list[list[int]]) -> None:
    csv_path = results_dir / "confusion_matrices" / f"{key}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=["true_legitimate", "true_phishing"], columns=["pred_legitimate", "pred_phishing"]).to_csv(csv_path)
    fig, axis = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["legitimate", "phishing"],
        yticklabels=["legitimate", "phishing"],
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(key.replace("__", " | "))
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrices" / f"{key}.png", dpi=120)
    plt.close(fig)


def plot_roc_curves(results_dir: Path, key: str, curves: dict[str, tuple]) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5))
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axis.plot(fpr, tpr, label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"ROC curves ({key})")
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "graphs" / f"roc__{key}.png", dpi=120)
    plt.close(fig)


def plot_metric_bars(results_dir: Path, key: str, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(7, 4))
    plot_frame = frame.set_index("model")[["accuracy", "f1", "roc_auc", "false_positive_rate"]]
    plot_frame.plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"Model comparison ({key})")
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "graphs" / f"model_comparison__{key}.png", dpi=120)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    frame.set_index("model")[["precision", "recall", "f1"]].plot(kind="bar", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title(f"Precision, recall and F1 ({key})")
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "graphs" / f"prf__{key}.png", dpi=120)
    plt.close(fig)


def plot_importance(results_dir: Path, key: str, names: list[str], values: list[float], limit: int = 20) -> None:
    order = np.argsort(values)[-limit:]
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.barh([names[index] for index in order], [values[index] for index in order])
    axis.set_title(f"Feature importance ({key})")
    fig.tight_layout()
    fig.savefig(results_dir / "graphs" / f"feature_importance__{key}.png", dpi=120)
    plt.close(fig)


def plot_dataset_overview(results_dir: Path, phiusiil: pd.DataFrame, phresh: pd.DataFrame) -> None:
    graphs = results_dir / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for axis, frame, title in (
        (axes[0], phiusiil, "PhiUSIIL sample"),
        (axes[1], phresh, "PhreshPhish sample"),
    ):
        counts = frame["label"].value_counts().sort_index()
        axis.bar(["legitimate (0)", "phishing (1)"], [int(counts.get(0, 0)), int(counts.get(1, 0))])
        axis.set_title(title)
        axis.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(graphs / "class_distribution.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, frame, title in (
        (axes[0], phiusiil.drop(columns=[c for c in phiusiil.columns if c.startswith("split_") or c in {"URL", "Domain", "registrable_domain"}], errors="ignore"), "PhiUSIIL"),
        (axes[1], phresh[[c for c in phresh.columns if c.startswith("url_") or c.startswith("n_") or c.startswith("is_") or c.startswith("has_") or c in {"label", "date"}]], "PhreshPhish"),
    ):
        missing = frame.isna().mean().sort_values(ascending=False).head(12)
        axis.barh(missing.index.astype(str)[::-1], missing.values[::-1])
        axis.set_xlim(0, 1)
        axis.set_title(f"Missing fraction ({title})")
    fig.tight_layout()
    fig.savefig(graphs / "missing_values.png", dpi=120)
    plt.close(fig)

    if "date" in phresh.columns:
        month = phresh["date"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
        table = phresh.assign(month=month).groupby(["month", "label"]).size().unstack(fill_value=0).sort_index()
        fig, axis = plt.subplots(figsize=(10, 4))
        table.plot(kind="bar", stacked=True, ax=axis)
        axis.set_title("PhreshPhish sample by month")
        axis.set_xlabel("Month")
        axis.legend(["legitimate", "phishing"][: table.shape[1]])
        fig.tight_layout()
        fig.savefig(graphs / "phreshphish_monthly_counts.png", dpi=120)
        plt.close(fig)


def verify_saved_metrics(results_dir: Path) -> dict:
    """Recompute metrics from saved predictions and compare with recorded JSON."""
    mismatches = []
    checked = 0
    metrics_dir = results_dir / "metrics"
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name in {"summary.json", "verification.json", "environment.json"}:
            continue
        payload = json.loads(path.read_text())
        pred_path = results_dir / "predictions" / f"{path.stem}.parquet"
        if not pred_path.exists() or "metrics" not in payload:
            continue
        frame = pd.read_parquet(pred_path)
        recomputed = compute_metrics(frame["y_true"], frame["y_pred"], frame["y_prob"] if "y_prob" in frame else None)
        checked += 1
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"):
            left = payload["metrics"].get(name)
            right = recomputed.get(name)
            if left is None and right is None:
                continue
            if left is None or right is None or abs(float(left) - float(right)) > 1e-8:
                mismatches.append({"file": path.name, "metric": name, "saved": left, "recomputed": right})
    report = {"checked_result_files": checked, "mismatches": mismatches, "passed": not mismatches and checked > 0}
    (metrics_dir / "verification.json").write_text(json.dumps(report, indent=2))
    return report
