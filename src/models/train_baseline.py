"""Train logistic regression, random forest, XGBoost, and SGD baselines."""

from __future__ import annotations

import json
import logging
import platform
import time
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.models.evaluate import (
    compute_metrics,
    plot_importance,
    plot_metric_bars,
    plot_roc_curves,
    save_confusion_artifacts,
    save_predictions,
)
from src.utils.config import append_event, path_of

LOGGER = logging.getLogger(__name__)


def _estimator(name: str, params: dict, y_train: pd.Series, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, **params)
    if name == "sgd_classifier":
        return SGDClassifier(random_state=seed, **params)
    if name == "xgboost":
        positive = int((y_train == 1).sum())
        negative = int((y_train == 0).sum())
        return XGBClassifier(
            random_state=seed,
            scale_pos_weight=negative / max(positive, 1),
            **params,
        )
    raise KeyError(name)


def _pipeline(estimator, numeric: list[str], categorical: list[str], scale: bool, min_frequency: int) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("num", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=min_frequency,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _feature_lists(dataset: str, frame: pd.DataFrame, metadata: dict) -> tuple[list[str], list[str]]:
    features = list(metadata["model_features"])
    categorical = [column for column in metadata.get("categorical_features", []) if column in frame.columns]
    numeric = [column for column in features if column not in categorical and column in frame.columns]
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{dataset} is missing model features: {missing}")
    return numeric, categorical


def _importance(pipeline: Pipeline) -> tuple[list[str], list[float]] | None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    names = list(preprocessor.get_feature_names_out())
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float).ravel())
    else:
        return None
    if len(names) != len(values):
        return None
    return names, values.tolist()


def train_all(config: dict) -> dict:
    processed_dir = path_of(config, "processed_dir")
    metadata_dir = path_of(config, "metadata_dir")
    models_dir = path_of(config, "models_dir")
    results_dir = path_of(config, "results_dir")
    seed = int(config["seed"])
    records = []
    errors = []
    roc_payload: dict[str, dict] = {}

    for dataset in ("phiusiil", "phreshphish"):
        frame = pd.read_parquet(processed_dir / f"{dataset}.parquet")
        metadata = json.loads((metadata_dir / f"{dataset}_processed.json").read_text())
        numeric, categorical = _feature_lists(dataset, frame, metadata)
        min_frequency = int(config["phiusiil"].get("one_hot_min_frequency", 100))
        split_names = ["split_random"]
        if frame["split_domain"].notna().all():
            split_names.append("split_domain")
        else:
            errors.append({"dataset": dataset, "error": "Domain-disjoint split was not available."})
        for split_name in split_names:
            curves = {}
            split_rows = []
            train_frame = frame.loc[frame[split_name].eq("train")]
            test_frame = frame.loc[frame[split_name].eq("test")]
            feature_columns = numeric + categorical
            x_train = train_frame[feature_columns]
            y_train = train_frame["label"].astype(int)
            x_test = test_frame[feature_columns]
            y_test = test_frame["label"].astype(int)
            for model_name, params in config["models"].items():
                key = f"{dataset}__{split_name}__{model_name}"
                LOGGER.info("Training %s", key)
                try:
                    scale = model_name in {"logistic_regression", "sgd_classifier"}
                    estimator = _estimator(model_name, dict(params), y_train, seed)
                    pipeline = _pipeline(estimator, numeric, categorical, scale, min_frequency)
                    start = time.perf_counter()
                    pipeline.fit(x_train, y_train)
                    train_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    probabilities = pipeline.predict_proba(x_test)[:, 1]
                    predictions = (probabilities >= 0.5).astype(int)
                    predict_seconds = time.perf_counter() - start
                    metrics = compute_metrics(y_test, predictions, probabilities)
                    metrics["train_seconds"] = train_seconds
                    metrics["predict_seconds"] = predict_seconds
                    metrics["n_train"] = int(len(train_frame))
                    metrics["n_test"] = int(len(test_frame))
                    payload = {
                        "dataset": dataset,
                        "split": split_name,
                        "model": model_name,
                        "positive_class": "phishing (1)",
                        "metrics": metrics,
                        "params": pipeline.named_steps["model"].get_params(),
                        "features": feature_columns,
                        "preprocessing_fit_on": "train split only",
                    }
                    # get_params includes large objects for some estimators. Keep JSON-safe scalars.
                    payload["params"] = {
                        name: value
                        for name, value in payload["params"].items()
                        if isinstance(value, (str, int, float, bool, type(None)))
                    }
                    (results_dir / "metrics" / f"{key}.json").write_text(json.dumps(payload, indent=2))
                    save_predictions(results_dir / "predictions" / f"{key}.parquet", y_test, predictions, probabilities)
                    save_confusion_artifacts(results_dir, key, metrics["confusion_matrix"])
                    joblib.dump(pipeline, models_dir / f"{key}.joblib")
                    curves[model_name] = (y_test.to_numpy(), probabilities)
                    split_rows.append({"model": model_name, **{name: metrics[name] for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")}})
                    records.append(payload)
                    if split_name == "split_random":
                        importance = _importance(pipeline)
                        if importance is not None:
                            names, values = importance
                            plot_importance(results_dir, key, names, values)
                            order = np.argsort(values)[::-1][:20]
                            payload["top_features"] = [
                                {"feature": names[index], "value": float(values[index])} for index in order
                            ]
                            (results_dir / "metrics" / f"{key}.json").write_text(json.dumps(payload, indent=2))
                    append_event(config, "train", f"Trained {key}", f1=metrics["f1"], roc_auc=metrics["roc_auc"])
                except Exception as exc:
                    LOGGER.exception("Model failed: %s", key)
                    errors.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
                    append_event(config, "train", f"Failed {key}", error=str(exc))
            if curves:
                plot_roc_curves(results_dir, f"{dataset}__{split_name}", curves)
                roc_payload[f"{dataset}__{split_name}"] = True
            if split_rows:
                plot_metric_bars(results_dir, f"{dataset}__{split_name}", split_rows)

    summary = {
        "n_models_trained": len(records),
        "results": [
            {
                "dataset": item["dataset"],
                "split": item["split"],
                "model": item["model"],
                **item["metrics"],
            }
            for item in records
        ],
        "errors": errors,
    }
    (results_dir / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    comparison = pd.DataFrame(summary["results"])
    if not comparison.empty:
        comparison.to_csv(results_dir / "metrics" / "comparison.csv", index=False)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            name: version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "pyarrow", "tldextract"]
        },
    }
    (results_dir / "metrics" / "environment.json").write_text(json.dumps(environment, indent=2))
    if errors:
        (results_dir / "metrics" / "errors.json").write_text(json.dumps(errors, indent=2))
    return summary
