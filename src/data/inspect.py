"""Dataset quality summaries. These functions do not print raw records."""

from __future__ import annotations

import pandas as pd


def missing_summary(frame: pd.DataFrame) -> dict:
    missing = frame.isna().sum()
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "columns_with_missing": int((missing > 0).sum()),
        "missing_counts": {column: int(count) for column, count in missing.items() if count > 0},
    }


def class_summary(frame: pd.DataFrame, column: str) -> dict:
    counts = frame[column].value_counts(dropna=False).to_dict()
    total = max(len(frame), 1)
    return {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "proportions": {str(key): float(value) / total for key, value in counts.items()},
    }


def numeric_profile(frame: pd.DataFrame, columns: list[str]) -> dict:
    profile = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        profile[column] = {
            "dtype": str(frame[column].dtype),
            "missing": int(series.isna().sum()),
            "nunique": int(series.nunique(dropna=True)),
            "min": None if series.notna().sum() == 0 else float(series.min()),
            "max": None if series.notna().sum() == 0 else float(series.max()),
            "mean": None if series.notna().sum() == 0 else float(series.mean()),
        }
    return profile
