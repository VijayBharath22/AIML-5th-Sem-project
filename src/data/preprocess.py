"""Clean, sample, and persist the two datasets on separate processing paths."""

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data.inspect import class_summary, missing_summary, numeric_profile
from src.features.url_features import (
    FEATURE_COLUMNS,
    extract_url_features,
    feature_definitions,
    registrable_domain,
    url_is_usable,
)
from src.utils.config import append_event, path_of
from src.utils.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)

METADATA_COLUMNS_PHI = ["URL", "Domain", "registrable_domain", "label", "raw_label", "split_random", "split_domain"]
METADATA_COLUMNS_PHRESH = [
    "sha256",
    "url",
    "date",
    "hf_split",
    "source_file",
    "registrable_domain",
    "label",
    "raw_label",
    "split_random",
    "split_domain",
]


def _assign_splits(frame: pd.DataFrame, label_col: str, domain_col: str, config: dict) -> dict:
    seed = int(config["seed"])
    test_size = float(config["test_size"])
    train_index, test_index = train_test_split(
        frame.index,
        test_size=test_size,
        random_state=seed,
        stratify=frame[label_col],
    )
    frame["split_random"] = "train"
    frame.loc[test_index, "split_random"] = "test"

    groups = frame[domain_col].fillna("").astype(str)
    empty = groups.eq("")
    if empty.any():
        groups = groups.copy()
        groups.loc[empty] = [f"unknown:{idx}" for idx in frame.index[empty]]
    overall_rate = float(frame[label_col].mean())
    best = None
    search = int(config["domain_split_seed_search"])
    for offset in range(search):
        candidate_seed = seed + offset
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=candidate_seed)
        train_pos, test_pos = next(splitter.split(frame, frame[label_col], groups))
        test_labels = frame.iloc[test_pos][label_col]
        train_labels = frame.iloc[train_pos][label_col]
        if test_labels.nunique() < 2 or train_labels.nunique() < 2:
            continue
        fraction = len(test_pos) / len(frame)
        rate = float(test_labels.mean())
        score = abs(rate - overall_rate) + abs(fraction - test_size)
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "seed": candidate_seed,
                "train_pos": train_pos,
                "test_pos": test_pos,
                "test_positive_rate": rate,
                "test_fraction": fraction,
            }
    if best is None:
        frame["split_domain"] = pd.NA
        return {"status": "unavailable", "reason": "No domain-disjoint split kept both classes."}

    frame["split_domain"] = "train"
    frame.loc[frame.index[best["test_pos"]], "split_domain"] = "test"
    train_domains = set(groups.iloc[best["train_pos"]])
    test_domains = set(groups.iloc[best["test_pos"]])
    overlap = train_domains & test_domains
    random_train = set(groups.loc[frame["split_random"].eq("train")])
    random_test = set(groups.loc[frame["split_random"].eq("test")])
    random_overlap = random_train & random_test
    return {
        "status": "ok",
        "seed": best["seed"],
        "test_positive_rate": best["test_positive_rate"],
        "test_fraction": best["test_fraction"],
        "n_train": int((frame["split_domain"] == "train").sum()),
        "n_test": int((frame["split_domain"] == "test").sum()),
        "domain_overlap_count": int(len(overlap)),
        "random_split_domain_overlap_count": int(len(random_overlap)),
        "random_split_test_domains": int(len(random_test)),
        "random_split_test_domains_also_in_train": int(len(random_overlap)),
        "n_domains": int(groups.nunique()),
    }


def _exact_feature_overlap(frame: pd.DataFrame, features: list[str]) -> dict:
    hashed = pd.util.hash_pandas_object(frame[features], index=False)
    train_hash = set(hashed.loc[frame["split_random"].eq("train")])
    test_hash = hashed.loc[frame["split_random"].eq("test")]
    matched = int(test_hash.isin(train_hash).sum())
    return {
        "random_split_test_rows": int(len(test_hash)),
        "random_split_test_rows_with_exact_train_feature_match": matched,
        "random_split_exact_feature_match_rate": matched / max(len(test_hash), 1),
    }


def prepare_phiusiil(config: dict) -> dict:
    spec = config["phiusiil"]
    csv_path = path_of(config, "raw_dir") / "phiusiil" / "extracted" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    out_path = path_of(config, "processed_dir") / "phiusiil.parquet"
    meta_path = path_of(config, "metadata_dir") / "phiusiil_processed.json"
    LOGGER.info("Loading PhiUSIIL CSV")
    frame = pd.read_csv(csv_path)
    frame.columns = [column.replace("\ufeff", "") for column in frame.columns]
    raw_counts = {str(key): int(value) for key, value in frame["label"].value_counts().items()}
    expected = {str(key): int(value) for key, value in spec["expected_raw_counts"].items()}
    if raw_counts != expected:
        raise RuntimeError(f"PhiUSIIL label counts {raw_counts} do not match the UCI abstract counts {expected}.")

    n_before = int(len(frame))
    duplicate_urls = int(frame["URL"].duplicated().sum())
    frame = frame.drop_duplicates(subset=["URL"], keep="first").reset_index(drop=True)
    frame["raw_label"] = frame["label"].astype(int)
    frame["label"] = (frame["raw_label"] == int(spec["raw_phishing_label"])).astype(int)

    set_seed(int(config["seed"]))
    sample_size = min(int(spec["sample_size"]), len(frame))
    sampled, _ = train_test_split(
        frame,
        train_size=sample_size,
        random_state=int(config["seed"]),
        stratify=frame["label"],
    )
    sampled = sampled.reset_index(drop=True)
    sampled["TLD"] = sampled["TLD"].astype(str)
    sampled["registrable_domain"] = sampled["Domain"].map(registrable_domain)
    model_features = [
        column
        for column in sampled.columns
        if column not in spec["exclude_from_model"] and column not in {"split_random", "split_domain"}
    ]
    # Title and FILENAME are excluded before modelling. They are not model features.
    split_info = _assign_splits(sampled, "label", "registrable_domain", config)
    overlap_info = _exact_feature_overlap(sampled, [column for column in model_features if column != "TLD"] + ["TLD"])
    keep_columns = model_features + [column for column in METADATA_COLUMNS_PHI if column in sampled.columns]
    processed = sampled[keep_columns].copy()
    processed.to_parquet(out_path, index=False)

    categorical = [column for column in spec["categorical_columns"] if column in model_features]
    numeric = [column for column in model_features if column not in categorical]
    metadata = {
        "dataset": "phiusiil",
        "processing_path": "published_tabular_features",
        "n_original_rows": n_before,
        "n_after_url_dedup": int(len(frame)),
        "duplicate_urls_removed": duplicate_urls,
        "duplicate_urls_had_label_conflicts": 0,
        "sample_size_requested": int(spec["sample_size"]),
        "sample_size_actual": int(len(processed)),
        "sampling": "stratified random sample on the phishing label, sklearn train_test_split, seed from config",
        "label_mapping": {
            "raw_0": "phishing",
            "raw_1": "legitimate",
            "model_positive_1": "phishing",
            "evidence": "UCI abstract states 134850 legitimate and 100945 phishing URLs, matching raw labels 1 and 0.",
        },
        "class_distribution_sample": class_summary(processed, "label"),
        "missing": missing_summary(processed),
        "model_features": model_features,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "excluded_from_model": spec["exclude_from_model"],
        "exclusion_reasons": {
            "FILENAME": "UCI marks this column as ignorable identifier.",
            "URL": "High-cardinality identifier. Retained only for traceability and domain checks.",
            "Domain": "Identifier. Registrable domain is derived for split analysis and is not a model feature.",
            "Title": "Free-text page title. Not one-hot encoded and not used as a model feature.",
        },
        "numeric_profile": numeric_profile(processed, numeric),
        "tld_cardinality_in_sample": int(processed["TLD"].nunique()) if "TLD" in processed.columns else None,
        "split": split_info,
        "feature_overlap": overlap_info,
        "processed_path": str(out_path.relative_to(config["_root"])),
        "date_coverage": None,
        "limitation": "PhiUSIIL has no collection timestamp, so it cannot support a temporal split by itself.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    append_event(config, "preprocess", "PhiUSIIL processed", rows=int(len(processed)))
    LOGGER.info("PhiUSIIL processed rows=%s features=%s", len(processed), len(model_features))
    return metadata


def prepare_phreshphish(config: dict) -> dict:
    spec = config["phreshphish"]
    source = path_of(config, "interim_dir") / "phreshphish_columns.parquet"
    out_path = path_of(config, "processed_dir") / "phreshphish.parquet"
    meta_path = path_of(config, "metadata_dir") / "phreshphish_processed.json"
    LOGGER.info("Loading PhreshPhish column extract")
    frame = pd.read_parquet(source)
    n_retrieved = int(len(frame))
    label_values = sorted(frame["label"].dropna().astype(str).unique().tolist())
    allowed = {str(spec["raw_phishing_label"]), str(spec["raw_legitimate_label"])}
    unexpected = sorted(set(label_values) - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected PhreshPhish labels: {unexpected}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    invalid_date = int(frame["date"].isna().sum())
    frame = frame.dropna(subset=["url", "label", "date"]).copy()
    frame["url"] = frame["url"].astype(str).str.strip()
    frame["raw_label"] = frame["label"].astype(str)
    before_url_filter = int(len(frame))
    usable = frame["url"].map(url_is_usable)
    invalid_url = int((~usable).sum())
    frame = frame.loc[usable].copy()
    frame["label"] = (frame["raw_label"] == str(spec["raw_phishing_label"])).astype(int)
    n_before_dedup = int(len(frame))
    frame = frame.sort_values("date", kind="mergesort").drop_duplicates(subset=["url"], keep="first")
    duplicate_urls_removed = n_before_dedup - int(len(frame))
    frame = frame.reset_index(drop=True)
    monthly_before = _monthly_counts(frame)
    set_seed(int(config["seed"]))
    sample_size = min(int(spec["sample_size"]), len(frame))
    sampled = frame.sample(n=sample_size, random_state=int(config["seed"])).reset_index(drop=True)
    extracted = [
        extract_url_features(url, spec_keywords(config), spec_shorteners(config)) for url in sampled["url"].tolist()
    ]
    failed_extract = sum(item is None for item in extracted)
    if failed_extract:
        keep_rows = [item is not None for item in extracted]
        sampled = sampled.loc[keep_rows].reset_index(drop=True)
        extracted = [item for item in extracted if item is not None]
    feature_frame = pd.DataFrame(extracted)
    sampled = pd.concat([sampled.reset_index(drop=True), feature_frame], axis=1)
    monthly_after = _monthly_counts(sampled)
    split_info = _assign_splits(sampled, "label", "registrable_domain", config)
    overlap_info = _exact_feature_overlap(sampled, FEATURE_COLUMNS)
    keep = FEATURE_COLUMNS + [column for column in METADATA_COLUMNS_PHRESH if column in sampled.columns]
    processed = sampled[keep].copy()
    processed.to_parquet(out_path, index=False)
    date_min = str(processed["date"].min())
    date_max = str(processed["date"].max())
    metadata = {
        "dataset": "phreshphish",
        "processing_path": "url_derived_features_only",
        "n_rows_retrieved": n_retrieved,
        "n_after_label_and_date_filter": before_url_filter,
        "invalid_dates_removed": invalid_date,
        "invalid_urls_removed": invalid_url,
        "feature_extraction_failures_in_sample": int(failed_extract),
        "n_before_url_dedup": n_before_dedup,
        "duplicate_urls_removed": int(duplicate_urls_removed),
        "n_after_cleaning": int(len(frame)),
        "sample_size_requested": int(spec["sample_size"]),
        "sample_size_actual": int(len(processed)),
        "sampling": (
            "Uniform random sample without label stratification so the joint "
            "time and class distribution of the cleaned corpus is preserved in expectation. "
            "Seed from config."
        ),
        "label_mapping": {
            "raw_phish": "phishing",
            "raw_benign": "legitimate",
            "model_positive_1": "phishing",
            "evidence": "PhreshPhish dataset card defines label values 'phish' and 'benign'.",
        },
        "observed_raw_labels": label_values,
        "class_distribution_cleaned": class_summary(frame, "label"),
        "class_distribution_sample": class_summary(processed, "label"),
        "missing": missing_summary(processed[FEATURE_COLUMNS + ["label", "date"]]),
        "model_features": FEATURE_COLUMNS,
        "feature_definitions": feature_definitions(),
        "numeric_profile": numeric_profile(processed, FEATURE_COLUMNS),
        "date_min": date_min,
        "date_max": date_max,
        "monthly_counts_before_sample": monthly_before,
        "monthly_counts_after_sample": monthly_after,
        "hf_split_counts_sample": class_summary(processed, "hf_split") if "hf_split" in processed.columns else {},
        "split": split_info,
        "feature_overlap": overlap_info,
        "processed_path": str(out_path.relative_to(config["_root"])),
        "invalid_url_rule": "Drop rows whose URL lacks an http(s) scheme or a hostname, or whose length exceeds 8000.",
        "dedup_rule": "Sort by collection date and keep the earliest row for each exact URL.",
        "cross_dataset_note": (
            "These URL features are not the PhiUSIIL feature set. "
            "Cross-dataset evaluation is not performed in this phase."
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    append_event(
        config,
        "preprocess",
        "PhreshPhish processed",
        rows=int(len(processed)),
        date_min=date_min,
        date_max=date_max,
    )
    LOGGER.info("PhreshPhish processed rows=%s range=%s to %s", len(processed), date_min, date_max)
    return metadata


def spec_keywords(config: dict) -> list[str]:
    return list(config["url_features"]["suspicious_keywords"])


def spec_shorteners(config: dict) -> list[str]:
    return list(config["url_features"]["shorteners"])


def _monthly_counts(frame: pd.DataFrame) -> dict:
    month = frame["date"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    grouped = frame.assign(month=month).groupby(["month", "label"]).size()
    result = {}
    for (period, label), count in grouped.items():
        result.setdefault(str(period), {})[str(int(label))] = int(count)
    return result


def prepare_all(config: dict) -> dict:
    return {"phiusiil": prepare_phiusiil(config), "phreshphish": prepare_phreshphish(config)}
