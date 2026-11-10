"""Write reports and project context from saved experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.config import path_of


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _fmt(value, digits=4):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_table(rows: list[dict], dataset: str, split: str) -> str:
    selected = [row for row in rows if row["dataset"] == dataset and row["split"] == split]
    if not selected:
        return "_No results for this split._\n"
    header = "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in selected:
        lines.append(
            "| {model} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} | {fpr} | {train} | {pred} |".format(
                model=row["model"],
                accuracy=_fmt(row.get("accuracy")),
                precision=_fmt(row.get("precision")),
                recall=_fmt(row.get("recall")),
                f1=_fmt(row.get("f1")),
                roc_auc=_fmt(row.get("roc_auc")),
                fpr=_fmt(row.get("false_positive_rate")),
                train=_fmt(row.get("train_seconds"), 2),
                pred=_fmt(row.get("predict_seconds"), 3),
            )
        )
    return "\n".join(lines) + "\n"


def _best(rows: list[dict], dataset: str, split: str = "split_random") -> dict | None:
    selected = [row for row in rows if row["dataset"] == dataset and row["split"] == split]
    if not selected:
        return None
    return max(selected, key=lambda row: (row.get("f1") or -1, row.get("roc_auc") or -1))


def _events(config: dict) -> list[dict]:
    path = path_of(config, "logs_dir") / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_reports(config: dict) -> None:
    metadata_dir = path_of(config, "metadata_dir")
    results_dir = path_of(config, "results_dir")
    reports = path_of(config, "reports_dir")
    phi = _load(metadata_dir / "phiusiil_processed.json")
    phresh = _load(metadata_dir / "phreshphish_processed.json")
    summary = _load(results_dir / "metrics" / "summary.json")
    environment = _load(results_dir / "metrics" / "environment.json")
    verification = _load(results_dir / "metrics" / "verification.json")
    rows = summary["results"]
    phi_best = _best(rows, "phiusiil")
    phresh_best = _best(rows, "phreshphish")
    errors = summary.get("errors") or []

    (reports / "dataset_analysis.md").write_text(_dataset_report(phi, phresh, config))
    (reports / "baseline_report.md").write_text(_baseline_report(phi, phresh, rows, environment, verification, errors, phi_best, phresh_best))
    (reports / "experiment_log.md").write_text(_log_report(_events(config), errors, verification))
    (reports / "research_paper_draft.md").write_text(_paper(phi, phresh, rows, environment, phi_best, phresh_best))
    _write_context_files(config, phi, phresh, rows, environment, phi_best, phresh_best, errors)


def _dataset_report(phi: dict, phresh: dict, config: dict) -> str:
    phi_features = ", ".join(f"`{name}`" for name in phi["model_features"])
    phresh_defs = "\n".join(f"- `{name}`: {text}" for name, text in phresh["feature_definitions"].items())
    return f"""# Dataset Analysis

This report describes the data actually retrieved and processed for Phase 1. It does not report model adaptation or temporal test results.

## PhiUSIIL Phishing URL Dataset

- Source page: {config['phiusiil']['uci_page']}
- Download: `{config['phiusiil']['zip_url']}`
- DOI recorded by UCI: {config['phiusiil']['doi']}
- UCI creation year: 2024. UCI citation text dates the paper 2023. The article number in the DOI is 103545.
- Original rows: {phi['n_original_rows']}
- Rows after exact-URL deduplication: {phi['n_after_url_dedup']}
- Duplicate URLs removed: {phi['duplicate_urls_removed']}
- Label conflicts among duplicate URLs: {phi['duplicate_urls_had_label_conflicts']}
- Sampled rows: {phi['sample_size_actual']} of {phi['sample_size_requested']} requested
- Sampling: {phi['sampling']}
- Missing values in the processed sample: {phi['missing']['columns_with_missing']} columns
- Date coverage: none. The published table has no collection timestamp.

### Label mapping

UCI abstract: 134,850 legitimate and 100,945 phishing URLs. Those counts match raw `label` 1 and raw `label` 0 before deduplication. Modelling uses 1 for phishing and 0 for legitimate.

Sample class counts (model label): `{json.dumps(phi['class_distribution_sample']['counts'])}`

### Features

Model features ({len(phi['model_features'])}): {phi_features}

Categorical feature: {', '.join(phi['categorical_features'])} ({phi['tld_cardinality_in_sample']} distinct values in the sample). Rare TLD levels are collapsed with `OneHotEncoder(min_frequency={config['phiusiil']['one_hot_min_frequency']})`, fit on the training split only.

Excluded from the model:

- `FILENAME`: UCI says this column can be ignored.
- `URL` and `Domain`: identifiers. `registrable_domain` is derived for split diagnostics and is not a feature.
- `Title`: free text, not encoded.
- `label` and `raw_label`: targets.

### Split diagnostics

Domain-disjoint status: `{json.dumps(phi['split'])}`

Exact feature-vector overlap on the random split: `{json.dumps(phi['feature_overlap'])}`

### Limitations

PhiUSIIL cannot support a temporal evaluation on its own. Several published features (`URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`, title-match scores, and webpage features) are specific to this dataset and are not available for PhreshPhish. High random-split accuracy would not, by itself, show robustness to drift.

## PhreshPhish

- Dataset card: {config['phreshphish']['hf_page']}
- Paper linked from the dataset card: {config['phreshphish']['paper_url']}
- Card version note: v1.0.1 (2026-02-07), with v1.0.0 released 2025-05-14.
- Rows retrieved (URL, label, date, sha256 only): {phresh['n_rows_retrieved']}
- Invalid dates removed: {phresh['invalid_dates_removed']}
- Invalid URLs removed: {phresh['invalid_urls_removed']}
- Duplicate URLs removed: {phresh['duplicate_urls_removed']}
- Rows after cleaning: {phresh['n_after_cleaning']}
- Sampled rows: {phresh['sample_size_actual']} of {phresh['sample_size_requested']} requested
- Sampling: {phresh['sampling']}
- Date range in the sample: {phresh['date_min']} to {phresh['date_max']}
- HTML, brand target, language, and language score were not downloaded.

### Label mapping

Dataset card values are `phish` and `benign`. Model label 1 is phishing and 0 is legitimate.

Cleaned class counts: `{json.dumps(phresh['class_distribution_cleaned']['counts'])}`

Sample class counts: `{json.dumps(phresh['class_distribution_sample']['counts'])}`

### URL features

{phresh_defs}

Keywords and shorteners are listed in `configs/experiment_config.yaml`. The label, date, sha256, and source file are not features.

### Split diagnostics

Domain-disjoint status: `{json.dumps(phresh['split'])}`

Exact feature-vector overlap on the random split: `{json.dumps(phresh['feature_overlap'])}`

### Limitations

The original Hugging Face train and test splits were not used as this project's evaluation protocol. Preserving dates does not mean a temporal evaluation has been run. URL lexical features omit page HTML, which the PhreshPhish paper treats as part of the benchmark. Feature definitions differ from PhiUSIIL, so the two processed tables are not a cross-dataset experiment.

## Preprocessing controls

- Transformations inside model pipelines are fit on the training split only.
- Sampling and splits use seed {config['seed']}.
- The positive class is phishing for every metric.
"""


def _baseline_report(phi, phresh, rows, environment, verification, errors, phi_best, phresh_best) -> str:
    packages = ", ".join(f"{name} {version}" for name, version in environment["packages"].items())
    error_text = "None." if not errors else "```json\n" + json.dumps(errors, indent=2) + "\n```"
    return f"""# Baseline Experiment Report

## Objective

Estimate conventional baseline phishing-detection performance for four classifiers. These numbers are a reference point for later temporal, cross-dataset, and drift-adaptation experiments. They are not those experiments.

## Environment

- Python {environment['python']}
- Platform: {environment['platform']}
- Packages: {packages}
- Hardware intent: CPU training. NVIDIA drivers were not available in this environment, and no GPU training was used.

## Data used

- PhiUSIIL processed rows: {phi['sample_size_actual']}. Random split and domain-disjoint split were taken from this sample.
- PhreshPhish processed rows: {phresh['sample_size_actual']}. Date range {phresh['date_min']} to {phresh['date_max']}.
- Positive class: phishing, encoded as 1.
- Decision threshold: 0.5 on the predicted phishing probability.

## Models

1. Logistic regression, `lbfgs`, `class_weight=balanced`, median imputation and standard scaling inside the training pipeline.
2. Random forest, 100 trees, `max_depth=18`, `min_samples_leaf=2`, `n_jobs=2`.
3. XGBoost, 120 trees, `max_depth=6`, `learning_rate=0.1`, histogram tree method, `scale_pos_weight` set from the training class counts.
4. SGDClassifier, `loss=log_loss`, `class_weight=balanced`, same scaling pipeline as logistic regression.

No test labels or test-only category levels were used to fit imputation, scaling, or one-hot encoding. PhiUSIIL `TLD` is the only categorical model input. PhreshPhish uses the 15 URL features only. Separate pipelines were fit for each dataset and each split.

## Metric definitions

- Precision, recall, and F1 use phishing as the positive class.
- False positive rate is false phishing decisions divided by all legitimate test rows.
- ROC-AUC uses the predicted phishing probability.
- Confusion-matrix axes are legitimate (0) and phishing (1), with rows as actual labels.

## PhiUSIIL, stratified random split

{_metric_table(rows, 'phiusiil', 'split_random')}

## PhiUSIIL, domain-disjoint split

{_metric_table(rows, 'phiusiil', 'split_domain')}

Domain overlap count for this split: {phi['split'].get('domain_overlap_count')}

## PhreshPhish, stratified random split

The split is stratified by label. It is not a temporal split.

{_metric_table(rows, 'phreshphish', 'split_random')}

## PhreshPhish, domain-disjoint split

{_metric_table(rows, 'phreshphish', 'split_domain')}

Domain overlap count for this split: {phresh['split'].get('domain_overlap_count')}

## Confusion matrices and graphs

PNG and CSV confusion matrices are in `results/confusion_matrices/`. Comparison bars, precision-recall-F1 bars, and ROC curves are in `results/graphs/` with names prefixed by dataset and split. Feature-importance plots for the random-split models are in the same folder. Class balance, missing-value fractions, and the PhreshPhish monthly sample counts are `results/graphs/class_distribution.png`, `missing_values.png`, and `phreshphish_monthly_counts.png`.

## Comparison

Best PhiUSIIL random-split model by F1: {_fmt_best(phi_best)}

Best PhreshPhish random-split model by F1: {_fmt_best(phresh_best)}

The two scores are not a cross-dataset comparison. The feature sets differ. A higher PhiUSIIL score is expected if webpage and corpus-derived features are easier than lexical URL features, but that interpretation is a hypothesis supported only by the separate within-dataset numbers above.

Random-split domain overlap remains a leakage risk even when the domain string is not a feature. The domain-disjoint tables are the stricter baseline collected in this phase. Exact duplicate feature vectors between train and test are quantified in `reports/dataset_analysis.md`.

## Observations

- Metrics were recomputed from saved predictions. Verification status: passed={verification.get('passed')}, files checked={verification.get('checked_result_files')}, mismatches={len(verification.get('mismatches') or [])}.
- Training and prediction times are wall-clock seconds on CPU.
- Errors recorded by the trainer: {error_text}

## Limitations

- Hyperparameters are fixed baselines, not a tuned search.
- No temporal holdout was trained or scored.
- No model was trained on one dataset and tested on the other.
- Drift detectors and adaptation policies were not implemented.
- PhreshPhish HTML was not used.
- PhiUSIIL has no timestamps.
"""


def _fmt_best(row: dict | None) -> str:
    if row is None:
        return "not available."
    return (
        f"{row['model']} with F1 {_fmt(row.get('f1'))}, ROC-AUC {_fmt(row.get('roc_auc'))}, "
        f"precision {_fmt(row.get('precision'))}, recall {_fmt(row.get('recall'))}, "
        f"FPR {_fmt(row.get('false_positive_rate'))}, accuracy {_fmt(row.get('accuracy'))}."
    )


def _log_report(events: list[dict], errors: list, verification: dict) -> str:
    lines = ["# Experiment Log", "", "Times are UTC.", ""]
    for event in events:
        extra = {key: value for key, value in event.items() if key not in {"time", "stage", "message"}}
        suffix = f" {json.dumps(extra)}" if extra else ""
        lines.append(f"- {event['time']} [{event['stage']}] {event['message']}{suffix}")
    lines.append("")
    lines.append(f"Verification: `{json.dumps(verification)}`")
    if errors:
        lines.append("")
        lines.append("Errors:")
        lines.append("```json")
        lines.append(json.dumps(errors, indent=2))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _paper(phi, phresh, rows, environment, phi_best, phresh_best) -> str:
    return f"""# Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning

## Abstract

Phishing URLs change over time, so a classifier that scores well on a random split can still fail on later or independently collected data. This draft reports Phase 1 and Phase 2 of that study: preparation of the PhiUSIIL and PhreshPhish datasets, and CPU baselines for logistic regression, random forest, XGBoost, and SGDClassifier. On the processed PhiUSIIL sample of {phi['sample_size_actual']} rows, the best random-split F1 was {_fmt_best(phi_best)} On the processed PhreshPhish sample of {phresh['sample_size_actual']} rows ({phresh['date_min']} to {phresh['date_max']}), the best random-split F1 was {_fmt_best(phresh_best)} Temporal testing, cross-dataset testing, drift detection, and adaptation have not been run. The baseline numbers are not evidence that those problems have been solved.

## Keywords

phishing URL detection; distribution shift; concept drift; dataset shift; baseline classification

## 1. Introduction

Phishing detectors trained on historical URLs can look accurate when the test set is a random subset of the same pool. Attackers change domains, hosting, and URL structure, and a second collection effort will not reproduce the first feature distribution. The research programme is to measure that degradation and to test whether limited labelled updates restore performance. This document only covers data preparation and static baselines.

## 2. Problem Statement

Three questions define the project:

1. Does a model trained on older phishing URLs lose performance on newer URLs?
2. Does a model trained on one public dataset generalise to another?
3. Can drift be detected, and can performance be recovered with a limited number of new labels?

Only the infrastructure required to ask these questions has been built. None of the three questions has been answered experimentally yet.

## 3. Literature Review

The PhiUSIIL data and feature design come from Prasad and Chandra, as recorded by the UCI repository: a tabular URL and webpage feature set with a similarity-index component, published in Computers & Security (DOI 10.1016/j.cose.2023.103545). PhreshPhish is described by its Hugging Face dataset card as a large HTML-URL benchmark and is linked to Dalton et al., arXiv:2507.10854. The card reports a train split of 498,255 samples and a test split of 168,060 samples, and version 1.0.1 adds samples collected between March and December 2025.

A broader review of drift detection and phishing benchmarks is not claimed here. Additional papers should be added only after their bibliographic details are checked. No citation in this draft was invented to fill the section.

## 4. Research Gap

Public phishing benchmarks are often evaluated with random splits or with a fixed official split. That does not measure temporal degradation, transfer to a dataset with a different feature schema, or recovery under a label budget. PhiUSIIL and PhreshPhish are useful for different parts of this gap: PhiUSIIL supplies a rich static feature table without timestamps, and PhreshPhish supplies URLs, labels, and collection dates. Their original columns are not interchangeable.

## 5. Research Objectives

1. Build a reproducible phishing URL pipeline on public data.
2. Establish baselines with several classical classifiers.
3. Measure temporal and cross-dataset shift.
4. Implement statistical drift detection.
5. Compare static, periodic, online, and drift-triggered updates.
6. Evaluate limited label budgets.
7. Report detection quality, false-positive rate, cost, and adaptation gain.

Objectives 1 and 2 are addressed in this draft for the samples described below. Objectives 3 to 7 are planned and not yet executed.

## 6. Proposed Methodology

Completed:

- Download PhiUSIIL from UCI and keep the zip unchanged.
- Read PhreshPhish columns `sha256`, `url`, `label`, and `date` from the parquet shards. Do not store HTML.
- Clean duplicates and invalid URLs, then sample with the configured seed.
- Fit preprocessing only on the training portion of each split.
- Evaluate a stratified random split and, where class balance allows, a domain-disjoint split.

Planned, not done:

- Chronological train/test cuts on PhreshPhish.
- Training on one dataset and testing on the other only after a shared feature schema is defined.
- Drift statistics such as population stability or Kolmogorov-Smirnov tests on feature streams.
- Periodic retraining, `partial_fit` updates with SGD, and retraining triggered by a drift signal.
- Label-budget curves.

The seed actually used is recorded in `configs/experiment_config.yaml`.

## 7. Dataset Description

PhiUSIIL contributed {phi['n_original_rows']} original rows, {phi['n_after_url_dedup']} after URL deduplication, and a stratified sample of {phi['sample_size_actual']}. Raw label 0 is phishing. The model uses the published numeric features and `TLD`, not the URL, domain, filename, or title.

PhreshPhish contributed {phresh['n_rows_retrieved']} retrieved rows before cleaning and {phresh['sample_size_actual']} sampled rows after URL validation and deduplication. Raw label `phish` is the positive class. Features are lexical URL statistics only. The sample date range is {phresh['date_min']} to {phresh['date_max']}. Monthly counts before and after sampling are stored in `data/metadata/phreshphish_processed.json`.

## 8. Baseline Experimental Setup

Python {environment['python']} on {environment['platform']}. Models: logistic regression, random forest, XGBoost, and SGDClassifier, with the parameters in `configs/experiment_config.yaml`. Metrics: accuracy, precision, recall, F1, ROC-AUC, false-positive rate, confusion matrix, training time, and prediction time. Predictions and fitted pipelines were saved.

## 9. Preliminary Results

### PhiUSIIL random split

{_metric_table(rows, 'phiusiil', 'split_random')}

### PhiUSIIL domain-disjoint split

{_metric_table(rows, 'phiusiil', 'split_domain')}

### PhreshPhish random split

{_metric_table(rows, 'phreshphish', 'split_random')}

### PhreshPhish domain-disjoint split

{_metric_table(rows, 'phreshphish', 'split_domain')}

These figures were computed from the saved prediction files. They are within-dataset baselines only.

## 10. Discussion

The random-split results describe how separable each prepared sample is under an i.i.d. assumption. They should not be read as operational phishing-detection performance under drift. Domain-disjoint evaluation reduces one leakage path, shared registrable domains, but it is still not a time-based evaluation. PhreshPhish is the dataset that can support that later test because the collection date was retained.

## 11. Limitations

The PhreshPhish sample is a subset of the full corpus, and HTML was discarded. PhiUSIIL has no dates. Hyperparameters were not searched. Class weighting and `scale_pos_weight` make the linear and boosting decisions sensitive to the training base rate. Keyword features depend on a hand-written list. No cross-dataset score is reported because the feature spaces differ.

## 12. Future Work

Phase 3 should cut PhreshPhish by time, train on earlier months, and test on later months. Phase 4 should add drift statistics and compare static, periodic, online, and drift-triggered updates at several label budgets. A cross-dataset test should wait until both sides use the same URL feature function, or until a separate aligned feature table is defined. Webpage features from PhiUSIIL should not be imputed into PhreshPhish.

## References

1. Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning. Computers & Security, 103545. https://doi.org/10.1016/j.cose.2023.103545. Citation text as provided by the UCI dataset record. UCI also lists the introductory paper year as 2024.
2. UCI Machine Learning Repository. PhiUSIIL Phishing URL Dataset. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset. Counts and the filename exclusion note were taken from this record.
3. Dalton, T., Gowda, H., Rao, G., Pargi, S., Hadj Khodabakhshi, A., Rombs, J., Jou, S., & Marwah, M. (2025). PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark. arXiv:2507.10854. https://arxiv.org/abs/2507.10854. Authors, title, and identifier were taken from the dataset card BibTeX. The PDF was not re-typeset for this draft.
4. Hugging Face dataset card, phreshphish/phreshphish. https://huggingface.co/datasets/phreshphish/phreshphish. Split sizes, field names, and version notes were taken from this card.

References that were not opened and checked are intentionally omitted.
"""


def _write_context_files(config, phi, phresh, rows, environment, phi_best, phresh_best, errors) -> None:
    root: Path = config["_root"]
    (root / "README.md").write_text(f"""# Drift-Aware Phishing URL Detection

Research title: Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning.

Phase 1 prepares PhiUSIIL and PhreshPhish. Phase 2 trains logistic regression, random forest, XGBoost, and SGDClassifier baselines. Temporal evaluation, cross-dataset transfer, drift detection, and adaptation are not implemented yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```

`python -m src.pipeline` reuses the PhiUSIIL zip and the PhreshPhish column cache when those files already exist, then rebuilds the processed samples, refits the models, and rewrites the reports.

## Sample sizes from the latest run

- PhiUSIIL: {phi['sample_size_actual']} rows after cleaning and stratified sampling.
- PhreshPhish: {phresh['sample_size_actual']} rows. Dates {phresh['date_min']} to {phresh['date_max']}.
- Best PhiUSIIL random-split model: {_fmt_best(phi_best)}
- Best PhreshPhish random-split model: {_fmt_best(phresh_best)}

## Layout

- `configs/experiment_config.yaml` — sources, sample sizes, seed, model parameters.
- `src/` — download, cleaning, URL features, training, evaluation, reports.
- `data/processed/` — modelling tables.
- `models/` — fitted pipelines.
- `results/` — metrics, predictions, graphs, confusion matrices.
- `reports/` — dataset analysis, baseline report, experiment log, paper draft.

## Reports

- [Dataset analysis](reports/dataset_analysis.md)
- [Baseline report](reports/baseline_report.md)
- [Experiment log](reports/experiment_log.md)
- [Paper draft](reports/research_paper_draft.md)
""")
    (root / "PROJECT_CONTEXT.md").write_text("""# Project Context

## Problem

Phishing URLs change. A model can score highly on a random split of historical data and still fail on newer URLs or on a dataset collected by someone else.

## Questions

1. Temporal shift: does performance fall when the test URLs are more recent than the training URLs?
2. Cross-dataset shift: does a model trained on one dataset generalise to another?
3. Adaptation: can drift be detected, and can a small number of new labels recover performance?

## Scope

The programme covers data preparation, baselines, temporal and cross-dataset tests, drift detection, and comparison of static, periodic, online, and drift-triggered updates under label budgets.

## Current status

Phase 1 and Phase 2 are the implemented portion. They produce within-dataset baselines on PhiUSIIL tabular features and on PhreshPhish URL features. They do not answer the three research questions above.

## Label convention

Model label 1 is phishing. Model label 0 is legitimate. PhiUSIIL raw label 0 is phishing. PhreshPhish raw label `phish` is phishing.
""")
    (root / "RESEARCH_METHODOLOGY.md").write_text("""# Research Methodology

## Completed

- PhiUSIIL: UCI zip preserved, URL deduplicated, stratified sample, published features except identifiers and title.
- PhreshPhish: column-selective read of URL, label, date, and sha256. HTML is not stored. Invalid URLs and duplicate URLs removed. Random sample preserves the cleaned time distribution in expectation.
- Splits: stratified random 80/20, and a domain-disjoint group split whose seed is chosen from a short search so both classes remain present. Seed search rule: smallest combined deviation of test fraction from 0.2 and positive rate from the sample base rate, among seeds `seed` through `seed + 19`.
- Preprocessing is inside each sklearn pipeline and is fit on that run's training rows only.
- Models: logistic regression, random forest, XGBoost, SGDClassifier.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, false-positive rate, confusion matrix, train time, predict time. Positive class is phishing.

## Not completed

- Temporal train/test using PhreshPhish `date`.
- Cross-dataset train/test. Feature sets are intentionally separate until an aligned schema exists.
- Drift detectors.
- Periodic, online, or drift-triggered retraining.
- Label-budget sweeps.
- Hyperparameter search.

## Leakage controls

Identifiers, dates, raw labels, and registrable domains are not model features. Domain overlap is measured for the random split and required to be empty for the domain-disjoint split. Exact feature-vector collisions between random-split train and test are recorded.
""")
    (root / "DATASET_GUIDE.md").write_text(f"""# Dataset Guide

## PhiUSIIL

- Page: {config['phiusiil']['uci_page']}
- File: `data/raw/phiusiil/phiusiil_phishing_url_dataset.zip` (unchanged archive) and the extracted CSV.
- Original rows: {phi['n_original_rows']}. Sample: {phi['sample_size_actual']}.
- Raw labels: 0 phishing ({config['phiusiil']['expected_raw_counts']['0']}), 1 legitimate ({config['phiusiil']['expected_raw_counts']['1']}), confirmed by matching the UCI abstract before deduplication.
- No timestamp column.
- Model features: {len(phi['model_features'])} columns, listed in `data/metadata/phiusiil_processed.json`.
- Excluded: `FILENAME`, `URL`, `Domain`, `Title`, labels, and `registrable_domain`.
- Sample limit in config: {config['phiusiil']['sample_size']}.

## PhreshPhish

- Card: {config['phreshphish']['hf_page']}
- Paper: {config['phreshphish']['paper_url']}
- Fields used: `sha256`, `url`, `label`, `date`. Fields not stored: `html`, `target`, `lang`, `lang_score`.
- Retrieved rows: {phresh['n_rows_retrieved']}. Cleaned: {phresh['n_after_cleaning']}. Sample: {phresh['sample_size_actual']}.
- Raw labels: `phish` and `benign`.
- Sample date range: {phresh['date_min']} to {phresh['date_max']}.
- URL feature definitions are in `data/metadata/phreshphish_processed.json`.
- Invalid URL rule: require an http or https scheme and a hostname, and length at most 8000.
- Dedup rule: keep the earliest dated row for each exact URL.
- Sample limit in config: {config['phreshphish']['sample_size']}.

## Compatibility

Do not join the two processed feature matrices column-wise. Cross-dataset evaluation is future work and needs an explicit shared feature function.
""")
    error_line = "None." if not errors else json.dumps(errors)
    (root / "IMPLEMENTATION_PLAN.md").write_text(f"""# Implementation Plan

## Phase 1 — data (completed for the samples in this run)

Download, inspect, clean, sample, and save both datasets. Metadata: `data/metadata/`.

## Phase 2 — baselines (completed unless errors remain)

Train four models on each prepared dataset and on the random and domain-disjoint splits. Errors from the latest run: {error_line}

Latest environment: Python {environment['python']}.

## Phase 3 — temporal shift (next)

On PhreshPhish only, define a cutoff date from the saved `date` column, train on earlier URLs, and test on later URLs. Repeat for more than one cutoff. Do not use the date as a feature. Compare against the Phase 2 random-split scores.

## Phase 4 — drift detection

Compute feature-level drift between the temporal train window and later windows. Candidates: PSI and Kolmogorov-Smirnov statistics. Record detections without feeding the label into the detector.

## Phase 5 — adaptation

Compare a frozen model, periodic refits, SGD `partial_fit`, and refits triggered by the Phase 4 detector. Vary the number of new labels.

## Phase 6 — cross-dataset

Apply `src/features/url_features.py` to PhiUSIIL URLs and evaluate transfer in both directions. Do not use PhiUSIIL webpage features in that comparison.

## Module map

- `src/data/download.py` — retrieval.
- `src/data/preprocess.py` — cleaning, sampling, splits.
- `src/features/url_features.py` — PhreshPhish features.
- `src/models/train_baseline.py` — model fit.
- `src/models/evaluate.py` — metrics and plots.
- `src/reporting/build_reports.py` — reports and these context files.
- `src/pipeline.py` — entry point.
""")
    (root / "AGENT_RULES.md").write_text("""# Agent Rules

- Read this file and only the context file needed for the current task before editing.
- Do not reload every source file or dataset into the conversation. Inspect the specific module you will change.
- Reuse `src/` and `configs/experiment_config.yaml`. Do not create a second training path.
- Keep new functions small and in the existing module that owns that step.
- Add a dependency only if the current pipeline cannot do the job.
- Keep seeds, sample sizes, and paths in the YAML config. Paths stay project-relative.
- Do not invent metrics, citations, or dataset counts. Read `results/metrics/summary.json` and `data/metadata/*_processed.json`.
- Update `reports/` and the context file that changed when a method or result changes.
- Do not overwrite `results/` or `models/` without recording the reason in `reports/experiment_log.md`.
- Do not claim temporal, cross-dataset, drift, or adaptation results unless that experiment was run and saved.
- Do not put labels, dates, identifiers, or HTML into model features.
- Fit preprocessing on the training split only.
- Leave the PhiUSIIL zip unchanged. Do not download PhreshPhish HTML for routine runs.
""")

