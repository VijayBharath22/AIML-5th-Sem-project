# Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning

## Abstract

Phishing URLs change over time, so a classifier that scores well on a random split can still fail on later or independently collected data. This draft reports Phase 1 and Phase 2 of that study: preparation of the PhiUSIIL and PhreshPhish datasets, and CPU baselines for logistic regression, random forest, XGBoost, and SGDClassifier. On the processed PhiUSIIL sample of 70000 rows, the best random-split F1 was xgboost with F1 0.9999, ROC-AUC 1.0000, precision 0.9998, recall 1.0000, FPR 0.0001, accuracy 0.9999. On the processed PhreshPhish sample of 40000 rows (2024-07-02 00:00:00+00:00 to 2025-12-16 00:00:00+00:00), the best random-split F1 was xgboost with F1 0.9065, ROC-AUC 0.9717, precision 0.9102, recall 0.9028, FPR 0.0730, accuracy 0.9161. Temporal testing, cross-dataset testing, drift detection, and adaptation have not been run. The baseline numbers are not evidence that those problems have been solved.

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

PhiUSIIL contributed 235795 original rows, 235370 after URL deduplication, and a stratified sample of 70000. Raw label 0 is phishing. The model uses the published numeric features and `TLD`, not the URL, domain, filename, or title.

PhreshPhish contributed 666315 retrieved rows before cleaning and 40000 sampled rows after URL validation and deduplication. Raw label `phish` is the positive class. Features are lexical URL statistics only. The sample date range is 2024-07-02 00:00:00+00:00 to 2025-12-16 00:00:00+00:00. Monthly counts before and after sampling are stored in `data/metadata/phreshphish_processed.json`.

## 8. Baseline Experimental Setup

Python 3.14.10 on Linux-7.3.3-200.fc44.x86_64-x86_64-with-glibc2.43. Models: logistic regression, random forest, XGBoost, and SGDClassifier, with the parameters in `configs/experiment_config.yaml`. Metrics: accuracy, precision, recall, F1, ROC-AUC, false-positive rate, confusion matrix, training time, and prediction time. Predictions and fitted pipelines were saved.

## 9. Preliminary Results

### PhiUSIIL random split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.9996 | 0.9997 | 0.9993 | 0.9995 | 1.0000 | 0.0002 | 0.62 | 0.022 |
| random_forest | 0.9999 | 1.0000 | 0.9998 | 0.9999 | 1.0000 | 0.0000 | 2.38 | 0.046 |
| xgboost | 0.9999 | 0.9998 | 1.0000 | 0.9999 | 1.0000 | 0.0001 | 1.18 | 0.022 |
| sgd_classifier | 0.9994 | 0.9993 | 0.9993 | 0.9993 | 1.0000 | 0.0005 | 0.61 | 0.020 |


### PhiUSIIL domain-disjoint split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.9995 | 0.9995 | 0.9993 | 0.9994 | 1.0000 | 0.0004 | 0.54 | 0.022 |
| random_forest | 0.9998 | 1.0000 | 0.9995 | 0.9997 | 1.0000 | 0.0000 | 2.44 | 0.046 |
| xgboost | 0.9998 | 0.9995 | 1.0000 | 0.9997 | 1.0000 | 0.0004 | 0.90 | 0.021 |
| sgd_classifier | 0.9996 | 0.9997 | 0.9993 | 0.9995 | 1.0000 | 0.0002 | 0.55 | 0.015 |


### PhreshPhish random split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8488 | 0.8231 | 0.8459 | 0.8343 | 0.9057 | 0.1489 | 0.48 | 0.009 |
| random_forest | 0.9144 | 0.9110 | 0.8976 | 0.9042 | 0.9737 | 0.0719 | 0.66 | 0.045 |
| xgboost | 0.9161 | 0.9102 | 0.9028 | 0.9065 | 0.9717 | 0.0730 | 0.22 | 0.007 |
| sgd_classifier | 0.8484 | 0.8229 | 0.8451 | 0.8339 | 0.9050 | 0.1489 | 0.16 | 0.004 |


### PhreshPhish domain-disjoint split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8396 | 0.8119 | 0.8417 | 0.8266 | 0.9117 | 0.1622 | 0.50 | 0.006 |
| random_forest | 0.8915 | 0.9038 | 0.8517 | 0.8770 | 0.9515 | 0.0754 | 0.64 | 0.045 |
| xgboost | 0.8887 | 0.8939 | 0.8565 | 0.8748 | 0.9503 | 0.0845 | 0.19 | 0.007 |
| sgd_classifier | 0.8329 | 0.8050 | 0.8340 | 0.8192 | 0.9088 | 0.1680 | 0.16 | 0.003 |


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
