# Project Context

## Problem

Phishing URLs change continuously. A model can score highly on a random split of historical data and still fail on newer URLs or on a dataset collected by an independent party.

## Questions

1. **Temporal shift:** Does model performance fall when test URLs are more recent than training URLs?
2. **Cross-dataset shift:** Does a model trained on one public dataset generalize to another independently collected dataset?
3. **Adaptation:** Can distribution drift be detected, and can performance be recovered with a limited number of new labels?

## Scope

The programme covers data acquisition, preprocessing, baseline model training, baseline methodology auditing, temporal and cross-dataset shift evaluations, statistical drift detection, and adaptive retraining (periodic, online, drift-triggered) under label budgets.

## Current Status

- **Phase 1 (Data Acquisition & Preparation):** Complete. PhiUSIIL (70,000 samples) and PhreshPhish (40,000 samples) are processed and metadata-documented.
- **Phase 2 (Baseline Model Training & Evaluation):** Complete. Baselines for Logistic Regression, Random Forest, XGBoost, and SGDClassifier trained and verified on random and domain-disjoint splits.
- **Methodology Audit:** Complete (`reports/methodology_audit.md`). Identified and excluded target leakage feature (`URLSimilarityIndex`) in PhiUSIIL; mapped valid chronological windows in PhreshPhish for Phase 3.
- **Phase 3+:** Pending. None of the three core research questions above have been evaluated experimentally yet.

## Label Convention

- **Model Label 1:** Phishing (Positive class).
- **Model Label 0:** Legitimate (Negative class).
- **PhiUSIIL Raw:** `0` is phishing, `1` is legitimate (inverted during preprocessing).
- **PhreshPhish Raw:** `'phish'` is phishing, `'benign'` is legitimate.
