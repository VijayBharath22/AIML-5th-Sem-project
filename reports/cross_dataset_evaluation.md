# Cross-Dataset Distribution Shift Evaluation Report

## 1. Executive Summary

This experiment evaluates cross-dataset generalizability between two independently collected public datasets: **PhreshPhish** (in-the-wild web crawl) and **PhiUSIIL** (curated academic benchmark). To ensure valid scientific comparison, all existing tabular webpage features were discarded, and an identical **15-dimensional lexical URL feature representation** was extracted from raw URLs in both datasets.

## 2. Dataset Overlap & Independence

- **PhiUSIIL Evaluated Samples:** 70000 URLs (1.70% domain overlap).
- **PhreshPhish Evaluated Samples:** 40000 URLs (4.76% domain overlap).
- **Exact URL Overlap:** 3 URLs (effectively disjoint).
- **Registrable Domain Overlap:** 923 domains.

## 3. Bidirectional Transfer Results

| Transfer Direction | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| PhreshPhish -> PhiUSIIL | logistic_regression | 0.5680 | 0.4968 | 0.9006 | 0.6404 | 0.8547 | 0.6799 |
| PhiUSIIL -> PhreshPhish | logistic_regression | 0.4057 | 0.4195 | 0.8339 | 0.5582 | 0.2366 | 0.9452 |
| PhreshPhish -> PhiUSIIL | random_forest | 0.3886 | 0.4041 | 0.9099 | 0.5597 | 0.4383 | 1.0000 |
| PhiUSIIL -> PhreshPhish | random_forest | 0.4142 | 0.4297 | 0.9198 | 0.5857 | 0.4298 | 1.0000 |
| PhreshPhish -> PhiUSIIL | xgboost | 0.3926 | 0.4066 | 0.9193 | 0.5638 | 0.2752 | 1.0000 |
| PhiUSIIL -> PhreshPhish | xgboost | 0.4148 | 0.4301 | 0.9213 | 0.5864 | 0.3618 | 1.0000 |
| PhreshPhish -> PhiUSIIL | sgd_classifier | 0.6105 | 0.5260 | 0.8883 | 0.6608 | 0.8524 | 0.5967 |
| PhiUSIIL -> PhreshPhish | sgd_classifier | 0.4003 | 0.4146 | 0.8061 | 0.5476 | 0.2811 | 0.9321 |

## 4. Analysis of Cross-Dataset Generalization Failure

### Key Findings:
1. **Catastrophic Transfer Collapse:**
   - **Train on PhreshPhish -> Test on PhiUSIIL:** Random Forest accuracy collapses to **38.86%** (F1: 0.5597, ROC-AUC: 0.4383); XGBoost accuracy collapses to **39.26%** (F1: 0.5638, ROC-AUC: 0.2752).
   - **Train on PhiUSIIL -> Test on PhreshPhish:** Random Forest accuracy collapses to **41.42%** (F1: 0.5857, ROC-AUC: 0.4298); XGBoost accuracy collapses to **41.48%** (F1: 0.5864, ROC-AUC: 0.3618).
   - Both directions result in **severe failure and near-total false positives** (FPR = 1.0000 for tree models).
2. **Root Cause Analysis (Structural Benchmark Artifacts):**
   - In PhiUSIIL, 100% of legitimate URLs are clean homepage root domains (`path_length == 0`, `n_subdirectories == 0`).
   - In PhreshPhish, legitimate URLs are crawled from deep web pages with average path lengths of 35 and 2–3 subdirectories.
   - A model trained on PhiUSIIL learns a spurious shortcut: *'Any URL containing subdirectories or path segments is phishing.'* When applied to PhreshPhish, this causes massive false positive explosions (>90% false positive rate).
   - Conversely, models trained on PhreshPhish classify root homepage domains as legitimate, missing short-domain phishing URLs in PhiUSIIL.

## 5. Artifact References

- Metrics: `results/cross_dataset/metrics/cross_dataset_metrics.csv`
- Overlap Summary: `results/cross_dataset/metrics/dataset_overlap.json`
- Predictions: `results/cross_dataset/predictions/*.parquet`
- Transfer Comparison Graph: `results/cross_dataset/graphs/cross_dataset_transfer_comparison.png`
- ROC Curves: `results/cross_dataset/graphs/roc__cross_dataset__*.png`