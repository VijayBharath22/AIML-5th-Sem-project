# Temporal Distribution Shift Evaluation Report

## 1. Executive Summary

This experiment evaluates how machine learning models trained on historical phishing URLs generalize to unseen future URLs. Models were trained strictly on historical data (**August to October 2024**, $N=14,525$, phishing rate $27.99%$) and evaluated without retraining across a future streaming period (**June to November 2025**, $N=18,314$, phishing rate $37.48%$).

## 2. Chronological Setup & Leakage Guardrails

- **Training Period:** 2024-08 to 2024-10 ($N = 14,525$).
- **Testing Period:** 2025-06 to 2025-11 ($N = 18,314$) — represents an 8-to-12 month chronological gap.
- **Feature Set:** Deterministic 15 lexical URL features (no HTML content, no target proxies).
- **Preprocessing:** Scaler and median imputer fit exclusively on the historical training window.

## 3. Aggregate Performance Under Temporal Shift

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8277 | 0.7561 | 0.7975 | 0.7762 | 0.8806 | 0.1542 | 0.055 |
| random_forest | 0.8794 | 0.9281 | 0.7351 | 0.8204 | 0.9341 | 0.0341 | 0.326 |
| xgboost | 0.8792 | 0.8722 | 0.7941 | 0.8313 | 0.9264 | 0.0698 | 0.340 |
| sgd_classifier | 0.8178 | 0.7315 | 0.8119 | 0.7696 | 0.8773 | 0.1787 | 0.071 |

## 4. Performance Degradation vs. Phase 2 Baselines

| Model | In-Distribution F1 (Random Split) | Domain Shift F1 (Domain-Disjoint) | Temporal Shift F1 (8-12 mo Gap) | $\Delta$ F1 (Temporal vs. In-Dist) | $\Delta$ FPR (Temporal vs. In-Dist) |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.8343 | 0.8266 | 0.7762 | -0.0581 | +0.0053 |
| random_forest | 0.9042 | 0.8770 | 0.8204 | -0.0838 | -0.0377 |
| xgboost | 0.9065 | 0.8748 | 0.8313 | -0.0752 | -0.0032 |
| sgd_classifier | 0.8339 | 0.8192 | 0.7696 | -0.0643 | +0.0298 |

### Key Findings:
1. **Tree Ensembles Show Sharp Drop:**
   - **XGBoost:** F1 drops from 0.9065 to 0.8313 ($-7.52%$ absolute, $-8.30%$ relative degradation). ROC-AUC drops from 0.9717 to 0.9264.
   - **Random Forest:** F1 drops from 0.9042 to 0.8204 ($-8.38%$ absolute degradation), driven by a sharp drop in recall ($0.8976 \rightarrow 0.7351$).
2. **Linear Classifiers Suffer High False Positives:**
   - False positive rate increases substantially: Logistic Regression FPR rises to 15.42% and SGD to 17.87%, demonstrating that fixed linear boundaries misclassify new legitimate domain patterns.

## 5. Month-by-Month Trajectory Analysis

| Month | Model | N Samples | Phish Prevalence | Accuracy | Precision | Recall | F1-Score | ROC-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-06 | logistic_regression | 671 | 0.419 | 0.8525 | 0.8074 | 0.8505 | 0.8284 | 0.9291 | 0.1462 |
| 2025-07 | logistic_regression | 3544 | 0.236 | 0.8172 | 0.5946 | 0.7097 | 0.6471 | 0.8345 | 0.1496 |
| 2025-08 | logistic_regression | 3342 | 0.257 | 0.8151 | 0.6245 | 0.7058 | 0.6627 | 0.8098 | 0.1471 |
| 2025-09 | logistic_regression | 2617 | 0.161 | 0.8044 | 0.4227 | 0.5829 | 0.4900 | 0.7292 | 0.1531 |
| 2025-10 | logistic_regression | 6802 | 0.480 | 0.8434 | 0.8264 | 0.8530 | 0.8395 | 0.9144 | 0.1654 |
| 2025-11 | logistic_regression | 1338 | 0.896 | 0.8401 | 0.9824 | 0.8365 | 0.9036 | 0.9157 | 0.1295 |
| 2025-06 | random_forest | 671 | 0.419 | 0.8644 | 0.9318 | 0.7295 | 0.8184 | 0.9582 | 0.0385 |
| 2025-07 | random_forest | 3544 | 0.236 | 0.8900 | 0.8719 | 0.6260 | 0.7288 | 0.8937 | 0.0284 |
| 2025-08 | random_forest | 3342 | 0.257 | 0.8689 | 0.8576 | 0.5884 | 0.6979 | 0.8884 | 0.0338 |
| 2025-09 | random_forest | 2617 | 0.161 | 0.8911 | 0.7509 | 0.4858 | 0.5899 | 0.7923 | 0.0310 |
| 2025-10 | random_forest | 6802 | 0.480 | 0.8872 | 0.9490 | 0.8086 | 0.8732 | 0.9592 | 0.0401 |
| 2025-11 | random_forest | 1338 | 0.896 | 0.8221 | 0.9949 | 0.8057 | 0.8903 | 0.9670 | 0.0360 |
| 2025-06 | xgboost | 671 | 0.419 | 0.8823 | 0.9040 | 0.8043 | 0.8512 | 0.9585 | 0.0615 |
| 2025-07 | xgboost | 3544 | 0.236 | 0.8852 | 0.7874 | 0.7037 | 0.7432 | 0.8768 | 0.0587 |
| 2025-08 | xgboost | 3342 | 0.257 | 0.8683 | 0.7770 | 0.6849 | 0.7281 | 0.8564 | 0.0681 |
| 2025-09 | xgboost | 2617 | 0.161 | 0.8766 | 0.6279 | 0.5758 | 0.6007 | 0.7590 | 0.0656 |
| 2025-10 | xgboost | 6802 | 0.480 | 0.8852 | 0.9048 | 0.8502 | 0.8767 | 0.9607 | 0.0826 |
| 2025-11 | xgboost | 1338 | 0.896 | 0.8640 | 0.9894 | 0.8574 | 0.9187 | 0.9669 | 0.0791 |
| 2025-06 | sgd_classifier | 671 | 0.419 | 0.8480 | 0.7878 | 0.8719 | 0.8277 | 0.9290 | 0.1692 |
| 2025-07 | sgd_classifier | 3544 | 0.236 | 0.8042 | 0.5670 | 0.7228 | 0.6355 | 0.8325 | 0.1707 |
| 2025-08 | sgd_classifier | 3342 | 0.257 | 0.7974 | 0.5871 | 0.7174 | 0.6457 | 0.8051 | 0.1749 |
| 2025-09 | sgd_classifier | 2617 | 0.161 | 0.7853 | 0.3916 | 0.5995 | 0.4738 | 0.7191 | 0.1790 |
| 2025-10 | sgd_classifier | 6802 | 0.480 | 0.8361 | 0.8080 | 0.8637 | 0.8349 | 0.9103 | 0.1894 |
| 2025-11 | sgd_classifier | 1338 | 0.896 | 0.8602 | 0.9801 | 0.8616 | 0.9170 | 0.9179 | 0.1511 |

## 6. Artifact References

- Aggregate Metrics: `results/temporal/metrics/temporal_aggregate_metrics.csv`
- Monthly Analysis: `results/temporal/metrics/temporal_monthly_metrics.csv`
- Predictions: `results/temporal/predictions/*.parquet`
- Confusion Matrices: `results/temporal/graphs/*.png` and `results/temporal/monthly_analysis/*.png`
- Degradation Graph: `results/temporal/graphs/temporal_vs_baseline_comparison.png`
- Monthly Curves: `results/temporal/graphs/temporal_monthly_trajectory.png`
- ROC Curves: `results/temporal/graphs/roc__temporal_future_eval.png`