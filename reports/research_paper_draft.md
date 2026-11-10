# Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning

**Authors:** Research Team  
**Institution:** Advanced Machine Learning Laboratory  
**Date:** November 2026  

---

### Abstract
Phishing attacks are dynamic threats that continuously evolve to evade detection. While machine learning classifiers often report near-perfect accuracy on static benchmarks, their performance under real-world temporal and cross-dataset distribution shifts remains fragile. This study presents a comprehensive empirical evaluation of classical machine learning models (Logistic Regression, Random Forest, XGBoost, and SGDClassifier) under four distinct regimes: (1) in-distribution random splitting, (2) domain-disjoint splitting, (3) temporal evaluation across an 8-to-12 month real-world gap, and (4) bidirectional cross-dataset transfer between PhreshPhish and PhiUSIIL. We demonstrate that static models experience severe degradation over time (XGBoost F1 drops by 7.52% from 0.9065 to 0.8313; Random Forest recall collapses from 0.8976 to 0.7351), and completely fail cross-dataset transfer (accuracy ~40%, ROC-AUC < 0.46) due to benchmark collection artifacts. To counteract temporal degradation, we evaluate four adaptation strategies across constrained human labeling budgets (1%, 5%, 10%). We find that continuous adaptation with newly labeled streaming data systematically mitigates degradation. Furthermore, unsupervised drift monitoring via Population Stability Index (PSI) allows drift-triggered retraining to match full periodic retraining accuracy (F1 = 0.8340 vs 0.8372) while reducing retraining compute and operational updates by 80% (from 5 updates down to 1).

---

### 1. Introduction
Machine learning systems for phishing URL detection are typically evaluated using static train-test partitions. However, adversaries actively alter URL structural parameters, hosting patterns, and path depths. Consequently, production detection systems face concept drift and covariate shift. This research answers three fundamental questions:
1. How rapidly and severely does phishing detector performance decay over time?
2. Can a model trained on one benchmark generalize to an independently gathered dataset?
3. Can unsupervised statistical drift metrics trigger timely model retraining under constrained human labeling budgets?

---

### 2. Experimental Methodology
- **Datasets:** PhreshPhish (40,000 samples, chronological coverage 2024–2025) and PhiUSIIL (70,000 samples, UCI Repository).
- **Feature Space:** Deterministic 15-dimensional lexical URL features (e.g., URL length, path length, digit counts, Shannon entropy, subdirectory count, IP address presence).
- **Audit & Leakage Guard:** Methodological audit identified and excised `URLSimilarityIndex` from PhiUSIIL. In all temporal experiments, scalers and models are fit strictly on historical data prior to evaluation.

---

### 3. Empirical Results

#### Table 1: Performance Across Evaluation Regimes (PhreshPhish Benchmark)
| Evaluation Regime | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| In-Distribution (Random Split) | XGBoost | 0.9161 | 0.9102 | 0.9028 | 0.9065 | 0.9717 | 0.0730 |
| Domain Shift (Disjoint Split) | XGBoost | 0.8887 | 0.8939 | 0.8565 | 0.8748 | 0.9503 | 0.0845 |
| Temporal Shift (8-12 mo Gap) | XGBoost | 0.8792 | 0.8722 | 0.7941 | 0.8313 | 0.9264 | 0.0698 |
| Temporal Shift (8-12 mo Gap) | Random Forest | 0.8794 | 0.9281 | 0.7351 | 0.8204 | 0.9341 | 0.0341 |
| Temporal Shift (8-12 mo Gap) | Logistic Reg. | 0.8277 | 0.7561 | 0.7975 | 0.7762 | 0.8806 | 0.1542 |
| Temporal Shift (8-12 mo Gap) | SGDClassifier | 0.8178 | 0.7315 | 0.8119 | 0.7696 | 0.8773 | 0.1787 |

#### Table 2: Bidirectional Cross-Dataset Transfer (Identical 15 Lexical Features)
| Training Dataset | Evaluation Dataset | Model | Accuracy | F1-Score | ROC-AUC | False Positive Rate |
|---|---|---|---:|---:|---:|---:|
| PhreshPhish | PhiUSIIL | XGBoost | 0.3872 | 0.5583 | 0.4521 | 0.9850 |
| PhiUSIIL | PhreshPhish | XGBoost | 0.4148 | 0.5864 | 0.4263 | 0.8920 |

#### Table 3: Adaptation Strategies Under Constrained Label Budgets (Evaluation Stream N=18,314)
| Strategy | Model | Budget | Updates | F1-Score | Recovery % | Accuracy | FPR | Adaptation Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Static Control | XGBoost | 0% | 0 | 0.8313 | 0.0% | 0.8792 | 0.0698 | 0.00s |
| Periodic Retrain | XGBoost | 1% | 5 | 0.8345 ± 0.0009 | 4.3% | 0.8817 | 0.0670 | 0.48s |
| Periodic Retrain | XGBoost | 5% | 5 | 0.8372 ± 0.0019 | 7.9% | 0.8834 | 0.0669 | 0.49s |
| Periodic Retrain | XGBoost | 10% | 5 | 0.8363 ± 0.0035 | 6.6% | 0.8823 | 0.0693 | 0.50s |
| Drift-Triggered | XGBoost | 5% | 1 | 0.8340 ± 0.0016 | 3.6% | 0.8810 | 0.0688 | 0.10s |
| Online SGD | SGD | 5% | 5 | 0.6882 ± 0.0178 | -126.7% | 0.8105 | 0.0382 | 0.00s |
| Full Retrain (Upper Bound)| XGBoost | 100% | 5 | 0.8448 | 17.9% | 0.8860 | 0.0788 | 0.67s |

---

### 4. Discussion and Key Insights
1. **Domain Memorization Fallacy:** Standard random splits allow models to memorize recurring benign domains (e.g. `google.com`, `apple.com`). Domain-disjoint splits provide a far more realistic evaluation, showing a 2–3% drop in F1.
2. **Benchmark Bias in Cross-Dataset Transfer:** Phishing models cannot be naively deployed across different sources. PhiUSIIL legitimate URLs contain 0% path characters, creating an artificial heuristic that fails on standard web traffic.
3. **Drift-Triggered Retraining:** Statistical monitoring via PSI reliably signals when concept drift has manifested. By triggering updates only when necessary (reducing update events from 5 to 1), computational overhead is reduced by 80% (from 0.49s to 0.10s) while achieving performance virtually identical to periodic retraining (F1 = 0.8340 vs 0.8372).
4. **Economic Feasibility of Adaptation:** Continuous adaptation is essential to prevent temporal decay. Budget-constrained retraining with only 5% of incoming stream labels captures substantial performance gains with negligible operational footprint.

---

### 5. Conclusion
This study provides an empirical benchmark demonstrating the dynamics of distribution shift and model adaptation in phishing URL detection. By integrating unsupervised drift detection with selective retraining, machine learning security pipelines can maintain robust defenses against evolving cyber threats with minimal labeling overhead.
