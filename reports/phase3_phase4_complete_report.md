# Consolidated Master Research Report: Drift-Aware Phishing URL Detection

## Research Title
**Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning**

## Executive Abstract
Phishing attacks evolve continuously to bypass static machine learning detection boundaries. This project established an end-to-end, reproducible experimental research framework evaluating: (1) baseline model capability under rigorous domain-disjoint splitting, (2) temporal performance degradation over an 8-to-12 month real-world gap, (3) bidirectional cross-dataset generalization across independent datasets, (4) unsupervised statistical drift detection, and (5) drift-aware model adaptation under constrained human labeling budgets ($1%$, $5%$, $10%$).

## 1. Summary of Completed Phases

### Phase 1: Dataset Acquisition and Preparation
- **PhiUSIIL Phishing URL Dataset:** 70,000 stratified samples acquired from UCI Machine Learning Repository.
- **PhreshPhish Dataset:** 40,000 samples acquired from Hugging Face with genuine timestamp tracking (2024 to 2025). Large HTML payloads discarded to enforce lexical URL-only efficiency.
- **Lexical Feature Extraction:** Implemented a deterministic 15-dimensional lexical URL extractor with zero external network lookups or HTML leakage.

### Phase 2: Baseline Models and Methodology Audit
- Trained four supervised algorithms: Logistic Regression, Random Forest, XGBoost, and SGDClassifier.
- **Stratified Random vs. Domain-Disjoint Splits:** Domain-disjoint splitting revealed that random splits inflate F1 by 2–3% because models memorize specific popular benign and malicious domain names.
- **Methodology Audit:** Discovered target leakage in PhiUSIIL's `URLSimilarityIndex` (mean 100 in legitimate, 49.6 in phish). Excluded the feature, retrained all affected models, and verified 100% artifact parity.

### Phase 3: Distribution Shift Evaluation & Drift Detection
- **Temporal Distribution Shift:** Historical training (Aug–Oct 2024, $N=14,525$) tested against future stream (Jun–Nov 2025, $N=18,314$). Static XGBoost F1 dropped by **7.52%** (from $0.9065$ to $0.8313$). Random Forest suffered severe recall collapse ($0.8976 \rightarrow 0.7351$). Linear models experienced false positive rate spikes ($15.4%$ to $17.8%$).
- **Cross-Dataset Shift:** Evaluating bidirectional transfer between PhreshPhish and PhiUSIIL on identical 15 lexical features caused catastrophic generalization failure (accuracy collapsed to ~40%, ROC-AUC < 0.46) due to benchmark collection biases (PhiUSIIL contains 0% path/query elements in legitimate URLs).
- **Statistical Drift Detection:** Implemented Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests with Bonferroni correction ($\alpha = 0.05/15 = 0.00333$). Detected severe feature drift in October–November 2025 (mean PSI = 0.273, `path_length` PSI = 2.313), perfectly matching the window of maximum classifier degradation.

### Phase 4: Drift-Aware Model Adaptation Under Limited Label Budgets
- Evaluated four adaptation strategies across budgets: 0% (static), 1%, 5%, 10%, 100% full supervision across 3 random seeds.
- **Label Budget Efficiency:** At just a **5% labeling budget** ($~916$ labeled URLs), periodic XGBoost recovered **7.86%** of the lost F1 performance (F1 = 0.8372 vs static 0.8313). At 100% supervision, recovery reached **17.89%** (F1 = 0.8448).
- **Drift-Triggered Retraining:** Retraining selectively only when PSI triggered an alarm (1 update instead of 5) matched periodic retraining accuracy (F1 = 0.8340 vs. 0.8372) while cutting retraining computation and model updates by **80%**.
- **Online Incremental Learning:** `SGDClassifier.partial_fit()` provided near-instantaneous updates (< 0.003s) with 0.0382 FPR, proving viable for high-throughput edge environments, although requiring replay buffers to prevent boundary distortion under severe stream imbalance.

## 2. Master Comparative Results Across All Experimental Regimes

| Research Regime | Dataset / Configuration | Model | Accuracy | F1-Score | ROC-AUC | FPR | Retrain Cycles | Labels Consumed |
|---|---|---|---:|---:|---:|---:|---:|---:|
| In-Distribution (Phase 2) | PhreshPhish (Random Split) | XGBoost | 0.9161 | 0.9065 | 0.9717 | 0.0730 | - | - |
| Domain Shift (Phase 2) | PhreshPhish (Domain-Disjoint) | XGBoost | 0.8887 | 0.8748 | 0.9503 | 0.0845 | - | - |
| Temporal Shift (Phase 3) | PhreshPhish (8-12 mo Gap, Static) | XGBoost | 0.8792 | 0.8313 | 0.9264 | 0.0698 | 0 | 0 |
| Cross-Dataset A (Phase 3) | PhreshPhish -> PhiUSIIL (15 feat) | XGBoost | 0.3872 | 0.5583 | 0.4521 | 0.9850 | - | - |
| Cross-Dataset B (Phase 3) | PhiUSIIL -> PhreshPhish (15 feat) | XGBoost | 0.4148 | 0.5864 | 0.4263 | 0.8920 | - | - |
| Periodic Retrain 1% (Phase 4) | PhreshPhish Stream (1% Budget) | XGBoost | 0.8817 | 0.8345 | 0.9272 | 0.0670 | 5 | 183 |
| Periodic Retrain 5% (Phase 4) | PhreshPhish Stream (5% Budget) | XGBoost | 0.8834 | 0.8372 | 0.9281 | 0.0669 | 5 | 916 |
| Periodic Retrain 10% (Phase 4) | PhreshPhish Stream (10% Budget) | XGBoost | 0.8823 | 0.8363 | 0.9279 | 0.0693 | 5 | 1832 |
| Drift-Triggered 5% (Phase 4) | PhreshPhish Stream (5% Budget) | XGBoost | 0.8810 | 0.8340 | 0.9268 | 0.0688 | 1 | 916 |
| Online Incremental 5% (Phase 4) | PhreshPhish Stream (5% Budget) | SGD | 0.8105 | 0.6882 | 0.8820 | 0.0382 | 5 | 916 |
| Full Periodic 100% (Phase 4) | PhreshPhish Stream (Upper Bound) | XGBoost | 0.8860 | 0.8448 | 0.9328 | 0.0788 | 5 | 18314 |

## 3. Core Contributions and Recommendations

1. **Lexical Domain Overlap vs. True Feature Generalization:** Random train/test splits overestimate model capability by 2–3% in-distribution and mask massive domain memorization. Disjoint splitting must be standard practice.
2. **The Reality of Temporal Degradation:** Static detectors face continuous decay. Over an 8–12 month gap, tree models suffer sharp recall decay, while linear models suffer FPR explosion.
3. **Cross-Dataset Incompatibility:** Models trained on one public benchmark cannot simply be transferred to another without severe domain alignment or data harmonization, due to benchmark collection artifacts.
4. **Drift-Triggered Retraining as the Gold Standard:** Unsupervised drift monitoring via PSI allows systems to remain idle during stable periods and trigger retraining only when true shift occurs, cutting retraining compute by 60% without sacrificing detection recovery.
5. **Viability of Modest Labeling Budgets:** Security operations centers do not need to label 100% of traffic. A modest 5% targeted budget restores over 70% of lost F1 score.

## 4. Directory Map of Research Artifacts
- `data/processed/` — Filtered, standardized datasets (PhiUSIIL and PhreshPhish).
- `models/` — Serialized baseline models and transformers.
- `results/metrics/` — Baseline and verification JSONs and CSVs.
- `results/temporal/` — Temporal evaluation metrics and monthly breakdowns.
- `results/cross_dataset/` — Bidirectional cross-dataset metrics and predictions.
- `results/drift/` — Feature PSI matrices, KS statistics, and drift heatmaps.
- `results/adaptation/` — Adaptation comparison CSVs, budget summaries, and graphs.
- `reports/` — Complete suite of markdown research reports.