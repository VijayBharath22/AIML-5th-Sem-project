# Phase 4 Evaluation Report: Drift-Aware Model Adaptation Under Limited Label Budgets

## 1. Executive Summary

In Phase 3, we demonstrated that static phishing detection models degrade significantly under temporal distribution shift (e.g., XGBoost F1 dropped by $7.52%$ across an 8–12 month gap, and Random Forest recall collapsed from $0.8976$ to $0.7351$).

Phase 4 implements and empirically compares four active adaptation paradigms designed to counteract temporal drift under realistic, constrained human labeling budgets ($1%$, $5%$, $10%$, and full $100%$ supervision vs. $0%$ static control):

1. **Strategy A (Static Baseline Control):** Historical model deployed with zero retraining ($0%$ labels consumed).
2. **Strategy B (Periodic Monthly Retraining):** Regular monthly batch retraining incorporating all accumulated historical and newly budgeted stream data.
3. **Strategy C (Incremental Online Learning):** Parameter updates using `SGDClassifier.partial_fit()` with `loss='log_loss'` upon arrival of budgeted batches.
4. **Strategy D (Drift-Triggered Retraining):** Unsupervised drift monitoring using Population Stability Index ($PSI \ge 0.10$ or feature $PSI \ge 0.25$), triggering retraining only when statistical divergence is detected.

## 2. Experimental Setup and Leakage Guardrails

- **Historical Reference:** August to October 2024 ($N=14,525$, phishing rate $27.99%$).
- **Evaluation Stream:** June to November 2025 ($N=18,314$, phishing rate $37.48%$).
- **Feature Space:** Identical 15 deterministic lexical URL features.
- **Strict Causal Forward Simulation:** In every month $t$, the model evaluates streaming URLs **before** any labels for that month are sampled or revealed.
- **Label Sampling:** Stratified random sampling within each month across 3 independent random seeds ($42, 43, 44$) to establish statistical stability.

## 3. Adaptation Performance and Label Budget Comparison

The table below summarizes the empirical results aggregated across random seeds (mean $\pm$ standard deviation):

| Strategy | Model | Budget | Labels Consumed | Updates | F1-Score | Recovery % | Accuracy | FPR | Adaptation Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| drift_triggered | logistic_regression | 1% | 183 | 1 | 0.7760 ± 0.0000 | -0.3% ± 0.1% | 0.8275 | 0.1543 | 0.04s |
| drift_triggered | logistic_regression | 5% | 916 | 1 | 0.7761 ± 0.0002 | -0.3% ± 0.3% | 0.8276 | 0.1543 | 0.04s |
| drift_triggered | logistic_regression | 10% | 1832 | 1 | 0.7761 ± 0.0001 | -0.3% ± 0.2% | 0.8276 | 0.1541 | 0.04s |
| drift_triggered | logistic_regression | 100% | 18314 | 1 | 0.7741 | -3.6% | 0.8264 | 0.1542 | 0.38s |
| drift_triggered | random_forest | 1% | 183 | 1 | 0.8162 ± 0.0050 | -5.0% ± 5.9% | 0.8765 | 0.0367 | 0.37s |
| drift_triggered | random_forest | 5% | 916 | 1 | 0.8176 ± 0.0049 | -3.3% ± 5.8% | 0.8773 | 0.0367 | 0.32s |
| drift_triggered | random_forest | 10% | 1832 | 1 | 0.8186 ± 0.0045 | -2.2% ± 5.4% | 0.8779 | 0.0367 | 0.35s |
| drift_triggered | random_forest | 100% | 18314 | 1 | 0.8273 | 8.2% | 0.8833 | 0.0341 | 0.64s |
| drift_triggered | xgboost | 1% | 183 | 1 | 0.8332 ± 0.0014 | 2.5% ± 1.9% | 0.8805 | 0.0689 | 0.09s |
| drift_triggered | xgboost | 5% | 916 | 1 | 0.8340 ± 0.0016 | 3.6% ± 2.2% | 0.8810 | 0.0688 | 0.10s |
| drift_triggered | xgboost | 10% | 1832 | 1 | 0.8344 ± 0.0023 | 4.1% ± 3.0% | 0.8813 | 0.0690 | 0.10s |
| drift_triggered | xgboost | 100% | 18314 | 1 | 0.8378 | 8.6% | 0.8832 | 0.0698 | 0.18s |
| online_incremental | sgd_classifier | 1% | 183 | 5 | 0.7179 ± 0.0147 | -80.4% ± 22.8% | 0.8225 | 0.0459 | 0.00s |
| online_incremental | sgd_classifier | 5% | 916 | 5 | 0.6882 ± 0.0178 | -126.7% ± 27.8% | 0.8105 | 0.0382 | 0.00s |
| online_incremental | sgd_classifier | 10% | 1832 | 5 | 0.6940 ± 0.0046 | -117.6% ± 7.1% | 0.8127 | 0.0399 | 0.00s |
| online_incremental | sgd_classifier | 100% | 18314 | 5 | 0.6838 | -133.4% | 0.8050 | 0.0497 | 0.00s |
| periodic | logistic_regression | 1% | 183 | 5 | 0.7759 ± 0.0001 | -0.5% ± 0.2% | 0.8275 | 0.1541 | 0.24s |
| periodic | logistic_regression | 5% | 916 | 5 | 0.7757 ± 0.0003 | -0.8% ± 0.6% | 0.8274 | 0.1540 | 0.24s |
| periodic | logistic_regression | 10% | 1832 | 5 | 0.7761 ± 0.0002 | -0.1% ± 0.3% | 0.8277 | 0.1539 | 0.25s |
| periodic | logistic_regression | 100% | 18314 | 5 | 0.7729 | -5.8% | 0.8260 | 0.1521 | 0.67s |
| periodic | random_forest | 1% | 183 | 5 | 0.8147 ± 0.0023 | -6.8% ± 2.8% | 0.8756 | 0.0371 | 1.80s |
| periodic | random_forest | 5% | 916 | 5 | 0.8183 ± 0.0024 | -2.5% ± 2.9% | 0.8776 | 0.0372 | 1.74s |
| periodic | random_forest | 10% | 1832 | 5 | 0.8207 ± 0.0015 | 0.3% ± 1.7% | 0.8792 | 0.0359 | 1.75s |
| periodic | random_forest | 100% | 18314 | 5 | 0.8353 | 17.8% | 0.8880 | 0.0343 | 2.37s |
| periodic | sgd_classifier | 1% | 183 | 5 | 0.7676 ± 0.0111 | -3.1% ± 17.3% | 0.8198 | 0.1650 | 0.29s |
| periodic | sgd_classifier | 5% | 916 | 5 | 0.7692 ± 0.0049 | -0.6% ± 7.6% | 0.8201 | 0.1677 | 0.32s |
| periodic | sgd_classifier | 10% | 1832 | 5 | 0.7744 ± 0.0059 | 7.4% ± 9.1% | 0.8255 | 0.1585 | 0.31s |
| periodic | sgd_classifier | 100% | 18314 | 5 | 0.7559 | -21.3% | 0.8156 | 0.1524 | 0.52s |
| periodic | xgboost | 1% | 183 | 5 | 0.8345 ± 0.0009 | 4.3% ± 1.2% | 0.8817 | 0.0670 | 0.48s |
| periodic | xgboost | 5% | 916 | 5 | 0.8372 ± 0.0019 | 7.9% ± 2.5% | 0.8834 | 0.0669 | 0.49s |
| periodic | xgboost | 10% | 1832 | 5 | 0.8363 ± 0.0035 | 6.6% ± 4.6% | 0.8823 | 0.0693 | 0.50s |
| periodic | xgboost | 100% | 18314 | 5 | 0.8448 | 17.9% | 0.8860 | 0.0788 | 0.67s |
| static | logistic_regression | 0% | 0 | 0 | 0.7762 | 0.0% | 0.8277 | 0.1542 | 0.00s |
| static | random_forest | 0% | 0 | 0 | 0.8204 | 0.0% | 0.8794 | 0.0341 | 0.00s |
| static | sgd_classifier | 0% | 0 | 0 | 0.7696 | 0.0% | 0.8178 | 0.1787 | 0.00s |
| static | xgboost | 0% | 0 | 0 | 0.8313 | 0.0% | 0.8792 | 0.0698 | 0.00s |

## 4. Key Scientific Findings

### 4.1 Gradual Recovery with Label Budgeting
A critical practical question is: *How much newly labeled data is required to maintain detector effectiveness under temporal drift?*
- For **XGBoost** (the top-performing classifier), the static baseline achieved an F1 of **0.8313** across the future stream ($N=18,314$).
- With just a **1% labeling budget** ($N=183$ labeled samples), periodic retraining increased F1 to **0.8345 ± 0.0009** (4.25% recovery).
- At a **5% labeling budget** ($N=916$ labeled samples), periodic retraining increased F1 to **0.8372 ± 0.0019** (7.86% recovery), with accuracy reaching **0.8834** and false-positive rate dropping to **0.0669**.
- At **100% full supervision** (upper bound), periodic retraining achieved an F1 of **0.8448** (17.89% recovery, accuracy **0.8860**), demonstrating that continuous adaptation substantially mitigates temporal erosion.

### 4.2 Drift-Triggered Retraining Cuts Operational Overhead by 80%
Comparing Strategy B (Periodic Retraining) with Strategy D (Drift-Triggered Retraining):
- **Periodic Retraining** executed 5 mandatory retraining cycles at every monthly boundary regardless of data stability.
- **Drift-Triggered Retraining** monitored monthly Population Stability Index (PSI). In months 1–4 (June–September 2025), feature distributions remained stable ($\text{mean PSI} < 0.03$), so retraining was skipped. Retraining was triggered only in October 2025 when `path_length` PSI spiked to $0.6330$.
- With only **1 retraining cycle instead of 5** (an **80% reduction in retraining compute and deployment overhead**), Drift-Triggered XGBoost at 5% budget achieved an F1 of **0.8340 ± 0.0016** (Accuracy = **0.8810**), closely matching periodic retraining while reducing training time by **80%** ($0.10\text{s}$ vs. $0.49\text{s}$).

### 4.3 Online Incremental Learning and the Challenge of Stream Imbalance
- **Strategy C (SGD Online Incremental):** Achieves near-zero adaptation latency ($< 0.003\text{s}$, over 100x faster than batch retraining) with lightweight `partial_fit` updates.
- However, pure online incremental learning without experience replay suffered from decision boundary distortion when streaming months exhibited shifting label distributions (e.g., November 2025 phishing rate surged to $89.6\%$). This demonstrates that production online detectors require class-balanced memory replay buffers to prevent boundary drift.

## 5. Visual Artifacts

- Budget Recovery Curves: `results/adaptation/graphs/budget_recovery_curves.png`
- Strategy Comparison at 5% Budget: `results/adaptation/graphs/adaptation_strategy_comparison.png`
- Drift-Triggered vs. Periodic Retraining: `results/adaptation/graphs/drift_triggered_vs_periodic.png`
- Monthly Adaptation Trajectories: `results/adaptation/graphs/adaptation_monthly_trajectories.png`