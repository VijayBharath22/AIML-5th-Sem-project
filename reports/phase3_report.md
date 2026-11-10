# Phase 3 Synthesis Report: Distribution Shift Evaluation and Drift Detection

## 1. Overview of Phase 3 Accomplishments

Phase 3 rigorously tested the three core research hypotheses regarding distribution shifts in machine learning for phishing detection:
1. **Temporal Shift:** Do static models decay over time? **Yes.** XGBoost F1 dropped by $7.52%$ (from $0.9065$ to $0.8313$), and Random Forest recall dropped from $0.8976$ to $0.7351$.
2. **Cross-Dataset Shift:** Do models generalize across independent datasets? **No.** When models trained on one dataset are evaluated on another using identical 15 lexical features, accuracy collapsed to $38.7%–41.5%$ (ROC-AUC $< 0.46$).
3. **Statistical Drift Detection:** Can drift be detected unsupervised? **Yes.** Population Stability Index (PSI) and two-sample KS tests reliably flagged distribution divergence, showing strong inverse correlation with classifier F1.

## 2. Comprehensive Performance Summary

| Evaluation Regime | Dataset Pair | Model | Accuracy | F1-Score | ROC-AUC | False Positive Rate | Key Phenomenon |
|---|---|---|---:|---:|---:|---:|---|
| In-Distribution (Phase 2) | PhreshPhish (Random) | XGBoost | 0.9161 | 0.9065 | 0.9717 | 0.0730 | Baseline benchmark |
| Domain Shift (Phase 2) | PhreshPhish (Disjoint) | XGBoost | 0.8887 | 0.8748 | 0.9503 | 0.0845 | Domain memorization lost |
| Temporal Shift (Phase 3) | PhreshPhish (8-12 mo) | XGBoost | 0.8792 | 0.8313 | 0.9264 | 0.0698 | Attacker evolution decay |
| Cross-Dataset A (Phase 3) | PhreshPhish -> PhiUSIIL | XGBoost | 0.3872 | 0.5583 | 0.4521 | 0.9850 | Benchmark artifact mismatch |
| Cross-Dataset B (Phase 3) | PhiUSIIL -> PhreshPhish | XGBoost | 0.4148 | 0.5864 | 0.4263 | 0.8920 | Benchmark artifact mismatch |

## 3. Visual Summaries

1. **Temporal Degradation:** `results/temporal/graphs/temporal_vs_baseline_comparison.png`
2. **Monthly Performance Curves:** `results/temporal/graphs/temporal_monthly_trajectory.png`
3. **Cross-Dataset Generalization Collapse:** `results/cross_dataset/graphs/cross_dataset_transfer_comparison.png`
4. **Feature Drift Heatmap:** `results/drift/graphs/drift_feature_heatmap.png`
5. **Drift vs. F1 Inverse Correlation:** `results/drift/graphs/drift_vs_f1_correlation.png`

## 4. Phase 4 Implementation Roadmap (Model Adaptation)

Having demonstrated that static models fail under both temporal and cross-dataset shifts, Phase 4 will develop adaptive retraining strategies:
1. **Periodic Retraining:** Sliding-window retraining at fixed monthly intervals.
2. **Online Learning:** Incremental weight updates using `SGDClassifier.partial_fit()`.
3. **Drift-Triggered Adaptation:** Retraining triggered autonomously when PSI exceeds $0.10$ or $0.25$.
4. **Budgeted Label Curves:** Evaluating how many newly labelled samples ($N=100, 250, 500, 1000$) are necessary to restore model F1.