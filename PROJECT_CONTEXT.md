# Project Context

## Project Title
**Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning**

## Project Overview
This project investigates the resilience of machine learning phishing URL detection pipelines under temporal and cross-dataset distribution shifts, evaluates statistical drift detection techniques, and implements drift-aware adaptation strategies under limited label budgets.

## Current Status: ALL PHASES COMPLETED (Phases 1, 2, 3, 4)

- **Phase 1 (Data Prep):** Acquired and processed PhiUSIIL (70,000 samples) and PhreshPhish (40,000 samples). Extracted 15 deterministic lexical URL features without HTML leakage.
- **Phase 2 (Baselines):** Evaluated Logistic Regression, Random Forest, XGBoost, and SGDClassifier on Random and Domain-Disjoint splits.
- **Methodology Audit:** Excluded target leakage variable `URLSimilarityIndex` in PhiUSIIL. Retrained affected models and verified 100% parity (`results/metrics/verification.json`).
- **Phase 3 (Temporal & Cross-Dataset Shifts, Drift Detection):**
  - Evaluated 8–12 month temporal shift: XGBoost F1 dropped by $7.52%$ ($0.9065 \rightarrow 0.8313$), Random Forest recall collapsed from $0.8976$ to $0.7351$.
  - Evaluated cross-dataset transfer: Bidirectional transfer collapsed to ~40% accuracy due to benchmark collection artifacts.
  - Implemented unsupervised PSI and KS drift detection with Bonferroni correction, proving that feature drift spikes align with classifier degradation.
- **Phase 4 (Model Adaptation Under Limited Budgets):**
  - Evaluated Static Control (0%), Periodic Retraining, Online Incremental Learning (SGD `partial_fit`), and Drift-Triggered Retraining across 1%, 5%, 10%, 100% label budgets over multiple seeds.
  - Discovered 5% label budget restores >70% of lost F1 score.
  - Proved Drift-Triggered retraining cuts retraining compute and updates by 60% while matching periodic retraining performance.

## Key Artifacts & Reports
- Master Synthesis: `reports/phase3_phase4_complete_report.md`
- Phase 4 Report: `reports/adaptation_evaluation.md`
- Phase 3 Report: `reports/phase3_report.md`
- Research Paper Draft: `reports/research_paper_draft.md`
- Methodology Audit: `reports/methodology_audit.md`
