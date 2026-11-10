# Project Context

## Project Title
**Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning**

## Project Overview
This project investigates the resilience of machine learning phishing URL detection pipelines under temporal and cross-dataset distribution shifts, evaluates statistical drift detection techniques, and implements drift-aware adaptation strategies under limited label budgets.

## Current Status: ALL PHASES COMPLETED (Phases 1, 2, 3, 4, 5)

- **Phase 1 (Data Prep):** Acquired and processed PhiUSIIL (70,000 samples) and PhreshPhish (40,000 samples). Extracted 15 deterministic lexical URL features without HTML leakage.
- **Phase 2 (Baselines):** Evaluated Logistic Regression, Random Forest, XGBoost, and SGDClassifier on Random and Domain-Disjoint splits.
- **Methodology Audit:** Excluded target leakage variable `URLSimilarityIndex` in PhiUSIIL. Retrained affected models and verified 100% parity (`results/metrics/verification.json`).
- **Phase 3 (Temporal & Cross-Dataset Shifts, Drift Detection):**
  - Evaluated 8–12 month temporal shift: XGBoost F1 dropped by $7.52%$ ($0.9065 \rightarrow 0.8313$), Random Forest recall collapsed from $0.8976$ to $0.7351$.
  - Evaluated cross-dataset transfer: Bidirectional transfer collapsed to ~40% accuracy with 100% false positive rates due to root-homepage collection bias in PhiUSIIL.
  - Implemented unsupervised PSI and KS drift detection with Bonferroni correction ($\alpha = 0.05/15 = 0.00333$), proving feature drift spikes align with classifier degradation.
- **Phase 4 (Model Adaptation Under Limited Budgets):**
  - Evaluated Static Control (0%), Periodic Retraining, Online Incremental Learning (SGD `partial_fit`), and Drift-Triggered Retraining across 1%, 5%, 10%, 100% label budgets over multiple seeds.
  - Proved that Drift-Triggered retraining at 5% budget matches periodic retraining accuracy (F1 = $0.8340$ vs. $0.8372$) while cutting retraining compute runtime by **80.26%** and update cycles by **80.0%** (1 update vs. 5).
- **Phase 5 (Synthesis, Visualizations & Academic Paper):**
  - Consolidated authoritative results index (`results/final/final_results_summary.csv`, `experiment_registry.csv` indexing 128 experiments).
  - Generated 9 publication-grade figures at 300 DPI (`results/final/figures/`).
  - Completed academic research paper (`reports/final_research_paper.md`), master report (`reports/phase5_complete_report.md`), and reproduction guide (`reports/reproducibility_guide.md`).
  - Executed automated verification suite confirming 20/20 checks passed.

## Final Scientific Verdict
**Scientifically Defensible With Documented Limitations**

## Key Deliverables & Reports
- Final Research Paper: `reports/final_research_paper.md`
- Master Synthesis: `reports/phase5_complete_report.md`
- Final Methodology Audit: `reports/final_methodology_audit.md`
- Reproducibility Guide: `reports/reproducibility_guide.md`
- Authoritative Results: `results/final/final_results_summary.csv`
- Publication Figures: `results/final/figures/`
