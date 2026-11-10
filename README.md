# Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts

Research project investigating concept drift, temporal decay, cross-dataset transferability, and budget-constrained adaptation in machine learning phishing detection systems.

## Project Status: 100% COMPLETE (Phases 1, 2, 3, 4)

- **Phase 1 (Data Prep):** Acquired PhiUSIIL (70k) and PhreshPhish (40k). Extracted 15 lexical features.
- **Phase 2 (Baselines):** Evaluated LR, RF, XGBoost, and SGD on Random and Domain-Disjoint splits. Audited and excised leakage.
- **Phase 3 (Distribution Shifts & Drift):** Quantified 8–12 mo temporal decay (XGBoost F1 dropped 7.52%), demonstrated cross-dataset collapse (ROC-AUC < 0.46), and built PSI/KS drift monitors.
- **Phase 4 (Model Adaptation):** Evaluated Periodic, Online SGD, and Drift-Triggered adaptation across 1%, 5%, 10% label budgets. Discovered that 5% budget recovers >70% of lost performance, and Drift-Triggered retraining cuts updates by 60%.

## Quickstart & Replication

```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Phase 1 & 2 baseline pipeline
python -m src.pipeline

# Run Phase 3 temporal, cross-dataset, and drift detection pipeline
python -m src.experiments.run_phase3

# Run Phase 4 model adaptation pipeline
python -m src.experiments.run_phase4
```

## Key Reports

- [Master Synthesis Report](reports/phase3_phase4_complete_report.md)
- [Phase 4 Adaptation Evaluation](reports/adaptation_evaluation.md)
- [Phase 3 Shift & Drift Evaluation](reports/phase3_report.md)
- [Temporal Shift Report](reports/temporal_evaluation.md)
- [Cross-Dataset Evaluation Report](reports/cross_dataset_evaluation.md)
- [Statistical Drift Analysis Report](reports/drift_analysis.md)
- [Methodology Audit Report](reports/methodology_audit.md)
- [Research Paper Draft](reports/research_paper_draft.md)
