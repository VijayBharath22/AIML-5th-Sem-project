# Drift-Aware Phishing URL Detection

Research title: Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning.

Phase 1 prepares PhiUSIIL and PhreshPhish. Phase 2 trains logistic regression, random forest, XGBoost, and SGDClassifier baselines. Temporal evaluation, cross-dataset transfer, drift detection, and adaptation are not implemented yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
```

`python -m src.pipeline` reuses the PhiUSIIL zip and the PhreshPhish column cache when those files already exist, then rebuilds the processed samples, refits the models, and rewrites the reports.

## Sample sizes from the latest run

- PhiUSIIL: 70000 rows after cleaning and stratified sampling.
- PhreshPhish: 40000 rows. Dates 2024-07-02 00:00:00+00:00 to 2025-12-16 00:00:00+00:00.
- Best PhiUSIIL random-split model: xgboost with F1 0.9999, ROC-AUC 1.0000, precision 0.9998, recall 1.0000, FPR 0.0001, accuracy 0.9999.
- Best PhreshPhish random-split model: xgboost with F1 0.9065, ROC-AUC 0.9717, precision 0.9102, recall 0.9028, FPR 0.0730, accuracy 0.9161.

## Layout

- `configs/experiment_config.yaml` — sources, sample sizes, seed, model parameters.
- `src/` — download, cleaning, URL features, training, evaluation, reports.
- `data/processed/` — modelling tables.
- `models/` — fitted pipelines.
- `results/` — metrics, predictions, graphs, confusion matrices.
- `reports/` — dataset analysis, baseline report, experiment log, paper draft.

## Reports

- [Methodology audit](reports/methodology_audit.md)
- [Temporal shift experiment report](reports/temporal_shift_report.md)
- [Dataset analysis](reports/dataset_analysis.md)
- [Baseline report](reports/baseline_report.md)
- [Experiment log](reports/experiment_log.md)
- [Paper draft](reports/research_paper_draft.md)
