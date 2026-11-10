# Implementation Plan

## Phase 1 — Data Preparation (Completed)

- Downloaded PhiUSIIL from UCI ML repository and extracted 70,000 stratified samples.
- Retrieved 77 parquet shards of PhreshPhish from Hugging Face via column projection (dropping HTML) and sampled 40,000 records.
- Saved processed datasets to `data/processed/` and metadata to `data/metadata/`.

## Phase 2 — Baseline Models (Completed)

- Trained 4 algorithms (Logistic Regression, Random Forest, XGBoost, SGDClassifier) on both datasets.
- Evaluated on both Stratified Random (80/20) and Domain-Disjoint splits (0% test domain overlap).
- Saved 16 fitted `.joblib` models, prediction `.parquet` tables, confusion matrices, ROC curves, and metric summaries.
- Verified 100% agreement between recorded metrics and recomputed prediction files (`results/metrics/verification.json`).

## Baseline Methodology Audit (Completed)

- Investigated PhiUSIIL near-perfect scores: identified direct target leakage in `URLSimilarityIndex`, severe HTML disparity, and root-homepage sampling bias.
- Excluded `URLSimilarityIndex` from the primary feature pipeline; retrained and verified all 8 PhiUSIIL models.
- Audited PhreshPhish: identified zero-benign period (2024-11 to 2025-04) and established balanced evaluation windows for temporal testing.
- Published comprehensive audit report: `reports/methodology_audit.md`.

## Phase 3 — Temporal Distribution Shift (Next Phase)

- On PhreshPhish only, construct a chronological train/test split:
  - *Train Window:* 2024-08 to 2024-10 (historical baseline, ~14.5k URLs).
  - *Test Window:* 2025-06 to 2025-11 (temporal shift, ~17.3k URLs).
- Evaluate frozen Phase 2 baseline models without retraining.
- Measure performance degradation (F1 drop, FPR increase, ROC-AUC decay) under pure temporal shift.
- Compare temporal performance against random-split and domain-disjoint baselines.

## Phase 4 — Drift Detection (Planned)

- Implement unsupervised feature drift detectors (PSI, Kolmogorov-Smirnov).
- Track drift scores over monthly streaming batches without using labels.

## Phase 5 — Model Adaptation (Planned)

- Evaluate adaptation strategies under constrained label budgets:
  - Static baseline (no update).
  - Periodic sliding-window retraining.
  - Online incremental updates (SGD `partial_fit`).
  - Drift-triggered retraining (triggered by Phase 4 detector).

## Phase 6 — Cross-Dataset Transfer (Planned)

- Apply `src/features/url_features.py` to PhiUSIIL URLs.
- Evaluate transfer generalizability in both directions (PhiUSIIL -> PhreshPhish and PhreshPhish -> PhiUSIIL).

## Module Map

- `src/data/download.py` — Parallel downloading and caching.
- `src/data/preprocess.py` — Cleaning, sampling, domain-disjoint split logic.
- `src/features/url_features.py` — Deterministic lexical URL feature extraction.
- `src/models/train_baseline.py` — Baseline model training.
- `src/models/evaluate.py` — Evaluation metrics, ROC, confusion matrices, artifact verification.
- `src/reporting/build_reports.py` — Report generation and context maintenance.
- `src/pipeline.py` — End-to-end pipeline CLI.
