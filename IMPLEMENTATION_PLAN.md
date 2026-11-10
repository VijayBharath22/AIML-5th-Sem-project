# Implementation Plan

## Project Roadmap

| Phase | Description | Status | Key Deliverable |
|---|---|---|---|
| Phase 1 | Dataset Acquisition and Feature Extraction | Completed | 15 deterministic lexical URL features, PhiUSIIL (70k), PhreshPhish (40k) |
| Phase 2 | Baseline Modeling & Evaluation | Completed | 16 baseline models, Random & Domain-Disjoint splits, parity verified |
| Audit | Methodology Audit & Leakage Remediation | Completed | Discovered & removed `URLSimilarityIndex` leakage, retrained baselines |
| Phase 3 | Temporal & Cross-Dataset Evaluation, Drift Detection | Completed | 8–12 mo temporal gap, bidirectional transfer, PSI/KS drift analysis |
| Phase 4 | Model Adaptation Under Constrained Budgets | Completed | Periodic, Online SGD, and Drift-Triggered adaptation across 1%, 5%, 10% budgets |
| Phase 5 | Research Synthesis, Visualization & Final Verification | Completed | Authoritative results index, 9 publication figures, research paper, validation harness |

## Core Codebase Structure
- `src/data/download.py` — Parallel downloading and caching.
- `src/data/preprocess.py` — Preprocessing, sampling, domain-disjoint splitting.
- `src/features/url_features.py` — 15 deterministic lexical URL feature extractors.
- `src/models/train_baseline.py` — Baseline model training and pipeline building.
- `src/models/evaluate.py` — Evaluation metrics and parity verification.
- `src/experiments/temporal_eval.py` — Phase 3 temporal evaluation module.
- `src/experiments/cross_dataset.py` — Phase 3 bidirectional transfer module.
- `src/experiments/drift_analysis.py` — Phase 3 statistical drift detection module.
- `src/experiments/run_phase3.py` — Master orchestrator for Phase 3.
- `src/experiments/adaptation.py` — Phase 4 model adaptation engine.
- `src/experiments/run_phase4.py` — Master orchestrator for Phase 4.
- `src/experiments/run_phase5.py` — Master orchestrator and verification suite for Phase 5.
- `src/pipeline.py` — End-to-end CLI pipeline.
