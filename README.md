# Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scientific Verdict](https://img.shields.io/badge/Scientific%20Verdict-Defensible%20with%20Limitations-success.svg)](reports/final_methodology_audit.md)

An empirical research framework investigating temporal degradation, cross-dataset distribution shifts, statistical drift detection, and budget-constrained adaptation in machine learning phishing URL detection systems.

---

## 1. Motivation & Problem Statement

Machine learning systems for phishing URL detection routinely achieve reported detection accuracies exceeding 99% on static benchmark partitions. However, adversaries continuously evolve URL token structures, hosting patterns, and path depths to evade detection. When static models are deployed in production, they suffer from:
1. **Domain Memorization:** Models learn to recognize popular benign or malicious domains rather than robust structural features.
2. **Temporal Degradation:** Performance degrades over time as attacker tactics evolve.
3. **Cross-Benchmark Incompatibility:** Models trained on curated benchmarks fail catastrophically on live web traffic.

---

## 2. Research Objectives & Methodology

The experimental framework spans five completed phases:
- **Phase 1 (Data Acquisition & Preprocessing):** Standardized **PhiUSIIL** ($N=70,000$, UCI) and **PhreshPhish** ($N=40,000$, Hugging Face with genuine timestamps). Extracted a deterministic, lightweight 15-dimensional lexical URL feature representation.
- **Phase 2 (Baselines & Leakage Audit):** Evaluated Logistic Regression, Random Forest, XGBoost, and SGDClassifier on Stratified Random and Domain-Disjoint splits. Audited and excised severe target leakage in PhiUSIIL (`URLSimilarityIndex`).
- **Phase 3 (Distribution Shifts & Drift Detection):** Evaluated an 8–12 month temporal shift and bidirectional cross-dataset transfer. Implemented Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift detectors with Bonferroni correction ($\alpha = 0.05/15 = 0.00333$).
- **Phase 4 (Drift-Aware Model Adaptation):** Evaluated Static Control, Periodic Monthly Retraining, Online Incremental Learning (SGD `partial_fit`), and Drift-Triggered Retraining under constrained human labeling budgets ($1\%, 5\%, 10\%, 100\%$) across multiple random seeds ($42, 43, 44$).
- **Phase 5 (Synthesis, Visualizations & Validation):** Created an authoritative experiment registry ($128$ indexed runs), generated 9 publication-grade 300 DPI figures, compiled the final research paper, and established a reproducible validation suite.

---

## 3. Major Research Findings

1. **Domain Memorization Penalty:** Standard random splits overestimate model F1 by **2–3%** because models memorize recurring domains. Enforcing domain-disjoint splitting drops XGBoost F1 from $0.9065$ to $0.8748$.
2. **Temporal Degradation:** Over an 8-to-12 month gap, static XGBoost F1 dropped by **-7.52%** ($0.9065 \rightarrow 0.8313$), Random Forest recall collapsed from $0.8976$ to $0.7351$, and linear models suffered significant false positive rate spikes ($15.4\%$ to $17.8\%$).
3. **Cross-Dataset Failure:** Bidirectional transfer between PhiUSIIL and PhreshPhish collapsed with a **100% false positive rate** ($\text{FPR} = 1.0000$). Inspection revealed a severe benchmark curation bias: **100% of legitimate URLs in PhiUSIIL are root homepages with 0.0 path length**, whereas in-the-wild web crawls feature deep paths (mean 35.0).
4. **Drift-Triggered Retraining Supremacy:** Drift-Triggered XGBoost at a 5% budget matched periodic retraining accuracy (F1 = $0.8340$ vs. $0.8372$) while requiring only **1 update cycle instead of 5**—cutting retraining compute runtime by **80.26%** ($0.096\text{s}$ vs. $0.487\text{s}$) and update disruption by **80.0%**.
5. **Online Incremental Learning Trade-offs:** Incremental SGD executes in under $0.003\text{s}$ (>100x faster than batch), but suffered boundary distortion when stream prevalence shifted sharply (89.6% phishing in Nov 2025), showing the necessity of memory replay buffers.

---

## 4. Repository Structure

```text
├── configs/
│   └── experiment_config.yaml         # Central hyperparameters & relative paths
├── data/
│   ├── metadata/                      # Provenance logs & schema definitions
│   └── processed/                     # Standardized Parquet datasets
├── models/                            # Serialized baseline joblib pipelines
├── results/
│   ├── metrics/                       # Baseline JSON/CSV evaluation outputs
│   ├── temporal/                      # Temporal aggregate & monthly evaluations
│   ├── cross_dataset/                 # Cross-dataset transfer outputs & confusion matrices
│   ├── drift/                         # PSI matrices, KS p-values & drift summaries
│   ├── adaptation/                    # Adaptation metrics across budgets and seeds
│   └── final/                         # Phase 5 authoritative registry, manifest & figures
│       ├── figures/                   # 9 publication-grade figures (300 DPI)
│       ├── final_results_summary.csv  # Authoritative master metrics
│       ├── experiment_registry.csv    # Index of all 128 experimental runs
│       ├── reproducibility_manifest.json # System & environment manifest
│       └── validation_results.json    # Verification test suite results
├── reports/
│   ├── final_research_paper.md        # Comprehensive academic research manuscript
│   ├── final_methodology_audit.md     # Independent methodology audit report
│   ├── phase5_complete_report.md      # Self-contained Phase 5 master synthesis
│   ├── reproducibility_guide.md       # Step-by-step reproduction instructions
│   └── experiment_log.md              # Chronological execution log
└── src/
    ├── data/                          # Download and preprocessing modules
    ├── features/                      # Deterministic 15-feature lexical URL extractor
    ├── models/                        # Estimators, pipelines, evaluation routines
    └── experiments/                   # Phase 3, 4, and 5 orchestration engines
```

---

## 5. Quickstart & Reproduction

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Experimental Pipelines
```bash
# 1. Run baseline pipeline (Phases 1 & 2)
python -m src.pipeline

# 2. Run temporal, cross-dataset & drift detection pipeline (Phase 3)
python -m src.experiments.run_phase3

# 3. Run model adaptation pipeline (Phase 4)
python -m src.experiments.run_phase4

# 4. Run final research synthesis, figure generation & validation (Phase 5)
python -m src.experiments.run_phase5

# 5. Run lightweight validation-only verification check
python -m src.experiments.run_phase5 --validate-only
```

---

## 6. Key Deliverables & Reports

- **Academic Paper:** [Final Research Paper](reports/final_research_paper.md)
- **Methodology Audit:** [Final Methodology Audit Report](reports/final_methodology_audit.md)
- **Consolidated Synthesis:** [Phase 5 Master Report](reports/phase5_complete_report.md)
- **Reproduction Guide:** [Reproducibility Guide](reports/reproducibility_guide.md)
- **Figures:** Available under `results/final/figures/fig1_research_pipeline.png` through `fig9_adaptation_efficiency.png`
- **Master Metrics Table:** [Authoritative Results Summary](results/final/final_results_summary.csv)
