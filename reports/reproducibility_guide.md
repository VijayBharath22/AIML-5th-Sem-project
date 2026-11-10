# Reproducibility Guide: Drift-Aware Phishing URL Detection

**Research Project:** *Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning*  
**Repository:** `VijayBharath22/AIML-5th-Sem-project`  
**Execution Environment:** Linux / Python 3.14 (CPU ML)

---

## 1. Environment Requirements

- **Python Version:** Python 3.14.x (standard virtual environment)
- **Primary Dependencies:** `scikit-learn>=1.5.0`, `xgboost>=2.1.0`, `pandas>=2.2.0`, `numpy>=2.0.0`, `scipy>=1.14.0`, `matplotlib>=3.9.0`, `pyarrow>=17.0.0`, `tqdm>=4.66.0`, `pyyaml>=6.0.0`.
- **Hardware Footprint:** Classical CPU ML (No GPU required). Memory requirement: ~4 GB RAM. Total runtime for full Phase 1–5 pipeline: ~3–5 minutes.

---

## 2. Directory Structure and Pre-cached Datasets

```text
AIML Project 5th sem/
├── configs/
│   └── experiment_config.yaml         # Centralized hyperparameters & relative paths
├── data/
│   ├── raw/                           # Raw UCI PhiUSIIL zip & PhreshPhish parquet shards
│   └── processed/                     # Pre-cached standardized parquets
│       ├── phiusiil.parquet           # 70,000 samples, 50 features (no URLSimilarityIndex)
│       ├── phiusiil_url_features.parquet # 70,000 samples, 15 lexical URL features
│       └── phreshphish.parquet        # 40,000 samples, 15 lexical features + timestamps
├── models/                            # Serialized joblib baseline estimators
├── results/
│   ├── metrics/                       # Phase 1/2 baseline evaluation JSONs and CSVs
│   ├── temporal/                      # Phase 3 temporal evaluation metrics & monthly traces
│   ├── cross_dataset/                 # Phase 3 cross-dataset transfer evaluation
│   ├── drift/                         # Phase 3 statistical drift (PSI & KS) tables
│   ├── adaptation/                    # Phase 4 adaptation metrics across budgets & seeds
│   └── final/                         # Phase 5 consolidated summaries, registry & figures
└── reports/                           # Academic reports, audits, and research paper
```

---

## 3. End-to-End Execution Commands

### Step 1: Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Baseline Pipeline (Phases 1 & 2)
```bash
python -m src.pipeline
```
*Expected Outputs:*
- `results/metrics/comparison.csv`
- `results/metrics/summary.json`
- `results/metrics/verification.json` (`"passed": true`)

### Step 3: Run Shift & Drift Pipeline (Phase 3)
```bash
python -m src.experiments.run_phase3
```
*Expected Outputs:*
- `results/temporal/metrics/temporal_aggregate_metrics.csv`
- `results/cross_dataset/metrics/cross_dataset_metrics.csv`
- `results/drift/metrics/drift_monthly_summary.csv`
- `reports/phase3_report.md`

### Step 4: Run Adaptation Pipeline (Phase 4)
```bash
python -m src.experiments.run_phase4
```
*Expected Outputs:*
- `results/adaptation/metrics/adaptation_budget_summary.csv`
- `results/adaptation/metrics/adaptation_comparison.csv`
- `reports/adaptation_evaluation.md`

### Step 5: Run Phase 5 Synthesis & Validation
```bash
# Generate consolidated summaries, publication figures, and final paper
python -m src.experiments.run_phase5

# Run in lightweight validation-only mode
python -m src.experiments.run_phase5 --validate-only
```

---

## 4. Key Reproducibility Controls & Audit Constraints

1. **Fixed Random Seeds:** Experiments use seed `42` for baselines and `[42, 43, 44]` for adaptation budget sampling.
2. **Leakage Exclusion:** Feature `URLSimilarityIndex` is strictly excluded from all models.
3. **Prequential Integrity:** Streaming predictions are generated prior to revealing monthly budget labels.
4. **Authoritative Verdict:** *Scientifically Defensible With Documented Limitations.*
