# Research Methodology

## Completed

- **PhiUSIIL Preparation:** UCI ZIP archive preserved; URL duplicates dropped; stratified sample of 70,000 rows. Excluded target leakage variable `URLSimilarityIndex`, identifiers (`URL`, `Domain`, `FILENAME`), free-text `Title`, and derived labels.
- **PhreshPhish Preparation:** Column-selective projection of `sha256`, `url`, `label`, and `date` across all 77 parquet shards (666k records). Raw HTML dropped to avoid disk bloat. Invalid and duplicate URLs dropped. 40,000-sample corpus extracted with 15 deterministic lexical URL features.
- **Splitting Strategy:**
  - *Stratified Random (80/20):* Baseline reference split.
  - *Domain-Disjoint Group Split:* Grouped by `registrable_domain` using a search over random seeds to minimize class and fraction distortion. Domain overlap is verified to be strictly $0.0\%$.
- **Preprocessing Guardrails:** Imputation, standard scaling, and One-Hot Encoding (for categorical TLDs) are enclosed within `sklearn.pipeline.Pipeline` and fitted exclusively on training splits.
- **Baseline Models:** Logistic Regression (`lbfgs`, balanced), Random Forest (100 estimators, max depth 18), XGBoost (`hist`, scale_pos_weight), and SGDClassifier (`log_loss`).
- **Evaluation & Verification:** Metrics include Accuracy, Precision, Recall, F1, ROC-AUC, FPR, confusion matrices, and latency. All 16 saved models verified against raw parquet predictions.
- **Methodology Audit:** Documented in `reports/methodology_audit.md`. Identified PhiUSIIL webpage/URL sampling bias and mapped PhreshPhish chronological windows.

## Pending (Phases 3 to 6)

- **Phase 3 (Temporal Distribution Shift):** Train on PhreshPhish Historical Window (2024-08 to 2024-10) and evaluate without retraining across Recent Window (2025-06 to 2025-11).
- **Phase 4 (Drift Detection):** Unsupervised statistical drift tracking on streaming lexical features (PSI, Kolmogorov-Smirnov).
- **Phase 5 (Drift Adaptation):** Evaluate static vs. periodic refits, online updates (SGD `partial_fit`), and drift-triggered retrains under varying label budgets.
- **Phase 6 (Cross-Dataset Evaluation):** Extract identical 15 lexical URL features on PhiUSIIL URLs to test bidirectional transfer against PhreshPhish without webpage features.

## Leakage Controls

- Identifiers, dates, raw labels, registrable domains, and target-derived similarity indexes (`URLSimilarityIndex`) are excluded from model features.
- Domain overlap is measured for random splits (28.4% in PhreshPhish) and enforced to be 0% in domain-disjoint splits.
