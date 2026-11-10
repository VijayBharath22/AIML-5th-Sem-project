# Research Methodology

## Experimental Architecture

### 1. In-Distribution and Domain-Disjoint Baselines (Phase 1 & 2)
- Models: Logistic Regression, Random Forest, XGBoost, SGDClassifier.
- Evaluated on Stratified Random Split (in-distribution) and Domain-Disjoint Split (evaluates memorization vs generalizability).
- Leakage exclusion: `URLSimilarityIndex` excised following empirical audit.

### 2. Temporal Distribution Shift (Phase 3)
- Historical training window: August–October 2024 ($N=14,525$).
- Future evaluation stream: June–November 2025 ($N=18,314$) across 6 individual monthly slices.
- Causal guarantee: All scalers and models fit strictly on training period.

### 3. Cross-Dataset Generalization (Phase 3)
- Direction A: PhreshPhish -> PhiUSIIL.
- Direction B: PhiUSIIL -> PhreshPhish.
- Enforced identical 15-dimensional lexical feature representation extracted directly from raw URLs.

### 4. Statistical Drift Detection (Phase 3)
- Unsupervised monitoring: Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests.
- Bonferroni multiple-comparison correction ($\alpha = 0.05 / 15 = 0.00333$).
- Quantified correlation between feature drift spikes and classifier F1 degradation.

### 5. Drift-Aware Model Adaptation (Phase 4)
- **Strategy A (Static Control):** Model frozen after historical training ($0%$ budget).
- **Strategy B (Periodic Retraining):** Monthly batch retraining on historical + accumulated budgeted labels.
- **Strategy C (Online Incremental Learning):** Parameter updates using `SGDClassifier.partial_fit()`.
- **Strategy D (Drift-Triggered Retraining):** Retrains only when monthly PSI exceeds threshold ($\text{mean\_psi} \ge 0.10$ or $\text{max\_psi} \ge 0.25$).
- **Budgets Evaluated:** $1\%$, $5\%$, $10\%$, and $100\%$ full supervision across 3 random seeds ($42, 43, 44$).
- **Recovery Formula:**
  $$\text{Recovery} = \frac{\text{F1}_{\text{adapted}} - \text{F1}_{\text{static}}}{\max(10^{-5}, \text{F1}_{\text{in-dist}} - \text{F1}_{\text{static}})} \times 100\%$$
