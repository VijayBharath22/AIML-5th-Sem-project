# Methodology Audit Report: Pre-Phase 3 Baseline Verification

**Project Title:** Drift-Aware Phishing URL Detection Under Temporal and Cross-Dataset Distribution Shifts Using Machine Learning  
**Date:** 2026-11-11  
**Scope:** In-depth audit of Phase 1 (Dataset Preparation) and Phase 2 (Baseline Training & Evaluation) across both PhiUSIIL and PhreshPhish datasets before commencing Phase 3.

---

## 1. Executive Audit Summary

A rigorous audit was conducted on the baseline pipelines, dataset structures, feature distributions, splitting strategies, and saved artifacts. The key findings are:

1. **PhiUSIIL Near-Perfect Performance (99.99%–100%) Explained**:
   - **Target Leakage (`URLSimilarityIndex`)**: In the original dataset construction (Prasad & Chandra, 2023), `URLSimilarityIndex` was computed by comparing candidate URLs against the targeted legitimate brand. For 100% of legitimate URLs in the dataset (40,105 in sample, 134,850 in raw), `URLSimilarityIndex` is identically `100.0` ($\sigma = 0.0$). For phishing URLs, the mean is $49.62$. This feature is unavailable at real-time inference on unseen URLs without knowing ground truth.
   - **Dataset Construction Artifacts (Webpage Features)**: Webpage features exhibit massive class disparity caused by crawling live, mature Alexa/Cisco homepages for legitimate instances versus parked, dead, or stub login pages for phishing instances:
     - `LineOfCode`: Median of $1,098$ for legitimate homepages vs. $11$ for phishing stubs.
     - `HasSocialNet`: $79.3\%$ of legitimate instances have social media links vs. $0.48\%$ of phishing instances.
     - `HasCopyrightInfo`: $80.7\%$ of legitimate instances have copyright info vs. $5.58\%$ of phishing instances.
   - **URL Structural Bias in Collection**: Across all 134,850 legitimate URLs in PhiUSIIL, exactly $0.0\%$ contain a path segment ($>1$ char) or query parameters (`?`, `=`, `&`). Every legitimate URL is a homepage root (`https://www.domain.com`), whereas phishing URLs contain deep subdirectories, query parameters, or raw IPs.
2. **PhreshPhish Data Quality & Temporal Split Preparedness**:
   - The 15 lexical URL features have realistic, moderate correlations (ranging from $-0.42$ for `n_subdirectories` to $+0.39$ for `hostname_length`).
   - Domain overlap between train and test in the random split is $28.38\%$, dropping to strictly **$0.0\%$** in the domain-disjoint split.
   - **Critical Temporal Collection Finding**: In the original PhreshPhish corpus (and our 40,000-sample extraction), there are severe collection gaps:
     - **2024-11 through 2025-04 contains zero benign samples** (only phishing URLs).
     - **2025-05 contains zero total records**.
     - **2025-12 contains zero phishing samples** (only benign).
     - Consequently, Phase 3 temporal splits must use balanced evaluation windows (e.g., Train: 2024-08 to 2024-10; Test: 2025-06 to 2025-11) rather than naive calendar cutoffs.
3. **Artifact Integrity**:
   - All 16 model pipelines (`.joblib`), predictions (`.parquet`), confusion matrices (`.csv`/`.png`), and metrics (`.json`) were re-verified against raw predictions (`verification.json`: passed).

---

## 2. PhiUSIIL Leakage & Artifact Investigation

### 2.1 The Role of `URLSimilarityIndex`
- **Definition in Literature**: In the PhiUSIIL framework paper (Prasad & Chandra, Computers & Security, 2023), `URLSimilarityIndex` measures character/subdomain similarity against an established set of legitimate target websites.
- **Observed Sample Statistics**:
  - Legitimate ($y=0$): $N = 40,105$, $\mu = 100.00$, $\sigma = 0.00$, Min = $100.0$, Max = $100.0$.
  - Phishing ($y=1$): $N = 29,895$, $\mu = 49.62$, $\sigma = 22.47$, Min = $0.0$, Max = $100.0$.
- **Leakage Assessment**: Direct target leakage. At inference time on arbitrary unlabelled URLs, the classifier cannot know which target brand is supposedly spoofed. Including this feature artificially guarantees near-zero misclassification.

### 2.2 Webpage Content Disparities
When examining the 31 webpage-derived features:
| Feature | Legitimate Mean | Phishing Mean | Disparity Mechanism |
| :--- | :---: | :---: | :--- |
| `LineOfCode` | 1,962.6 (median 1,098) | 65.9 (median 11) | Live enterprise homepages vs. short single-file phishing templates |
| `HasSocialNet` | 79.26% | 0.49% | Established web properties link to social platforms; phishing kits do not |
| `HasCopyrightInfo` | 80.69% | 5.58% | Production footers contain legal notices |
| `DomainTitleMatchScore` | 75.29 | 16.56 | Legitimate title matches domain; phishing sites have generic/missing titles |
| `NoOfExternalRef` | 32.5 | 1.8 | High-traffic homepages reference external CDNs and partners |

### 2.3 URL Structural Sampling Artifact
- Legitimate URLs in PhiUSIIL were scraped from top-domain lists (Alexa/Cisco Umbrella) as apex/www root domains without crawling internal links.
  - Queries (`NoOfQMarkInURL > 0`): Legitimate = **0.0%**, Phishing = **5.9%**
  - Parameter Delimiters (`NoOfEqualsInURL > 0`): Legitimate = **0.0%**, Phishing = **5.2%**
  - Subdirectories (`path > 1 char`): Legitimate = **0.0%**, Phishing = **27.0%**
- **Ablation Study on PhiUSIIL**:
  - Full Features (with `URLSimilarityIndex`): Random F1 = $0.9999$, Domain F1 = $1.0000$
  - Full Features (without `URLSimilarityIndex`): Random F1 = $0.9999$, Domain F1 = $0.9997$
  - Lexical URL-only (20 features, no HTML): Random F1 = $0.9962$, Domain F1 = $0.9936$
  - This ablation proves that high scores persist because of the fundamental homepage vs. deep-phish URL structure divergence present in the dataset itself.

---

## 3. PhreshPhish Data-Quality & Temporal Investigation

### 3.1 Feature Extraction Integrity
All 15 features in `src/features/url_features.py` are strictly lexical and deterministic.
- Correlation with Phishing Label:
  - `n_subdirectories`: $-0.424$
  - `hostname_length`: $+0.391$
  - `path_length`: $-0.281$
  - `is_https`: $-0.264$
  - `n_query_params`: $+0.165$
  - `n_suspicious_keywords`: $+0.127$
- No single feature dominates. Features behave as expected in modern phishing literature: attackers use longer spoofed subdomains/hostnames on root paths with fewer legitimate subdirectories.

### 3.2 Chronological Distribution & Temporal Gaps
Analysis of the full 666,315-record dataset and the 40,000-sample extracted corpus revealed severe collection irregularities:

| Period | Benign Records | Phish Records | Phish Ratio | Viability for Temporal Testing |
| :--- | :---: | :---: | :---: | :--- |
| **2024-07** | 7 | 3,172 | 99.8% | Incomplete benign collection |
| **2024-08 to 2024-10** | 10,459 | 4,066 | 28.0% | **Window 1: Strong continuous representation** |
| **2024-11 to 2025-04** | **0** | 4,909 | **100.0%** | **Unusable as standalone test split** (zero negative class) |
| **2025-05** | **0** | **0** | — | Collection blackout |
| **2025-06 to 2025-11** | 11,450 | 5,864 | 33.9% | **Window 2: Strong continuous representation** |
| **2025-12** | 73 | 0 | 0.0% | Incomplete phish collection |

**Temporal Strategy for Phase 3**:
To perform methodologically sound temporal drift evaluation without single-class evaluation collapse:
- **Training Window (Historical Baseline):** 2024-08 to 2024-10 ($N \approx 14,500$).
- **Test Window (Temporal Shift):** 2025-06 to 2025-11 ($N \approx 17,300$).
- This provides a clean 8-month temporal gap between train and test distributions where both classes are realistically represented.

---

## 4. Split Validation & Domain Overlap

| Dataset | Split Method | Train Rows | Test Rows | Test Domain Overlap | Domain Leakage |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PhiUSIIL** | Random Stratified | 56,000 | 14,000 | 6.56% (741 domains) | Moderate domain reuse |
| **PhiUSIIL** | Domain-Disjoint | 56,054 | 13,946 | **0.0% (0 domains)** | **Cleanly eliminated** |
| **PhreshPhish** | Random Stratified | 32,000 | 8,000 | 28.38% (1,346 domains) | High domain reuse |
| **PhreshPhish** | Domain-Disjoint | 31,789 | 8,211 | **0.0% (0 domains)** | **Cleanly eliminated** |

In PhreshPhish, eliminating domain overlap caused model performance to drop realistically:
- Random Forest F1: $0.9042 \rightarrow 0.8770$
- XGBoost F1: $0.9065 \rightarrow 0.8748$
- False Positive Rate: $7.19\% \rightarrow 7.54\%$ (RF), $7.30\% \rightarrow 8.45\%$ (XGBoost).

---

## 5. Identified Issues & Corrections Made

| Component | Identified Issue | Severity | Correction Implemented |
| :--- | :--- | :--- | :--- |
| **PhiUSIIL Config** | `URLSimilarityIndex` was included in active features despite being target-derived | High (Target Leakage) | Added `URLSimilarityIndex` to `exclude_from_model` in `configs/experiment_config.yaml`. Retrained all 8 PhiUSIIL models. |
| **Version Preservation** | Previous results needed to be preserved for comparative documentation | Medium | Saved previous outputs to `results/metrics/summary_v1_baseline.json` and `results/metrics/comparison_v1_baseline.csv`. |
| **PhreshPhish Temporal Plan** | Uniform chronological splitting would produce 100% phishing test sets between Nov 2024 and Apr 2025 | High (Methodological Flaw) | Documented the exact valid temporal windows (Window 1: 2024-08/10, Window 2: 2025-06/11) for Phase 3. |

---

## 6. Corrected Baseline Metrics

### 6.1 PhiUSIIL Comparison: With vs. Without `URLSimilarityIndex`

| Split | Model | Metric | V1 (With Leakage) | V2 (Audited & Corrected) | Delta |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **split_random** | Logistic Regression | F1-Score | 1.000000 | 0.999498 | -0.000502 |
| | | False Positive Rate | 0.000000 | 0.000249 | +0.000249 |
| | Random Forest | F1-Score | 0.999916 | 0.999916 | 0.000000 |
| | XGBoost | F1-Score | 0.999833 | 0.999916 | +0.000083 |
| | SGDClassifier | F1-Score | 0.999833 | 0.999331 | -0.000502 |
| **split_domain** | Logistic Regression | F1-Score | 0.999832 | 0.999411 | -0.000421 |
| | | False Positive Rate | 0.000000 | 0.000375 | +0.000375 |
| | Random Forest | F1-Score | 1.000000 | 0.999747 | -0.000253 |
| | XGBoost | F1-Score | 1.000000 | 0.999748 | -0.000252 |
| | SGDClassifier | F1-Score | 0.999832 | 0.999495 | -0.000337 |

Removing `URLSimilarityIndex` eliminates the direct label proxy and introduces realistic false positives and false negatives, while highlighting that webpage features and root-URL structural dichotomy maintain the remaining high separation.

---

## 7. Remaining Limitations

1. **PhiUSIIL Cross-Dataset Unsuitability**: PhiUSIIL cannot be directly evaluated against PhreshPhish using its published tabular features because PhreshPhish lacks HTML webpage features. Cross-dataset testing in Phase 6 will require extracting common lexical URL features using `src/features/url_features.py` on PhiUSIIL URLs.
2. **PhreshPhish Label Seasonality**: Attack campaigns fluctuate, and months with high phishing volumes reflect bursty scraping of particular sources (e.g. OpenPhish, PhishTank).

---

## 8. Readiness Assessment for Phase 3

- [x] Baseline pipelines are deterministic and reproducible.
- [x] Leakage feature `URLSimilarityIndex` has been removed and documented.
- [x] Domain-disjoint splits have verified zero domain overlap.
- [x] All 16 models have saved pipelines, predictions, and verified metric summaries.
- [x] PhreshPhish temporal windows are identified and verified for valid binary classification evaluation.

**Verdict: READY FOR PHASE 3 (Temporal Distribution Shift Experiments).**

