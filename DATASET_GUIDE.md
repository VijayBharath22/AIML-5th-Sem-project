# Dataset Guide

## PhiUSIIL Phishing URL Dataset

- **UCI Dataset Page:** https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- **Raw Files:** `data/raw/phiusiil/phiusiil_phishing_url_dataset.zip` (original, unchanged) and extracted CSV.
- **Original Counts:** 235,795 rows (100,945 phishing, 134,850 legitimate).
- **Sampled Subset:** 70,000 rows (stratified random sample).
- **Label Mapping:** Raw `0` is phishing (mapped to model `1`), Raw `1` is legitimate (mapped to model `0`).
- **Timestamp Availability:** None. PhiUSIIL has no collection dates and cannot be used for temporal evaluations.
- **Model Features (50 features):** 49 numerical features and 1 categorical feature (`TLD`).
- **Excluded Features:**
  - `URLSimilarityIndex`: Target-derived leakage variable (identically 100 for all legitimate, mean 49.6 for phishing).
  - Identifiers: `URL`, `Domain`, `FILENAME`.
  - Free-text: `Title`.
  - Diagnostics/Metadata: `label`, `raw_label`, `registrable_domain`, `split_random`, `split_domain`.
- **Known Structural Limitations:**
  - 100% of legitimate URLs in PhiUSIIL are root domain homepages without paths or query parameters.
  - Webpage features (`LineOfCode`, `HasSocialNet`, `HasCopyrightInfo`) reflect differences between live mature enterprise sites and short-lived phishing stub forms.

## PhreshPhish Dataset

- **Hugging Face Repository:** https://huggingface.co/datasets/phreshphish/phreshphish
- **Paper:** Dalton et al., *PhreshPhish*, arXiv:2507.10854 (2025).
- **Fields Processed:** `sha256`, `url`, `label`, `date`.
- **Fields Intentionally Excluded:** `html` (multi-gigabyte payload), `target`, `lang`, `lang_score`.
- **Retrieved Corpus:** 666,315 rows across 77 parquet shards.
- **Sampled Subset:** 40,000 rows (uniform random sample preserving chronological distribution).
- **Label Mapping:** Raw `'phish'` is phishing (model `1`), `'benign'` is legitimate (model `0`).
- **Date Range:** 2024-07-02 to 2025-12-16.
- **Features Extracted (15 features):** `url_length`, `hostname_length`, `path_length`, `n_dots`, `n_hyphens`, `n_digits`, `n_special_chars`, `n_subdirectories`, `n_query_params`, `is_https`, `is_ip_hostname`, `n_suspicious_keywords`, `n_subdomains`, `has_at_symbol`, `is_shortened`.
- **Critical Collection Gaps Identified in Audit:**
  - *2024-11 through 2025-04:* Contains **0 benign samples** (100% phishing).
  - *2025-05:* Contains **0 records** (collection blackout).
  - *2025-12:* Contains **0 phishing samples** (100% benign).
  - *Viable Temporal Windows:* Window 1 (2024-08 to 2024-10) and Window 2 (2025-06 to 2025-11).

## Cross-Dataset Compatibility

The processed feature tables are intentionally separated:
- PhiUSIIL: 50 tabular/webpage features.
- PhreshPhish: 15 lexical URL features.
Cross-dataset transfer (Phase 6) must extract the 15 lexical URL features from PhiUSIIL's raw URLs rather than concatenating or imputing missing webpage features.
