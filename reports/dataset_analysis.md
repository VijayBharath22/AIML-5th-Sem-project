# Dataset Analysis

This report describes the data actually retrieved and processed for Phase 1. It does not report model adaptation or temporal test results.

## PhiUSIIL Phishing URL Dataset

- Source page: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- Download: `https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip`
- DOI recorded by UCI: https://doi.org/10.1016/j.cose.2023.103545
- UCI creation year: 2024. UCI citation text dates the paper 2023. The article number in the DOI is 103545.
- Original rows: 235795
- Rows after exact-URL deduplication: 235370
- Duplicate URLs removed: 425
- Label conflicts among duplicate URLs: 0
- Sampled rows: 70000 of 70000 requested
- Sampling: stratified random sample on the phishing label, sklearn train_test_split, seed from config
- Missing values in the processed sample: 0 columns
- Date coverage: none. The published table has no collection timestamp.

### Label mapping

UCI abstract: 134,850 legitimate and 100,945 phishing URLs. Those counts match raw `label` 1 and raw `label` 0 before deduplication. Modelling uses 1 for phishing and 0 for legitimate.

Sample class counts (model label): `{"0": 40105, "1": 29895}`

### Features

Model features (50): `URLLength`, `DomainLength`, `IsDomainIP`, `TLD`, `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`, `LineOfCode`, `LargestLineLength`, `HasTitle`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`

Categorical feature: TLD (525 distinct values in the sample). Rare TLD levels are collapsed with `OneHotEncoder(min_frequency=100)`, fit on the training split only.

Excluded from the model:

- `FILENAME`: UCI says this column can be ignored.
- `URL` and `Domain`: identifiers. `registrable_domain` is derived for split diagnostics and is not a feature.
- `Title`: free text, not encoded.
- `label` and `raw_label`: targets.

### Split diagnostics

Domain-disjoint status: `{"status": "ok", "seed": 43, "test_positive_rate": 0.4260719919690234, "test_fraction": 0.19922857142857142, "n_train": 56054, "n_test": 13946, "domain_overlap_count": 0, "random_split_domain_overlap_count": 741, "random_split_test_domains": 11288, "random_split_test_domains_also_in_train": 741, "n_domains": 54202}`

Exact feature-vector overlap on the random split: `{"random_split_test_rows": 14000, "random_split_test_rows_with_exact_train_feature_match": 40, "random_split_exact_feature_match_rate": 0.002857142857142857}`

### Limitations

PhiUSIIL cannot support a temporal evaluation on its own. Several published features (`URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`, title-match scores, and webpage features) are specific to this dataset and are not available for PhreshPhish. High random-split accuracy would not, by itself, show robustness to drift.

## PhreshPhish

- Dataset card: https://huggingface.co/datasets/phreshphish/phreshphish
- Paper linked from the dataset card: https://arxiv.org/abs/2507.10854
- Card version note: v1.0.1 (2026-02-07), with v1.0.0 released 2025-05-14.
- Rows retrieved (URL, label, date, sha256 only): 666315
- Invalid dates removed: 0
- Invalid URLs removed: 5744
- Duplicate URLs removed: 0
- Rows after cleaning: 660571
- Sampled rows: 40000 of 40000 requested
- Sampling: Uniform random sample without label stratification so the joint time and class distribution of the cleaned corpus is preserved in expectation. Seed from config.
- Date range in the sample: 2024-07-02 00:00:00+00:00 to 2025-12-16 00:00:00+00:00
- HTML, brand target, language, and language score were not downloaded.

### Label mapping

Dataset card values are `phish` and `benign`. Model label 1 is phishing and 0 is legitimate.

Cleaned class counts: `{"0": 362248, "1": 298323}`

Sample class counts: `{"0": 21989, "1": 18011}`

### URL features

- `url_length`: Character length of the raw URL.
- `hostname_length`: Character length of the parsed hostname.
- `path_length`: Character length of the URL path.
- `n_dots`: Count of '.' characters in the raw URL.
- `n_hyphens`: Count of '-' characters in the raw URL.
- `n_digits`: Count of digit characters in the raw URL.
- `n_special_chars`: Count of characters that are not alphanumeric and not in .:/-_?=&%+.
- `n_subdirectories`: Count of non-empty path segments.
- `n_query_params`: Count of query-string keys.
- `is_https`: 1 if the scheme is https, else 0.
- `is_ip_hostname`: 1 if the hostname is an IP address, else 0.
- `n_suspicious_keywords`: Count of distinct configured keywords present as substrings.
- `n_subdomains`: Number of subdomain labels from the public-suffix parse.
- `has_at_symbol`: 1 if '@' appears in the URL, else 0.
- `is_shortened`: 1 if the hostname is a configured shortening service, else 0.

Keywords and shorteners are listed in `configs/experiment_config.yaml`. The label, date, sha256, and source file are not features.

### Split diagnostics

Domain-disjoint status: `{"status": "ok", "seed": 57, "test_positive_rate": 0.45402508829618804, "test_fraction": 0.205275, "n_train": 31789, "n_test": 8211, "domain_overlap_count": 0, "random_split_domain_overlap_count": 1346, "random_split_test_domains": 4742, "random_split_test_domains_also_in_train": 1346, "n_domains": 19399}`

Exact feature-vector overlap on the random split: `{"random_split_test_rows": 8000, "random_split_test_rows_with_exact_train_feature_match": 3795, "random_split_exact_feature_match_rate": 0.474375}`

### Limitations

The original Hugging Face train and test splits were not used as this project's evaluation protocol. Preserving dates does not mean a temporal evaluation has been run. URL lexical features omit page HTML, which the PhreshPhish paper treats as part of the benchmark. Feature definitions differ from PhiUSIIL, so the two processed tables are not a cross-dataset experiment.

## Preprocessing controls

- Transformations inside model pipelines are fit on the training split only.
- Sampling and splits use seed 42.
- The positive class is phishing for every metric.
