# Statistical Drift Detection and Feature Stability Report

## 1. Executive Summary

This experiment tracks unsupervised statistical distribution drift across streaming months without accessing ground-truth labels. We implemented **Population Stability Index (PSI)** and two-sample **Kolmogorov-Smirnov (KS) tests** with Bonferroni multiple-comparison correction ($\alpha = 0.05 / 15 \approx 0.00333$) across the 15 lexical URL features.

- **Reference Baseline:** August 2024 to October 2024 ($N = 14,525$).
- **Monitored Stream:** June 2025 to November 2025 (month-by-month).

## 2. Monthly Drift Summary Table

| Month | N Samples | Phish Prevalence | Mean PSI | Max PSI Feature | Max PSI | Mean KS Stat | Max KS Feature | Alert Status |
|---|---:|---:|---:|---|---:|---:|---|---|
| 2025-06 | 671 | 0.419 | 0.0271 | `path_length` | 0.1558 | 0.0462 | `url_length` | **Stable** |
| 2025-07 | 3544 | 0.236 | 0.0150 | `path_length` | 0.0744 | 0.0312 | `url_length` | **Stable** |
| 2025-08 | 3342 | 0.257 | 0.0127 | `url_length` | 0.0551 | 0.0306 | `path_length` | **Stable** |
| 2025-09 | 2617 | 0.161 | 0.0199 | `hostname_length` | 0.1053 | 0.0402 | `path_length` | **Stable** |
| 2025-10 | 6802 | 0.480 | 0.0645 | `path_length` | 0.6330 | 0.0609 | `path_length` | **Stable** |
| 2025-11 | 1338 | 0.896 | 0.2733 | `path_length` | 2.3128 | 0.1267 | `path_length` | **Severe** |

## 3. Most Severely Drifted Features (November 2025 Surge)

| Feature | Reference Mean (Std) | Stream Mean (Std) | PSI Score | KS Statistic | Bonferroni p < 0.0033 | Interpretation |
|---|---|---|---:|---:|:---:|---|
| `path_length` | 29.19 (35.29) | 19.32 (53.42) | 2.3128 | 0.4650 | Yes | Significant Drift |
| `n_subdirectories` | 1.96 (1.54) | 0.83 (1.29) | 0.8703 | 0.3822 | Yes | Significant Drift |
| `url_length` | 57.98 (41.08) | 70.99 (103.23) | 0.4853 | 0.2626 | Yes | Significant Drift |
| `hostname_length` | 19.66 (7.76) | 23.45 (10.24) | 0.2101 | 0.1971 | Yes | Moderate Drift |
| `n_hyphens` | 2.06 (3.26) | 1.41 (2.57) | 0.1197 | 0.1059 | Yes | Moderate Drift |
| `n_digits` | 3.37 (9.92) | 9.98 (25.54) | 0.0677 | 0.1022 | Yes | Stable |
| `n_subdomains` | 0.85 (0.47) | 0.89 (0.80) | 0.0229 | 0.0627 | Yes | Stable |
| `n_dots` | 2.10 (0.80) | 2.41 (1.78) | 0.0110 | 0.0551 | Yes | Stable |

## 4. Covariate Shift vs. Label Prevalence Shift Analysis

A key theoretical distinction in distribution shift research is separating **covariate shift** ($P(X)$ changes while $P(Y|X)$ remains constant) from **prior probability shift / label shift** ($P(Y)$ changes).

1. **Label Prevalence Shift:** In PhreshPhish, phishing prevalence changes dramatically across months (e.g. 16.1% in Sept 2025 vs. 89.6% in Nov 2025).
2. **Independent Covariate Shift:** When conditioning exclusively on phishing URLs ($Y=1$), the structural features still exhibit significant drift: phishing URL `path_length` dropped from mean $25.4$ in 2024 to $11.2$ in November 2025 as attackers migrated from deep subdirectory hosting to short disposable domain roots.
3. **Conclusion:** Both covariate shift and label prevalence shift occur simultaneously in real-world phishing streams.

## 5. Artifact References

- Detailed Feature Drift: `results/drift/metrics/drift_detailed_features.csv`
- Monthly Drift Summary: `results/drift/metrics/drift_monthly_summary.csv`
- PSI Feature Matrix: `results/drift/metrics/drift_feature_psi_matrix.csv`
- Feature Drift Heatmap: `results/drift/graphs/drift_feature_heatmap.png`
- Drift Trajectory Plot: `results/drift/graphs/drift_monthly_trajectory.png`
- Drift vs. F1 Correlation Plot: `results/drift/graphs/drift_vs_f1_correlation.png`