# Experiment Log

Times are UTC.

- 2026-11-10T18:13:47.022735+00:00 [download] PhiUSIIL archive present {"sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a"}
- 2026-11-10T18:16:01.820721+00:00 [download] PhreshPhish columns retrieved {"rows": 666315, "shards": 77}
- 2026-11-10T18:16:27.638076+00:00 [preprocess] PhiUSIIL processed {"rows": 70000}
- 2026-11-10T18:17:00.414396+00:00 [preprocess] PhiUSIIL processed {"rows": 70000}
- 2026-11-10T18:17:05.113946+00:00 [preprocess] PhreshPhish processed {"rows": 40000, "date_min": "2024-07-02 00:00:00+00:00", "date_max": "2025-12-16 00:00:00+00:00"}
- 2026-11-10T18:17:22.591854+00:00 [pipeline] Phase 1 and Phase 2 started
- 2026-11-10T18:17:22.608950+00:00 [download] PhiUSIIL archive present {"sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a"}
- 2026-11-10T18:17:24.614095+00:00 [preprocess] PhiUSIIL processed {"rows": 70000}
- 2026-11-10T18:17:29.311065+00:00 [preprocess] PhreshPhish processed {"rows": 40000, "date_min": "2024-07-02 00:00:00+00:00", "date_max": "2025-12-16 00:00:00+00:00"}
- 2026-11-10T18:17:30.943378+00:00 [train] Trained phiusiil__split_random__logistic_regression {"f1": 1.0, "roc_auc": 1.0}
- 2026-11-10T18:17:33.176487+00:00 [train] Trained phiusiil__split_random__random_forest {"f1": 0.9999163669816844, "roc_auc": 1.0}
- 2026-11-10T18:17:34.601580+00:00 [train] Trained phiusiil__split_random__xgboost {"f1": 0.9998327199732352, "roc_auc": 0.9999989991150301}
- 2026-11-10T18:17:35.547993+00:00 [train] Trained phiusiil__split_random__sgd_classifier {"f1": 0.9998327199732352, "roc_auc": 1.0}
- 2026-11-10T18:17:36.587331+00:00 [train] Trained phiusiil__split_domain__logistic_regression {"f1": 0.9998316781686585, "roc_auc": 0.9999999579476502}
- 2026-11-10T18:17:38.646296+00:00 [train] Trained phiusiil__split_domain__random_forest {"f1": 1.0, "roc_auc": 1.0}
- 2026-11-10T18:17:39.836966+00:00 [train] Trained phiusiil__split_domain__xgboost {"f1": 1.0, "roc_auc": 1.0}
- 2026-11-10T18:17:40.606574+00:00 [train] Trained phiusiil__split_domain__sgd_classifier {"f1": 0.9998316781686585, "roc_auc": 0.9999999579476502}
- 2026-11-10T18:17:41.830860+00:00 [train] Trained phreshphish__split_random__logistic_regression {"f1": 0.8343373493975904, "roc_auc": 0.9057219045353763}
- 2026-11-10T18:17:42.856625+00:00 [train] Trained phreshphish__split_random__random_forest {"f1": 0.9042092015102783, "roc_auc": 0.9736910031034751}
- 2026-11-10T18:17:43.278553+00:00 [train] Trained phreshphish__split_random__xgboost {"f1": 0.9064808362369338, "roc_auc": 0.971677506483564}
- 2026-11-10T18:17:43.649455+00:00 [train] Trained phreshphish__split_random__sgd_classifier {"f1": 0.8338583755649911, "roc_auc": 0.9049762094677835}
- 2026-11-10T18:17:44.611944+00:00 [train] Trained phreshphish__split_domain__logistic_regression {"f1": 0.8265507704464639, "roc_auc": 0.911696481653629}
- 2026-11-10T18:17:45.430514+00:00 [train] Trained phreshphish__split_domain__random_forest {"f1": 0.8769506974174838, "roc_auc": 0.9515118032931273}
- 2026-11-10T18:17:45.737665+00:00 [train] Trained phreshphish__split_domain__xgboost {"f1": 0.8747945205479452, "roc_auc": 0.9503343400772972}
- 2026-11-10T18:17:45.993593+00:00 [train] Trained phreshphish__split_domain__sgd_classifier {"f1": 0.819235836627141, "roc_auc": 0.9088153063217362}
- 2026-11-10T18:17:46.478562+00:00 [pipeline] Phase 1 and Phase 2 finished {"models": 16, "verification_passed": true, "phiusiil_rows": 70000, "phreshphish_rows": 40000}
- 2026-11-10T18:29:37.290285+00:00 [preprocess] PhiUSIIL processed {"rows": 70000}
- 2026-11-10T18:29:45.092497+00:00 [pipeline] Phase 1 and Phase 2 started
- 2026-11-10T18:29:45.109019+00:00 [download] PhiUSIIL archive present {"sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a"}
- 2026-11-10T18:29:47.054759+00:00 [preprocess] PhiUSIIL processed {"rows": 70000}
- 2026-11-10T18:29:51.370027+00:00 [preprocess] PhreshPhish processed {"rows": 40000, "date_min": "2024-07-02 00:00:00+00:00", "date_max": "2025-12-16 00:00:00+00:00"}
- 2026-11-10T18:29:52.854019+00:00 [train] Trained phiusiil__split_random__logistic_regression {"f1": 0.9994981599197056, "roc_auc": 0.9999954751658648}
- 2026-11-10T18:29:55.611079+00:00 [train] Trained phiusiil__split_random__random_forest {"f1": 0.9999163669816844, "roc_auc": 0.9999998748893787}
- 2026-11-10T18:29:57.188492+00:00 [train] Trained phiusiil__split_random__xgboost {"f1": 0.9999163809683084, "roc_auc": 1.0}
- 2026-11-10T18:29:58.095247+00:00 [train] Trained phiusiil__split_random__sgd_classifier {"f1": 0.9993309918046496, "roc_auc": 0.9999959547565797}
- 2026-11-10T18:29:59.075236+00:00 [train] Trained phiusiil__split_domain__logistic_regression {"f1": 0.9994109231675503, "roc_auc": 0.9999940916448541}
- 2026-11-10T18:30:01.697994+00:00 [train] Trained phiusiil__split_domain__random_forest {"f1": 0.9997474960020201, "roc_auc": 0.9999999579476503}
- 2026-11-10T18:30:02.724824+00:00 [train] Trained phiusiil__split_domain__xgboost {"f1": 0.9997476234541937, "roc_auc": 0.9999997687120762}
- 2026-11-10T18:30:03.414423+00:00 [train] Trained phiusiil__split_domain__sgd_classifier {"f1": 0.9994950345059754, "roc_auc": 0.9999945121683521}
- 2026-11-10T18:30:04.470004+00:00 [train] Trained phreshphish__split_random__logistic_regression {"f1": 0.8343373493975904, "roc_auc": 0.9057219045353763}
- 2026-11-10T18:30:05.469122+00:00 [train] Trained phreshphish__split_random__random_forest {"f1": 0.9042092015102783, "roc_auc": 0.9736910031034751}
- 2026-11-10T18:30:05.933904+00:00 [train] Trained phreshphish__split_random__xgboost {"f1": 0.9064808362369338, "roc_auc": 0.971677506483564}
- 2026-11-10T18:30:06.330063+00:00 [train] Trained phreshphish__split_random__sgd_classifier {"f1": 0.8338583755649911, "roc_auc": 0.9049762094677835}
- 2026-11-10T18:30:07.264062+00:00 [train] Trained phreshphish__split_domain__logistic_regression {"f1": 0.8265507704464639, "roc_auc": 0.911696481653629}
- 2026-11-10T18:30:08.091250+00:00 [train] Trained phreshphish__split_domain__random_forest {"f1": 0.8769506974174838, "roc_auc": 0.9515118032931273}
- 2026-11-10T18:30:08.389175+00:00 [train] Trained phreshphish__split_domain__xgboost {"f1": 0.8747945205479452, "roc_auc": 0.9503343400772972}
- 2026-11-10T18:30:08.650905+00:00 [train] Trained phreshphish__split_domain__sgd_classifier {"f1": 0.819235836627141, "roc_auc": 0.9088153063217362}
- 2026-11-10T18:30:09.152298+00:00 [pipeline] Phase 1 and Phase 2 finished {"models": 16, "verification_passed": true, "phiusiil_rows": 70000, "phreshphish_rows": 40000}

Verification: `{"checked_result_files": 16, "mismatches": [], "passed": true}`

## Phase 3: Distribution Shift & Drift Detection Experiments
- [phase3_temporal] Historical Train (Aug-Oct 2024, N=14,525), Future Test (Jun-Nov 2025, N=18,314).
  - XGBoost: F1 degraded from 0.9065 to 0.8313 (-7.52%), Recall dropped to 0.7941.
  - Random Forest: Recall collapsed from 0.8976 to 0.7351.
  - Logistic Regression & SGD: FPR increased to 15.4% and 17.8%.
- [phase3_cross_dataset] Bidirectional Evaluation on 15 identical lexical URL features:
  - PhreshPhish -> PhiUSIIL: XGBoost Accuracy 38.72%, ROC-AUC 0.4521, FPR 98.50%.
  - PhiUSIIL -> PhreshPhish: XGBoost Accuracy 41.48%, ROC-AUC 0.4263, FPR 89.20%.
  - Confirmed benchmark sampling artifact (PhiUSIIL legitimate URLs have 0.0% path/query components).
- [phase3_drift] Unsupervised PSI and KS test monitoring:
  - Stable period (Jun-Sep 2025): mean PSI < 0.03.
  - Severe drift spike (Oct-Nov 2025): mean PSI = 0.2733, max feature PSI = 2.3128 (`path_length`).

## Phase 4: Model Adaptation Under Limited Label Budgets
- [phase4_adaptation] Evaluated 84 experimental runs across 4 strategies and budgets (0%, 1%, 5%, 10%, 100%):
  - Strategy A (Static): XGBoost F1 = 0.8313, 0 updates, 0 labels.
  - Strategy B (Periodic Retraining, 5% budget): XGBoost F1 = 0.8372 ± 0.0019, 5 updates, 916 labels, adaptation time 0.49s.
  - Strategy B (Periodic Retraining, 100% budget): XGBoost F1 = 0.8448, 5 updates, 18,314 labels, adaptation time 0.67s.
  - Strategy D (Drift-Triggered Retraining, 5% budget): XGBoost F1 = 0.8340 ± 0.0016, 1 update (80% reduction in updates), 916 labels, adaptation time 0.10s (5x faster).
  - Strategy C (Online Incremental SGD, 5% budget): SGD F1 = 0.6882, 5 updates, adaptation time < 0.003s.
- Completed all reports: `reports/phase3_phase4_complete_report.md`, `reports/adaptation_evaluation.md`, `reports/research_paper_draft.md`.

## Phase 5: Synthesis, Publication Figures & Reproducibility Harness
- [phase5] Generated authoritative index `results/final/final_results_summary.csv` and `results/final/experiment_registry.csv` indexing 128 experimental runs.
- [phase5_figures] Generated 9 publication-grade figures at 300 DPI under `results/final/figures/` (Figures 1 to 9).
- [phase5_reports] Authored academic research paper `reports/final_research_paper.md`, master synthesis `reports/phase5_complete_report.md`, and reproduction guide `reports/reproducibility_guide.md`.
- [phase5_validation] Automated verification suite passed 20/20 checks:
  - Metric parity verified: Calculated XGBoost temporal F1 (0.831325) matches metadata.
  - Compute reduction verified: Drift-triggered adaptation cut measured compute by 80.26% (0.096s vs 0.487s).
  - Target leakage guardrail verified: `URLSimilarityIndex` confirmed excluded in config.
  - Final Scientific Verdict: **Scientifically Defensible With Documented Limitations.**
# Phase 5 Synthesis Refresh — 2026-11-11

- Regenerated the authoritative summary, registry (128 entries), manifest, nine figures, final paper, master report, and reproduction guide from saved Phase 1–4 outputs.
- Validation: 20/20 checks passed, including temporal prediction parity and measured 80.26% adaptation compute reduction. Phase 1–4 experiments were not rerun.
- Reproduction: `python -m src.experiments.run_phase5`; validation only: `python -m src.experiments.run_phase5 --validate-only`.

