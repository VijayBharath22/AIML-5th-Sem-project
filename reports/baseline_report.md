# Baseline Experiment Report

## Objective

Estimate conventional baseline phishing-detection performance for four classifiers. These numbers are a reference point for later temporal, cross-dataset, and drift-adaptation experiments. They are not those experiments.

## Environment

- Python 3.14.10
- Platform: Linux-7.3.3-200.fc44.x86_64-x86_64-with-glibc2.43
- Packages: numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, matplotlib 3.11.2, pyarrow 26.0.1, tldextract 5.4.0
- Hardware intent: CPU training. NVIDIA drivers were not available in this environment, and no GPU training was used.

## Data used

- PhiUSIIL processed rows: 70000. Random split and domain-disjoint split were taken from this sample.
- PhreshPhish processed rows: 40000. Date range 2024-07-02 00:00:00+00:00 to 2025-12-16 00:00:00+00:00.
- Positive class: phishing, encoded as 1.
- Decision threshold: 0.5 on the predicted phishing probability.

## Models

1. Logistic regression, `lbfgs`, `class_weight=balanced`, median imputation and standard scaling inside the training pipeline.
2. Random forest, 100 trees, `max_depth=18`, `min_samples_leaf=2`, `n_jobs=2`.
3. XGBoost, 120 trees, `max_depth=6`, `learning_rate=0.1`, histogram tree method, `scale_pos_weight` set from the training class counts.
4. SGDClassifier, `loss=log_loss`, `class_weight=balanced`, same scaling pipeline as logistic regression.

No test labels or test-only category levels were used to fit imputation, scaling, or one-hot encoding. PhiUSIIL `TLD` is the only categorical model input. PhreshPhish uses the 15 URL features only. Separate pipelines were fit for each dataset and each split.

## Metric definitions

- Precision, recall, and F1 use phishing as the positive class.
- False positive rate is false phishing decisions divided by all legitimate test rows.
- ROC-AUC uses the predicted phishing probability.
- Confusion-matrix axes are legitimate (0) and phishing (1), with rows as actual labels.

## PhiUSIIL, stratified random split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.9996 | 0.9997 | 0.9993 | 0.9995 | 1.0000 | 0.0002 | 0.62 | 0.022 |
| random_forest | 0.9999 | 1.0000 | 0.9998 | 0.9999 | 1.0000 | 0.0000 | 2.38 | 0.046 |
| xgboost | 0.9999 | 0.9998 | 1.0000 | 0.9999 | 1.0000 | 0.0001 | 1.18 | 0.022 |
| sgd_classifier | 0.9994 | 0.9993 | 0.9993 | 0.9993 | 1.0000 | 0.0005 | 0.61 | 0.020 |


## PhiUSIIL, domain-disjoint split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.9995 | 0.9995 | 0.9993 | 0.9994 | 1.0000 | 0.0004 | 0.54 | 0.022 |
| random_forest | 0.9998 | 1.0000 | 0.9995 | 0.9997 | 1.0000 | 0.0000 | 2.44 | 0.046 |
| xgboost | 0.9998 | 0.9995 | 1.0000 | 0.9997 | 1.0000 | 0.0004 | 0.90 | 0.021 |
| sgd_classifier | 0.9996 | 0.9997 | 0.9993 | 0.9995 | 1.0000 | 0.0002 | 0.55 | 0.015 |


Domain overlap count for this split: 0

## PhreshPhish, stratified random split

The split is stratified by label. It is not a temporal split.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8488 | 0.8231 | 0.8459 | 0.8343 | 0.9057 | 0.1489 | 0.48 | 0.009 |
| random_forest | 0.9144 | 0.9110 | 0.8976 | 0.9042 | 0.9737 | 0.0719 | 0.66 | 0.045 |
| xgboost | 0.9161 | 0.9102 | 0.9028 | 0.9065 | 0.9717 | 0.0730 | 0.22 | 0.007 |
| sgd_classifier | 0.8484 | 0.8229 | 0.8451 | 0.8339 | 0.9050 | 0.1489 | 0.16 | 0.004 |


## PhreshPhish, domain-disjoint split

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Train s | Predict s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8396 | 0.8119 | 0.8417 | 0.8266 | 0.9117 | 0.1622 | 0.50 | 0.006 |
| random_forest | 0.8915 | 0.9038 | 0.8517 | 0.8770 | 0.9515 | 0.0754 | 0.64 | 0.045 |
| xgboost | 0.8887 | 0.8939 | 0.8565 | 0.8748 | 0.9503 | 0.0845 | 0.19 | 0.007 |
| sgd_classifier | 0.8329 | 0.8050 | 0.8340 | 0.8192 | 0.9088 | 0.1680 | 0.16 | 0.003 |


Domain overlap count for this split: 0

## Confusion matrices and graphs

PNG and CSV confusion matrices are in `results/confusion_matrices/`. Comparison bars, precision-recall-F1 bars, and ROC curves are in `results/graphs/` with names prefixed by dataset and split. Feature-importance plots for the random-split models are in the same folder. Class balance, missing-value fractions, and the PhreshPhish monthly sample counts are `results/graphs/class_distribution.png`, `missing_values.png`, and `phreshphish_monthly_counts.png`.

## Comparison

Best PhiUSIIL random-split model by F1: xgboost with F1 0.9999, ROC-AUC 1.0000, precision 0.9998, recall 1.0000, FPR 0.0001, accuracy 0.9999.

Best PhreshPhish random-split model by F1: xgboost with F1 0.9065, ROC-AUC 0.9717, precision 0.9102, recall 0.9028, FPR 0.0730, accuracy 0.9161.

The two scores are not a cross-dataset comparison. The feature sets differ. A higher PhiUSIIL score is expected if webpage and corpus-derived features are easier than lexical URL features, but that interpretation is a hypothesis supported only by the separate within-dataset numbers above.

Random-split domain overlap remains a leakage risk even when the domain string is not a feature. The domain-disjoint tables are the stricter baseline collected in this phase. Exact duplicate feature vectors between train and test are quantified in `reports/dataset_analysis.md`.

## Observations

- Metrics were recomputed from saved predictions. Verification status: passed=True, files checked=16, mismatches=0.
- Training and prediction times are wall-clock seconds on CPU.
- Errors recorded by the trainer: None.

## Limitations

- Hyperparameters are fixed baselines, not a tuned search.
- No temporal holdout was trained or scored.
- No model was trained on one dataset and tested on the other.
- Drift detectors and adaptation policies were not implemented.
- PhreshPhish HTML was not used.
- PhiUSIIL has no timestamps.
