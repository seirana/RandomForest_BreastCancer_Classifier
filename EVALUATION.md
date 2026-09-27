# Evaluation protocol

## Goal

The maintained benchmark evaluates Logistic Regression and Random Forest on scikit-learn's Wisconsin Diagnostic Breast Cancer dataset using reproducible out-of-fold cross-validation. The goal is methodology and software quality, not clinical validation.

## Class definition

The source dataset uses 0 = malignant and 1 = benign. The maintained pipeline remaps this to 0 = benign and 1 = malignant. This makes malignancy the positive class for precision, recall, F1, Average Precision, and the confusion matrix.

## Outer cross-validation

The benchmark uses shuffled `StratifiedKFold`. Every sample appears in exactly one held-out fold, and a fresh clone of the estimator is trained for each fold. The pooled probability vector therefore contains one out-of-fold probability for every sample.

This avoids reporting in-sample predictions as validation performance.

## Preprocessing leakage

Logistic Regression uses a scikit-learn `Pipeline` containing `StandardScaler` followed by `LogisticRegression`. The scaler is therefore fit only on each outer training fold. Random Forest does not require feature standardization.

## Calibration

If sigmoid or isotonic calibration is requested, `CalibratedClassifierCV` is fit inside each outer training fold. The outer test fold is never used to fit the classifier or probability calibrator.

## Threshold

The default classification threshold is 0.5. The pipeline does not optimize this threshold using outer test-fold outcomes. A user-supplied alternative should be treated as a pre-specified operating point.

## Metrics

Threshold-independent metrics are ROC-AUC and Average Precision. Threshold-dependent metrics are accuracy, balanced accuracy, F1, precision, sensitivity/recall, specificity, and confusion matrix. Probability-quality metrics are Brier score and log loss, plus a calibration curve.

No single metric is sufficient for medical classification.

## Fold summary versus pooled OOF metrics

The project reports both. The fold summary describes how metrics vary across cross-validation folds. The pooled OOF metric evaluates all held-out predictions together. Fold standard deviation is not presented as a confidence interval.

## Bootstrap interval

The pipeline uses a stratified nonparametric sample bootstrap over OOF predictions. Malignant and benign samples are resampled separately with replacement, preserving the original class counts. The 2.5th and 97.5th percentiles define the reported 95% interval.

This bootstrap conditions on the OOF prediction process. It does not reproduce full model retraining for every bootstrap sample, model dataset shift, or establish external clinical generalization.

## Final serialized model

After evaluation, the selected estimator is fit on the complete built-in dataset and serialized. That full-data model is an artifact for demonstration. Because it is fit on all samples, it does not have a separate hold-out estimate beyond the cross-validation results.

## Model comparison

`--model all` evaluates both model families with the same dataset, target mapping, number of folds, shuffle seed, threshold, calibration setting, and bootstrap configuration.

The command reports measured benchmark metrics for both models. Results on this fixed dataset do not establish performance on external populations.

## External validity

The evaluation does not test another hospital or laboratory, temporal drift, demographic subgroup performance, prospective deployment, decision-curve utility, or clinical harms from false positives and false negatives. Those questions require additional data and a different validation design.
