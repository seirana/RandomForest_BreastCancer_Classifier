# Model card

## System

**Project:** RandomForest_BreastCancer_Classifier / Breast Cancer Baseline

**Models:** Logistic Regression and Random Forest

**Dataset:** `sklearn.datasets.load_breast_cancer`

**Task:** binary classification of the built-in diagnostic dataset.

## Intended use

The repository is intended for learning reproducible binary-classification evaluation, comparing a linear and non-linear baseline, demonstrating leakage-safe scaling and nested probability calibration, and generating auditable machine-readable experiment artifacts.

## Not intended for

The serialized model is not intended for diagnosis, screening, treatment decisions, clinical triage, patient-specific risk estimation, or deployment as a medical device.

## Target semantics

The sklearn source target is remapped so `1 = malignant` and `0 = benign`. This is explicitly recorded in metrics and metadata.

## Inputs

The dataset contains 30 continuous features from the classic Wisconsin Diagnostic Breast Cancer dataset distributed by scikit-learn. The maintained repository does not ingest real patient records.

## Evaluation

The project uses stratified out-of-fold cross-validation. Optional calibration is nested inside each outer training fold. Uncertainty is summarized with a stratified bootstrap over OOF predictions.

## Interpretability artifacts

For Logistic Regression, standardized coefficients are exported. For Random Forest, mean-decrease-in-impurity feature importance is exported. Neither artifact is a causal explanation of malignancy.

## Known limitations

- small benchmark dataset;
- no independent external validation cohort;
- no demographic subgroup analysis;
- no prospective validation;
- no threshold optimization tied to clinical costs;
- no decision-curve analysis;
- no missing-data study;
- no robustness analysis for measurement shift;
- bootstrap intervals do not include full retraining uncertainty;
- model artifacts are fit on the entire dataset after evaluation.

## Appropriate interpretation

Appropriate: under the specified cross-validation procedure on the built-in sklearn dataset, the model produced the reported out-of-fold discrimination, threshold, and probability-quality metrics.

Not appropriate: the model is sufficiently accurate to diagnose breast cancer in clinical practice.

## Reproducibility

Preserve the Git commit SHA, CLI arguments, random seed, cross-validation fold count, calibration method, decision threshold, bootstrap repetitions, package versions, dataset SHA-256 fingerprint, generated metrics, and OOF predictions.
