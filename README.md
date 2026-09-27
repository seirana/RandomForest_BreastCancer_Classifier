# RandomForest_BreastCancer_Classifier

This repository is a reproducible binary-classification benchmark built on scikit-learn's Wisconsin Diagnostic Breast Cancer dataset. Despite the historical repository name, the maintained project compares **Logistic Regression** and **Random Forest**.

The upgrade focuses on correct out-of-fold evaluation, explicit class semantics, probability calibration, uncertainty reporting, reproducible artifacts, tests, CI, and clear limits on medical interpretation.

> **Important:** this project is an educational/research benchmark. It is not a clinical diagnostic device and must not be used to diagnose patients.

## The most important correction: malignant is the positive class

The scikit-learn dataset encodes `0 = malignant` and `1 = benign`. The historical implementation passed those labels directly to standard binary metrics, so F1, recall, precision, and average precision treated **benign** as the positive class.

The maintained implementation explicitly remaps the target to `0 = benign` and `1 = malignant`. Therefore sensitivity/recall refers to malignant cases, precision refers to predicted malignant cases, and average precision treats malignancy as the positive event.

## Dataset

Source: `sklearn.datasets.load_breast_cancer`.

The bundled dataset contains 569 samples, 30 continuous features, 212 malignant cases, and 357 benign cases. Because the dataset ships with scikit-learn, the benchmark does not require downloading patient data.

## Evaluation design

The core evaluation uses shuffled, stratified K-fold cross-validation. For every outer fold, a **fresh clone** of the estimator is fit on the training partition; optional probability calibration is performed only inside that training partition; probabilities are generated for the held-out fold; and held-out predictions are written back into their original sample positions.

After all folds, every sample has exactly one **out-of-fold (OOF)** prediction from a model that was not trained on that sample.

The pipeline reports both per-fold metrics with mean and sample standard deviation, and pooled OOF metrics calculated across all held-out predictions. See [EVALUATION.md](EVALUATION.md).

## Metrics

The maintained pipeline reports ROC-AUC, Average Precision / PR-AUC, Accuracy, Balanced Accuracy, F1, Precision, Sensitivity / Recall, Specificity, Brier score, Log loss, and a confusion matrix.

A fixed threshold of 0.5 is used by default for threshold-dependent metrics. It can be changed explicitly with `--threshold`, but it is **not tuned on held-out folds**.

## Confidence intervals

The pipeline can perform a stratified sample bootstrap over pooled OOF predictions. By default it uses 1000 bootstrap replicates and a 95% percentile interval while preserving the malignant/benign class counts in each resample.

These intervals characterize uncertainty in the evaluated sample predictions. They do not capture every source of uncertainty, such as retraining on a new population or changes in data collection.

## Probability calibration

Calibration is optional with `--calibration none`, `--calibration sigmoid`, or `--calibration isotonic`.

When calibration is enabled, `CalibratedClassifierCV` runs **inside each outer training fold**. The outer held-out fold is not used to fit the calibrator.

The historical `--calibrate` flag remains available as an alias for `--calibration isotonic`.

## Models

### Logistic Regression

The logistic baseline uses `StandardScaler`, class-balanced Logistic Regression, and deterministic seed configuration. The exported `feature_coefficients.csv` contains standardized coefficients from an uncalibrated full-data reference fit. Coefficient magnitude is model interpretation, not causal importance.

### Random Forest

The Random Forest uses 300 trees, balanced subsample class weights, a deterministic random seed, and configurable parallelism. The exported `feature_importances.csv` contains mean-decrease-in-impurity feature importance from a full-data reference fit. These importances are not causal effects.

## Installation

```bash
git clone https://github.com/seirana/RandomForest_BreastCancer_Classifier.git
cd RandomForest_BreastCancer_Classifier

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

## Run one model

Logistic Regression:

```bash
breast-cancer-baseline --model logreg --cv 5 --seed 42 --bootstrap-reps 1000
```

Random Forest:

```bash
breast-cancer-baseline --model rf --cv 5 --seed 42 --bootstrap-reps 1000
```

The historical entry point remains available after installation:

```bash
python main.py --model rf --cv 5
```

## Compare both models

```bash
breast-cancer-baseline --model all --cv 5 --seed 42
```

Both models use the same cross-validation configuration. Results are stored under separate subdirectories plus a machine-readable `comparison.json`.

## Generated artifacts

A single-model run writes `metrics.json`, `oof_predictions.csv`, `confusion_matrix.csv`, ROC/PR/calibration plots, a model-interpretation CSV, `best_model_<model>.joblib`, and `run_metadata.json`.

For `--model all`, model-specific outputs are stored under `artifacts/logreg/` and `artifacts/rf/`; the root directory also contains `comparison.json` and `run_metadata.json`.

Generated models, plots, and metrics are not committed to Git.

## Reproducibility metadata

`run_metadata.json` records command arguments, the resolved calibration setting, Python/platform information, NumPy/pandas/scikit-learn versions, Git commit when available, dataset sample/feature counts, class semantics, and a SHA-256 fingerprint of the exact feature matrix, remapped target, and feature-name list.

## Testing

```bash
python -m pytest
python -m ruff check src tests main.py
```

Tests cover malignant-positive target remapping, sensitivity/specificity semantics, complete OOF prediction coverage, fixed-seed reproducibility, inner-fold calibration, bootstrap reproducibility, end-to-end artifact generation, and two-model comparison output.

GitHub Actions runs the maintained project on Python 3.10, 3.11, and 3.12 and builds the Docker image.

## Docker

```bash
docker build -t breast-cancer-baseline .
mkdir -p artifacts

docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" \
  breast-cancer-baseline \
  --model all \
  --outdir /app/artifacts
```

## What this repository demonstrates

The useful engineering/research outcome is not simply that a classifier receives a high score on a small benchmark dataset. It demonstrates a reusable evaluation pattern: explicit target semantics, fresh estimator per fold, optional train-only calibration, OOF probabilities, discrimination/threshold/calibration metrics, bootstrap uncertainty, and reproducible artifacts.

See [MODEL_CARD.md](MODEL_CARD.md) for limitations.

## License

No explicit license file is currently included. Repository visibility alone does not grant reuse rights.
