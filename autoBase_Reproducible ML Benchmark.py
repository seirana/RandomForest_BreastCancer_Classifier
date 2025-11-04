#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: seirana
"""

import argparse, json, os, joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# get_models(seed)
# -----------------------------------------------------------------------------
# Returns a dictionary of baseline machine learning models initialized
# with consistent random seeds for reproducibility.
#
# Models included:
#   1. "logreg" – Logistic Regression wrapped in a pipeline with StandardScaler.
#      - StandardScaler normalizes features to zero mean and unit variance.
#      - LogisticRegression is configured for balanced class weights,
#        increased max iterations (2000), and deterministic randomness.
#
#   2. "rf" – Random Forest Classifier.
#      - Ensemble of 300 decision trees trained on balanced subsamples.
#      - Runs in parallel using all CPU cores (n_jobs = -1).
#      - Provides a strong non-linear baseline.
#
# The function enables easy selection of model types by key ("logreg" or "rf")
# and ensures both models share the same random seed for reproducibility.
# -----------------------------------------------------------------------------
def get_models(seed):
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        ]),
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=seed
        ),
    }

# -----------------------------------------------------------------------------
# evaluate_cv(model, X, y, cv_splits=5, seed=0, calibrate=False)
# -----------------------------------------------------------------------------
# Performs stratified k-fold cross-validation on the provided model and dataset,
# computes evaluation metrics, and aggregates results across folds.
#
# Parameters
# ----------
# model : sklearn estimator
#     The machine learning model to be trained and evaluated (e.g., LogisticRegression, RandomForest).
# X : ndarray
#     Feature matrix (samples × features).
# y : ndarray
#     Target vector (class labels).
# cv_splits : int, optional (default=5)
#     Number of cross-validation folds (StratifiedKFold ensures balanced class ratios per fold).
# seed : int, optional (default=0)
#     Random seed for reproducibility of data splits.
# calibrate : bool, optional (default=False)
#     If True, wraps the model in CalibratedClassifierCV with isotonic regression
#     to improve probability calibration.
#
# Workflow
# --------
# 1. Initializes Stratified K-Fold cross-validation.
# 2. For each fold:
#    - Splits the data into training and test subsets.
#    - Optionally applies isotonic calibration to the model.
#    - Trains the model and predicts probabilities on the test data.
#    - Falls back to `decision_function` if `predict_proba` is unavailable,
#      rescaling the outputs to [0, 1].
#    - Converts probabilities to binary predictions (threshold = 0.5).
#    - Collects evaluation metrics for each fold:
#         * ROC-AUC
#         * Average Precision (PR-AUC)
#         * F1-score
#         * Accuracy
#         * Confusion matrix
# 3. Aggregates results across folds and computes mean ± std for each metric.
#
# Returns
# -------
# metrics : dict
#     Mean and standard deviation for all evaluation metrics across folds,
#     plus per-fold confusion matrices.
# y_true : ndarray
#     Concatenated true labels from all folds.
# y_prob : ndarray
#     Concatenated predicted probabilities from all folds.
# y_pred : ndarray
#     Concatenated binary predictions from all folds.
# mdl : sklearn estimator
#     The last trained (optionally calibrated) model.
#
# Purpose
# -------
# Provides a robust, reproducible evaluation of a model’s predictive
# performance and calibration quality through cross-validation.
# -----------------------------------------------------------------------------

def evaluate_cv(model, X, y, cv_splits=5, seed=0, calibrate=False):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    y_true_all, y_prob_all, y_pred_all = [], [], []
    cms = []
    roc_aucs, pr_aucs, f1s, accs = [], [], [], []

    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        mdl = model
        if calibrate:
            mdl = CalibratedClassifierCV(model, cv=3, method="isotonic")
        mdl.fit(Xtr, ytr)
        if hasattr(mdl, "predict_proba"):
            yprob = mdl.predict_proba(Xte)[:, 1]
        else:
            # fall back to decision_function if available
            yprob = mdl.decision_function(Xte)
            # scale to [0,1]
            yprob = (yprob - yprob.min()) / (yprob.max() - yprob.min() + 1e-12)
        ypred = (yprob >= 0.5).astype(int)

        y_true_all.append(yte)
        y_prob_all.append(yprob)
        y_pred_all.append(ypred)

        roc_aucs.append(roc_auc_score(yte, yprob))
        pr_aucs.append(average_precision_score(yte, yprob))
        f1s.append(f1_score(yte, ypred))
        accs.append(accuracy_score(yte, ypred))
        cms.append(confusion_matrix(yte, ypred))

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = np.concatenate(y_pred_all)
    metrics = {
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "cms": [cm.tolist() for cm in cms],
    }
    return metrics, y_true, y_prob, y_pred, mdl

# -----------------------------------------------------------------------------
# plot_and_save_curves(y_true, y_prob, outdir)
# -----------------------------------------------------------------------------
# Generates and saves three key evaluation plots (ROC, Precision–Recall,
# and Calibration curves) for binary classification performance.
#
# Parameters
# ----------
# y_true : ndarray
#     True class labels (0 or 1) collected from all cross-validation folds.
# y_prob : ndarray
#     Predicted probabilities for the positive class, concatenated across folds.
# outdir : str
#     Directory path where the plots will be saved. Created automatically if it
#     does not exist.
#
# Workflow
# --------
# 1. Creates the output directory (if missing).
# 2. Uses sklearn's built-in visualization utilities:
#    - RocCurveDisplay.from_predictions()
#    - PrecisionRecallDisplay.from_predictions()
#    - CalibrationDisplay.from_predictions()
# 3. Each plot is saved as a high-quality PNG file with a descriptive title:
#    - "roc_curve.png"
#    - "pr_curve.png"
#    - "calibration_curve.png"
# 4. Closes figures after saving to prevent memory leaks in repeated runs.
#
# Purpose
# -------
# This function provides a quick and consistent way to visualize model
# performance in terms of discrimination (ROC), precision–recall trade-off,
# and calibration reliability. These plots are critical for evaluating both
# how well a model separates classes and how trustworthy its probability
# estimates are.

def plot_and_save_curves(y_true, y_prob, outdir):
    os.makedirs(outdir, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    plt.savefig(os.path.join(outdir, "roc_curve.png"), bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(outdir, "pr_curve.png"), bbox_inches="tight")
    plt.close()

    # calibration
    fig = plt.figure()
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10)
    plt.title("Calibration Curve")
    plt.savefig(os.path.join(outdir, "calibration_curve.png"), bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# main()
# -----------------------------------------------------------------------------
# Entry point of the AutoBase toolkit.
# Orchestrates the entire machine learning workflow — from argument parsing,
# data loading, model training, evaluation, visualization, and artifact saving.
#
# Workflow
# --------
# 1. **Parse Command-Line Arguments**
#    - --model      : Choose between "logreg" (Logistic Regression) or "rf" (Random Forest)
#    - --cv         : Number of cross-validation folds (default: 5)
#    - --seed       : Random seed for reproducibility
#    - --calibrate  : Apply isotonic calibration for probability reliability
#    - --outdir     : Output directory to save artifacts (default: "artifacts")
#
# 2. **Load Dataset**
#    - Uses the built-in Breast Cancer dataset from scikit-learn.
#    - Extracts the feature matrix (X) and binary target vector (y).
#
# 3. **Select and Evaluate Model**
#    - Calls get_models(seed) to build model dictionary.
#    - Selects the model specified by --model argument.
#    - Calls evaluate_cv() to perform stratified cross-validation,
#      compute metrics, and return predictions and trained model.
#
# 4. **Save Evaluation Results**
#    - Writes metrics summary to "metrics.json".
#    - Calls plot_and_save_curves() to generate ROC, PR, and Calibration plots.
#
# 5. **Train Final Model on Full Dataset**
#    - Fits the chosen model (optionally calibrated) on all data.
#    - Saves the trained model to "best_model_<model>.joblib".
#
# 6. **Generate Confusion Matrix**
#    - Computes confusion matrix on pooled CV predictions.
#    - Saves it as "confusion_matrix.csv".
#
# 7. **Print Summary**
#    - Displays metrics and artifact directory location in the console.
#
# Purpose
# -------
# This function integrates all modular components (data loading,
# model training, evaluation, and visualization) into a single,
# reproducible experiment pipeline that can be executed via CLI.
#
# Example Usage
# -------------
# python main.py --model logreg --cv 5 --calibrate --outdir artifacts
# python main.py --model rf --cv 10 --seed 123
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Baseline ML Classifier + Evaluation Toolkit")
    ap.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"])
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true", help="Apply isotonic calibration via CV")
    ap.add_argument("--outdir", type=str, default="artifacts")
    args = ap.parse_args()

    data = load_breast_cancer()
    X = data["data"]
    y = data["target"]

    models = get_models(args.seed)
    base_model = models[args.model]

    metrics, y_true, y_prob, y_pred, fitted = evaluate_cv(
        base_model, X, y, cv_splits=args.cv, seed=args.seed, calibrate=args.calibrate
    )

    # -----------------------------------------------------------------------------
    # Save evaluation metrics as a JSON file
    # -----------------------------------------------------------------------------
    # Writes the performance summary (ROC-AUC, PR-AUC, F1, Accuracy, etc.)
    # returned by evaluate_cv() into a human-readable JSON file.
    # This file acts as a permanent record of model evaluation results
    # for later comparison or visualization.
    #
    # Example output path:
    #   artifacts/metrics.json
    # -----------------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_and_save_curves(y_true, y_prob, args.outdir)

    # final fit on all data for deployment
    final_model = base_model
    if args.calibrate:
        final_model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
    final_model.fit(X, y)
    joblib.dump(final_model, os.path.join(args.outdir, f"best_model_{args.model}.joblib"))

    # confusion matrix on pooled CV predictions
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(args.outdir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    print("Done. Artifacts in:", args.outdir)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
