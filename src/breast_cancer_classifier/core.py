from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    positive_class: str
    negative_class: str
    source_name: str


@dataclass(frozen=True)
class EvaluationResult:
    summary: dict[str, Any]
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray


def load_dataset() -> DatasetBundle:
    """Load sklearn's Wisconsin breast-cancer dataset with malignant=1.

    sklearn stores malignant as 0 and benign as 1. For diagnostic reporting,
    this project explicitly remaps the clinically adverse class (malignant) to
    the positive label 1 so that recall means malignant-case sensitivity.
    """

    data = load_breast_cancer()
    X = np.asarray(data.data, dtype=np.float64)
    original_target = np.asarray(data.target, dtype=np.int64)
    y = (original_target == 0).astype(np.int64)

    if X.ndim != 2:
        raise ValueError("Dataset features must be a 2D array.")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("Dataset target must align with feature rows.")
    if not np.isfinite(X).all():
        raise ValueError("Dataset contains non-finite feature values.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Dataset must contain both binary classes.")

    return DatasetBundle(
        X=X,
        y=y,
        feature_names=[str(name) for name in data.feature_names],
        positive_class="malignant",
        negative_class="benign",
        source_name="sklearn.datasets.load_breast_cancer",
    )


def build_model(
    name: str,
    *,
    seed: int,
    n_jobs: int = -1,
) -> BaseEstimator:
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=int(seed),
                    ),
                ),
            ]
        )

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=int(n_jobs),
            random_state=int(seed),
        )

    raise ValueError(
        f"Unknown model {name!r}; expected 'logreg' or 'rf'."
    )


def _validate_binary_inputs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int,
) -> None:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y must contain both binary labels 0 and 1.")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2.")

    class_counts = np.bincount(y.astype(int), minlength=2)
    if int(class_counts.min()) < cv_splits:
        raise ValueError(
            "Each class must contain at least cv_splits samples."
        )


def _fitted_estimator(
    base_model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    calibration: str,
    calibration_cv: int,
) -> BaseEstimator:
    estimator = clone(base_model)

    if calibration == "none":
        estimator.fit(X_train, y_train)
        return estimator

    if calibration not in {"sigmoid", "isotonic"}:
        raise ValueError(
            "calibration must be one of: none, sigmoid, isotonic"
        )
    if calibration_cv < 2:
        raise ValueError("calibration_cv must be at least 2.")

    class_counts = np.bincount(y_train.astype(int), minlength=2)
    if int(class_counts.min()) < calibration_cv:
        raise ValueError(
            "Each outer-training class must contain at least calibration_cv samples."
        )

    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=calibration,
        cv=calibration_cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def fit_final_model(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    calibration: str = "none",
    calibration_cv: int = 3,
) -> BaseEstimator:
    """Fit a fresh estimator on all supplied data for demonstration/deployment use."""

    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    _validate_binary_inputs(
        features,
        labels,
        cv_splits=2,
    )
    return _fitted_estimator(
        model,
        features,
        labels,
        calibration=calibration,
        calibration_cv=calibration_cv,
    )


def _positive_probability(
    estimator: BaseEstimator,
    X: np.ndarray,
) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(
            estimator.predict_proba(X),
            dtype=float,
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                "predict_proba must return two binary-class columns."
            )
        values = probabilities[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(
            estimator.decision_function(X),
            dtype=float,
        )
        values = 1.0 / (
            1.0 + np.exp(-np.clip(scores, -500.0, 500.0))
        )
    else:
        raise TypeError(
            "Estimator must expose predict_proba or decision_function."
        )

    if not np.isfinite(values).all():
        raise ValueError("Estimator returned non-finite probabilities.")
    return np.clip(values, 0.0, 1.0)


def scalar_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)

    if truth.ndim != 1 or prob.ndim != 1 or len(truth) != len(prob):
        raise ValueError("y_true and y_prob must be aligned 1D arrays.")
    if set(np.unique(truth)) != {0, 1}:
        raise ValueError("y_true must contain both binary classes.")
    if not np.isfinite(prob).all():
        raise ValueError("y_prob must contain only finite values.")
    if np.any(prob < 0.0) or np.any(prob > 1.0):
        raise ValueError("y_prob values must be in [0, 1].")

    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        truth,
        pred,
        labels=[0, 1],
    ).ravel()
    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else float("nan")
    )

    return {
        "roc_auc": float(roc_auc_score(truth, prob)),
        "average_precision": float(
            average_precision_score(truth, prob)
        ),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, pred)
        ),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "precision": float(
            precision_score(truth, pred, zero_division=0)
        ),
        "sensitivity_recall": float(
            recall_score(truth, pred, zero_division=0)
        ),
        "specificity": specificity,
        "brier_score": float(brier_score_loss(truth, prob)),
        "log_loss": float(
            log_loss(truth, prob, labels=[0, 1])
        ),
    }


def _fold_summary(
    fold_records: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    metric_names = list(fold_records[0]["metrics"])
    output: dict[str, dict[str, float]] = {}

    for metric_name in metric_names:
        values = np.asarray(
            [
                fold["metrics"][metric_name]
                for fold in fold_records
            ],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        output[metric_name] = {
            "mean": float(finite.mean()),
            "std": float(
                finite.std(ddof=1)
                if len(finite) > 1
                else 0.0
            ),
        }

    return output


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float = 0.5,
    reps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    if reps < 0:
        raise ValueError("reps must be non-negative.")
    if reps == 0:
        return {
            "reps_requested": 0,
            "reps_valid": 0,
            "method": "stratified sample bootstrap",
            "ci95": {},
        }

    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if truth.shape != prob.shape or truth.ndim != 1:
        raise ValueError("y_true and y_prob must be aligned 1D arrays.")

    negative_idx = np.flatnonzero(truth == 0)
    positive_idx = np.flatnonzero(truth == 1)
    if len(negative_idx) == 0 or len(positive_idx) == 0:
        raise ValueError("Both classes are required for stratified bootstrap.")

    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {}

    for _ in range(reps):
        drawn_negative = rng.choice(
            negative_idx,
            size=len(negative_idx),
            replace=True,
        )
        drawn_positive = rng.choice(
            positive_idx,
            size=len(positive_idx),
            replace=True,
        )
        indices = np.concatenate(
            [drawn_negative, drawn_positive]
        )
        metrics = scalar_metrics(
            truth[indices],
            prob[indices],
            threshold=threshold,
        )
        for name, value in metrics.items():
            if np.isfinite(value):
                samples.setdefault(name, []).append(float(value))

    intervals: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        lower, upper = np.quantile(
            np.asarray(values, dtype=float),
            [0.025, 0.975],
        )
        intervals[name] = {
            "low": float(lower),
            "high": float(upper),
        }

    valid_counts = [len(values) for values in samples.values()]
    return {
        "reps_requested": int(reps),
        "reps_valid": int(min(valid_counts)) if valid_counts else 0,
        "method": "stratified sample bootstrap",
        "ci95": intervals,
    }


def evaluate_cv(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int = 5,
    seed: int = 42,
    calibration: str = "none",
    calibration_cv: int = 3,
    threshold: float = 0.5,
    bootstrap_reps: int = 1000,
) -> EvaluationResult:
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    _validate_binary_inputs(
        features,
        labels,
        cv_splits=cv_splits,
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    splitter = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=int(seed),
    )
    oof_prob = np.full(
        len(labels),
        np.nan,
        dtype=float,
    )
    fold_records: list[dict[str, Any]] = []

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(features, labels),
        start=1,
    ):
        estimator = _fitted_estimator(
            model,
            features[train_idx],
            labels[train_idx],
            calibration=calibration,
            calibration_cv=calibration_cv,
        )
        probability = _positive_probability(
            estimator,
            features[test_idx],
        )
        oof_prob[test_idx] = probability
        metrics = scalar_metrics(
            labels[test_idx],
            probability,
            threshold=threshold,
        )
        fold_records.append(
            {
                "fold": fold_number,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_malignant": int(labels[train_idx].sum()),
                "n_test_malignant": int(labels[test_idx].sum()),
                "metrics": metrics,
            }
        )

    if np.isnan(oof_prob).any():
        raise RuntimeError("Cross-validation did not generate every OOF prediction.")

    oof_pred = (oof_prob >= threshold).astype(int)
    pooled_metrics = scalar_metrics(
        labels,
        oof_prob,
        threshold=threshold,
    )
    pooled_confusion = confusion_matrix(
        labels,
        oof_pred,
        labels=[0, 1],
    )

    summary = {
        "evaluation": "stratified out-of-fold cross-validation",
        "positive_class": "malignant",
        "negative_class": "benign",
        "threshold": float(threshold),
        "cv_splits": int(cv_splits),
        "seed": int(seed),
        "calibration": calibration,
        "calibration_cv": (
            int(calibration_cv)
            if calibration != "none"
            else None
        ),
        "folds": fold_records,
        "fold_metric_summary": _fold_summary(fold_records),
        "oof_metrics": pooled_metrics,
        "oof_confusion_matrix": {
            "labels": ["benign", "malignant"],
            "matrix": pooled_confusion.tolist(),
        },
        "bootstrap": bootstrap_confidence_intervals(
            labels,
            oof_prob,
            threshold=threshold,
            reps=bootstrap_reps,
            seed=int(seed) + 10_000,
        ),
    }

    return EvaluationResult(
        summary=summary,
        y_true=labels.copy(),
        y_prob=oof_prob,
        y_pred=oof_pred,
    )
