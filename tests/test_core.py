import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from breast_cancer_classifier.core import (
    bootstrap_confidence_intervals,
    build_model,
    evaluate_cv,
    load_dataset,
    scalar_metrics,
)


def test_dataset_uses_malignant_as_positive_class():
    dataset = load_dataset()

    assert dataset.positive_class == "malignant"
    assert dataset.negative_class == "benign"
    assert dataset.X.shape == (569, 30)
    assert int(dataset.y.sum()) == 212
    assert int((dataset.y == 0).sum()) == 357


def test_scalar_metrics_have_expected_medical_orientation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.9, 0.7])

    metrics = scalar_metrics(
        y_true,
        y_prob,
        threshold=0.5,
    )

    assert metrics["sensitivity_recall"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(2 / 3)


def test_out_of_fold_evaluation_covers_every_sample_once():
    dataset = load_dataset()
    model = LogisticRegression(
        max_iter=3000,
        random_state=7,
    )

    result = evaluate_cv(
        model,
        dataset.X,
        dataset.y,
        cv_splits=4,
        seed=7,
        bootstrap_reps=20,
    )

    assert len(result.y_true) == len(dataset.y)
    assert len(result.y_prob) == len(dataset.y)
    assert np.isfinite(result.y_prob).all()
    assert ((result.y_prob >= 0) & (result.y_prob <= 1)).all()
    assert len(result.summary["folds"]) == 4
    assert (
        sum(fold["n_test"] for fold in result.summary["folds"])
        == len(dataset.y)
    )


def test_cv_evaluation_is_reproducible_for_same_seed():
    dataset = load_dataset()
    model = build_model(
        "logreg",
        seed=11,
    )

    first = evaluate_cv(
        model,
        dataset.X,
        dataset.y,
        cv_splits=3,
        seed=11,
        bootstrap_reps=10,
    )
    second = evaluate_cv(
        model,
        dataset.X,
        dataset.y,
        cv_splits=3,
        seed=11,
        bootstrap_reps=10,
    )

    np.testing.assert_allclose(
        first.y_prob,
        second.y_prob,
    )
    assert (
        first.summary["oof_metrics"]
        == second.summary["oof_metrics"]
    )


def test_sigmoid_calibration_runs_inside_outer_cv():
    dataset = load_dataset()
    model = build_model(
        "logreg",
        seed=3,
    )

    result = evaluate_cv(
        model,
        dataset.X,
        dataset.y,
        cv_splits=3,
        seed=3,
        calibration="sigmoid",
        calibration_cv=3,
        bootstrap_reps=0,
    )

    assert result.summary["calibration"] == "sigmoid"
    assert result.summary["calibration_cv"] == 3
    assert result.summary["bootstrap"]["reps_requested"] == 0


def test_bootstrap_intervals_are_reproducible():
    y_true = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1]
    )
    y_prob = np.array(
        [0.1, 0.2, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9]
    )

    first = bootstrap_confidence_intervals(
        y_true,
        y_prob,
        reps=30,
        seed=5,
    )
    second = bootstrap_confidence_intervals(
        y_true,
        y_prob,
        reps=30,
        seed=5,
    )

    assert first == second
    assert first["reps_valid"] == 30
    assert "roc_auc" in first["ci95"]


def test_invalid_cv_is_rejected():
    dataset = load_dataset()

    with pytest.raises(ValueError, match="cv_splits"):
        evaluate_cv(
            build_model("logreg", seed=1),
            dataset.X,
            dataset.y,
            cv_splits=1,
            bootstrap_reps=0,
        )
