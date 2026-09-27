import json
import sys

import pandas as pd

from breast_cancer_classifier.cli import main


def test_cli_writes_oof_artifacts(tmp_path, monkeypatch):
    outdir = tmp_path / "artifacts"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "breast-cancer-baseline",
            "--model",
            "logreg",
            "--cv",
            "3",
            "--bootstrap-reps",
            "20",
            "--outdir",
            str(outdir),
        ],
    )

    assert main() == 0

    expected = {
        "metrics.json",
        "oof_predictions.csv",
        "confusion_matrix.csv",
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "feature_coefficients.csv",
        "best_model_logreg.joblib",
        "run_metadata.json",
    }
    assert expected.issubset(
        {path.name for path in outdir.iterdir()}
    )

    metrics = json.loads(
        (outdir / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (outdir / "run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    oof = pd.read_csv(
        outdir / "oof_predictions.csv"
    )

    assert metrics["positive_class"] == "malignant"
    assert metrics["negative_class"] == "benign"
    assert len(oof) == 569
    assert metadata["dataset"]["n_samples"] == 569
    assert metadata["dataset"]["sha256"]


def test_all_models_use_separate_output_directories(tmp_path, monkeypatch):
    outdir = tmp_path / "comparison"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "breast-cancer-baseline",
            "--model",
            "all",
            "--cv",
            "3",
            "--bootstrap-reps",
            "0",
            "--outdir",
            str(outdir),
        ],
    )

    assert main() == 0
    assert (outdir / "logreg" / "metrics.json").exists()
    assert (outdir / "rf" / "metrics.json").exists()
    assert (outdir / "comparison.json").exists()
