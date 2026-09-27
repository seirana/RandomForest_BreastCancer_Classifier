from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from .core import (
    build_model,
    evaluate_cv,
    fit_final_model,
    load_dataset,
)


def _save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _dataset_sha256(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.asarray(X, dtype="<f8").tobytes(order="C")
    )
    digest.update(
        np.asarray(y, dtype=np.int8).tobytes(order="C")
    )
    digest.update("\n".join(feature_names).encode("utf-8"))
    return digest.hexdigest()


def _save_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    outdir: Path,
    *,
    model_name: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        name=model_name,
    )
    plt.title(
        f"ROC Curve — {model_name} (malignant = positive)"
    )
    plt.savefig(
        outdir / "roc_curve.png",
        bbox_inches="tight",
    )
    plt.close()

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob,
        name=model_name,
    )
    plt.title(
        f"Precision–Recall Curve — {model_name} "
        "(malignant = positive)"
    )
    plt.savefig(
        outdir / "pr_curve.png",
        bbox_inches="tight",
    )
    plt.close()

    CalibrationDisplay.from_predictions(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
        name=model_name,
    )
    plt.title(
        f"Calibration Curve — {model_name} "
        "(malignant = positive)"
    )
    plt.savefig(
        outdir / "calibration_curve.png",
        bbox_inches="tight",
    )
    plt.close()


def _save_interpretation(
    model_name: str,
    base_model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    outdir: Path,
) -> None:
    estimator = clone(base_model)
    estimator.fit(X, y)

    if model_name == "logreg":
        classifier = estimator.named_steps["clf"]
        values = classifier.coef_[0]
        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "standardized_coefficient": values,
                "abs_standardized_coefficient": np.abs(values),
            }
        ).sort_values(
            "abs_standardized_coefficient",
            ascending=False,
        )
        frame.to_csv(
            outdir / "feature_coefficients.csv",
            index=False,
        )
        return

    if model_name == "rf":
        values = estimator.feature_importances_
        frame = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_decrease_impurity": values,
            }
        ).sort_values(
            "mean_decrease_impurity",
            ascending=False,
        )
        frame.to_csv(
            outdir / "feature_importances.csv",
            index=False,
        )
        return

    raise ValueError(
        f"Unsupported interpretation model: {model_name}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible breast-cancer baseline evaluation "
            "using sklearn's Wisconsin Diagnostic dataset."
        )
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf", "all"],
        default="logreg",
        help="Model to evaluate. 'all' runs both on identical CV settings.",
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration",
        choices=["none", "sigmoid", "isotonic"],
        default="none",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Backward-compatible alias for --calibration isotonic. "
            "Do not combine with a non-none --calibration value."
        ),
    )
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Classification threshold for malignant probability. "
            "This is fixed before evaluation and is not tuned on test folds."
        ),
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1000,
        help=(
            "Stratified sample bootstrap repetitions for 95%% "
            "confidence intervals on pooled OOF metrics."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for Random Forest.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts"),
    )
    return parser


def _resolve_calibration(args: argparse.Namespace) -> str:
    if args.calibrate:
        if args.calibration != "none":
            raise ValueError(
                "--calibrate cannot be combined with an explicit "
                "non-none --calibration value."
            )
        return "isotonic"
    return str(args.calibration)


def _run_one_model(
    *,
    model_name: str,
    args: argparse.Namespace,
    calibration: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model = build_model(
        model_name,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    evaluation = evaluate_cv(
        base_model,
        X,
        y,
        cv_splits=args.cv,
        seed=args.seed,
        calibration=calibration,
        calibration_cv=args.calibration_cv,
        threshold=args.threshold,
        bootstrap_reps=args.bootstrap_reps,
    )

    metrics = {
        "model": model_name,
        **evaluation.summary,
    }
    _save_json(
        metrics,
        output_dir / "metrics.json",
    )

    oof = pd.DataFrame(
        {
            "sample_index": np.arange(len(y), dtype=int),
            "true_label": evaluation.y_true.astype(int),
            "true_class": np.where(
                evaluation.y_true == 1,
                "malignant",
                "benign",
            ),
            "probability_malignant": evaluation.y_prob,
            "predicted_label": evaluation.y_pred.astype(int),
            "predicted_class": np.where(
                evaluation.y_pred == 1,
                "malignant",
                "benign",
            ),
        }
    )
    oof.to_csv(
        output_dir / "oof_predictions.csv",
        index=False,
    )

    matrix = np.asarray(
        metrics["oof_confusion_matrix"]["matrix"],
        dtype=int,
    )
    pd.DataFrame(
        matrix,
        index=["actual_benign", "actual_malignant"],
        columns=["predicted_benign", "predicted_malignant"],
    ).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    _save_curves(
        evaluation.y_true,
        evaluation.y_prob,
        output_dir,
        model_name=model_name,
    )

    final_model = fit_final_model(
        base_model,
        X,
        y,
        calibration=calibration,
        calibration_cv=args.calibration_cv,
    )
    model_path = (
        output_dir / f"best_model_{model_name}.joblib"
    )
    joblib.dump(final_model, model_path)

    _save_interpretation(
        model_name,
        base_model,
        X,
        y,
        feature_names,
        output_dir,
    )

    return {
        "model": model_name,
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "oof_metrics": metrics["oof_metrics"],
        "bootstrap": metrics["bootstrap"],
    }


def main() -> int:
    args = build_parser().parse_args()
    calibration = _resolve_calibration(args)

    if args.cv < 2:
        raise ValueError("--cv must be at least 2")
    if args.calibration_cv < 2:
        raise ValueError("--calibration-cv must be at least 2")
    if args.bootstrap_reps < 0:
        raise ValueError("--bootstrap-reps must be non-negative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")

    dataset = load_dataset()
    model_names = (
        ["logreg", "rf"]
        if args.model == "all"
        else [args.model]
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []

    for model_name in model_names:
        model_outdir = (
            args.outdir / model_name
            if len(model_names) > 1
            else args.outdir
        )
        runs.append(
            _run_one_model(
                model_name=model_name,
                args=args,
                calibration=calibration,
                X=dataset.X,
                y=dataset.y,
                feature_names=dataset.feature_names,
                output_dir=model_outdir,
            )
        )

    metadata = {
        "command": "breast-cancer-baseline",
        "arguments": {
            key: (
                str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        "resolved_calibration": calibration,
        "dataset": {
            "source": dataset.source_name,
            "n_samples": int(dataset.X.shape[0]),
            "n_features": int(dataset.X.shape[1]),
            "positive_class": dataset.positive_class,
            "negative_class": dataset.negative_class,
            "sha256": _dataset_sha256(
                dataset.X,
                dataset.y,
                dataset.feature_names,
            ),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "git_commit": _git_commit(),
        "runs": runs,
    }
    _save_json(
        metadata,
        args.outdir / "run_metadata.json",
    )

    if len(runs) > 1:
        comparison = {
            run["model"]: {
                "oof_metrics": run["oof_metrics"],
                "bootstrap": run["bootstrap"],
            }
            for run in runs
        }
        _save_json(
            comparison,
            args.outdir / "comparison.json",
        )

    print(
        json.dumps(
            {
                "output_dir": str(args.outdir.resolve()),
                "runs": runs,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
