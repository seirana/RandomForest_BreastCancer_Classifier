"""Breast-cancer baseline classification toolkit."""

from .core import (
    DatasetBundle,
    EvaluationResult,
    bootstrap_confidence_intervals,
    build_model,
    evaluate_cv,
    fit_final_model,
    load_dataset,
    scalar_metrics,
)

__all__ = [
    "DatasetBundle",
    "EvaluationResult",
    "bootstrap_confidence_intervals",
    "build_model",
    "evaluate_cv",
    "fit_final_model",
    "load_dataset",
    "scalar_metrics",
]
