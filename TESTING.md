# Quality checks

The automated test suite focuses on evaluation semantics and reproducibility rather than asserting one exact benchmark score.

Key regression checks include:

1. **Target semantics** — the 212 malignant cases are encoded as positive.
2. **Metric orientation** — sensitivity and specificity use malignant-positive semantics.
3. **OOF completeness** — every sample receives one finite held-out probability.
4. **Determinism** — the same model, seed, and split settings reproduce the same probabilities.
5. **Calibration isolation** — calibrated evaluation runs inside the outer cross-validation structure.
6. **Bootstrap determinism** — a fixed bootstrap seed reproduces the same intervals.
7. **Artifact contract** — CLI runs create metrics, OOF predictions, plots, model files, and provenance metadata.
8. **Comparison mode** — Logistic Regression and Random Forest outputs are kept separate.

The suite intentionally does not require a fixed ROC-AUC or accuracy value because dependency-version changes can cause small numerical differences without invalidating the methodology.
