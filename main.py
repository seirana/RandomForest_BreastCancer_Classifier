#!/usr/bin/env python3
"""Backward-compatible entry point for the breast-cancer baseline toolkit."""

from breast_cancer_classifier.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
