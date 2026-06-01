"""Backward-compatible import path; use notebooks.eda instead."""

from notebooks.eda import CHARTS_DIR, generate_eda_charts

__all__ = ["CHARTS_DIR", "generate_eda_charts"]
