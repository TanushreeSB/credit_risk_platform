"""Model evaluation metrics for Home Credit Default Risk."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import METRICS_PATH

logger = logging.getLogger(__name__)


class TrainingMetrics(TypedDict):
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> TrainingMetrics:
    """Compute classification metrics on held-out predictions."""
    return TrainingMetrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def save_metrics(metrics: TrainingMetrics, path: Path | str = METRICS_PATH) -> None:
    """Persist evaluation metrics to a JSON file."""
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving training metrics to %s", output_path)
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def load_metrics(path: Path | str = METRICS_PATH) -> dict[str, float] | None:
    """Load metrics JSON if available."""
    metrics_path = Path(path)
    if not metrics_path.is_file():
        return None
    with metrics_path.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)
