"""Inference and scoring for Home Credit Default Risk."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from src.data.preprocessor import HomeCreditPreprocessor
from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH
from src.utils.helpers import predict_default_probability

logger = logging.getLogger(__name__)


def load_model(path: Path | str = MODEL_PATH) -> LGBMClassifier:
    """Load a trained LightGBM model from disk."""
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, LGBMClassifier):
        raise TypeError(f"Expected LGBMClassifier, got {type(model).__name__}")
    return model


def load_preprocessor(path: Path | str = PREPROCESSOR_PATH) -> HomeCreditPreprocessor:
    """Load a fitted preprocessor from disk."""
    return HomeCreditPreprocessor.load(path)


def score_applicants(
    applicants: pd.DataFrame,
    model: LGBMClassifier | None = None,
    preprocessor: HomeCreditPreprocessor | None = None,
) -> pd.Series:
    """Return default probabilities for one or more applicants."""
    model = model or load_model()
    preprocessor = preprocessor or load_preprocessor()
    logger.info("Scoring %d applicant(s)", len(applicants))
    return predict_default_probability(applicants, model, preprocessor)
