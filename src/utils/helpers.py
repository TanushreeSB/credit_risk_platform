"""Shared helper utilities for UI and inference."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier

from src.data.preprocessor import HomeCreditPreprocessor


def get_risk_band(probability: float) -> str:
    """Map a default probability to a business risk band."""
    if probability <= 0.30:
        return "Low Risk"
    if probability <= 0.70:
        return "Medium Risk"
    return "High Risk"


def get_risk_band_class(risk_band: str) -> str:
    """Return CSS class for a risk band card."""
    return {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }.get(risk_band, "risk-medium")


def clean_feature_name(feature_name: str) -> str:
    """Convert internal feature names to business-friendly labels."""
    for prefix in ("num__", "cat__"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :].replace("_", " ").title()
    return feature_name.replace("_", " ").title()


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded CSV file with common encoding fallbacks."""
    content = uploaded_file.getvalue()
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode CSV with supported encodings.")


def validate_applicant_dataframe(
    applicants: pd.DataFrame,
    preprocessor: HomeCreditPreprocessor,
) -> list[str]:
    """Return missing feature columns required by the fitted preprocessor."""
    if applicants.empty:
        return ["Dataset contains no rows."]

    if preprocessor.numerical_columns_ is None or preprocessor.categorical_columns_ is None:
        return ["Preprocessor feature columns are unavailable."]

    required_columns = set(preprocessor.numerical_columns_) | set(preprocessor.categorical_columns_)
    return sorted(required_columns - set(applicants.columns))


def build_prediction_results(
    applicants: pd.DataFrame,
    probabilities: pd.Series,
) -> pd.DataFrame:
    """Build a results dataframe with probabilities and risk bands."""
    results = pd.DataFrame(
        {
            "default_probability": probabilities.values,
            "risk_band": [get_risk_band(float(value)) for value in probabilities.values],
        },
        index=applicants.index,
    )
    if "SK_ID_CURR" in applicants.columns:
        results.insert(0, "SK_ID_CURR", applicants["SK_ID_CURR"].values)
    return results.reset_index(drop=True)


def predict_default_probability(
    applicants: pd.DataFrame,
    model: LGBMClassifier,
    preprocessor: HomeCreditPreprocessor,
) -> pd.Series:
    """Score applicants and return default probabilities."""
    features = preprocessor.transform_to_dataframe(applicants)
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=applicants.index, name="default_probability")
