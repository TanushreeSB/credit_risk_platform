"""FastAPI backend for the Credit Risk Intelligence Platform."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from lightgbm import LGBMClassifier
from pydantic import BaseModel, Field

from src.data.preprocessor import HomeCreditPreprocessor
from src.utils.config import DATABASE_PATH, METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Intelligence Platform API",
    description="REST API for model scoring, metrics, and health checks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: LGBMClassifier | None = None
_preprocessor: HomeCreditPreprocessor | None = None


class PredictRequest(BaseModel):
    """Payload for batch applicant scoring."""

    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    """Scoring response for one or more applicants."""

    predictions: list[dict[str, Any]]


def _load_artifacts() -> tuple[LGBMClassifier | None, HomeCreditPreprocessor | None]:
    """Load model artifacts once and cache them in process memory."""
    global _model, _preprocessor

    if _model is None and MODEL_PATH.is_file():
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, LGBMClassifier):
            _model = loaded

    if _preprocessor is None and PREPROCESSOR_PATH.is_file():
        _preprocessor = HomeCreditPreprocessor.load(PREPROCESSOR_PATH)

    return _model, _preprocessor


def _load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.is_file():
        return {}
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health probe used by Docker and orchestrators."""
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Return persisted model evaluation metrics."""
    metrics = _load_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics file not found.")
    return metrics


@app.get("/info")
def platform_info() -> dict[str, Any]:
    """Return platform metadata and artifact availability."""
    model, preprocessor = _load_artifacts()
    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": str(MODEL_PATH),
        "database_path": str(DATABASE_PATH),
        "database_exists": DATABASE_PATH.is_file(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score applicant records and return default probabilities."""
    model, preprocessor = _load_artifacts()
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not available.")

    applicants = pd.DataFrame(request.records)
    features = preprocessor.transform_to_dataframe(applicants)
    probabilities = model.predict_proba(features)[:, 1]

    predictions: list[dict[str, Any]] = []
    for index, probability in enumerate(probabilities):
        risk_band = "Low Risk" if probability <= 0.30 else "Medium Risk" if probability <= 0.70 else "High Risk"
        predictions.append(
            {
                "index": index,
                "default_probability": float(probability),
                "risk_band": risk_band,
            },
        )

    return PredictResponse(predictions=predictions)


@app.on_event("startup")
def startup_event() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    logger.info("Starting Credit Risk Platform API")
    _load_artifacts()
