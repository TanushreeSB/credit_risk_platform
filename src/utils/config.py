"""Central configuration and project paths."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ASSETS_DIR = PROJECT_ROOT / "assets"
SQL_DIR = PROJECT_ROOT / "sql"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

APPLICATION_TRAIN_PATH = DATA_DIR / "application_train.csv"
APPLICATION_TEST_PATH = DATA_DIR / "application_test.csv"
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", DATA_DIR / "home_credit.db"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "lightgbm.pkl"))
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RULES_PATH = DOCUMENTS_DIR / "business_rules.txt"
CHARTS_DIR = ASSETS_DIR / "charts"
SHAP_DIR = ASSETS_DIR / "shap"

TARGET_COLUMN = "TARGET"
RANDOM_STATE = 42
TEST_SIZE = 0.2
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
