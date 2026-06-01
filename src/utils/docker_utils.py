"""Docker and runtime path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.config import DATA_DIR, DATABASE_PATH, MODEL_PATH, PROJECT_ROOT


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def get_mounted_data_dir() -> Path:
    """Return the data directory (bind-mounted in Docker)."""
    return DATA_DIR.resolve()


def get_database_path() -> Path:
    """Return the SQLite database path used by Talk-to-Data."""
    return DATABASE_PATH.resolve()


def get_model_path() -> Path:
    """Return the LightGBM model artifact path."""
    return Path(os.getenv("MODEL_PATH", MODEL_PATH)).resolve()


def artifacts_available() -> dict[str, bool]:
    """Check whether key runtime artifacts exist."""
    return {
        "model": get_model_path().is_file(),
        "database": get_database_path().is_file(),
        "training_csv": (DATA_DIR / "application_train.csv").is_file(),
    }
