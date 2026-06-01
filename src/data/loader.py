"""Load Home Credit Default Risk training data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import APPLICATION_TRAIN_PATH

logger = logging.getLogger(__name__)


def _resolve_path(file_path: Path | str | None) -> Path:
    """Resolve the CSV path, using the project default when none is given."""
    if file_path is None:
        return APPLICATION_TRAIN_PATH
    return Path(file_path).expanduser().resolve()


def load_application_train(
    file_path: Path | str | None = None,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load the Home Credit application_train.csv dataset.

    Parameters
    ----------
    file_path:
        Path to the CSV file. Defaults to ``data/application_train.csv``
        relative to the project root.
    **read_csv_kwargs:
        Additional keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Training applications with TARGET and feature columns.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist or is not a file.
    """
    path = _resolve_path(file_path)

    if not path.exists():
        logger.error("Training data file not found: %s", path)
        raise FileNotFoundError(f"Training data file not found: {path}")

    if not path.is_file():
        logger.error("Path is not a file: %s", path)
        raise FileNotFoundError(f"Path is not a file: {path}")

    logger.info("Loading Home Credit training data from %s", path)

    try:
        df = pd.read_csv(path, **read_csv_kwargs)
    except Exception:
        logger.exception("Failed to read training data from %s", path)
        raise

    logger.info(
        "Loaded training data: %d rows, %d columns",
        len(df),
        len(df.columns),
    )
    return df
