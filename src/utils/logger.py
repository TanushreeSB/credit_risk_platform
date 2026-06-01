"""Logging setup for the credit risk platform."""

from __future__ import annotations

import logging

from src.utils.config import LOG_LEVEL


def setup_logging() -> None:
    """Configure root logging once for CLI and services."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
