"""Versioned prompt templates for Talk-to-Data (NL-to-SQL)."""

from __future__ import annotations

PROMPT_VERSION = "1.0.0"

UNSUPPORTED_QUESTION_MESSAGE = (
    "Question not currently supported. Try asking about counts, default rate, "
    "average income/credit/annuity, gender, or occupation."
)

EMPTY_QUESTION_MESSAGE = "Question cannot be empty."
