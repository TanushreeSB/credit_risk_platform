"""SQL safety validation for Talk-to-Data queries."""

from __future__ import annotations

import re

FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
)

FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    """Raised when generated SQL fails safety checks."""


def validate_sql(sql: str) -> str:
    """
    Validate that SQL is a single safe SELECT statement.

    Raises
    ------
    SQLValidationError
        If the statement is empty, multi-statement, non-SELECT, or unsafe.
    """
    if not sql or not sql.strip():
        raise SQLValidationError("SQL statement is empty.")

    statements = [part.strip() for part in sql.split(";") if part.strip()]
    if len(statements) != 1:
        raise SQLValidationError("Only a single SQL statement is allowed.")

    normalized = statements[0].strip()
    if not normalized.upper().startswith("SELECT"):
        raise SQLValidationError("Only SELECT statements are allowed.")

    if FORBIDDEN_PATTERN.search(normalized):
        raise SQLValidationError("Unsafe SQL keyword detected.")

    return normalized
