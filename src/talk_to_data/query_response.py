"""Standardized response object for Talk-to-Data queries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class QueryResponse:
    """Structured response from a natural language data question."""

    question: str
    generated_sql: str
    result_df: pd.DataFrame
    business_summary: str
    success: bool
    error_message: str | None = None

    @classmethod
    def failure(
        cls,
        question: str,
        error_message: str,
        generated_sql: str = "",
    ) -> QueryResponse:
        """Build a failed response that keeps the UI stable."""
        return cls(
            question=question,
            generated_sql=generated_sql,
            result_df=pd.DataFrame(),
            business_summary="Unable to answer this question.",
            success=False,
            error_message=error_message,
        )
