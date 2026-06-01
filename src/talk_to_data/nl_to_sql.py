"""Schema-aware natural language to SQL for Home Credit analytics."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_application_train
from src.talk_to_data.prompt_templates import (
    EMPTY_QUESTION_MESSAGE,
    UNSUPPORTED_QUESTION_MESSAGE,
)
from src.talk_to_data.query_response import QueryResponse
from src.talk_to_data.query_runner import DEFAULT_DB_PATH, QueryExecutionError, QueryRunner
from src.talk_to_data.sql_validator import SQLValidationError, validate_sql

logger = logging.getLogger(__name__)

TABLE_NAME = "application_train"

FRIENDLY_METRIC_NAMES: dict[str, str] = {
    "AMT_INCOME_TOTAL": "income",
    "AMT_CREDIT": "credit amount",
    "AMT_ANNUITY": "annuity",
    "AMT_GOODS_PRICE": "goods price",
}

METRIC_COLUMNS: dict[str, str] = {
    "income": "AMT_INCOME_TOTAL",
    "credit": "AMT_CREDIT",
    "annuity": "AMT_ANNUITY",
    "goods price": "AMT_GOODS_PRICE",
    "age": "DAYS_BIRTH",
}

GROUP_COLUMNS: dict[str, str] = {
    "gender": "CODE_GENDER",
    "male": "CODE_GENDER",
    "female": "CODE_GENDER",
    "occupation": "OCCUPATION_TYPE",
    "education": "NAME_EDUCATION_TYPE",
    "contract": "NAME_CONTRACT_TYPE",
    "housing": "NAME_HOUSING_TYPE",
    "family": "NAME_FAMILY_STATUS",
    "default status": "TARGET",
    "target": "TARGET",
}

COUNT_KEYWORDS = (
    "how many",
    "count",
    "number of",
    "total customers",
    "total applicants",
    "customer count",
    "customers in the dataset",
    "customers are in",
)

DEFAULT_RATE_KEYWORDS = (
    "default rate",
    "percentage defaulted",
    "percent default",
    "percentage of customers defaulted",
    "what percent",
    "what percentage",
)

AVG_KEYWORDS = ("average", "mean", "avg")
TOP_KEYWORDS = ("highest", "top", "most", "largest", "maximum")
SHOW_KEYWORDS = ("show", "list", "display")


@dataclass
class ResolvedIntent:
    """Structured representation of a parsed analytical question."""

    name: str
    aggregation: str | None = None
    metric_column: str | None = None
    filter_target: int | None = None
    group_column: str | None = None
    order_desc: bool = True
    limit: int | None = None
    compute_default_rate: bool = False


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _extract_limit(text: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\blimit\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    if "top 10" in text or "top ten" in text:
        return 10
    return None


def _extract_metric_column(text: str, schema_columns: set[str]) -> str | None:
    for alias, column in METRIC_COLUMNS.items():
        if alias in text and column in schema_columns:
            return column
    for column in schema_columns:
        readable = column.lower().replace("_", " ")
        if readable in text or column.lower() in text:
            return column
    return None


def _extract_group_column(text: str, schema_columns: set[str]) -> str | None:
    for alias, column in GROUP_COLUMNS.items():
        if alias in text and column in schema_columns:
            return column
    if "by default status" in text or "default status" in text:
        return "TARGET" if "TARGET" in schema_columns else None
    return None


def _mentions_defaulters(text: str) -> bool:
    return any(
        token in text
        for token in ("defaulter", "defaulters", "defaulted", "target = 1", "who default")
    )


def _mentions_non_defaulters(text: str) -> bool:
    return any(
        token in text
        for token in ("non-defaulter", "non defaulter", "non-default", "did not default", "target = 0")
    )


def _resolve_intent(question: str, schema_columns: set[str]) -> ResolvedIntent | None:
    """Map a normalized question to a structured analytical intent."""
    text = _normalize_question(question)
    limit = _extract_limit(text)
    metric = _extract_metric_column(text, schema_columns)
    group = _extract_group_column(text, schema_columns)

    if _contains_any(text, DEFAULT_RATE_KEYWORDS):
        if group == "CODE_GENDER" or _contains_any(text, ("male", "female", "gender", "males or females")):
            return ResolvedIntent(
                name="gender_default_rate",
                group_column="CODE_GENDER",
                compute_default_rate=True,
                order_desc=True,
            )
        if group == "OCCUPATION_TYPE" or "occupation" in text:
            return ResolvedIntent(
                name="occupation_default_rate",
                group_column="OCCUPATION_TYPE",
                compute_default_rate=True,
                order_desc=True,
                limit=limit or 10,
            )
        return ResolvedIntent(name="default_rate", compute_default_rate=True)

    if _contains_any(text, COUNT_KEYWORDS):
        if _mentions_defaulters(text) or ("default" in text and "how many" in text):
            return ResolvedIntent(name="count_defaulters", aggregation="COUNT")
        if any(token in text for token in ("customer", "customers", "applicant", "applicants", "dataset")):
            return ResolvedIntent(name="count_total", aggregation="COUNT")

    if _contains_any(text, AVG_KEYWORDS) and metric:
        filter_target: int | None = None
        if _mentions_non_defaulters(text):
            filter_target = 0
        elif _mentions_defaulters(text):
            filter_target = 1

        if metric == "AMT_ANNUITY" and ("default status" in text or group == "TARGET"):
            return ResolvedIntent(
                name="avg_annuity_by_default",
                aggregation="AVG",
                metric_column="AMT_ANNUITY",
                group_column="TARGET",
            )

        if group == "OCCUPATION_TYPE" or ("occupation" in text and metric == "AMT_INCOME_TOTAL"):
            return ResolvedIntent(
                name="occupation_avg_metric",
                aggregation="AVG",
                metric_column=metric,
                group_column="OCCUPATION_TYPE",
                order_desc=True,
                limit=limit or (10 if _contains_any(text, TOP_KEYWORDS + SHOW_KEYWORDS) else None),
            )

        if group == "CODE_GENDER" and metric == "AMT_CREDIT":
            return ResolvedIntent(
                name="avg_credit_by_gender",
                aggregation="AVG",
                metric_column="AMT_CREDIT",
                group_column="CODE_GENDER",
            )

        return ResolvedIntent(
            name="avg_metric",
            aggregation="AVG",
            metric_column=metric,
            filter_target=filter_target,
        )

    if group == "OCCUPATION_TYPE" and _contains_any(text, TOP_KEYWORDS + SHOW_KEYWORDS):
        metric_column = metric or "AMT_INCOME_TOTAL"
        if "default" in text:
            return ResolvedIntent(
                name="occupation_default_rate",
                group_column="OCCUPATION_TYPE",
                compute_default_rate=True,
                order_desc=True,
                limit=limit or 10,
            )
        return ResolvedIntent(
            name="occupation_avg_metric",
            aggregation="AVG",
            metric_column=metric_column,
            group_column="OCCUPATION_TYPE",
            order_desc=True,
            limit=limit or 10,
        )

    return None


def _build_where_clause(filter_target: int | None) -> str:
    if filter_target is None:
        return ""
    return f"WHERE TARGET = {filter_target}"


def _build_sql(intent: ResolvedIntent, table: str = TABLE_NAME) -> str:
    """Generate deterministic SQL from a resolved intent."""
    if intent.name == "count_total":
        return f"SELECT COUNT(*) AS total_customers FROM {table}"

    if intent.name == "count_defaulters":
        return (
            f"SELECT COUNT(*) AS defaulted_customers "
            f"FROM {table} WHERE TARGET = 1"
        )

    if intent.name == "default_rate":
        return f"""
SELECT ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) AS default_rate
FROM {table}
""".strip()

    if intent.name == "avg_metric" and intent.metric_column:
        alias = intent.metric_column.lower()
        where = _build_where_clause(intent.filter_target)
        return f"""
SELECT ROUND(AVG({intent.metric_column}), 2) AS avg_{alias}
FROM {table}
{where}
""".strip()

    if intent.name == "gender_default_rate":
        return f"""
SELECT
    CODE_GENDER,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate
FROM {table}
GROUP BY CODE_GENDER
ORDER BY default_rate DESC
""".strip()

    if intent.name == "occupation_default_rate":
        limit = intent.limit or 10
        return f"""
SELECT
    OCCUPATION_TYPE,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate
FROM {table}
WHERE OCCUPATION_TYPE IS NOT NULL
GROUP BY OCCUPATION_TYPE
ORDER BY default_rate DESC
LIMIT {limit}
""".strip()

    if intent.name == "occupation_avg_metric" and intent.metric_column:
        limit_sql = f"LIMIT {intent.limit}" if intent.limit else ""
        alias = intent.metric_column.lower()
        order = "DESC" if intent.order_desc else "ASC"
        return f"""
SELECT
    OCCUPATION_TYPE,
    ROUND(AVG({intent.metric_column}), 2) AS avg_{alias}
FROM {table}
WHERE OCCUPATION_TYPE IS NOT NULL
GROUP BY OCCUPATION_TYPE
ORDER BY avg_{alias} {order}
{limit_sql}
""".strip()

    if intent.name == "avg_annuity_by_default":
        return f"""
SELECT
    TARGET,
    ROUND(AVG(AMT_ANNUITY), 2) AS avg_annuity
FROM {table}
GROUP BY TARGET
""".strip()

    if intent.name == "avg_credit_by_gender":
        return f"""
SELECT
    CODE_GENDER,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit
FROM {table}
GROUP BY CODE_GENDER
""".strip()

    raise SQLValidationError(f"Unable to build SQL for intent '{intent.name}'.")


class NLToSQLConverter:
    """
    Convert business questions into SQL using schema-aware semantic intent matching.

    Uses a file-backed SQLite database with per-query connections for Streamlit
    thread safety.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        db_path: Path | str = DEFAULT_DB_PATH,
    ) -> None:
        self._dataframe = dataframe
        self._query_runner = QueryRunner(db_path=db_path)
        self._schema: dict[str, Any] | None = None
        self._last_intent: ResolvedIntent | None = None
        self._initialize_database()

    def _get_dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            logger.info("Loading application_train.csv for NL-to-SQL queries")
            self._dataframe = load_application_train()
        return self._dataframe

    def _initialize_database(self) -> None:
        """Ensure the SQLite database file exists before queries run."""
        self._query_runner.ensure_database(self._get_dataframe(), TABLE_NAME)

    def get_schema(self) -> dict[str, Any]:
        """Read SQLite schema dynamically from the loaded dataset."""
        if self._schema is None:
            self._schema = self._query_runner.get_schema(TABLE_NAME)
        return self._schema

    def _resolve_sql_and_intent(self, question: str) -> tuple[str, ResolvedIntent]:
        """Resolve intent and build validated SQL for a question."""
        if not question.strip():
            raise ValueError(EMPTY_QUESTION_MESSAGE)

        schema = self.get_schema()
        schema_columns = {column["name"] for column in schema["columns"][TABLE_NAME]}
        intent = _resolve_intent(question, schema_columns)
        if intent is None:
            raise SQLValidationError(UNSUPPORTED_QUESTION_MESSAGE)

        sql = validate_sql(_build_sql(intent))
        return sql, intent

    def generate_sql(self, question: str) -> str:
        """Convert a natural language question into a validated SELECT query."""
        sql, intent = self._resolve_sql_and_intent(question)
        self._last_intent = intent
        return sql

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute validated SQL and return a pandas DataFrame."""
        return self._query_runner.execute(sql)

    def generate_business_summary(
        self,
        question: str,
        result_df: pd.DataFrame,
        intent: ResolvedIntent,
    ) -> str:
        """Generate a plain-language summary from the question and query result."""
        if result_df.empty:
            return "The query completed successfully but returned no rows."

        if intent.name == "count_total":
            return f"The dataset contains {int(result_df.iloc[0, 0]):,} customers."

        if intent.name == "count_defaulters":
            return f"{int(result_df.iloc[0, 0]):,} customers defaulted."

        if intent.name == "default_rate":
            return f"The overall default rate is {float(result_df.iloc[0, 0]):.2f}%."

        if intent.name == "avg_metric" and intent.metric_column:
            value = float(result_df.iloc[0, 0])
            label = FRIENDLY_METRIC_NAMES.get(
                intent.metric_column,
                intent.metric_column.replace("_", " ").lower(),
            )
            if intent.filter_target == 1:
                return f"The average {label} of defaulters is {value:,.2f}."
            if intent.filter_target == 0:
                return f"The average {label} for non-defaulters is {value:,.2f}."
            return f"The average {label} is {value:,.2f}."

        if intent.name == "gender_default_rate":
            top = result_df.iloc[0]
            gender = "males" if top["CODE_GENDER"] == "M" else "females"
            return (
                f"{gender.title()} show the higher default rate at "
                f"{float(top['default_rate']):.2f}%."
            )

        if intent.name == "occupation_default_rate":
            occupations = ", ".join(result_df["OCCUPATION_TYPE"].head(3).astype(str))
            return f"{occupations} show the highest observed default rates."

        if intent.name == "occupation_avg_metric":
            top = result_df.iloc[0]
            value_columns = [column for column in result_df.columns if column != "OCCUPATION_TYPE"]
            metric_value = float(top[value_columns[0]])
            metric_label = FRIENDLY_METRIC_NAMES.get(
                intent.metric_column or "",
                intent.metric_column.replace("_", " ").lower() if intent.metric_column else "value",
            )
            return (
                f"'{top['OCCUPATION_TYPE']}' leads with the highest average {metric_label} "
                f"at {metric_value:,.2f}."
            )

        if intent.name == "avg_annuity_by_default":
            parts: list[str] = []
            for _, row in result_df.iterrows():
                label = "defaulters" if int(row["TARGET"]) == 1 else "non-defaulters"
                parts.append(f"{label}: {float(row['avg_annuity']):,.2f}")
            return "Average annuity by default status - " + "; ".join(parts) + "."

        if intent.name == "avg_credit_by_gender":
            top = result_df.iloc[0]
            gender = "Females" if top["CODE_GENDER"] == "F" else "Males"
            return f"{gender} have the highest average credit amount at {float(top['avg_credit']):,.2f}."

        if len(result_df) == 1 and len(result_df.columns) == 1:
            return f"The answer is {result_df.iloc[0, 0]}."

        return f"The query returned {len(result_df):,} rows."

    def ask(self, question: str) -> QueryResponse:
        """End-to-end question answering that never raises to the UI layer."""
        generated_sql = ""
        try:
            generated_sql, intent = self._resolve_sql_and_intent(question)
            result_df = self.execute_query(generated_sql)
            business_summary = self.generate_business_summary(question, result_df, intent)
            return QueryResponse(
                question=question,
                generated_sql=generated_sql,
                result_df=result_df,
                business_summary=business_summary,
                success=True,
            )
        except ValueError as error:
            logger.warning("Invalid question: %s", error)
            return QueryResponse.failure(question=question, error_message=str(error))
        except SQLValidationError as error:
            logger.warning("SQL generation failed: %s", error)
            return QueryResponse.failure(question=question, error_message=str(error))
        except QueryExecutionError as error:
            logger.exception("Query execution failed")
            return QueryResponse.failure(
                question=question,
                error_message=str(error),
                generated_sql=generated_sql,
            )
        except Exception as error:
            logger.exception("Unexpected Talk-to-Data failure")
            return QueryResponse.failure(
                question=question,
                error_message=str(error),
                generated_sql=generated_sql,
            )


def ask_data_question(question: str) -> QueryResponse:
    """Convenience wrapper for one-off natural language queries."""
    return NLToSQLConverter().ask(question)
