"""Thread-safe SQLite query execution for Talk-to-Data."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from src.talk_to_data.sql_validator import validate_sql
from src.utils.config import DATABASE_PATH

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = DATABASE_PATH


class QueryExecutionError(RuntimeError):
    """Raised when a validated SQL query fails at execution time."""


class QueryRunner:
    """
    Execute read-only SQL against a file-backed SQLite database.

    Opens a fresh connection for each query so callers remain thread-safe
    (required for Streamlit).
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path).expanduser().resolve()

    def ensure_database(
        self,
        dataframe: pd.DataFrame,
        table_name: str,
        force_refresh: bool = False,
    ) -> None:
        """Materialize the dataset into SQLite when the database file is missing."""
        if self.db_path.is_file() and not force_refresh:
            logger.debug("SQLite database already exists at %s", self.db_path)
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Building SQLite database at %s", self.db_path)

        with sqlite3.connect(self.db_path, check_same_thread=False) as connection:
            dataframe.to_sql(table_name, connection, index=False, if_exists="replace")

        logger.info(
            "SQLite database ready with %d rows in table '%s'",
            len(dataframe),
            table_name,
        )

    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a validated SELECT query using a new SQLite connection."""
        safe_sql = validate_sql(sql)
        logger.info("Executing SQL...")

        if not self.db_path.is_file():
            raise QueryExecutionError(
                f"SQLite database not found at {self.db_path}. "
                "Initialize the database before executing queries.",
            )

        try:
            with sqlite3.connect(self.db_path, check_same_thread=False) as connection:
                result = pd.read_sql_query(safe_sql, connection)
        except sqlite3.Error as error:
            logger.exception("SQLite execution failed")
            raise QueryExecutionError(f"Query execution failed: {error}") from error
        except Exception as error:
            logger.exception("Unexpected query execution failure")
            raise QueryExecutionError(f"Query execution failed: {error}") from error

        logger.info("Rows returned: %d", len(result))
        return result

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """Read table schema via a short-lived SQLite connection."""
        if not self.db_path.is_file():
            raise QueryExecutionError(
                f"SQLite database not found at {self.db_path}. "
                "Initialize the database before reading schema.",
            )

        columns: list[dict[str, str]] = []
        with sqlite3.connect(self.db_path, check_same_thread=False) as connection:
            for row in connection.execute(f"PRAGMA table_info({table_name})"):
                columns.append({"name": str(row[1]), "type": str(row[2])})

        logger.info(
            "Loaded schema for table '%s' with %d columns",
            table_name,
            len(columns),
        )
        return {
            "tables": [table_name],
            "columns": {table_name: columns},
        }
