"""Sklearn preprocessing pipeline for Home Credit Default Risk features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_COLUMNS: tuple[str, ...] = ("SK_ID_CURR", "TARGET")


def identify_column_types(
    df: pd.DataFrame,
    exclude_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split feature columns into numerical and categorical groups by dtype."""
    excluded = set(exclude_columns or ())
    feature_df = df.drop(columns=list(excluded), errors="ignore")

    numerical_columns = feature_df.select_dtypes(
        include=["number", "bool"],
    ).columns.tolist()
    categorical_columns = feature_df.select_dtypes(
        include=["object", "category", "string"],
    ).columns.tolist()

    unassigned = [
        column
        for column in feature_df.columns
        if column not in numerical_columns and column not in categorical_columns
    ]
    if unassigned:
        logger.warning(
            "Columns with unsupported dtypes excluded from preprocessing: %s",
            unassigned,
        )

    logger.info(
        "Identified %d numerical and %d categorical feature columns",
        len(numerical_columns),
        len(categorical_columns),
    )
    return numerical_columns, categorical_columns


def build_column_transformer(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer with numeric and categorical sub-pipelines."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ],
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ],
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


class HomeCreditPreprocessor:
    """
    Fit/transform wrapper around a sklearn ColumnTransformer for Home Credit data.

    Automatically detects numerical and categorical columns, applies median
    imputation to numeric features and most-frequent imputation plus ordinal
    encoding to categorical features.
    """

    def __init__(
        self,
        exclude_columns: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.exclude_columns: tuple[str, ...] = tuple(
            exclude_columns if exclude_columns is not None else DEFAULT_EXCLUDE_COLUMNS,
        )
        self.numerical_columns_: list[str] | None = None
        self.categorical_columns_: list[str] | None = None
        self.preprocessor_: ColumnTransformer | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True when the underlying transformer has been fit."""
        return self.preprocessor_ is not None

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(X).__name__}")

    def _validate_fitted_columns(self, X: pd.DataFrame) -> None:
        if self.numerical_columns_ is None or self.categorical_columns_ is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit before transform.")

        expected_columns = set(self.numerical_columns_) | set(self.categorical_columns_)
        missing_columns = expected_columns - set(X.columns)
        if missing_columns:
            raise ValueError(
                "Input is missing columns required by the fitted preprocessor: "
                f"{sorted(missing_columns)}",
            )

    def _build_feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.numerical_columns_ is not None
        assert self.categorical_columns_ is not None
        feature_columns = self.numerical_columns_ + self.categorical_columns_
        return X[feature_columns]

    def fit(self, X: pd.DataFrame, y: Any | None = None) -> Self:
        """Learn preprocessing parameters from training data."""
        del y
        self._validate_input(X)

        logger.info("Fitting Home Credit preprocessor on %d rows", len(X))
        self.numerical_columns_, self.categorical_columns_ = identify_column_types(
            X,
            exclude_columns=self.exclude_columns,
        )

        if not self.numerical_columns_ and not self.categorical_columns_:
            raise ValueError("No feature columns found for preprocessing.")

        self.preprocessor_ = build_column_transformer(
            self.numerical_columns_,
            self.categorical_columns_,
        )
        feature_frame = self._build_feature_frame(X)
        self.preprocessor_.fit(feature_frame)

        logger.info(
            "Preprocessor fitted on %d output features",
            self.preprocessor_.transform(feature_frame).shape[1],
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing to raw feature data."""
        self._validate_input(X)
        self._validate_fitted_columns(X)

        assert self.preprocessor_ is not None
        feature_frame = self._build_feature_frame(X)

        logger.info("Transforming %d rows with Home Credit preprocessor", len(X))
        transformed = self.preprocessor_.transform(feature_frame)
        logger.info("Transformed output shape: %s", transformed.shape)
        return transformed

    def transform_to_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing and return a DataFrame with training feature names."""
        transformed = self.transform(X)
        feature_names = self.get_feature_names_out()
        return pd.DataFrame(transformed, columns=feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Any | None = None) -> np.ndarray:
        """Fit the preprocessor and transform the input data."""
        return self.fit(X, y=y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        """Return output feature names from the fitted ColumnTransformer."""
        if not self.is_fitted or self.preprocessor_ is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit before accessing feature names.")
        return list(self.preprocessor_.get_feature_names_out())

    def save(self, path: Path | str) -> None:
        """Persist the fitted preprocessor to disk with joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")

        output_path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving Home Credit preprocessor to %s", output_path)
        joblib.dump(self, output_path)
        logger.info("Preprocessor saved successfully")

    @classmethod
    def load(cls, path: Path | str) -> HomeCreditPreprocessor:
        """Load a fitted preprocessor from disk."""
        input_path = Path(path).expanduser().resolve()

        if not input_path.exists():
            logger.error("Preprocessor file not found: %s", input_path)
            raise FileNotFoundError(f"Preprocessor file not found: {input_path}")

        if not input_path.is_file():
            logger.error("Path is not a file: %s", input_path)
            raise FileNotFoundError(f"Path is not a file: {input_path}")

        logger.info("Loading Home Credit preprocessor from %s", input_path)
        preprocessor = joblib.load(input_path)

        if not isinstance(preprocessor, cls):
            raise TypeError(
                f"Expected loaded object of type {cls.__name__}, "
                f"got {type(preprocessor).__name__}",
            )

        logger.info("Preprocessor loaded successfully")
        return preprocessor
