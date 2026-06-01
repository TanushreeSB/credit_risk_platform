"""Train a LightGBM classifier for Home Credit Default Risk."""

from __future__ import annotations

import logging

import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.data.loader import load_application_train
from src.data.preprocessor import HomeCreditPreprocessor
from src.ml.evaluate import TrainingMetrics, calculate_metrics, save_metrics
from src.utils.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def save_model(model: LGBMClassifier, path: str | None = None) -> None:
    """Persist the trained model to disk."""
    output_path = MODEL_PATH if path is None else path
    output_path = str(output_path)
    logger.info("Saving LightGBM model to %s", output_path)
    joblib.dump(model, output_path)


def train_model(
    data_path: str | None = None,
) -> tuple[LGBMClassifier, HomeCreditPreprocessor, TrainingMetrics]:
    """Load data, preprocess features, train LightGBM, and evaluate on a hold-out set."""
    df = load_application_train(data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training data.")

    y = df[TARGET_COLUMN]
    train_df, test_df, y_train, y_test = train_test_split(
        df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    logger.info("Train rows: %d, test rows: %d", len(train_df), len(test_df))

    preprocessor = HomeCreditPreprocessor()
    preprocessor.fit(train_df)
    X_train = preprocessor.transform(train_df)
    X_test = preprocessor.transform(test_df)

    model = LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    save_model(model)
    preprocessor.save(PREPROCESSOR_PATH)
    save_metrics(metrics)

    return model, preprocessor, metrics


def main() -> None:
    """Entry point for training the Home Credit LightGBM model."""
    setup_logging()
    train_model()


if __name__ == "__main__":
    main()
