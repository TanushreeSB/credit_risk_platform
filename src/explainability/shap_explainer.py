"""SHAP-based explainability for the Home Credit LightGBM model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from src.data.loader import load_application_train
from src.data.preprocessor import HomeCreditPreprocessor
from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH, SHAP_DIR

logger = logging.getLogger(__name__)

OUTPUT_DIR = SHAP_DIR

SUMMARY_PLOT_PATH = OUTPUT_DIR / "summary_plot.png"
GLOBAL_IMPORTANCE_PLOT_PATH = OUTPUT_DIR / "global_importance.png"
APPLICANT_FORCE_PLOT_PATH = OUTPUT_DIR / "applicant_force_plot.png"


class RiskDriver(TypedDict):
    feature: str
    shap_value: float
    feature_value: float | str | None


class ApplicantExplanation(TypedDict):
    applicant_id: int | None
    base_value: float
    predicted_probability: float
    top_positive_drivers: list[RiskDriver]
    top_negative_drivers: list[RiskDriver]


def load_model(path: Path | str = MODEL_PATH) -> LGBMClassifier:
    """Load a trained LightGBM model from disk."""
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Loading LightGBM model from %s", model_path)
    model = joblib.load(model_path)
    if not isinstance(model, LGBMClassifier):
        raise TypeError(
            f"Expected LGBMClassifier at {model_path}, got {type(model).__name__}",
        )
    return model


def load_preprocessor(
    path: Path | str = PREPROCESSOR_PATH,
) -> HomeCreditPreprocessor:
    """Load a fitted preprocessor from disk."""
    return HomeCreditPreprocessor.load(path)


def _normalize_shap_values(shap_values: Any) -> np.ndarray:
    """Return SHAP values for the positive (default) class."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    return np.asarray(shap_values)


def _normalize_base_value(base_value: Any) -> float:
    """Return the expected value for the positive (default) class."""
    if isinstance(base_value, (list, np.ndarray)):
        return float(base_value[1])
    return float(base_value)


class HomeCreditShapExplainer:
    """Generate global and local SHAP explanations for credit risk predictions."""

    def __init__(
        self,
        model: LGBMClassifier,
        preprocessor: HomeCreditPreprocessor,
        output_dir: Path | str = OUTPUT_DIR,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names_: list[str] | None = None

    @classmethod
    def from_artifacts(
        cls,
        model_path: Path | str = MODEL_PATH,
        preprocessor_path: Path | str = PREPROCESSOR_PATH,
        output_dir: Path | str = OUTPUT_DIR,
    ) -> HomeCreditShapExplainer:
        """Build an explainer from saved model and preprocessor artifacts."""
        model = load_model(model_path)
        preprocessor = load_preprocessor(preprocessor_path)
        return cls(model=model, preprocessor=preprocessor, output_dir=output_dir)

    def _prepare_features(self, raw_df: pd.DataFrame) -> np.ndarray:
        if not self.preprocessor.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before generating SHAP values.")
        return self.preprocessor.transform(raw_df)

    def _get_feature_names(self) -> list[str]:
        if self.feature_names_ is None:
            self.feature_names_ = self.preprocessor.get_feature_names_out()
        return self.feature_names_

    def compute_shap_values(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for preprocessed applicant features."""
        features = self._prepare_features(raw_df)
        logger.info("Computing SHAP values for %d applicants", len(raw_df))
        shap_values = _normalize_shap_values(self.explainer.shap_values(features))
        logger.info("SHAP values shape: %s", shap_values.shape)
        return shap_values

    def global_feature_importance(
        self,
        raw_df: pd.DataFrame,
        top_n: int | None = 20,
    ) -> pd.DataFrame:
        """
        Rank features by mean absolute SHAP value across applicants.

        Returns
        -------
        pd.DataFrame
            Columns: feature, mean_abs_shap (sorted descending).
        """
        shap_values = self.compute_shap_values(raw_df)
        feature_names = self._get_feature_names()

        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            },
        ).sort_values("mean_abs_shap", ascending=False)

        if top_n is not None:
            importance = importance.head(top_n).reset_index(drop=True)

        logger.info("Computed global feature importance for %d features", len(importance))
        return importance

    def plot_summary(
        self,
        raw_df: pd.DataFrame,
        save_path: Path | str = SUMMARY_PLOT_PATH,
        max_display: int = 20,
    ) -> Path:
        """Generate and save a SHAP summary plot."""
        features = self._prepare_features(raw_df)
        shap_values = self.compute_shap_values(raw_df)
        feature_names = self._get_feature_names()

        output_path = Path(save_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving SHAP summary plot to %s", output_path)
        plt.figure()
        shap.summary_plot(
            shap_values,
            features,
            feature_names=feature_names,
            show=False,
            max_display=max_display,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    def plot_global_importance(
        self,
        raw_df: pd.DataFrame,
        save_path: Path | str = GLOBAL_IMPORTANCE_PLOT_PATH,
        top_n: int = 20,
    ) -> Path:
        """Generate and save a bar chart of global SHAP feature importance."""
        importance = self.global_feature_importance(raw_df, top_n=top_n)
        output_path = Path(save_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Saving global importance plot to %s", output_path)
        plt.figure(figsize=(10, 8))
        plt.barh(
            importance["feature"][::-1],
            importance["mean_abs_shap"][::-1],
        )
        plt.xlabel("Mean |SHAP value|")
        plt.title("Global Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    def explain_applicant(
        self,
        applicant: pd.DataFrame | pd.Series,
        top_n: int = 5,
    ) -> ApplicantExplanation:
        """
        Explain a single applicant and return top risk-increasing and risk-decreasing drivers.

        Positive SHAP values increase default risk; negative values decrease it.
        """
        if isinstance(applicant, pd.Series):
            applicant_df = applicant.to_frame().T
        else:
            applicant_df = applicant.copy()

        if len(applicant_df) != 1:
            raise ValueError("explain_applicant expects exactly one applicant row.")

        applicant_id: int | None = None
        if "SK_ID_CURR" in applicant_df.columns:
            applicant_id = int(applicant_df["SK_ID_CURR"].iloc[0])

        feature_frame = self.preprocessor.transform_to_dataframe(applicant_df)
        shap_values = self.compute_shap_values(applicant_df)
        feature_names = self._get_feature_names()

        row_shap = shap_values[0]
        predicted_probability = float(self.model.predict_proba(feature_frame)[0, 1])
        base_value = _normalize_base_value(self.explainer.expected_value)

        drivers = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_value": row_shap,
                "feature_value": feature_frame.iloc[0].to_numpy(),
            },
        )

        positive_drivers = (
            drivers[drivers["shap_value"] > 0]
            .sort_values("shap_value", ascending=False)
            .head(top_n)
        )
        negative_drivers = (
            drivers[drivers["shap_value"] < 0]
            .sort_values("shap_value", ascending=True)
            .head(top_n)
        )

        explanation: ApplicantExplanation = {
            "applicant_id": applicant_id,
            "base_value": base_value,
            "predicted_probability": predicted_probability,
            "top_positive_drivers": [
                {
                    "feature": row.feature,
                    "shap_value": float(row.shap_value),
                    "feature_value": _format_feature_value(row.feature_value),
                }
                for row in positive_drivers.itertuples(index=False)
            ],
            "top_negative_drivers": [
                {
                    "feature": row.feature,
                    "shap_value": float(row.shap_value),
                    "feature_value": _format_feature_value(row.feature_value),
                }
                for row in negative_drivers.itertuples(index=False)
            ],
        }

        logger.info(
            "Explained applicant %s (predicted probability=%.4f)",
            applicant_id if applicant_id is not None else "unknown",
            predicted_probability,
        )
        return explanation

    def plot_applicant_force(
        self,
        applicant: pd.DataFrame | pd.Series,
        save_path: Path | str = APPLICANT_FORCE_PLOT_PATH,
        top_n: int = 15,
    ) -> Path:
        """Generate and save a SHAP force plot for a single applicant."""
        if isinstance(applicant, pd.Series):
            applicant_df = applicant.to_frame().T
        else:
            applicant_df = applicant.copy()

        features = self._prepare_features(applicant_df)
        shap_values = self.compute_shap_values(applicant_df)
        feature_names = self._get_feature_names()
        base_value = _normalize_base_value(self.explainer.expected_value)

        output_path = Path(save_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        force_plot = shap.force_plot(
            base_value,
            shap_values[0],
            features[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
        logger.info("Saving applicant force plot to %s", output_path)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(force_plot)
        return output_path

    def generate_explanations(
        self,
        raw_df: pd.DataFrame,
        applicant_index: int = 0,
        sample_size: int | None = 1000,
        top_n: int = 5,
    ) -> tuple[pd.DataFrame, ApplicantExplanation]:
        """
        Generate global importance, summary plot, and a single-applicant explanation.

        Returns
        -------
        tuple[pd.DataFrame, ApplicantExplanation]
            Global feature importance table and local applicant explanation.
        """
        if sample_size is not None and len(raw_df) > sample_size:
            logger.info("Sampling %d rows for SHAP analysis", sample_size)
            analysis_df = raw_df.sample(n=sample_size, random_state=42)
        else:
            analysis_df = raw_df

        importance = self.global_feature_importance(analysis_df)
        self.plot_summary(analysis_df)
        self.plot_global_importance(analysis_df)

        applicant = raw_df.iloc[[applicant_index]]
        explanation = self.explain_applicant(applicant, top_n=top_n)
        self.plot_applicant_force(applicant)

        importance_path = self.output_dir / "global_importance.csv"
        importance.to_csv(importance_path, index=False)
        logger.info("Saved global importance table to %s", importance_path)

        return importance, explanation


def _format_feature_value(value: Any) -> float | str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return float(value)
    return str(value)


def main() -> None:
    """Generate SHAP explanations and save plots to assets/shap."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    df = load_application_train()
    explainer = HomeCreditShapExplainer.from_artifacts()
    importance, explanation = explainer.generate_explanations(df)

    logger.info("Top global features:\n%s", importance.head(10).to_string(index=False))
    logger.info(
        "Applicant %s top positive drivers: %s",
        explanation["applicant_id"],
        explanation["top_positive_drivers"],
    )
    logger.info(
        "Applicant %s top negative drivers: %s",
        explanation["applicant_id"],
        explanation["top_negative_drivers"],
    )


if __name__ == "__main__":
    main()
