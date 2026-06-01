"""Streamlit UI for the AI-Powered Credit Risk Intelligence Platform."""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_application_train
from src.data.preprocessor import HomeCreditPreprocessor
from src.explainability.shap_explainer import HomeCreditShapExplainer
from src.talk_to_data.nl_to_sql import NLToSQLConverter
from src.talk_to_data.query_response import QueryResponse
from src.utils.config import (
    CHARTS_DIR,
    METRICS_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RULES_PATH,
    SHAP_DIR,
    TARGET_COLUMN,
)
from notebooks.eda import generate_eda_charts

logger = logging.getLogger(__name__)

PAGES: tuple[str, ...] = (
    "Dashboard",
    "EDA",
    "Risk Prediction",
    "Explainability",
    "Business Rules",
    "Talk-to-Data Chatbot",
)

EDA_CHARTS: dict[str, str] = {
    "Default Distribution": "default_distribution.png",
    "Income Distribution": "income_distribution.png",
    "Correlation Heatmap": "correlation_heatmap.png",
    "Age vs Default": "age_vs_default.png",
    "Occupation vs Default": "occupation_vs_default.png",
}

RISK_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Low Risk", 0.00, 0.30),
    ("Medium Risk", 0.31, 0.70),
    ("High Risk", 0.71, 1.00),
)

BUSINESS_INSIGHTS: tuple[str, ...] = (
    "External credit bureau scores are the strongest predictors of default risk.",
    "Default rate is approximately 8%.",
    "Income and employment stability significantly influence risk.",
    "Explainability improves regulatory transparency.",
)

PLATFORM_CAPABILITIES: tuple[str, ...] = (
    "EDA & Insights",
    "Risk Prediction",
    "Explainable AI",
    "Business Rules",
    "Talk-to-Data Chatbot",
)

RISK_SAMPLE_SIZE = 20_000


def configure_logging() -> None:
    """Configure application logging once per session."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def inject_custom_styles() -> None:
    """Apply minimalist styling with dark-theme and light-theme compatibility."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        h1, h2, h3 {
            font-weight: 700 !important;
        }

        [data-theme="dark"] h1,
        [data-theme="dark"] h2,
        [data-theme="dark"] h3,
        [data-theme="dark"] h4,
        [data-theme="dark"] h5,
        [data-theme="dark"] h6 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        [data-theme="light"] h1,
        [data-theme="light"] h2,
        [data-theme="light"] h3,
        [data-theme="light"] h4,
        [data-theme="light"] h5,
        [data-theme="light"] h6 {
            color: #0f172a !important;
            font-weight: 700 !important;
        }

        .insight-card {
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-left: 4px solid #3b82f6;
            border-radius: 0.75rem;
            padding: 1rem 1.15rem;
            margin-bottom: 0.75rem;
            line-height: 1.55;
        }
        [data-theme="dark"] .insight-card,
        [data-theme="dark"] .capability-card {
            color: #f8fafc;
        }
        [data-theme="light"] .insight-card,
        [data-theme="light"] .capability-card {
            color: #1e293b;
        }
        .capability-card {
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        .risk-card {
            padding: 1.25rem;
            border-radius: 0.75rem;
            color: white;
            font-weight: 600;
            text-align: center;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
        }
        .risk-low { background: linear-gradient(135deg, #059669, #10b981); }
        .risk-medium { background: linear-gradient(135deg, #d97706, #f59e0b); color: #1f2937; }
        .risk-high { background: linear-gradient(135deg, #dc2626, #ef4444); }
        div[data-testid="stMetric"] {
            background: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 0.75rem;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"] {
            opacity: 0.85;
        }

        /* Chatbot layout */
        .main .block-container {
            overflow-x: hidden;
        }
        [data-testid="stVerticalBlock"] > div {
            max-width: 100%;
        }
        .chat-sql-block pre {
            max-height: 180px;
            overflow-y: auto;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }
        [data-testid="stCodeBlock"] pre {
            max-height: 180px;
            overflow-y: auto;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }
        [data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
            z-index: 999;
            background: inherit;
            padding-top: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    """Initialize shared Streamlit session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "upload_filename" not in st.session_state:
        st.session_state.upload_filename = None


def load_json_file(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists."""
    if not path.is_file():
        logger.warning("Missing JSON file: %s", path)
        return None
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load JSON file: %s", path)
        return None


def load_text_file(path: Path) -> str | None:
    """Load a text file if it exists."""
    if not path.is_file():
        logger.warning("Missing text file: %s", path)
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Failed to load text file: %s", path)
        return None


@st.cache_data(show_spinner="Loading dataset...")
def get_training_data() -> pd.DataFrame:
    """Cache the training dataset for dashboard and selection widgets."""
    return load_application_train()


@st.cache_resource(show_spinner=False)
def load_prediction_artifacts() -> tuple[LGBMClassifier | None, HomeCreditPreprocessor | None]:
    """Load model and preprocessor artifacts for scoring."""
    model: LGBMClassifier | None = None
    preprocessor: HomeCreditPreprocessor | None = None

    if MODEL_PATH.is_file():
        try:
            loaded_model = joblib.load(MODEL_PATH)
            if isinstance(loaded_model, LGBMClassifier):
                model = loaded_model
        except Exception:
            logger.exception("Failed to load model from %s", MODEL_PATH)
    else:
        logger.warning("Model file not found: %s", MODEL_PATH)

    if PREPROCESSOR_PATH.is_file():
        try:
            preprocessor = HomeCreditPreprocessor.load(PREPROCESSOR_PATH)
        except Exception:
            logger.exception("Failed to load preprocessor from %s", PREPROCESSOR_PATH)
    else:
        logger.warning("Preprocessor file not found: %s", PREPROCESSOR_PATH)

    return model, preprocessor


@st.cache_resource(show_spinner=False)
def get_shap_explainer() -> HomeCreditShapExplainer | None:
    """Load SHAP explainer when model artifacts are available."""
    if not MODEL_PATH.is_file() or not PREPROCESSOR_PATH.is_file():
        return None
    try:
        return HomeCreditShapExplainer.from_artifacts()
    except Exception:
        logger.exception("Failed to initialize SHAP explainer")
        return None


@st.cache_resource(show_spinner=False)
def get_nl_converter() -> NLToSQLConverter:
    """Return a cached NL-to-SQL converter backed by the training dataset."""
    return NLToSQLConverter(dataframe=get_training_data())


@st.cache_data(show_spinner="Generating charts...")
def ensure_eda_charts() -> dict[str, Path]:
    """Ensure EDA charts exist and return their paths."""
    chart_paths: dict[str, Path] = {}
    missing = False
    for title, filename in EDA_CHARTS.items():
        path = CHARTS_DIR / filename
        chart_paths[title] = path
        if not path.is_file():
            missing = True

    if missing:
        logger.info("Generating missing EDA charts")
        generate_eda_charts(dataframe=get_training_data(), output_dir=CHARTS_DIR)
        for title, filename in EDA_CHARTS.items():
            chart_paths[title] = CHARTS_DIR / filename

    return chart_paths


def get_risk_band(probability: float) -> str:
    """Map a default probability to a business risk band."""
    if probability <= 0.30:
        return "Low Risk"
    if probability <= 0.70:
        return "Medium Risk"
    return "High Risk"


def get_risk_band_class(risk_band: str) -> str:
    """Return CSS class for a risk band card."""
    return {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }.get(risk_band, "risk-medium")


def clean_feature_name(feature_name: str) -> str:
    """Convert internal feature names to business-friendly labels."""
    for prefix in ("num__", "cat__"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):].replace("_", " ").title()
    return feature_name.replace("_", " ").title()


def predict_default_probability(
    applicants: pd.DataFrame,
    model: LGBMClassifier,
    preprocessor: HomeCreditPreprocessor,
) -> pd.Series:
    """Score applicants and return default probabilities."""
    features = preprocessor.transform_to_dataframe(applicants)
    probabilities = model.predict_proba(features)[:, 1]
    return pd.Series(probabilities, index=applicants.index, name="default_probability")


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read an uploaded CSV file with common encoding fallbacks."""
    content = uploaded_file.getvalue()
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode CSV with supported encodings.")


def validate_applicant_dataframe(
    applicants: pd.DataFrame,
    preprocessor: HomeCreditPreprocessor,
) -> list[str]:
    """Return missing feature columns required by the fitted preprocessor."""
    if applicants.empty:
        return ["Dataset contains no rows."]

    if preprocessor.numerical_columns_ is None or preprocessor.categorical_columns_ is None:
        return ["Preprocessor feature columns are unavailable."]

    required_columns = set(preprocessor.numerical_columns_) | set(preprocessor.categorical_columns_)
    missing_columns = sorted(required_columns - set(applicants.columns))
    return missing_columns


def build_prediction_results(
    applicants: pd.DataFrame,
    probabilities: pd.Series,
) -> pd.DataFrame:
    """Build a results dataframe with probabilities and risk bands."""
    results = pd.DataFrame(
        {
            "default_probability": probabilities.values,
            "risk_band": [get_risk_band(float(value)) for value in probabilities.values],
        },
        index=applicants.index,
    )
    if "SK_ID_CURR" in applicants.columns:
        results.insert(0, "SK_ID_CURR", applicants["SK_ID_CURR"].values)
    return results.reset_index(drop=True)


def parse_business_rules(rules_text: str) -> list[str]:
    """Split the rules document into individual rule blocks."""
    rules: list[str] = []
    current_lines: list[str] = []

    for line in rules_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("=") or stripped == "BUSINESS CREDIT RISK RULES":
            continue
        if stripped.startswith("Rule "):
            if current_lines:
                rules.append("\n".join(current_lines))
            current_lines = [stripped]
        elif current_lines:
            current_lines.append(stripped)

    if current_lines:
        rules.append("\n".join(current_lines))
    return rules


def render_page_header(title: str, subtitle: str) -> None:
    """Render a consistent page header using native Streamlit components."""
    st.header(title)
    st.caption(subtitle)


@st.cache_data(show_spinner=False)
def get_risk_band_distribution() -> pd.Series | None:
    """Estimate portfolio risk-band distribution using the trained model."""
    model, preprocessor = load_prediction_artifacts()
    if model is None or preprocessor is None:
        return None

    df = get_training_data()
    sample_size = min(RISK_SAMPLE_SIZE, len(df))
    sample = df.sample(n=sample_size, random_state=42)
    probabilities = predict_default_probability(sample, model, preprocessor)
    bands = probabilities.apply(get_risk_band)
    return bands.value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)


def build_metrics_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """Format evaluation metrics for tabular display."""
    display_names = {
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
    }
    rows = [
        {"Metric": display_names[key], "Score": round(float(metrics[key]), 4)}
        for key in display_names
        if key in metrics
    ]
    return pd.DataFrame(rows)


def render_metric_card(label: str, value: str, css_class: str) -> None:
    """Render a colored HTML metric card."""
    st.markdown(
        f'<div class="risk-card {css_class}"><div>{label}</div><div style="font-size:1.5rem;">{value}</div></div>',
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    """Render the executive dashboard page."""
    st.header("AI-Powered Credit Risk Intelligence Platform")
    st.caption(
        "Predict loan default risk, explain decisions, explore data, "
        "and query insights using natural language.",
    )

    df = get_training_data()
    metrics = load_json_file(METRICS_PATH)
    default_rate = float(df[TARGET_COLUMN].mean()) if TARGET_COLUMN in df.columns else 0.0
    high_risk_count = int(df[TARGET_COLUMN].sum()) if TARGET_COLUMN in df.columns else 0
    roc_auc = float(metrics.get("roc_auc", 0.0)) if metrics else 0.0

    st.subheader("Executive KPIs")
    kpi_row = st.columns(4)
    kpi_row[0].metric("Total Applicants", f"{len(df):,}")
    kpi_row[1].metric("Default Rate", f"{default_rate:.1%}")
    kpi_row[2].metric("ROC-AUC", f"{roc_auc:.3f}" if metrics else "N/A")
    kpi_row[3].metric("High Risk Applicants", f"{high_risk_count:,}")

    st.subheader("Model Quality Metrics")
    if metrics is None:
        st.warning("Model metrics file not found at `models/metrics.json`. Train the model first.")
    else:
        metric_row = st.columns(4)
        metric_row[0].metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
        metric_row[1].metric("Precision", f"{metrics.get('precision', 0):.3f}")
        metric_row[2].metric("Recall", f"{metrics.get('recall', 0):.3f}")
        metric_row[3].metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")

    st.divider()

    insight_col, capability_col = st.columns([1.4, 1])

    with insight_col:
        st.subheader("Business Insights")
        with st.container(border=True):
            for insight in BUSINESS_INSIGHTS:
                st.markdown(
                    f'<div class="insight-card">📌 {insight}</div>',
                    unsafe_allow_html=True,
                )

    with capability_col:
        st.subheader("Platform Capabilities")
        with st.container(border=True):
            for capability in PLATFORM_CAPABILITIES:
                st.markdown(
                    f'<div class="capability-card">✅ {capability}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    performance_col, distribution_col = st.columns(2)

    with performance_col:
        st.subheader("Model Performance Overview")
        if metrics is None:
            st.info("Train the model to view performance analytics.")
        else:
            metrics_df = build_metrics_dataframe(metrics)
            st.dataframe(metrics_df, width="stretch", hide_index=True)
            chart = px.bar(
                metrics_df,
                x="Metric",
                y="Score",
                text="Score",
                color="Score",
                color_continuous_scale="Blues",
                title="Evaluation Metrics",
            )
            chart.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            chart.update_layout(
                yaxis_range=[0, 1],
                coloraxis_showscale=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(chart, width="stretch")

    with distribution_col:
        st.subheader("Risk Distribution")
        band_counts = get_risk_band_distribution()
        if band_counts is None:
            st.info("Load model artifacts to view predicted risk-band distribution.")
        else:
            distribution_df = band_counts.rename_axis("Risk Band").reset_index(name="Applicants")
            pie_chart = px.pie(
                distribution_df,
                names="Risk Band",
                values="Applicants",
                color="Risk Band",
                color_discrete_map={
                    "Low Risk": "#10b981",
                    "Medium Risk": "#f59e0b",
                    "High Risk": "#ef4444",
                },
                hole=0.45,
            )
            pie_chart.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(pie_chart, width="stretch")
            st.caption(f"Based on a sample of {band_counts.sum():,} scored applicants.")

    st.divider()
    st.subheader("Risk Band Definitions")
    band_cols = st.columns(3)
    for column, (label, lower, upper) in zip(band_cols, RISK_BANDS):
        column.info(f"**{label}**: {lower:.2f} – {upper:.2f}")


def render_eda() -> None:
    """Render exploratory data analysis charts."""
    render_page_header("Exploratory Data Analysis", "Visual summaries of portfolio characteristics.")

    try:
        chart_paths = ensure_eda_charts()
    except Exception as error:
        logger.exception("Failed to generate EDA charts")
        st.error(f"Unable to load charts: {error}")
        return

    available = 0
    for chart_title, chart_path in chart_paths.items():
        st.subheader(chart_title)
        if chart_path.is_file():
            st.image(str(chart_path), width="stretch")
            available += 1
        else:
            st.warning(f"Chart not found: `{chart_path}`")

    if available == 0:
        st.info("No EDA charts are available yet.")


def render_risk_prediction() -> None:
    """Render applicant scoring workflow."""
    render_page_header("Risk Prediction", "Score applicants with the trained LightGBM model.")

    model, preprocessor = load_prediction_artifacts()
    if model is None or preprocessor is None:
        st.error(
            "Prediction artifacts are missing. Expected files:\n"
            f"- `{MODEL_PATH}`\n"
            f"- `{PREPROCESSOR_PATH}`"
        )
        return

    input_mode = st.radio(
        "Input Source",
        options=("Select Applicant from Training Data", "Upload CSV"),
        horizontal=True,
    )

    applicants: pd.DataFrame | None = None

    if input_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload applicant CSV", type=["csv"])
        if uploaded_file is not None:
            if st.session_state.upload_filename != uploaded_file.name:
                try:
                    uploaded_df = read_uploaded_csv(uploaded_file)
                    validation_issues = validate_applicant_dataframe(uploaded_df, preprocessor)
                    if validation_issues:
                        st.session_state.uploaded_df = None
                        st.session_state.upload_filename = None
                        st.error(
                            "Uploaded file is missing required feature columns: "
                            f"{', '.join(validation_issues)}",
                        )
                    else:
                        st.session_state.uploaded_df = uploaded_df
                        st.session_state.upload_filename = uploaded_file.name
                        logger.info(
                            "Uploaded file '%s' with %d rows and %d columns",
                            uploaded_file.name,
                            len(uploaded_df),
                            len(uploaded_df.columns),
                        )
                        st.success(f"{len(uploaded_df):,} applicants loaded successfully.")
                except Exception as error:
                    logger.exception("Failed to read uploaded CSV")
                    st.error(f"Unable to read uploaded CSV: {error}")

        if st.session_state.uploaded_df is not None:
            applicants = st.session_state.uploaded_df.copy()
            preview_cols = st.columns(3)
            preview_cols[0].metric("Rows", f"{len(applicants):,}")
            preview_cols[1].metric("Columns", f"{len(applicants.columns):,}")
            preview_cols[2].metric("Source File", st.session_state.upload_filename or "uploaded.csv")
            st.subheader("Preview")
            st.dataframe(applicants.head(5), width="stretch", hide_index=True)
    else:
        st.session_state.uploaded_df = None
        st.session_state.upload_filename = None
        df = get_training_data()
        if "SK_ID_CURR" not in df.columns:
            st.error("Training data does not contain `SK_ID_CURR` for applicant selection.")
            return

        applicant_ids = df["SK_ID_CURR"].dropna().astype(int).tolist()
        selected_id = st.selectbox("Select Applicant ID", options=applicant_ids)
        applicants = df[df["SK_ID_CURR"] == selected_id].copy()
        st.dataframe(applicants.head(1), width="stretch", hide_index=True)

    if st.button("Predict", type="primary"):
        if applicants is None or applicants.empty:
            st.warning("Provide applicant data before running prediction.")
            return

        validation_issues = validate_applicant_dataframe(applicants, preprocessor)
        if validation_issues:
            st.error(f"Cannot run prediction: {', '.join(validation_issues)}")
            return

        try:
            probabilities = predict_default_probability(applicants, model, preprocessor)
            logger.info("Generated predictions for %d applicant(s)", len(probabilities))
        except Exception as error:
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {error}")
            return

        if len(applicants) == 1:
            st.subheader("Prediction Results")
            probability = float(probabilities.iloc[0])
            risk_band = get_risk_band(probability)
            css_class = get_risk_band_class(risk_band)
            applicant_label = "Uploaded Applicant"
            if "SK_ID_CURR" in applicants.columns:
                applicant_label = f"Applicant {applicants.iloc[0]['SK_ID_CURR']}"

            st.subheader(applicant_label)
            result_cols = st.columns(2)
            with result_cols[0]:
                render_metric_card("Default Probability", f"{probability:.2%}", css_class)
            with result_cols[1]:
                render_metric_card("Risk Band", risk_band, css_class)

            explainer = get_shap_explainer()
            if explainer is not None:
                with st.spinner("Generating SHAP explanation..."):
                    try:
                        explanation = explainer.explain_applicant(applicants.iloc[[0]])
                        st.subheader("SHAP Explanation")
                        driver_cols = st.columns(2)
                        with driver_cols[0]:
                            st.markdown("**Top Positive Risk Drivers**")
                            for driver in explanation["top_positive_drivers"]:
                                feature = clean_feature_name(driver["feature"])
                                st.markdown(
                                    f"- **{feature}** (+{driver['shap_value']:.3f})",
                                )
                        with driver_cols[1]:
                            st.markdown("**Top Negative Risk Drivers**")
                            for driver in explanation["top_negative_drivers"]:
                                feature = clean_feature_name(driver["feature"])
                                st.markdown(
                                    f"- **{feature}** ({driver['shap_value']:.3f})",
                                )
                    except Exception as error:
                        logger.exception("SHAP explanation failed during prediction")
                        st.warning(f"SHAP explanation unavailable: {error}")
            else:
                st.info("SHAP explainer is not available. Train model artifacts to enable explanations.")
        else:
            st.subheader("Batch Prediction Results")
            results_df = build_prediction_results(applicants, probabilities)
            st.dataframe(results_df, width="stretch", hide_index=True)
            st.caption(f"Scored {len(results_df):,} applicants.")


def render_explainability() -> None:
    """Render global and local SHAP explanations."""
    render_page_header("Model Explainability", "Global feature influence and applicant-level drivers.")

    summary_plot = SHAP_DIR / "summary_plot.png"
    importance_plot = SHAP_DIR / "global_importance.png"

    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.subheader("SHAP Summary Plot")
        if summary_plot.is_file():
            st.image(str(summary_plot), width="stretch")
        else:
            st.warning(f"Missing SHAP summary plot at `{summary_plot}`.")

    with plot_cols[1]:
        st.subheader("Feature Importance Plot")
        if importance_plot.is_file():
            st.image(str(importance_plot), width="stretch")
        else:
            st.warning(f"Missing feature importance plot at `{importance_plot}`.")

    explainer = get_shap_explainer()
    if explainer is None:
        st.error("SHAP explainer could not be loaded. Train model artifacts first.")
        return

    df = get_training_data()
    if "SK_ID_CURR" not in df.columns:
        st.error("Training data does not contain applicant IDs for local explanations.")
        return

    applicant_ids = df["SK_ID_CURR"].dropna().astype(int).tolist()
    selected_id = st.selectbox("Applicant for Local Explanation", options=applicant_ids, key="shap_applicant")
    applicant = df[df["SK_ID_CURR"] == selected_id].iloc[[0]]

    if st.button("Explain Applicant", type="primary"):
        with st.spinner("Generating SHAP explanation..."):
            try:
                explanation = explainer.explain_applicant(applicant)
            except Exception as error:
                logger.exception("Applicant explanation failed")
                st.error(f"Could not explain applicant: {error}")
                return

        probability = explanation["predicted_probability"]
        risk_band = get_risk_band(probability)
        st.info(
            f"Applicant {selected_id} has a predicted default probability of {probability:.1%} "
            f"({risk_band}). Positive drivers increase risk; negative drivers reduce it."
        )

        driver_cols = st.columns(2)
        with driver_cols[0]:
            st.subheader("Top Positive Risk Drivers")
            if explanation["top_positive_drivers"]:
                for driver in explanation["top_positive_drivers"]:
                    feature = clean_feature_name(driver["feature"])
                    st.markdown(
                        f"- **{feature}** increases risk "
                        f"(impact: {driver['shap_value']:.3f}, value: {driver['feature_value']})"
                    )
            else:
                st.info("No strong positive risk drivers were identified.")

        with driver_cols[1]:
            st.subheader("Top Negative Risk Drivers")
            if explanation["top_negative_drivers"]:
                for driver in explanation["top_negative_drivers"]:
                    feature = clean_feature_name(driver["feature"])
                    st.markdown(
                        f"- **{feature}** reduces risk "
                        f"(impact: {driver['shap_value']:.3f}, value: {driver['feature_value']})"
                    )
            else:
                st.info("No strong negative risk drivers were identified.")


def render_business_rules() -> None:
    """Render generated business rules."""
    render_page_header("Business Rules", "Human-readable credit policy rules from a decision tree.")

    rules_text = load_text_file(RULES_PATH)
    if rules_text is None:
        st.warning(
            f"Business rules file not found at `{RULES_PATH}`. "
            "Run `python -m src.explainability.rule_generator` first."
        )
        return

    rules = parse_business_rules(rules_text)
    if not rules:
        st.warning("No rules were found in the business rules document.")
        return

    st.success(f"Loaded {len(rules)} business rules.")
    for index, rule in enumerate(rules, start=1):
        with st.expander(f"Rule {index}", expanded=index == 1):
            st.code(rule, language="text")


def render_chat_response(response: QueryResponse, question_number: int) -> None:
    """Render one chatbot interaction in a bordered container."""
    with st.container(border=True):
        st.markdown(f"## Question {question_number}")
        st.markdown(f"❓ **Question**")
        st.write(response.question)

        if not response.success:
            st.warning(response.error_message or "Unable to process this question.")
            if response.generated_sql:
                st.markdown("🧠 **Generated SQL**")
                st.code(response.generated_sql, language="sql")
            return

        st.markdown("🧠 **Generated SQL**")
        st.code(response.generated_sql, language="sql")

        st.markdown("📊 **Result**")
        if response.result_df.empty:
            st.info("The query returned no rows.")
        else:
            display_df = response.result_df.copy()
            if len(display_df.columns) > 6:
                st.caption(f"Showing all {len(display_df.columns)} columns.")
            st.dataframe(display_df, width="stretch", hide_index=True)

        st.markdown("💡 **Business Summary**")
        st.info(response.business_summary)


def render_chatbot() -> None:
    """Render the natural language analytics chatbot."""
    render_page_header(
        "Talk-to-Data Chatbot",
        "Ask business questions about the Home Credit portfolio in plain language.",
    )

    st.caption("Try asking about defaults, income, occupations, or credit amounts.")

    if not st.session_state.chat_history:
        st.info("Start by asking a question about customers, defaults, income, or occupations.")
    else:
        for question_number, response in enumerate(st.session_state.chat_history, start=1):
            render_chat_response(response, question_number)
            if question_number < len(st.session_state.chat_history):
                st.divider()

    question = st.chat_input("Ask a question about the portfolio...")
    if question:
        with st.spinner("Translating question and running query..."):
            response = get_nl_converter().ask(question)
            st.session_state.chat_history.append(response)
        st.rerun()


def render_sidebar() -> str:
    """Render sidebar navigation and return selected page."""
    st.sidebar.markdown("## Credit Risk Platform")
    st.sidebar.caption("AI-Powered Analytics")
    st.sidebar.divider()
    selected = st.sidebar.radio("Navigation", options=PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Home Credit Default Risk")
    return selected


def main() -> None:
    """Launch the Streamlit application."""
    configure_logging()
    init_session_state()

    st.set_page_config(
        page_title="Credit Risk Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_styles()

    selected_page = render_sidebar()
    page_renderers = {
        "Dashboard": render_dashboard,
        "EDA": render_eda,
        "Risk Prediction": render_risk_prediction,
        "Explainability": render_explainability,
        "Business Rules": render_business_rules,
        "Talk-to-Data Chatbot": render_chatbot,
    }
    page_renderers[selected_page]()


if __name__ == "__main__":
    main()
