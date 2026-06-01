"""Generate EDA chart assets for the Streamlit dashboard."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loader import load_application_train
from src.utils.config import CHARTS_DIR, TARGET_COLUMN
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

NUMERIC_SUBSET = (
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_BIRTH",
)


def generate_eda_charts(
    dataframe: pd.DataFrame | None = None,
    output_dir: Path | str = CHARTS_DIR,
) -> list[Path]:
    """
    Build and save the standard EDA chart set for the Home Credit dataset.

    Returns
    -------
    list[Path]
        Paths to generated chart files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = dataframe if dataframe is not None else load_application_train()
    sns.set_theme(style="whitegrid", palette="muted")
    generated: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    default_counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Non-Default", "Default"]
    ax.bar(labels, [default_counts.get(0, 0), default_counts.get(1, 0)], color=["#4c78a8", "#e45756"])
    ax.set_title("Default Distribution")
    ax.set_ylabel("Applicant Count")
    path = output_path / "default_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    income = df["AMT_INCOME_TOTAL"].dropna()
    ax.hist(income, bins=50, color="#4c78a8", edgecolor="white")
    ax.set_title("Income Distribution")
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Frequency")
    path = output_path / "income_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    numeric_df = df[[col for col in NUMERIC_SUBSET if col in df.columns]].dropna()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    path = output_path / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    age_years = (-df["DAYS_BIRTH"] / 365.25).clip(lower=0)
    plot_df = pd.DataFrame({"age_years": age_years, TARGET_COLUMN: df[TARGET_COLUMN]})
    sns.boxplot(data=plot_df, x=TARGET_COLUMN, y="age_years", ax=ax)
    ax.set_xticklabels(["Non-Default", "Default"])
    ax.set_title("Age vs Default")
    ax.set_xlabel("Default Status")
    ax.set_ylabel("Age (Years)")
    path = output_path / "age_vs_default.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    occupation_rates = (
        df.dropna(subset=["OCCUPATION_TYPE"])
        .groupby("OCCUPATION_TYPE")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    occupation_rates.plot(kind="barh", color="#e45756", ax=ax)
    ax.set_title("Top 10 Occupations by Default Rate")
    ax.set_xlabel("Default Rate")
    ax.invert_yaxis()
    path = output_path / "occupation_vs_default.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)

    logger.info("Generated %d EDA charts in %s", len(generated), output_path)
    return generated


def main() -> None:
    """CLI entrypoint for chart generation."""
    setup_logging()
    paths = generate_eda_charts()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
