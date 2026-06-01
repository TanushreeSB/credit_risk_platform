"""Business rule generation from a shallow decision tree for credit risk."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.data.loader import load_application_train
from src.data.preprocessor import HomeCreditPreprocessor
from src.utils.config import PREPROCESSOR_PATH, RANDOM_STATE, RULES_PATH, TARGET_COLUMN

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = RULES_PATH

TREE_MAX_DEPTH = 3
TREE_MIN_SAMPLES_LEAF = 500

RULES_HEADER = "\n".join(
    [
        "=" * 50,
        "BUSINESS CREDIT RISK RULES",
        "=" * 30,
        "",
    ],
)
RULES_FOOTER = "\n" + "=" * 50


class CreditRiskRuleGenerator:
    """
    Train a shallow decision tree and convert its splits into business rules.

    Rules describe applicant profiles as Low Risk or High Risk using plain
    language IF-THEN statements suitable for analyst review or UI display.
    """

    def __init__(
        self,
        preprocessor_path: Path | str = PREPROCESSOR_PATH,
        rules_output_path: Path | str = DEFAULT_RULES_PATH,
        max_depth: int = TREE_MAX_DEPTH,
        min_samples_leaf: int = TREE_MIN_SAMPLES_LEAF,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.preprocessor_path = Path(preprocessor_path).expanduser().resolve()
        self.rules_output_path = Path(rules_output_path).expanduser().resolve()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.preprocessor_: HomeCreditPreprocessor | None = None
        self.model_: DecisionTreeClassifier | None = None
        self.feature_names_: list[str] | None = None
        self.rules_: list[str] = []

    @property
    def is_trained(self) -> bool:
        """Return True when the decision tree has been fit."""
        return self.model_ is not None

    def train_rule_model(
        self,
        data_path: Path | str | None = None,
    ) -> DecisionTreeClassifier:
        """
        Load training data, preprocess features, and fit a shallow decision tree.

        Returns
        -------
        DecisionTreeClassifier
            Fitted rule model.

        Raises
        ------
        FileNotFoundError
            If training data or the saved preprocessor is missing.
        ValueError
            If the target column is absent from the training data.
        """
        try:
            df = load_application_train(data_path)
        except FileNotFoundError:
            logger.exception("Unable to load training data for rule generation.")
            raise

        if TARGET_COLUMN not in df.columns:
            message = f"Target column '{TARGET_COLUMN}' not found in training data."
            logger.error(message)
            raise ValueError(message)

        try:
            self.preprocessor_ = HomeCreditPreprocessor.load(self.preprocessor_path)
        except FileNotFoundError:
            logger.exception(
                "Saved preprocessor not found at %s. Train the LightGBM model first.",
                self.preprocessor_path,
            )
            raise

        logger.info("Transforming training data for rule model")
        features = self.preprocessor_.transform(df)
        self.feature_names_ = [
            _clean_feature_name(name)
            for name in self.preprocessor_.get_feature_names_out()
        ]
        target = df[TARGET_COLUMN].to_numpy()

        logger.info(
            "Training decision tree (max_depth=%d, min_samples_leaf=%d)",
            self.max_depth,
            self.min_samples_leaf,
        )
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model_.fit(features, target)

        logger.info("Decision tree trained with %d leaves", self.model_.get_n_leaves())
        self.rules_ = self.extract_rules()
        return self.model_

    def extract_rules(self) -> list[str]:
        """
        Convert fitted tree splits into human-readable IF-THEN rule strings.

        Returns
        -------
        list[str]
            One formatted rule block per leaf node.

        Raises
        ------
        RuntimeError
            If called before the rule model has been trained.
        """
        if not self.is_trained or self.model_ is None:
            message = "Rule model is not trained. Call train_rule_model() first."
            logger.error(message)
            raise RuntimeError(message)

        if self.feature_names_ is None:
            message = "Feature names are unavailable for rule extraction."
            logger.error(message)
            raise RuntimeError(message)

        tree = self.model_.tree_
        rule_paths: list[tuple[list[str], str]] = []

        def _walk_tree(node_id: int, conditions: list[str]) -> None:
            if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                feature = self.feature_names_[tree.feature[node_id]]
                threshold = float(tree.threshold[node_id])

                left_conditions = conditions + [
                    f"{feature} <= {_format_threshold(threshold)}",
                ]
                right_conditions = conditions + [
                    f"{feature} > {_format_threshold(threshold)}",
                ]
                _walk_tree(tree.children_left[node_id], left_conditions)
                _walk_tree(tree.children_right[node_id], right_conditions)
                return

            class_counts = tree.value[node_id][0]
            predicted_class = int(np.argmax(class_counts))
            outcome = "HIGH RISK" if predicted_class == 1 else "LOW RISK"
            rule_paths.append((conditions, outcome))

        _walk_tree(0, [])

        formatted_rules: list[str] = []
        for index, (conditions, outcome) in enumerate(rule_paths, start=1):
            if conditions:
                condition_lines = "\nAND ".join(
                    [f"IF {conditions[0]}"] + conditions[1:],
                )
            else:
                condition_lines = "IF no specific conditions apply"

            formatted_rules.append(
                "\n".join(
                    [
                        f"Rule {index}:",
                        condition_lines,
                        f"THEN {outcome}",
                    ],
                ),
            )

        self.rules_ = formatted_rules
        logger.info("Extracted %d business rules from decision tree", len(formatted_rules))
        return formatted_rules

    def get_rules(self) -> list[str]:
        """
        Return the latest generated business rules.

        Raises
        ------
        RuntimeError
            If rules have not been extracted yet.
        """
        if not self.rules_:
            if self.is_trained:
                return self.extract_rules()
            message = "No rules available. Call train_rule_model() first."
            logger.error(message)
            raise RuntimeError(message)
        return self.rules_

    def save_rules(self, output_path: Path | str | None = None) -> Path:
        """
        Persist generated rules to a text file.

        Returns
        -------
        Path
            Path to the saved rules file.

        Raises
        ------
        RuntimeError
            If no rules are available to save.
        """
        rules = self.get_rules()
        destination = Path(
            output_path if output_path is not None else self.rules_output_path,
        ).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        document = RULES_HEADER + "\n\n".join(rules) + RULES_FOOTER + "\n"
        try:
            destination.write_text(document, encoding="utf-8")
        except OSError:
            logger.exception("Failed to save business rules to %s", destination)
            raise

        logger.info("Saved %d business rules to %s", len(rules), destination)
        return destination

    def format_rules_for_display(self) -> str:
        """Return the full rules document as a single string for UI rendering."""
        return RULES_HEADER + "\n\n".join(self.get_rules()) + RULES_FOOTER


def _clean_feature_name(feature_name: str) -> str:
    """Remove sklearn transformer prefixes from feature names."""
    for prefix in ("num__", "cat__"):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):]
    return feature_name


def _format_threshold(value: float) -> str:
    """Format split thresholds for business-facing rule text."""
    if np.isfinite(value) and abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def main() -> None:
    """Train the rule model, print rules, and save them to disk."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    generator = CreditRiskRuleGenerator()

    try:
        generator.train_rule_model()
        rules = generator.get_rules()
        output_path = generator.save_rules()

        print(generator.format_rules_for_display())

        logger.info("Generated %d rules and saved them to %s", len(rules), output_path)
    except (FileNotFoundError, ValueError, RuntimeError, OSError):
        logger.exception("Business rule generation failed.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
