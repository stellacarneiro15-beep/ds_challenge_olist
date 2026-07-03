"""Production model and scoring helpers for late-delivery risk."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluate import best_f1_threshold, time_based_split
from src.features import (
    CATEGORICAL_FEATURES,
    CUSTOMER_ID_COLUMN,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    ORDER_ID_COLUMN,
    TARGET_COLUMN,
)

DEFAULT_MODEL_PATH = Path("artifacts/late_delivery_model.pkl")
DEFAULT_THRESHOLD = 0.5
THRESHOLD_VALIDATION_FRACTION = 0.2


class SupportsPredictProba(Protocol):
    def predict_proba(self, x: pd.DataFrame) -> Any: ...


@dataclass
class ModelArtifact:
    """Serializable bundle for production inference."""

    model_name: str
    estimator: SupportsPredictProba
    feature_columns: list[str] = field(default_factory=lambda: FEATURE_COLUMNS.copy())
    threshold: float = DEFAULT_THRESHOLD


def _preprocessor() -> ColumnTransformer:
    """Impute numerics and one-hot encode categoricals for the production model."""
    numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipeline() -> Pipeline:
    """Build the single model used by ``python -m src.main``."""
    return Pipeline(
        steps=[
            ("prep", _preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.1,
                    max_iter=300,
                    max_depth=None,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def select_operating_threshold(
    estimator: SupportsPredictProba,
    validation_set: pd.DataFrame,
) -> float:
    """Choose the production alert threshold on held-out validation rows."""
    if validation_set[TARGET_COLUMN].nunique() < 2:
        return DEFAULT_THRESHOLD

    probabilities = estimator.predict_proba(validation_set[FEATURE_COLUMNS])[:, 1]
    threshold, _ = best_f1_threshold(validation_set[TARGET_COLUMN], probabilities)
    return threshold


def train_model_artifact(dataset: pd.DataFrame) -> ModelArtifact:
    """Train the production model and carry forward a validation-selected threshold."""
    train_set, validation_set = time_based_split(
        dataset,
        test_frac=THRESHOLD_VALIDATION_FRACTION,
    )

    threshold = DEFAULT_THRESHOLD
    if train_set[TARGET_COLUMN].nunique() >= 2:
        threshold_estimator = build_model_pipeline()
        threshold_estimator.fit(train_set[FEATURE_COLUMNS], train_set[TARGET_COLUMN])
        threshold = select_operating_threshold(threshold_estimator, validation_set)

    estimator = build_model_pipeline()
    estimator.fit(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN])
    return ModelArtifact(
        model_name="gradient_boosting",
        estimator=estimator,
        threshold=threshold,
    )


def save_model(model: object, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Persist a fitted model artifact for later CLI/API inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> object:
    """Load a fitted model artifact produced by ``save_model``."""
    with Path(path).open("rb") as f:
        return pickle.load(f)  # noqa: S301 - local model artifact, not untrusted input


def score_customer_orders(
    artifact: ModelArtifact,
    dataset: pd.DataFrame,
    customer_id: str,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return the customer's highest-risk orders for late delivery."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    customer_rows = dataset[
        dataset[CUSTOMER_ID_COLUMN].astype(str) == str(customer_id)
    ].copy()
    columns = [
        ORDER_ID_COLUMN,
        CUSTOMER_ID_COLUMN,
        "late_delivery_risk",
        "predicted_late",
        TARGET_COLUMN,
    ]
    if customer_rows.empty:
        return pd.DataFrame(columns=columns)

    probabilities = artifact.estimator.predict_proba(customer_rows[artifact.feature_columns])[:, 1]
    customer_rows["late_delivery_risk"] = probabilities
    customer_rows["predicted_late"] = (probabilities >= artifact.threshold).astype(int)

    return (
        customer_rows.sort_values("late_delivery_risk", ascending=False)
        .head(top_k)[columns]
        .reset_index(drop=True)
    )
