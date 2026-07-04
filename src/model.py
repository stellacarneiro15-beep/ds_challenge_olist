"""Production model and scoring helpers for late-delivery risk."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

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
DEFAULT_SCALE_POS_WEIGHT = 1.0


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


def _negative_to_positive_ratio(target: pd.Series) -> float:
    """Return the class-imbalance ratio used by XGBoost."""
    positives = int((target == 1).sum())
    negatives = int((target == 0).sum())
    if positives == 0:
        return DEFAULT_SCALE_POS_WEIGHT
    return negatives / positives


def build_model_pipeline(scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT) -> Pipeline:
    """Build the single model used by ``python -m src.main``."""
    return Pipeline(
        steps=[
            ("prep", _preprocessor()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
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
        threshold_estimator = build_model_pipeline(
            scale_pos_weight=_negative_to_positive_ratio(train_set[TARGET_COLUMN])
        )
        threshold_estimator.fit(train_set[FEATURE_COLUMNS], train_set[TARGET_COLUMN])
        threshold = select_operating_threshold(threshold_estimator, validation_set)

    estimator = build_model_pipeline(
        scale_pos_weight=_negative_to_positive_ratio(dataset[TARGET_COLUMN])
    )
    estimator.fit(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN])
    return ModelArtifact(
        model_name="xgboost_scale_pos_weight",
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
