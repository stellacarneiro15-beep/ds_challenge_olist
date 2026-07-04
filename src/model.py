"""Production model and scoring helpers for late-delivery risk."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.evaluate import best_f1_threshold, time_based_split
from src.features import (
    CATEGORICAL_FEATURES,
    CUSTOMER_ID_COLUMN,
    CUSTOMER_UNIQUE_ID_COLUMN,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    ORDER_ID_COLUMN,
    TARGET_COLUMN,
)

DEFAULT_MODEL_PATH = Path("artifacts/late_delivery_model.pkl")
DEFAULT_THRESHOLD = 0.5
THRESHOLD_VALIDATION_FRACTION = 0.2
DEFAULT_SCALE_POS_WEIGHT = 1.0
# `scale_pos_weight` optimises the tree for the rare late class but leaves the raw
# scores badly miscalibrated (they run ~6x the true late rate). We therefore wrap
# the fitted tree in isotonic calibration so `predict_proba` is an honest
# probability that can be surfaced to users as a real late-delivery risk.
CALIBRATION_METHOD = "isotonic"


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


def _fit_calibrated_pipeline(
    features: pd.DataFrame,
    target: pd.Series,
    scale_pos_weight: float,
) -> CalibratedClassifierCV:
    """Fit the weighted tree and wrap it in cross-validated isotonic calibration.

    Calibration runs an internal CV over the *training* rows only, so the returned
    estimator never sees the held-out threshold rows and its ``predict_proba`` is a
    calibrated late-delivery probability rather than a `scale_pos_weight`-inflated
    score.
    """
    base = build_model_pipeline(scale_pos_weight=scale_pos_weight)
    calibrated = CalibratedClassifierCV(base, method=CALIBRATION_METHOD, cv=5)
    calibrated.fit(features, target)
    return calibrated


def train_model_artifact(dataset: pd.DataFrame) -> ModelArtifact:
    """Train the calibrated production model and select its operating threshold.

    The dataset is split chronologically. The model is trained and calibrated on
    the earlier rows; the operating threshold is then chosen on the most-recent
    held-out rows using the **exact estimator that ships**, so the threshold and
    the probabilities it is applied to always come from the same model (the old
    code tuned the threshold on a model trained on a subset, then shipped a
    different model refit on all the data).
    """
    train_set, threshold_set = time_based_split(
        dataset,
        test_frac=THRESHOLD_VALIDATION_FRACTION,
    )

    # Calibration and F1-threshold selection both need both classes present in
    # their respective folds. Fall back to an uncalibrated, default-threshold model
    # trained on all rows when the split cannot support that.
    if (
        train_set[TARGET_COLUMN].nunique() < 2
        or threshold_set[TARGET_COLUMN].nunique() < 2
    ):
        estimator = build_model_pipeline(
            scale_pos_weight=_negative_to_positive_ratio(dataset[TARGET_COLUMN])
        )
        estimator.fit(dataset[FEATURE_COLUMNS], dataset[TARGET_COLUMN])
        return ModelArtifact(
            model_name="xgboost_scale_pos_weight",
            estimator=estimator,
            threshold=DEFAULT_THRESHOLD,
        )

    estimator = _fit_calibrated_pipeline(
        train_set[FEATURE_COLUMNS],
        train_set[TARGET_COLUMN],
        scale_pos_weight=_negative_to_positive_ratio(train_set[TARGET_COLUMN]),
    )
    threshold = select_operating_threshold(estimator, threshold_set)
    return ModelArtifact(
        model_name="xgboost_scale_pos_weight_isotonic",
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
    customer_unique_id: str,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return a customer's highest-risk orders for late delivery.

    Orders are grouped by ``customer_unique_id`` (the person-level key). Grouping
    by ``customer_id`` would be a no-op, since Olist assigns a fresh ``customer_id``
    to every order, so a customer would never have more than one row to rank.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    customer_rows = dataset[
        dataset[CUSTOMER_UNIQUE_ID_COLUMN].astype(str) == str(customer_unique_id)
    ].copy()
    columns = [
        ORDER_ID_COLUMN,
        CUSTOMER_UNIQUE_ID_COLUMN,
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
