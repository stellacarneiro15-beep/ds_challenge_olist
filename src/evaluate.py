"""Evaluation utilities for the late-delivery model.

The split is **chronological**: the most-recent orders are held out for testing,
so evaluation mimics predicting future orders from past ones (no look-ahead).
The operating threshold is chosen to maximise F1 rather than defaulting to 0.5,
because late deliveries are the rare, costly positive class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.features import TIME_COLUMN


def time_based_split(
    dataset: pd.DataFrame,
    test_frac: float = 0.2,
    time_column: str = TIME_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically into (train, test), holding out the most-recent rows.

    ``test_frac`` is the fraction of the most-recent orders reserved for testing.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1 (exclusive).")

    ordered = dataset.sort_values(time_column).reset_index(drop=True)
    test_start = int(len(ordered) * (1 - test_frac))
    return ordered.iloc[:test_start].copy(), ordered.iloc[test_start:].copy()


def best_f1_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    """Return the probability threshold that maximises F1, and that F1 value."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_values = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    # precision/recall carry one extra point with no corresponding threshold.
    best_idx = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def evaluate_classifier(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Score probabilistic predictions with ranking, calibration and F1 metrics.

    If ``threshold`` is ``None`` it is selected to maximise F1 on the given data.
    """
    if threshold is None:
        threshold, _ = best_f1_threshold(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
