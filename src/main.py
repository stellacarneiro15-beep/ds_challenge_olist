"""CLI entrypoint for customer late-delivery risk scoring.

Usage:
    python -m src.main --customer_unique_id <ID> --top_k 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.model import ModelArtifact

if __name__ == "__main__" and "LOKY_MAX_CPU_COUNT" not in os.environ:
    env = os.environ.copy()
    env["LOKY_MAX_CPU_COUNT"] = "1"
    os.execvpe(sys.executable, [sys.executable, "-m", "src.main", *sys.argv[1:]], env)  # noqa: S606


def parse_args() -> argparse.Namespace:
    from src.data_loader import DEFAULT_DATA_DIR
    from src.model import DEFAULT_MODEL_PATH

    parser = argparse.ArgumentParser(description="Score Olist late-delivery risk")
    parser.add_argument(
        "--customer_unique_id",
        required=True,
        help="Customer unique ID (person-level key) whose orders to score",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of highest-risk orders to print",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("DATA_DIR", DEFAULT_DATA_DIR),
        help="Directory containing the Olist CSVs",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path for the pickled production model artifact",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining and overwrite --model_path instead of loading it",
    )
    return parser.parse_args()


def _load_or_train_artifact(
    dataset: pd.DataFrame,
    model_path: Path,
    *,
    retrain: bool,
) -> ModelArtifact:
    from src.model import ModelArtifact, load_model, save_model, train_model_artifact

    if model_path.exists() and not retrain:
        artifact = load_model(model_path)
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(
                f"Artifact at {model_path} is not a ModelArtifact. "
                "Run with --retrain to replace it."
            )
        print(f"Loaded model artifact: {model_path}")
        return artifact

    artifact = train_model_artifact(dataset)
    saved_path = save_model(artifact, model_path)
    print(f"Saved model artifact: {saved_path}")
    return artifact


def _print_predictions(
    predictions: pd.DataFrame, customer_unique_id: str, threshold: float
) -> None:
    if predictions.empty:
        print(f"No delivered orders found for customer_unique_id={customer_unique_id!r}.")
        return

    print(
        f"Top {len(predictions)} late-delivery risk predictions "
        f"for customer_unique_id={customer_unique_id} (threshold={threshold:.3f}):"
    )
    for i, row in enumerate(predictions.itertuples(index=False), start=1):
        print(
            f"{i}. order_id={row.order_id} | risk={row.late_delivery_risk:.2%} | "
            f"predicted_late={row.predicted_late} | actual_late={row.late}"
        )


def main() -> None:
    from src.data_loader import load_delivery_dataset
    from src.model import score_customer_orders

    args = parse_args()

    dataset = load_delivery_dataset(args.data_dir)
    artifact = _load_or_train_artifact(
        dataset,
        args.model_path,
        retrain=args.retrain,
    )
    predictions = score_customer_orders(
        artifact,
        dataset,
        customer_unique_id=args.customer_unique_id,
        top_k=args.top_k,
    )
    _print_predictions(predictions, args.customer_unique_id, artifact.threshold)


if __name__ == "__main__":
    main()
