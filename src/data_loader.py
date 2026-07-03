"""Load and join the Olist Brazilian e-commerce CSVs.

Files are expected in DATA_DIR (default: "data"), using the original
Kaggle file names, e.g. olist_orders_dataset.csv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "data")

FILES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


@dataclass
class OlistData:
    """Container for the raw Olist tables."""

    customers: pd.DataFrame
    orders: pd.DataFrame
    order_items: pd.DataFrame
    order_payments: pd.DataFrame
    order_reviews: pd.DataFrame
    products: pd.DataFrame
    sellers: pd.DataFrame
    geolocation: pd.DataFrame
    category_translation: pd.DataFrame


def load_raw_tables(data_dir: str | Path = DEFAULT_DATA_DIR) -> OlistData:
    """Read every Olist CSV into a DataFrame."""
    data_dir = Path(data_dir)
    missing = [name for name in FILES.values() if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}. Set DATA_DIR or pass data_dir explicitly."
        )

    tables = {key: pd.read_csv(data_dir / filename) for key, filename in FILES.items()}
    return OlistData(**tables)


def load_delivery_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Convenience wrapper: load raw tables and build the modeling table.

    Returns the order-level, leakage-safe feature/target table produced by
    ``src.features.build_delivery_dataset`` (imported lazily to avoid a
    circular import, since ``features`` depends on ``OlistData``).
    """
    from src.features import build_delivery_dataset

    return build_delivery_dataset(load_raw_tables(data_dir))
