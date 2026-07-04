"""Feature engineering for the on-time delivery prediction task.

Target
------
`late` = 1 if an order was delivered *after* the customer-facing estimated
delivery date, else 0. We keep only orders with `order_status == "delivered"`
that have both a real delivery date and an estimated date.

We treat `order_estimated_delivery_date` as a hard deadline (the "limit"): the
order must be delivered *up to* that instant. Since the estimate is stored at
midnight (00:00) of the promised day, any delivery after that instant counts as
late, including deliveries later on the same calendar day. This is a deliberate
choice to read the estimate as "deliver by this date/time", not "any time on
this date is fine".

Leakage policy
--------------
Every feature must be knowable at **purchase time**, because the intended use
is to flag at-risk orders early enough for operations to intervene. Concretely
we only use:

- `order_purchase_timestamp` and `order_estimated_delivery_date` (the promise
  shown to the customer at checkout),
- `shipping_limit_date` (the seller's ship-by deadline, fixed at order time),
- order-item, product, payment and geography attributes.

We deliberately DO NOT use `order_approved_at`, `order_delivered_carrier_date`
or `order_delivered_customer_date` (the last one *is* the target), since they
are only known after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import OlistData

# Feature groups, consumed by the model pipelines in model.py.
NUMERIC_FEATURES: list[str] = [
    "estimated_delivery_days",
    "shipping_limit_days",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "n_items",
    "n_distinct_products",
    "n_sellers",
    "price_total",
    "freight_total",
    "freight_ratio",
    "product_weight_g_total",
    "product_volume_cm3_total",
    "product_photos_qty_avg",
    "product_description_length_avg",
    "payment_installments_max",
    "payment_value_total",
    "geo_distance_km",
    "same_state",
]

CATEGORICAL_FEATURES: list[str] = [
    "customer_state",
    "seller_state",
    "payment_type",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "late"
TIME_COLUMN = "order_purchase_timestamp"
ORDER_ID_COLUMN = "order_id"
# `customer_id` is a per-order key in Olist (one row per order), so it cannot be
# used to group a person's orders. `customer_unique_id` is the stable person-level
# identifier and is what scoring/aggregation must key on.
CUSTOMER_ID_COLUMN = "customer_id"
CUSTOMER_UNIQUE_ID_COLUMN = "customer_unique_id"

_EARTH_RADIUS_KM = 6371.0

# Continental Brazil bounding box, used to drop corrupt geolocation rows
# (some Olist lat/lng points fall in the ocean or other continents).
_BRAZIL_LAT_BOUNDS = (-33.75, 5.27)
_BRAZIL_LNG_BOUNDS = (-73.99, -34.79)


def _haversine_km(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Great-circle distance in km between two arrays of lat/lon points."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _zip_centroids(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Mean lat/lng per zip-code prefix (Olist geolocation has many rows/zip).

    Coordinates outside continental Brazil are dropped first, since a handful of
    corrupt rows otherwise inflate `geo_distance_km` to physically impossible
    values (>8,000 km).
    """
    lat_lo, lat_hi = _BRAZIL_LAT_BOUNDS
    lng_lo, lng_hi = _BRAZIL_LNG_BOUNDS
    in_brazil = (
        geolocation["geolocation_lat"].between(lat_lo, lat_hi)
        & geolocation["geolocation_lng"].between(lng_lo, lng_hi)
    )
    return (
        geolocation[in_brazil]
        .groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "mean"), lng=("geolocation_lng", "mean"))
        .reset_index()
    )


def _order_item_features(order_items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-items + product attributes to one row per order."""
    items = order_items.merge(
        products[
            [
                "product_id",
                "product_photos_qty",
                "product_description_lenght",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
        ],
        on="product_id",
        how="left",
    )
    items["product_volume_cm3"] = (
        items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    )
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])

    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_distinct_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        product_weight_g_total=("product_weight_g", lambda s: s.sum(min_count=1)),
        product_volume_cm3_total=("product_volume_cm3", lambda s: s.sum(min_count=1)),
        product_photos_qty_avg=("product_photos_qty", "mean"),
        product_description_length_avg=("product_description_lenght", "mean"),
        shipping_limit_date=("shipping_limit_date", "max"),
    )
    agg["freight_ratio"] = agg["freight_total"] / (agg["price_total"] + 1.0)

    # Primary seller = the seller on the first line item, used for geography.
    primary_seller = (
        items.sort_values("order_item_id")
        .groupby("order_id")["seller_id"]
        .first()
        .rename("primary_seller_id")
    )
    return agg.join(primary_seller)


def _payment_features(order_payments: pd.DataFrame) -> pd.DataFrame:
    """One row per order: total paid, max installments, dominant payment type."""
    agg = order_payments.groupby("order_id").agg(
        payment_value_total=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
    )
    dominant_type = (
        order_payments.sort_values("payment_value", ascending=False)
        .groupby("order_id")["payment_type"]
        .first()
        .rename("payment_type")
    )
    return agg.join(dominant_type)


def build_delivery_dataset(data: OlistData) -> pd.DataFrame:
    """Build the order-level modeling table for late-delivery prediction.

    Returns a DataFrame with FEATURE_COLUMNS, TARGET_COLUMN and TIME_COLUMN,
    one row per delivered order that has both a delivery and estimated date.
    """
    orders = data.orders.copy()
    for col in [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[col] = pd.to_datetime(orders[col])

    delivered = orders[orders["order_status"] == "delivered"].dropna(
        subset=["order_delivered_customer_date", "order_estimated_delivery_date"]
    )

    df = delivered[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
        ]
    ].copy()

    # --- Target ---
    # `order_estimated_delivery_date` is treated as the delivery deadline/limit
    # (stored at midnight of the promised day). A strict `>` comparison flags any
    # delivery after that instant as late, so same-day deliveries after 00:00 also
    # count as late by design.
    df[TARGET_COLUMN] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)

    # --- Purchase-time temporal features ---
    purchase = df["order_purchase_timestamp"]
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - purchase
    ).dt.total_seconds() / 86400.0
    df["purchase_month"] = purchase.dt.month
    df["purchase_dayofweek"] = purchase.dt.dayofweek
    df["purchase_hour"] = purchase.dt.hour

    # --- Order-item / product / payment aggregates ---
    item_feats = _order_item_features(data.order_items, data.products)
    pay_feats = _payment_features(data.order_payments)
    df = df.merge(item_feats, on="order_id", how="left").merge(
        pay_feats, on="order_id", how="left"
    )

    df["shipping_limit_days"] = (
        df["shipping_limit_date"] - purchase.values
    ).dt.total_seconds() / 86400.0

    # --- Geography: customer & primary-seller states + centroid distance ---
    df = df.merge(
        data.customers[
            [
                "customer_id",
                "customer_unique_id",
                "customer_state",
                "customer_zip_code_prefix",
            ]
        ],
        on="customer_id",
        how="left",
    )
    sellers = data.sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]].rename(
        columns={"seller_id": "primary_seller_id"}
    )
    df = df.merge(sellers, on="primary_seller_id", how="left")
    # Leave `same_state` as NaN when either state is unknown, rather than
    # asserting "different state" for a failed join.
    both_states_known = df["customer_state"].notna() & df["seller_state"].notna()
    df["same_state"] = (
        (df["customer_state"] == df["seller_state"]).where(both_states_known).astype("float")
    )

    centroids = _zip_centroids(data.geolocation)
    cust_geo = centroids.rename(
        columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix", "lat": "c_lat", "lng": "c_lng"}
    )
    sell_geo = centroids.rename(
        columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix", "lat": "s_lat", "lng": "s_lng"}
    )
    df = df.merge(cust_geo, on="customer_zip_code_prefix", how="left").merge(
        sell_geo, on="seller_zip_code_prefix", how="left"
    )
    df["geo_distance_km"] = _haversine_km(df["c_lat"], df["c_lng"], df["s_lat"], df["s_lng"])

    keep = [
        ORDER_ID_COLUMN,
        CUSTOMER_ID_COLUMN,
        CUSTOMER_UNIQUE_ID_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]
    return df[keep].sort_values(TIME_COLUMN).reset_index(drop=True)
