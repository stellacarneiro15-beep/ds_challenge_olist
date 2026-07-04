import numpy as np
import pandas as pd

from src.data_loader import OlistData
from src.evaluate import evaluate_classifier, time_based_split
from src.features import (
    CATEGORICAL_FEATURES,
    CUSTOMER_ID_COLUMN,
    FEATURE_COLUMNS,
    ORDER_ID_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    build_delivery_dataset,
)
from src.model import (
    ModelArtifact,
    build_model_pipeline,
    load_model,
    save_model,
    score_customer_orders,
    select_operating_threshold,
    train_model_artifact,
)

NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


class EstimatedDaysRiskModel:
    """Tiny test double: higher promised window -> higher late risk."""

    def predict_proba(self, X):
        probabilities = np.clip(X["estimated_delivery_days"].to_numpy(dtype=float) / 100.0, 0, 1)
        return np.column_stack([1 - probabilities, probabilities])


class FitEstimatedDaysRiskModel(EstimatedDaysRiskModel):
    """Fittable version of the deterministic risk model used by artifact training."""

    def fit(self, X, y):
        return self


def _toy_olist() -> OlistData:
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "delivered", "canceled"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01", "2018-02-01", "2018-03-01"]
            ),
            "order_approved_at": pd.to_datetime(["2018-01-01", "2018-02-01", pd.NaT]),
            "order_delivered_carrier_date": pd.to_datetime(
                ["2018-01-02", "2018-02-02", pd.NaT]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2018-01-08", "2018-02-20", pd.NaT]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2018-01-10", "2018-02-10", "2018-03-10"]
            ),
        }
    )
    order_items = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "order_item_id": [1, 1, 1],
            "product_id": ["p1", "p2", "p1"],
            "seller_id": ["s1", "s2", "s1"],
            "shipping_limit_date": pd.to_datetime(
                ["2018-01-05", "2018-02-05", "2018-03-05"]
            ),
            "price": [100.0, 50.0, 100.0],
            "freight_value": [10.0, 25.0, 10.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2"],
            "product_category_name": ["toys", "electronics"],
            "product_name_lenght": [40, 55],
            "product_description_lenght": [500, 900],
            "product_photos_qty": [2, 4],
            "product_weight_g": [500, 1500],
            "product_length_cm": [20, 30],
            "product_height_cm": [10, 15],
            "product_width_cm": [15, 20],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "customer_unique_id": ["u1", "u2", "u3"],
            "customer_zip_code_prefix": [1000, 5000, 1000],
            "customer_city": ["sp", "rj", "sp"],
            "customer_state": ["SP", "RJ", "SP"],
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["s1", "s2"],
            "seller_zip_code_prefix": [1000, 9000],
            "seller_city": ["sp", "ba"],
            "seller_state": ["SP", "BA"],
        }
    )
    order_payments = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "payment_sequential": [1, 1, 1],
            "payment_type": ["credit_card", "boleto", "credit_card"],
            "payment_installments": [3, 1, 1],
            "payment_value": [110.0, 75.0, 110.0],
        }
    )
    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1000, 5000, 9000],
            "geolocation_lat": [-23.5, -22.9, -12.9],
            "geolocation_lng": [-46.6, -43.2, -38.5],
            "geolocation_city": ["sp", "rj", "ba"],
            "geolocation_state": ["SP", "RJ", "BA"],
        }
    )
    return OlistData(
        customers=customers,
        orders=orders,
        order_items=order_items,
        order_payments=order_payments,
        order_reviews=pd.DataFrame({"order_id": [], "review_score": []}),
        products=products,
        sellers=sellers,
        geolocation=geolocation,
        category_translation=pd.DataFrame(
            {"product_category_name": [], "product_category_name_english": []}
        ),
    )


def test_build_delivery_dataset_shape_and_target():
    df = build_delivery_dataset(_toy_olist())

    assert len(df) == 2
    assert set([ORDER_ID_COLUMN, CUSTOMER_ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]).issubset(df.columns)
    assert df[TARGET_COLUMN].tolist() == [0, 1]
    assert df[ORDER_ID_COLUMN].tolist() == ["o1", "o2"]
    assert df[CUSTOMER_ID_COLUMN].tolist() == ["c1", "c2"]


def test_build_delivery_dataset_features_are_leakage_safe():
    df = build_delivery_dataset(_toy_olist())

    for leaked in [
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "order_approved_at",
    ]:
        assert leaked not in df.columns
    assert (df["estimated_delivery_days"] > 0).all()
    assert df["same_state"].tolist() == [1, 0]


def test_production_pipeline_fits_and_scores():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    df["customer_state"] = "SP"
    df["seller_state"] = "SP"
    df["payment_type"] = "credit_card"
    y = pd.Series([0, 1] * (n // 2))

    model = build_model_pipeline()
    model.fit(df[FEATURE_COLUMNS], y)
    probabilities = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]

    assert probabilities.shape == (n,)
    assert np.isfinite(probabilities).all()


def test_score_customer_orders_returns_top_k_riskiest_orders():
    df = pd.DataFrame({col: [0.0, 0.0, 0.0] for col in FEATURE_COLUMNS})
    df["estimated_delivery_days"] = [10.0, 80.0, 50.0]
    df["customer_state"] = ["SP", "SP", "RJ"]
    df["seller_state"] = ["SP", "RJ", "RJ"]
    df["payment_type"] = ["credit_card", "boleto", "credit_card"]
    df[ORDER_ID_COLUMN] = ["low", "high", "other_customer"]
    df[CUSTOMER_ID_COLUMN] = ["c1", "c1", "c2"]
    df[TARGET_COLUMN] = [0, 1, 0]
    df[TIME_COLUMN] = pd.date_range("2018-01-01", periods=3, freq="h")

    artifact = ModelArtifact(
        model_name="test_model",
        estimator=EstimatedDaysRiskModel(),
        threshold=0.6,
    )

    scored = score_customer_orders(artifact, df, customer_id="c1", top_k=1)

    assert scored[ORDER_ID_COLUMN].tolist() == ["high"]
    assert scored["late_delivery_risk"].tolist() == [0.8]
    assert scored["predicted_late"].tolist() == [1]


def test_select_operating_threshold_uses_validation_f1():
    validation = pd.DataFrame({col: [0.0, 0.0, 0.0, 0.0] for col in FEATURE_COLUMNS})
    validation["estimated_delivery_days"] = [10.0, 20.0, 30.0, 40.0]
    validation[TARGET_COLUMN] = [0, 0, 1, 1]

    threshold = select_operating_threshold(EstimatedDaysRiskModel(), validation)

    assert threshold == 0.3


def test_train_model_artifact_carries_validation_threshold(monkeypatch):
    frame = _synthetic_modeling_frame(n=20)
    frame["estimated_delivery_days"] = np.linspace(10.0, 90.0, len(frame))
    frame[TARGET_COLUMN] = [0, 1] * 8 + [0, 0, 1, 1]

    monkeypatch.setattr(
        "src.model.build_model_pipeline",
        lambda scale_pos_weight=1.0: FitEstimatedDaysRiskModel(),
    )

    artifact = train_model_artifact(frame)

    assert np.isclose(artifact.threshold, 0.8578947368421052)


def test_model_artifact_roundtrip(tmp_path):
    artifact = ModelArtifact(
        model_name="test_model",
        estimator=EstimatedDaysRiskModel(),
        threshold=0.4,
    )
    X = pd.DataFrame({"estimated_delivery_days": [10.0, 50.0]})

    path = save_model(artifact, tmp_path / "model.pkl")
    loaded = load_model(path)

    assert isinstance(loaded, ModelArtifact)
    assert loaded.model_name == "test_model"
    assert loaded.threshold == 0.4
    assert np.allclose(loaded.estimator.predict_proba(X)[:, 1], [0.1, 0.5])


def _synthetic_modeling_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A modeling-table-shaped frame with both classes present in every fold."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({col: rng.normal(size=n) for col in NUMERIC_FEATURES})
    for col in CATEGORICAL_FEATURES:
        frame[col] = rng.choice(["SP", "RJ"], size=n)
    # Alternating labels guarantee both classes in any contiguous chronological fold.
    frame[TARGET_COLUMN] = np.tile([0, 1], n // 2)
    frame[ORDER_ID_COLUMN] = [f"o{i}" for i in range(n)]
    frame[CUSTOMER_ID_COLUMN] = rng.choice(["c1", "c2", "c3"], size=n)
    frame[TIME_COLUMN] = pd.date_range("2018-01-01", periods=n, freq="h")
    return frame


def test_time_based_split_is_chronological_and_sized():
    frame = _synthetic_modeling_frame(n=100)
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)

    train, test = time_based_split(shuffled, test_frac=0.2)

    assert len(train) == 80
    assert len(test) == 20
    # Every train timestamp precedes every test timestamp.
    assert train[TIME_COLUMN].max() < test[TIME_COLUMN].min()


def test_evaluate_classifier_returns_expected_metric_keys():
    y_true = pd.Series([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3])

    metrics = evaluate_classifier(y_true, y_proba)

    assert set(metrics) == {"pr_auc", "roc_auc", "brier", "f1", "threshold", "accuracy"}
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
