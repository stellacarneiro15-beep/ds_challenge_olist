# Olist Delivery Risk Insights

Source: Olist CSVs in `data/`, using delivered orders with a known customer delivery date and estimated delivery date. Reference files: `src/features.py` defines the leakage-safe target/features, `src/model.py` defines the production model pipeline, and the notebooks below contain the reported statistics.

## 1. Late delivery is uncommon, but customer impact is severe

Only 7,826 of 96,470 delivered orders were late, an 8.1% late rate. A naive model that always predicts "on time" would still be 91.9% accurate, but would catch zero late deliveries, so accuracy is not the right main metric.

Reviews show the business cost clearly: late orders averaged 2.57 stars across 7,700 reviewed late deliveries, while on-time orders averaged 4.29 stars across 88,661 reviewed on-time deliveries. One-star reviews appeared on 46.2% of late orders versus 6.6% of on-time orders.

Where to find it: `notebooks/02_delivery_feature_exploration.ipynb`, section `1. Target balance`, insight cell; `notebooks/01_descriptive_analytics.ipynb`, section `3.1 Does delivery lateness pull the star rating down?`, late-delivery review table.

## 2. Geography is the strongest operational pattern

The categorical feature table shows strong geographic concentration. Among customer states with at least 100 orders, AL had a 23.9% late rate, MA 19.7%, PI 16.0%, CE 15.3%, BA 14.0%, and RJ 13.5%, all above the 8.1% overall late rate.

The same table also flags seller geography: seller_state MA had a 23.2% late rate, while high-volume seller_state SP was 8.8%. Payment type is less separated: boleto was 8.9%, debit card and credit card were both 8.0%, and voucher was 7.0%.

Where to find it: `notebooks/02_delivery_feature_exploration.ipynb`, section `6. Categorical feature signal`, categorical summary table; same table is repeated in `notebooks/02_model.ipynb`, section `6. Categorical feature signal`.

## 3. Promise accuracy matters more than raw shipping time

The descriptive notebook compares review score against both absolute wait time and lateness versus the promised ETA. Both relationships are negative and statistically significant, with about -0.045 stars per extra day from purchase and about -0.034 stars per day late versus ETA.

The notebook's key interpretation is a cliff at the promised date: early or on-time orders sit around 4.2-4.3 stars, but scores drop to about 3.5 when 0-5 days late, about 1.9 when 5-10 days late, and about 1.7 beyond that.

Where to find it: `notebooks/01_descriptive_analytics.ipynb`, `Insight 4 - the broken promise that hurts`.

## 4. The coded model is a leakage-safe risk ranker

The model notebook uses a chronological train/validation/test split: 61,740 train rows, 15,436 validation rows, and 19,294 test rows. The test late rate is 0.053, so the no-skill PR-AUC floor on that period is about 0.053 rather than the overall 8.1%.

In `src/model.py`, the shipped pipeline trains a scikit-learn `HistGradientBoostingClassifier` after numeric imputation, categorical imputation, and one-hot encoding. The feature table comes from `src/features.py` and excludes post-purchase carrier/customer delivery timestamps to avoid leakage.

The benchmark tested weighted logistic regression, XGBoost, LightGBM, and HistGradientBoosting. XGBoost had the strongest test PR-AUC in the displayed benchmark at 0.139, versus HistGradientBoosting at 0.118, but XGBoost's thresholded F1 was only 0.021. HistGradientBoosting was promoted because the notebook frames it as simpler to ship in scikit-learn, serializable, and better behaved for the CLI workflow.

On the held-out test period, using the validation-selected artifact threshold, the shipped gradient-boosting model achieved PR-AUC 0.104, ROC-AUC 0.723, Brier score 0.142, and F1 0.094 at threshold 0.673. The no-skill PR-AUC baseline in that test period is about the late rate, 5.3%, so the model ranks risk better than chance. Still, the hard threshold is conservative and misses many late orders, so the right deployment threshold should depend on the business action: monitoring, seller contact, ETA adjustment, shipping upgrade, or proactive customer messaging.

Where to find it: `notebooks/02_model.ipynb`, sections `Chronological Train / Validation / Test Split`, `Fit and Evaluate`, `Reading the Benchmark`, and `Production API (promoted to src/)`.
