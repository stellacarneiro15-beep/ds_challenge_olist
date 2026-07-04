# Olist Delivery Risk Insights

Source: Olist CSVs in `data/`, using delivered orders with a known customer delivery date and estimated delivery date. Target follows `src/features.py`: `late = delivered_customer_date > estimated_delivery_date`. Analysis base is 96,470 delivered orders from 2016-09-15 to 2018-08-29.

## 1. Late delivery is uncommon, but customer impact is severe

Only 7,826 of 96,470 delivered orders were late, an 8.1% late rate. The median late order arrived 5.8 days after the promised date and took 29.2 days from purchase to delivery, versus 9.7 days for on-time orders. On-time orders were typically delivered 12.3 days before the estimate.

Reviews show the business cost clearly: late orders averaged 2.57 stars, while on-time orders averaged 4.29 stars. Low reviews, defined as 1-2 stars, appeared on 52.8% of late orders versus 9.1% of on-time orders.

## 2. Geography is the strongest operational pattern

Cross-state fulfillment was materially riskier: 9.3% late versus 6.1% when customer and seller were in the same state. Distance shows the same shape. Orders in the longest-distance quartile, above 798 km, were 10.3% late, compared with 6.3% in the shortest-distance quartile, below 186 km.

The highest-risk customer states with at least 500 delivered orders were MA at 19.7%, CE at 15.3%, BA at 14.0%, and RJ at 13.5%. Lower-risk large states included PR at 5.0%, MG at 5.6%, and SP at 5.9%.

## 3. Promise setting matters as much as item complexity

Multi-item and multi-seller baskets were not the main source of lateness in this dataset. Single-item orders were 8.3% late, while multi-item orders were 6.5% late; multi-seller orders were rare and only 1.4% late.

The customer-facing delivery estimate is more informative. Orders promised in 18.3-23.2 days were 9.7% late, while orders promised in more than 28.4 days were 5.8% late. Longer promises appear to absorb logistics variability, especially for distant routes.

## 4. Risk varies over time and can be ranked, not perfectly predicted

Late rates spiked in specific periods: March 2018 reached 21.4%, February 2018 reached 16.0%, and November 2017 reached 14.3% among months with more than 1,000 delivered orders. Monitoring drift by month should be part of any operations workflow.

A chronological holdout model trained on the first 80% of orders and tested on the latest 20% achieved ROC-AUC 0.718, PR-AUC 0.099, and F1 0.172 at a 0.378 threshold. That is useful for prioritizing outreach or carrier review, but the low base rate means predictions should be treated as a triage signal rather than an automated SLA decision.
