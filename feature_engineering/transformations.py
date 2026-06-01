"""Pandas feature transformations used for local training and unit tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_sample_sources(sample_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(sample_dir / "customers.csv", parse_dates=["signup_date"])
    subscriptions = pd.read_csv(
        sample_dir / "subscriptions.csv",
        parse_dates=["subscription_start", "subscription_end"],
    )
    events = pd.read_csv(sample_dir / "product_events.csv", parse_dates=["event_timestamp"])
    return customers, subscriptions, events


def build_customer_churn_features(
    customers: pd.DataFrame,
    subscriptions: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build model-ready customer churn features from CRM, billing, and events."""

    event_features = (
        events.assign(event_count=1)
        .pivot_table(
            index="customer_id",
            columns="event_type",
            values="event_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    event_features.columns.name = None

    engagement = (
        events.groupby("customer_id")
        .agg(
            total_events=("event_id", "count"),
            active_days=("event_timestamp", lambda values: values.dt.date.nunique()),
            last_event_at=("event_timestamp", "max"),
        )
        .reset_index()
    )

    latest_subscription = (
        subscriptions.sort_values("subscription_start")
        .groupby("customer_id", as_index=False)
        .tail(1)[["customer_id", "billing_interval", "auto_renew", "subscription_status"]]
    )

    features = (
        customers.merge(latest_subscription, on="customer_id", how="left")
        .merge(event_features, on="customer_id", how="left")
        .merge(engagement, on="customer_id", how="left")
    )

    event_columns = [
        column
        for column in ["login", "support_ticket", "feature_use", "payment_failed"]
        if column in features
    ]
    features[event_columns] = features[event_columns].fillna(0)
    features["total_events"] = features["total_events"].fillna(0)
    features["active_days"] = features["active_days"].fillna(0)
    features["days_since_signup"] = (pd.Timestamp.utcnow().tz_localize(None) - features["signup_date"]).dt.days
    features["support_ticket_rate"] = features.get("support_ticket", 0) / features["total_events"].clip(lower=1)
    features["payment_failure_rate"] = features.get("payment_failed", 0) / features["total_events"].clip(lower=1)
    features["mrr_per_active_day"] = features["monthly_recurring_revenue"] / features["active_days"].clip(lower=1)
    features["feature_created_at"] = pd.Timestamp.utcnow()

    return features.fillna(
        {
            "billing_interval": "unknown",
            "auto_renew": False,
            "subscription_status": "unknown",
        }
    )


def load_training_frame(sample_dir: Path) -> pd.DataFrame:
    customers, subscriptions, events = load_sample_sources(sample_dir)
    return build_customer_churn_features(customers, subscriptions, events)
