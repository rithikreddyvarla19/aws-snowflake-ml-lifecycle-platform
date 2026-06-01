"""AWS Glue compatible PySpark ETL for churn feature engineering."""

from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_churn_features(customers: DataFrame, subscriptions: DataFrame, events: DataFrame) -> DataFrame:
    event_features = (
        events.groupBy("customer_id")
        .pivot("event_type")
        .agg(F.count("event_id"))
        .fillna(0)
    )

    engagement = events.groupBy("customer_id").agg(
        F.count("event_id").alias("total_events"),
        F.countDistinct(F.to_date("event_timestamp")).alias("active_days"),
        F.max("event_timestamp").alias("last_event_at"),
    )

    latest_subscription_start = subscriptions.groupBy("customer_id").agg(
        F.max("subscription_start").alias("latest_subscription_start")
    )
    latest_subscription = (
        subscriptions.join(
            latest_subscription_start,
            (subscriptions.customer_id == latest_subscription_start.customer_id)
            & (subscriptions.subscription_start == latest_subscription_start.latest_subscription_start),
            "inner",
        )
        .select(
            subscriptions.customer_id,
            "billing_interval",
            "auto_renew",
            "subscription_status",
        )
    )

    features = (
        customers.join(latest_subscription, "customer_id", "left")
        .join(event_features, "customer_id", "left")
        .join(engagement, "customer_id", "left")
        .fillna(
            0,
            subset=["login", "support_ticket", "feature_use", "payment_failed", "total_events", "active_days"],
        )
        .withColumn("days_since_signup", F.datediff(F.current_date(), F.to_date("signup_date")))
        .withColumn("support_ticket_rate", F.col("support_ticket") / F.greatest(F.col("total_events"), F.lit(1)))
        .withColumn("payment_failure_rate", F.col("payment_failed") / F.greatest(F.col("total_events"), F.lit(1)))
        .withColumn(
            "mrr_per_active_day",
            F.col("monthly_recurring_revenue") / F.greatest(F.col("active_days"), F.lit(1)),
        )
        .withColumn("feature_created_at", F.current_timestamp())
    )
    return features


def run_job(spark: SparkSession, input_root: str, output_uri: str) -> None:
    customers = spark.read.option("header", True).option("inferSchema", True).csv(f"{input_root}/customers.csv")
    subscriptions = spark.read.option("header", True).option("inferSchema", True).csv(f"{input_root}/subscriptions.csv")
    events = spark.read.option("header", True).option("inferSchema", True).csv(f"{input_root}/product_events.csv")

    features = build_churn_features(customers, subscriptions, events)
    features.write.mode("overwrite").format("parquet").save(output_uri)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build churn features with PySpark.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-uri", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    spark_session = SparkSession.builder.appName("churn-feature-engineering").getOrCreate()
    run_job(spark_session, args.input_root, args.output_uri)
