"""Snowflake-backed offline/online feature store helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    dtype: str
    description: str


CUSTOMER_CHURN_FEATURES = [
    FeatureDefinition("customer_id", "STRING", "Stable customer identifier"),
    FeatureDefinition("monthly_recurring_revenue", "FLOAT", "Current subscription MRR"),
    FeatureDefinition("days_since_signup", "INTEGER", "Customer tenure in days"),
    FeatureDefinition("support_ticket_rate", "FLOAT", "Share of support events among all events"),
    FeatureDefinition("payment_failure_rate", "FLOAT", "Share of failed payment events among all events"),
    FeatureDefinition("mrr_per_active_day", "FLOAT", "Revenue normalized by active product days"),
    FeatureDefinition("feature_created_at", "TIMESTAMP_NTZ", "Feature calculation timestamp"),
]


class SnowflakeFeatureStore:
    """DDL and merge contract for a Snowflake feature store table."""

    def __init__(self, offline_table: str, online_table: str, entity_key: str = "customer_id") -> None:
        self.offline_table = offline_table
        self.online_table = online_table
        self.entity_key = entity_key

    def create_table_sql(self, table_name: str | None = None) -> str:
        target = table_name or self.offline_table
        columns = ",\n  ".join(f"{feature.name} {feature.dtype}" for feature in CUSTOMER_CHURN_FEATURES)
        return f"CREATE TABLE IF NOT EXISTS {target} (\n  {columns},\n  PRIMARY KEY ({self.entity_key})\n);"

    def merge_sql(self, source_table: str, target_table: str | None = None) -> str:
        target = target_table or self.offline_table
        update_columns = [feature.name for feature in CUSTOMER_CHURN_FEATURES if feature.name != self.entity_key]
        assignments = ", ".join(f"{column} = source.{column}" for column in update_columns)
        insert_columns = ", ".join(feature.name for feature in CUSTOMER_CHURN_FEATURES)
        insert_values = ", ".join(f"source.{feature.name}" for feature in CUSTOMER_CHURN_FEATURES)

        return f"""
MERGE INTO {target} AS target
USING {source_table} AS source
ON target.{self.entity_key} = source.{self.entity_key}
WHEN MATCHED THEN UPDATE SET {assignments}
WHEN NOT MATCHED THEN INSERT ({insert_columns})
VALUES ({insert_values});
""".strip()

