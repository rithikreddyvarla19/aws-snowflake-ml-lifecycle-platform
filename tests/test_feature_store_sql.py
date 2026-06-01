from __future__ import annotations

from feature_engineering.feature_store import SnowflakeFeatureStore


def test_feature_store_merge_sql_contains_contract() -> None:
    store = SnowflakeFeatureStore(
        offline_table="CHURN_ML.FEATURE_STORE.CUSTOMER_CHURN_FEATURES",
        online_table="CHURN_ML.FEATURE_STORE.CUSTOMER_CHURN_FEATURES_ONLINE",
    )
    sql = store.merge_sql("TEMP_CHURN_FEATURES")
    assert "MERGE INTO CHURN_ML.FEATURE_STORE.CUSTOMER_CHURN_FEATURES" in sql
    assert "ON target.customer_id = source.customer_id" in sql
    assert "payment_failure_rate" in sql

