from __future__ import annotations

import pandas as pd

from feature_engineering.data_quality import DataQualityValidator, assert_quality


def test_default_customer_quality_checks_pass() -> None:
    frame = pd.DataFrame(
        [
            {
                "customer_id": "C001",
                "signup_date": "2024-01-01",
                "plan_type": "pro",
                "monthly_recurring_revenue": 99.0,
            },
            {
                "customer_id": "C002",
                "signup_date": "2024-01-02",
                "plan_type": "basic",
                "monthly_recurring_revenue": 29.0,
            },
        ]
    )
    results = DataQualityValidator(frame).run_default_customer_checks()
    assert all(result.passed for result in results)
    assert_quality(results)


def test_unique_key_detects_duplicates() -> None:
    frame = pd.DataFrame({"customer_id": ["C001", "C001"], "plan_type": ["pro", "pro"]})
    result = DataQualityValidator(frame).unique_key("customer_id")
    assert not result.passed
    assert result.details["duplicate_count"] == 1

