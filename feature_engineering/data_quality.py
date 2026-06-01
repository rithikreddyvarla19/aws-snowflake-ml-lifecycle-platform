"""Reusable data quality checks for raw and feature datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    check_name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class DataQualityValidator:
    """Small validation engine that works with pandas-like DataFrames."""

    def __init__(self, dataframe: Any) -> None:
        self.df = dataframe

    def not_null(self, columns: list[str]) -> ValidationResult:
        missing = {col: int(self.df[col].isna().sum()) for col in columns}
        return ValidationResult("not_null", all(count == 0 for count in missing.values()), missing)

    def unique_key(self, key: str) -> ValidationResult:
        duplicate_count = int(self.df.duplicated(subset=[key]).sum())
        return ValidationResult("unique_key", duplicate_count == 0, {"key": key, "duplicate_count": duplicate_count})

    def accepted_values(self, column: str, values: set[Any]) -> ValidationResult:
        observed = set(self.df[column].dropna().unique())
        unexpected = sorted(observed - values)
        return ValidationResult(
            "accepted_values",
            len(unexpected) == 0,
            {"column": column, "unexpected_values": unexpected},
        )

    def numeric_range(
        self,
        column: str,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> ValidationResult:
        series = self.df[column].dropna()
        below = int((series < minimum).sum()) if minimum is not None else 0
        above = int((series > maximum).sum()) if maximum is not None else 0
        return ValidationResult(
            "numeric_range",
            below == 0 and above == 0,
            {"column": column, "below_min": below, "above_max": above, "minimum": minimum, "maximum": maximum},
        )

    def run_default_customer_checks(self) -> list[ValidationResult]:
        return [
            self.not_null(["customer_id", "signup_date", "plan_type", "monthly_recurring_revenue"]),
            self.unique_key("customer_id"),
            self.accepted_values("plan_type", {"basic", "pro", "enterprise"}),
            self.numeric_range("monthly_recurring_revenue", minimum=0),
        ]


def assert_quality(results: list[ValidationResult]) -> None:
    failed = [result for result in results if not result.passed]
    if failed:
        summary = {result.check_name: result.details for result in failed}
        raise ValueError(f"Data quality validation failed: {summary}")
