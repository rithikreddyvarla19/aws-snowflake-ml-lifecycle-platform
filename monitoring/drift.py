"""Drift and prediction distribution shift detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftResult:
    feature: str
    psi: float
    drifted: bool


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))
    if len(breakpoints) <= 2:
        breakpoints = np.linspace(expected.min(), expected.max() + epsilon, buckets + 1)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), epsilon, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), epsilon, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def detect_feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: list[str],
    threshold: float = 0.2,
) -> list[DriftResult]:
    results = []
    for feature in numeric_features:
        psi = population_stability_index(reference[feature], current[feature])
        results.append(DriftResult(feature=feature, psi=psi, drifted=psi >= threshold))
    return results


def prediction_distribution_shift(
    reference_scores: pd.Series,
    current_scores: pd.Series,
    threshold: float = 0.15,
) -> dict[str, float | bool]:
    reference_mean = float(reference_scores.mean())
    current_mean = float(current_scores.mean())
    absolute_shift = abs(current_mean - reference_mean)
    return {
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "absolute_shift": absolute_shift,
        "shifted": absolute_shift >= threshold,
    }

