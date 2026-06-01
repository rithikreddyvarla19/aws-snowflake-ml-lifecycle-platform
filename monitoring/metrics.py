"""Monitoring metrics for labels, predictions, and business KPIs."""

from __future__ import annotations

import numpy as np


def accuracy_degradation(
    reference_accuracy: float,
    current_accuracy: float,
    threshold: float = 0.05,
) -> dict[str, float | bool]:
    drop = reference_accuracy - current_accuracy
    return {
        "reference_accuracy": float(reference_accuracy),
        "current_accuracy": float(current_accuracy),
        "absolute_drop": float(drop),
        "degraded": bool(drop >= threshold),
    }


def monitoring_scorecard(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    from training.evaluate import business_kpis, classification_metrics

    y_pred = (y_score >= 0.5).astype(int)
    metrics = classification_metrics(y_true, y_pred, y_score).as_dict()
    metrics.update(business_kpis(y_true, y_score))
    return metrics
