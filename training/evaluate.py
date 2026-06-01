"""Model evaluation utilities for churn classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
    )


def business_kpis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    save_offer_cost: float = 25.0,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    true_positives = int(((y_true == 1) & (y_pred == 1)).sum())
    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    missed_churners = int(((y_true == 1) & (y_pred == 0)).sum())

    return {
        "customers_flagged_for_retention": float(y_pred.sum()),
        "true_churners_captured": float(true_positives),
        "false_positive_outreach": float(false_positives),
        "missed_churners": float(missed_churners),
        "estimated_retention_spend": float(y_pred.sum() * save_offer_cost),
    }
