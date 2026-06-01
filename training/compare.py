"""Automated champion selection for trained model candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelRun:
    model_name: str
    estimator: Any
    metrics: dict[str, float]
    params: dict[str, Any]


def select_champion(runs: list[ModelRun], metric: str = "roc_auc", minimum_score: float | None = None) -> ModelRun:
    if not runs:
        raise ValueError("No model runs supplied for champion selection.")

    champion = max(runs, key=lambda run: run.metrics[metric])
    if minimum_score is not None and champion.metrics[metric] < minimum_score:
        raise ValueError(
            f"Best model {champion.model_name} scored {champion.metrics[metric]:.4f}, below {minimum_score:.4f}."
        )
    return champion

