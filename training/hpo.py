"""Hyperparameter optimization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from training.models import ModelSpec


@dataclass(frozen=True)
class HpoResult:
    model_name: str
    estimator: Any
    best_params: dict[str, Any]
    best_score: float


def run_grid_search(spec: ModelSpec, x_train: Any, y_train: Any, metric: str = "roc_auc", folds: int = 3) -> HpoResult:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=spec.estimator,
        param_grid=spec.param_grid,
        scoring=metric,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    return HpoResult(spec.name, search.best_estimator_, search.best_params_, float(search.best_score_))

