"""Train, evaluate, compare, and register customer churn models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from feature_engineering.transformations import load_training_frame
from training.compare import ModelRun, select_champion
from training.evaluate import business_kpis, classification_metrics
from training.hpo import run_grid_search
from training.model_registry import log_sklearn_model
from training.models import CATEGORICAL_FEATURES, NUMERIC_FEATURES, candidate_specs

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _positive_scores(estimator: Any, x_frame: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_frame)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_frame)
        return 1 / (1 + np.exp(-scores))
    return estimator.predict(x_frame)


def train_from_config(config_path: Path, sample_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training_config = config["training"]
    frame = load_training_frame(sample_dir)

    x = frame[FEATURE_COLUMNS]
    y = frame[training_config["target_column"]].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=training_config["test_size"],
        random_state=training_config["random_state"],
        stratify=y,
    )

    runs: list[ModelRun] = []
    for spec in candidate_specs(training_config["candidates"]):
        hpo_result = run_grid_search(spec, x_train, y_train, metric=training_config["metric_to_optimize"], folds=3)
        y_score = _positive_scores(hpo_result.estimator, x_test)
        y_pred = (y_score >= 0.5).astype(int)
        metrics = classification_metrics(y_test.to_numpy(), y_pred, y_score).as_dict()
        metrics.update(business_kpis(y_test.to_numpy(), y_score))
        runs.append(ModelRun(spec.name, hpo_result.estimator, metrics, hpo_result.best_params))

    champion = select_champion(
        runs,
        metric=training_config["metric_to_optimize"],
        minimum_score=training_config.get("champion_min_roc_auc"),
    )
    registry_id = log_sklearn_model(
        champion.estimator,
        model_name=f"churn_{champion.model_name}",
        metrics=champion.metrics,
        params=champion.params,
        artifact_dir=artifact_dir,
    )

    report = {
        "champion": champion.model_name,
        "registry_id_or_artifact": registry_id,
        "runs": [
            {"model_name": run.model_name, "metrics": run.metrics, "params": run.params}
            for run in sorted(runs, key=lambda item: item.metrics[training_config["metric_to_optimize"]], reverse=True)
        ],
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register churn models.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--sample-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/models"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(train_from_config(args.config, args.sample_dir, args.artifact_dir), indent=2))

