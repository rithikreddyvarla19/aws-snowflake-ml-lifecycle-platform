"""MLflow model registry helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def log_sklearn_model(
    estimator: Any,
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any],
    artifact_dir: Path,
) -> str:
    """Log a model to MLflow when available and always persist a local artifact."""

    import joblib

    artifact_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = artifact_dir / f"{model_name}.joblib"
    joblib.dump(estimator, local_model_path)

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return str(local_model_path)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(estimator, artifact_path="model", registered_model_name=model_name)
        mlflow.log_artifact(str(local_model_path))
        return mlflow.active_run().info.run_id

