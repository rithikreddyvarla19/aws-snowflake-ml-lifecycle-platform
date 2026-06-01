"""Airflow DAG orchestrating ingestion, features, training, deployment, monitoring, and retraining."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}


with DAG(
    dag_id="churn_ml_lifecycle",
    description="Customer churn ML lifecycle on AWS, Snowflake, SageMaker, MLflow, and Airflow.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["churn", "mlops", "sagemaker", "snowflake"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_multi_source_data",
        bash_command="python -m data_ingestion.ingestion_job --config config/config.yaml --raw-root data/raw",
    )

    validate = BashOperator(
        task_id="validate_raw_data",
        bash_command="python -m pytest tests/test_data_quality.py -q",
    )

    build_features = BashOperator(
        task_id="build_feature_store",
        bash_command=(
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            "from feature_engineering.transformations import load_training_frame\n"
            "df = load_training_frame(Path('data/sample'))\n"
            "Path('data/features').mkdir(parents=True, exist_ok=True)\n"
            "df.to_parquet('data/features/customer_churn_features.parquet')\n"
            "PY"
        ),
    )

    train = BashOperator(
        task_id="train_compare_register_models",
        bash_command=(
            "python -m training.train --config config/config.yaml "
            "--sample-dir data/sample --artifact-dir artifacts/models"
        ),
    )

    deploy = BashOperator(
        task_id="deploy_champion_to_sagemaker",
        bash_command="echo 'Deploy with deployment/sagemaker_deploy.py after model artifact is published to S3.'",
    )

    monitor = BashOperator(
        task_id="monitor_model_health",
        bash_command="echo 'Run monitoring.monitor with reference/current scored feature snapshots.'",
    )

    retrain = BashOperator(
        task_id="trigger_retraining_when_needed",
        bash_command="echo 'Retraining policy evaluated by monitoring/retraining.py.'",
    )

    ingest >> validate >> build_features >> train >> deploy >> monitor >> retrain
