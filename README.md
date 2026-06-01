# AWS Snowflake ML Lifecycle Platform

Production-grade machine learning lifecycle platform for customer churn prediction in a subscription business. The repository covers multi-source ingestion, PySpark ETL, Snowflake feature storage, model training, hyperparameter optimization, MLflow tracking, champion selection, SageMaker deployment, REST inference, monitoring, drift detection, and automated retraining orchestration.

## Platform Scope

- Ingestion: local CSV, S3 CSV, JSONL, and Snowflake extraction patterns.
- Processing: AWS Glue compatible PySpark job plus local pandas transformations.
- Quality: reusable validation checks for keys, nulls, accepted values, and numeric ranges.
- Feature store: Snowflake offline and online feature table contracts.
- Training: Logistic Regression, Random Forest, XGBoost, TensorFlow DNN, and PyTorch DNN factory support.
- Optimization: sklearn grid search with stratified cross-validation.
- Tracking: MLflow metrics, params, model artifacts, and registry hooks.
- Comparison: automatic champion model selection by configurable metric.
- Deployment: SageMaker endpoint automation and FastAPI REST endpoint.
- Monitoring: PSI drift detection, accuracy degradation, prediction distribution shifts, business KPIs.
- Retraining: policy-based retraining decision designed for Airflow.
- Operations: Docker, GitHub Actions, Terraform, diagrams, docs, sample datasets, and unit tests.

## Repository Layout

```text
data_ingestion/        Multi-source ingestion and Snowflake raw loading helpers
feature_engineering/   PySpark ETL, pandas transformations, validation, feature store SQL
training/              Model factories, HPO, evaluation, MLflow registry integration
deployment/            FastAPI inference and SageMaker deployment automation
monitoring/            Drift, degradation, scorecards, retraining policy
airflow/dags/          End-to-end orchestration DAG
dashboards/            Streamlit dashboard for model metrics and business KPIs
terraform/             AWS and Snowflake infrastructure templates
docs/                  Architecture, runbook, diagrams, monitoring design
tests/                 Unit tests for quality, features, monitoring, SQL contracts
data/sample/           Synthetic subscription churn sample datasets
```

## Quickstart

```bash
cd aws-snowflake-ml-lifecycle-platform
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

Run ingestion against the sample sources:

```bash
python -m data_ingestion.ingestion_job --config config/config.yaml --raw-root data/raw
```

Train local model candidates:

```bash
python -m pip install -e ".[ml]"
python -m training.train --config config/config.yaml --sample-dir data/sample --artifact-dir artifacts/models
```

Serve a trained artifact locally:

```bash
MODEL_PATH=artifacts/models/churn_random_forest.joblib uvicorn deployment.api:app --reload --port 8000
```

## Docker

```bash
docker compose up --build
```

Services:

- `http://localhost:8000/health` for the prediction API.
- `http://localhost:5000` for MLflow.
- `http://localhost:8501` for the metrics dashboard.

## Production Path

1. Land raw source snapshots in S3 and Snowflake raw tables.
2. Run the Glue PySpark feature job into curated S3 and Snowflake feature tables.
3. Validate data quality before model training.
4. Train candidates, run HPO, log experiments to MLflow, and select a champion.
5. Publish the champion artifact to S3 and deploy to SageMaker.
6. Capture predictions, labels, and business outcomes in Snowflake monitoring tables.
7. Evaluate drift, performance degradation, and prediction shifts.
8. Trigger retraining through Airflow when the policy threshold is met.

## Key Commands

```bash
make lint
make test
make compile
make ingest
make train
make api
make dashboard
```

## Infrastructure

Terraform templates create AWS S3 buckets, IAM roles, a Glue catalog database/job, SageMaker model and endpoint scaffolding, and Snowflake database/schema/warehouse primitives. See [terraform/README.md](terraform/README.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Monitoring Design](docs/monitoring.md)
- [Operational Runbook](docs/runbook.md)
- [Architecture Diagram](docs/diagrams/architecture.mmd)
- [Data Flow Diagram](docs/diagrams/data_flow.mmd)
- [ML Lifecycle Diagram](docs/diagrams/ml_lifecycle.mmd)

## Security Notes

- No credentials are committed. Use `.env.example` as the contract for local development.
- Prefer IAM roles, AWS Secrets Manager, and Snowflake key-pair auth in production.
- Enable S3 encryption, versioning, lifecycle policies, and public access blocks.
- Separate Snowflake raw, feature store, and monitoring schemas with role-based access.

