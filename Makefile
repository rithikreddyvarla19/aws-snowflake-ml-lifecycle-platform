.PHONY: install install-full lint test compile ingest train api dashboard

install:
	python -m pip install -e ".[dev]"

install-full:
	python -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	python -m ruff check .

test:
	python -m pytest -q

compile:
	python -m compileall data_ingestion feature_engineering training deployment monitoring airflow dashboards tests

ingest:
	python -m data_ingestion.ingestion_job --config config/config.yaml --raw-root data/raw

train:
	python -m training.train --config config/config.yaml --sample-dir data/sample --artifact-dir artifacts/models

api:
	python -m uvicorn deployment.api:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboards/app.py

