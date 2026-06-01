FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY data_ingestion /app/data_ingestion
COPY feature_engineering /app/feature_engineering
COPY training /app/training
COPY deployment /app/deployment
COPY monitoring /app/monitoring
COPY dashboards /app/dashboards

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[cloud,dashboard]"

EXPOSE 8000

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
