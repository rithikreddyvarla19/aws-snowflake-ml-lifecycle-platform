"""Production REST API for churn prediction."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from deployment.inference import ChurnPredictor


class ChurnRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class ChurnResponse(BaseModel):
    predictions: list[dict[str, float]]


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    model_path = Path(os.environ.get("MODEL_PATH", "artifacts/models/churn_random_forest.joblib"))
    return ChurnPredictor(model_path)


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(payload: ChurnRequest) -> ChurnResponse:
    return ChurnResponse(predictions=get_predictor().predict(payload.records))

