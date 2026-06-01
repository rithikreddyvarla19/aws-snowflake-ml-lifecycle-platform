"""Inference helpers shared by FastAPI and SageMaker serving containers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class ChurnPredictor:
    """Thin wrapper around a trained sklearn-compatible churn model."""

    def __init__(self, model_path: Path) -> None:
        import joblib

        self.model_path = model_path
        self.model = joblib.load(model_path)

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, float]]:
        frame = pd.DataFrame(records)
        probabilities = self.model.predict_proba(frame)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        return [
            {"churn_probability": float(probability), "churn_prediction": int(prediction)}
            for probability, prediction in zip(probabilities, predictions, strict=False)
        ]

