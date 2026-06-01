"""Batch monitor that writes a model health report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from monitoring.drift import detect_feature_drift, prediction_distribution_shift
from monitoring.metrics import accuracy_degradation
from monitoring.retraining import evaluate_retraining_policy
from training.models import NUMERIC_FEATURES


def build_monitoring_report(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_accuracy: float,
    current_accuracy: float,
    min_records: int,
) -> dict[str, object]:
    drift_results = detect_feature_drift(reference_features, current_features, NUMERIC_FEATURES)
    prediction_shift = prediction_distribution_shift(
        reference_features["churn_probability"],
        current_features["churn_probability"],
    )
    degradation = accuracy_degradation(reference_accuracy, current_accuracy)
    decision = evaluate_retraining_policy(
        drifted_features=[result.feature for result in drift_results if result.drifted],
        accuracy_degraded=bool(degradation["degraded"]),
        prediction_shifted=bool(prediction_shift["shifted"]),
        min_records_met=len(current_features) >= min_records,
    )
    return {
        "feature_drift": [result.__dict__ for result in drift_results],
        "accuracy_degradation": degradation,
        "prediction_distribution_shift": prediction_shift,
        "retraining_decision": decision.__dict__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model monitoring report.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/monitoring/model_health_report.json"))
    parser.add_argument("--reference-accuracy", type=float, required=True)
    parser.add_argument("--current-accuracy", type=float, required=True)
    parser.add_argument("--min-records", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = build_monitoring_report(
        pd.read_parquet(args.reference),
        pd.read_parquet(args.current),
        args.reference_accuracy,
        args.current_accuracy,
        args.min_records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

