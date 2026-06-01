"""Automated retraining trigger policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainingDecision:
    should_retrain: bool
    reasons: list[str]


def evaluate_retraining_policy(
    drifted_features: list[str],
    accuracy_degraded: bool,
    prediction_shifted: bool,
    min_records_met: bool,
) -> RetrainingDecision:
    reasons: list[str] = []
    if not min_records_met:
        return RetrainingDecision(False, ["minimum scored record threshold not met"])
    if drifted_features:
        reasons.append(f"feature drift detected: {', '.join(drifted_features)}")
    if accuracy_degraded:
        reasons.append("accuracy degradation detected")
    if prediction_shifted:
        reasons.append("prediction distribution shift detected")
    return RetrainingDecision(bool(reasons), reasons)

