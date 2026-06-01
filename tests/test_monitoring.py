from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.drift import population_stability_index, prediction_distribution_shift
from monitoring.metrics import accuracy_degradation
from monitoring.retraining import evaluate_retraining_policy


def test_population_stability_index_detects_large_shift() -> None:
    reference = pd.Series(np.repeat([1.0, 2.0, 3.0], repeats=50))
    current = pd.Series(np.repeat([8.0, 9.0, 10.0], repeats=50))
    assert population_stability_index(reference, current) > 0.2


def test_prediction_shift_contract() -> None:
    result = prediction_distribution_shift(pd.Series([0.1, 0.2, 0.3]), pd.Series([0.8, 0.9, 0.7]), threshold=0.15)
    assert result["shifted"] is True
    assert result["absolute_shift"] > 0.15


def test_retraining_policy_requires_minimum_records() -> None:
    decision = evaluate_retraining_policy(["support_ticket_rate"], True, True, min_records_met=False)
    assert decision.should_retrain is False


def test_accuracy_degradation() -> None:
    result = accuracy_degradation(reference_accuracy=0.9, current_accuracy=0.82, threshold=0.05)
    assert result["degraded"] is True

