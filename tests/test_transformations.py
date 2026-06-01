from __future__ import annotations

from pathlib import Path

from feature_engineering.transformations import load_training_frame


def test_sample_feature_frame_contains_model_columns() -> None:
    frame = load_training_frame(Path("data/sample"))
    expected = {"customer_id", "support_ticket_rate", "payment_failure_rate", "mrr_per_active_day", "churned"}
    assert expected.issubset(frame.columns)
    assert len(frame) == 16

