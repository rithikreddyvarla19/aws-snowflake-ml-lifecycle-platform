"""Streamlit metrics dashboard for churn model performance and business KPIs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPORT_PATH = Path("artifacts/models/training_report.json")
MONITORING_PATH = Path("artifacts/monitoring/model_health_report.json")


st.set_page_config(page_title="Churn ML Lifecycle Dashboard", layout="wide")
st.title("Customer Churn ML Lifecycle")


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


training_report = load_json(REPORT_PATH)
monitoring_report = load_json(MONITORING_PATH)

if training_report:
    st.subheader("Model Comparison")
    runs = pd.DataFrame(training_report["runs"])
    metrics = pd.json_normalize(runs["metrics"])
    model_rows = pd.concat([runs[["model_name"]], metrics], axis=1)
    metric_columns = ["precision", "recall", "f1", "roc_auc"]
    cols = st.columns(len(metric_columns))
    champion_row = model_rows.iloc[0]
    for col, metric in zip(cols, metric_columns, strict=False):
        col.metric(metric.upper(), f"{champion_row[metric]:.3f}")
    st.plotly_chart(
        px.bar(model_rows, x="model_name", y=metric_columns, barmode="group", title="Offline Model Metrics"),
        use_container_width=True,
    )
    kpi_cols = [
        "customers_flagged_for_retention",
        "true_churners_captured",
        "false_positive_outreach",
        "estimated_retention_spend",
    ]
    st.dataframe(model_rows[["model_name", *kpi_cols]], use_container_width=True)
else:
    st.info("Run training first to populate artifacts/models/training_report.json.")

if monitoring_report:
    st.subheader("Monitoring")
    drift = pd.DataFrame(monitoring_report["feature_drift"])
    st.plotly_chart(
        px.bar(drift, x="feature", y="psi", color="drifted", title="Feature Drift PSI"),
        use_container_width=True,
    )
    st.json(monitoring_report["retraining_decision"])
else:
    st.info("Run monitoring to populate artifacts/monitoring/model_health_report.json.")
