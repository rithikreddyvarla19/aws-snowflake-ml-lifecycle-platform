# Monitoring Design

## Signals

The monitoring package tracks three classes of production risk:

- Feature drift: population stability index for numeric features.
- Accuracy degradation: drop from reference accuracy after labels arrive.
- Prediction distribution shift: absolute movement in average churn probability.

## Metrics Dashboard

The dashboard surfaces:

- Precision
- Recall
- F1
- ROC AUC
- Accuracy
- Customers flagged for retention
- True churners captured
- False-positive outreach
- Missed churners
- Estimated retention spend

## Thresholds

Default thresholds live in `config/config.yaml`:

- `psi_threshold`: `0.2`
- `accuracy_drop_threshold`: `0.05`
- `prediction_shift_threshold`: `0.15`
- `min_scored_records`: `1000`

## Recommended Production Tables

```sql
CREATE TABLE IF NOT EXISTS CHURN_ML.MONITORING.PREDICTION_LOGS (
  request_id STRING,
  customer_id STRING,
  churn_probability FLOAT,
  churn_prediction INTEGER,
  model_name STRING,
  model_version STRING,
  scored_at TIMESTAMP_NTZ
);

CREATE TABLE IF NOT EXISTS CHURN_ML.MONITORING.LABELS (
  customer_id STRING,
  churned INTEGER,
  label_observed_at TIMESTAMP_NTZ
);

CREATE TABLE IF NOT EXISTS CHURN_ML.MONITORING.MODEL_HEALTH_REPORTS (
  report_id STRING,
  report VARIANT,
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
```

## Retraining Policy

Retraining is triggered only when the minimum scored-record threshold is met and at least one of these signals is true:

- One or more features exceed the PSI threshold.
- Current accuracy drops beyond the configured threshold.
- Mean predicted churn probability shifts beyond the configured threshold.

