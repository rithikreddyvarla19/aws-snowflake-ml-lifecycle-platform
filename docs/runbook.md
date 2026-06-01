# Operational Runbook

## Daily Checks

1. Confirm the Airflow DAG completed.
2. Review ingestion manifest record counts and checksums.
3. Check Glue job duration and failed task counts.
4. Review MLflow experiment status for scheduled training runs.
5. Inspect dashboard precision, recall, F1, ROC AUC, and business KPIs.
6. Review monitoring report for drift and retraining decisions.

## Incident: Data Quality Failure

1. Open the failed Airflow task logs.
2. Identify the failing validation result and source table.
3. Compare current raw record counts against the prior successful manifest.
4. Quarantine the affected source snapshot.
5. Re-run ingestion after source owner confirmation.

## Incident: Accuracy Degradation

1. Validate delayed labels are complete for the evaluation window.
2. Review segment-level performance by plan, region, and company size.
3. Compare current feature distributions to the training reference.
4. Trigger retraining if the policy emits a positive decision.
5. Deploy the new champion only after offline and shadow evaluation pass.

## Incident: SageMaker Endpoint Failure

1. Check endpoint status and CloudWatch logs.
2. Roll back to the previous endpoint configuration if health checks fail.
3. Confirm model artifact availability in S3.
4. Validate container image digest and environment variables.
5. Re-run a smoke prediction before shifting traffic.

