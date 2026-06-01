output "raw_bucket" {
  value       = aws_s3_bucket.raw.bucket
  description = "Raw data S3 bucket."
}

output "curated_bucket" {
  value       = aws_s3_bucket.curated.bucket
  description = "Curated data S3 bucket."
}

output "artifact_bucket" {
  value       = aws_s3_bucket.artifacts.bucket
  description = "Model artifact S3 bucket."
}

output "glue_job_name" {
  value       = aws_glue_job.feature_engineering.name
  description = "Glue feature engineering job name."
}

output "sagemaker_endpoint_name" {
  value       = aws_sagemaker_endpoint.churn.name
  description = "SageMaker endpoint name."
}

output "snowflake_database" {
  value       = snowflake_database.churn_ml.name
  description = "Snowflake database name."
}

