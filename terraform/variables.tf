variable "aws_region" {
  type        = string
  description = "AWS region for the ML platform."
  default     = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Deployment environment name."
  default     = "dev"
}

variable "raw_bucket_name" {
  type        = string
  description = "S3 bucket for raw immutable snapshots."
}

variable "curated_bucket_name" {
  type        = string
  description = "S3 bucket for curated feature outputs."
}

variable "artifact_bucket_name" {
  type        = string
  description = "S3 bucket for model and pipeline artifacts."
}

variable "glue_script_s3_uri" {
  type        = string
  description = "S3 URI of the PySpark Glue job script."
}

variable "sagemaker_image_uri" {
  type        = string
  description = "ECR image URI for the SageMaker inference container."
}

variable "sagemaker_model_data_url" {
  type        = string
  description = "S3 URL for the trained champion model artifact."
}

variable "snowflake_organization_name" {
  type        = string
  description = "Snowflake organization name."
}

variable "snowflake_account_name" {
  type        = string
  description = "Snowflake account name."
}

variable "snowflake_user" {
  type        = string
  description = "Snowflake Terraform service user."
}

variable "snowflake_password" {
  type        = string
  description = "Snowflake service user password."
  sensitive   = true
}

variable "snowflake_role" {
  type        = string
  description = "Snowflake role used by Terraform."
  default     = "SYSADMIN"
}

