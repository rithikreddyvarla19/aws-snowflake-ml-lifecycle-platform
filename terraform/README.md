# Terraform

This directory provisions the platform foundation:

- S3 raw, curated, and artifact buckets with versioning, encryption, and public access blocks.
- IAM roles and policies for AWS Glue and SageMaker.
- AWS Glue catalog database and feature engineering job.
- SageMaker model, endpoint configuration, and endpoint scaffolding.
- Snowflake database, raw schema, feature store schema, monitoring schema, and warehouse.

## Usage

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
terraform init
terraform plan
terraform apply
```

Upload `feature_engineering/etl_pyspark.py` to the `glue_script_s3_uri` path before applying the Glue job in a real environment.

## Snowflake Notes

The Terraform configuration uses the official `snowflakedb/snowflake` provider namespace. Table DDL is kept in `docs/snowflake_bootstrap.sql` and mirrored by Python SQL helpers so table-level contracts can evolve without tightly coupling every schema change to provider resource syntax.

