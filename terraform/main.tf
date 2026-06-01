locals {
  name_prefix = "churn-ml-${var.environment}"
  common_tags = {
    Project     = "aws-snowflake-ml-lifecycle-platform"
    Environment = var.environment
    Owner       = "ml-platform"
  }
}

resource "aws_s3_bucket" "raw" {
  bucket = var.raw_bucket_name
  tags   = local.common_tags
}

resource "aws_s3_bucket" "curated" {
  bucket = var.curated_bucket_name
  tags   = local.common_tags
}

resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifact_bucket_name
  tags   = local.common_tags
}

resource "aws_s3_bucket_versioning" "all" {
  for_each = {
    raw       = aws_s3_bucket.raw.id
    curated   = aws_s3_bucket.curated.id
    artifacts = aws_s3_bucket.artifacts.id
  }

  bucket = each.value
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "all" {
  for_each = {
    raw       = aws_s3_bucket.raw.id
    curated   = aws_s3_bucket.curated.id
    artifacts = aws_s3_bucket.artifacts.id
  }

  bucket                  = each.value
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "all" {
  for_each = {
    raw       = aws_s3_bucket.raw.id
    curated   = aws_s3_bucket.curated.id
    artifacts = aws_s3_bucket.artifacts.id
  }

  bucket = each.value
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

data "aws_iam_policy_document" "glue_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["glue.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "glue" {
  name               = "${local.name_prefix}-glue-role"
  assume_role_policy = data.aws_iam_policy_document.glue_assume_role.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

data "aws_iam_policy_document" "glue_data_access" {
  statement {
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.raw.arn,
      "${aws_s3_bucket.raw.arn}/*",
      aws_s3_bucket.curated.arn,
      "${aws_s3_bucket.curated.arn}/*",
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*"
    ]
  }
}

resource "aws_iam_role_policy" "glue_data_access" {
  name   = "${local.name_prefix}-glue-data-access"
  role   = aws_iam_role.glue.id
  policy = data.aws_iam_policy_document.glue_data_access.json
}

resource "aws_glue_catalog_database" "churn_ml" {
  name = "churn_ml_${var.environment}"
}

resource "aws_glue_job" "feature_engineering" {
  name              = "${local.name_prefix}-feature-engineering"
  role_arn          = aws_iam_role.glue.arn
  glue_version      = "4.0"
  worker_type       = "G.1X"
  number_of_workers = 2

  command {
    name            = "glueetl"
    script_location = var.glue_script_s3_uri
    python_version  = "3"
  }

  default_arguments = {
    "--job-language"                     = "python"
    "--enable-metrics"                   = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--input-root"                       = "s3://${aws_s3_bucket.raw.bucket}/"
    "--output-uri"                       = "s3://${aws_s3_bucket.curated.bucket}/features/customer_churn/"
  }

  tags = local.common_tags
}

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${local.name_prefix}-sagemaker-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_sagemaker_model" "churn" {
  name               = "${local.name_prefix}-model"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = var.sagemaker_image_uri
    model_data_url = var.sagemaker_model_data_url
  }

  tags = local.common_tags
}

resource "aws_sagemaker_endpoint_configuration" "churn" {
  name = "${local.name_prefix}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.churn.name
    initial_instance_count = 1
    instance_type          = "ml.m5.large"
    initial_variant_weight = 1
  }

  tags = local.common_tags
}

resource "aws_sagemaker_endpoint" "churn" {
  name                 = "${local.name_prefix}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.churn.name
  tags                 = local.common_tags
}

resource "snowflake_database" "churn_ml" {
  name    = "CHURN_ML_${upper(var.environment)}"
  comment = "Customer churn ML lifecycle platform database."
}

resource "snowflake_schema" "raw" {
  database = snowflake_database.churn_ml.name
  name     = "RAW"
  comment  = "Raw ingested snapshots."
}

resource "snowflake_schema" "feature_store" {
  database = snowflake_database.churn_ml.name
  name     = "FEATURE_STORE"
  comment  = "Offline and online churn feature tables."
}

resource "snowflake_schema" "monitoring" {
  database = snowflake_database.churn_ml.name
  name     = "MONITORING"
  comment  = "Prediction logs, labels, and model health reports."
}

resource "snowflake_warehouse" "ml" {
  name           = "ML_WH_${upper(var.environment)}"
  warehouse_size = "MEDIUM"
  auto_suspend   = 60
  auto_resume    = true
  comment        = "Warehouse for ML feature engineering and monitoring queries."
}
