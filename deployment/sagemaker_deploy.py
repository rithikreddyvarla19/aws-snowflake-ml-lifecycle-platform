"""SageMaker deployment automation for the champion churn model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SageMakerDeploymentConfig:
    model_data_url: str
    image_uri: str
    role_arn: str
    model_name: str
    endpoint_config_name: str
    endpoint_name: str
    instance_type: str = "ml.m5.large"
    initial_instance_count: int = 1


def deploy(config: SageMakerDeploymentConfig) -> dict[str, str]:
    import boto3

    client = boto3.client("sagemaker")
    client.create_model(
        ModelName=config.model_name,
        ExecutionRoleArn=config.role_arn,
        PrimaryContainer={"Image": config.image_uri, "ModelDataUrl": config.model_data_url},
    )
    client.create_endpoint_config(
        EndpointConfigName=config.endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": config.model_name,
                "InitialInstanceCount": config.initial_instance_count,
                "InstanceType": config.instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    client.create_endpoint(EndpointName=config.endpoint_name, EndpointConfigName=config.endpoint_config_name)
    return {
        "model_name": config.model_name,
        "endpoint_config_name": config.endpoint_config_name,
        "endpoint_name": config.endpoint_name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy champion churn model to SageMaker.")
    parser.add_argument("--model-data-url", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--model-name", default="churn-prediction-model")
    parser.add_argument("--endpoint-config-name", default="churn-prediction-endpoint-config")
    parser.add_argument("--endpoint-name", default="churn-prediction-prod")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    deployment = SageMakerDeploymentConfig(**vars(args))
    print(deploy(deployment))

