"""Model factory for classical ML, XGBoost, TensorFlow, and PyTorch candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]


NUMERIC_FEATURES = [
    "monthly_recurring_revenue",
    "days_since_signup",
    "support_ticket_rate",
    "payment_failure_rate",
    "mrr_per_active_day",
    "total_events",
    "active_days",
]

CATEGORICAL_FEATURES = ["plan_type", "billing_interval", "subscription_status"]


def preprocessing_pipeline() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def logistic_regression_spec() -> ModelSpec:
    estimator = Pipeline(
        steps=[
            ("preprocess", preprocessing_pipeline()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    return ModelSpec(
        "logistic_regression",
        estimator,
        {"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l2"]},
    )


def random_forest_spec() -> ModelSpec:
    estimator = Pipeline(
        steps=[
            ("preprocess", preprocessing_pipeline()),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=42)),
        ]
    )
    return ModelSpec(
        "random_forest",
        estimator,
        {"model__n_estimators": [100, 250], "model__max_depth": [4, 8, None]},
    )


def xgboost_spec() -> ModelSpec:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("Install xgboost to train the xgboost candidate.") from exc

    estimator = Pipeline(
        steps=[
            ("preprocess", preprocessing_pipeline()),
            (
                "model",
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return ModelSpec(
        "xgboost",
        estimator,
        {"model__max_depth": [3, 5], "model__learning_rate": [0.03, 0.1], "model__n_estimators": [100, 250]},
    )


def tensorflow_dnn_spec(input_dim: int | None = None) -> ModelSpec:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install tensorflow to train the TensorFlow DNN candidate.") from exc

    if input_dim is None:
        input_dim = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC", "Precision", "Recall"])
    return ModelSpec("tensorflow_dnn", model, {"epochs": [25], "batch_size": [64, 128]})


def pytorch_dnn(input_dim: int):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch to instantiate the PyTorch DNN.") from exc

    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )


def candidate_specs(names: list[str]) -> list[ModelSpec]:
    registry = {
        "logistic_regression": logistic_regression_spec,
        "random_forest": random_forest_spec,
        "xgboost": xgboost_spec,
    }
    specs: list[ModelSpec] = []
    for name in names:
        if name == "tensorflow_dnn":
            continue
        if name not in registry:
            raise ValueError(f"Unknown model candidate: {name}")
        specs.append(registry[name]())
    return specs

