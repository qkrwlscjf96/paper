import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from .func_model import model_training


def get_mlflow_config(base_path: Path) -> dict:
    tracking_dir = Path(os.getenv("MLFLOW_TRACKING_DIR", str(base_path / "mlflow_data"))).expanduser()
    artifact_dir = Path(os.getenv("MLFLOW_ARTIFACT_DIR", str(tracking_dir / "artifacts"))).expanduser()

    default_tracking_uri = f"sqlite:///{(tracking_dir / 'mlflow.db').resolve().as_posix()}"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    artifact_uri = os.getenv("MLFLOW_ARTIFACT_ROOT", artifact_dir.resolve().as_uri())

    return {
        "tracking_dir": tracking_dir,
        "artifact_dir": artifact_dir,
        "tracking_uri": tracking_uri,
        "artifact_uri": artifact_uri,
    }


def configure_mlflow(base_path: Path, experiment_name: str) -> dict:
    config = get_mlflow_config(base_path)
    tracking_dir = config["tracking_dir"]
    artifact_dir = config["artifact_dir"]
    tracking_uri = config["tracking_uri"]
    artifact_uri = config["artifact_uri"]

    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_uri)

    mlflow.set_experiment(experiment_name)
    return config


def make_experiment_name(prefix: str, data_name: str) -> str:
    return f"{prefix}-{data_name}"


def parse_multiplier_values(env_name: str, default_values: list[int]) -> list[int]:
    raw = os.getenv(env_name)
    if not raw:
        return default_values

    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))

    return values or default_values


def log_dict_artifact(data: dict, artifact_path: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as temp_file:
        json.dump(data, temp_file, ensure_ascii=False, indent=2)
        temp_path = temp_file.name

    try:
        mlflow.log_artifact(temp_path, artifact_path=artifact_path)
    finally:
        os.remove(temp_path)


def run_and_log_model(
    run_name: str,
    df_to_train: pd.DataFrame,
    check_cols: list[str],
    target_col: list[str],
    params: dict,
    tags: dict,
    nested: bool = False,
) -> dict:
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        log_dict_artifact({"params": params, "tags": tags}, artifact_path="config")

        metrics = model_training(df_to_train, check_cols, target_col)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        log_dict_artifact(
            {
                "confusion_matrix": metrics["confusion_matrix"],
                "classification_report": metrics["classification_report"],
                "model_params": metrics["model_params"],
            },
            artifact_path="metrics",
        )
        mlflow.log_text(metrics["classification_report_text"], "metrics/classification_report.txt")
        return metrics
