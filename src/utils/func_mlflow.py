import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from .func_model import model_training


def get_mlflow_config(base_path: Path) -> dict:
    TRACKING_DIR = Path(os.getenv("MLFLOW_TRACKING_DIR", str(base_path / "mlflow_data"))).expanduser()
    ARTIFACT_DIR = Path(os.getenv("MLFLOW_ARTIFACT_DIR", str(TRACKING_DIR / "artifacts"))).expanduser()

    DEFAULT_TRACKING_URI = f"sqlite:///{(TRACKING_DIR / 'mlflow.db').resolve().as_posix()}"
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_ROOT", ARTIFACT_DIR.resolve().as_uri())

    return {
        "tracking_dir": TRACKING_DIR,
        "artifact_dir": ARTIFACT_DIR,
        "tracking_uri": TRACKING_URI,
        "artifact_uri": ARTIFACT_URI,
    }


def configure_mlflow(base_path: Path, experiment_name: str) -> dict:
    config = get_mlflow_config(base_path)
    TRACKING_DIR = config["tracking_dir"]
    ARTIFACT_DIR = config["artifact_dir"]
    TRACKING_URI = config["tracking_uri"]
    ARTIFACT_URI = config["artifact_uri"]

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)

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
        mlflow.log_metric("validation_accuracy", metrics["validation_accuracy"])
        mlflow.log_metric("cv_accuracy_mean", metrics["cv_accuracy_mean"])
        mlflow.log_metric("cv_accuracy_std", metrics["cv_accuracy_std"])
        log_dict_artifact(
            {
                "confusion_matrix": metrics["confusion_matrix"],
                "classification_report": metrics["classification_report"],
                "cv_fold_accuracies": metrics["cv_fold_accuracies"],
                "model_params": metrics["model_params"],
            },
            artifact_path="metrics",
        )
        mlflow.log_text(metrics["classification_report_text"], "metrics/classification_report.txt")
        return metrics
