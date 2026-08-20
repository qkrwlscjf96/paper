import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import ViewType

from .func_model import cross_validate_model


def get_mlflow_config(base_path: Path) -> dict:
    TRACKING_DIR = Path(os.getenv("MLFLOW_TRACKING_DIR", str(base_path / "result" / "mlflow"))).expanduser()
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


def _is_remote_tracking_uri(tracking_uri: str) -> bool:
    scheme = urlparse(tracking_uri).scheme
    return scheme in {"http", "https"}


def _get_experiment_any_state(client: MlflowClient, experiment_name: str):
    experiments = client.search_experiments(view_type=ViewType.ALL)
    for experiment in experiments:
        if experiment.name == experiment_name:
            return experiment
    return None


def _restore_if_deleted(client: MlflowClient, experiment):
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        return client.get_experiment(experiment.experiment_id)
    return experiment


def _uses_mlflow_artifact_proxy(experiment) -> bool:
    artifact_location = getattr(experiment, "artifact_location", "") or ""
    return artifact_location.startswith("mlflow-artifacts:/")


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

    experiment = _get_experiment_any_state(client, experiment_name)
    experiment = _restore_if_deleted(client, experiment)

    resolved_experiment_name = experiment_name

    if (
        experiment is not None
        and not _is_remote_tracking_uri(TRACKING_URI)
        and _uses_mlflow_artifact_proxy(experiment)
    ):
        resolved_experiment_name = f"{experiment_name}--local-artifacts"
        experiment = _get_experiment_any_state(client, resolved_experiment_name)
        experiment = _restore_if_deleted(client, experiment)

    if experiment is None:
        if _is_remote_tracking_uri(TRACKING_URI):
            # Let the remote tracking server apply its own default artifact root.
            client.create_experiment(resolved_experiment_name)
        else:
            client.create_experiment(resolved_experiment_name, artifact_location=ARTIFACT_URI)

    mlflow.set_experiment(resolved_experiment_name)
    config["resolved_experiment_name"] = resolved_experiment_name
    return config


def _sanitize_name_part(value) -> str:
    return (
        str(value)
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "x")
        .replace("[", "")
        .replace("]", "")
    )


def _format_param_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(map(str, value))
    return str(value)


def make_name_from_params(prefix: str, params: dict, include_keys: list[str]) -> str:
    parts = [prefix]
    for key in include_keys:
        if key not in params:
            continue
        value = _sanitize_name_part(_format_param_value(params[key]))
        parts.append(f"{key}={value}")
    return "--".join(parts)


def log_dataframe_artifact(df: pd.DataFrame, artifact_path: str, filename: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / filename
        df.to_csv(output_path, index=False)
        mlflow.log_artifact(str(output_path), artifact_path=artifact_path)


def run_and_log_cv_model(
    run_name: str,
    cv_folds_by_weighting: dict[str, list[dict]],
    target_col: list[str],
    model_name: str,
    model_params: dict,
    params: dict,
    tags: dict,
    artifact_dataframes: dict[str, pd.DataFrame] | None = None,
    nested: bool = False,
) -> dict:
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        if artifact_dataframes:
            for artifact_name, artifact_df in artifact_dataframes.items():
                log_dataframe_artifact(
                    artifact_df,
                    artifact_path="weighting_details",
                    filename=f"{artifact_name}.csv",
                )

        metrics_by_weighting = {}
        for weighting in ("baseline", "weighted"):
            metrics = cross_validate_model(
                cv_folds_by_weighting[weighting],
                target_col,
                model_name=model_name,
                model_config=model_params,
            )
            metrics_by_weighting[weighting] = metrics
            for metric_name in (
                "cv_auc",
                "cv_f1score",
                "cv_precision",
                "cv_recall",
                "cv_accuracy",
            ):
                mlflow.log_metric(
                    f"{weighting}_{metric_name}", metrics[metric_name]
                )
        return metrics_by_weighting
