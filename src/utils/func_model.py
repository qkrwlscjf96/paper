import os

import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


SCALE_REQUIRED_MODELS = {"MLPClassifier", "LogisticRegression"}


def _validate_unit_interval(metric_name: str, value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{metric_name} must be finite, got {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{metric_name} must be between 0 and 1, got {value}")
    return float(value)


def _parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    values = tuple(int(size.strip()) for size in value.split(",") if size.strip())
    return values or default


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none"}:
        return None
    return int(normalized)


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def get_supported_model_names() -> list[str]:
    return [
        "MLPClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "XGBClassifier",
    ]


def resolve_model_names(raw_model_names: str | None = None) -> list[str]:
    supported_model_names = get_supported_model_names()
    raw_value = raw_model_names or os.getenv("MODEL_NAMES", os.getenv("MODEL_NAME", "MLPClassifier"))
    model_names = [model_name.strip() for model_name in raw_value.split(",") if model_name.strip()]
    if not model_names:
        model_names = ["MLPClassifier"]

    if len(model_names) == 1 and model_names[0].lower() == "all":
        return supported_model_names

    invalid_model_names = [
        model_name for model_name in model_names
        if model_name not in supported_model_names
    ]
    if invalid_model_names:
        supported = ", ".join(supported_model_names)
        invalid = ", ".join(invalid_model_names)
        raise ValueError(f"Unsupported model_name={invalid}. Supported: {supported}")

    return model_names


def get_model_config_from_env(model_name: str) -> dict:
    if model_name == "MLPClassifier":
        return {
            "hidden_layer_sizes": _parse_int_tuple(
                os.getenv("MLP_HIDDEN_LAYER_SIZES", "64,32"),
                (64, 32),
            ),
            "activation": os.getenv("MLP_ACTIVATION", "relu"),
            "solver": os.getenv("MLP_SOLVER", "adam"),
            "max_iter": int(os.getenv("MLP_MAX_ITER", "1000")),
            "random_state": int(os.getenv("MLP_RANDOM_STATE", "42")),
        }

    if model_name == "LogisticRegression":
        return {
            "C": float(os.getenv("LOGISTIC_REGRESSION_C", "1.0")),
            "solver": os.getenv("LOGISTIC_REGRESSION_SOLVER", "lbfgs"),
            "max_iter": int(os.getenv("LOGISTIC_REGRESSION_MAX_ITER", "1000")),
            "random_state": int(os.getenv("LOGISTIC_REGRESSION_RANDOM_STATE", "42")),
        }

    if model_name == "RandomForestClassifier":
        return {
            "n_estimators": int(os.getenv("RANDOM_FOREST_N_ESTIMATORS", "200")),
            "max_depth": _parse_optional_int(os.getenv("RANDOM_FOREST_MAX_DEPTH")),
            "random_state": int(os.getenv("RANDOM_FOREST_RANDOM_STATE", "42")),
        }

    if model_name == "GradientBoostingClassifier":
        return {
            "n_estimators": int(os.getenv("GRADIENT_BOOSTING_N_ESTIMATORS", "100")),
            "learning_rate": float(os.getenv("GRADIENT_BOOSTING_LEARNING_RATE", "0.1")),
            "max_depth": int(os.getenv("GRADIENT_BOOSTING_MAX_DEPTH", "3")),
            "random_state": int(os.getenv("GRADIENT_BOOSTING_RANDOM_STATE", "42")),
        }

    if model_name == "XGBClassifier":
        return {
            "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "200")),
            "max_depth": int(os.getenv("XGB_MAX_DEPTH", "6")),
            "learning_rate": float(os.getenv("XGB_LEARNING_RATE", "0.1")),
            "subsample": float(os.getenv("XGB_SUBSAMPLE", "1.0")),
            "colsample_bytree": float(os.getenv("XGB_COLSAMPLE_BYTREE", "1.0")),
            "random_state": int(os.getenv("XGB_RANDOM_STATE", "42")),
            "eval_metric": os.getenv("XGB_EVAL_METRIC", "logloss"),
        }

    supported = ", ".join(get_supported_model_names())
    raise ValueError(f"Unsupported model_name={model_name}. Supported: {supported}")


def get_model_run_configs_from_env(data_name: str = "") -> list[dict]:
    model_names = resolve_model_names()

    return [
        {
            "data_name": data_name or os.getenv("DATA_NAME", ""),
            "model_name": model_name,
            **get_model_logging_params(model_name, get_model_config_from_env(model_name)),
        }
        for model_name in model_names
    ]


def get_model_logging_params(model_name: str, model_config: dict) -> dict:
    loggable_config = {}
    for key, value in model_config.items():
        if isinstance(value, tuple):
            loggable_config[f"{model_name}_{key}"] = ",".join(map(str, value))
        else:
            loggable_config[f"{model_name}_{key}"] = value
    return loggable_config


def resolve_model_config(modeling_config: dict) -> dict:
    model_name = modeling_config["model_name"]
    return get_model_config_from_env(model_name)


def _build_model(model_name: str, model_config: dict | None = None):
    model_config = model_config or {}

    if model_name == "MLPClassifier":
        return MLPClassifier(**model_config)
    if model_name == "LogisticRegression":
        return LogisticRegression(**model_config)
    if model_name == "RandomForestClassifier":
        return RandomForestClassifier(**model_config)
    if model_name == "GradientBoostingClassifier":
        return GradientBoostingClassifier(**model_config)
    if model_name == "XGBClassifier":
        return xgb.XGBClassifier(**model_config)

    supported = ", ".join(get_supported_model_names())
    raise ValueError(f"Unsupported model_name={model_name}. Supported: {supported}")


def _get_metric_average(y_true):
    return "binary" if y_true.nunique() == 2 else "weighted"


def _prepare_features(
    X_train,
    X_other,
    model_name: str,
):
    """Fit scaling on fold train only when the model requires it."""
    if model_name in SCALE_REQUIRED_MODELS:
        scaler = StandardScaler()
        train_values = scaler.fit_transform(X_train)
        other_values = scaler.transform(X_other)
    else:
        train_values = X_train.to_numpy(copy=True)
        other_values = X_other.to_numpy(copy=True)

    return train_values, other_values


def build_stratified_cv_folds(
    train_df,
    target_col: list[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> list[dict]:
    """Materialize fixed fold frames so every candidate uses identical rows."""
    y_train = train_df[target_col[0]]
    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return [
        {
            "train_df": train_df.iloc[fold_train_idx].reset_index(drop=True),
            "valid_df": train_df.iloc[fold_valid_idx].reset_index(drop=True),
        }
        for fold_train_idx, fold_valid_idx in kfold.split(train_df, y_train)
    ]


def cross_validate_model(
    cv_folds: list[dict],
    target_col,
    model_name: str = "MLPClassifier",
    model_config: dict | None = None,
):
    if not cv_folds:
        raise ValueError("cv_folds must not be empty")
    cv_aucs = []
    cv_accuracies = []
    cv_f1_scores = []
    cv_precisions = []
    cv_recalls = []

    for fold_idx, fold in enumerate(cv_folds, start=1):
        fold_train_df = fold["train_df"]
        fold_valid_df = fold["valid_df"]
        fold_check_cols = fold["check_cols"]
        X_fold_train = fold_train_df[fold_check_cols]
        X_fold_valid = fold_valid_df[fold_check_cols]
        y_fold_train = fold_train_df[target_col[0]]
        y_fold_valid = fold_valid_df[target_col[0]]
        metric_average = _get_metric_average(y_fold_train)

        X_fold_train_scaled, X_fold_valid_scaled = _prepare_features(
            X_fold_train, X_fold_valid, model_name
        )

        model = clone(_build_model(model_name, model_config))
        fit_kwargs = {}
        if fold.get("sample_weight") is not None:
            fit_kwargs["sample_weight"] = np.asarray(fold["sample_weight"], dtype=float)
        model.fit(X_fold_train_scaled, y_fold_train, **fit_kwargs)
        y_fold_pred = model.predict(X_fold_valid_scaled)
        y_fold_score = model.predict_proba(X_fold_valid_scaled)[:, 1]
        fold_auc = _validate_unit_interval(
            f"{model_name}.fold_{fold_idx}.auc",
            roc_auc_score(y_fold_valid, y_fold_score),
        )
        fold_accuracy = _validate_unit_interval(
            f"{model_name}.fold_{fold_idx}.accuracy",
            accuracy_score(y_fold_valid, y_fold_pred),
        )
        fold_f1 = _validate_unit_interval(
            f"{model_name}.fold_{fold_idx}.f1_score",
            f1_score(y_fold_valid, y_fold_pred, average=metric_average, zero_division=0),
        )
        fold_precision = _validate_unit_interval(
            f"{model_name}.fold_{fold_idx}.precision",
            precision_score(
                y_fold_valid, y_fold_pred, average=metric_average, zero_division=0
            ),
        )
        fold_recall = _validate_unit_interval(
            f"{model_name}.fold_{fold_idx}.recall",
            recall_score(
                y_fold_valid, y_fold_pred, average=metric_average, zero_division=0
            ),
        )
        cv_aucs.append(float(fold_auc))
        cv_accuracies.append(float(fold_accuracy))
        cv_f1_scores.append(float(fold_f1))
        cv_precisions.append(float(fold_precision))
        cv_recalls.append(float(fold_recall))
        print(
            f"[CV {fold_idx}/{len(cv_folds)}] "
            f"auc={fold_auc:.4f} | f1={fold_f1:.4f} | "
            f"precision={fold_precision:.4f} | recall={fold_recall:.4f} | "
            f"accuracy={fold_accuracy:.4f}"
        )

    print(
        f"[RESULT] cv_auc={np.mean(cv_aucs):.4f} | "
        f"cv_f1score={np.mean(cv_f1_scores):.4f} | "
        f"cv_precision={np.mean(cv_precisions):.4f} | "
        f"cv_recall={np.mean(cv_recalls):.4f} | "
        f"cv_accuracy={np.mean(cv_accuracies):.4f}"
    )

    cv_auc = _validate_unit_interval(f"{model_name}.cv_auc", np.mean(cv_aucs))
    cv_accuracy = _validate_unit_interval(f"{model_name}.cv_accuracy", np.mean(cv_accuracies))
    cv_f1_score_mean = _validate_unit_interval(f"{model_name}.cv_f1_score", np.mean(cv_f1_scores))
    cv_precision_mean = _validate_unit_interval(f"{model_name}.cv_precision", np.mean(cv_precisions))
    cv_recall_mean = _validate_unit_interval(f"{model_name}.cv_recall", np.mean(cv_recalls))

    return {
        "cv_auc": cv_auc,
        "cv_f1score": cv_f1_score_mean,
        "cv_precision": cv_precision_mean,
        "cv_recall": cv_recall_mean,
        "cv_accuracy": cv_accuracy,
    }
