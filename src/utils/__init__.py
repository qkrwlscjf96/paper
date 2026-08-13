"""Public utilities used by ``main.py`` and other modules."""

from .func_common import get_available_data_names, load_data_df
from .func_eda import plot_boxplots_by_date
from .func_static import (
    date_trend,
    outlier_remover,
)
from .func_weight import get_weighted_df

try:
    from .func_feat_imp import xgboost_feature_importance
except ModuleNotFoundError:
    xgboost_feature_importance = None

try:
    from .func_mlflow import (
        configure_mlflow,
        get_mlflow_config,
        make_name_from_params,
        run_and_log_model,
    )
except ModuleNotFoundError:
    configure_mlflow = None
    get_mlflow_config = None
    make_name_from_params = None
    run_and_log_model = None

try:
    from .func_model import (
        get_model_config_from_env,
        get_model_run_configs_from_env,
        get_supported_model_names,
        model_training,
        resolve_model_config,
    )
except ModuleNotFoundError:
    get_model_config_from_env = None
    get_model_run_configs_from_env = None
    get_supported_model_names = None
    model_training = None
    resolve_model_config = None

__all__ = [
    "configure_mlflow",
    "date_trend",
    "get_available_data_names",
    "get_model_config_from_env",
    "get_model_run_configs_from_env",
    "get_supported_model_names",
    "get_weighted_df",
    "get_mlflow_config",
    "load_data_df",
    "make_name_from_params",
    "model_training",
    "outlier_remover",
    "plot_boxplots_by_date",
    "resolve_model_config",
    "run_and_log_model",
    "xgboost_feature_importance",
]
