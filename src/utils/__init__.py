"""Public utilities used by ``main.py`` and other modules."""

from .func_common import get_available_data_names, load_data_df
from .func_eda import generate_eda_outputs
from .func_static import (
    FuncStaticV1,
    FuncStaticV2,
    anchor_window_date_trend,
    iqr_remover,
    pchart_remover,
    pelt_cpd_date_trend,
)
from .func_weight import build_statistical_sample_weights

try:
    from .func_feat_imp import xgboost_feature_importance
except ModuleNotFoundError:
    xgboost_feature_importance = None

try:
    from .func_mlflow import (
        configure_mlflow,
        get_mlflow_config,
        make_name_from_params,
        run_and_log_cv_model,
    )
except ModuleNotFoundError:
    configure_mlflow = None
    get_mlflow_config = None
    make_name_from_params = None
    run_and_log_cv_model = None

try:
    from .func_model import (
        build_stratified_cv_folds,
        get_model_config_from_env,
        get_model_run_configs_from_env,
        get_supported_model_names,
        cross_validate_model,
        resolve_model_config,
    )
except ModuleNotFoundError:
    build_stratified_cv_folds = None
    get_model_config_from_env = None
    get_model_run_configs_from_env = None
    get_supported_model_names = None
    cross_validate_model = None
    resolve_model_config = None

__all__ = [
    "configure_mlflow",
    "build_stratified_cv_folds",
    "anchor_window_date_trend",
    "FuncStaticV1",
    "FuncStaticV2",
    "get_available_data_names",
    "get_model_config_from_env",
    "get_model_run_configs_from_env",
    "get_supported_model_names",
    "build_statistical_sample_weights",
    "get_mlflow_config",
    "load_data_df",
    "make_name_from_params",
    "cross_validate_model",
    "iqr_remover",
    "pchart_remover",
    "pelt_cpd_date_trend",
    "generate_eda_outputs",
    "resolve_model_config",
    "run_and_log_cv_model",
    "xgboost_feature_importance",
]
