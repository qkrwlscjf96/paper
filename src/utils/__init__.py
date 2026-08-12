"""Public utilities used by ``main.py`` and other modules."""

from .func_common import load_data_df
from .func_eda import plot_boxplots_by_date
from .func_static import (
    date_group_test,
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
        make_experiment_name,
        parse_multiplier_values,
        run_and_log_model,
    )
except ModuleNotFoundError:
    configure_mlflow = None
    get_mlflow_config = None
    make_experiment_name = None
    parse_multiplier_values = None
    run_and_log_model = None

try:
    from .func_model import model_training
except ModuleNotFoundError:
    model_training = None

__all__ = [
    "configure_mlflow",
    "date_group_test",
    "date_trend",
    "get_weighted_df",
    "get_mlflow_config",
    "load_data_df",
    "make_experiment_name",
    "model_training",
    "outlier_remover",
    "parse_multiplier_values",
    "plot_boxplots_by_date",
    "run_and_log_model",
    "xgboost_feature_importance",
]
