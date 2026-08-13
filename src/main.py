#%%
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.func_common import get_available_data_names, load_data_df
from utils.func_eda import plot_boxplots_by_date
from utils.func_model import (
    get_model_run_configs_from_env,
    resolve_model_names,
    resolve_model_config,
)
from utils.func_static import date_trend, outlier_remover
from utils.func_weight import get_weighted_df


# =============================================================================
# Path Config
# =============================================================================
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / "data"
RESULT_PATH = BASE_PATH / "result"
EDA_RESULT_PATH = RESULT_PATH / "eda"


# =============================================================================
# Run Switch
# 1. Data loader
# 2. Feature importance + Statistical analysis
# 3. EDA
# --- MLflow setup
# 4. Modeling
# =============================================================================
RUN_DATA_LOADER = os.getenv("RUN_DATA_LOADER", "1") == "1"
RUN_EDA = os.getenv("RUN_EDA", "1") == "1"
RUN_ANALYSIS = os.getenv("RUN_ANALYSIS", "1") == "1"
RUN_MODELING = os.getenv("RUN_MODELING", "1") == "1"


# =============================================================================
# Experiment / Data Config
# =============================================================================
DATA_NAME = os.getenv("DATA_NAME", "사출성형기")
DATA_NAMES = os.getenv("DATA_NAMES", DATA_NAME)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "").strip()
EXPERIMENT_PREFIX = os.getenv("EXPERIMENT_PREFIX", "model-weighting-comparison").strip()


# =============================================================================
# Hyperparameter Config
# =============================================================================

# 변수 중요도 관련

FEATURE_IMPORTANCE_F1_THRESHOLD = float(
    os.getenv("FEATURE_IMPORTANCE_F1_THRESHOLD", "0.0")
)
FEATURE_IMPORTANCE_RUNS = int(os.getenv("FEATURE_IMPORTANCE_RUNS", "30"))
FEATURE_IMPORTANCE_TOP_K = int(os.getenv("FEATURE_IMPORTANCE_TOP_K", "10"))

# 통계 가중치 관련
IDX_OUTLIER_DIRECTIONAL_CORR_THRESHOLD = float(
    os.getenv(
        "IDX_OUTLIER_DIRECTIONAL_CORR_THRESHOLD",
        os.getenv("OUTLIER_DIRECTIONAL_CORR_THRESHOLD", "0.3"),
    )
)
DATE_TREND_CONTEXT_DAYS = int(os.getenv("DATE_TREND_CONTEXT_DAYS", "2"))
DATE_TREND_MIN_WINDOW_POINTS = int(os.getenv("DATE_TREND_MIN_WINDOW_POINTS", "4"))
DATE_TREND_MIN_ANCHOR_RATE_QUANTILE = float(
    os.getenv("DATE_TREND_MIN_ANCHOR_RATE_QUANTILE", "0.8")
)
DATE_TREND_MIN_LEVEL_SCORE = float(os.getenv("DATE_TREND_MIN_LEVEL_SCORE", "0.5"))
DATE_TREND_MIN_SIGN_AGREEMENT = float(os.getenv("DATE_TREND_MIN_SIGN_AGREEMENT", "0.5"))

# 실험 조합 관련
BASELINE_INDEX_MUL = int(os.getenv("BASELINE_INDEX_MUL", "2"))
BASELINE_DATE_MUL = int(os.getenv("BASELINE_DATE_MUL", "2"))


if RUN_MODELING:
    from utils.func_mlflow import (
        configure_mlflow,
        get_mlflow_config,
        log_dataframe_artifact,
        make_name_from_params,
        run_and_log_model,
    )

    import mlflow

ANALYSIS_PARAMS = {
    "idx_outlier_directional_corr_threshold": IDX_OUTLIER_DIRECTIONAL_CORR_THRESHOLD,
    "feature_importance_f1_threshold": FEATURE_IMPORTANCE_F1_THRESHOLD,
    "feature_importance_runs": FEATURE_IMPORTANCE_RUNS,
    "feature_importance_top_k": FEATURE_IMPORTANCE_TOP_K,
    "date_trend_context_days": DATE_TREND_CONTEXT_DAYS,
    "date_trend_min_window_points": DATE_TREND_MIN_WINDOW_POINTS,
    "date_trend_min_anchor_rate_quantile": DATE_TREND_MIN_ANCHOR_RATE_QUANTILE,
    "date_trend_min_level_score": DATE_TREND_MIN_LEVEL_SCORE,
    "date_trend_min_sign_agreement": DATE_TREND_MIN_SIGN_AGREEMENT,
}

SELECTED_MODEL_NAMES = resolve_model_names()
WEIGHT_PARAMS = {
    "baseline_index_mul": BASELINE_INDEX_MUL,
    "baseline_date_mul": BASELINE_DATE_MUL
}


def resolve_data_names(raw_data_names: str, data_path: Path) -> list[str]:
    available_data_names = get_available_data_names(data_path)
    selected_data_names = [data_name.strip() for data_name in raw_data_names.split(",") if data_name.strip()]
    if not selected_data_names:
        selected_data_names = [DATA_NAME]

    if len(selected_data_names) == 1 and selected_data_names[0].lower() == "all":
        return available_data_names

    invalid_data_names = [
        data_name for data_name in selected_data_names
        if data_name not in available_data_names
    ]
    if invalid_data_names:
        available = ", ".join(available_data_names)
        invalid = ", ".join(invalid_data_names)
        raise ValueError(f"Unsupported data_name={invalid}. Available: {available}")

    return selected_data_names


def build_experiment_name(
    data_name: str,
    experiment_name: str | None = None,
    experiment_prefix: str | None = None,
) -> str:
    if experiment_name:
        return experiment_name
    prefix = (experiment_prefix or "model-weighting-comparison").strip() or "model-weighting-comparison"
    return f"{prefix}--{data_name}"


def build_parent_run_params(model_run_config: dict) -> dict:
    return {
        **model_run_config,
        **ANALYSIS_PARAMS,
        **WEIGHT_PARAMS,
    }


def build_child_run_params(model_run_config: dict, test_config: dict) -> dict:
    return {
        **build_parent_run_params(model_run_config),
        **test_config,
    }


def build_dataset_tags(data_name: str, df, ng_df, check_cols, date_col: list[str]) -> dict:
    date_key = date_col[0]
    return {
        "data_name": data_name,
        "dataset_rows": str(len(df)),
        "dataset_ng_rows": str(len(ng_df)),
        "dataset_feature_count": str(len(check_cols)),
        "dataset_target_col": "TAG",
        "dataset_date_col": date_key,
        "dataset_date_start": str(df[date_key].min()),
        "dataset_date_end": str(df[date_key].max()),
    }


def build_child_run_tags(weighting: str, dataset_tags: dict) -> dict:
    run_type = "baseline" if weighting == "baseline" else "weighted"
    return {
        **dataset_tags,
        "stage": "modeling",
        "run_type": run_type,
        "comparison_group": "baseline_vs_weighted",
        "weighting": weighting,
    }


def build_weighting_artifacts(weight_diagnostics: dict, weight_df) -> dict[str, object]:
    artifacts = {
        "index_weighted_rows": weight_diagnostics["index_weighted_rows"],
        "date_feature_weighted_rows": weight_diagnostics["date_feature_weighted_rows"],
        "feature_importance_weights": weight_diagnostics["feature_importance_weights"],
        "weighting_summary": weight_diagnostics["weighting_summary"],
        "weighted_data_preview": weight_df.head(100).copy(),
    }

    return {
        name: artifact_df
        for name, artifact_df in artifacts.items()
        if artifact_df is not None
    }


def log_comparison_metrics(metrics_by_weighting: dict) -> None:
    baseline_metrics = metrics_by_weighting.get("baseline")
    weighted_metrics = metrics_by_weighting.get("weighted")

    if baseline_metrics is None or weighted_metrics is None:
        return

    metric_pairs = {
        "test_accuracy": "accuracy",
        "validation_accuracy": "validation_accuracy",
        "cv_accuracy": "cv_accuracy",
        "cv_f1score": "cv_f1_score",
        "cv_precision": "cv_precision",
        "cv_recall": "cv_recall",
    }

    for metric_name, source_key in metric_pairs.items():
        baseline_value = baseline_metrics[source_key]
        weighted_value = weighted_metrics[source_key]
        mlflow.log_metric(f"baseline_{metric_name}", baseline_value)
        mlflow.log_metric(f"weighted_{metric_name}", weighted_value)
        mlflow.log_metric(f"delta_{metric_name}", weighted_value - baseline_value)


run_data_loader = RUN_DATA_LOADER or RUN_EDA or RUN_ANALYSIS or RUN_MODELING
run_analysis = RUN_ANALYSIS or RUN_EDA or RUN_MODELING
SELECTED_DATA_NAMES = resolve_data_names(DATA_NAMES, DATA_PATH)

if RUN_MODELING and not RUN_ANALYSIS:
    print("RUN_MODELING=1 이므로 모델링에 필요한 분석 단계도 함께 실행합니다.")
if RUN_EDA and not RUN_ANALYSIS:
    print("RUN_EDA=1 이므로 EDA 시각화에 필요한 분석 단계도 함께 실행합니다.")


print("Run config:")
print(
    f"DATA_NAMES={SELECTED_DATA_NAMES}, "
    f"RUN_DATA_LOADER={RUN_DATA_LOADER}, "
    f"RUN_EDA={RUN_EDA}, "
    f"RUN_ANALYSIS={RUN_ANALYSIS}, "
    f"RUN_MODELING={RUN_MODELING}"
)
print(
    f"MODEL_NAMES={SELECTED_MODEL_NAMES}"
)
print(
    f"IDX_OUTLIER_DIRECTIONAL_CORR_THRESHOLD={ANALYSIS_PARAMS['idx_outlier_directional_corr_threshold']}, "
    f"FEATURE_IMPORTANCE_F1_THRESHOLD={ANALYSIS_PARAMS['feature_importance_f1_threshold']}, "
    f"FEATURE_IMPORTANCE_RUNS={ANALYSIS_PARAMS['feature_importance_runs']}, "
    f"FEATURE_IMPORTANCE_TOP_K={ANALYSIS_PARAMS['feature_importance_top_k']}, "
    f"DATE_TREND_CONTEXT_DAYS={ANALYSIS_PARAMS['date_trend_context_days']}, "
    f"DATE_TREND_MIN_WINDOW_POINTS={ANALYSIS_PARAMS['date_trend_min_window_points']}, "
    f"DATE_TREND_MIN_ANCHOR_RATE_QUANTILE={ANALYSIS_PARAMS['date_trend_min_anchor_rate_quantile']}, "
    f"DATE_TREND_MIN_LEVEL_SCORE={ANALYSIS_PARAMS['date_trend_min_level_score']}, "
    f"DATE_TREND_MIN_SIGN_AGREEMENT={ANALYSIS_PARAMS['date_trend_min_sign_agreement']}"
)
if RUN_MODELING:
    print(f"BASELINE_INDEX_MUL={BASELINE_INDEX_MUL}")
    print(f"BASELINE_DATE_MUL={BASELINE_DATE_MUL}")


def run_pipeline_for_data(data_name: str) -> None:
    model_run_configs = get_model_run_configs_from_env(data_name=data_name)

    print("\n" + "=" * 80)
    print(f"DATASET: {data_name}")
    print("=" * 80)

    # 1. Data loader
    df = None
    ng_df = None
    target_col = None
    check_cols = None
    date_col = None

    if run_data_loader:
        print("\n[1] Data loader")
        df, ng_df, target_col, check_cols, date_col = load_data_df(
            data_name=data_name,
            data_path=DATA_PATH,
        )
    else:
        print("\n[1] Data loader skipped")

    # 2. Feature importance + Statistical analysis
    feature_importance_result = None
    static_idx_result = None
    static_idx_detail_result = None
    static_date_result = None

    if run_analysis:
        print("\n[2] Feature importance + Statistical analysis")

        try:
            from utils.func_feat_imp import xgboost_feature_importance

            print(
                "Feature importance config: "
                f"f1_threshold={ANALYSIS_PARAMS['feature_importance_f1_threshold']}, "
                f"runs={ANALYSIS_PARAMS['feature_importance_runs']}, "
                f"top_k={ANALYSIS_PARAMS['feature_importance_top_k']}"
            )
            feature_importance_result = xgboost_feature_importance(
                full_df=df,
                target_col=target_col,
                check_cols=check_cols,
                f1_threshold=ANALYSIS_PARAMS["feature_importance_f1_threshold"],
                n_runs=ANALYSIS_PARAMS["feature_importance_runs"],
                top_k=ANALYSIS_PARAMS["feature_importance_top_k"],
            )
        except ModuleNotFoundError as exc:
            feature_importance_result = None
            print(f"Feature importance skipped: {exc}")

        print(
            "Outlier remover config: "
            f"directional_corr_threshold={ANALYSIS_PARAMS['idx_outlier_directional_corr_threshold']}"
        )
        print(
            "Date trend config: "
            f"context_days={ANALYSIS_PARAMS['date_trend_context_days']}, "
            f"min_window_points={ANALYSIS_PARAMS['date_trend_min_window_points']}, "
            f"min_anchor_rate_quantile={ANALYSIS_PARAMS['date_trend_min_anchor_rate_quantile']}, "
            f"min_level_score={ANALYSIS_PARAMS['date_trend_min_level_score']}, "
            f"min_sign_agreement={ANALYSIS_PARAMS['date_trend_min_sign_agreement']}"
        )
        static_idx_result, static_idx_detail_result = outlier_remover(
            df,
            target_col[0],
            check_cols,
            directional_corr_threshold=ANALYSIS_PARAMS["idx_outlier_directional_corr_threshold"],
            return_details=True,
        )
        static_date_result = date_trend(
            df,
            check_cols,
            target_col[0],
            context_days=ANALYSIS_PARAMS["date_trend_context_days"],
            min_window_points=ANALYSIS_PARAMS["date_trend_min_window_points"],
            min_anchor_rate_quantile=ANALYSIS_PARAMS["date_trend_min_anchor_rate_quantile"],
            min_level_score=ANALYSIS_PARAMS["date_trend_min_level_score"],
            min_sign_agreement=ANALYSIS_PARAMS["date_trend_min_sign_agreement"],
        )
    else:
        print("\n[2] Feature importance + Statistical analysis skipped")

    # 3. EDA
    if RUN_EDA:
        print("\n[3] EDA")
        eda_output_dir = EDA_RESULT_PATH / data_name
        print(f"EDA output_dir={eda_output_dir}")
        plot_boxplots_by_date(
            df=df,
            check_cols=check_cols,
            target_col=target_col[0],
            output_dir=eda_output_dir,
            static_idx_result=static_idx_result,
            static_idx_detail_result=static_idx_detail_result,
            static_date_result=static_date_result,
        )
    else:
        print("\n[3] EDA skipped")

    # MLflow setup
    if not RUN_MODELING:
        print("\n[MLflow] Setup skipped")
        print("\n[4] Modeling skipped")
        return

    print("\n[MLflow] Setup")
    mlflow_config = get_mlflow_config(base_path=BASE_PATH)
    print(f"MLflow tracking_dir: {mlflow_config['tracking_dir']}")
    print(f"MLflow artifact_dir: {mlflow_config['artifact_dir']}")
    print(f"MLflow tracking_uri: {mlflow_config['tracking_uri']}")

    experiment_name = build_experiment_name(
        data_name=data_name,
        experiment_name=EXPERIMENT_NAME,
        experiment_prefix=EXPERIMENT_PREFIX,
    )
    print(f"MLflow experiment_name: {experiment_name}")
    mlflow_config = configure_mlflow(base_path=BASE_PATH, experiment_name=experiment_name)
    if mlflow_config.get("resolved_experiment_name") != experiment_name:
        print(
            "MLflow resolved_experiment_name: "
            f"{mlflow_config['resolved_experiment_name']}"
        )

    # 4. Modeling
    print("\n[4] Modeling")

    dataset_tags = build_dataset_tags(data_name, df, ng_df, check_cols, date_col)
    selected_run = None

    weight_df, weight_diagnostics = get_weighted_df(
        df,
        check_cols,
        BASELINE_INDEX_MUL,
        BASELINE_DATE_MUL,
        static_idx_result,
        static_date_result,
        feature_importance_result=(
            feature_importance_result.copy()
            if feature_importance_result is not None
            else None
        ),
        return_diagnostics=True,
    )

    model_test_configs = []
    for model_run_config in model_run_configs:
        model_test_configs.extend(
            [
                {
                    **model_run_config,
                    "weighting": "baseline",
                },
                {
                    **model_run_config,
                    "weighting": "weighted",
                    "index_mul": BASELINE_INDEX_MUL,
                    "date_mul": BASELINE_DATE_MUL,
                },
            ]
        )

    for model_run_config in model_run_configs:
        parent_run_params = build_parent_run_params(model_run_config)
        parent_run_name = model_run_config["model_name"]
        model_name = model_run_config["model_name"]

        with mlflow.start_run(run_name=parent_run_name):
            mlflow.log_params(parent_run_params)
            mlflow.set_tags(
                {
                    **dataset_tags,
                    "stage": "comparison",
                    "comparison_group": "baseline_vs_weighted",
                    "experiment_role": "parent",
                    "model_name": model_name,
                }
            )
            log_dataframe_artifact(
                weight_diagnostics["weighting_summary"],
                artifact_path="weighting_details",
                filename="weighting_summary.csv",
            )

            metrics_by_weighting = {}
            for test_config in model_test_configs:
                if test_config["model_name"] != model_name:
                    continue

                weighting = test_config["weighting"]
                df_to_train = weight_df if weighting == "weighted" else df
                artifact_dataframes = None
                if weighting == "weighted":
                    artifact_dataframes = build_weighting_artifacts(weight_diagnostics, weight_df)

                print("\n")
                message = (
                    f"{model_name} / {weighting} 모델 학습결과: data_name={data_name}"
                )
                if weighting == "weighted":
                    message += (
                        f", index_mul={test_config['index_mul']}, "
                        f"date_mul={test_config['date_mul']}"
                    )
                print(message)

                run_name = (
                    f"{model_name}-baseline"
                    if weighting == "baseline"
                    else f"{model_name}-weighted"
                )
                metrics = run_and_log_model(
                    run_name=run_name,
                    df_to_train=df_to_train,
                    check_cols=check_cols,
                    target_col=target_col,
                    model_name=model_name,
                    model_params=resolve_model_config(model_run_config),
                    params=build_child_run_params(model_run_config, test_config),
                    tags={
                        **build_child_run_tags(weighting, dataset_tags),
                        "model_name": model_name,
                    },
                    artifact_dataframes=artifact_dataframes,
                    nested=True,
                )
                metrics_by_weighting[weighting] = metrics

                if selected_run is None or metrics["cv_accuracy"] > selected_run["cv_accuracy"]:
                    selected_run = {
                        "data_name": data_name,
                        "model_name": model_name,
                        "run_name": run_name,
                        "cv_accuracy": metrics["cv_accuracy"],
                        "index_mul": test_config.get("index_mul"),
                        "date_mul": test_config.get("date_mul"),
                        "params": test_config.copy(),
                    }

            log_comparison_metrics(metrics_by_weighting)

    print("\nSelected run summary:")
    print(selected_run)


for selected_data_name in SELECTED_DATA_NAMES:
    run_pipeline_for_data(selected_data_name)
