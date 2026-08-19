#%%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.func_common import (
    build_pipeline_child_run_params,
    build_pipeline_child_run_tags,
    build_pipeline_dataset_tags,
    build_pipeline_experiment_name,
    build_pipeline_model_test_configs,
    build_pipeline_parent_run_params,
    build_pipeline_weighting_artifacts,
    load_data_df,
    load_pipeline_config,
    log_pipeline_comparison_metrics,
    log_pipeline_run_configuration,
    resolve_pipeline_data_names,
    run_pipeline_statistical_analysis,
)
from utils.func_eda import generate_eda_outputs
from utils.func_model import (
    get_model_run_configs_from_env,
    resolve_model_config,
    resolve_model_names,
)
from utils.func_weight import get_weighted_df


BASE_PATH = Path(__file__).resolve().parent.parent

# =============================================================================
# User Inputs
# 여기만 수정해서 실행 범위와 파라미터를 제어합니다.
# =============================================================================
USER_INPUTS = {
    "data": {
        "data_name": "사출성형기",
        "data_names": "사출성형기",  # 예: "all", "사출성형기,살균기"
        "eda_output_subdir": "",
        "experiment_name": "",
        "experiment_prefix": "model-weighting-comparison",
    },
    "steps": {
        "run_data_loader": True,
        "run_analysis": True,
        "run_eda": True,
        "run_modeling": False,
    },
    "feature_importance": {
        "f1_threshold": 0.0,
        "runs": 30,
        "top_k": 10,
    },
    "static": {
        "version": "v2",  # "v1" or "v2"
        "v1": {
            "iqr_directional_corr_threshold": 0.3,
            "anchor_context_days": 2,
            "anchor_min_window_points": 4,
            "anchor_min_rate_quantile": 0.8,
            "anchor_min_level_score": 0.5,
            "anchor_min_sign_agreement": 0.5,
        },
        "v2": {
            "p_chart_sigma_level": 3.0,
            "p_chart_min_subgroup_size": 1,
            "pelt_penalty_scale": 1.0,
            "pelt_min_segment_size": 3,
            "pelt_change_point_tolerance_days": 1,
            "pelt_min_effect_size": 0.0,
        },
    },
    "weighting": {
        "baseline_index_mul": 2,
        "baseline_date_mul": 2,
    },
}


CONFIG = load_pipeline_config(BASE_PATH, user_inputs=USER_INPUTS)
SELECTED_MODEL_NAMES = resolve_model_names()
SELECTED_DATA_NAMES = resolve_pipeline_data_names(
    raw_data_names=CONFIG.data_names_raw,
    data_name=CONFIG.data_name,
    data_path=CONFIG.paths.data_path,
)

if CONFIG.switches.run_modeling:
    from utils.func_mlflow import (
        configure_mlflow,
        get_mlflow_config,
        log_dataframe_artifact,
        run_and_log_model,
    )

    import mlflow


def load_dataset(data_name: str):
    print("\n[1] Data loader")
    return load_data_df(
        data_name=data_name,
        data_path=CONFIG.paths.data_path,
    )


def run_analysis(df, target_col: list[str], check_cols: list[str]) -> dict:
    print("\n[2] Feature importance + Statistical analysis")
    return run_pipeline_statistical_analysis(
        df=df,
        target_col=target_col[0],
        check_cols=check_cols,
        config=CONFIG,
    )


def run_eda(
    data_name: str,
    df,
    check_cols: list[str],
    target_col: list[str],
    analysis_results: dict,
) -> None:
    print("\n[3] EDA")
    eda_output_dir = CONFIG.paths.eda_result_path / CONFIG.static_analysis.version / data_name
    print(f"EDA output_dir={eda_output_dir}")
    generate_eda_outputs(
        df=df,
        check_cols=check_cols,
        target_col=target_col[0],
        output_dir=eda_output_dir,
        feature_importance_result=analysis_results["feature_importance_result"],
        static_idx_result=analysis_results["static_idx_result"],
        static_idx_detail_result=analysis_results["static_idx_detail_result"],
        static_date_result=analysis_results["static_date_result"],
    )


def run_modeling(
    data_name: str,
    df,
    ng_df,
    target_col: list[str],
    check_cols: list[str],
    date_col: list[str],
    analysis_results: dict,
) -> None:
    print("\n[MLflow] Setup")
    mlflow_config = get_mlflow_config(base_path=CONFIG.paths.base_path)
    print(f"MLflow tracking_dir: {mlflow_config['tracking_dir']}")
    print(f"MLflow artifact_dir: {mlflow_config['artifact_dir']}")
    print(f"MLflow tracking_uri: {mlflow_config['tracking_uri']}")

    experiment_name = build_pipeline_experiment_name(
        data_name=data_name,
        experiment_name=CONFIG.experiment_name,
        experiment_prefix=CONFIG.experiment_prefix,
    )
    print(f"MLflow experiment_name: {experiment_name}")
    mlflow_config = configure_mlflow(
        base_path=CONFIG.paths.base_path,
        experiment_name=experiment_name,
    )
    if mlflow_config.get("resolved_experiment_name") != experiment_name:
        print(f"MLflow resolved_experiment_name: {mlflow_config['resolved_experiment_name']}")

    print("\n[4] Modeling")
    model_run_configs = get_model_run_configs_from_env(data_name=data_name)
    dataset_tags = build_pipeline_dataset_tags(data_name, df, ng_df, check_cols, date_col)
    selected_run = None

    weight_df, weight_diagnostics = get_weighted_df(
        df,
        check_cols,
        CONFIG.weight_params["baseline_index_mul"],
        CONFIG.weight_params["baseline_date_mul"],
        analysis_results["static_idx_result"],
        analysis_results["static_date_result"],
        feature_importance_result=(
            analysis_results["feature_importance_result"].copy()
            if analysis_results["feature_importance_result"] is not None
            else None
        ),
        return_diagnostics=True,
    )

    model_test_configs = build_pipeline_model_test_configs(model_run_configs, CONFIG)

    for model_run_config in model_run_configs:
        parent_run_params = build_pipeline_parent_run_params(model_run_config, CONFIG)
        model_name = model_run_config["model_name"]

        with mlflow.start_run(run_name=model_name):
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
                    artifact_dataframes = build_pipeline_weighting_artifacts(weight_diagnostics, weight_df)
                feature_weights = None
                if weighting == "weighted" and not weight_diagnostics["feature_importance_weights"].empty:
                    feature_weights = dict(
                        zip(
                            weight_diagnostics["feature_importance_weights"]["FEATURE"],
                            weight_diagnostics["feature_importance_weights"]["WEIGHT"],
                        )
                    )

                print("\n")
                message = f"{model_name} / {weighting} 모델 학습결과: data_name={data_name}"
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
                    params=build_pipeline_child_run_params(model_run_config, test_config, CONFIG),
                    tags={
                        **build_pipeline_child_run_tags(weighting, dataset_tags),
                        "model_name": model_name,
                    },
                    artifact_dataframes=artifact_dataframes,
                    feature_weights=feature_weights,
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

            log_pipeline_comparison_metrics(metrics_by_weighting, mlflow)

    print("\nSelected run summary:")
    print(selected_run)


def run_pipeline_for_data(data_name: str) -> None:
    print("\n" + "=" * 80)
    print(f"DATASET: {data_name}")
    print("=" * 80)

    df = ng_df = target_col = check_cols = date_col = None
    analysis_results = {
        "feature_importance_result": None,
        "static_idx_result": None,
        "static_idx_detail_result": None,
        "static_date_result": None,
    }

    if CONFIG.switches.effective_run_data_loader:
        df, ng_df, target_col, check_cols, date_col = load_dataset(data_name)
    else:
        print("\n[1] Data loader skipped")

    if CONFIG.switches.effective_run_analysis:
        analysis_results = run_analysis(df, target_col, check_cols)
    else:
        print("\n[2] Feature importance + Statistical analysis skipped")

    if CONFIG.switches.run_eda:
        run_eda(data_name, df, check_cols, target_col, analysis_results)
    else:
        print("\n[3] EDA skipped")

    if CONFIG.switches.run_modeling:
        run_modeling(data_name, df, ng_df, target_col, check_cols, date_col, analysis_results)
    else:
        print("\n[MLflow] Setup skipped")
        print("\n[4] Modeling skipped")


def main() -> None:
    log_pipeline_run_configuration(CONFIG, SELECTED_DATA_NAMES, SELECTED_MODEL_NAMES)
    for data_name in SELECTED_DATA_NAMES:
        run_pipeline_for_data(data_name)


if __name__ == "__main__":
    main()
