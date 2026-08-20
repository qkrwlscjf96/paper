#%%
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import TextIO

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.func_common import (
    build_pipeline_dataset_tags,
    build_pipeline_experiment_name,
    load_data_df,
    load_pipeline_config,
    log_pipeline_run_configuration,
    resolve_pipeline_data_names,
    run_pipeline_statistical_analysis,
)
from utils.func_eda import generate_eda_outputs
from utils.func_model import (
    build_stratified_cv_folds,
    get_model_run_configs_from_env,
    resolve_model_config,
    resolve_model_names,
)
from utils.func_weight import build_statistical_sample_weights


BASE_PATH = Path(__file__).resolve().parent.parent


class TimestampedTee:
    """Prefix output with wall/elapsed time and write it to multiple streams."""

    def __init__(self, *streams: TextIO):
        self.streams = streams
        self.started_at = perf_counter()
        self.at_line_start = True

    def _prefix(self) -> str:
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")
        elapsed = perf_counter() - self.started_at
        return f"[{timestamp}] [+{elapsed:,.3f}s] "

    def write(self, message: str) -> int:
        if not message:
            return 0

        for chunk in message.splitlines(keepends=True):
            rendered = f"{self._prefix() if self.at_line_start else ''}{chunk}"
            for stream in self.streams:
                stream.write(rendered)
            self.at_line_start = chunk.endswith(("\n", "\r"))
        return len(message)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

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
        "run_pipeline": True,
        "run_eda": False,
    },
    "feature_importance": {
        "f1_threshold": 0.6,
        "runs": 30,
        "top_k": 5,
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
        "sample_weight_mul": 2.0,
    },
}


CONFIG = load_pipeline_config(BASE_PATH, user_inputs=USER_INPUTS)
SELECTED_MODEL_NAMES = resolve_model_names()
SELECTED_DATA_NAMES = resolve_pipeline_data_names(
    raw_data_names=CONFIG.data_names_raw,
    data_name=CONFIG.data_name,
    data_path=CONFIG.paths.data_path,
)

if CONFIG.switches.run_pipeline:
    from utils.func_mlflow import (
        configure_mlflow,
        run_and_log_cv_model,
    )


def load_dataset(data_name: str):
    print(f"[DATA] Loading dataset: {data_name}")
    return load_data_df(
        data_name=data_name,
        data_path=CONFIG.paths.data_path,
    )


def run_analysis(df, target_col: list[str], check_cols: list[str]) -> dict:
    print("[EDA] Running full-dataset analysis")
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
    print("[EDA] Generating outputs")
    eda_output_dir = CONFIG.paths.eda_result_path / CONFIG.static_analysis.version / data_name
    print(f"[EDA] Output directory: {eda_output_dir}")
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


def prepare_modeling_folds(df, target_col: list[str], check_cols: list[str]):
    """Prepare identical baseline/weighted CV folds without validation leakage."""
    baseline_folds = build_stratified_cv_folds(df, target_col)
    weighted_folds = []
    diagnostics = []

    for fold_number, baseline_fold in enumerate(baseline_folds, start=1):
        fold_train_df = baseline_fold["train_df"]
        with redirect_stdout(StringIO()):
            analysis = run_pipeline_statistical_analysis(
                df=fold_train_df,
                target_col=target_col[0],
                check_cols=check_cols,
                config=CONFIG,
            )
        selected_features = analysis["selected_features"]
        sample_weights, fold_diagnostics = build_statistical_sample_weights(
            fold_train_df,
            analysis["static_idx_result"],
            analysis["static_date_result"],
            weight_mul=CONFIG.weight_params["sample_weight_mul"],
        )
        weighted_rows = int((sample_weights > 1.0).sum())
        static_row_count = (
            analysis["static_idx_result"]["INDEX"].nunique()
            if not analysis["static_idx_result"].empty
            else 0
        )
        date_feature_count = len(analysis["static_date_result"])
        index_method = "IQR" if CONFIG.static_analysis.version == "v1" else "P-chart"
        date_method = "high-defect" if CONFIG.static_analysis.version == "v1" else "PELT"
        print(
            f"[FOLD {fold_number}/{len(baseline_folds)}] "
            f"features={selected_features} | "
            f"{index_method}_rows={static_row_count} | "
            f"{date_method}_date_feature_pairs={date_feature_count} | "
            f"weighted_rows={weighted_rows}/{len(fold_train_df)}"
        )

        baseline_fold["check_cols"] = selected_features
        weighted_folds.append(
            {
                "train_df": fold_train_df,
                "valid_df": baseline_fold["valid_df"],
                "check_cols": selected_features,
                "sample_weight": sample_weights,
            }
        )
        diagnostics.append(fold_diagnostics)

    return {"baseline": baseline_folds, "weighted": weighted_folds}, diagnostics


def configure_tracking(data_name: str) -> None:
    experiment_name = build_pipeline_experiment_name(
        data_name=data_name,
        experiment_name=CONFIG.experiment_name,
        experiment_prefix=CONFIG.experiment_prefix,
    )
    mlflow_config = configure_mlflow(
        base_path=CONFIG.paths.base_path,
        experiment_name=experiment_name,
        dataset_name=data_name,
    )
    print(f"[MLFLOW] Tracking URI: {mlflow_config['tracking_uri']}")
    print(f"[MLFLOW] Experiment: {mlflow_config['resolved_experiment_name']}")
    print(f"[MLFLOW] Artifact URI: {mlflow_config['dataset_artifact_uri']}")


def run_cv_comparison(
    data_name: str,
    df,
    ng_df,
    target_col: list[str],
    check_cols: list[str],
    date_col: list[str],
) -> None:
    print("[PIPELINE] Starting fold-local baseline/weighted CV")
    configure_tracking(data_name)
    cv_folds_by_weighting, weighting_diagnostics = prepare_modeling_folds(
        df, target_col, check_cols
    )
    model_run_configs = get_model_run_configs_from_env(data_name=data_name)
    dataset_tags = {
        **build_pipeline_dataset_tags(data_name, df, ng_df, check_cols, date_col),
        "evaluation_method": "fold_local_weighting_5_fold_cv",
        "independent_test_set": "false",
    }

    for model_run_config in model_run_configs:
        model_name = model_run_config["model_name"]
        model_params = resolve_model_config(model_run_config)
        run_params = {
            **model_run_config,
            **CONFIG.analysis_params,
            **CONFIG.weight_params,
        }

        print(
            f"[MODEL] {model_name} | baseline vs weighted | dataset={data_name}"
            f" | sample_weight_mul={CONFIG.weight_params['sample_weight_mul']}"
        )
        artifacts = {
            f"fold_{fold_number}_sample_weights": fold_diagnostics
            for fold_number, fold_diagnostics in enumerate(
                weighting_diagnostics, start=1
            )
        }
        run_and_log_cv_model(
            run_name=model_name,
            cv_folds_by_weighting=cv_folds_by_weighting,
            target_col=target_col,
            model_name=model_name,
            model_params=model_params,
            params=run_params,
            tags={
                **dataset_tags,
                "stage": "modeling",
                "comparison_group": "baseline_vs_weighted",
                "model_name": model_name,
                "evaluation_phase": "fold_local_weighting_cv",
            },
            artifact_dataframes=artifacts,
        )
    print(f"[PIPELINE] Completed dataset: {data_name}")


def run_dataset(data_name: str) -> None:
    print("\n" + "=" * 72)
    print(f"[DATASET] {data_name}")
    print("=" * 72)

    if not (CONFIG.switches.run_pipeline or CONFIG.switches.run_eda):
        print("[SKIP] RUN_PIPELINE=0 and RUN_EDA=0")
        return

    df, ng_df, target_col, check_cols, date_col = load_dataset(data_name)
    if CONFIG.switches.run_eda:
        analysis_results = run_analysis(df, target_col, check_cols)
        run_eda(data_name, df, check_cols, target_col, analysis_results)

    if CONFIG.switches.run_pipeline:
        run_cv_comparison(data_name, df, ng_df, target_col, check_cols, date_col)


def run() -> None:
    log_pipeline_run_configuration(CONFIG, SELECTED_DATA_NAMES, SELECTED_MODEL_NAMES)
    for data_name in SELECTED_DATA_NAMES:
        run_dataset(data_name)


def main() -> None:
    log_dir = CONFIG.paths.result_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    log_path = log_dir / f"pipeline_{run_id}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        output = TimestampedTee(sys.stdout, log_file)
        with redirect_stdout(output), redirect_stderr(output):
            print(f"[LOG] File: {log_path}")
            try:
                run()
            except Exception:
                print("[PIPELINE] Failed; traceback follows")
                traceback.print_exc()
                raise
            finally:
                print("[LOG] Run finished")


if __name__ == "__main__":
    main()
