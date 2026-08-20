import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .func_static import (
    anchor_window_date_trend,
    iqr_remover,
    pchart_remover,
    pelt_cpd_date_trend,
)

"""파일 읽기 (CSV, Excel)"""
# 1추상 클래스 (공통 인터페이스)
class FileReader(ABC):
    def __init__(self, filepath: str):
        self.filepath = filepath

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass


# CSV 전용 클래스
class CSVReader(FileReader):
    def read(self) -> pd.DataFrame:
        print(f"[DATA] Reading CSV: {self.filepath}")
        return pd.read_csv(self.filepath).reset_index(drop=True)


# Excel 전용 클래스
class ExcelReader(FileReader):
    def read(self) -> pd.DataFrame:
        print(f"[DATA] Reading Excel: {self.filepath}")
        return pd.read_excel(self.filepath).reset_index(drop=True)


# Factory 함수 (확장자 기반 객체 생성)
def get_reader(filepath: str) -> FileReader:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return CSVReader(filepath)
    elif ext in [".xlsx", ".xls"]:
        return ExcelReader(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

"""실제 파일 읽기"""

def load_data_df(data_name: str, data_path: str | os.PathLike):
    """
    data_name : '용해탱크', '사출성형기', '살균기', '소성가공'
    return : df, ng_df, target_col, check_cols
    """

    data_dir = Path(data_path)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    supported_suffixes = {".csv", ".xlsx", ".xls"}
    matches = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_suffixes
        and path.stem == data_name
    )
    if not matches:
        available = ", ".join(get_available_data_names(data_dir)) or "(none)"
        raise FileNotFoundError(
            f"Dataset '{data_name}' was not found in {data_dir}. Available: {available}"
        )
    if len(matches) > 1:
        matched_names = ", ".join(path.name for path in matches)
        raise ValueError(
            f"Dataset '{data_name}' is ambiguous. Matching files: {matched_names}"
        )
    file_path = matches[0]

    df = get_reader(str(file_path)).read()

    required_columns = {"DATE", "TAG"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"Dataset '{data_name}' is missing required columns: {', '.join(missing_columns)}"
        )

    # 날짜 컬럼 처리 (STD_DT 예외 처리)
    if data_name != "소성가공":
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize() # yyyy-mm-dd 00:00:00 형태로 변환
    else:
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.floor("min") # yyyy-mm-dd HH:MM:00 형태로 변환

    # 기본 설정
    target_col = ["TAG"]
    date_col = ["DATE"]
    check_cols = [x for x in df.columns if x not in target_col + date_col]
    non_numeric_cols = [col for col in check_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_cols:
        raise ValueError(
            "All feature columns must be numeric. Non-numeric columns: "
            + ", ".join(non_numeric_cols)
        )

    # TAG → 1/0 변환
    df[target_col[0]] = (df[target_col[0]] == "NG").astype(int)

    # 결측치 처리 (수치형만 평균 대체)
    df = df.fillna(df.mean(numeric_only=True))

    # NG 데이터
    ng_df = df[df[target_col[0]] == 1]

    return df, ng_df, target_col, check_cols, date_col


def get_available_data_names(data_path: str | os.PathLike) -> list[str]:
    data_dir = Path(data_path)
    supported_suffixes = {".csv", ".xlsx", ".xls"}
    data_names = sorted(
        file_path.stem
        for file_path in data_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes
    )
    return data_names


"""파이프라인 설정 / 실행 공통"""


@dataclass(frozen=True)
class PathConfig:
    base_path: Path
    data_path: Path
    result_path: Path
    eda_result_path: Path


@dataclass(frozen=True)
class RunSwitches:
    run_pipeline: bool
    run_eda: bool


@dataclass(frozen=True)
class StaticAnalysisConfig:
    version: str
    index_fn: Callable
    date_fn: Callable
    params: dict[str, object]


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathConfig
    switches: RunSwitches
    data_name: str
    data_names_raw: str
    experiment_name: str
    experiment_prefix: str
    feature_importance_params: dict[str, object]
    static_analysis: StaticAnalysisConfig
    weight_params: dict[str, object]

    @property
    def analysis_params(self) -> dict[str, object]:
        return {
            **self.static_analysis.params,
            **self.feature_importance_params,
        }


def _read_user_input(user_inputs: dict[str, Any] | None, *keys: str) -> Any:
    current = user_inputs or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _resolve_user_or_env(
    user_inputs: dict[str, Any] | None,
    input_keys: tuple[str, ...],
    env_key: str,
    default: Any,
    caster: Callable[[Any], Any] | None = None,
) -> Any:
    env_value = os.getenv(env_key)
    if env_value is not None:
        return caster(env_value) if caster is not None else env_value
    user_value = _read_user_input(user_inputs, *input_keys)
    if user_value is not None:
        return caster(user_value) if caster is not None else user_value
    return caster(default) if caster is not None else default


def load_pipeline_config(base_path: Path, user_inputs: dict[str, Any] | None = None) -> PipelineConfig:
    result_path = base_path / "result"
    eda_result_path = result_path / "eda"
    eda_output_subdir = _resolve_user_or_env(
        user_inputs,
        ("data", "eda_output_subdir"),
        "EDA_OUTPUT_SUBDIR",
        "",
        str,
    ).strip()
    if eda_output_subdir:
        eda_result_path = eda_result_path / eda_output_subdir

    run_pipeline = _resolve_user_or_env(
        user_inputs, ("steps", "run_pipeline"), "RUN_PIPELINE", True, lambda value: str(value) == "1" if isinstance(value, str) else bool(value)
    )
    run_eda = _resolve_user_or_env(
        user_inputs, ("steps", "run_eda"), "RUN_EDA", False, lambda value: str(value) == "1" if isinstance(value, str) else bool(value)
    )
    switches = RunSwitches(
        run_pipeline=run_pipeline,
        run_eda=run_eda,
    )

    static_version = _resolve_user_or_env(
        user_inputs, ("static", "version"), "STATIC_VERSION", "v2", str
    ).strip().lower()
    if static_version == "v1":
        static_analysis = StaticAnalysisConfig(
            version=static_version,
            index_fn=iqr_remover,
            date_fn=anchor_window_date_trend,
            params={
                "static_version": static_version,
                "iqr_directional_corr_threshold": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v1", "iqr_directional_corr_threshold"),
                    "IQR_DIRECTIONAL_CORR_THRESHOLD",
                    os.getenv("OUTLIER_DIRECTIONAL_CORR_THRESHOLD", "0.3"),
                    float,
                ),
                "anchor_context_days": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v1", "anchor_context_days"),
                    "ANCHOR_CONTEXT_DAYS",
                    os.getenv("DATE_TREND_CONTEXT_DAYS", "2"),
                    int,
                ),
                "anchor_min_window_points": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v1", "anchor_min_window_points"),
                    "ANCHOR_MIN_WINDOW_POINTS",
                    os.getenv("DATE_TREND_MIN_WINDOW_POINTS", "4"),
                    int,
                ),
                "anchor_min_rate_quantile": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v1", "anchor_min_rate_quantile"),
                    "ANCHOR_MIN_RATE_QUANTILE",
                    os.getenv("DATE_TREND_MIN_ANCHOR_RATE_QUANTILE", "0.8"),
                    float,
                ),
                "anchor_min_level_score": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v1", "anchor_min_level_score"),
                    "ANCHOR_MIN_LEVEL_SCORE",
                    os.getenv("DATE_TREND_MIN_LEVEL_SCORE", "0.5"),
                    float,
                ),
                "anchor_min_sign_agreement": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v1", "anchor_min_sign_agreement"),
                    "ANCHOR_MIN_SIGN_AGREEMENT",
                    os.getenv("DATE_TREND_MIN_SIGN_AGREEMENT", "0.5"),
                    float,
                ),
            },
        )
    elif static_version == "v2":
        static_analysis = StaticAnalysisConfig(
            version=static_version,
            index_fn=pchart_remover,
            date_fn=pelt_cpd_date_trend,
            params={
                "static_version": static_version,
                "p_chart_sigma_level": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v2", "p_chart_sigma_level"),
                    "P_CHART_SIGMA_LEVEL",
                    "3.0",
                    float,
                ),
                "p_chart_min_subgroup_size": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v2", "p_chart_min_subgroup_size"),
                    "P_CHART_MIN_SUBGROUP_SIZE",
                    "1",
                    int,
                ),
                "pelt_penalty_scale": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v2", "pelt_penalty_scale"),
                    "PELT_PENALTY_SCALE",
                    "1.0",
                    float,
                ),
                "pelt_min_segment_size": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v2", "pelt_min_segment_size"),
                    "PELT_MIN_SEGMENT_SIZE",
                    "3",
                    int,
                ),
                "pelt_change_point_tolerance_days": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v2", "pelt_change_point_tolerance_days"),
                    "PELT_CHANGE_POINT_TOLERANCE_DAYS",
                    "1",
                    int,
                ),
                "pelt_min_effect_size": _resolve_user_or_env(
                    user_inputs,
                    ("static", "v2", "pelt_min_effect_size"),
                    "PELT_MIN_EFFECT_SIZE",
                    "0.0",
                    float,
                ),
            },
        )
    else:
        raise ValueError("STATIC_VERSION must be one of: v1, v2")

    return PipelineConfig(
        paths=PathConfig(
            base_path=base_path,
            data_path=base_path / "data",
            result_path=result_path,
            eda_result_path=eda_result_path,
        ),
        switches=switches,
        data_name=_resolve_user_or_env(user_inputs, ("data", "data_name"), "DATA_NAME", "사출성형기", str),
        data_names_raw=_resolve_user_or_env(
            user_inputs,
            ("data", "data_names"),
            "DATA_NAMES",
            _resolve_user_or_env(user_inputs, ("data", "data_name"), "DATA_NAME", "사출성형기", str),
            str,
        ),
        experiment_name=_resolve_user_or_env(user_inputs, ("data", "experiment_name"), "EXPERIMENT_NAME", "", str).strip(),
        experiment_prefix=_resolve_user_or_env(
            user_inputs,
            ("data", "experiment_prefix"),
            "EXPERIMENT_PREFIX",
            "model-weighting-comparison",
            str,
        ).strip(),
        feature_importance_params={
            "feature_importance_f1_threshold": _resolve_user_or_env(
                user_inputs, ("feature_importance", "f1_threshold"), "FEATURE_IMPORTANCE_F1_THRESHOLD", "0.6", float
            ),
            "feature_importance_runs": _resolve_user_or_env(
                user_inputs, ("feature_importance", "runs"), "FEATURE_IMPORTANCE_RUNS", "30", int
            ),
            "feature_importance_top_k": _resolve_user_or_env(
                user_inputs, ("feature_importance", "top_k"), "FEATURE_IMPORTANCE_TOP_K", "5", int
            ),
        },
        weight_params={
            "sample_weight_mul": _resolve_user_or_env(
                user_inputs,
                ("weighting", "sample_weight_mul"),
                "SAMPLE_WEIGHT_MUL",
                "2.0",
                float,
            ),
        },
        static_analysis=static_analysis,
    )


def resolve_pipeline_data_names(raw_data_names: str, data_name: str, data_path: Path) -> list[str]:
    available_data_names = get_available_data_names(data_path)
    selected_data_names = [name.strip() for name in raw_data_names.split(",") if name.strip()]
    if not selected_data_names:
        selected_data_names = [data_name]

    if len(selected_data_names) == 1 and selected_data_names[0].lower() == "all":
        return available_data_names

    invalid_data_names = [
        selected_name for selected_name in selected_data_names
        if selected_name not in available_data_names
    ]
    if invalid_data_names:
        available = ", ".join(available_data_names)
        invalid = ", ".join(invalid_data_names)
        raise ValueError(f"Unsupported data_name={invalid}. Available: {available}")

    return selected_data_names


def build_pipeline_experiment_name(
    data_name: str,
    experiment_name: str | None = None,
    experiment_prefix: str | None = None,
) -> str:
    if experiment_name:
        return experiment_name
    prefix = (experiment_prefix or "model-weighting-comparison").strip() or "model-weighting-comparison"
    return f"{prefix}--{data_name}"


def build_pipeline_parent_run_params(model_run_config: dict, config: PipelineConfig) -> dict:
    return {
        **model_run_config,
        **config.analysis_params,
        **config.weight_params,
    }


def build_pipeline_child_run_params(model_run_config: dict, test_config: dict, config: PipelineConfig) -> dict:
    return {
        **build_pipeline_parent_run_params(model_run_config, config),
        **test_config,
    }


def build_pipeline_dataset_tags(data_name: str, df, ng_df, check_cols, date_col: list[str]) -> dict:
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


def build_pipeline_child_run_tags(weighting: str, dataset_tags: dict) -> dict:
    run_type = "baseline" if weighting == "baseline" else "weighted"
    return {
        **dataset_tags,
        "stage": "modeling",
        "run_type": run_type,
        "comparison_group": "baseline_vs_weighted",
        "weighting": weighting,
    }


def build_pipeline_model_test_configs(model_run_configs: list[dict], config: PipelineConfig) -> list[dict]:
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
                    "sample_weight_mul": config.weight_params["sample_weight_mul"],
                },
            ]
        )
    return model_test_configs


def log_pipeline_run_configuration(
    config: PipelineConfig,
    selected_data_names: list[str],
    selected_model_names: list[str],
) -> None:
    print("[CONFIG] Run settings")
    print(
        f"[CONFIG] datasets={selected_data_names} | "
        f"pipeline={config.switches.run_pipeline} | eda={config.switches.run_eda}"
    )
    print(f"[CONFIG] models={selected_model_names}")
    print(
        f"[CONFIG] static={config.static_analysis.version} | "
        f"fi_threshold={config.feature_importance_params['feature_importance_f1_threshold']} | "
        f"fi_runs={config.feature_importance_params['feature_importance_runs']} | "
        f"top_k={config.feature_importance_params['feature_importance_top_k']}"
    )

    params = config.static_analysis.params
    if config.static_analysis.version == "v1":
        print(
            f"IQR_DIRECTIONAL_CORR_THRESHOLD={params['iqr_directional_corr_threshold']}, "
            f"ANCHOR_CONTEXT_DAYS={params['anchor_context_days']}, "
            f"ANCHOR_MIN_WINDOW_POINTS={params['anchor_min_window_points']}, "
            f"ANCHOR_MIN_RATE_QUANTILE={params['anchor_min_rate_quantile']}, "
            f"ANCHOR_MIN_LEVEL_SCORE={params['anchor_min_level_score']}, "
            f"ANCHOR_MIN_SIGN_AGREEMENT={params['anchor_min_sign_agreement']}"
        )
    else:
        print(
            f"P_CHART_SIGMA_LEVEL={params['p_chart_sigma_level']}, "
            f"P_CHART_MIN_SUBGROUP_SIZE={params['p_chart_min_subgroup_size']}, "
            f"PELT_PENALTY_SCALE={params['pelt_penalty_scale']}, "
            f"PELT_MIN_SEGMENT_SIZE={params['pelt_min_segment_size']}, "
            f"PELT_CHANGE_POINT_TOLERANCE_DAYS={params['pelt_change_point_tolerance_days']}, "
            f"PELT_MIN_EFFECT_SIZE={params['pelt_min_effect_size']}"
        )

    if config.switches.run_pipeline:
        print(f"[CONFIG] sample_weight_mul={config.weight_params['sample_weight_mul']}")


def run_pipeline_statistical_analysis(df, target_col: str, check_cols: list[str], config: PipelineConfig) -> dict:
    feature_importance_result = None
    try:
        from utils.func_feat_imp import xgboost_feature_importance

        print(
            "[ANALYSIS] Feature importance: "
            f"f1_threshold={config.feature_importance_params['feature_importance_f1_threshold']}, "
            f"runs={config.feature_importance_params['feature_importance_runs']}, "
            f"top_k={config.feature_importance_params['feature_importance_top_k']}"
        )
        feature_importance_result = xgboost_feature_importance(
            full_df=df,
            target_col=[target_col],
            check_cols=check_cols,
            f1_threshold=config.feature_importance_params["feature_importance_f1_threshold"],
            n_runs=config.feature_importance_params["feature_importance_runs"],
            top_k=config.feature_importance_params["feature_importance_top_k"],
        )
    except ModuleNotFoundError as exc:
        print(f"[ANALYSIS] Feature importance skipped: {exc}")

    top_k = config.feature_importance_params["feature_importance_top_k"]
    if feature_importance_result is not None and not feature_importance_result.empty:
        selected_features = (
            feature_importance_result.head(top_k)["FEATURE"].tolist()
        )
    else:
        selected_features = list(check_cols[:top_k])
    if not selected_features:
        raise ValueError("No features are available for statistical analysis")
    print(f"[ANALYSIS] Selected features ({len(selected_features)}): {selected_features}")

    params = config.static_analysis.params
    if config.static_analysis.version == "v1":
        print(
            "[ANALYSIS] IQR: "
            f"directional_corr_threshold={params['iqr_directional_corr_threshold']}"
        )
        print(
            "[ANALYSIS] Anchor window: "
            f"context_days={params['anchor_context_days']}, "
            f"min_window_points={params['anchor_min_window_points']}, "
            f"min_rate_quantile={params['anchor_min_rate_quantile']}, "
            f"min_level_score={params['anchor_min_level_score']}, "
            f"min_sign_agreement={params['anchor_min_sign_agreement']}"
        )
        static_idx_result, static_idx_detail_result = config.static_analysis.index_fn(
            df,
            target_col,
            selected_features,
            directional_corr_threshold=params["iqr_directional_corr_threshold"],
            return_details=True,
        )
        static_date_result = config.static_analysis.date_fn(
            df,
            selected_features,
            target_col,
            context_days=params["anchor_context_days"],
            min_window_points=params["anchor_min_window_points"],
            min_anchor_rate_quantile=params["anchor_min_rate_quantile"],
            min_level_score=params["anchor_min_level_score"],
            min_sign_agreement=params["anchor_min_sign_agreement"],
        )
    else:
        print(
            "[ANALYSIS] P-chart: "
            f"sigma_level={params['p_chart_sigma_level']}, "
            f"min_outlier_features={params['p_chart_min_subgroup_size']}"
        )
        print(
            "[ANALYSIS] PELT: "
            f"penalty_scale={params['pelt_penalty_scale']}, "
            f"min_segment_size={params['pelt_min_segment_size']}, "
            f"change_point_tolerance_days={params['pelt_change_point_tolerance_days']}, "
            f"min_effect_size={params['pelt_min_effect_size']}"
        )
        static_idx_result, static_idx_detail_result = config.static_analysis.index_fn(
            df,
            target_col,
            check_cols=selected_features,
            sigma_level=params["p_chart_sigma_level"],
            min_outlier_features=params["p_chart_min_subgroup_size"],
            return_details=True,
        )
        static_date_result = config.static_analysis.date_fn(
            df,
            selected_features,
            target_col,
            penalty_scale=params["pelt_penalty_scale"],
            min_segment_size=params["pelt_min_segment_size"],
            change_point_tolerance_days=params["pelt_change_point_tolerance_days"],
            min_effect_size=params["pelt_min_effect_size"],
        )

    return {
        "feature_importance_result": feature_importance_result,
        "selected_features": selected_features,
        "static_idx_result": static_idx_result,
        "static_idx_detail_result": static_idx_detail_result,
        "static_date_result": static_date_result,
    }
