#%%
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.func_common import load_data_df
from utils.func_eda import plot_boxplots_by_date
from utils.func_static import date_group_test, date_trend, outlier_remover
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
# 2. EDA
# --- MLflow setup
# 3. Feature importance + Statistical analysis
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
EXPERIMENT_PREFIX = os.getenv("EXPERIMENT_PREFIX", "weight-multiplier-search")


# =============================================================================
# Hyperparameter Config
# =============================================================================
OUTLIER_DIRECTIONAL_CORR_THRESHOLD = float(
    os.getenv("OUTLIER_DIRECTIONAL_CORR_THRESHOLD", "0.3")
)
FEATURE_IMPORTANCE_F1_THRESHOLD = float(
    os.getenv("FEATURE_IMPORTANCE_F1_THRESHOLD", "0.0")
)
FEATURE_IMPORTANCE_RUNS = int(os.getenv("FEATURE_IMPORTANCE_RUNS", "30"))
FEATURE_IMPORTANCE_TOP_K = int(os.getenv("FEATURE_IMPORTANCE_TOP_K", "10"))
DATE_TREND_MIN_DEVIATION = float(os.getenv("DATE_TREND_MIN_DEVIATION", "-2"))
DATE_TREND_WINDOW_SHORT = int(os.getenv("DATE_TREND_WINDOW_SHORT", "3"))
DATE_TREND_WINDOW_LONG = int(os.getenv("DATE_TREND_WINDOW_LONG", "7"))


if RUN_MODELING:
    from utils.func_mlflow import (
        configure_mlflow,
        get_mlflow_config,
        make_experiment_name,
        parse_multiplier_values,
        run_and_log_model,
    )

    import mlflow

    INDEX_MUL_VALUES = parse_multiplier_values("INDEX_MULS", [1, 2, 3])
    DATE_MUL_VALUES = parse_multiplier_values("DATE_MULS", [1, 2, 3])
else:
    INDEX_MUL_VALUES = []
    DATE_MUL_VALUES = []


run_data_loader = RUN_DATA_LOADER or RUN_EDA or RUN_ANALYSIS or RUN_MODELING
run_analysis = RUN_ANALYSIS or RUN_MODELING

if RUN_MODELING and not RUN_ANALYSIS:
    print("RUN_MODELING=1 이므로 모델링에 필요한 분석 단계도 함께 실행합니다.")


print("Run config:")
print(
    f"DATA_NAME={DATA_NAME}, "
    f"RUN_DATA_LOADER={RUN_DATA_LOADER}, "
    f"RUN_EDA={RUN_EDA}, "
    f"RUN_ANALYSIS={RUN_ANALYSIS}, "
    f"RUN_MODELING={RUN_MODELING}"
)
print(
    f"OUTLIER_DIRECTIONAL_CORR_THRESHOLD={OUTLIER_DIRECTIONAL_CORR_THRESHOLD}, "
    f"FEATURE_IMPORTANCE_F1_THRESHOLD={FEATURE_IMPORTANCE_F1_THRESHOLD}, "
    f"FEATURE_IMPORTANCE_RUNS={FEATURE_IMPORTANCE_RUNS}, "
    f"FEATURE_IMPORTANCE_TOP_K={FEATURE_IMPORTANCE_TOP_K}, "
    f"DATE_TREND_MIN_DEVIATION={DATE_TREND_MIN_DEVIATION}, "
    f"DATE_TREND_WINDOW_SHORT={DATE_TREND_WINDOW_SHORT}, "
    f"DATE_TREND_WINDOW_LONG={DATE_TREND_WINDOW_LONG}"
)
if RUN_MODELING:
    print(f"INDEX_MULS={INDEX_MUL_VALUES}")
    print(f"DATE_MULS={DATE_MUL_VALUES}")


#%%
# 1. Data loader
df = None
ng_df = None
target_col = None
check_cols = None
date_col = None

if run_data_loader:
    print("\n[1] Data loader")
    df, ng_df, target_col, check_cols, date_col = load_data_df(
        data_name=DATA_NAME,
        data_path=DATA_PATH,
    )
else:
    print("\n[1] Data loader skipped")


#%%
# 2. EDA
eda_result = {}

if RUN_EDA:
    print("\n[2] EDA")
    eda_output_dir = EDA_RESULT_PATH / DATA_NAME
    print(f"EDA output_dir={eda_output_dir}")
    eda_result = plot_boxplots_by_date(
        df=df,
        check_cols=check_cols,
        target_col=target_col[0],
        output_dir=eda_output_dir,
    )
else:
    print("\n[2] EDA skipped")


#%%
# --- MLflow setup
mlflow_config = None
experiment_name = None

if RUN_MODELING:
    print("\n[MLflow] Setup")
    mlflow_config = get_mlflow_config(base_path=BASE_PATH)
    print(f"MLflow tracking_dir: {mlflow_config['tracking_dir']}")
    print(f"MLflow artifact_dir: {mlflow_config['artifact_dir']}")
    print(f"MLflow tracking_uri: {mlflow_config['tracking_uri']}")

    experiment_name = make_experiment_name(EXPERIMENT_PREFIX, DATA_NAME)
    print(f"MLflow experiment_name: {experiment_name}")
    configure_mlflow(base_path=BASE_PATH, experiment_name=experiment_name)
else:
    print("\n[MLflow] Setup skipped")


#%%
# 3. Feature importance + Statistical analysis
feature_importance_result = None
static_1_result = None
static_2_result = None
static_3_result = None

if run_analysis:
    print("\n[3] Feature importance + Statistical analysis")

    try:
        from utils.func_feat_imp import xgboost_feature_importance

        print(
            "Feature importance config: "
            f"f1_threshold={FEATURE_IMPORTANCE_F1_THRESHOLD}, "
            f"runs={FEATURE_IMPORTANCE_RUNS}, "
            f"top_k={FEATURE_IMPORTANCE_TOP_K}"
        )
        feature_importance_result = xgboost_feature_importance(
            full_df=df,
            target_col=target_col,
            check_cols=check_cols,
            f1_threshold=FEATURE_IMPORTANCE_F1_THRESHOLD,
            n_runs=FEATURE_IMPORTANCE_RUNS,
            top_k=FEATURE_IMPORTANCE_TOP_K,
        )
    except ModuleNotFoundError as exc:
        feature_importance_result = None
        print(f"Feature importance skipped: {exc}")

    print(
        "Outlier remover config: "
        f"directional_corr_threshold={OUTLIER_DIRECTIONAL_CORR_THRESHOLD}"
    )
    print(
        "Date trend config: "
        f"min_deviation={DATE_TREND_MIN_DEVIATION}, "
        f"window_short={DATE_TREND_WINDOW_SHORT}, "
        f"window_long={DATE_TREND_WINDOW_LONG}"
    )
    static_1_result = date_group_test(ng_df, check_cols)
    static_2_result = outlier_remover(
        df,
        target_col[0],
        check_cols,
        directional_corr_threshold=OUTLIER_DIRECTIONAL_CORR_THRESHOLD,
    )
    static_3_result = date_trend(
        df,
        check_cols,
        target_col[0],
        min_deviation=DATE_TREND_MIN_DEVIATION,
        window_short=DATE_TREND_WINDOW_SHORT,
        window_long=DATE_TREND_WINDOW_LONG,
    )
else:
    print("\n[3] Feature importance + Statistical analysis skipped")


#%%
# 4. Modeling
if RUN_MODELING:
    print("\n[4] Modeling")

    parent_run_name = f"{DATA_NAME}-search"
    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_params(
            {
                "data_name": DATA_NAME,
                "index_mul_values": ",".join(map(str, INDEX_MUL_VALUES)),
                "date_mul_values": ",".join(map(str, DATE_MUL_VALUES)),
                "outlier_directional_corr_threshold": OUTLIER_DIRECTIONAL_CORR_THRESHOLD,
                "feature_importance_f1_threshold": FEATURE_IMPORTANCE_F1_THRESHOLD,
                "feature_importance_runs": FEATURE_IMPORTANCE_RUNS,
                "feature_importance_top_k": FEATURE_IMPORTANCE_TOP_K,
            }
        )
        mlflow.set_tags({"stage": "grid_search", "data_name": DATA_NAME})

        print("일반 모델 학습결과:")
        baseline_metrics = run_and_log_model(
            run_name=f"{DATA_NAME}-baseline",
            df_to_train=df,
            check_cols=check_cols,
            target_col=target_col,
            params={
                "data_name": DATA_NAME,
                "weighting": "baseline",
                "index_mul": 1,
                "date_mul": 1,
            },
            tags={
                "stage": "baseline",
                "data_name": DATA_NAME,
            },
            nested=True,
        )

        best_run = {
            "run_name": f"{DATA_NAME}-baseline",
            "accuracy": baseline_metrics["accuracy"],
            "index_mul": 1,
            "date_mul": 1,
        }

        for index_mul in INDEX_MUL_VALUES:
            for date_mul in DATE_MUL_VALUES:
                print("\n")
                print(
                    "가중치 적용 모델 학습결과: "
                    f"data_name={DATA_NAME}, index_mul={index_mul}, date_mul={date_mul}"
                )

                weight_df = get_weighted_df(
                    df,
                    check_cols,
                    index_mul,
                    date_mul,
                    static_1_result,
                    static_2_result,
                    static_3_result,
                    feature_importance_result=(
                        feature_importance_result.copy()
                        if feature_importance_result is not None
                        else None
                    ),
                )

                weighted_metrics = run_and_log_model(
                    run_name=f"{DATA_NAME}-index{index_mul}-date{date_mul}",
                    df_to_train=weight_df,
                    check_cols=check_cols,
                    target_col=target_col,
                    params={
                        "data_name": DATA_NAME,
                        "weighting": "weighted",
                        "index_mul": index_mul,
                        "date_mul": date_mul,
                    },
                    tags={
                        "stage": "weighted",
                        "data_name": DATA_NAME,
                    },
                    nested=True,
                )

                if weighted_metrics["accuracy"] > best_run["accuracy"]:
                    best_run = {
                        "run_name": f"{DATA_NAME}-index{index_mul}-date{date_mul}",
                        "accuracy": weighted_metrics["accuracy"],
                        "index_mul": index_mul,
                        "date_mul": date_mul,
                    }

        mlflow.log_metric("best_accuracy", best_run["accuracy"])

    print("\nBest run summary:")
    print(best_run)
else:
    print("\n[4] Modeling skipped")
