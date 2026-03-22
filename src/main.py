#%%
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
                    
#%%
from pathlib import Path
import mlflow

from utils import (
    configure_mlflow,
    corr_with_defect,
    date_group_test,
    date_trend,
    get_weighted_df,
    get_mlflow_config,
    load_data_df,
    make_experiment_name,
    outlier_remover,
    parse_multiplier_values,
    run_and_log_model,
    xgboost_feature_importance,
)


base_path = Path(__file__).parent.parent
data_path = base_path / 'data'
file_paths = os.listdir(data_path)
mlflow_config = get_mlflow_config(base_path=base_path)
print(f"MLflow tracking_dir: {mlflow_config['tracking_dir']}")
print(f"MLflow artifact_dir: {mlflow_config['artifact_dir']}")
print(f"MLflow tracking_uri: {mlflow_config['tracking_uri']}")

#TODO 기본꼴 : 날짜 DATE (datetime) / 검사 TAG (1: NG / 0 : OK) / 측정값 칼럼들 -> df / ng_df / target_col / check_cols / date_col
# data_name : '용해탱크', '사출성형기', '살균기', '소성가공'
data_name = os.getenv("DATA_NAME", "사출성형기")
experiment_name = make_experiment_name("weight-multiplier-search", data_name)
print(f"MLflow experiment_name: {experiment_name}")
configure_mlflow(base_path=base_path, experiment_name=experiment_name)
df, ng_df, target_col, check_cols, date_col = load_data_df(data_name=data_name, base_path=base_path)


#%%
# Feature Importance
print("\n")
feature_importance_result = xgboost_feature_importance(df, target_col, model_type="class")

#%%
# Statistical Analysis
print("\n")
static_1_result = date_group_test(ng_df,check_cols)
static_2_result = outlier_remover(df,target_col[0],check_cols)
static_3_result = date_trend(df,check_cols)
static_4_result = corr_with_defect(df, target_col[0], check_cols)


#%%
# MLflow 실험 파라미터
print("\n")
index_mul_values = parse_multiplier_values("INDEX_MULS", [1, 2, 3])
date_mul_values = parse_multiplier_values("DATE_MULS", [1, 2, 3])

print(f"INDEX_MULS: {index_mul_values}")
print(f"DATE_MULS: {date_mul_values}")

# # %%
# # EDA
# plot_boxplots_by_date(df, check_cols)

# %%
# 모델 학습 및 MLflow 기록
print("\n")
parent_run_name = f"{data_name}-search"
with mlflow.start_run(run_name=parent_run_name):
    mlflow.log_params(
        {
            "data_name": data_name,
            "index_mul_values": ",".join(map(str, index_mul_values)),
            "date_mul_values": ",".join(map(str, date_mul_values)),
        }
    )
    mlflow.set_tags({"stage": "grid_search", "data_name": data_name})

    print("일반 모델 학습결과: ")
    baseline_metrics = run_and_log_model(
        run_name=f"{data_name}-baseline",
        df_to_train=df,
        check_cols=check_cols,
        target_col=target_col,
        params={
            "data_name": data_name,
            "weighting": "baseline",
            "index_mul": 1,
            "date_mul": 1,
        },
        tags={
            "stage": "baseline",
            "data_name": data_name,
        },
        nested=True,
    )

    best_run = {
        "run_name": f"{data_name}-baseline",
        "accuracy": baseline_metrics["accuracy"],
        "index_mul": 1,
        "date_mul": 1,
    }

    for index_mul in index_mul_values:
        for date_mul in date_mul_values:
            print("\n")
            print(f"가중치 적용 모델 학습결과: data_name={data_name}, index_mul={index_mul}, date_mul={date_mul}")

            weight_df = get_weighted_df(
                df,
                check_cols,
                index_mul,
                date_mul,
                static_1_result,
                static_2_result,
                static_3_result,
                static_4_result,
                feature_importance_result.copy(),
            )

            weighted_metrics = run_and_log_model(
                run_name=f"{data_name}-index{index_mul}-date{date_mul}",
                df_to_train=weight_df,
                check_cols=check_cols,
                target_col=target_col,
                params={
                    "data_name": data_name,
                    "weighting": "weighted",
                    "index_mul": index_mul,
                    "date_mul": date_mul,
                },
                tags={
                    "stage": "weighted",
                    "data_name": data_name,
                },
                nested=True,
            )

            if weighted_metrics["accuracy"] > best_run["accuracy"]:
                best_run = {
                    "run_name": f"{data_name}-index{index_mul}-date{date_mul}",
                    "accuracy": weighted_metrics["accuracy"],
                    "index_mul": index_mul,
                    "date_mul": date_mul,
                }

    mlflow.log_metric("best_accuracy", best_run["accuracy"])

print("\nBest run summary:")
print(best_run)
# %%
