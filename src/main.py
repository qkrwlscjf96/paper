#%%
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
                    
#%%
from pathlib import Path
import pandas as pd
import numpy as np

from utils import (
    corr_with_defect,
    date_group_test,
    date_trend,
    get_weighted_df,
    load_data_df,
    model_training,
    outlier_remover,
    xgboost_feature_importance,
)

base_path = Path(__file__).parent.parent
data_path = base_path / 'data'
file_paths = os.listdir(data_path)

#TODO 기본꼴 : 날짜 DATE (datetime) / 검사 TAG (1: NG / 0 : OK) / 측정값 칼럼들 -> df / ng_df / target_col / check_cols / date_col
# data_name : '용해탱크', '사출성형기', '살균기', '소성가공'
df, ng_df, target_col, check_cols, date_col = load_data_df(data_name="사출성형기", base_path=base_path)


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
# 가중치 적용된 DataFrame 생성
print("\n")
weight_df = get_weighted_df(df, check_cols, static_1_result, static_2_result, static_3_result, static_4_result, feature_importance_result)

# # %%
# # EDA
# plot_boxplots_by_date(df, check_cols)

# %%
# 모델 학습
print("\n")
print("일반 모델 학습결과: ")
model_training(df, check_cols, target_col)

print("가중치 적용 모델 학습결과: ")
model_training(weight_df, check_cols, target_col)
# %%
