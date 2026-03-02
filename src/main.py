#%%
import sys
import os

from pyparsing import col
ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
                    
#%%
from pathlib import Path
import pandas as pd
import numpy as np
from utils.func_common import *
from utils.func_static import *
from utils.func_eda import *
from utils.func_feat_imp import *
from utils.func_weight import *
from utils.func_model import *


base_path = Path(__file__).parent.parent
data_path = base_path / 'data'
file_paths = os.listdir(data_path)

#TODO 기본꼴 : 날짜 DATE (datetime) / 검사 TAG (1: NG / 0 : OK) / 측정값 칼럼들 -> df / ng_df / target_col / check_cols

# #용해탱크
# file_path= data_path / file_paths[1]
# df = get_reader(file_path).read()

# ## Case별
# df["DATE"] = pd.to_datetime(df["STD_DT"]).dt.normalize()
# df = df.drop(columns=["STD_DT"])
# target_col = ["TAG"]
# check_cols = ["MELT_TEMP","MOTORSPEED","MELT_WEIGHT"]
# date_col = ["DATE"]
# df[target_col[0]] = df[target_col[0]].apply(lambda x: 1 if x == "NG" else 0)

# ## 일반
# df = df.fillna(df.mean())
# ng_df = df[df[target_col[0]].astype(int) == 1]


#사출성형기
file_path = data_path / file_paths[0]
df = get_reader(file_path).read()

## Case별
df["DATE"] = pd.to_datetime(df["DATE"]).dt.normalize()
target_col = ["TAG"]
date_col = ["DATE"]
check_cols = [x for x in df.columns.tolist() if x not in target_col + date_col]
df[target_col[0]] = df[target_col[0]].apply(lambda x: 1 if x == "NG" else 0)

## 일반
df = df.fillna(df.mean())
ng_df = df[df[target_col[0]].astype(int) == 1]

#%%
# Feature Importance
feature_importance_result = xgboost_feature_importance(df, target_col, model_type="class")

#%%
# Statistical Analysis
static_1_result = date_group_test(ng_df,check_cols)
static_2_result = outlier_remover(df,target_col[0],check_cols)
static_3_result = date_trend(df,check_cols)
static_4_result = corr_with_defect(df, target_col[0], check_cols)


#%%
# 가중치 적용된 DataFrame 생성
weight_df = get_weighted_df(df, check_cols, static_1_result, static_2_result, static_3_result, static_4_result, feature_importance_result)

# # %%
# # EDA
# plot_boxplots_by_date(df, check_cols)

# %%
# 모델 학습
print("일반 모델 학습결과: ")
model_training(df, check_cols, target_col)

print("가중치 적용 모델 학습결과: ")
model_training(weight_df, check_cols, target_col)
# %%
