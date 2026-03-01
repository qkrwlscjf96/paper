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
from utils.func_common import *
from utils.func_static import *
from utils.func_eda import *
from utils.func_feat_imp import *


base_path = Path(__file__).parent.parent
data_path = base_path / 'data'
file_path = os.listdir(data_path)

#TODO 기본꼴 : 날짜 DATE (datetime) / 검사 TAG (1: NG / 0 : OK) / 측정값 칼럼들 -> df / ng_df / target_col / check_cols

#용해탱크
file_path_1 = data_path / file_path[0]
df = get_reader(file_path_1).read()

target_col = "TAG"

df["DATE"] = pd.to_datetime(df["STD_DT"]).dt.date
df = df.drop(columns=["STD_DT"])
df = df[["DATE", "MELT_TEMP", "MOTORSPEED", "MELT_WEIGHT", target_col]].dropna()
df[target_col] = df[target_col].apply(lambda x: 1 if x == "NG" else 0)

ng_df = df[df[target_col] == 1]
check_cols = df.columns.difference(["DATE", target_col])


#%%
# Feature Importance

feature_importance_result = xgboost_feature_importance(df, target_col, model_type="class")

#%%
# Statistical Analysis

static_1_result = date_group_test(ng_df,check_cols)
static_2_result = outlier_remover(df,target_col,check_cols)
static_3_result = date_trend(df,check_cols)
static_4_result = corr_with_defect(df, target_col, check_cols)

# %%
# EDA
plot_boxplots_by_date(df, check_cols)

