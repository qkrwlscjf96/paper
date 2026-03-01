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
from utils.func_weight import *


base_path = Path(__file__).parent.parent
data_path = base_path / 'data'
file_path = os.listdir(data_path)

#TODO 기본꼴 : 날짜 DATE (datetime) / 검사 TAG (1: NG / 0 : OK) / 측정값 칼럼들 -> df / ng_df / target_col / check_cols

#용해탱크
file_path_1 = data_path / file_path[0]
df = get_reader(file_path_1).read()

## Case별
df["DATE"] = pd.to_datetime(df["STD_DT"]).dt.normalize()
df = df.drop(columns=["STD_DT"])
target_col = ["TAG"]
check_cols = ["MELT_TEMP","MOTORSPEED","MELT_WEIGHT"]
date_col = ["DATE"]
df[target_col[0]] = df[target_col[0]].apply(lambda x: 1 if x == "NG" else 0)

## 일반
df = df[date_col + check_cols + target_col].dropna()
ng_df = df[df[target_col[0]] == 1]

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

# %%
# EDA
plot_boxplots_by_date(df, check_cols)

# %%
# 모델 학습

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Feature / Target 분리
X = weight_df[check_cols]
y = weight_df[target_col]

# train/test 분리 (시계열이면 shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# 스케일링 (MLP는 필수)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 평가
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# %%
