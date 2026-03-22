import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

def xgboost_feature_importance(df: pd.DataFrame, target_col : list, model_type: str) -> pd.DataFrame:
    """XGBoost를 활용한 Feature Importance 분석 함수"""
    
    df = df.drop("DATE", axis=1)  # 날짜 컬럼 제거
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "reg":
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )

    model.fit(X_train, y_train)

    importance = model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        importance.items(), columns=["FEATURE", "importance"]
    ).sort_values("importance", ascending=False)
    print("Feature Importance 분석 결과:")
    print(importance_df)
    return importance_df
