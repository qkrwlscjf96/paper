import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

def xgboost_feature_importance(df, target_col, model_type):
    
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
            random_state=42
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

    model.fit(X_train, y_train)

    importance = model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        importance.items(), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)

    return importance_df
