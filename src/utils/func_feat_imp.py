import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def xgboost_feature_importance(
    full_df: pd.DataFrame,
    target_col: list,
    check_cols: list,
    f1_threshold: float = 0.0,
    n_runs: int = 30,
    top_k: int = 10,
) -> pd.DataFrame:
    """XGBoost 기반 feature importance를 반복 실행 후 집계한다."""

    X = full_df[check_cols].copy()
    y = full_df[target_col[0]]

    aggregate = {
        feature: {
            "importance_sum": 0.0,
            "selected_count": 0,
            "top_k_count": 0,
            "rank_sum": 0.0,
            "best_rank": None,
        }
        for feature in X.columns
    }
    run_scores = []
    selected_run_count = 0
    score_label = "F1"
    best_run_score = float("-inf")
    best_run_importance = None

    for run_idx in range(n_runs):
        random_state = 42 + run_idx
        split_kwargs = {
            "test_size": 0.2,
            "random_state": random_state,
            "stratify": y,
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state,
            n_jobs=1,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        run_score = f1_score(y_test, y_pred, average="binary", zero_division=0)
        run_scores.append(run_score)

        raw_importance = model.get_booster().get_score(importance_type="gain")
        run_importance = (
            pd.DataFrame(raw_importance.items(), columns=["FEATURE", "importance"])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        is_better_run = run_score > best_run_score
        if is_better_run:
            best_run_score = run_score
            best_run_importance = run_importance.copy()

        if run_score < f1_threshold:
            continue

        selected_run_count += 1
        top_features = set(run_importance.head(top_k)["FEATURE"])

        for rank, row in enumerate(run_importance.itertuples(index=False), start=1):
            feature = row.FEATURE
            aggregate[feature]["importance_sum"] += float(row.importance)
            aggregate[feature]["selected_count"] += 1
            aggregate[feature]["rank_sum"] += rank
            aggregate[feature]["best_rank"] = rank if aggregate[feature]["best_rank"] is None else min(
                aggregate[feature]["best_rank"], rank
            )
            if feature in top_features:
                aggregate[feature]["top_k_count"] += 1

    if selected_run_count == 0:
        threshold_message = f"F1 threshold {f1_threshold} 이상인 실행이 없어 "
        print(
            f"{threshold_message}최고 {score_label} 실행 결과를 사용합니다. "
            f"(best_{score_label.lower()}={best_run_score:.4f})"
        )
        if best_run_importance is None:
            return pd.DataFrame(columns=["FEATURE", "importance"])
        print("Feature Importance 분석 결과:")
        print(best_run_importance)
        return best_run_importance

    summary_rows = []
    for feature, stats in aggregate.items():
        if stats["selected_count"] == 0:
            continue
        mean_importance = stats["importance_sum"] / stats["selected_count"]
        mean_rank = stats["rank_sum"] / stats["selected_count"]
        summary_rows.append(
            {
                "FEATURE": feature,
                "importance": mean_importance,
                "top_k_count": stats["top_k_count"],
                "top_k_ratio": stats["top_k_count"] / selected_run_count,
                "selected_count": stats["selected_count"],
                "mean_rank": mean_rank,
                "best_rank": stats["best_rank"],
            }
        )

    importance_df = (
        pd.DataFrame(summary_rows)
        .sort_values(
            by=["top_k_count", "top_k_ratio", "importance", "mean_rank"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )

    print("Feature Importance 분석 결과:")
    threshold_summary = f"F1 threshold={f1_threshold}"
    print(
        f"총 실행 수={n_runs}, {threshold_summary}, "
        f"채택 실행 수={selected_run_count}, 평균 {score_label}={sum(run_scores) / len(run_scores):.4f}"
    )
    print(importance_df)
    return importance_df
