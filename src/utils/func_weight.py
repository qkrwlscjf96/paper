import pandas as pd


def get_weighted_df(
    df,
    check_cols,
    index_mul,
    date_mul,
    static_idx_result,
    static_date_result,
    feature_importance_result=None,
    return_diagnostics: bool = False,
):
    """가중치 적용된 DataFrame 생성 함수"""

    diagnostics = {}
    idx_weight_cols = list(check_cols)
    print("Index 가중치는 feature importance와 무관하게 전체 check_cols에 적용합니다.")

    static_idx_result = (
        static_idx_result.copy()
        if static_idx_result is not None
        else pd.DataFrame(columns=["INDEX"])
    )
    static_date_result = (
        static_date_result.copy()
        if static_date_result is not None
        else pd.DataFrame(columns=["DATE", "FEATURE"])
    )

    # 가중치 (통계 분석 기반)
    weight_base_df = df.copy()
    weight_base_df["INDEX"] = weight_base_df.index

    ## index별 가중치
    if not static_idx_result.empty:
        static_idx_flags = (
            static_idx_result[["INDEX"]]
            .drop_duplicates()
            .assign(flag_static_idx=1)
        )
        weight_base_df = weight_base_df.merge(
            static_idx_flags,
            on="INDEX",
            how="left"
        )
    else:
        weight_base_df["flag_static_idx"] = pd.NA

    if idx_weight_cols:
        weight_base_df.loc[weight_base_df["flag_static_idx"] == 1, idx_weight_cols] *= index_mul
    weight_base_df = weight_base_df.drop(columns=["flag_static_idx"])
    diagnostics["index_weighted_rows"] = static_idx_result.copy()
    diagnostics["index_weighted_features"] = pd.DataFrame({"FEATURE": idx_weight_cols})

    ## 날짜별 가중치
    temp_weight_df = weight_base_df.melt(id_vars=["INDEX", "DATE"], var_name="FEATURE", value_name="VALUE")

    static_list = []
    if not static_date_result.empty and {"DATE", "FEATURE"}.issubset(static_date_result.columns):
        static_list.append(("flag_static_date", static_date_result))

    for flag_name, static_df in static_list:
        temp_weight_df = temp_weight_df.merge(
            static_df[["DATE", "FEATURE"]].assign(**{flag_name: 1}),
            on=["DATE", "FEATURE"],
            how="left"
        )

    flag_cols = [name for name, _ in static_list]

    if flag_cols:
        multiplier = (
            temp_weight_df[flag_cols]
            .fillna(0)
            .replace({0: 1, 1: date_mul})
            .prod(axis=1)
        )
    else:
        multiplier = 1

    temp_weight_df["VALUE"] *= multiplier
    if flag_cols:
        diagnostics["date_feature_weighted_rows"] = temp_weight_df[
            temp_weight_df[flag_cols].fillna(0).any(axis=1)
        ].copy()
    else:
        diagnostics["date_feature_weighted_rows"] = pd.DataFrame(columns=temp_weight_df.columns)

    weight_df = (
        temp_weight_df[["INDEX", "DATE", "FEATURE", "VALUE"]]
        .pivot(index=["INDEX", "DATE"], columns="FEATURE", values="VALUE")
        .reset_index()
    )

    weight_df = weight_df.drop(columns=["INDEX"])
    
    # 가중치 (Feature Importance 기반)
    if feature_importance_result is None or feature_importance_result.empty:
        print("Feature importance 결과가 없어 통계 기반 가중치만 적용합니다.")
        diagnostics["feature_importance_weights"] = pd.DataFrame(columns=["FEATURE", "WEIGHT"])
        diagnostics["weighting_summary"] = pd.DataFrame(
            [
                {
                    "index_weighted_row_count": len(static_idx_result),
                    "index_weighted_feature_count": len(idx_weight_cols),
                    "date_feature_weighted_count": len(diagnostics["date_feature_weighted_rows"]),
                    "feature_importance_feature_count": 0,
                }
            ]
        )
        if return_diagnostics:
            return weight_df, diagnostics
        return weight_df

    feature_importance_result = feature_importance_result.copy()
    feature_importance_result["WEIGHT"] = 1 + (
        feature_importance_result["importance"] / feature_importance_result["importance"].sum()
    )

    weight_map = dict(
        zip(
            feature_importance_result["FEATURE"],
            feature_importance_result["WEIGHT"]
        )
    )

    common_cols = df.columns.intersection(weight_map.keys())

    weight_df[common_cols] = (
        weight_df[common_cols]
        .mul(pd.Series(weight_map), axis=1)
    )

    diagnostics["feature_importance_weights"] = feature_importance_result.copy()
    diagnostics["weighting_summary"] = pd.DataFrame(
        [
            {
                "index_weighted_row_count": len(static_idx_result),
                "index_weighted_feature_count": len(idx_weight_cols),
                "date_feature_weighted_count": len(diagnostics["date_feature_weighted_rows"]),
                "feature_importance_feature_count": len(feature_importance_result),
            }
        ]
    )
    
    print("가중치 적용된 df 생성 완료")
    if return_diagnostics:
        return weight_df, diagnostics
    return weight_df
