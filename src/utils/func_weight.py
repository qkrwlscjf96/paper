import pandas as pd


def get_weighted_df(
    df,
    check_cols,
    index_mul,
    date_mul,
    static_1_result,
    static_2_result,
    static_3_result,
    feature_importance_result=None,
):
    """가중치 적용된 DataFrame 생성 함수"""
    
    # 가중치 (통계 분석 기반)
    weight_base_df = df.copy()

    ## index별 가중치
    static_2_result["flag_static2"] = 1

    weight_base_df = weight_base_df.merge(
        static_2_result,
        on=weight_base_df.columns.tolist(),
        how="left"
    )

    weight_base_df.loc[weight_base_df["flag_static2"] == 1, check_cols] *= index_mul
    weight_base_df = weight_base_df.drop(columns=["flag_static2"])

    ## 날짜별 가중치
    weight_base_df["INDEX"] = weight_base_df.index
    temp_weight_df = weight_base_df.melt(id_vars=["INDEX", "DATE"], var_name="FEATURE", value_name="VALUE")

    static_list = [
        ("flag_static1", static_1_result),
        ("flag_static3", static_3_result),
    ]

    for flag_name, static_df in static_list:
        temp_weight_df = temp_weight_df.merge(
            static_df[["DATE", "FEATURE"]].assign(**{flag_name: 1}),
            on=["DATE", "FEATURE"],
            how="left"
        )

    flag_cols = [name for name, _ in static_list]

    multiplier = (
        temp_weight_df[flag_cols]
        .fillna(0)
        .replace({0: 1, 1: date_mul})
        .prod(axis=1)
    )

    temp_weight_df["VALUE"] *= multiplier

    weight_df = (
        temp_weight_df[["INDEX", "DATE", "FEATURE", "VALUE"]]
        .pivot(index=["INDEX", "DATE"], columns="FEATURE", values="VALUE")
        .reset_index()
    )

    weight_df = weight_df.drop(columns=["INDEX"])
    
    # 가중치 (Feature Importance 기반)
    if feature_importance_result is None or feature_importance_result.empty:
        print("Feature importance 결과가 없어 통계 기반 가중치만 적용합니다.")
        return weight_df

    feature_importance_result["WEIGHT"] = 1 + (feature_importance_result["importance"] / feature_importance_result["importance"].sum())

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
    
    print("가중치 적용된 df 생성 완료")
    return weight_df
