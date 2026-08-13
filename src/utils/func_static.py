import pandas as pd
import numpy as np

def outlier_remover(
    full_df: pd.DataFrame,
    target_col: str,
    check_cols: list,
    directional_corr_threshold: float = 0.0,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """IQR 기반 이상치 탐지 함수
    
    변수와 불량(target) 간 방향성을 확인하여
    - |corr|가 threshold 이상으로 강하면
      - 망대(불량과 반비례, 음의 상관): lower만 이상치로 판단
      - 망소(불량과 비례, 양의 상관): upper만 이상치로 판단
    - |corr|가 threshold 미만이면 분석 안함
    """
    
    iqr_bounds = {}
    
    # IQR 계산 및 이상치 경계 설정
    # 전체 데이터 기준으로 사분위 범위를 계산한다.
    for col in check_cols:
        q1 = full_df[col].quantile(0.25)
        q3 = full_df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        iqr_bounds[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper
        }

    # 이상치 여부 판단
    outlier_mask = pd.DataFrame(False, index=full_df.index, columns=check_cols)

    for col in check_cols:
        lower = iqr_bounds[col]["lower"]
        upper = iqr_bounds[col]["upper"]
        valid = full_df[[col, target_col]].dropna()
        corr = np.nan

        if len(valid) >= 2 and valid[col].nunique() >= 2 and valid[target_col].nunique() >= 2:
            corr = valid[col].corr(valid[target_col], method="pearson")

        if pd.isna(corr) or abs(corr) < directional_corr_threshold:
            continue
        elif corr < 0:
            # 망대: 값이 작을수록 불량 가능성이 높으므로 lower만 확인
            outlier_mask[col] = full_df[col] < lower
        else:
            # 망소: 값이 클수록 불량 가능성이 높으므로 upper만 확인
            outlier_mask[col] = full_df[col] > upper

    outlier_rows = full_df[outlier_mask.any(axis=1)]
    outlier_rows = outlier_rows.drop_duplicates().reset_index(drop=True)
    outlier_detail_list = []

    for col in check_cols:
        col_mask = outlier_mask[col].fillna(False)
        if not col_mask.any():
            continue

        detail_df = full_df.loc[col_mask, ["DATE", col]].copy()
        detail_df = detail_df.rename(columns={col: "VALUE"})
        detail_df["FEATURE"] = col
        outlier_detail_list.append(detail_df[["DATE", "FEATURE", "VALUE"]])

    if outlier_detail_list:
        outlier_detail_df = pd.concat(outlier_detail_list, ignore_index=True)
    else:
        outlier_detail_df = pd.DataFrame(columns=["DATE", "FEATURE", "VALUE"])

    print(
        "IQR 기반 이상치 탐지 결과: "
        f"corr_threshold={directional_corr_threshold}, "
        f"총 {len(outlier_rows)}개 행이 이상치로 감지"
    )
    
    if return_details:
        return outlier_rows, outlier_detail_df
    return outlier_rows


def date_trend(
    full_df: pd.DataFrame,
    check_cols: list,
    target_col: str,
    context_days: int = 2,
    min_window_points: int = 4,
    min_anchor_rate_quantile: float = 0.8,
    min_level_score: float = 0.5,
    min_sign_agreement: float = 0.5,
) -> dict:
    """날짜별 트렌드 분석 함수

    defect_rate가 높은 날짜를 anchor로 잡고,
    해당 날짜 주변 구간에서 feature가 같은 방향의 극단값과 유사한 트렌드를 보이는 경우만 반환한다.
    """
    date_col = "DATE"

    def add_series_features(daily_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        daily_df = daily_df.sort_values(date_col).copy()
        mean_value = daily_df[value_name].mean()
        std_value = daily_df[value_name].std()
        if pd.isna(std_value) or std_value == 0:
            daily_df["level_zscore"] = 0.0
        else:
            daily_df["level_zscore"] = (daily_df[value_name] - mean_value) / std_value

        daily_df["delta"] = daily_df[value_name].diff()
        daily_df["trend_direction"] = np.where(
            daily_df["delta"] > 0,
            "up",
            np.where(daily_df["delta"] < 0, "down", None),
        )
        return daily_df

    def extract_window_pattern(daily_df: pd.DataFrame, value_name: str, center_date: pd.Timestamp) -> pd.DataFrame:
        window_start = center_date - pd.Timedelta(days=context_days)
        window_end = center_date + pd.Timedelta(days=context_days)
        window_df = daily_df[
            (daily_df[date_col] >= window_start) &
            (daily_df[date_col] <= window_end)
        ][[date_col, value_name]].copy()
        if len(window_df) < min_window_points:
            return pd.DataFrame(columns=[date_col, "delta"])

        window_df["delta"] = window_df[value_name].diff()
        return window_df.dropna(subset=["delta"])[[date_col, "delta"]].reset_index(drop=True)

    defect_daily_df = (
        full_df.groupby(date_col)[target_col]
        .mean()
        .reset_index(name="defect_rate")
    )
    defect_daily_df = add_series_features(defect_daily_df, "defect_rate")
    anchor_threshold = defect_daily_df["defect_rate"].quantile(min_anchor_rate_quantile)
    defect_anchor_df = defect_daily_df[
        (defect_daily_df["defect_rate"] >= anchor_threshold) &
        (defect_daily_df["defect_rate"] > 0)
    ].copy()
    if defect_anchor_df.empty:
        print("날짜별 high defect 분석 결과: defect_rate 높은 anchor 날짜를 찾지 못함")
        return pd.DataFrame()

    result_list = []

    feature_daily_map = {
        value_col: add_series_features(
            full_df.groupby(date_col)[value_col].mean().reset_index(name="daily_mean"),
            "daily_mean",
        )
        for value_col in check_cols
    }

    for defect_row in defect_anchor_df.itertuples(index=False):
        center_date = defect_row.DATE
        defect_pattern = extract_window_pattern(defect_daily_df, "defect_rate", center_date)
        if defect_pattern.empty:
            continue

        defect_direction = "up" if defect_pattern["delta"].sum() > 0 else "down"

        for value_col, daily_df in feature_daily_map.items():
            valid = daily_df.merge(
                defect_daily_df[[date_col, "defect_rate"]],
                on=date_col,
                how="inner",
            )
            if len(valid) < min_window_points:
                continue

            corr = valid["daily_mean"].corr(valid["defect_rate"])
            align_sign = -1 if pd.notna(corr) and corr < 0 else 1
            feature_state = "low" if align_sign < 0 else "high"

            feature_pattern = extract_window_pattern(daily_df, "daily_mean", center_date)
            if feature_pattern.empty:
                continue

            merged_pattern = defect_pattern.merge(
                feature_pattern,
                on=date_col,
                how="inner",
                suffixes=("_defect", "_feature"),
            )
            if len(merged_pattern) < min_window_points - 1:
                continue

            defect_delta = merged_pattern["delta_defect"]
            feature_delta = merged_pattern["delta_feature"] * align_sign
            if defect_delta.nunique() < 2 or feature_delta.nunique() < 2:
                pattern_corr = 0.0
            else:
                pattern_corr = defect_delta.corr(feature_delta)
                if pd.isna(pattern_corr):
                    pattern_corr = 0.0

            sign_agreement = (
                np.sign(defect_delta).eq(np.sign(feature_delta)).mean()
            )
            feature_direction = "up" if feature_delta.sum() > 0 else "down"
            center_feature = daily_df.loc[daily_df[date_col] == center_date].copy()
            if center_feature.empty:
                continue
            level_score = float(center_feature["level_zscore"].iloc[0] * align_sign)

            if (
                level_score >= min_level_score and
                sign_agreement >= min_sign_agreement and
                feature_direction == defect_direction
            ):
                result_list.append(
                    pd.DataFrame(
                        {
                            "DATE": [center_date],
                            "FEATURE": [value_col],
                            "defect_rate": [defect_row.defect_rate],
                            "defect_level_zscore": [defect_row.level_zscore],
                            "defect_trend_direction": [defect_direction],
                            "feature_state": [feature_state],
                            "feature_trend_direction": [feature_direction],
                            "feature_level_score": [level_score],
                            "pattern_corr": [pattern_corr],
                            "sign_agreement": [sign_agreement],
                            "window_points": [len(merged_pattern)],
                        }
                    )
                )

    if result_list:
        result = pd.concat(result_list, ignore_index=True)
    else:
        result = pd.DataFrame()
    
    if not result.empty:
        cols = ["DATE", "FEATURE"] + [col for col in result.columns if col not in ["DATE", "FEATURE"]]
        result = result[cols]
    
    print(
        "날짜별 high defect 분석 결과: "
        f"defect_rate가 높은 날짜에 feature 값과 트렌드가 함께 맞는 총 {len(result)}개 날짜-변수 조합 감지"
    )
    return result
