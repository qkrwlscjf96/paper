import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def date_group_test(ng_df :pd.DataFrame,check_cols: list) -> pd.DataFrame:
    """날짜별 집단차이 통계적 검정 함수"""

    # 통계적 검정 함수 정의
    def t_test(x, y):
        x = x.dropna()
        y = y.dropna()

        if len(x) < 2 or len(y) < 2:
            return np.nan

        stat, p = ttest_ind(x, y, equal_var=False)  # Welch
        return p

    def wilcoxon_rank_sum(x, y):
        x = x.dropna()
        y = y.dropna()

        if len(x) < 2 or len(y) < 2:
            return np.nan

        # 두 집단 값이 완전히 동일하면 에러 발생 가능
        try:
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            return p
        except ValueError:
            return np.nan


    def cohens_d(x, y):
        x = x.dropna()
        y = y.dropna()

        if len(x) < 2 or len(y) < 2:
            return np.nan

        nx, ny = len(x), len(y)
        vx, vy = x.var(ddof=1), y.var(ddof=1)

        pooled_std = np.sqrt(
            ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
        )

        if pooled_std == 0:
            return np.nan

        return (x.mean() - y.mean()) / pooled_std

    
    # 비교 결과 저장 리스트
    results = []
    
    #TODO: NG가 많이 없는 것은 OK끼리도 비교 추가? (현재는 NG끼리만 비교)
    unique_dates = ng_df["DATE"].unique()

    for target_date in unique_dates:

        g1 = ng_df[ng_df["DATE"] == target_date]
        g2 = ng_df[ng_df["DATE"] != target_date]

        ## 비교 불가 케이스 제거
        if len(g1) < 2 or len(g2) < 2:
            continue

        for col in check_cols:
            cd = cohens_d(g1[col], g2[col])
            t_p = t_test(g1[col], g2[col])
            w_p = wilcoxon_rank_sum(g1[col], g2[col])

            if any(pd.notna(v) for v in [cd, t_p, w_p]):
                results.append({
                    "target_date": target_date,
                    "column": col,
                    "cohens_d": cd,
                    "t_pvalue": t_p,
                    "wilcoxon_pvalue": w_p,
                    "compare": "NG_target_vs_NG_others"
                })

    stat_df = pd.DataFrame(results)

    # 결과해석
    summary_df = (
        stat_df
        .assign(
            cohens_d_sig = lambda x: x["cohens_d"].abs() > 0.8,
            ttest_sig    = lambda x: x["t_pvalue"] < 0.05,
            wilcox_sig   = lambda x: x["wilcoxon_pvalue"] < 0.05,
        )
        .groupby(["target_date", "column"])
        .agg(
            cohens_d_mean=("cohens_d", "mean"),
            cohens_d_max =("cohens_d", lambda x: x.abs().max()),
            cohens_d_sig =("cohens_d_sig", "any"),
            ttest_sig    =("ttest_sig", "any"),
            wilcox_sig   =("wilcox_sig", "any")
        )
        .reset_index()
    )

    final_flagged = summary_df[
        summary_df["cohens_d_sig"] &
        (summary_df["ttest_sig"] | summary_df["wilcox_sig"])
    ].reset_index(drop=True)
    
    #RESULT 해석을 위해 컬럼명 변경
    final_flagged = final_flagged.rename(columns={"target_date": "DATE"})
    final_flagged = final_flagged.rename(columns={"column": "FEATURE"})
    
    print(f"날짜별 집단차이 통계적 검정 결과: 총 {len(final_flagged)}개 날짜-변수 조합에서 유의미한 차이 감지")
    return final_flagged

def outlier_remover(
    full_df: pd.DataFrame,
    target_col: str,
    check_cols: list,
    directional_corr_threshold: float = 0.0,
) -> pd.DataFrame:
    """IQR 기반 이상치 탐지 함수
    
    변수와 불량(target) 간 방향성을 확인하여
    - |corr|가 threshold 이상으로 강하면
      - 망대(불량과 반비례, 음의 상관): lower만 이상치로 판단
      - 망소(불량과 비례, 양의 상관): upper만 이상치로 판단
    - |corr|가 threshold 미만이면 기존처럼 양측 이상치로 판단
    """
    
    iqr_bounds = {}
    ng_df = full_df[full_df[target_col] == 1].reset_index(drop=True)  # NG 데이터만 추출
    
    # IQR 계산 및 이상치 경계 설정
    #TODO: NG가 많이 없으면 전체 데이터로 IQR 산출 (현재는 NG 데이터로만 IQR 산출)
    for col in check_cols:
        q1 = ng_df[col].quantile(0.25)
        q3 = ng_df[col].quantile(0.75)
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
            outlier_mask[col] = (full_df[col] < lower) | (full_df[col] > upper)
        elif corr < 0:
            # 망대: 값이 작을수록 불량 가능성이 높으므로 lower만 확인
            outlier_mask[col] = full_df[col] < lower
        else:
            # 망소: 값이 클수록 불량 가능성이 높으므로 upper만 확인
            outlier_mask[col] = full_df[col] > upper

    outlier_rows = full_df[outlier_mask.any(axis=1)]
    outlier_rows = outlier_rows.drop_duplicates().reset_index(drop=True)
    print(
        "IQR 기반 이상치 탐지 결과: "
        f"corr_threshold={directional_corr_threshold}, "
        f"총 {len(outlier_rows)}개 행이 이상치로 감지"
    )
    
    return outlier_rows


def date_trend(
    full_df: pd.DataFrame,
    check_cols: list,
    target_col: str,
    min_deviation: float = -2,
    window_short: int = 3,
    window_long: int = 7,
) -> dict:
    """날짜별 트렌드 분석 함수

    변수별 평균 트렌드 이상감지와 불량률 트렌드 이상감지를 함께 수행하고,
    두 이상이 같은 날짜에 동시에 발생한 경우만 반환한다.
    """
    
    date_col = "DATE"
    def build_trend_flags(daily_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        daily_df = daily_df.sort_values(date_col).copy()

        daily_df["ma_short"] = daily_df[value_name].rolling(window_short).mean()
        daily_df["ma_long"] = daily_df[value_name].rolling(window_long).mean()

        daily_df["break_ma_short"] = daily_df[value_name] < daily_df["ma_short"]
        daily_df["break_ma_long"] = daily_df[value_name] < daily_df["ma_long"]
        daily_df["deviation_pct"] = (
            (daily_df[value_name] - daily_df["ma_long"]) / daily_df["ma_long"] * 100
        )

        return daily_df[
            (daily_df["break_ma_short"]) &
            (daily_df["break_ma_long"]) &
            (daily_df["deviation_pct"] <= min_deviation) &
            (daily_df["ma_short"].notna()) &
            (daily_df["ma_long"].notna())
        ].copy()

    defect_daily_df = (
        full_df.groupby(date_col)[target_col]
        .mean()
        .reset_index(name="defect_rate")
    )
    defect_filtered = build_trend_flags(defect_daily_df, "defect_rate")
    defect_filtered = defect_filtered.rename(
        columns={
            "ma_short": "defect_ma_short",
            "ma_long": "defect_ma_long",
            "break_ma_short": "defect_break_ma_short",
            "break_ma_long": "defect_break_ma_long",
            "deviation_pct": "defect_deviation_pct",
        }
    )
    defect_filtered = defect_filtered[
        [
            "DATE",
            "defect_rate",
            "defect_ma_short",
            "defect_ma_long",
            "defect_break_ma_short",
            "defect_break_ma_long",
            "defect_deviation_pct",
        ]
    ]

    result_list = []

    for value_col in check_cols:
        daily_df = (
            full_df.groupby(date_col)[value_col]
            .mean()
            .reset_index(name="daily_mean")
        )
        filtered = build_trend_flags(daily_df, "daily_mean")
        filtered = filtered.merge(defect_filtered, on="DATE", how="inner")
        
        filtered["FEATURE"] = value_col  # 어떤 변수인지 표시
        result_list.append(filtered.reset_index(drop=True))

    if result_list:
        result = pd.concat(result_list, ignore_index=True)
    else:
        result = pd.DataFrame()
    
    if not result.empty:
        cols = ["DATE", "FEATURE"] + [col for col in result.columns if col not in ["DATE", "FEATURE"]]
        result = result[cols]
    
    print(
        "날짜별 트렌드 분석 결과: "
        f"변수 트렌드와 불량 트렌드가 동시에 이상인 총 {len(result)}개 날짜-변수 조합 감지"
    )
    return result
