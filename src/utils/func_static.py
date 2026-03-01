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
    
    return final_flagged

def outlier_remover(df :pd.DataFrame, target_col : str, check_cols: list) -> pd.DataFrame:
    """IQR 기반 이상치 탐지 함수"""
    
    iqr_bounds = {}
    ng_df =  df[df[target_col] == 1].reset_index(drop=True)  # NG 데이터만 추출
    
    # IQR 계산 및 이상치 경계 설정
    #TODO: NG가 많이 없으면 전체 데이터로 IQR 산출 (혀재는 NG 데이터로만 IQR 산출)
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
    outlier_mask = pd.DataFrame(False, index=df.index, columns=check_cols)

    for col in check_cols:
        lower = iqr_bounds[col]["lower"]
        upper = iqr_bounds[col]["upper"]

        outlier_mask[col] = (df[col] < lower) | (df[col] > upper)

    outlier_rows = df[outlier_mask.any(axis=1)].reset_index(drop=True)
    
    return outlier_rows


def date_trend(df: pd.DataFrame, check_cols: list) -> dict:
    """날짜별 트렌드 분석 함수"""
    
    date_col = "DATE"
    window_short = 3
    window_long = 7
    
    # 이탈 강도 기준 (%)
    min_deviation = -2  # -2% 이하일 때만 유효

    result_list = []

    for value_col in check_cols:

        # 날짜별 평균 생성
        daily_df = (
            df.groupby(date_col)[value_col]
            .mean()
            .reset_index(name="daily_mean")
        )

        daily_df = daily_df.sort_values(date_col)

        daily_df["ma_short"] = daily_df["daily_mean"].rolling(window_short).mean()
        daily_df["ma_long"] = daily_df["daily_mean"].rolling(window_long).mean()

        daily_df["break_ma_short"] = daily_df["daily_mean"] < daily_df["ma_short"]
        daily_df["break_ma_long"] = daily_df["daily_mean"] < daily_df["ma_long"]

        # 이탈 강도 계산
        daily_df["deviation_pct"] = (
            (daily_df["daily_mean"] - daily_df["ma_long"]) / daily_df["ma_long"] * 100
        )

        # short & long 동시 이탈 + 강도 필터
        filtered = daily_df[
            (daily_df["break_ma_short"]) &
            (daily_df["break_ma_long"]) &
            (daily_df["deviation_pct"] <= min_deviation) &
            (daily_df["ma_short"].notna()) &
            (daily_df["ma_long"].notna())
        ].copy() 
        
        filtered["FEATURE"] = value_col  # 어떤 변수인지 표시
        result_list.append(filtered.reset_index(drop=True))
        
    result = pd.concat(result_list, ignore_index=True)
    
    cols = ["DATE", "FEATURE"] + [col for col in result.columns if col not in ["DATE", "FEATURE"]]
    result = result[cols]
        
    return result

def corr_with_defect(df: pd.DataFrame, target_col: str, check_cols: list) -> pd.DataFrame:
    """
    각 변수(col)와 target_col 간 피어슨/스피어만 상관계수 계산
    DATE 컬럼은 datetime → int64 timestamp로 변환 후 상관계수 계산
    """

    date_col = "DATE"
    top_n = 10
    
    # df[date_col] = pd.to_datetime(df[date_col])

    results = []

    # 날짜별 상관계수 계산
    for date, subdf in df.groupby(date_col):

        for col in check_cols:

            # 수치형만 처리
            if not np.issubdtype(subdf[col].dtype, np.number):
                continue

            # 데이터 부족 시 상관계수 불가
            valid = subdf[[col, target_col]].dropna()
            
            # 데이터 부족 or 값이 constant한 경우 → corr 계산 불가
            if len(valid) < 2 or valid[col].nunique() < 2 or valid[target_col].nunique() < 2:
                pearson_corr = np.nan
                spearman_corr = np.nan
            else:
                pearson_corr = valid[col].corr(valid[target_col], method="pearson")
                spearman_corr = valid[col].corr(valid[target_col], method="spearman")


            results.append({
                "DATE" : date,
                "FEATURE": col,
                "pearson": pearson_corr,
                "spearman": spearman_corr
            })

    # DataFrame 변환
    corr_df = pd.DataFrame(results)

    # 변수별 평균 상관계수 계산
    corr_df = corr_df.groupby(["DATE","FEATURE"]).agg({
        "pearson": "mean",
        "spearman": "mean"
    }).reset_index()

    # 절댓값 기준 상위 n개 변수 선택
    corr_df["abs_mean_corr"] = corr_df["pearson"].abs()  # 기준: Pearson
    corr_df = corr_df.sort_values("abs_mean_corr", ascending=False).head(top_n).reset_index(drop=True)

    return corr_df