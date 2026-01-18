import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

def static_1(df :pd.DataFrame,check_cols: list) -> pd.DataFrame:
    """용해탱크 데이터에 대한 통계적 이상치 탐지 함수"""
        
    # ---------------------------
    # 통계 함수 정의
    # ---------------------------

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

    # ---------------------------
    # 통계 지 계산
    # ---------------------------
    results = []
    
    #TODO: NG가 많이 없는 것은 OK끼리도 비교 추가?
    ng_df = df[df["TAG"] == "NG"]
    unique_dates = ng_df["date"].unique()

    for target_date in unique_dates:

        g1 = ng_df[ng_df["date"] == target_date]
        g2 = ng_df[ng_df["date"] != target_date]

        # 비교 불가 케이스 제거
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

    # ---------------------------
    # 필터링 및 결과 해석
    # ---------------------------

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
            wilcox_sig   =("wilcox_sig", "any"),
            n_compare    =("column", "size"),
        )
        .reset_index()
    )

    final_flagged = summary_df[
        summary_df["cohens_d_sig"] &
        (summary_df["ttest_sig"] | summary_df["wilcox_sig"])
    ].reset_index(drop=True)
    
    return final_flagged