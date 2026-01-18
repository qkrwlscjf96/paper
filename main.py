
#%%
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind, mannwhitneyu

#base_path = Path(__file__).parent
base_path = Path("/Users/danielpark/Documents/서강대/pgm/논문/src").parent
data_path = base_path / 'data'

#xlsx_files = list(data_path.glob('*.xlsx'))
csv_files = list(data_path.glob('*.csv'))

#용접기
#df1 = pd.read_excel(xlsx_files[0], sheet_name='Raw')

#용해탱크
df2 = pd.read_csv(csv_files[0])
# %%
#용해탱크

# ---------------------------
# 1. 날짜 전처리 (yyyy-mm-dd)
# ---------------------------
df2 = df2.copy()

df2["date"] = pd.to_datetime(df2["STD_DT"]).dt.date
#df2 = df2.drop(columns="STD_DT")

# ---------------------------
# 2. 그룹 생성 (date + TAG)
# ---------------------------
# groups[(date, tag)] 형태
groups = dict(tuple(df2.groupby(["date", "TAG"])))

# 숫자 컬럼만 선택
check_cols = ["MELT_TEMP","MOTORSPEED","MELT_WEIGHT"]

# ---------------------------
# 3. 통계 함수 정의
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
# 4. 통계 지 계산
# ---------------------------
results = []

ng_df = df2[df2["TAG"] == "NG"]
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
# 5. 필터링 및 결과 해석
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
