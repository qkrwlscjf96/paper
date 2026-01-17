
#%%
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations

#base_path = Path(__file__).parent
base_path = Path('C:/Users/wlscj/coding/paper/src').parent
data_path = base_path / '1.data'
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
# 3. Cohen's d 함수
# ---------------------------
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
# 4. Cohen's d 계산
# ---------------------------
results = []

for (d1, tag1), (d2, tag2) in combinations(groups.keys(), 2):

    # 같은 날짜 내 OK vs NG 비교만 하고 싶으면 ↓
    if d1 != d2:
        continue

    # OK vs NG만 비교 (같은 TAG끼리는 제외)
    if tag1 == tag2:
        continue

    g1, g2 = groups[(d1, tag1)], groups[(d2, tag2)]

    for col in check_cols:
        d = cohens_d(g1[col], g2[col])

        if pd.notna(d):
            results.append({
                "date": d1,
                "tag_1": tag1,
                "tag_2": tag2,
                "column": col,
                "cohens_d": d
            })

cohen_df = pd.DataFrame(results)

# ---------------------------
# 5. |Cohen's d| > 0.05 추출
# ---------------------------
threshold = 0.05

small_effect_df = cohen_df[
    cohen_df["cohens_d"].abs() > threshold
].reset_index(drop=True)


# ---------------------------
# 6. 결과 확인
# ---------------------------
print("=== date 내 OK vs NG |Cohen's d| > 0.05 ===")
print(small_effect_df["column"].value_counts())


# %%
