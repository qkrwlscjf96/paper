
#%%
import pandas as pd
from pathlib import Path
import numpy as np
from utils.func_static import *
from utils.func_eda import *

#base_path = Path(__file__).parent
base_path = Path("/Users/danielpark/Documents/서강대/pgm/논문/src").parent
data_path = base_path / 'data'

#xlsx_files = list(data_path.glob('*.xlsx'))
csv_files = list(data_path.glob('*.csv'))

# 기본꼴 : 날짜 DATE / 검사 TAG 

#용접기
#df1 = pd.read_excel(xlsx_files[0], sheet_name='Raw')

#용해탱크
df2 = pd.read_csv(csv_files[0])
df2["DATE"] = pd.to_datetime(df2["STD_DT"]).dt.date
df2 = df2.drop(columns=["STD_DT"])
ng_df2 = df2[df2["TAG"] == "NG"]
check_cols = ["MELT_TEMP","MOTORSPEED","MELT_WEIGHT"]


#%%
# Statistical Analysis

#static_1_result = static_1(ng_df2,check_cols)
#static_2_result = static_2(ng_df2,check_cols,df2)


# %%

# EDA
plot_boxplots_by_date(df2, check_cols)


# %%
