
#%%
import pandas as pd
from pathlib import Path
import numpy as np
from utils.func_static import *

#base_path = Path(__file__).parent
base_path = Path("/Users/danielpark/Documents/서강대/pgm/논문/src").parent
data_path = base_path / 'data'

#xlsx_files = list(data_path.glob('*.xlsx'))
csv_files = list(data_path.glob('*.csv'))

#용접기
#df1 = pd.read_excel(xlsx_files[0], sheet_name='Raw')

#용해탱크
df2 = pd.read_csv(csv_files[0])
df2["date"] = pd.to_datetime(df2["STD_DT"]).dt.date
check_cols = ["MELT_TEMP","MOTORSPEED","MELT_WEIGHT"]
static_1(df2,check_cols)


# %%
#용해탱크

#%%

