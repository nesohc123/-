import numpy as np
import pandas as pd
import pickle

filename = r"C:\Users\dell\Desktop\各指数1990-2020 _dqw.xlsx"
df = pd.ExcelFile(filename)
sheet_names = df.sheet_names
df_concat = pd.DataFrame(index=pd.read_excel(filename,sheet_names[1],index_col=0).index,
                         columns = sheet_names)

for sheet_name in sheet_names:
    for time in df_concat.index:
        try:
            df_concat.loc[time,sheet_name] = df.parse(sheet_name, index_col=0).loc[time,r"收盘"]
        except:
            df_concat.loc[time,sheet_name] = np.nan

df_concat = df_concat.sort_index()
df_concat['return_SP500'] = df_concat.iloc[:, 1].diff() / df_concat.iloc[:, 1].shift()
df_concat['return_N225'] = df_concat.iloc[:, 2].diff() / df_concat.iloc[:, 2].shift()
df_concat['return_FTSE'] = df_concat.iloc[:, 3].diff() / df_concat.iloc[:, 3].shift()
df_concat['return_DAX'] = df_concat.iloc[:, 4].diff() / df_concat.iloc[:, 4].shift()
df_concat['return_HSI'] = df_concat.iloc[:, 5].diff() / df_concat.iloc[:, 5].shift()

with open('prepared_data.pkl','wb') as fl:
    pickle.dump(df_concat, fl)
