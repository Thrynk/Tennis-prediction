import pandas as pd
import os

df = pd.read_csv(os.path.abspath('../input/balanced-featured-2018.csv'))

df = df.dropna(subset=['p_fs', 'p_serve_adv'])

df['direct'].fillna(value=0, inplace=True)

print(df.isnull().sum())

df.to_csv(os.path.abspath('../input/balanced-cleaned-featured-2018.csv'), index=False)
