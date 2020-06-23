import pandas as pd
import os

import math

test_df = pd.read_csv(os.path.abspath('../input/2018.csv'), delimiter=';')

test_df_1 = test_df.drop_duplicates(subset='match_id', keep='first')

test_df_0 = test_df.drop_duplicates(subset='match_id', keep='last')

test_df_1.shape[0]

size = math.ceil(test_df_1.shape[0]/2) - 1

test_df = pd.concat([test_df_1.head(n=size), test_df_0.tail(n=size+1)])

print(test_df_1['p_matches'].value_counts())

print(test_df_0['p_matches'].value_counts())

print(test_df['p_matches'].value_counts())

test_df = test_df.dropna(subset=['p_1st_in', 'player_rank', 'opponent_rank', 'surface'])
test_df = test_df.drop(columns=['minutes'])

test_df.isnull().sum()

test_df.to_csv(os.path.abspath('../input/balanced-cleaned-2018.csv'), index=False)
