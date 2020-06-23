# -*- coding: utf-8 -*-

import pandas as pd
import os

train_df = pd.read_csv(os.path.abspath('../input/2007-2017.csv'), delimiter=';')

# train
train_df = train_df.dropna(subset=['match_id'])
train_df = train_df.dropna(subset=['p_1st_in', 'player_rank', 'opponent_rank', 'surface'])
train_df = train_df.drop(train_df[train_df['surface'] == 'P'].index)
train_df = train_df.drop(columns=['minutes'])

train_df.isnull().sum()

# save it to csv

train_df.to_csv(os.path.abspath('../input/cleaned-2007-2017.csv'), index=False)

# validation
validation_df = pd.read_csv(os.path.abspath('../input/2018.csv'), delimiter=';')

validation_df = validation_df.dropna(subset=['p_1st_in', 'player_rank', 'opponent_rank', 'surface'])
validation_df = validation_df.drop(columns=['minutes'])

validation_df.isnull().sum()

validation_df.to_csv(os.path.abspath('../input/cleaned-2018.csv'), index=False)

# test
test_df = pd.read_csv(os.path.abspath('../input/2019.csv'), delimiter=';')

test_df = test_df.dropna(subset=['p_1st_in', 'player_rank', 'opponent_rank', 'surface'])
test_df = test_df.drop(columns=['minutes'])

test_df.isnull().sum()

test_df.to_csv(os.path.abspath('../input/cleaned-2019.csv'), index=False)