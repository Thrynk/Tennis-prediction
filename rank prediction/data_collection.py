# -*- coding: utf-8 -*-

import pandas as pd
from random import randint
import matplotlib.pyplot as plt

df = pd.read_csv('input/2014-2017.csv', delimiter=';')

def resample(df):
    # creating arrays to put in DataFrame at the end
    match_id = []
    date = []
    player1_id = []
    player2_id = []
    player1_rank = []
    player2_rank = []
    player1_rank_points = []
    player2_rank_points = []
    player1_elo_rating = []
    player2_elo_rating = []
    won = []
    
    for i in range(0, df.shape[0]):

        # create random number to choose win or loose (we have a high number of matches so the sampling should be 50/50)
        rand = randint(0, 1)

        match_id.append(df.iloc[i]['match_id'])
        date.append(df.iloc[i]['date'])

        if rand == 0:
            # ids
            player1_id.append(df.iloc[i]['loser_id'])
            player2_id.append(df.iloc[i]['winner_id'])
            # ranks
            player1_rank.append(df.iloc[i]['loser_rank'])
            player2_rank.append(df.iloc[i]['winner_rank'])
            # rank points
            player1_rank_points.append(df.iloc[i]['loser_rank_points'])
            player2_rank_points.append(df.iloc[i]['winner_rank_points'])
            # elo ratings
            player1_elo_rating.append(df.iloc[i]['loser_elo_rating'])
            player2_elo_rating.append(df.iloc[i]['winner_elo_rating'])
            # won
            won.append(0)
        else:
            # ids
            player1_id.append(df.iloc[i]['winner_id'])
            player2_id.append(df.iloc[i]['loser_id'])
            # ranks
            player1_rank.append(df.iloc[i]['winner_rank'])
            player2_rank.append(df.iloc[i]['loser_rank'])
            # rank points
            player1_rank_points.append(df.iloc[i]['winner_rank_points'])
            player2_rank_points.append(df.iloc[i]['loser_rank_points'])
            # elo ratings
            player1_elo_rating.append(df.iloc[i]['winner_elo_rating'])
            player2_elo_rating.append(df.iloc[i]['loser_elo_rating'])
            # won
            won.append(1)

    return pd.DataFrame({
        'match_id': match_id,
        'date': date,
        'player1_id': player1_id,
        'player2_id': player2_id,
        'player1_rank': player1_rank,
        'player2_rank': player2_rank,
        'player1_rank_points': player1_rank_points,
        'player2_rank_points': player2_rank_points,
        'player1_elo_rating': player1_elo_rating,
        'player2_elo_rating': player2_elo_rating,
        'won': won
    })
    
oversampled_df = resample(df)

plt.hist(oversampled_df['won'])
plt.show()

oversampled_df['won'].value_counts()

oversampled_df.to_csv('input/balanced-2014-2017.csv', index=False)

data_2018 = pd.read_csv('input/2018.csv', delimiter=';')

resampled_2018 = resample(data_2018)

plt.hist(resampled_2018['won'])
plt.show()

resampled_2018.to_csv('input/balanced-2018.csv', index=False)