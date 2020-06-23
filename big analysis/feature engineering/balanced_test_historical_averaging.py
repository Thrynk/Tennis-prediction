import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

test_df = pd.read_csv(os.path.abspath('../input/balanced-cleaned-2018.csv'))

test_df['datetime'] = test_df['date'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))

statistics_df_2007_2017 = pd.read_csv(os.path.abspath('../input/cleaned-2007-2017.csv'))

frames = [statistics_df_2007_2017, test_df]
statistics_df = pd.concat(frames)

# player 1 columns
p_fs_column = []
p_w1sp_column = []
p_w2sp_column = []
p_wrp_column = []
p_tpw_column = []
p_tmw_column = []
p_aces_per_game_column = []
p_df_per_game_column = []
p_bp_won_column = []

# Combining statistics
p_wsp_column = []
p_completeness_column = []

p_h2h_column = []

p_number_of_matches_used_column = []

#player 2 columns
o_fs_column = []
o_w1sp_column = []
o_w2sp_column = []
o_wrp_column = []
o_tpw_column = []
o_tmw_column = []
o_aces_per_game_column = []
o_df_per_game_column = []
o_bp_won_column = []

# Combining statistics
o_wsp_column = []
o_completeness_column = []

o_h2h_column = []

o_number_of_matches_used_column = []

# Combining statistics from the two players
p_serve_adv_column = []
o_serve_adv_column = []
direct_column = []

# matches to include (for development, no need for final dataset generation)
# number_of_matches_to_include = 2000

# time discounting coefficient
time_discounting = 0.8

pbar = tqdm(total=test_df.shape[0])

for index, row in test_df.iterrows():
    # filter dataset to keep only player matches prior to current match
    filtered_dataset = statistics_df[
        (statistics_df['datetime'] < datetime.strptime(row['date'], '%d/%m/%Y')) 
        & (statistics_df['player_id'] == row['player_id'])
    ]

    # initialize temp variables to calculate features
    p_1st_in = p_sv_pt = p_1st_won = p_2nd_won = o_sv_pt = o_1st_won = o_2nd_won = o_sv_pt = p_matches \
    = o_matches = p_ace = p_df = p_sv_gms = p_bp_sv = o_bp_sv = p_bp_fc = o_bp_fc = p_number_of_matches_used = 0

    # initialize variable features
    p_fs = p_w1sp = p_w2sp = p_wrp = p_tpw = p_tmw = p_aces_per_game = p_df_per_game = p_bp_won = p_wsp = p_completeness = p_h2h = np.nan

    # calculate statistics averaging with time discounting and surface weighting for player 1
    if(filtered_dataset.shape[0] != 0):
        for f_index, f_row in filtered_dataset.iterrows():
            time_diff = row['datetime'].year - f_row['datetime'].year
            surface_weight = surface_correlations[row['surface']].loc[f_row['surface']]
            p_1st_in += f_row['p_1st_in'] * (time_discounting ** time_diff) * surface_weight
            p_sv_pt += f_row['p_sv_pt'] * (time_discounting ** time_diff) * surface_weight
            p_1st_won += f_row['p_1st_won'] * (time_discounting ** time_diff) * surface_weight
            p_2nd_won += f_row['p_2nd_won'] * (time_discounting ** time_diff) * surface_weight
            o_sv_pt += f_row['o_sv_pt'] * (time_discounting ** time_diff) * surface_weight
            o_1st_won += f_row['o_1st_won'] * (time_discounting ** time_diff) * surface_weight
            o_2nd_won += f_row['o_2nd_won'] * (time_discounting ** time_diff) * surface_weight
            o_sv_pt += f_row['o_sv_pt'] * (time_discounting ** time_diff) * surface_weight
            p_matches += f_row['p_matches'] * (time_discounting ** time_diff) * surface_weight
            o_matches += f_row['o_matches'] * (time_discounting ** time_diff) * surface_weight
            p_ace += f_row['p_ace'] * (time_discounting ** time_diff) * surface_weight
            p_df += f_row['p_df'] * (time_discounting ** time_diff) * surface_weight
            p_sv_gms += f_row['p_sv_gms'] * (time_discounting ** time_diff) * surface_weight
            p_bp_sv += f_row['p_bp_sv'] * (time_discounting ** time_diff) * surface_weight
            o_bp_sv += f_row['o_bp_sv'] * (time_discounting ** time_diff) * surface_weight
            p_bp_fc += f_row['p_bp_fc'] * (time_discounting ** time_diff) * surface_weight
            o_bp_fc += f_row['o_bp_fc'] * (time_discounting ** time_diff) * surface_weight

        ## Compute features 

        # first serve sucess percentage
        p_fs = p_1st_in / p_sv_pt

        # winning on first serve percentage
        p_w1sp = p_1st_won / p_1st_in

        # winning on second serve percentage
        p_w2sp = p_2nd_won / (p_sv_pt - p_1st_in)
        
        # winning on return percentage
        p_wrp = (o_sv_pt - o_1st_won - o_2nd_won) / o_sv_pt
        
        # percentage of all points won
        p_tpw = (p_1st_won + p_2nd_won + (o_sv_pt - o_1st_won - o_2nd_won) ) / (p_sv_pt + o_sv_pt)
        
        # percentage of all matches won
        p_tmw = p_matches / (p_matches + o_matches)
        
        # Average number of aces per game
        p_aces_per_game = p_ace / p_sv_gms
        
        # Average number of double faults per game
        p_df_per_game = p_df / p_sv_gms
        
        # Percentage of breakpoints won
        p_bp_won = (p_bp_sv + o_bp_sv) / (p_bp_fc + o_bp_fc)
        
        # overall winning percentage
        p_wsp = p_w1sp * p_fs + p_w2sp * (1-p_fs)
        
        # completeness
        p_completeness = p_wsp * p_wrp
        
        
        
        # number of matches used to compute statistics
        p_number_of_matches_used = filtered_dataset.shape[0]

    p_fs_column.append(p_fs)
    p_w1sp_column.append(p_w1sp)
    p_w2sp_column.append(p_w2sp)
    p_wrp_column.append(p_wrp)
    p_tpw_column.append(p_tpw)
    p_tmw_column.append(p_tmw)
    p_aces_per_game_column.append(p_aces_per_game)
    p_df_per_game_column.append(p_df_per_game)
    p_bp_won_column.append(p_bp_won)
    
    # Combining statistics
    p_wsp_column.append(p_wsp)
    p_completeness_column.append(p_completeness)
    
    ## head to head feature
    # filter dataset to keep only player matches prior to current match
    filtered_dataset_h2h = statistics_df[
        (statistics_df['datetime'] < datetime.strptime(row['date'], '%d/%m/%Y')) 
        & (statistics_df['player_id'] == row['player_id'])
        & (statistics_df['opponent_id'] == row['opponent_id'])
    ]
    if(filtered_dataset_h2h.shape[0] != 0):
        p_h2h = filtered_dataset_h2h['p_matches'].sum() / filtered_dataset_h2h['p_matches'].count()
    p_h2h_column.append(p_h2h)
    

    # number of matches computed to have stats
    p_number_of_matches_used_column.append(p_number_of_matches_used)
    
    #------------------------------------------------------------------------------------------------------------------
    
    # filter dataset to keep only opponent matches prior to current match
    filtered_dataset = statistics_df[
        (statistics_df['datetime'] < datetime.strptime(row['date'], '%d/%m/%Y')) 
        & (statistics_df['player_id'] == row['opponent_id'])
    ]

    # initialize temp variables to calculate features
    p_1st_in = p_sv_pt = p_1st_won = p_2nd_won = o_sv_pt = o_1st_won = o_2nd_won = o_sv_pt = p_matches \
    = o_matches = p_ace = p_df = p_sv_gms = p_bp_sv = o_bp_sv = p_bp_fc = o_bp_fc = o_number_of_matches_used = 0

    # initialize variable features
    o_fs = o_w1sp = o_w2sp = o_wrp = o_tpw = o_tmw = o_aces_per_game = o_df_per_game = o_bp_won = o_wsp = o_completeness = o_h2h =  np.nan

    # calculate statistics averaging with time discounting and surface weighting for player 2
    if(filtered_dataset.shape[0] != 0):
        for f_index, f_row in filtered_dataset.iterrows():
            time_diff = row['datetime'].year - f_row['datetime'].year
            surface_weight = surface_correlations[row['surface']].loc[f_row['surface']]
            p_1st_in += f_row['p_1st_in'] * (time_discounting ** time_diff) * surface_weight
            p_sv_pt += f_row['p_sv_pt'] * (time_discounting ** time_diff) * surface_weight
            p_1st_won += f_row['p_1st_won'] * (time_discounting ** time_diff) * surface_weight
            p_2nd_won += f_row['p_2nd_won'] * (time_discounting ** time_diff) * surface_weight
            o_sv_pt += f_row['o_sv_pt'] * (time_discounting ** time_diff) * surface_weight
            o_1st_won += f_row['o_1st_won'] * (time_discounting ** time_diff) * surface_weight
            o_2nd_won += f_row['o_2nd_won'] * (time_discounting ** time_diff) * surface_weight
            o_sv_pt += f_row['o_sv_pt'] * (time_discounting ** time_diff) * surface_weight
            p_matches += f_row['p_matches'] * (time_discounting ** time_diff) * surface_weight
            o_matches += f_row['o_matches'] * (time_discounting ** time_diff) * surface_weight
            p_ace += f_row['p_ace'] * (time_discounting ** time_diff) * surface_weight
            p_df += f_row['p_df'] * (time_discounting ** time_diff) * surface_weight
            p_sv_gms += f_row['p_sv_gms'] * (time_discounting ** time_diff) * surface_weight
            p_bp_sv += f_row['p_bp_sv'] * (time_discounting ** time_diff) * surface_weight
            o_bp_sv += f_row['o_bp_sv'] * (time_discounting ** time_diff) * surface_weight
            p_bp_fc += f_row['p_bp_fc'] * (time_discounting ** time_diff) * surface_weight
            o_bp_fc += f_row['o_bp_fc'] * (time_discounting ** time_diff) * surface_weight
            
        ## Compute features 

        # first serve sucess percentage
        o_fs = p_1st_in / p_sv_pt

        # winning on first serve percentage
        o_w1sp = p_1st_won / p_1st_in

        # winning on second serve percentage
        o_w2sp = p_2nd_won / (p_sv_pt - p_1st_in)
        
        # winning on return percentage
        o_wrp = (o_sv_pt - o_1st_won - o_2nd_won) / o_sv_pt
        
        # percentage of all points won
        o_tpw = (p_1st_won + p_2nd_won + (o_sv_pt - o_1st_won - o_2nd_won) ) / (p_sv_pt + o_sv_pt)
        
        # percentage of all matches won
        o_tmw = p_matches / (p_matches + o_matches)
        
        # Average number of aces per game
        o_aces_per_game = p_ace / p_sv_gms
        
        # Average number of double faults per game
        o_df_per_game = p_df / p_sv_gms
        
        # Percentage of breakpoints won
        o_bp_won = (p_bp_sv + o_bp_sv) / (p_bp_fc + o_bp_fc)
        
        ## Combining features to obtain more stats
        
        # overall winning percentage
        o_wsp = o_w1sp * o_fs + o_w2sp * (1-o_fs)
        
        # completeness
        o_completeness = o_wsp * o_wrp
        
        
        
        # number of matches used to compute statistics
        o_number_of_matches_used = filtered_dataset.shape[0]

    o_fs_column.append(o_fs)
    o_w1sp_column.append(o_w1sp)
    o_w2sp_column.append(o_w2sp)
    o_wrp_column.append(o_wrp)
    o_tpw_column.append(o_tpw)
    o_tmw_column.append(o_tmw)
    o_aces_per_game_column.append(o_aces_per_game)
    o_df_per_game_column.append(o_df_per_game)
    o_bp_won_column.append(o_bp_won)
    
    # Combining statistics
    o_wsp_column.append(o_wsp)
    o_completeness_column.append(o_completeness)
    
    ## head to head feature
    # filter dataset to keep only player matches prior to current match
    filtered_dataset_h2h = statistics_df[
        (statistics_df['datetime'] < datetime.strptime(row['date'], '%d/%m/%Y')) 
        & (statistics_df['player_id'] == row['opponent_id'])
        & (statistics_df['opponent_id'] == row['player_id'])
    ]
    if(filtered_dataset_h2h.shape[0] != 0):
        o_h2h = filtered_dataset_h2h['p_matches'].sum() / filtered_dataset_h2h['p_matches'].count()
    o_h2h_column.append(o_h2h)
    

    # number of matches computed to have stats
    o_number_of_matches_used_column.append(o_number_of_matches_used)
    
    # Combining statistics from the two players
    p_serve_adv_column = (np.array(p_wsp_column) - np.array(o_wrp_column)).tolist()
    o_serve_adv_column = (np.array(o_wsp_column) - np.array(p_wrp_column)).tolist()
    direct_column = (np.array(p_h2h_column) - np.array(o_h2h_column)).tolist()
    
    # progress
    pbar.update(1)
    
pbar.close()

result_test_df = pd.DataFrame({'p_fs': p_fs_column, 'p_w1sp': p_w1sp_column, 'p_w2sp': p_w2sp_column, 'p_wrp': p_wrp_column, 
                               'p_tpw': p_tpw_column, 'p_tmw': p_tmw_column, 'p_aces_per_game': p_aces_per_game_column,
                               'p_df_per_game': p_df_per_game_column, 'p_bp_won': p_bp_won_column,
                               'p_wsp': p_wsp_column, 'p_completeness': p_completeness_column, 'p_serve_adv': p_serve_adv_column,
                               'p_number_of_matches_used': p_number_of_matches_used_column,
                               'o_fs': o_fs_column, 'o_w1sp': o_w1sp_column, 'o_w2sp': o_w2sp_column, 'o_wrp': o_wrp_column, 
                               'o_tpw': o_tpw_column, 'o_tmw': o_tmw_column, 'o_aces_per_game': o_aces_per_game_column,
                               'o_df_per_game': o_df_per_game_column, 'o_bp_won': o_bp_won_column,
                               'o_wsp': o_wsp_column, 'o_completeness': o_completeness_column, 'o_serve_adv': o_serve_adv_column,
                               'direct': direct_column,
                               'o_number_of_matches_used': o_number_of_matches_used_column
                              })
result_test_df['player_id'] = test_df['player_id'].tolist()
result_test_df['opponent_id'] = test_df['opponent_id'].tolist()
result_test_df['tournament_name'] = test_df['tournament_name'].tolist()
result_test_df['date'] = test_df['date'].tolist()
result_test_df['p_matches'] = test_df['p_matches'].tolist()

result_test_df.to_csv(os.path.abspath('../input/balanced-featured-2018.csv'), index=False)