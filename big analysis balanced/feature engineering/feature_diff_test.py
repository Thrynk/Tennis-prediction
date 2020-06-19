import pandas as pd
import os

test_df = pd.read_csv(os.path.abspath('../input/cleaned-featured-2018.csv'))

test_df['fs_diff'] = test_df['p_fs'] - test_df['o_fs']
test_df['w1sp_diff'] = test_df['p_w1sp'] - test_df['o_w1sp']
test_df['w2sp_diff'] = test_df['p_w2sp'] - test_df['o_w2sp']
test_df['wrp_diff'] = test_df['p_wrp'] - test_df['o_wrp']
test_df['tpw_diff'] = test_df['p_tpw'] - test_df['o_tpw']
test_df['tmw_diff'] = test_df['p_tmw'] - test_df['o_tmw']
test_df['aces_per_game_diff'] = test_df['p_aces_per_game'] - test_df['o_aces_per_game']
test_df['df_per_game_diff'] = test_df['p_df_per_game'] - test_df['o_df_per_game']
test_df['bp_won_diff'] = test_df['p_bp_won'] - test_df['o_bp_won']
test_df['wsp_diff'] = test_df['p_wsp'] - test_df['o_wsp']
test_df['completeness_diff'] = test_df['p_completeness'] - test_df['o_completeness']
test_df['serve_adv_diff'] = test_df['p_serve_adv'] - test_df['o_serve_adv']
test_df['elo_rating_diff'] = test_df['player_elo_rating'] - test_df['opponent_elo_rating']


test_df = test_df.drop(['p_fs', 'o_fs', 'p_w1sp','o_w1sp','p_w2sp','o_w2sp','p_wrp','o_wrp','p_tpw','o_tpw','p_tmw','o_tmw','p_aces_per_game','o_aces_per_game'
               ,'p_df_per_game','o_df_per_game','p_bp_won','o_bp_won','p_wsp','o_wsp','p_completeness','o_completeness','p_serve_adv','o_serve_adv', 'player_elo_rating', 'opponent_elo_rating'], axis = 1)

test_df.to_csv(os.path.abspath('../input/diff-cleaned-featured-2018.csv'), index=False)