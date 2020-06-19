import pandas as pd
import os

df = pd.read_csv(os.path.abspath('../input/diff-cleaned-featured-2018.csv'))

df['fs_diff'] = df['fs_diff'] / df['fs_diff'].std()
df['w1sp_diff'] = df['w1sp_diff'] / df['w1sp_diff'].std()
df['w2sp_diff'] = df['w2sp_diff'] / df['w2sp_diff'].std()
df['wrp_diff'] = df['wrp_diff'] / df['wrp_diff'].std()
df['tpw_diff'] = df['tpw_diff'] / df['tpw_diff'].std()
df['tmw_diff'] = df['tmw_diff'] / df['tmw_diff'].std()
df['aces_per_game_diff'] = df['aces_per_game_diff'] / df['aces_per_game_diff'].std()
df['df_per_game_diff'] = df['df_per_game_diff'] / df['df_per_game_diff'].std()
df['bp_won_diff'] = df['bp_won_diff'] / df['bp_won_diff'].std()
df['wsp_diff'] = df['wsp_diff'] / df['wsp_diff'].std()
df['completeness_diff'] = df['completeness_diff'] / df['completeness_diff'].std()
df['serve_adv_diff'] = df['serve_adv_diff'] / df['serve_adv_diff'].std()
df['elo_rating_diff'] = df['elo_rating_diff'] / df['elo_rating_diff'].std()

df.to_csv(os.path.abspath('../input/standardized-diff-cleaned-featured-2018.csv'), index=False)