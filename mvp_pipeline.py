"""
Fantasy Premier League Predictor MVP Pipeline
Step 1: Data Reshaping Agent
- Reshape FPL player dataset into time-series format: (player_id, gameweek)
- For each row, include all stats up to GW_t as features
- Target: actual points in GW_t+1
- Compute lag features (lag-1, lag-3, lag-5) and rolling averages (last 3, last 5)
"""
import pandas as pd
import numpy as np

def load_player_data(csv_path):
    """Load raw player data from CSV."""
    return pd.read_csv(csv_path)

def reshape_to_timeseries(df, player_id_col='player_id', gw_col='gameweek', target_col='total_points'):
    """
    Reshape player stats into time-series format with lag and rolling features.
    Each row: (player_id, gameweek), features up to GW_t, target = points in GW_t+1.
    """
    # Use last5_total_points_* columns as pseudo-time-series
    points_cols = [f'last5_total_points_{i}' for i in range(1, 6)]
    ts_rows = []
    for _, row in df.iterrows():
        player_id = row['player_id']
        player_name = row.get('player_name', '')
        team_name = row.get('team_name', '')
        position = row.get('position', '')
        for gw in range(5):
            features = {f'lag_{lag}_points': row[points_cols[gw-lag]] if gw-lag >= 0 else np.nan for lag in [1, 3, 5]}
            rolling_3 = np.mean([row[points_cols[max(0, gw-i)]] for i in range(3)])
            rolling_5 = np.mean([row[points_cols[max(0, gw-i)]] for i in range(5)])
            form_acceleration = rolling_3 - rolling_5
            target_next_gw = row[points_cols[gw+1]] if gw < 4 else np.nan
            # Per-90 stats
            per90_stats = {}
            for stat in ['goals_scored', 'assists', 'saves', 'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded']:
                stat_val = row.get(stat, np.nan)
                minutes = row.get('minutes', np.nan)
                per90_stats[f'{stat}_per90'] = stat_val / minutes * 90 if minutes and minutes > 0 else np.nan
            # Expected metrics, attacking, defensive, bonus, discipline
            expected_metrics = {k: row.get(k, np.nan) for k in [
                'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
                'expected_goals_per_90', 'expected_assists_per_90', 'expected_goal_involvements_per_90', 'expected_goals_conceded_per_90']}
            attacking = {k: row.get(k, np.nan) for k in ['influence', 'creativity', 'threat', 'ict_index']}
            defensive = {k: row.get(k, np.nan) for k in ['saves', 'clean_sheets', 'goals_conceded', 'recoveries', 'tackles', 'defensive_contribution']}
            bonus = {k: row.get(k, np.nan) for k in ['bps', 'bonus']}
            discipline = {k: row.get(k, np.nan) for k in ['yellow_cards', 'red_cards']}
            # Market/price
            market = {k: row.get(k, np.nan) for k in ['now_cost', 'cost_change_event', 'selected_by_percent', 'transfers_in_event', 'transfers_out_event', 'value_form', 'value_season']}
            # Availability
            availability = {k: row.get(k, np.nan) for k in ['chance_of_playing_next_round', 'chance_of_playing_this_round', 'minutes', 'starts']}
            # Fixtures
            fixtures = {k: row.get(k, np.nan) for k in [
                'next_fixture_id', 'next_fixture_is_home', 'next_fixture_difficulty', 'next_opponent_name', 'next_opponent_strength',
                'next_opponent_strength_defence_home', 'next_opponent_strength_defence_away', 'next_opponent_strength_attack_home', 'next_opponent_strength_attack_away']}
            # Categorical encoding (simple: keep as string, ML can encode)
            # Scaling (defer to model pipeline)
            ts_rows.append({
                'player_id': player_id,
                'player_name': player_name,
                'team_name': team_name,
                'position': position,
                'pseudo_gameweek': gw+1,
                'points': row[points_cols[gw]],
                'lag_1_points': features['lag_1_points'],
                'lag_3_points': features['lag_3_points'],
                'lag_5_points': features['lag_5_points'],
                'rolling_3_points': rolling_3,
                'rolling_5_points': rolling_5,
                'form_acceleration': form_acceleration,
                'target_next_gw': target_next_gw,
                **per90_stats,
                **expected_metrics,
                **attacking,
                **defensive,
                **bonus,
                **discipline,
                **market,
                **availability,
                **fixtures
            })
    ts_df = pd.DataFrame(ts_rows)
    return ts_df

def main():
    # Example usage
    df = load_player_data('data/players.csv')
    ts_df = reshape_to_timeseries(df)
    # --- Feature Engineering ---
    # Opponent defensive/attacking strength
    opponent_strength_cols = ['next_opponent_strength', 'next_opponent_strength_defence_home', 'next_opponent_strength_defence_away', 'next_opponent_strength_attack_home', 'next_opponent_strength_attack_away']
    for col in opponent_strength_cols:
        ts_df[col] = df[col].values.repeat(5)
    # Home/Away flag
    ts_df['is_home'] = df['next_fixture_is_home'].values.repeat(5)
    # Availability features
    ts_df['chance_of_playing_next_round'] = df['chance_of_playing_next_round'].values.repeat(5)
    # Minutes trend (last 5)
    for i in range(1, 6):
        ts_df[f'last5_minutes_{i}'] = df[f'last5_minutes_{i}'].values.repeat(5)
    ts_df['minutes_trend'] = ts_df[[f'last5_minutes_{i}' for i in range(1, 6)]].mean(axis=1)
    # Team form (team-level rolling stats, using last5_team_goals_scored and last5_team_goals_conceded if available)
    if 'last5_team_goals_scored_1' in df.columns:
        for i in range(1, 6):
            ts_df[f'last5_team_goals_scored_{i}'] = df[f'last5_team_goals_scored_{i}'].values.repeat(5)
            ts_df[f'last5_team_goals_conceded_{i}'] = df[f'last5_team_goals_conceded_{i}'].values.repeat(5)
        ts_df['team_goals_scored_rolling'] = ts_df[[f'last5_team_goals_scored_{i}' for i in range(1, 6)]].mean(axis=1)
        ts_df['team_goals_conceded_rolling'] = ts_df[[f'last5_team_goals_conceded_{i}' for i in range(1, 6)]].mean(axis=1)
    # Save expanded dataset
    ts_df.to_csv('data/players_timeseries.csv', index=False)
    print('Reshaped and feature-engineered time-series data saved to data/players_timeseries.csv')

if __name__ == '__main__':
    main()
