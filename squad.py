
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
try:
    import pulp
except Exception as e:
    raise ImportError("PuLP is required for the optimization. Install with: pip install pulp")

CSV_PATH = 'data/players.csv'
BUDGET = 1000  # FPL uses 1000 tenths = 100.0m
MAX_PER_TEAM = 3

VALID_POSITIONS = ['GKP', 'DEF', 'MID', 'FWD']
FORMATIONS = [
    {'name':'3-4-3','DEF':3,'MID':4,'FWD':3},
    {'name':'3-5-2','DEF':3,'MID':5,'FWD':2},
    {'name':'4-4-2','DEF':4,'MID':4,'FWD':2},
    {'name':'4-3-3','DEF':4,'MID':3,'FWD':3},
    {'name':'5-3-2','DEF':5,'MID':3,'FWD':2},
    {'name':'4-5-1','DEF':4,'MID':5,'FWD':1},
]

def load_data(path=CSV_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Provide your player CSV in that path.")
    df = pd.read_csv(path)
    return df


def featurize(df):
    df = df.copy()
    features = pd.DataFrame()
    for col in ['ep_next','ep_this','form','points_per_game','now_cost','total_points','minutes','selected_by_percent','value_form','value_season','ict_index']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    features['player_id'] = df.get('player_id', df.get('id'))
    features['player_name'] = df.get('player_name', df.get('web_name'))
    features['team'] = df.get('team_name', df.get('team'))
    def normalize_position(pos):
        if not isinstance(pos, str):
            return None
        pos_clean = pos.strip().upper().replace('GOALKEEPER', 'GKP').replace('GK', 'GKP').replace('GOL', 'GKP')
        if pos_clean.startswith('GKP'):
            return 'GKP'
        elif pos_clean.startswith('DEF'):
            return 'DEF'
        elif pos_clean.startswith('MID'):
            return 'MID'
        elif pos_clean.startswith('FWD') or pos_clean.startswith('FOR') or pos_clean.startswith('STR'):
            return 'FWD'
        return None
    features['position'] = df.get('position').apply(normalize_position)
    predictive_features = [
        'ep_next', 'form', 'points_per_game', 'minutes', 'starts', 'chance_of_playing_next_round', 'status',
        'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
        'expected_goals_per_90', 'expected_assists_per_90', 'expected_goal_involvements_per_90', 'expected_goals_conceded_per_90',
        'total_points', 'bps', 'bonus', 'value_form', 'value_season',
        'clean_sheets', 'clean_sheets_per_90', 'goals_scored', 'assists', 'saves', 'saves_per_90',
        'selected_by_percent', 'transfers_in_event', 'transfers_out_event', 'now_cost'
    ]
    for col in predictive_features:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    per90_map = {
        'goals_scored': 'goals_scored_per_90',
        'assists': 'assists_per_90',
        'clean_sheets': 'clean_sheets_per_90',
        'saves': 'saves_per_90',
    }
    for raw, per90 in per90_map.items():
        if raw in features.columns and 'minutes' in features.columns:
            if per90 not in features.columns:
                features[per90] = features.apply(lambda r: (r[raw] / r['minutes'] * 90) if r['minutes'] > 0 else 0, axis=1)
    if 'minutes' in features.columns and 'total_points' in features.columns:
        features['minutes_per_point'] = features.apply(lambda r: r['minutes'] / r['total_points'] if r['total_points']>0 else 9999, axis=1)
    if 'form' in features.columns and 'points_per_game' in features.columns:
        features['is_inform'] = (features['form'] > features['points_per_game']).astype(int)
    for p in VALID_POSITIONS:
        features[f'pos_{p}'] = (features['position']==p).astype(int)
    features['orig_index'] = df.index
    return df, features


def train_model(features, df, target_col='ep_next'):
    if 'ep_next' in df.columns:
        y = pd.to_numeric(df['ep_next'], errors='coerce').fillna(0)
        print("Using provided target column 'ep_next' for supervised training.")
    else:
        y = pd.to_numeric(df.get('ep_next', 0), errors='coerce').fillna(0)
        print("No labeled 'ep_next' column found. Using 'ep_next' as proxy target for MVP training.")
    X = features.drop(['player_id','player_name','team','position','orig_index'], axis=1, errors='ignore')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    if XGB_AVAILABLE:
        model = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"Validation RMSE: {rmse:.4f}")
    all_preds = model.predict(X)
    pred_df = pd.DataFrame({
        'player_id': features['player_id'],
        'predicted_points': all_preds
    })
    return model, all_preds, pred_df


def optimize_squad(df, features, preds, budget=BUDGET, max_per_team=MAX_PER_TEAM):
    N = len(df)
    indices = list(range(N))
    prob = pulp.LpProblem('FPL_squad_selection', pulp.LpMaximize)
    pick = pulp.LpVariable.dicts('pick', indices, lowBound=0, upBound=1, cat='Integer')
    prob += pulp.lpSum([preds[i] * pick[i] for i in indices])
    prob += pulp.lpSum([pick[i] for i in indices]) == 15
    pos_idx = {p: [i for i in indices if str(df.iloc[i].get('position')).upper()==p] for p in VALID_POSITIONS}
    prob += pulp.lpSum([pick[i] for i in pos_idx['GKP']]) == 2
    prob += pulp.lpSum([pick[i] for i in pos_idx['DEF']]) == 5
    prob += pulp.lpSum([pick[i] for i in pos_idx['MID']]) == 5
    prob += pulp.lpSum([pick[i] for i in pos_idx['FWD']]) == 3
    prob += pulp.lpSum([df.iloc[i].get('now_cost',0) * pick[i] for i in indices]) <= budget
    teams = df['team_name'].fillna(df.get('team','')).unique()
    for t in teams:
        team_indices = [i for i in indices if df.iloc[i].get('team_name', df.iloc[i].get('team'))==t]
        if team_indices:
            prob += pulp.lpSum([pick[i] for i in team_indices]) <= max_per_team
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)
    selected_idx = [i for i in indices if pulp.value(pick[i])>0.5]
    squad = df.iloc[selected_idx].copy()
    squad['predicted_points'] = [preds[i] for i in selected_idx]
    squad = squad.reset_index(drop=True)
    out_cols = ['player_name','team_name','position','now_cost','predicted_points']
    print('\n=== Optimized 15-man squad ===')
    pos_order = ['GKP', 'DEF', 'MID', 'FWD']
    squad_sorted = squad.copy()
    squad_sorted['pos_order'] = squad_sorted['position'].apply(lambda x: pos_order.index(str(x).upper()) if str(x).upper() in pos_order else 99)
    squad_sorted = squad_sorted.sort_values(['pos_order', 'predicted_points'], ascending=[True, False]).drop(columns=['pos_order'])
    print(squad_sorted[out_cols].to_string(index=False))
    return squad


def pick_starting_xi_and_captain(squad):
    best_lineup = None
    best_score = -1
    best_formation = None
    squad = squad.copy().reset_index(drop=True)
    squad['position'] = squad['position'].astype(str).str.upper()
    idx_by_pos = {p: squad[squad['position']==p].index.tolist() for p in VALID_POSITIONS}
    for f in FORMATIONS:
        ok = True
        for p in ['DEF','MID','FWD']:
            if len(idx_by_pos.get(p, [])) < f[p]:
                ok = False
        if len(idx_by_pos.get('GKP', [])) < 1:
            ok = False
        if not ok:
            continue
        lineup_idx = []
        gkp_idx = squad[squad['position']=='GKP'].nlargest(1, 'predicted_points').index.tolist()
        lineup_idx.extend(gkp_idx)

        for p in ['DEF','MID','FWD']:
            candidates = squad[squad['position']==p].nlargest(f[p], 'predicted_points').index.tolist()
            lineup_idx.extend(candidates)

        lineup = squad.loc[lineup_idx].copy()
        total = lineup['predicted_points'].sum()
        if total > best_score:
            best_score = total
            best_lineup = lineup
            best_formation = f['name']

    if best_lineup is None:
        print("\nERROR: Could not find a valid starting XI with given squad.")
        return None
    position_order = ['GKP', 'DEF', 'MID', 'FWD']
    best_lineup_ordered = best_lineup.copy()
    best_lineup_ordered['position'] = best_lineup_ordered['position'].astype(str).str.upper()
    best_lineup_ordered['pos_order'] = best_lineup_ordered['position'].apply(lambda x: position_order.index(x) if x in position_order else 99)
    best_lineup_ordered = best_lineup_ordered.sort_values(['pos_order', 'predicted_points'], ascending=[True, False]).drop(columns=['pos_order']).reset_index(drop=True)

    ordered = best_lineup_ordered.reset_index(drop=True)
    captain = ordered.loc[0]
    vice = ordered.loc[1]
    bench = squad[~squad.index.isin(best_lineup.index)].sort_values('predicted_points', ascending=False)

    return {
        'formation': best_formation,
        'starting_xi': best_lineup_ordered,
        'captain': captain,
        'vice_captain': vice,
        'bench': bench
    }


def main():
    df = load_data(CSV_PATH)
    df_original, features = featurize(df)
    model, preds, pred_df = train_model(features, df_original, target_col='ep_next')

    df_original['predicted_points'] = pred_df['predicted_points']
    pred_df.to_csv('data/predicted.csv', index=False)
    print("Saved predicted points to data/predicted.csv")
    squad = optimize_squad(df_original, features, pred_df['predicted_points'], budget=BUDGET, max_per_team=MAX_PER_TEAM)

    selection = pick_starting_xi_and_captain(squad)
    if selection is None:
        print("\nNo valid starting XI could be formed. See above for diagnostics.")
        return

    print('\n=== Best Starting XI & Captain ===')
    out_cols = ['player_name','team_name','position','now_cost','predicted_points']
    print(f"Formation: {selection['formation']}")
    pos_order = ['GKP', 'DEF', 'MID', 'FWD']
    xi_sorted = selection['starting_xi'].copy()
    xi_sorted['pos_order'] = xi_sorted['position'].apply(lambda x: pos_order.index(str(x).upper()) if str(x).upper() in pos_order else 99)
    xi_sorted = xi_sorted.sort_values(['pos_order', 'predicted_points'], ascending=[True, False]).drop(columns=['pos_order'])
    print(xi_sorted[out_cols].to_string(index=False))
    print(f"Captain: {selection['captain']['player_name']} ({selection['captain']['predicted_points']:.2f})")
    print(f"Vice-Captain: {selection['vice_captain']['player_name']} ({selection['vice_captain']['predicted_points']:.2f})")

    print('\n=== Bench ===')
    bench_sorted = selection['bench'].copy()
    bench_sorted['pos_order'] = bench_sorted['position'].apply(lambda x: pos_order.index(str(x).upper()) if str(x).upper() in pos_order else 99)
    bench_sorted = bench_sorted.sort_values(['pos_order', 'predicted_points'], ascending=[True, False]).drop(columns=['pos_order'])
    print(bench_sorted[out_cols].to_string(index=False))


if __name__ == '__main__':
    main()
