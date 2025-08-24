import sys
if sys.version_info >= (3, 13):
    import multiprocessing.resource_tracker as rt
    def fix_semlock_leak():
        orig = rt.register
        def register(name, rtype):
            if rtype == "semlock":
                return
            return orig(name, rtype)
        rt.register = register
    fix_semlock_leak()

def safe_n_jobs():
    return 1 if sys.version_info >= (3, 13) else -1

def safe_parallel_fit(model, X, y):
    if sys.version_info >= (3, 13):
        from joblib import parallel_backend
        with parallel_backend("threading"):
            return model.fit(X, y)
    else:
        return model.fit(X, y)

def safe_cross_val_predict(model, X, y, cv):
    if sys.version_info >= (3, 13):
        from joblib import parallel_backend
        with parallel_backend("threading"):
            return cross_val_predict(model, X, y, cv=cv, n_jobs=1)
    else:
        return cross_val_predict(model, X, y, cv=cv, n_jobs=-1)


import pandas as pd
import numpy as np
import os
import pulp
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import optuna

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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
    # Use all numeric columns except identifiers and text columns for modeling
    exclude_cols = ['player_id', 'player_name', 'web_name', 'first_name', 'second_name', 'team_name', 'team', 'position', 'status', 'photo', 'region', 'team_join_date', 'birth_date', 'player_name', 'team', 'team_code', 'team_short_name', 'next_opponent_name', 'next_opponent_short_name']
    # Always keep identifiers and key info
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
    # Add all numeric columns except excluded ones
    for col in df.columns:
        if col not in exclude_cols:
            # Try to convert to numeric, fillna with 0
            features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    # Add per-90 features if not present
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
    if target_col in df.columns:
        # Use all history, last5, and fixture columns as features
        for col in df.columns:
            if 'history' in col or 'last5' in col or 'fixture' in col:
                features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        y = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
        print(f"Using provided target column '{target_col}' for supervised training.")
    else:
        y = pd.to_numeric(df.get(target_col, 0), errors='coerce').fillna(0)
        print(f"No labeled '{target_col}' column found. Using '{target_col}' as proxy target for MVP training.")
    X = features.drop(['player_id','player_name','team','position','orig_index'], axis=1, errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED)

    # Hyperparameter tuning for XGBoost
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'random_state': RANDOM_SEED,
            'verbosity': 0,
            'n_jobs': safe_n_jobs()
        }
        model = XGBRegressor(**params)
        safe_parallel_fit(model, X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse
    import optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    xgb_params = study.best_params
    xgb_params['random_state'] = RANDOM_SEED
    xgb_params['verbosity'] = 0
    xgb_params['n_jobs'] = safe_n_jobs()
    # Estimate uncertainty using cross-validation std
    xgb = XGBRegressor(**xgb_params)
    safe_parallel_fit(xgb, X_train, y_train)
    xgb_preds = xgb.predict(X_scaled)
    # Use cross_val_predict to get CV predictions for std estimation
    cv_preds = safe_cross_val_predict(xgb, X_scaled, y, cv=5)
    # For each player, estimate std as abs diff between model pred and CV pred
    uncertainty = np.abs(xgb_preds - cv_preds)

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=safe_n_jobs())
    safe_parallel_fit(rf, X_train, y_train)
    rf_preds = rf.predict(X_scaled)

    # Ridge Regression
    from sklearn.linear_model import Ridge
    ridge = Ridge(random_state=RANDOM_SEED)
    safe_parallel_fit(ridge, X_train, y_train)
    ridge_preds = ridge.predict(X_scaled)

    # Stacking ensemble
    stack_X = np.vstack([xgb_preds, rf_preds, ridge_preds]).T
    meta = Ridge(random_state=RANDOM_SEED)
    safe_parallel_fit(meta, stack_X, y)
    stacked_preds = meta.predict(stack_X)

    rmse = np.sqrt(mean_squared_error(y, stacked_preds))
    print(f"Stacked Model RMSE: {rmse:.4f}")
    pred_df = pd.DataFrame({
        'player_id': features['player_id'],
        'predicted_points': stacked_preds,
        'predicted_points_std': uncertainty
    })
    return (xgb, rf, ridge, meta), stacked_preds, pred_df


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
    # Ensure captain and vice-captain have chance_of_playing_this_round > 75
    xi = ordered.copy()
    xi = xi[xi['chance_of_playing_this_round'] > 75]
    if len(xi) < 2:
        print("ERROR: Not enough starters with >75% chance of playing for captain/vice.")
        captain = ordered.loc[0]
        vice = ordered.loc[1]
    else:
        captain = xi.iloc[0]
        vice = xi.iloc[1]
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
    df_original['predicted_points_std'] = pred_df['predicted_points_std']
    pred_df.to_csv('data/predicted.csv', index=False)
    print("Saved predicted points to data/predicted.csv")
    squad = optimize_squad(df_original, features, pred_df['predicted_points'], budget=BUDGET, max_per_team=MAX_PER_TEAM)

    selection = pick_starting_xi_and_captain(squad)
    if selection is None:
        print("\nNo valid starting XI could be formed. See above for diagnostics.")
        return

    print('\n=== Best Starting XI & Captain ===')
    out_cols = ['player_name','team_name','position','now_cost','predicted_points','predicted_points_std']
    print(f"Formation: {selection['formation']}")
    pos_order = ['GKP', 'DEF', 'MID', 'FWD']
    xi_sorted = selection['starting_xi'].copy()
    xi_sorted['pos_order'] = xi_sorted['position'].apply(lambda x: pos_order.index(str(x).upper()) if str(x).upper() in pos_order else 99)
    xi_sorted = xi_sorted.sort_values(['pos_order', 'predicted_points'], ascending=[True, False]).drop(columns=['pos_order'])
    print(xi_sorted[out_cols].to_string(index=False))
    print(f"Captain: {selection['captain']['player_name']} ({selection['captain']['predicted_points']:.2f} ± {selection['captain']['predicted_points_std']:.2f})")
    print(f"Vice-Captain: {selection['vice_captain']['player_name']} ({selection['vice_captain']['predicted_points']:.2f} ± {selection['vice_captain']['predicted_points_std']:.2f})")

    print('\n=== Bench ===')
    bench_sorted = selection['bench'].copy()
    bench_sorted['pos_order'] = bench_sorted['position'].apply(lambda x: pos_order.index(str(x).upper()) if str(x).upper() in pos_order else 99)
    bench_sorted = bench_sorted.sort_values(['pos_order', 'predicted_points'], ascending=[True, False]).drop(columns=['pos_order'])
    print(bench_sorted[out_cols].to_string(index=False))

    # Output projected team points for the gameweek
    projected_points = xi_sorted['predicted_points'].sum()
    print(f"\nProjected team points for the gameweek: {projected_points:.2f}")


if __name__ == '__main__':
    main()
