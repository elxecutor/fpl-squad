"""
FPL Predictor
Builds a machine learning model to predict FPL player points and optimizes squad selection.
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import optuna, pulp, os, pickle

def load_data():
    """Load JSON data into DataFrames."""
    with open('data/bootstrap.json', 'r') as f:
        bootstrap = json.load(f)
    
    players_df = pd.DataFrame(bootstrap['elements'])
    players_df['selected_by_percent'] = players_df['selected_by_percent'].str.rstrip('%').astype(float)
    players_df['form'] = pd.to_numeric(players_df['form'], errors='coerce')
    players_df['points_per_game'] = pd.to_numeric(players_df['points_per_game'], errors='coerce')
    players_df['clean_sheets_per_90'] = pd.to_numeric(players_df['clean_sheets_per_90'], errors='coerce')
    players_df['starts_per_90'] = pd.to_numeric(players_df['starts_per_90'], errors='coerce')
    teams_df = pd.DataFrame(bootstrap['teams'])
    gameweeks_df = pd.DataFrame(bootstrap['events'])
    
    with open('data/fixtures.json', 'r') as f:
        fixtures_df = pd.DataFrame(json.load(f))
    
    with open('data/player_histories.json', 'r') as f:
        player_histories = json.load(f)
    
    return players_df, teams_df, gameweeks_df, fixtures_df, player_histories

def create_player_history_df(player_histories):
    """Create a DataFrame from player histories."""
    all_histories = []
    for player_id, data in player_histories.items():
        if data['history']:
            history_df = pd.DataFrame(data['history'])
            history_df['player_id'] = int(player_id)
            all_histories.append(history_df)
    
    combined_df = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    combined_df = combined_df.sort_values(['player_id', 'round']).reset_index(drop=True)
    return combined_df

def create_target_variable(history_df):
    """Create target variable: points in next gameweek."""
    history_df = history_df.copy()
    history_df['target'] = history_df.groupby('player_id')['total_points'].shift(-1)
    # Drop rows where target is NaN (last gameweek for each player)
    history_df = history_df.dropna(subset=['target'])
    return history_df

def add_features(history_df, players_df, teams_df, fixtures_df):
    """Add feature engineering."""
    # Merge player info
    history_df = history_df.merge(players_df[['id', 'element_type', 'team', 'now_cost', 'selected_by_percent', 'form', 'points_per_game', 'clean_sheets_per_90', 'starts_per_90']], left_on='player_id', right_on='id', how='left')
    history_df.rename(columns={'element_type': 'position', 'team': 'team_id', 'now_cost': 'cost'}, inplace=True)
    
    # Add average points per player
    average_points_df = history_df.groupby('player_id')['total_points'].mean().reset_index(name='average_points')
    history_df = history_df.merge(average_points_df, on='player_id', how='left')
    
    # Merge team info for player team
    history_df = history_df.merge(teams_df[['id', 'name', 'strength']], left_on='team_id', right_on='id', how='left', suffixes=('', '_team'))
    history_df.rename(columns={'name': 'team_name', 'strength': 'team_strength'}, inplace=True)
    
    # Opponent team strength
    history_df = history_df.merge(teams_df[['id', 'strength']], left_on='opponent_team', right_on='id', how='left', suffixes=('', '_opp'))
    history_df.rename(columns={'strength': 'opponent_strength'}, inplace=True)
    
    # Home/away flag
    history_df['is_home'] = history_df['was_home'].astype(int)
    
    # Convert string columns to float
    float_cols = ['expected_goals', 'expected_assists', 'influence', 'creativity', 'threat', 'ict_index']
    for col in float_cols:
        history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
    
    # Rolling averages (last 3 and 5 games)
    for window in [3, 5]:
        history_df[f'rolling_points_{window}'] = history_df.groupby('player_id')['total_points'].rolling(window).mean().reset_index(0, drop=True)
        history_df[f'rolling_minutes_{window}'] = history_df.groupby('player_id')['minutes'].rolling(window).mean().reset_index(0, drop=True)
        history_df[f'rolling_goals_{window}'] = history_df.groupby('player_id')['goals_scored'].rolling(window).mean().reset_index(0, drop=True)
        history_df[f'rolling_assists_{window}'] = history_df.groupby('player_id')['assists'].rolling(window).mean().reset_index(0, drop=True)
        history_df[f'rolling_clean_sheets_{window}'] = history_df.groupby('player_id')['clean_sheets'].rolling(window).mean().reset_index(0, drop=True)
        history_df[f'rolling_tackles_{window}'] = history_df.groupby('player_id')['tackles'].rolling(window).mean().reset_index(0, drop=True)
        history_df[f'rolling_recoveries_{window}'] = history_df.groupby('player_id')['recoveries'].rolling(window).mean().reset_index(0, drop=True)
        history_df[f'rolling_bps_{window}'] = history_df.groupby('player_id')['bps'].rolling(window).mean().reset_index(0, drop=True)
    
    # Expected goals/assists
    history_df['expected_goals'] = history_df['expected_goals']
    history_df['expected_assists'] = history_df['expected_assists']
    
    # Other features
    history_df['minutes'] = history_df['minutes']
    history_df['bps'] = history_df['bps']
    history_df['influence'] = history_df['influence']
    history_df['creativity'] = history_df['creativity']
    history_df['threat'] = history_df['threat']
    history_df['ict_index'] = history_df['ict_index']
    
    return history_df

def objective(trial, X_train, y_train, X_test, y_test):
    """Optuna objective function with enhanced regularization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        
        # Sampling regularization (prevents overfitting)
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),  # NEW
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),    # NEW
        
        # L1/L2 regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),     # Increased range
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),   # Increased range
        
        # Tree structure regularization
        'gamma': trial.suggest_float('gamma', 0, 0.5),           # NEW: Min split loss
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # NEW: Min samples per leaf
        
        'random_state': 42
    }
    
    model = XGBRegressor(**params)
    
    # Use cross-validation for more robust evaluation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, 
                               scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()  # Convert back to positive MAE
    
    return cv_mae

def select_features_by_importance(model, feature_names, threshold=0.01):
    """Select features based on model importance scores."""
    feature_importance = model.feature_importances_
    important_features = []
    
    for i, importance in enumerate(feature_importance):
        if importance >= threshold:
            important_features.append(feature_names[i])
    
    print(f"Selected {len(important_features)}/{len(feature_names)} features above threshold {threshold}")
    return important_features

def train_model(df):
    """Train XGBoost model with Optuna tuning."""
    training_features = [
        'position', 'team_id', 'opponent_team', 'cost', 'team_strength', 'opponent_strength', 'is_home',
        'rolling_points_3', 'rolling_points_5', 'rolling_minutes_3', 'rolling_minutes_5',
        'rolling_goals_3', 'rolling_goals_5', 'rolling_assists_3', 'rolling_assists_5',
        'rolling_clean_sheets_3', 'rolling_clean_sheets_5', 'rolling_tackles_3', 'rolling_tackles_5',
        'rolling_recoveries_3', 'rolling_recoveries_5', 'rolling_bps_3', 'rolling_bps_5',
        'expected_goals', 'expected_assists', 'minutes', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'average_points', 'selected_by_percent', 'form', 'points_per_game', 'clean_sheets_per_90', 'starts_per_90'
    ]
    
    # Identify categorical features
    categorical_features = ['position', 'team_id', 'opponent_team']
    numeric_features = [f for f in training_features if f not in categorical_features]
    
    # Fill NaN values
    df[numeric_features] = df[numeric_features].fillna(0)
    df['position'] = df['position'].fillna(df['position'].mode()[0] if not df['position'].mode().empty else 1)
    df['team_id'] = df['team_id'].fillna(df['team_id'].mode()[0] if not df['team_id'].mode().empty else 1)
    df['opponent_team'] = df['opponent_team'].fillna(df['opponent_team'].mode()[0] if not df['opponent_team'].mode().empty else 1)
    
    # Drop rows with NaN target
    df = df.dropna(subset=['target'])
    
    X = df[training_features]
    y = df['target']
    
    # Split by time (last 3 gameweeks for test)
    max_round = df['round'].max()
    train_df = df[df['round'] < max_round - 3]
    test_df = df[df['round'] >= max_round - 3]
    
    X_train = train_df[training_features].copy()
    y_train = train_df['target']
    X_test = test_df[training_features].copy()
    y_test = test_df['target']
    
    # Fit OneHotEncoder on categorical features (position, team_id, opponent_team)
    position_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    team_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    opponent_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Encode position
    position_train_encoded = position_encoder.fit_transform(X_train[['position']])
    position_test_encoded = position_encoder.transform(X_test[['position']])
    position_feature_names = [f'position_{int(cat)}' for cat in position_encoder.categories_[0]]
    
    # Encode team_id
    team_train_encoded = team_encoder.fit_transform(X_train[['team_id']])
    team_test_encoded = team_encoder.transform(X_test[['team_id']])
    team_feature_names = [f'team_{int(cat)}' for cat in team_encoder.categories_[0]]
    
    # Encode opponent_team
    opponent_train_encoded = opponent_encoder.fit_transform(X_train[['opponent_team']])
    opponent_test_encoded = opponent_encoder.transform(X_test[['opponent_team']])
    opponent_feature_names = [f'opponent_{int(cat)}' for cat in opponent_encoder.categories_[0]]
    
    # Create DataFrames with encoded features
    position_train_df = pd.DataFrame(position_train_encoded, columns=position_feature_names, index=X_train.index)
    position_test_df = pd.DataFrame(position_test_encoded, columns=position_feature_names, index=X_test.index)
    
    team_train_df = pd.DataFrame(team_train_encoded, columns=team_feature_names, index=X_train.index)
    team_test_df = pd.DataFrame(team_test_encoded, columns=team_feature_names, index=X_test.index)
    
    opponent_train_df = pd.DataFrame(opponent_train_encoded, columns=opponent_feature_names, index=X_train.index)
    opponent_test_df = pd.DataFrame(opponent_test_encoded, columns=opponent_feature_names, index=X_test.index)
    
    # Remove original categorical columns and add encoded columns
    X_train_encoded = X_train.drop(['position', 'team_id', 'opponent_team'], axis=1)
    X_test_encoded = X_test.drop(['position', 'team_id', 'opponent_team'], axis=1)
    
    X_train_encoded = pd.concat([X_train_encoded, position_train_df, team_train_df, opponent_train_df], axis=1)
    X_test_encoded = pd.concat([X_test_encoded, position_test_df, team_test_df, opponent_test_df], axis=1)
    
    # Update numeric features list (all features are now numeric after one-hot encoding)
    all_numeric_features = list(X_train_encoded.columns)
    
    # Fit scaler on all numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    # Convert back to DataFrame for clarity (optional, XGBoost can handle numpy arrays)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=all_numeric_features, index=X_train_encoded.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=all_numeric_features, index=X_test_encoded.index)
    
    # Optuna study with more trials for better optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_test_scaled, y_test), n_trials=50)
    
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['early_stopping_rounds'] = 10  # Stop if no improvement for 10 rounds
    best_params['eval_metric'] = 'mae'
    
    model = XGBRegressor(**best_params)
    
    # Fit with early stopping validation
    model.fit(X_train_scaled, y_train, 
             eval_set=[(X_test_scaled, y_test)], 
             verbose=False)
    
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"Best MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    # Feature importance analysis
    important_features = select_features_by_importance(model, all_numeric_features, threshold=0.005)
    print(f"Top 10 most important features:")
    feature_importance_pairs = list(zip(all_numeric_features, model.feature_importances_))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in feature_importance_pairs[:10]:
        print(f"  {feat}: {imp:.4f}")
    
    # Save scaler and preprocessing info
    preprocessing_info = {
        'scaler': scaler,
        'position_encoder': position_encoder,
        'team_encoder': team_encoder,
        'opponent_encoder': opponent_encoder,
        'position_feature_names': position_feature_names,
        'team_feature_names': team_feature_names,
        'opponent_feature_names': opponent_feature_names,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'all_features': all_numeric_features
    }
    
    return model, training_features, preprocessing_info

def predict_upcoming(model, features, preprocessing_info, players_df, teams_df, fixtures_df, player_histories):
    """Predict points for upcoming gameweek."""
    scaler = preprocessing_info['scaler']
    position_encoder = preprocessing_info['position_encoder']
    team_encoder = preprocessing_info['team_encoder']
    opponent_encoder = preprocessing_info['opponent_encoder']
    position_feature_names = preprocessing_info['position_feature_names']
    team_feature_names = preprocessing_info['team_feature_names']
    opponent_feature_names = preprocessing_info['opponent_feature_names']
    all_features = preprocessing_info['all_features']
    
    # Get current gameweek
    current_gw = max([max([h['round'] for h in data['history']]) for data in player_histories.values() if data['history']])
    
    # For each player, get their next fixture
    predictions = []
    for player_id, data in player_histories.items():
        fixtures = data['fixtures']
        if fixtures:
            next_fixture = fixtures[0]  # Assuming sorted
            # Opponent team
            if next_fixture['is_home']:
                opponent_team = next_fixture['team_a']
            else:
                opponent_team = next_fixture['team_h']
            # Get latest history
            if data['history']:
                latest = data['history'][-1]
                # Create feature row
                row = {
                    'player_id': int(player_id),
                    'round': current_gw,
                    'opponent_team': opponent_team,
                    'is_home': next_fixture['is_home'],
                    'was_home': next_fixture['is_home'],
                    'minutes': latest.get('minutes', 0),
                    'total_points': latest.get('total_points', 0),
                    'goals_scored': latest.get('goals_scored', 0),
                    'assists': latest.get('assists', 0),
                    'expected_goals': float(latest.get('expected_goals', 0) or 0),
                    'expected_assists': float(latest.get('expected_assists', 0) or 0),
                    'bps': latest.get('bps', 0),
                    'influence': float(latest.get('influence', 0) or 0),
                    'creativity': float(latest.get('creativity', 0) or 0),
                    'threat': float(latest.get('threat', 0) or 0),
                    'ict_index': float(latest.get('ict_index', 0) or 0),
                }
                # Add rolling features (simplified, using last few)
                # For simplicity, use last 3 and 5 from history
                history = data['history']
                if len(history) >= 5:
                    row['rolling_points_3'] = np.mean([h['total_points'] for h in history[-3:]])
                    row['rolling_points_5'] = np.mean([h['total_points'] for h in history[-5:]])
                    row['rolling_minutes_3'] = np.mean([h['minutes'] for h in history[-3:]])
                    row['rolling_minutes_5'] = np.mean([h['minutes'] for h in history[-5:]])
                    row['rolling_goals_3'] = np.mean([h['goals_scored'] for h in history[-3:]])
                    row['rolling_goals_5'] = np.mean([h['goals_scored'] for h in history[-5:]])
                    row['rolling_assists_3'] = np.mean([h['assists'] for h in history[-3:]])
                    row['rolling_assists_5'] = np.mean([h['assists'] for h in history[-5:]])
                    row['rolling_clean_sheets_3'] = np.mean([h.get('clean_sheets', 0) for h in history[-3:]])
                    row['rolling_clean_sheets_5'] = np.mean([h.get('clean_sheets', 0) for h in history[-5:]])
                    row['rolling_tackles_3'] = np.mean([h.get('tackles', 0) for h in history[-3:]])
                    row['rolling_tackles_5'] = np.mean([h.get('tackles', 0) for h in history[-5:]])
                    row['rolling_recoveries_3'] = np.mean([h.get('recoveries', 0) for h in history[-3:]])
                    row['rolling_recoveries_5'] = np.mean([h.get('recoveries', 0) for h in history[-5:]])
                    row['rolling_bps_3'] = np.mean([h['bps'] for h in history[-3:]])
                    row['rolling_bps_5'] = np.mean([h['bps'] for h in history[-5:]])
                else:
                    for w in [3,5]:
                        row[f'rolling_points_{w}'] = np.mean([h['total_points'] for h in history[-w:]]) if history else 0
                        row[f'rolling_minutes_{w}'] = np.mean([h['minutes'] for h in history[-w:]]) if history else 0
                        row[f'rolling_goals_{w}'] = np.mean([h['goals_scored'] for h in history[-w:]]) if history else 0
                        row[f'rolling_assists_{w}'] = np.mean([h['assists'] for h in history[-w:]]) if history else 0
                        row[f'rolling_clean_sheets_{w}'] = np.mean([h.get('clean_sheets', 0) for h in history[-w:]]) if history else 0
                        row[f'rolling_tackles_{w}'] = np.mean([h.get('tackles', 0) for h in history[-w:]]) if history else 0
                        row[f'rolling_recoveries_{w}'] = np.mean([h.get('recoveries', 0) for h in history[-w:]]) if history else 0
                        row[f'rolling_bps_{w}'] = np.mean([h['bps'] for h in history[-w:]]) if history else 0
                
                # Merge player and team info
                player_info = players_df[players_df['id'] == int(player_id)].iloc[0]
                
                # Calculate rolling minutes for filtering
                rolling_minutes_5 = np.mean([h['minutes'] for h in history[-5:]]) if len(history) >= 5 else (np.mean([h['minutes'] for h in history]) if history else 0)
                
                # Filter out unavailable and non-playing players
                if (player_info['status'] != 'a' or  # Not available
                    player_info.get('chance_of_playing_next_round', 0) < 75 or  # Low chance of playing
                    rolling_minutes_5 < 60):  # Not getting regular minutes
                    continue  # Skip this player
                
                row['position'] = player_info['element_type']
                row['cost'] = player_info['now_cost']
                row['team_id'] = player_info['team']
                row['selected_by_percent'] = player_info['selected_by_percent']
                row['form'] = player_info['form']
                row['points_per_game'] = player_info['points_per_game']
                row['clean_sheets_per_90'] = player_info['clean_sheets_per_90']
                row['starts_per_90'] = player_info['starts_per_90']
                row['average_points'] = np.mean([h['total_points'] for h in history]) if history else 0
                team_info = teams_df[teams_df['id'] == row['team_id']].iloc[0]
                row['team_strength'] = team_info['strength']
                opp_team_info = teams_df[teams_df['id'] == row['opponent_team']].iloc[0]
                row['opponent_strength'] = opp_team_info['strength']
                
                # Create prediction dataframe with original features
                X_pred = pd.DataFrame([row])[features]
                
                # Apply OneHotEncoder to all categorical features
                position_encoded = position_encoder.transform(X_pred[['position']])
                position_encoded_df = pd.DataFrame(position_encoded, columns=position_feature_names, index=X_pred.index)
                
                team_encoded = team_encoder.transform(X_pred[['team_id']])
                team_encoded_df = pd.DataFrame(team_encoded, columns=team_feature_names, index=X_pred.index)
                
                opponent_encoded = opponent_encoder.transform(X_pred[['opponent_team']])
                opponent_encoded_df = pd.DataFrame(opponent_encoded, columns=opponent_feature_names, index=X_pred.index)
                
                # Remove original categorical columns and add encoded columns
                X_pred_encoded = X_pred.drop(['position', 'team_id', 'opponent_team'], axis=1)
                X_pred_encoded = pd.concat([X_pred_encoded, position_encoded_df, team_encoded_df, opponent_encoded_df], axis=1)
                
                # Ensure columns are in the same order as training
                X_pred_encoded = X_pred_encoded[all_features]
                
                # Scale all features using the fitted scaler
                X_pred_scaled = scaler.transform(X_pred_encoded)
                
                # Predict
                pred_points = model.predict(X_pred_scaled)[0]
                row['predicted_points'] = pred_points
                predictions.append(row)
    
    return pd.DataFrame(predictions)

def optimize_squad(predictions_df, players_df, teams_df):
    """Optimize squad using ILP."""
    # Positions: 1=GKP, 2=DEF, 3=MID, 4=FWD
    positions = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    limits = {1: 2, 2: 5, 3: 5, 4: 3}
    
    # Formation: e.g., 3-4-3
    formation = (3, 4, 3)  # DEF, MID, FWD for starting XI
    
    prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
    
    # Variables: select player or not
    player_vars = pulp.LpVariable.dicts("player", predictions_df.index, cat='Binary')
    
    # Objective: maximize predicted points
    prob += pulp.lpSum([predictions_df.loc[i, 'predicted_points'] * player_vars[i] for i in predictions_df.index])
    
    # Constraints
    # Total players: 15
    prob += pulp.lpSum([player_vars[i] for i in predictions_df.index]) == 15
    
    # Position limits
    for pos_id, limit in limits.items():
        prob += pulp.lpSum([player_vars[i] for i in predictions_df[predictions_df['position'] == pos_id].index]) == limit
    
    # Team limit: <=3 per team
    for team_id in predictions_df['team_id'].unique():
        prob += pulp.lpSum([player_vars[i] for i in predictions_df[predictions_df['team_id'] == team_id].index]) <= 3
    
    # Budget: <=1000
    prob += pulp.lpSum([predictions_df.loc[i, 'cost'] * player_vars[i] for i in predictions_df.index]) <= 1000
    
    # Solve
    prob.solve()
    
    # Get selected players
    selected = [i for i in predictions_df.index if player_vars[i].value() == 1]
    squad_df = predictions_df.loc[selected].copy()
    
    # Add player names and team names
    squad_df = squad_df.merge(players_df[['id', 'web_name']], left_on='player_id', right_on='id', how='left')
    squad_df = squad_df.merge(teams_df[['id', 'name']], left_on='team_id', right_on='id', how='left', suffixes=('', '_team'))
    squad_df.rename(columns={'web_name': 'player_name', 'name': 'team_name'}, inplace=True)
    
    # Add position name
    position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    squad_df['position_name'] = squad_df['position'].map(position_map)
    
    # Verify formation
    position_counts = squad_df['position'].value_counts().sort_index()
    print(f"Squad formation: GK={position_counts.get(1,0)}, DEF={position_counts.get(2,0)}, MID={position_counts.get(3,0)}, FWD={position_counts.get(4,0)}")
    
    return squad_df

def main():
    # Load data
    players_df, teams_df, gameweeks_df, fixtures_df, player_histories = load_data()
    
    # Create history DataFrame
    history_df = create_player_history_df(player_histories)
    
    # Create target
    history_df = create_target_variable(history_df)
    
    # Feature engineering
    history_df = add_features(history_df, players_df, teams_df, fixtures_df)
    
    # Train model
    model, features, preprocessing_info = train_model(history_df)
    
    # Predict upcoming (now includes filtering for availability and minutes)
    predictions_df = predict_upcoming(model, features, preprocessing_info, players_df, teams_df, fixtures_df, player_histories)
    print(f"Predictions shape (after filtering): {predictions_df.shape}")
    print(f"Sample predictions: {predictions_df[['player_id', 'predicted_points', 'cost', 'position']].head()}")
    
    # Optimize squad
    optimal_squad = optimize_squad(predictions_df, players_df, teams_df)
    print(f"Optimal squad shape: {optimal_squad.shape}")
    
    # Export
    optimal_squad.to_json('data/optimal_squad_new.json', orient='records', indent=2)
    print("Optimal squad saved to data/optimal_squad_new.json")

if __name__ == '__main__':
    main()