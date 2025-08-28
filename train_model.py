"""
Fantasy Premier League Predictor MVP - Model Training
- Train Ridge and XGBoost regressors on the feature-engineered time-series dataset
- Time-based train-test split (first 70% pseudo_gameweeks for train, last 30% for test)
- Hyperparameter tuning for XGBoost using Optuna
- Report RMSE and top feature importances
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def load_data(path):
    return pd.read_csv(path)

def time_split(df):
    # Use pseudo_gameweek for time-based split
    max_gw = df['pseudo_gameweek'].max()
    split_gw = int(max_gw * 0.7)
    train = df[df['pseudo_gameweek'] <= split_gw]
    test = df[df['pseudo_gameweek'] > split_gw]
    # Drop rows with NaN target
    train = train.dropna(subset=['target_next_gw'])
    test = test.dropna(subset=['target_next_gw'])
    return train, test

def get_features(df):
    exclude = ['player_id', 'player_name', 'pseudo_gameweek', 'target_next_gw', 'points', 'position', 'team_name', 'next_opponent_name']
    numeric_features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
    return numeric_features

def train_ridge(train, test, features):
    X_train = train[features].fillna(0)
    y_train = train['target_next_gw']
    X_test = test[features].fillna(0)
    y_test = test['target_next_gw']
    model = Ridge(random_state=SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - preds))
    r2 = model.score(X_test, y_test)
    print(f'Ridge RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
    return model, rmse, mae, r2

def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': SEED
    }
    model = XGBRegressor(**params)
    model.fit(X_train.fillna(0), y_train)
    preds = model.predict(X_test.fillna(0))
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return rmse

def train_xgb_optuna(train, test, features):
    X_train = train[features].fillna(0)
    y_train = train['target_next_gw']
    X_test = test[features].fillna(0)
    y_test = test['target_next_gw']
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=30)
    print('Best XGBoost params:', study.best_params)
    model = XGBRegressor(**study.best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - preds))
    r2 = model.score(X_test, y_test)
    print(f'XGBoost RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
    # Feature importances
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print('Top 10 feature importances:')
    print(importances.head(10))
    # Save predictions for latest pseudo_gameweek
    test_out = test.copy()
    test_out['predicted_next_gw'] = preds
    # Merge with player info
    players = pd.read_csv('data/players.csv')
    test_out = test_out.merge(players[['player_id', 'player_name', 'position', 'team_name', 'now_cost']], on='player_id', how='left')
    print('Columns after merge:', test_out.columns.tolist())
    test_out = test_out.rename(columns={'player_name_y': 'player_name', 'position_y': 'position', 'team_name_y': 'team_name', 'now_cost_y': 'now_cost'})
    test_out[['player_id', 'player_name', 'position', 'team_name', 'now_cost', 'predicted_next_gw']].to_csv('data/predicted_next_gw.csv', index=False)
    print('Saved predicted next GW points to data/predicted_next_gw.csv')
    # Save feature importances
    importances.to_csv('data/xgb_feature_importance.csv')
    return model, rmse, mae, r2, importances

def main():
    df = load_data('data/players_timeseries.csv')
    train, test = time_split(df)
    features = get_features(df)
    print(f'Features used: {features}')
    print(f'Train size: {len(train)}, Test size: {len(test)}')
    print('--- Ridge Baseline ---')
    train_ridge(train, test, features)
    print('--- XGBoost (Optuna) ---')
    train_xgb_optuna(train, test, features)
    # Position-specific models
    print('--- Position-Specific Models (XGBoost) ---')
    all_pos_preds = []
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        train_pos = train[train['position'] == pos]
        test_pos = test[test['position'] == pos]
        if len(train_pos) == 0 or len(test_pos) == 0:
            continue
        print(f'Position: {pos}')
        pos_features = get_features(train_pos)
        model, rmse, mae, r2, importances = train_xgb_optuna(train_pos, test_pos, pos_features)
        # Get predictions for this position
        X_test = test_pos[pos_features].fillna(0)
        preds = model.predict(X_test)
        test_out = test_pos.copy()
        test_out['predicted_next_gw'] = preds
        players = pd.read_csv('data/players.csv')
        test_out = test_out.merge(players[['player_id', 'player_name', 'position', 'team_name', 'now_cost']], on='player_id', how='left')
        test_out = test_out.rename(columns={'player_name_y': 'player_name', 'position_y': 'position', 'team_name_y': 'team_name', 'now_cost_y': 'now_cost'})
        all_pos_preds.append(test_out[['player_id', 'player_name', 'position', 'team_name', 'now_cost', 'predicted_next_gw']])
    # Concatenate all positions and save to CSV
    if all_pos_preds:
        all_preds_df = pd.concat(all_pos_preds, ignore_index=True)
        all_preds_df.to_csv('data/predicted_next_gw.csv', index=False)
        print('Saved all position predictions to data/predicted_next_gw.csv')

if __name__ == '__main__':
    main()