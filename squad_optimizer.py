"""
Fantasy Premier League Predictor MVP - Squad Optimization
- Select optimal 15-man squad under FPL constraints (budget, max 3 per team, position requirements)
- Pick best starting XI and captain
- Output DataFrame (console) and JSON (frontend)
"""
import pandas as pd
import numpy as np
import json
from collections import Counter


# FPL constraints
BUDGET = 1000  # 100.0 in FPL units
SQUAD_SIZE = 15
STARTING_XI = 11
MAX_PER_TEAM = 3
POSITION_LIMITS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}

# Valid FPL formations: (DEF, MID, FWD)
VALID_FORMATIONS = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 4, 2),
    (4, 3, 3),
    (4, 5, 1),
    (5, 3, 2),
    (5, 4, 1)
]

def get_starting_limits(formation):
    def_, mid, fwd = formation
    return {'GKP': 1, 'DEF': def_, 'MID': mid, 'FWD': fwd}

def select_starting_xi(squad_df, formation):
    limits = get_starting_limits(formation)
    starting = []
    start_pos_counter = Counter()
    for _, row in squad_df.sort_values('predicted_next_gw', ascending=False).iterrows():
        pos = row['position']
        if start_pos_counter[pos] < limits[pos]:
            starting.append(row)
            start_pos_counter[pos] += 1
        if len(starting) == 11:
            break
    return pd.DataFrame(starting)

# Load player predictions and info
df = pd.read_csv('data/predicted_next_gw.csv')
# Load additional player info for filtering
players_info = pd.read_csv('data/players.csv', usecols=['player_id','chance_of_playing_next_round','chance_of_playing_this_round','minutes'])
# Merge info into predictions
df = df.merge(players_info, on='player_id', how='left')
# Filter out players with low chance or no minutes
# Filter out players with low chance or no minutes, and exclude those flagged as injured/suspended if such columns exist
filter_criteria = (
    (df['chance_of_playing_next_round'] >= 90) &
    (df['chance_of_playing_this_round'] >= 90) &
    (df['minutes'] > 0)
)
if 'status' in df.columns:
    # Exclude players flagged as 'injured', 'suspended', or 'unavailable'
    filter_criteria &= ~df['status'].isin(['injured', 'suspended', 'unavailable'])
df = df[filter_criteria].copy()

# Select optimal squad

# Smarter greedy squad selection
def select_optimal_squad(df, budget, squad_size, max_per_team, position_limits):
    df = df.copy()
    df['value_per_cost'] = df['predicted_next_gw'] / df['now_cost']
    squad = []
    used_budget = 0
    team_counter = Counter()
    position_counter = Counter()
    remaining_positions = position_limits.copy()

    for _ in range(squad_size):
        # Scarcity: fewer remaining slots = higher scarcity
        df['scarcity'] = df['position'].map(lambda pos: 1 / (remaining_positions[pos] if remaining_positions[pos] > 0 else 1e-6))
        # Dynamic priority: value per cost * scarcity
        df['priority'] = df['value_per_cost'] * df['scarcity']
        # Filter out players who can't be picked due to constraints
        candidates = df[
            (df['now_cost'] + used_budget <= budget) &
            (df['team_name'].map(lambda t: team_counter[t] < max_per_team)) &
            (df['position'].map(lambda p: position_counter[p] < position_limits[p]))
        ]
        if candidates.empty:
            break
        # Pick the highest priority candidate
        pick = candidates.sort_values('priority', ascending=False).iloc[0]
        squad.append(pick)
        used_budget += int(pick['now_cost'])
        team_counter[pick['team_name']] += 1
        position_counter[pick['position']] += 1
        remaining_positions[pick['position']] -= 1
        # Remove picked player from pool
        df = df[df['player_name'] != pick['player_name']]

    return pd.DataFrame(squad)

squad_df = select_optimal_squad(
    df,
    BUDGET,
    SQUAD_SIZE,
    MAX_PER_TEAM,
    POSITION_LIMITS
)


# Choose formation: set here or optimize
import sys
if len(sys.argv) > 1:
    # User can pass formation as e.g. "3-4-3"
    formation_str = sys.argv[1]
    formation_tuple = tuple(map(int, formation_str.split('-')))
    if formation_tuple not in VALID_FORMATIONS:
        print(f"Invalid formation {formation_tuple}, using default 3-4-3.")
        formation_tuple = (3, 4, 3)
else:
    # Optimize for best formation
    best_total = -float('inf')
    best_formation = None
    best_starting_df = None
    for formation in VALID_FORMATIONS:
        starting_df = select_starting_xi(squad_df, formation)
        total_points = starting_df['predicted_next_gw'].sum()
        if total_points > best_total:
            best_total = total_points
            best_formation = formation
            best_starting_df = starting_df
    formation_tuple = best_formation
    starting_df = best_starting_df
    print(f"Selected optimal formation: {formation_tuple[0]}-{formation_tuple[1]}-{formation_tuple[2]}")
if 'starting_df' not in locals():
    starting_df = select_starting_xi(squad_df, formation_tuple)
    print(f"Selected formation: {formation_tuple[0]}-{formation_tuple[1]}-{formation_tuple[2]}")

starting_sorted = starting_df.sort_values('predicted_next_gw', ascending=False).reset_index(drop=True)
captain = starting_sorted.iloc[0]
vice_captain = starting_sorted.iloc[1]


print('Squad DataFrame columns:', squad_df.columns.tolist())
output_cols = ['player_name', 'position', 'team_name', 'now_cost', 'predicted_next_gw']
print('Optimal Squad:')
print(squad_df[output_cols])

# Sort starting XI by position order: GKP, DEF, MID, FWD
position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
starting_xi_sorted = starting_df.copy()
starting_xi_sorted['pos_order'] = starting_xi_sorted['position'].map(position_order)
starting_xi_sorted = starting_xi_sorted.sort_values(['pos_order', 'predicted_next_gw'], ascending=[True, False]).drop(columns='pos_order').reset_index(drop=True)

# Identify bench (not in starting XI)
starting_ids = set(starting_xi_sorted['player_id'])
bench_df = squad_df[~squad_df['player_id'].isin(starting_ids)].copy()
bench_sorted = bench_df.copy()
bench_sorted['pos_order'] = bench_sorted['position'].map(position_order)
bench_sorted = bench_sorted.sort_values(['pos_order', 'predicted_next_gw'], ascending=[True, False]).drop(columns='pos_order').reset_index(drop=True)

print('\nStarting XI (sorted):')
print(starting_xi_sorted[output_cols])
print('\nBench:')
print(bench_sorted[output_cols])
print(f'\nCaptain: {captain["player_name"]} ({captain["predicted_next_gw"]} pts predicted)')
print(f'Vice-Captain: {vice_captain["player_name"]} ({vice_captain["predicted_next_gw"]} pts predicted)')
print(f'Formation: {formation_tuple[0]}-{formation_tuple[1]}-{formation_tuple[2]}')

# Output JSON (frontend)
squad_json = {
    'squad': [
        {k: (int(v) if k == 'now_cost' else float(v) if k == 'predicted_next_gw' else str(v)) for k, v in rec.items()}
        for rec in squad_df[output_cols].to_dict(orient='records')
    ],
    'starting_xi': [
        {k: (int(v) if k == 'now_cost' else float(v) if k == 'predicted_next_gw' else str(v)) for k, v in rec.items()}
        for rec in starting_xi_sorted[output_cols].to_dict(orient='records')
    ],
    'bench': [
        {k: (int(v) if k == 'now_cost' else float(v) if k == 'predicted_next_gw' else str(v)) for k, v in rec.items()}
        for rec in bench_sorted[output_cols].to_dict(orient='records')
    ],
    'captain': {
        'player_name': str(captain['player_name']),
        'position': str(captain['position']),
        'team_name': str(captain['team_name']),
        'now_cost': int(captain['now_cost']),
        'predicted_next_gw': float(captain['predicted_next_gw'])
    }
    ,
    'vice_captain': {
        'player_name': str(vice_captain['player_name']),
        'position': str(vice_captain['position']),
        'team_name': str(vice_captain['team_name']),
        'now_cost': int(vice_captain['now_cost']),
        'predicted_next_gw': float(vice_captain['predicted_next_gw'])
    },
    'formation': f'{formation_tuple[0]}-{formation_tuple[1]}-{formation_tuple[2]}'
}
with open('data/optimal_squad.json', 'w') as f:
    json.dump(squad_json, f, indent=2)
print('\nSquad JSON saved to data/optimal_squad.json')
