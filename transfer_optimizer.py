"""
FPL Transfer Optimizer
Suggests the best transfer for a given current 15-man squad to maximize predicted points.
Each transfer costs -4 points.
"""
import pandas as pd
import numpy as np
import json
import sys
from collections import Counter


# FPL constraints
BUDGET = 1000
SQUAD_SIZE = 15
STARTING_XI = 11
MAX_PER_TEAM = 3
POSITION_LIMITS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}

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

def check_constraints(squad_df, new_player, out_player=None):
    """Check if adding new_player and removing out_player keeps squad valid"""
    temp_df = squad_df.copy()
    if out_player is not None:
        temp_df = temp_df[temp_df['player_name'] != out_player['player_name']]
    temp_df = pd.concat([temp_df, pd.DataFrame([new_player])], ignore_index=True)
    
    # Check position limits
    pos_count = temp_df['position'].value_counts()
    for pos, limit in POSITION_LIMITS.items():
        if pos_count.get(pos, 0) > limit:
            return False
    
    # Check team limits
    team_count = temp_df['team_name'].value_counts()
    if team_count.max() > MAX_PER_TEAM:
        return False
    
    # Check budget
    total_cost = temp_df['now_cost'].sum()
    if total_cost > BUDGET:
        return False
    
    return True

# Load predictions
df = pd.read_csv('data/predicted_next_gw.csv')
players_info = pd.read_csv('data/players.csv', usecols=['player_id','chance_of_playing_next_round','chance_of_playing_this_round','minutes'])
df = df.merge(players_info, on='player_id', how='left')
filter_criteria = (
    (df['chance_of_playing_next_round'] >= 90) &
    (df['chance_of_playing_this_round'] >= 90) &
    (df['minutes'] > 0)
)
df = df[filter_criteria].copy()

# Load current squad
if len(sys.argv) > 1:
    # Load from JSON file
    current_squad_file = sys.argv[1]
    with open(current_squad_file, 'r') as f:
        current_squad_data = json.load(f)
    current_squad_df = pd.DataFrame(current_squad_data['squad'])
    # Ensure dtypes match df
    current_squad_df['now_cost'] = current_squad_df['now_cost'].astype(float)
    current_squad_df['predicted_next_gw'] = current_squad_df['predicted_next_gw'].astype(float)
    print(f"Loaded squad from {current_squad_file}")
else:
    # Load current squad interactively
    print("Enter your current 15-man squad player names, one per line (press Enter twice when done):")
    player_names = []
    for line in sys.stdin:
        name = line.strip()
        if not name:
            break
        player_names.append(name)

    if len(player_names) != 15:
        print(f"Error: You entered {len(player_names)} players. Need exactly 15.")
        sys.exit(1)

    # Find players in the dataset
    current_squad_df = df[df['player_name'].isin(player_names)].copy()
    if len(current_squad_df) != 15:
        missing = set(player_names) - set(current_squad_df['player_name'])
        print(f"Error: Could not find the following players in the dataset: {missing}")
        sys.exit(1)

    print(f"Loaded squad with {len(current_squad_df)} players.")

# Determine current formation: optimize
formation_tuple = (3, 4, 3)  # Default, will be optimized

# Optimize formation for current squad
best_total = -float('inf')
best_formation = formation_tuple
for form in VALID_FORMATIONS:
    starting_df = select_starting_xi(current_squad_df, form)
    total_points = starting_df['predicted_next_gw'].sum()
    if total_points > best_total:
        best_total = total_points
        best_formation = form
formation_tuple = best_formation
current_starting = select_starting_xi(current_squad_df, formation_tuple)
current_points = current_starting['predicted_next_gw'].sum()

print(f"Current squad total starting XI predicted points: {current_points}")
print(f"Current formation: {formation_tuple[0]}-{formation_tuple[1]}-{formation_tuple[2]}")

# Find best transfer
best_transfer = None
best_net_gain = -float('inf')

# Get current team and position counters
current_team_count = current_squad_df['team_name'].value_counts().to_dict()
current_pos_count = current_squad_df['position'].value_counts().to_dict()

for _, out_player in current_squad_df.iterrows():
    print(f"Considering transferring out: {out_player['player_name']} ({out_player['predicted_next_gw']} pts)")
    
    # Candidates: players not in squad, with predicted points > out_player's, and can afford
    free_budget_after_sell = BUDGET - current_squad_df['now_cost'].sum() + out_player['now_cost']
    
    # Filter candidates
    candidates = df[
        ~df['player_name'].isin(current_squad_df['player_name']) &
        (df['now_cost'] <= free_budget_after_sell)
    ].copy()
    
    # Further filter by constraints
    valid_candidates = []
    for _, cand in candidates.iterrows():
        # Check if adding this player would violate constraints
        temp_team_count = current_team_count.copy()
        temp_team_count[out_player['team_name']] -= 1
        if temp_team_count[out_player['team_name']] == 0:
            del temp_team_count[out_player['team_name']]
        temp_team_count[cand['team_name']] = temp_team_count.get(cand['team_name'], 0) + 1
        if temp_team_count[cand['team_name']] > MAX_PER_TEAM:
            continue
        
        temp_pos_count = current_pos_count.copy()
        temp_pos_count[out_player['position']] -= 1
        temp_pos_count[cand['position']] = temp_pos_count.get(cand['position'], 0) + 1
        if temp_pos_count[cand['position']] > POSITION_LIMITS[cand['position']]:
            continue
        
        valid_candidates.append(cand)
    
    if not valid_candidates:
        continue
    
    # For each valid candidate, create new squad and calculate new points
    for cand in valid_candidates:
        # Create new squad
        new_squad_df = current_squad_df[current_squad_df['player_name'] != out_player['player_name']].copy()
        new_squad_df = pd.concat([new_squad_df, cand.to_frame().T], ignore_index=True)
        
        # Optimize formation for new squad
        best_new_total = -float('inf')
        best_new_formation = formation_tuple
        for form in VALID_FORMATIONS:
            new_starting = select_starting_xi(new_squad_df, form)
            new_total = new_starting['predicted_next_gw'].sum()
            if new_total > best_new_total:
                best_new_total = new_total
                best_new_formation = form
        new_points = best_new_total
        
        net_gain = new_points - current_points - 4  # Transfer cost
        
        if net_gain > 0 and net_gain > best_net_gain:
            best_net_gain = net_gain
            best_transfer = {
                'out': out_player,
                'in': cand,
                'net_gain': net_gain,
                'new_points': new_points,
                'new_formation': best_new_formation
            }

if best_transfer:
    print("\nBest Transfer Found:")
    print(f"Transfer out: {best_transfer['out']['player_name']} ({best_transfer['out']['position']}, {best_transfer['out']['team_name']}, {best_transfer['out']['now_cost']}m, {best_transfer['out']['predicted_next_gw']:.2f} pts)")
    print(f"Transfer in: {best_transfer['in']['player_name']} ({best_transfer['in']['position']}, {best_transfer['in']['team_name']}, {best_transfer['in']['now_cost']}m, {best_transfer['in']['predicted_next_gw']:.2f} pts)")
    print(f"Net gain: {best_transfer['net_gain']:.2f} points")
    print(f"New total starting XI points: {best_transfer['new_points']:.2f}")
    
    # Create updated squad
    new_squad_df = current_squad_df[current_squad_df['player_name'] != best_transfer['out']['player_name']].copy()
    new_squad_df = pd.concat([new_squad_df, best_transfer['in'].to_frame().T], ignore_index=True)
    
    # Select starting XI with new formation
    new_starting_df = select_starting_xi(new_squad_df, best_transfer['new_formation'])
    new_starting_sorted = new_starting_df.sort_values('predicted_next_gw', ascending=False).reset_index(drop=True)
    new_captain = new_starting_sorted.iloc[0]
    new_vice_captain = new_starting_sorted.iloc[1]
    
    # Sort starting XI by position
    position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
    new_starting_xi_sorted = new_starting_df.copy()
    new_starting_xi_sorted['pos_order'] = new_starting_xi_sorted['position'].map(position_order)
    new_starting_xi_sorted = new_starting_xi_sorted.sort_values(['pos_order', 'predicted_next_gw'], ascending=[True, False]).drop(columns='pos_order').reset_index(drop=True)
    
    # Identify bench
    new_starting_ids = set(new_starting_xi_sorted['player_id'])
    new_bench_df = new_squad_df[~new_squad_df['player_id'].isin(new_starting_ids)].copy()
    new_bench_sorted = new_bench_df.copy()
    new_bench_sorted['pos_order'] = new_bench_sorted['position'].map(position_order)
    new_bench_sorted = new_bench_sorted.sort_values(['pos_order', 'predicted_next_gw'], ascending=[True, False]).drop(columns='pos_order').reset_index(drop=True)
    
    # Output JSON
    output_cols = ['player_name', 'position', 'team_name', 'now_cost', 'predicted_next_gw']
    updated_squad_json = {
        'squad': [
            {k: (int(v) if k == 'now_cost' else float(v) if k == 'predicted_next_gw' else str(v)) for k, v in rec.items()}
            for rec in new_squad_df[output_cols].to_dict(orient='records')
        ],
        'starting_xi': [
            {k: (int(v) if k == 'now_cost' else float(v) if k == 'predicted_next_gw' else str(v)) for k, v in rec.items()}
            for rec in new_starting_xi_sorted[output_cols].to_dict(orient='records')
        ],
        'bench': [
            {k: (int(v) if k == 'now_cost' else float(v) if k == 'predicted_next_gw' else str(v)) for k, v in rec.items()}
            for rec in new_bench_sorted[output_cols].to_dict(orient='records')
        ],
        'captain': {
            'player_name': str(new_captain['player_name']),
            'position': str(new_captain['position']),
            'team_name': str(new_captain['team_name']),
            'now_cost': int(new_captain['now_cost']),
            'predicted_next_gw': float(new_captain['predicted_next_gw'])
        },
        'vice_captain': {
            'player_name': str(new_vice_captain['player_name']),
            'position': str(new_vice_captain['position']),
            'team_name': str(new_vice_captain['team_name']),
            'now_cost': int(new_vice_captain['now_cost']),
            'predicted_next_gw': float(new_vice_captain['predicted_next_gw'])
        },
        'formation': f"{best_transfer['new_formation'][0]}-{best_transfer['new_formation'][1]}-{best_transfer['new_formation'][2]}",
        'transfer': {
            'out': {
                'player_name': str(best_transfer['out']['player_name']),
                'position': str(best_transfer['out']['position']),
                'team_name': str(best_transfer['out']['team_name']),
                'now_cost': int(best_transfer['out']['now_cost']),
                'predicted_next_gw': float(best_transfer['out']['predicted_next_gw'])
            },
            'in': {
                'player_name': str(best_transfer['in']['player_name']),
                'position': str(best_transfer['in']['position']),
                'team_name': str(best_transfer['in']['team_name']),
                'now_cost': int(best_transfer['in']['now_cost']),
                'predicted_next_gw': float(best_transfer['in']['predicted_next_gw'])
            },
            'net_gain': float(best_transfer['net_gain'])
        }
    }
    with open('data/updated_squad.json', 'w') as f:
        json.dump(updated_squad_json, f, indent=2)
    print('\nUpdated squad JSON saved to data/updated_squad.json')
else:
    print("No beneficial transfers found.")