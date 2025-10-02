"""
Fantasy Premier League Predictor MVP - Squad Optimization
- Select optimal 15-man squad under FPL constraints (budget, max 3 per team, position requirements)
- Pick best starting XI and captain
- Output DataFrame (console) and JSON (frontend)
"""
import pandas as pd
import numpy as np
import json
import os
import copy
from collections import Counter


# FPL constraints
BUDGET = 1000  # 100.0 in FPL units
SQUAD_SIZE = 15
STARTING_XI = 11
MAX_PER_TEAM = 3
POSITION_LIMITS = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
TRANSFER_COST_POINTS = 4

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


def load_current_squad(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Warning: Unable to parse current squad JSON at {path}.")
        return None

    squad_records = data.get('squad', [])
    if not squad_records:
        return None

    df = pd.DataFrame(squad_records)
    return df if not df.empty else None


def _assess_player_availability(row):
    reasons = []

    if pd.isna(row.get('player_id')):
        reasons.append('no projection record')
    chance_next = row.get('chance_of_playing_next_round')
    if pd.notna(chance_next) and chance_next < 90:
        reasons.append(f'chance_next={chance_next}')
    chance_this = row.get('chance_of_playing_this_round')
    if pd.notna(chance_this) and chance_this < 90:
        reasons.append(f'chance_this={chance_this}')
    minutes = row.get('minutes')
    if pd.notna(minutes) and minutes <= 0:
        reasons.append('no recent minutes')

    is_available = len(reasons) == 0
    return is_available, reasons


def enrich_current_squad(current_df, df_full):
    if current_df is None or current_df.empty:
        return None

    rename_map = {
        'position': 'position_saved',
        'team_name': 'team_name_saved',
        'now_cost': 'now_cost_saved',
        'predicted_next_gw': 'predicted_next_gw_saved'
    }
    enriched = current_df.rename(columns=rename_map)

    merge_cols = [
        'player_id',
        'position',
        'team_name',
        'now_cost',
        'predicted_next_gw',
        'chance_of_playing_next_round',
        'chance_of_playing_this_round',
        'minutes'
    ]
    available_cols = [col for col in merge_cols if col in df_full.columns]
    enriched = enriched.merge(
        df_full[['player_name'] + available_cols],
        on='player_name',
        how='left'
    )

    if 'position' in enriched.columns:
        enriched['position_current'] = enriched['position'].fillna(enriched['position_saved'])
    else:
        enriched['position_current'] = enriched['position_saved']

    if 'team_name' in enriched.columns:
        enriched['team_name_current'] = enriched['team_name'].fillna(enriched['team_name_saved'])
    else:
        enriched['team_name_current'] = enriched['team_name_saved']

    if 'now_cost' in enriched.columns:
        enriched['now_cost_current'] = enriched['now_cost'].fillna(enriched['now_cost_saved'])
    else:
        enriched['now_cost_current'] = enriched['now_cost_saved']

    if 'predicted_next_gw' in enriched.columns:
        enriched['predicted_next_gw_current'] = enriched['predicted_next_gw'].fillna(enriched['predicted_next_gw_saved'])
    else:
        enriched['predicted_next_gw_current'] = enriched['predicted_next_gw_saved']

    availability_checks = enriched.apply(_assess_player_availability, axis=1)
    enriched['is_available'] = [item[0] for item in availability_checks]
    enriched['availability_notes'] = [
        '; '.join(item[1]) if item[1] else '' for item in availability_checks
    ]

    return enriched


def _first_valid(row, columns):
    for col in columns:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def format_player_payload(row, include_availability=False):
    if row is None:
        return None

    player_name = str(row['player_name'])
    position = _first_valid(row, ['position_current', 'position', 'position_saved'])
    team_name = _first_valid(row, ['team_name_current', 'team_name', 'team_name_saved'])
    now_cost = _first_valid(row, ['now_cost', 'now_cost_current', 'now_cost_saved'])
    predicted = _first_valid(
        row,
        ['predicted_next_gw', 'predicted_next_gw_current', 'predicted_next_gw_saved']
    )

    payload = {
        'player_name': player_name,
    }
    if position is not None:
        payload['position'] = str(position)
    if team_name is not None:
        payload['team_name'] = str(team_name)
    if now_cost is not None:
        payload['now_cost'] = int(now_cost)
    if predicted is not None:
        payload['predicted_next_gw'] = float(predicted)

    if include_availability:
        payload['is_available'] = bool(row.get('is_available', False))
        notes = row.get('availability_notes')
        if isinstance(notes, str) and notes:
            payload['availability_notes'] = notes

    return payload


def describe_player_short(payload):
    if payload is None:
        return 'None'

    name = payload.get('player_name', 'Unknown')
    team = payload.get('team_name')
    team_part = f" ({team})" if team else ''
    predicted = payload.get('predicted_next_gw')
    predicted_part = ''
    if predicted is not None:
        predicted_part = f" {predicted:.2f} pts"

    return f"{name}{team_part}{predicted_part}"


def compute_transfer_recommendations(current_enriched, new_squad_df):
    if current_enriched is None or current_enriched.empty:
        return [], {
            'total_transfers': 0,
            'points_gain_before_hits': 0.0,
            'points_hit': 0.0,
            'net_points_gain': 0.0
        }

    new_names = set(new_squad_df['player_name'])
    current_names = set(current_enriched['player_name'])

    transfers = []
    total_delta = 0.0
    total_transfers = 0
    position_order = ['GKP', 'DEF', 'MID', 'FWD']

    for pos in position_order:
        outs = current_enriched[
            (~current_enriched['player_name'].isin(new_names)) &
            (current_enriched['position_current'] == pos)
        ].sort_values('predicted_next_gw_current', ascending=False, na_position='last')

        ins = new_squad_df[
            (~new_squad_df['player_name'].isin(current_names)) &
            (new_squad_df['position'] == pos)
        ].sort_values('predicted_next_gw', ascending=False)

        max_len = max(len(outs), len(ins))
        for idx in range(max_len):
            out_row = outs.iloc[idx] if idx < len(outs) else None
            in_row = ins.iloc[idx] if idx < len(ins) else None

            reason = None
            delta = None
            net_gain = None
            forced = False

            out_pred = None
            if out_row is not None:
                out_pred = out_row.get('predicted_next_gw_current')
                if not out_row['is_available']:
                    notes = out_row['availability_notes']
                    reason = f"Availability: {notes}" if notes else 'Availability concern'
                    forced = True
                elif in_row is not None and pd.notna(out_pred):
                    delta = float(in_row['predicted_next_gw']) - float(out_pred)
                    if delta > 0:
                        reason = f"Predicted points upgrade (+{delta:.2f})"
                    elif delta < 0:
                        reason = f"Budget rebalancing ({delta:.2f} pts impact)"
                    else:
                        reason = 'Like-for-like swap'
            elif in_row is not None:
                reason = 'Squad depth upgrade'

            if delta is None and in_row is not None and out_row is not None and pd.notna(out_pred):
                delta = float(in_row['predicted_next_gw']) - float(out_pred)

            if delta is not None:
                net_gain = delta - TRANSFER_COST_POINTS

            keep_transfer = False
            if forced:
                keep_transfer = True
            elif delta is not None and net_gain is not None and net_gain > 0:
                keep_transfer = True
            elif delta is None and in_row is not None and out_row is None:
                predicted_value = float(in_row['predicted_next_gw'])
                keep_transfer = predicted_value - TRANSFER_COST_POINTS > 0

            if not keep_transfer:
                continue

            transfer_entry = {
                'position': pos,
                'out': format_player_payload(out_row, include_availability=True) if out_row is not None else None,
                'in': format_player_payload(in_row) if in_row is not None else None,
                'transfer_cost': TRANSFER_COST_POINTS
            }
            if reason:
                transfer_entry['reason'] = reason
            if delta is not None:
                transfer_entry['predicted_points_delta'] = round(delta, 3)
                transfer_entry['net_points_after_cost'] = round(net_gain, 3)

            transfers.append(transfer_entry)
            total_transfers += 1
            if delta is not None:
                total_delta += delta

    points_hit = total_transfers * TRANSFER_COST_POINTS
    net_points_gain = total_delta - points_hit
    summary = {
        'total_transfers': total_transfers,
        'points_gain_before_hits': round(total_delta, 3),
        'points_hit': round(points_hit, 3),
        'net_points_gain': round(net_points_gain, 3)
    }

    return transfers, summary


def build_final_squad(current_enriched, optimized_squad_df, transfers):
    columns = ['player_id', 'player_name', 'position', 'team_name', 'now_cost', 'predicted_next_gw']

    if current_enriched is None or current_enriched.empty:
        return optimized_squad_df[columns].copy().reset_index(drop=True)

    # Start from current squad projection
    final_df = current_enriched[[
        'player_id',
        'player_name',
        'position_current',
        'team_name_current',
        'now_cost_current',
        'predicted_next_gw_current'
    ]].rename(columns={
        'position_current': 'position',
        'team_name_current': 'team_name',
        'now_cost_current': 'now_cost',
        'predicted_next_gw_current': 'predicted_next_gw'
    }).copy()

    for transfer in transfers:
        out_payload = transfer.get('out')
        in_payload = transfer.get('in')

        if out_payload:
            final_df = final_df[final_df['player_name'] != out_payload.get('player_name')]

        if in_payload:
            in_name = in_payload.get('player_name')
            replacement = optimized_squad_df[optimized_squad_df['player_name'] == in_name]
            if replacement.empty:
                continue
            final_df = pd.concat([final_df, replacement[columns]], ignore_index=True)

    # Ensure squad size by topping up with optimized squad players if needed
    if len(final_df) < SQUAD_SIZE:
        remaining = optimized_squad_df[~optimized_squad_df['player_name'].isin(final_df['player_name'])]
        if not remaining.empty:
            final_df = pd.concat([final_df, remaining[columns].head(SQUAD_SIZE - len(final_df))], ignore_index=True)

    return final_df.reset_index(drop=True)

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
df_full = df.copy()
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

optimized_squad_df = select_optimal_squad(
    df,
    BUDGET,
    SQUAD_SIZE,
    MAX_PER_TEAM,
    POSITION_LIMITS
)

current_squad_path = 'data/updated_squad.json'
current_squad_df = load_current_squad(current_squad_path)
current_enriched = enrich_current_squad(current_squad_df, df_full) if current_squad_df is not None else None

transfer_payload, transfer_summary = compute_transfer_recommendations(current_enriched, optimized_squad_df)
final_squad_df = build_final_squad(current_enriched, optimized_squad_df, transfer_payload)


# Choose formation: set here or optimize
import sys
if len(sys.argv) > 1:
    formation_str = sys.argv[1]
    formation_tuple = tuple(map(int, formation_str.split('-')))
    if formation_tuple not in VALID_FORMATIONS:
        print(f"Invalid formation {formation_tuple}, using default 3-4-3.")
        formation_tuple = (3, 4, 3)
    starting_df = select_starting_xi(final_squad_df, formation_tuple)
else:
    best_total = -float('inf')
    best_formation = None
    best_starting_df = None
    for formation in VALID_FORMATIONS:
        candidate_df = select_starting_xi(final_squad_df, formation)
        total_points = candidate_df['predicted_next_gw'].sum()
        if total_points > best_total:
            best_total = total_points
            best_formation = formation
            best_starting_df = candidate_df
    formation_tuple = best_formation
    starting_df = best_starting_df
    print(f"Selected optimal formation for final squad: {formation_tuple[0]}-{formation_tuple[1]}-{formation_tuple[2]}")

if 'starting_df' not in locals():
    starting_df = select_starting_xi(final_squad_df, formation_tuple)
    print(f"Selected formation: {formation_tuple[0]}-{formation_tuple[1]}-{formation_tuple[2]}")

starting_sorted = starting_df.sort_values('predicted_next_gw', ascending=False).reset_index(drop=True)
captain = starting_sorted.iloc[0]
vice_captain = starting_sorted.iloc[1]

output_cols = ['player_name', 'position', 'team_name', 'now_cost', 'predicted_next_gw']
final_total_predicted = float(final_squad_df['predicted_next_gw'].sum())
optimized_total_predicted = float(optimized_squad_df['predicted_next_gw'].sum())
current_total_predicted = None
current_unavailable_payload = []

if current_enriched is not None:
    current_total_predicted = float(current_enriched['predicted_next_gw_current'].fillna(0).sum())
    summary_cols = [
        'player_name',
        'position_current',
        'team_name_current',
        'predicted_next_gw_current',
        'is_available',
        'availability_notes'
    ]
    display_df = current_enriched[summary_cols].rename(columns={
        'player_name': 'Player',
        'position_current': 'Pos',
        'team_name_current': 'Team',
        'predicted_next_gw_current': 'Predicted',
        'is_available': 'Available',
        'availability_notes': 'Notes'
    })
    print('\nCurrent squad health check:')
    print(display_df)

    current_unavailable_payload = [
        format_player_payload(row, include_availability=True)
        for _, row in current_enriched[~current_enriched['is_available']].iterrows()
    ]
else:
    print('\nNo existing squad found at data/updated_squad.json; using optimized squad as baseline.')

if current_total_predicted is not None:
    print(f"Total predicted points (current squad): {current_total_predicted:.2f}")
print(f"Total predicted points (final squad): {final_total_predicted:.2f}")
print(f"Total predicted points (optimized benchmark): {optimized_total_predicted:.2f}")

if transfer_payload:
    print('\nRecommended transfers (including 4pt hit each):')
    for transfer in transfer_payload:
        out_desc = describe_player_short(transfer.get('out'))
        in_desc = describe_player_short(transfer.get('in'))
        details = []
        reason = transfer.get('reason')
        if reason:
            details.append(reason)
        delta = transfer.get('predicted_points_delta')
        net = transfer.get('net_points_after_cost')
        if delta is not None:
            details.append(f"Δ {delta:+.2f} pts before hit")
        if net is not None:
            details.append(f"Net {net:+.2f} pts after hit")
        detail_str = f" ({'; '.join(details)})" if details else ''
        print(f" - {transfer['position']}: {out_desc} -> {in_desc}{detail_str}")

    if current_total_predicted is not None:
        projected_after_hits = current_total_predicted + transfer_summary['net_points_gain']
        print(
            f"Total transfers: {transfer_summary['total_transfers']} | "
            f"Gain before hits: {transfer_summary['points_gain_before_hits']:+.2f} | "
            f"Hits: {transfer_summary['points_hit']:+.2f} | "
            f"Net gain: {transfer_summary['net_points_gain']:+.2f}"
        )
        print(f"Projected total after hits: {projected_after_hits:.2f}")
else:
    print('\nNo transfers recommended; current squad is already optimal after accounting for hits.')

# Sort starting XI by position order: GKP, DEF, MID, FWD
position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
starting_xi_sorted = starting_df.copy()
starting_xi_sorted['pos_order'] = starting_xi_sorted['position'].map(position_order)
starting_xi_sorted = starting_xi_sorted.sort_values(['pos_order', 'predicted_next_gw'], ascending=[True, False]).drop(columns='pos_order').reset_index(drop=True)

# Identify bench (not in starting XI)
starting_ids = set(starting_xi_sorted['player_id'])
bench_df = final_squad_df[~final_squad_df['player_id'].isin(starting_ids)].copy()
bench_sorted = bench_df.copy()
bench_sorted['pos_order'] = bench_sorted['position'].map(position_order)
bench_sorted = bench_sorted.sort_values(['pos_order', 'predicted_next_gw'], ascending=[True, False]).drop(columns='pos_order').reset_index(drop=True)

print('\nFinal squad (all players):')
print(final_squad_df[output_cols])
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
        for rec in final_squad_df[output_cols].to_dict(orient='records')
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
    },
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

updated_squad_payload = copy.deepcopy(squad_json)
updated_squad_payload['transfer_summary'] = transfer_summary
if current_total_predicted is not None:
    updated_squad_payload['current_total_predicted_points'] = round(current_total_predicted, 3)
updated_squad_payload['final_total_predicted_points'] = round(final_total_predicted, 3)
updated_squad_payload['optimized_total_predicted_points'] = round(optimized_total_predicted, 3)
if transfer_payload:
    updated_squad_payload['recommended_transfers'] = transfer_payload
if current_unavailable_payload:
    updated_squad_payload['current_unavailable_players'] = current_unavailable_payload

with open('data/updated_squad.json', 'w') as f:
    json.dump(updated_squad_payload, f, indent=2)

print('\nSquad JSON saved to data/optimal_squad.json and data/updated_squad.json')
