
import requests
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

player_info_fields = [
    'can_transact','can_select','chance_of_playing_next_round','chance_of_playing_this_round','code','cost_change_event','cost_change_event_fall','cost_change_start','cost_change_start_fall','dreamteam_count','element_type','ep_next','ep_this','event_points','first_name','form','id','in_dreamteam','news','news_added','now_cost','photo','points_per_game','removed','second_name','selected_by_percent','special','squad_number','status','team','team_code','total_points','transfers_in','transfers_in_event','transfers_out','transfers_out_event','value_form','value_season','web_name','region','team_join_date','birth_date','has_temporary_code','opta_code','minutes','goals_scored','assists','clean_sheets','goals_conceded','own_goals','penalties_saved','penalties_missed','yellow_cards','red_cards','saves','bonus','bps','influence','creativity','threat','ict_index','clearances_blocks_interceptions','recoveries','tackles','defensive_contribution','starts','expected_goals','expected_assists','expected_goal_involvements','expected_goals_conceded','influence_rank','influence_rank_type','creativity_rank','creativity_rank_type','threat_rank','threat_rank_type','ict_index_rank','ict_index_rank_type','corners_and_indirect_freekicks_order','corners_and_indirect_freekicks_text','direct_freekicks_order','direct_freekicks_text','penalties_order','penalties_text','expected_goals_per_90','saves_per_90','expected_assists_per_90','expected_goal_involvements_per_90','expected_goals_conceded_per_90','goals_conceded_per_90','now_cost_rank','now_cost_rank_type','form_rank','form_rank_type','points_per_game_rank','points_per_game_rank_type','selected_rank','selected_rank_type','starts_per_90','clean_sheets_per_90','defensive_contribution_per_90'
]

extra_fields = ['team_name', 'position', 'player_id', 'player_name']

history_past_fields = [
    'season_name','element_code','start_cost','end_cost','total_points','minutes','goals_scored','assists','clean_sheets','goals_conceded','own_goals','penalties_saved','penalties_missed','yellow_cards','red_cards','saves','bonus','bps','influence','creativity','threat','ict_index','clearances_blocks_interceptions','recoveries','tackles','defensive_contribution','starts','expected_goals','expected_assists','expected_goal_involvements','expected_goals_conceded'
]

bootstrap_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
bootstrap_response = requests.get(bootstrap_url)
if bootstrap_response.status_code == 200:
    bootstrap_data = bootstrap_response.json()
else:
    raise Exception(f'Failed to fetch bootstrap-static data: {bootstrap_response.status_code}')


# Use bootstrap-static data for team and position mappings
players = bootstrap_data['elements']
teams = bootstrap_data['teams']
element_types = bootstrap_data['element_types']

team_id_to_name = {team['id']: team['name'] for team in teams}
position_id_to_short = {etype['id']: etype['singular_name_short'] for etype in element_types}

# Use bootstrap-static data for opponent details (strength, short_name, etc.)
team_id_to_details = {team['id']: team for team in teams}

def fetch_history_past(player):
    # Start with player info fields
    base = {k: player.get(k, 0) for k in player_info_fields}
    # Fill chance_of_playing_next_round with 100 if null or None
    if base.get('chance_of_playing_next_round') in [None, '', 'null']:
        base['chance_of_playing_next_round'] = 100
    # Fill chance_of_playing_this_round with 100 if null or None
    if base.get('chance_of_playing_this_round') in [None, '', 'null']:
        base['chance_of_playing_this_round'] = 100
    # Add derived fields
    base['team_name'] = team_id_to_name.get(player.get('team', 0), '')
    base['position'] = position_id_to_short.get(player.get('element_type', 0), '')
    base['player_id'] = player.get('id', None)
    base['player_name'] = player.get('web_name', None)
    player_id = player.get('id', None)
    # Fetch history_past from API
    if player_id is not None:
        url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
        try:
            resp = requests.get(url)
            if resp.status_code == 200:
                summary = resp.json()
                history_past = list(reversed(summary.get('history_past', [])))
                # Most recent season first
                for i in range(4):
                    if i < len(history_past):
                        season = history_past[i]
                        for k in history_past_fields:
                            base[f'history_past_{i+1}_{k}'] = season.get(k, 0)
                    else:
                        for k in history_past_fields:
                            base[f'history_past_{i+1}_{k}'] = 0
                # Add last 5 games stats
                history = summary.get('history', [])
                last5 = history[-5:] if len(history) >= 5 else [None]*(5-len(history)) + history
                last5_fields = [
                    'total_points','minutes','goals_scored','assists','clean_sheets','goals_conceded','own_goals',
                    'penalties_saved','penalties_missed','yellow_cards','red_cards','saves','bonus','bps',
                    'influence','creativity','threat','ict_index','clearances_blocks_interceptions','recoveries',
                    'tackles','defensive_contribution','starts','expected_goals','expected_assists',
                    'expected_goal_involvements','expected_goals_conceded'
                ]
                for idx, game in enumerate(last5):
                    for field in last5_fields:
                        colname = f'last5_{field}_{idx+1}'
                        if game is None:
                            base[colname] = 0
                        else:
                            val = game.get(field, 0)
                            # Convert string numbers to float if needed
                            if isinstance(val, str):
                                try:
                                    val = float(val)
                                except:
                                    val = 0
                            base[colname] = val
                    # Add next fixture and opponent details
                    fixtures = summary.get('fixtures', [])
                    next_fixture = None
                    for fx in fixtures:
                        if not fx.get('finished', False):
                            next_fixture = fx
                            break
                    if next_fixture:
                        base['next_fixture_id'] = next_fixture.get('id', 0)
                        base['next_fixture_event'] = next_fixture.get('event', 0)
                        base['next_fixture_is_home'] = next_fixture.get('is_home', False)
                        base['next_fixture_difficulty'] = next_fixture.get('difficulty', 0)
                        # Use team_h or team_a for opponent depending on is_home
                        if next_fixture.get('is_home', False):
                            opp_id = next_fixture.get('team_a', 0)
                        else:
                            opp_id = next_fixture.get('team_h', 0)
                        base['next_fixture_opponent_team'] = opp_id
                        opp = team_id_to_details.get(opp_id, {})
                        base['next_opponent_name'] = opp.get('name', '')
                        base['next_opponent_short_name'] = opp.get('short_name', '')
                        base['next_opponent_strength'] = opp.get('strength', 0)
                        base['next_opponent_strength_overall_home'] = opp.get('strength_overall_home', 0)
                        base['next_opponent_strength_overall_away'] = opp.get('strength_overall_away', 0)
                        base['next_opponent_strength_attack_home'] = opp.get('strength_attack_home', 0)
                        base['next_opponent_strength_attack_away'] = opp.get('strength_attack_away', 0)
                        base['next_opponent_strength_defence_home'] = opp.get('strength_defence_home', 0)
                        base['next_opponent_strength_defence_away'] = opp.get('strength_defence_away', 0)
                    else:
                        # No upcoming fixture
                        base['next_fixture_id'] = 0
                        base['next_fixture_event'] = 0
                        base['next_fixture_is_home'] = False
                        base['next_fixture_difficulty'] = 0
                        base['next_fixture_opponent_team'] = 0
                        base['next_opponent_name'] = ''
                        base['next_opponent_short_name'] = ''
                        base['next_opponent_strength'] = 0
                        base['next_opponent_strength_overall_home'] = 0
                        base['next_opponent_strength_overall_away'] = 0
                        base['next_opponent_strength_attack_home'] = 0
                        base['next_opponent_strength_attack_away'] = 0
                        base['next_opponent_strength_defence_home'] = 0
                        base['next_opponent_strength_defence_away'] = 0
            else:
                print(f'Failed to fetch summary for player {player_id}: {resp.status_code}')
        except Exception as e:
            print(f'Error fetching summary for player {player_id}: {e}')
    return base

all_rows = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(fetch_history_past, player) for player in players]
    for future in as_completed(futures):
        all_rows.append(future.result())

final_columns = player_info_fields + extra_fields + [
    f'history_past_1_{k}'
    for k in history_past_fields
]
final_columns += [
    f'last5_{field}_{i}'
    for field in [
        'total_points','minutes','goals_scored','assists','clean_sheets','goals_conceded','own_goals',
        'penalties_saved','penalties_missed','yellow_cards','red_cards','saves','bonus','bps',
        'influence','creativity','threat','ict_index','clearances_blocks_interceptions','recoveries',
        'tackles','defensive_contribution','starts','expected_goals','expected_assists',
        'expected_goal_involvements','expected_goals_conceded'
    ]
    for i in range(1,6)
]
final_columns += [
    'next_fixture_id','next_fixture_event','next_fixture_is_home','next_fixture_difficulty','next_fixture_opponent_team',
    'next_opponent_name','next_opponent_short_name','next_opponent_strength',
    'next_opponent_strength_overall_home','next_opponent_strength_overall_away',
    'next_opponent_strength_attack_home','next_opponent_strength_attack_away',
    'next_opponent_strength_defence_home','next_opponent_strength_defence_away'
]

df = pd.DataFrame(all_rows)
df = df.fillna(0).infer_objects(copy=False)
df = df[final_columns]
df = df.sort_values('player_id')
df.to_csv('data/players.csv', index=False)
print('Exported correct history_past data for all players to data/players.csv.')
