"""
FPL Data Exporter
Fetches data from FPL API endpoints and saves as JSON files.
"""
import requests
import json
import time
import os
import concurrent.futures

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# API URLs
BOOTSTRAP_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_URL = 'https://fantasy.premierleague.com/api/fixtures/'

def fetch_and_save(url, filename):
    """Fetch data from URL and save as JSON."""
    print(f"Fetching {url}...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    with open(f'data/{filename}', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to data/{filename}")
    return data

def fetch_player_history(player_id):
    """Fetch history for a single player."""
    url = f'https://fantasy.premierleague.com/api/element-summary/{player_id}/'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def main():
    # Fetch bootstrap static (players, teams, gameweeks)
    bootstrap_data = fetch_and_save(BOOTSTRAP_URL, 'bootstrap.json')

    # Fetch fixtures
    fetch_and_save(FIXTURES_URL, 'fixtures.json')

    # Fetch per-player detailed history using parallel processing
    player_histories = {}
    elements = bootstrap_data.get('elements', [])
    print(f"Fetching history for {len(elements)} players using parallel processing...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_player_history, player['id']): player['id'] for player in elements}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            player_id = futures[future]
            try:
                data = future.result()
                player_histories[player_id] = data
                if (i + 1) % 50 == 0:
                    print(f"Fetched {i + 1}/{len(elements)} players")
            except Exception as e:
                print(f"Error fetching player {player_id}: {e}")

    # Save player histories
    with open('data/player_histories.json', 'w') as f:
        json.dump(player_histories, f, indent=2)
    print("Saved player histories to data/player_histories.json")

if __name__ == '__main__':
    main()
