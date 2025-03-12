import time
import requests
import pandas as pd
import json
from datetime import datetime

# Get current FPL data
response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
fpl_data = json.loads(response.text)

# Extract defender data
defenders = [player for player in fpl_data['elements'] if player['element_type'] == 2]

# Create a DataFrame for defenders
defenders_df = pd.DataFrame(defenders)

# Get historical gameweek data for each defender
defender_history = []

for defender in defenders:
    player_id = defender['id']
    player_response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/')
    player_data = json.loads(player_response.text)

    # Extract history data and add player_id
    history = player_data['history']
    for match in history:
        match['player_id'] = player_id
        match['player_name'] = defender['web_name']
        match['team_id'] = defender['team']
        match['cost'] = defender['now_cost'] / 10  # Convert to millions
        defender_history.append(match)

    # Don't hit the API too hard
    time.sleep(0.1)

# Create DataFrame with all defender history
history_df = pd.DataFrame(defender_history)

# Save to CSV
history_df.to_csv('defender_history.csv', index=False)