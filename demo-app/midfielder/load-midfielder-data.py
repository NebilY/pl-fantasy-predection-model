import time
import requests
import pandas as pd
import json
from datetime import datetime

# Get current FPL data
response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
fpl_data = json.loads(response.text)

# Extract midfielder data
midfielders = [player for player in fpl_data['elements'] if player['element_type'] == 3]  # Element type 3 for midfielders

# Create a DataFrame for midfielders
midfielders_df = pd.DataFrame(midfielders)

# Get historical gameweek data for each midfielder
midfielder_history = []

for midfielder in midfielders:
    player_id = midfielder['id']
    
    try:
        player_response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/')
        player_data = json.loads(player_response.text)
        
        # Extract history data and add player context
        history = player_data['history']
        for match in history:
            match['player_id'] = player_id
            match['player_name'] = midfielder['web_name']
            match['team_id'] = midfielder['team']
            match['cost'] = midfielder['now_cost'] / 10  # Convert to millions
            match['team_goals_scored'] = None  # Placeholder for team goals
            
            midfielder_history.append(match)
        
        # Don't hit the API too hard
        time.sleep(0.1)
    
    except Exception as e:
        print(f"Error fetching data for midfielder {player_id} ({midfielder['web_name']}): {e}")
        continue

# Create DataFrame with all midfielder history
history_df = pd.DataFrame(midfielder_history)

# Try to get team goals for each match if possible
try:
    fixtures_response = requests.get('https://fantasy.premierleague.com/api/fixtures/')
    fixtures_data = json.loads(fixtures_response.text)
    
    # Create a mapping of fixtures to team goals
    fixture_goals = {}
    for fixture in fixtures_data:
        fixture_goals[fixture['id']] = {
            'home_team': fixture['team_h'],
            'away_team': fixture['team_a'],
            'home_goals': fixture.get('team_h_score', 0),
            'away_goals': fixture.get('team_a_score', 0)
        }
    
    # Update team goals in history
    for index, row in history_df.iterrows():
        fixture_id = row.get('fixture')
        if fixture_id in fixture_goals:
            match_goals = fixture_goals[fixture_id]
            if row['team_id'] == match_goals['home_team']:
                history_df.at[index, 'team_goals_scored'] = match_goals['home_goals']
            elif row['team_id'] == match_goals['away_team']:
                history_df.at[index, 'team_goals_scored'] = match_goals['away_goals']
except Exception as e:
    print(f"Could not fetch team goals: {e}")

# Save to CSV
history_df.to_csv('midfielder_history.csv', index=False)

print(f"Saved midfielder history data with {len(history_df)} records to midfielder_history.csv")