import requests
import json
import pandas as pd
import time
from datetime import datetime

print("Starting goalkeeper data collection...")

# Fetch current FPL data
try:
    print("Fetching data from FPL API...")
    response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    fpl_data = json.loads(response.text)
    print("FPL data fetched successfully")
except Exception as e:
    print(f"Error fetching FPL data: {e}")
    exit(1)

# Extract goalkeeper data (element_type 1 = goalkeepers)
goalkeepers = [player for player in fpl_data['elements'] if player['element_type'] == 1]
print(f"Found {len(goalkeepers)} goalkeepers in current FPL data")

# Team mapping for reference
team_map = {team['id']: team['name'] for team in fpl_data['teams']}

# Create a DataFrame for goalkeepers
goalkeepers_df = pd.DataFrame(goalkeepers)
print(f"Basic data extracted for {len(goalkeepers_df)} goalkeepers")

# Get historical gameweek data for each goalkeeper
goalkeeper_history = []

print("Fetching detailed history for each goalkeeper...")
for i, goalkeeper in enumerate(goalkeepers):
    player_id = goalkeeper['id']
    player_name = goalkeeper['web_name']
    
    print(f"Processing {i+1}/{len(goalkeepers)}: {player_name} (ID: {player_id})")
    
    try:
        player_response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/')
        player_data = json.loads(player_response.text)
        
        # Get this season's gameweek data
        if 'history' in player_data:
            history = player_data['history']
            for match in history:
                # Add player identification
                match['player_id'] = player_id
                match['player_name'] = player_name
                match['team_id'] = goalkeeper['team']
                match['team_name'] = team_map.get(goalkeeper['team'], "Unknown")
                match['position'] = "GK"
                match['cost'] = goalkeeper['now_cost'] / 10  # Convert to millions
                
                # Add to collection
                goalkeeper_history.append(match)
        
        # Get previous seasons' data
        if 'history_past' in player_data:
            for season in player_data['history_past']:
                # We'll add a summary row for each previous season
                season_summary = {
                    'player_id': player_id,
                    'player_name': player_name,
                    'team_id': goalkeeper['team'],
                    'team_name': team_map.get(goalkeeper['team'], "Unknown"),
                    'position': "GK",
                    'season': season['season_name'],
                    'cost': goalkeeper['now_cost'] / 10,  # Use current cost as estimate
                    'minutes': season['minutes'],
                    'total_points': season['total_points'],
                    'goals_scored': season['goals_scored'],
                    'assists': season['assists'],
                    'clean_sheets': season['clean_sheets'],
                    'goals_conceded': season['goals_conceded'],
                    'own_goals': season['own_goals'],
                    'penalties_saved': season['penalties_saved'],
                    'penalties_missed': season['penalties_missed'],
                    'yellow_cards': season['yellow_cards'],
                    'red_cards': season['red_cards'],
                    'saves': season['saves'],
                    'bonus': season['bonus'],
                    'bps': season['bps'],
                    'influence': season.get('influence', 0),
                    'creativity': season.get('creativity', 0),
                    'threat': season.get('threat', 0),
                    'ict_index': season.get('ict_index', 0),
                    'starts': season.get('starts', 0),
                    'expected_goals': season.get('expected_goals', 0),
                    'expected_assists': season.get('expected_assists', 0),
                    'expected_goal_involvements': season.get('expected_goal_involvements', 0),
                    'expected_goals_conceded': season.get('expected_goals_conceded', 0),
                    'is_past_season': True,
                }
                goalkeeper_history.append(season_summary)
        
        # Don't hit the API too hard
        time.sleep(0.5)
        
    except Exception as e:
        print(f"Error fetching history for {player_name} (ID: {player_id}): {e}")
        time.sleep(1)  # Wait a bit longer before next request

# Create DataFrame with all goalkeeper history
if goalkeeper_history:
    history_df = pd.DataFrame(goalkeeper_history)
    print(f"Created history DataFrame with {len(history_df)} records")
    
    # Add fixture difficulty
    fixtures = {}
    try:
        # Get fixture data
        fixtures_response = requests.get('https://fantasy.premierleague.com/api/fixtures/')
        fixtures_data = json.loads(fixtures_response.text)
        
        # Process fixtures to get difficulties
        for fixture in fixtures_data:
            if 'difficulty' in fixture:
                # Store difficulty for home and away teams
                fixtures[(fixture['team_h'], fixture['team_a'], fixture.get('event'))] = {
                    'home_difficulty': fixture['team_h_difficulty'],
                    'away_difficulty': fixture['team_a_difficulty']
                }
        
        # Add difficulty to history
        if 'event' in history_df.columns and 'team_h' in history_df.columns and 'team_a' in history_df.columns:
            def get_difficulty(row):
                key = (row['team_h'], row['team_a'], row['event'])
                if key in fixtures:
                    return fixtures[key]['home_difficulty'] if row['team_h'] == row['team_id'] else fixtures[key]['away_difficulty']
                return 3  # Default medium difficulty
            
            history_df['difficulty'] = history_df.apply(get_difficulty, axis=1)
    except Exception as e:
        print(f"Warning: Could not add fixture difficulty: {e}")
    
    # Save to CSV
    history_df.to_csv('goalkeeper_history.csv', index=False)
    print("Goalkeeper history data saved to 'goalkeeper_history.csv'")
    
    # Also save a basic profile of each goalkeeper
    goalkeepers_df.to_csv('goalkeeper_profiles.csv', index=False)
    print("Goalkeeper profiles saved to 'goalkeeper_profiles.csv'")
else:
    print("No history data collected. Please check the API or try again.")

print("Goalkeeper data collection completed.")