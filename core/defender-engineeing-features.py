import pandas as pd
import requests
import json
import numpy as np
import time
from datetime import datetime

print("Starting enhanced defender feature engineering...")

# Fetch FPL data
try:
    print("Fetching data from FPL API...")
    response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    fpl_data = json.loads(response.text)
    print("FPL data fetched successfully")
    
    # Get fixture data for difficulty calculations
    fixtures_response = requests.get('https://fantasy.premierleague.com/api/fixtures/')
    fixtures_data = json.loads(fixtures_response.text)
    print(f"Fetched {len(fixtures_data)} fixtures")
except Exception as e:
    print(f"Error fetching FPL data: {e}")
    exit(1)

# Load the historical data
try:
    print("Loading defender history data...")
    history_df = pd.read_csv('defender_history.csv')
    print(f"Loaded {len(history_df)} historical records")
except FileNotFoundError:
    print("Defender history file not found. Please run the load.py script first.")
    exit(1)

# Add team name mapping
team_map = {team['id']: team['name'] for team in fpl_data['teams']}
history_df['team_name'] = history_df['team_id'].map(team_map)

print("Creating base features...")

# ------------------ BASE FEATURES ------------------
# Standard point scoring components
history_df['clean_sheet_points'] = history_df['clean_sheets'] * 4
history_df['goals_points'] = history_df['goals_scored'] * 6
history_df['assists_points'] = history_df['assists'] * 3
history_df['bonus_points'] = history_df['bonus']

# ------------------ FORM FEATURES ------------------
print("Calculating form features...")

# Keep original names for compatibility
history_df['minutes_trend'] = history_df.groupby('player_id')['minutes'].transform(
    lambda x: x.rolling(3, min_periods=1).mean())
history_df['rolling_points'] = history_df.groupby('player_id')['total_points'].transform(
    lambda x: x.rolling(3, min_periods=1).mean())

# Additional form windows
history_df['rolling_points_5'] = history_df.groupby('player_id')['total_points'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())
history_df['minutes_trend_5'] = history_df.groupby('player_id')['minutes'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())

# Fix for minutes_consistency - using vectorized operations instead of lambda with if/else
history_df['minutes_rolling_std'] = history_df.groupby('player_id')['minutes'].transform(
    lambda x: x.rolling(5, min_periods=3).std())
history_df['minutes_rolling_mean'] = history_df.groupby('player_id')['minutes'].transform(
    lambda x: x.rolling(5, min_periods=3).mean())
# Calculate consistency (higher value = more consistent minutes)
history_df['minutes_consistency'] = 1.0
mask = history_df['minutes_rolling_mean'] > 0
history_df.loc[mask, 'minutes_consistency'] = 1 - (history_df.loc[mask, 'minutes_rolling_std'] / 90)
# Ensure values are between 0 and 1
history_df['minutes_consistency'] = history_df['minutes_consistency'].clip(0, 1)

# Value metrics
history_df['value'] = history_df['total_points'] / history_df['cost']
history_df['recent_value'] = history_df['rolling_points_5'] / history_df['cost']

# ------------------ HOME/AWAY FEATURES ------------------
print("Calculating home/away performance features...")

# Home/Away performance
history_df_home = history_df[history_df['was_home'] == True].groupby('player_id')['total_points'].mean().reset_index()
history_df_home.columns = ['player_id', 'avg_points_home']
history_df_away = history_df[history_df['was_home'] == False].groupby('player_id')['total_points'].mean().reset_index()
history_df_away.columns = ['player_id', 'avg_points_away']

# Home/Away clean sheet tendency
history_df_home_cs = history_df[history_df['was_home'] == True].groupby('player_id')['clean_sheets'].mean().reset_index()
history_df_home_cs.columns = ['player_id', 'home_clean_sheet_rate']
history_df_away_cs = history_df[history_df['was_home'] == False].groupby('player_id')['clean_sheets'].mean().reset_index()
history_df_away_cs.columns = ['player_id', 'away_clean_sheet_rate']

# Join home/away features
history_df = pd.merge(history_df, history_df_home, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_away, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_home_cs, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_away_cs, on='player_id', how='left')

# ------------------ OPPONENT FEATURES ------------------
print("Calculating opponent-based features...")

# Team strength data
team_strength = {team['id']: {
    'attack_home': team.get('strength_attack_home', 1000),
    'attack_away': team.get('strength_attack_away', 1000),
    'defence_home': team.get('strength_defence_home', 1000),
    'defence_away': team.get('strength_defence_away', 1000),
    'overall': team.get('strength', 1000),
} for team in fpl_data['teams']}

# Add opponent strength metrics
history_df['opponent_team_strength'] = history_df['opponent_team'].map({team['id']: team.get('strength', 1000) for team in fpl_data['teams']})
# Find max strength to normalize
max_strength = max([team.get('strength', 1000) for team in fpl_data['teams']])
history_df['opponent_strength_factor'] = history_df['opponent_team_strength'] / max_strength  # Normalize

# Add more detailed opponent metrics
def get_opponent_attack(row):
    opp_id = row['opponent_team']
    if opp_id in team_strength:
        if row['was_home']:
            return team_strength[opp_id]['attack_away']
        else:
            return team_strength[opp_id]['attack_home']
    return 1000  # Default value if team not found

history_df['opponent_attack_strength'] = history_df.apply(get_opponent_attack, axis=1) / 1200  # Normalize

# ------------------ ATTACKING FEATURES ------------------
print("Calculating attacking contribution features...")

# Attacking returns - recent goals and assists
history_df['recent_goals'] = history_df.groupby('player_id')['goals_scored'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())
history_df['recent_assists'] = history_df.groupby('player_id')['assists'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())
history_df['attacking_returns'] = history_df['recent_goals'] + history_df['recent_assists']

# Bonus point trend
history_df['bonus_trend'] = history_df.groupby('player_id')['bonus'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())

# ------------------ TEAM PERFORMANCE FEATURES ------------------
print("Calculating team performance features...")

# Create team performance metrics
team_performances = {}
if 'round' in history_df.columns:
    for team_id in history_df['team_id'].unique():
        team_matches = history_df[history_df['team_id'] == team_id]
        if 'round' in team_matches.columns:
            team_matches = team_matches.sort_values('round')
        if not team_matches.empty:
            # Calculate recent clean sheets for the team
            if 'clean_sheets' in team_matches.columns:
                recent_cs = team_matches['clean_sheets'].tail(5).sum() if len(team_matches) >= 5 else team_matches['clean_sheets'].sum()
            else:
                recent_cs = 0
                
            # Calculate recent goals conceded
            if 'goals_conceded' in team_matches.columns:
                recent_gc = team_matches['goals_conceded'].tail(5).sum() if len(team_matches) >= 5 else team_matches['goals_conceded'].sum()
            else:
                recent_gc = 0
                
            team_performances[team_id] = {
                'recent_clean_sheets': recent_cs,
                'recent_goals_conceded': recent_gc
            }

# Add team performance to player data
if team_performances:
    history_df['team_clean_sheets_last_5'] = history_df['team_id'].map({k: v['recent_clean_sheets'] for k, v in team_performances.items()})
    history_df['team_goals_conceded_last_5'] = history_df['team_id'].map({k: v['recent_goals_conceded'] for k, v in team_performances.items()})
else:
    history_df['team_clean_sheets_last_5'] = 0
    history_df['team_goals_conceded_last_5'] = 0

# ------------------ FIXTURE DIFFICULTY FEATURES ------------------
print("Calculating future fixture features...")

# Process fixture data to calculate future difficulty
current_gw = history_df['round'].max() if 'round' in history_df.columns else 38

# Prepare future fixtures (for planning multiple gameweeks ahead)
future_fixtures = []
for fixture in fixtures_data:
    if fixture.get('event') and fixture.get('event') > current_gw:
        home_team = fixture['team_h']
        away_team = fixture['team_a']
        difficulty = fixture.get('difficulty', 3)
        gw = fixture['event']
        
        future_fixtures.append({
            'gw': gw,
            'team': home_team,
            'opponent': away_team,
            'is_home': True,
            'difficulty': difficulty
        })
        
        future_fixtures.append({
            'gw': gw,
            'team': away_team,
            'opponent': home_team,
            'is_home': False,
            'difficulty': difficulty
        })

future_fixtures_df = pd.DataFrame(future_fixtures)

# Add difficulty data for players' upcoming fixtures
team_future_difficulty = {}
for team_id in history_df['team_id'].unique():
    team_fixtures = future_fixtures_df[future_fixtures_df['team'] == team_id]
    if not team_fixtures.empty:
        next_3_fixtures = team_fixtures.sort_values('gw').head(3)
        team_future_difficulty[team_id] = {
            'next_3_avg_difficulty': next_3_fixtures['difficulty'].mean(),
            'next_3_min_difficulty': next_3_fixtures['difficulty'].min(),
            'next_3_max_difficulty': next_3_fixtures['difficulty'].max(),
        }

# Add future fixture data to player data
if team_future_difficulty:
    history_df['next_3_avg_difficulty'] = history_df['team_id'].map({k: v['next_3_avg_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_min_difficulty'] = history_df['team_id'].map({k: v['next_3_min_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_max_difficulty'] = history_df['team_id'].map({k: v['next_3_max_difficulty'] for k, v in team_future_difficulty.items()})
else:
    history_df['next_3_avg_difficulty'] = 3  # Medium difficulty as default
    history_df['next_3_min_difficulty'] = 3
    history_df['next_3_max_difficulty'] = 3

# ------------------ SAVE PROCESSED DATA ------------------
print("Saving processed data...")

# Fill NaN values with appropriate defaults
history_df = history_df.fillna({
    'avg_points_home': history_df['total_points'].mean(),
    'avg_points_away': history_df['total_points'].mean(),
    'home_clean_sheet_rate': 0.3,  # Average clean sheet rate
    'away_clean_sheet_rate': 0.2,
    'opponent_team_strength': 1000,
    'opponent_attack_strength': 1000/1200,
    'minutes_consistency': 0,
    'opponent_strength_factor': 0.6,  # Middle value
    'team_clean_sheets_last_5': 1.5,  # Average values
    'team_goals_conceded_last_5': 5,
    'next_3_avg_difficulty': 3,
    'next_3_min_difficulty': 2,
    'next_3_max_difficulty': 4,
})

# Display the list of features we've created
feature_columns = [col for col in history_df.columns if col not in ['player_id', 'team_id', 'opponent_team', 'team_name']]
print("\nFeatures created:")
for col in feature_columns:
    print(f"- {col}")

# Save the processed data
history_df.to_csv('defender_processed.csv', index=False)
print(f"Enhanced processed data saved to 'defender_processed.csv' with {len(history_df)} rows and {len(feature_columns)} features")

# Save future fixtures for multi-gameweek planning
future_fixtures_df.to_csv('future_fixtures.csv', index=False)
print(f"Future fixture data saved to 'future_fixtures.csv' with {len(future_fixtures_df)} rows")

print("Feature engineering completed successfully!")