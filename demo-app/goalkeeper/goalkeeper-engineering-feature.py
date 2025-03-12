import pandas as pd
import requests
import json
import numpy as np
import time
from datetime import datetime

print("Starting enhanced goalkeeper feature engineering...")

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
    print("Loading goalkeeper history data...")
    history_df = pd.read_csv('goalkeeper_history.csv')
    print(f"Loaded {len(history_df)} historical records")
except FileNotFoundError:
    print("Goalkeeper history file not found. Please run the load.py script first.")
    exit(1)

# Print column names to debug
print("\nAvailable columns in goalkeeper_history.csv:")
print(history_df.columns.tolist())

# Add team name mapping
team_map = {team['id']: team['name'] for team in fpl_data['teams']}
history_df['team_name'] = history_df['team_id'].map(team_map)

print("Creating base features...")

# ------------------ BASE FEATURES ------------------
# Standard point scoring components for goalkeepers
history_df['clean_sheet_points'] = history_df['clean_sheets'] * 4  # 4 points per clean sheet
history_df['save_points'] = (history_df['saves'] / 3).apply(np.floor).astype(int)  # 1 point per 3 saves
history_df['penalty_save_points'] = history_df['penalties_saved'] * 5 if 'penalties_saved' in history_df.columns else 0
history_df['goals_conceded_points'] = -1 * (history_df['goals_conceded'] / 2).apply(np.floor).astype(int)  # -1 point per 2 goals conceded
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

# Calculate minutes consistency (higher value = more consistent minutes)
history_df['minutes_rolling_std'] = history_df.groupby('player_id')['minutes'].transform(
    lambda x: x.rolling(5, min_periods=3).std())
history_df['minutes_rolling_mean'] = history_df.groupby('player_id')['minutes'].transform(
    lambda x: x.rolling(5, min_periods=3).mean())
history_df['minutes_consistency'] = 1.0
mask = history_df['minutes_rolling_mean'] > 0
history_df.loc[mask, 'minutes_consistency'] = 1 - (history_df.loc[mask, 'minutes_rolling_std'] / 90)
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

# Home/Away saves
history_df_home_saves = history_df[history_df['was_home'] == True].groupby('player_id')['saves'].mean().reset_index()
history_df_home_saves.columns = ['player_id', 'home_saves_per_game']
history_df_away_saves = history_df[history_df['was_home'] == False].groupby('player_id')['saves'].mean().reset_index()
history_df_away_saves.columns = ['player_id', 'away_saves_per_game']

# Join home/away features
history_df = pd.merge(history_df, history_df_home, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_away, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_home_cs, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_away_cs, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_home_saves, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_away_saves, on='player_id', how='left')

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
if 'opponent_team' in history_df.columns:
    history_df['opponent_team_strength'] = history_df['opponent_team'].map({team['id']: team.get('strength', 1000) for team in fpl_data['teams']})
    max_strength = max([team.get('strength', 1000) for team in fpl_data['teams']])
    history_df['opponent_strength_factor'] = history_df['opponent_team_strength'] / max_strength  # Normalize

    # Add more detailed opponent metrics (focusing on attack strength - key for goalkeepers)
    def get_opponent_attack(row):
        opp_id = row['opponent_team']
        if opp_id in team_strength:
            if row['was_home']:
                return team_strength[opp_id]['attack_away']
            else:
                return team_strength[opp_id]['attack_home']
        return 1000  # Default value if team not found

    history_df['opponent_attack_strength'] = history_df.apply(get_opponent_attack, axis=1) / 1200  # Normalize
else:
    print("Warning: 'opponent_team' column not found. Skipping opponent-based features.")
    history_df['opponent_team_strength'] = 1000
    history_df['opponent_strength_factor'] = 0.5
    history_df['opponent_attack_strength'] = 1000/1200

# ------------------ GOALKEEPER-SPECIFIC FEATURES ------------------
print("Calculating goalkeeper-specific features...")

# Save metrics
history_df['saves_per_game'] = history_df['saves'].astype(float)
history_df['recent_saves'] = history_df.groupby('player_id')['saves'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())

# Save points rate (how often they get save points)
history_df['save_points_per_game'] = history_df['save_points'].astype(float)
history_df['recent_save_points'] = history_df.groupby('player_id')['save_points'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())

# Clean sheet metrics
history_df['recent_clean_sheets'] = history_df.groupby('player_id')['clean_sheets'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())
history_df['clean_sheet_rate'] = history_df.groupby('player_id')['clean_sheets'].transform(
    lambda x: x.mean())

# Goals conceded metrics
history_df['goals_conceded_per_game'] = history_df['goals_conceded'].astype(float)
history_df['recent_goals_conceded'] = history_df.groupby('player_id')['goals_conceded'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())

# Penalty save metrics (if available)
if 'penalties_saved' in history_df.columns:
    history_df['penalties_saved_per_game'] = history_df['penalties_saved'].astype(float)
    history_df['has_saved_penalty'] = (history_df['penalties_saved'] > 0).astype(int)
else:
    history_df['penalties_saved_per_game'] = 0
    history_df['has_saved_penalty'] = 0

# Bonus point metrics
history_df['bonus_per_game'] = history_df['bonus'].astype(float)
history_df['recent_bonus'] = history_df.groupby('player_id')['bonus'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())

# Clean sheet streaks (consecutive clean sheets)
if 'round' in history_df.columns:
    # Initialize streak column
    history_df['clean_sheet_streak'] = 0
    
    # Process each player separately
    for player_id in history_df['player_id'].unique():
        player_data = history_df[history_df['player_id'] == player_id].sort_values('round')
        
        # Calculate streaks
        streak = 0
        for i, row in player_data.iterrows():
            if row['clean_sheets'] > 0:
                streak += 1
            else:
                streak = 0
            history_df.at[i, 'clean_sheet_streak'] = streak

# ------------------ ADVANCED METRICS ------------------
print("Calculating advanced metrics...")

# Expected goals conceded (if available)
if 'expected_goals_conceded' in history_df.columns:
    history_df['xGC_per_game'] = history_df['expected_goals_conceded'].astype(float)
    history_df['recent_xGC'] = history_df.groupby('player_id')['expected_goals_conceded'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    history_df['goals_conceded_minus_xGC'] = history_df['goals_conceded'] - history_df['expected_goals_conceded']
else:
    history_df['xGC_per_game'] = history_df['goals_conceded_per_game']
    history_df['recent_xGC'] = history_df['recent_goals_conceded']
    history_df['goals_conceded_minus_xGC'] = 0

# Save percentage (if shots faced data available)
if 'shots_faced' in history_df.columns:
    history_df['save_percentage'] = history_df.apply(
        lambda row: row['saves'] / row['shots_faced'] if row['shots_faced'] > 0 else 0, axis=1)
else:
    # Estimate shots faced from saves and goals conceded
    history_df['estimated_shots_faced'] = history_df['saves'] + history_df['goals_conceded']
    history_df['save_percentage'] = history_df.apply(
        lambda row: row['saves'] / row['estimated_shots_faced'] if row['estimated_shots_faced'] > 0 else 0, axis=1)

# Recent save percentage
history_df['recent_save_percentage'] = history_df.groupby('player_id')['save_percentage'].transform(
    lambda x: x.rolling(5, min_periods=1).mean())

# Points per 90 minutes
history_df['points_per_90'] = 90 * history_df['total_points'] / history_df['minutes'].clip(lower=1)

# ------------------ TEAM DEFENSIVE FEATURES ------------------
print("Calculating team defensive features...")

# Team defensive metrics
team_defense_metrics = {}
if 'round' in history_df.columns:
    for team_id in history_df['team_id'].unique():
        team_matches = history_df[history_df['team_id'] == team_id]
        if 'round' in team_matches.columns:
            team_matches = team_matches.sort_values('round')
        
        if not team_matches.empty:
            team_defense_metrics[team_id] = {}
            
            # Recent clean sheets
            team_defense_metrics[team_id]['recent_clean_sheets'] = team_matches['clean_sheets'].tail(5).sum() if len(team_matches) >= 5 else 0
            
            # Recent goals conceded
            team_defense_metrics[team_id]['recent_goals_conceded'] = team_matches['goals_conceded'].tail(5).sum() if len(team_matches) >= 5 else 0

# Add team metrics to player data
if team_defense_metrics:
    history_df['team_recent_clean_sheets'] = history_df['team_id'].map({k: v.get('recent_clean_sheets', 0) for k, v in team_defense_metrics.items()})
    history_df['team_recent_goals_conceded'] = history_df['team_id'].map({k: v.get('recent_goals_conceded', 0) for k, v in team_defense_metrics.items()})
else:
    history_df['team_recent_clean_sheets'] = 0
    history_df['team_recent_goals_conceded'] = 0

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
        
        # For goalkeepers, we're more interested in clean sheet potential (facing weak attacks)
        home_cs_potential = 1 / (team_strength.get(away_team, {}).get('attack_away', 1000) / 1000)
        away_cs_potential = 1 / (team_strength.get(home_team, {}).get('attack_home', 1000) / 1000)
        
        future_fixtures.append({
            'gw': gw,
            'team': home_team,
            'opponent': away_team,
            'is_home': True,
            'difficulty': difficulty,
            'cs_potential': home_cs_potential,
            'save_potential': team_strength.get(away_team, {}).get('attack_away', 1000) / 1000  # Higher potential for saves against strong attacks
        })
        
        future_fixtures.append({
            'gw': gw,
            'team': away_team,
            'opponent': home_team,
            'is_home': False,
            'difficulty': difficulty,
            'cs_potential': away_cs_potential,
            'save_potential': team_strength.get(home_team, {}).get('attack_home', 1000) / 1000
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
            'next_3_cs_potential': next_3_fixtures['cs_potential'].mean(),
            'next_3_save_potential': next_3_fixtures['save_potential'].mean()
        }

# Add future fixture data to player data
if team_future_difficulty:
    history_df['next_3_avg_difficulty'] = history_df['team_id'].map({k: v['next_3_avg_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_min_difficulty'] = history_df['team_id'].map({k: v['next_3_min_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_max_difficulty'] = history_df['team_id'].map({k: v['next_3_max_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_cs_potential'] = history_df['team_id'].map({k: v['next_3_cs_potential'] for k, v in team_future_difficulty.items()})
    history_df['next_3_save_potential'] = history_df['team_id'].map({k: v['next_3_save_potential'] for k, v in team_future_difficulty.items()})
else:
    history_df['next_3_avg_difficulty'] = 3  # Medium difficulty as default
    history_df['next_3_min_difficulty'] = 3
    history_df['next_3_max_difficulty'] = 3
    history_df['next_3_cs_potential'] = 1
    history_df['next_3_save_potential'] = 0.5

# ------------------ SAVE PROCESSED DATA ------------------
print("Saving processed data...")

# Fill NaN values with appropriate defaults
history_df = history_df.fillna({
    'avg_points_home': history_df['total_points'].mean(),
    'avg_points_away': history_df['total_points'].mean(),
    'home_clean_sheet_rate': 0.3,  # Conservative defaults
    'away_clean_sheet_rate': 0.2,
    'home_saves_per_game': 3,
    'away_saves_per_game': 3,
    'opponent_team_strength': 1000,
    'opponent_attack_strength': 1000/1200,
    'minutes_consistency': 0,
    'opponent_strength_factor': 0.6,  # Middle value
    'team_recent_clean_sheets': 1,
    'team_recent_goals_conceded': 5,
    'next_3_avg_difficulty': 3,
    'next_3_min_difficulty': 2,
    'next_3_max_difficulty': 4,
    'next_3_cs_potential': 1,
    'next_3_save_potential': 0.5,
    'save_percentage': 0.66,  # Average save percentage
    'recent_save_percentage': 0.66,
    'goals_conceded_minus_xGC': 0,
    'clean_sheet_streak': 0,
    'has_saved_penalty': 0
})

# Display the list of features we've created
feature_columns = [col for col in history_df.columns if col not in ['player_id', 'team_id', 'opponent_team', 'team_name']]
print("\nFeatures created:")
for col in feature_columns:
    print(f"- {col}")

# Save the processed data
history_df.to_csv('goalkeeper_processed.csv', index=False)
print(f"Enhanced processed data saved to 'goalkeeper_processed.csv' with {len(history_df)} rows and {len(feature_columns)} features")

# Save future fixtures for multi-gameweek planning
future_fixtures_df.to_csv('future_fixtures.csv', index=False)
print(f"Future fixture data saved to 'future_fixtures.csv' with {len(future_fixtures_df)} rows")

print("Feature engineering completed successfully!")