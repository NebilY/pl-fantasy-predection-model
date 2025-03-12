import pandas as pd
import requests
import json
import numpy as np
import time
from datetime import datetime

print("Starting enhanced forward feature engineering...")

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
    print("Loading forward history data...")
    history_df = pd.read_csv('forward_history.csv')
    print(f"Loaded {len(history_df)} historical records")
except FileNotFoundError:
    print("Forward history file not found. Please run the load.py script first.")
    exit(1)

# Print column names to debug
print("\nAvailable columns in forward_history.csv:")
print(history_df.columns.tolist())

# Add team name mapping
team_map = {team['id']: team['name'] for team in fpl_data['teams']}
history_df['team_name'] = history_df['team_id'].map(team_map)

print("Creating base features...")

# ------------------ BASE FEATURES ------------------
# Standard point scoring components for forwards
history_df['goals_points'] = history_df['goals_scored'] * 4  # Forwards get 4 points per goal
history_df['assists_points'] = history_df['assists'] * 3
history_df['bonus_points'] = history_df['bonus']

# Forward-specific penalties
if 'penalties_scored' in history_df.columns:
    # Note: penalties already counted in goals, this is just for analysis
    history_df['penalties_scored_points'] = history_df['penalties_scored'] * 4
else:
    history_df['penalties_scored_points'] = 0

if 'penalties_missed' in history_df.columns:
    history_df['penalties_missed_points'] = history_df['penalties_missed'] * (-2)
else:
    history_df['penalties_missed_points'] = 0

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

# Home/Away goal scoring tendency
history_df_home_goals = history_df[history_df['was_home'] == True].groupby('player_id')['goals_scored'].mean().reset_index()
history_df_home_goals.columns = ['player_id', 'home_goals_per_game']
history_df_away_goals = history_df[history_df['was_home'] == False].groupby('player_id')['goals_scored'].mean().reset_index()
history_df_away_goals.columns = ['player_id', 'away_goals_per_game']

# Join home/away features
history_df = pd.merge(history_df, history_df_home, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_away, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_home_goals, on='player_id', how='left')
history_df = pd.merge(history_df, history_df_away_goals, on='player_id', how='left')

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

    # Add more detailed opponent metrics (focusing on defensive strength - key for forwards)
    def get_opponent_defense(row):
        opp_id = row['opponent_team']
        if opp_id in team_strength:
            if row['was_home']:
                return team_strength[opp_id]['defence_away']
            else:
                return team_strength[opp_id]['defence_home']
        return 1000  # Default value if team not found

    history_df['opponent_defense_strength'] = history_df.apply(get_opponent_defense, axis=1) / 1200  # Normalize
else:
    print("Warning: 'opponent_team' column not found. Skipping opponent-based features.")
    history_df['opponent_team_strength'] = 1000
    history_df['opponent_strength_factor'] = 0.5
    history_df['opponent_defense_strength'] = 1000/1200

# ------------------ ATTACKING FEATURES ------------------
print("Calculating attacking contribution features...")

# Basic attacking returns - recent goals and assists
history_df['recent_goals'] = history_df.groupby('player_id')['goals_scored'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())
history_df['recent_assists'] = history_df.groupby('player_id')['assists'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())
history_df['attacking_returns'] = history_df['recent_goals'] + history_df['recent_assists']

# Goal scoring streaks (consecutive games with goals)
# This requires sorting data properly
if 'round' in history_df.columns:
    # Initialize streak column
    history_df['goal_streak'] = 0
    
    # Process each player separately
    for player_id in history_df['player_id'].unique():
        player_data = history_df[history_df['player_id'] == player_id].sort_values('round')
        
        # Calculate streaks
        streak = 0
        for i, row in player_data.iterrows():
            if row['goals_scored'] > 0:
                streak += 1
            else:
                streak = 0
            history_df.at[i, 'goal_streak'] = streak

# Consistency of attacking returns
history_df['goals_per_90'] = 90 * history_df['goals_scored'] / history_df['minutes'].clip(lower=1)
history_df['assists_per_90'] = 90 * history_df['assists'] / history_df['minutes'].clip(lower=1)
history_df['goal_involvement_per_90'] = history_df['goals_per_90'] + history_df['assists_per_90']

# Add shots data if available
if 'shots' in history_df.columns:
    history_df['shots_per_90'] = 90 * history_df['shots'] / history_df['minutes'].clip(lower=1)
else:
    history_df['shots_per_90'] = 0

# Add big chances if available
if 'big_chances' in history_df.columns:
    history_df['big_chances_per_90'] = 90 * history_df['big_chances'] / history_df['minutes'].clip(lower=1)
else:
    history_df['big_chances_per_90'] = 0

# Add creativity and threat data if available (FPL-specific metrics)
if 'creativity' in history_df.columns:
    history_df['recent_creativity'] = history_df.groupby('player_id')['creativity'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
else:
    history_df['recent_creativity'] = 0

if 'threat' in history_df.columns:
    history_df['recent_threat'] = history_df.groupby('player_id')['threat'].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
else:
    history_df['recent_threat'] = 0

# Bonus point trend - especially important for forwards
history_df['bonus_trend'] = history_df.groupby('player_id')['bonus'].transform(
    lambda x: x.rolling(5, min_periods=1).sum())

# ------------------ ADVANCED METRICS ------------------
print("Calculating advanced metrics...")

# Shot conversion rate (if shots data available)
if 'shots' in history_df.columns and 'goals_scored' in history_df.columns:
    # Avoid division by zero
    history_df['shot_conversion'] = history_df.apply(
        lambda row: row['goals_scored'] / row['shots'] if row['shots'] > 0 else 0, axis=1)
    # Also calculate recent conversion rate
    history_df['recent_shots'] = history_df.groupby('player_id')['shots'].transform(
        lambda x: x.rolling(5, min_periods=1).sum())
    history_df['recent_shot_conversion'] = history_df.apply(
        lambda row: row['recent_goals'] / row['recent_shots'] if row['recent_shots'] > 0 else 0, axis=1)
else:
    history_df['shot_conversion'] = 0
    history_df['recent_shot_conversion'] = 0

# Expected goals and assists (if available)
if 'expected_goals' in history_df.columns:
    history_df['xG_per_90'] = 90 * history_df['expected_goals'] / history_df['minutes'].clip(lower=1)
    history_df['goals_minus_xG'] = history_df['goals_scored'] - history_df['expected_goals']
    # Recent xG
    history_df['recent_xG'] = history_df.groupby('player_id')['expected_goals'].transform(
        lambda x: x.rolling(5, min_periods=1).sum())
else:
    history_df['xG_per_90'] = 0
    history_df['goals_minus_xG'] = 0
    history_df['recent_xG'] = 0

if 'expected_assists' in history_df.columns:
    history_df['xA_per_90'] = 90 * history_df['expected_assists'] / history_df['minutes'].clip(lower=1)
    history_df['assists_minus_xA'] = history_df['assists'] - history_df['expected_assists']
    # Recent xA
    history_df['recent_xA'] = history_df.groupby('player_id')['expected_assists'].transform(
        lambda x: x.rolling(5, min_periods=1).sum())
else:
    history_df['xA_per_90'] = 0
    history_df['assists_minus_xA'] = 0
    history_df['recent_xA'] = 0

# Expected goal involvement (xG + xA)
if 'expected_goal_involvements' in history_df.columns:
    history_df['xGI_per_90'] = 90 * history_df['expected_goal_involvements'] / history_df['minutes'].clip(lower=1)
else:
    history_df['xGI_per_90'] = history_df['xG_per_90'] + history_df['xA_per_90']

# Goal involvement percentage - skip if team_goals_scored not available
print("Checking for team goals data...")
if 'team_goals_scored' in history_df.columns and 'round' in history_df.columns:
    print("Calculating goal involvement percentage...")
    # This requires mapping each player's goals and assists against their team's total goals
    team_total_goals = {}
    for team_id in history_df['team_id'].unique():
        for round_num in history_df['round'].unique():
            team_goals = history_df[(history_df['team_id'] == team_id) & 
                                  (history_df['round'] == round_num)]['team_goals_scored'].sum()
            if team_goals > 0:
                team_total_goals[(team_id, round_num)] = team_goals

    if team_total_goals:
        # Create a function to get team's goals for a specific match
        def get_team_goals(row):
            key = (row['team_id'], row['round'])
            return team_total_goals.get(key, 0)
        
        # Add team goals column
        history_df['team_match_goals'] = history_df.apply(get_team_goals, axis=1)
        
        # Calculate goal involvement percentage
        history_df['goal_involvement_pct'] = history_df.apply(
            lambda row: (row['goals_scored'] + row['assists']) / row['team_match_goals'] 
                        if row['team_match_goals'] > 0 else 0, axis=1)
        
        # Calculate recent goal involvement percentage (5 game average)
        history_df['recent_goal_inv_pct'] = history_df.groupby('player_id')['goal_involvement_pct'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
    else:
        history_df['goal_involvement_pct'] = 0
        history_df['recent_goal_inv_pct'] = 0
else:
    print("Skipping goal involvement percentage calculation (team_goals_scored not available)")
    history_df['goal_involvement_pct'] = 0
    history_df['recent_goal_inv_pct'] = 0

# Penalty duty (if data available)
if 'penalties_taken' in history_df.columns:
    history_df['penalty_taker'] = (history_df['penalties_taken'] > 0).astype(int)
else:
    history_df['penalty_taker'] = 0

# Successful dribbles (if available) - often correlates with forward performances
if 'successful_dribbles' in history_df.columns:
    history_df['dribbles_per_90'] = 90 * history_df['successful_dribbles'] / history_df['minutes'].clip(lower=1)
else:
    history_df['dribbles_per_90'] = 0

# ------------------ TEAM PERFORMANCE FEATURES ------------------
print("Calculating team performance features...")

# Skip team metrics if team_goals_scored not available
history_df['team_recent_goals'] = 0

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
        
        # For forwards, we're more interested in attacking potential (facing weak defenses)
        home_attacking_potential = 1 / (team_strength.get(away_team, {}).get('defence_away', 1000) / 1000)
        away_attacking_potential = 1 / (team_strength.get(home_team, {}).get('defence_home', 1000) / 1000)
        
        future_fixtures.append({
            'gw': gw,
            'team': home_team,
            'opponent': away_team,
            'is_home': True,
            'difficulty': difficulty,
            'attacking_potential': home_attacking_potential
        })
        
        future_fixtures.append({
            'gw': gw,
            'team': away_team,
            'opponent': home_team,
            'is_home': False,
            'difficulty': difficulty,
            'attacking_potential': away_attacking_potential
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
            'next_3_attacking_potential': next_3_fixtures['attacking_potential'].mean()
        }

# Add future fixture data to player data
if team_future_difficulty:
    history_df['next_3_avg_difficulty'] = history_df['team_id'].map({k: v['next_3_avg_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_min_difficulty'] = history_df['team_id'].map({k: v['next_3_min_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_max_difficulty'] = history_df['team_id'].map({k: v['next_3_max_difficulty'] for k, v in team_future_difficulty.items()})
    history_df['next_3_attacking_potential'] = history_df['team_id'].map({k: v['next_3_attacking_potential'] for k, v in team_future_difficulty.items()})
else:
    history_df['next_3_avg_difficulty'] = 3  # Medium difficulty as default
    history_df['next_3_min_difficulty'] = 3
    history_df['next_3_max_difficulty'] = 3
    history_df['next_3_attacking_potential'] = 1  # Neutral attacking potential

# ------------------ SAVE PROCESSED DATA ------------------
print("Saving processed data...")

# Fill NaN values with appropriate defaults
history_df = history_df.fillna({
    'avg_points_home': history_df['total_points'].mean(),
    'avg_points_away': history_df['total_points'].mean(),
    'home_goals_per_game': 0.2,  # Conservative defaults
    'away_goals_per_game': 0.15, 
    'opponent_team_strength': 1000,
    'opponent_defense_strength': 1000/1200,
    'minutes_consistency': 0,
    'opponent_strength_factor': 0.6,  # Middle value
    'team_recent_goals': 1.5,  # Average values
    'next_3_avg_difficulty': 3,
    'next_3_min_difficulty': 2,
    'next_3_max_difficulty': 4,
    'next_3_attacking_potential': 1,
    'goal_involvement_pct': 0,
    'recent_goal_inv_pct': 0,
    'shot_conversion': 0,
    'recent_shot_conversion': 0,
    'xG_per_90': 0,
    'xA_per_90': 0,
    'xGI_per_90': 0,
    'goals_minus_xG': 0,
    'assists_minus_xA': 0,
    'recent_xG': 0,
    'recent_xA': 0,
    'shots_per_90': 0,
    'big_chances_per_90': 0,
    'dribbles_per_90': 0,
    'recent_creativity': 0,
    'recent_threat': 0,
    'penalty_taker': 0,
    'goal_streak': 0
})

# Display the list of features we've created
feature_columns = [col for col in history_df.columns if col not in ['player_id', 'team_id', 'opponent_team', 'team_name']]
print("\nFeatures created:")
for col in feature_columns:
    print(f"- {col}")

# Save the processed data
history_df.to_csv('forward_processed.csv', index=False)
print(f"Enhanced processed data saved to 'forward_processed.csv' with {len(history_df)} rows and {len(feature_columns)} features")

# Save future fixtures for multi-gameweek planning
future_fixtures_df.to_csv('future_fixtures.csv', index=False)
print(f"Future fixture data saved to 'future_fixtures.csv' with {len(future_fixtures_df)} rows")

print("Feature engineering completed successfully!")