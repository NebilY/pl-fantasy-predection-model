import requests
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

print("Starting Midfielder Predictor...")

# First, fetch the FPL data
try:
    response = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    fpl_data = json.loads(response.text)
    print("Fetched current FPL data successfully")
    
    # Get fixture data
    fixtures_response = requests.get('https://fantasy.premierleague.com/api/fixtures/')
    fixtures = json.loads(fixtures_response.text)
    print(f"Fetched {len(fixtures)} fixtures")
except Exception as e:
    print(f"Error fetching FPL data: {e}")
    exit(1)

# Team mapping for reference
team_map = {team['id']: team['name'] for team in fpl_data['teams']}
team_strength = {team['id']: {
    'overall': team['strength'],
    'attack_home': team.get('strength_attack_home', 1000),
    'attack_away': team.get('strength_attack_away', 1000),
    'defence_home': team.get('strength_defence_home', 1000),
    'defence_away': team.get('strength_defence_away', 1000)
} for team in fpl_data['teams']}

# Get current gameweek
events = fpl_data['events']
current_event = next((event for event in events if event['is_current']), None)
if current_event:
    current_gw = current_event['id']
    print(f"Current gameweek: {current_gw}")
else:
    # Find the next gameweek if no current one
    next_event = next((event for event in events if event['is_next']), None)
    if next_event:
        current_gw = next_event['id']
        print(f"Next gameweek: {current_gw}")
    else:
        # Fallback
        current_gw = min([f['event'] for f in fixtures if f['event'] is not None])
        print(f"Estimated next gameweek: {current_gw}")

# Filter to get near-term fixtures
upcoming_fixtures = [f for f in fixtures if f['event'] == current_gw]
print(f"Found {len(upcoming_fixtures)} fixtures for gameweek {current_gw}")

# Get fixtures for future gameweeks (for multi-week planning)
future_gws = [current_gw + i for i in range(4)]  # Current GW plus next 3
future_fixtures = [f for f in fixtures if f['event'] in future_gws]
print(f"Found {len(future_fixtures)} fixtures for the next 4 gameweeks")

# Get current midfielder data
midfielders = [player for player in fpl_data['elements'] if player['element_type'] == 3]
midfielders_df = pd.DataFrame(midfielders)
print(f"Found {len(midfielders)} midfielders in current FPL data")

# Load historical data if available
try:
    history_df = pd.read_csv('midfielder_processed.csv')
    print(f"Loaded processed data with {len(history_df)} records")
except FileNotFoundError:
    history_df = pd.DataFrame()  # Empty DataFrame if file not found
    print("No processed midfielder data found. Will rely on current season data only.")

# Check if we have the feature list used for training
if os.path.exists('midfielder_model_features.txt'):
    with open('midfielder_model_features.txt', 'r') as f:
        training_features = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(training_features)} features from feature list file")
else:
    # Default to core features if file not found
    training_features = [
        'minutes_trend', 'rolling_points', 'value', 'avg_points_home', 'avg_points_away',
        'attacking_returns', 'opponent_strength_factor', 'was_home', 'recent_goals', 
        'recent_assists', 'bonus_trend'
    ]
    print("Feature list file not found, using default features")

# Create prediction data for each midfielder
prediction_data = []

for midfielder in midfielders:
    player_id = midfielder['id']
    team_id = midfielder['team']
    
    # Find this player's fixture for current gameweek
    player_fixture = next((f for f in upcoming_fixtures 
                       if f['team_h'] == team_id or f['team_a'] == team_id), None)
    
    if player_fixture:
        is_home = player_fixture['team_h'] == team_id
        opponent_id = player_fixture['team_a'] if is_home else player_fixture['team_h']
        
        # Get recent data for features
        player_history = pd.DataFrame()
        if not history_df.empty:
            player_history = history_df[history_df['player_id'] == player_id]
            if 'round' in player_history.columns:
                player_history = player_history.sort_values('round', ascending=False)
        
        # If no historical data, try to get data from element-summary API
        if player_history.empty:
            try:
                player_response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/')
                player_data = json.loads(player_response.text)
                
                if 'history' in player_data and player_data['history']:
                    print(f"Fetched recent history for {midfielder['web_name']} from API")
                    # Convert to DataFrame if needed
                    # For now, we'll use current season stats
                else:
                    print(f"No history found for {midfielder['web_name']} (ID: {player_id})")
            except Exception as e:
                print(f"Error fetching history for {midfielder['web_name']} (ID: {player_id}): {e}")
        
        # Get future fixtures for this player (for multi-gameweek planning)
        future_player_fixtures = [f for f in future_fixtures 
                                if f['team_h'] == team_id or f['team_a'] == team_id]
        
        # Calculate difficulty of future fixtures
        future_difficulty = []
        future_attacking_potential = []
        for f in future_player_fixtures:
            gw = f['event']
            is_f_home = f['team_h'] == team_id
            opp_id = f['team_a'] if is_f_home else f['team_h']
            
            # Calculate fixture difficulty based on opponent's defense vs team's attack
            if is_f_home:
                opp_defense = team_strength[opp_id]['defence_away']
                team_attack = team_strength[team_id]['attack_home']
            else:
                opp_defense = team_strength[opp_id]['defence_home']
                team_attack = team_strength[team_id]['attack_away']
            
            # Higher values are better for midfielders (easier to score against)
            attacking_potential = team_attack / opp_defense if opp_defense > 0 else 1
            
            future_difficulty.append({
                'gw': gw,
                'difficulty': f.get('difficulty', 3),
                'attacking_potential': attacking_potential,
                'is_home': is_f_home
            })
        
        # Calculate fixture metrics for next 3 gameweeks
        next_3_avg_difficulty = np.mean([f['difficulty'] for f in future_difficulty[:3]]) if future_difficulty else 3
        next_3_attacking_potential = np.mean([f['attacking_potential'] for f in future_difficulty[:3]]) if future_difficulty else 1
        
        # Calculate current fixture metrics
        if is_home:
            opp_defense = team_strength[opponent_id]['defence_away']
            team_attack = team_strength[team_id]['attack_home']
        else:
            opp_defense = team_strength[opponent_id]['defence_home']
            team_attack = team_strength[team_id]['attack_away']
        
        attacking_potential = team_attack / opp_defense if opp_defense > 0 else 1
        opponent_strength_factor = team_strength[opponent_id]['overall'] / 1200  # Normalize
        
        # Extract core metrics from player data
        minutes = midfielder.get('minutes', 0)
        total_points = midfielder.get('total_points', 0)
        cost = midfielder.get('now_cost', 0) / 10  # Convert to millions
        
        # Build basic prediction row with current season stats
        pred_row = {
            'player_id': player_id,
            'player_name': midfielder['web_name'],
            'team_id': team_id,
            'team_name': team_map.get(team_id, "Unknown"),
            'cost': cost,
            'total_points': total_points,
            'minutes': minutes,
            'selected_by_percent': float(midfielder.get('selected_by_percent', 0)),
            'transfers_in': midfielder.get('transfers_in', 0),
            'transfers_out': midfielder.get('transfers_out', 0),
            'was_home': is_home,
            'form': float(midfielder.get('form', 0)),
        }
        
        # Add current season stats for feature calculation
        goals = midfielder.get('goals_scored', 0)
        assists = midfielder.get('assists', 0)
        bonus = midfielder.get('bonus', 0)
        
        # Calculate basic features from current season stats if history not available
        if player_history.empty:
            # Estimate recent form based on current form
            pred_row['minutes_trend'] = minutes / midfielder.get('starts', 1) if midfielder.get('starts', 0) > 0 else 0
            pred_row['rolling_points'] = float(midfielder.get('form', 0))
            pred_row['value'] = total_points / cost if cost > 0 else 0
            pred_row['avg_points_home'] = float(midfielder.get('points_per_game', 0))
            pred_row['avg_points_away'] = float(midfielder.get('points_per_game', 0))
            pred_row['attacking_returns'] = goals + assists
            pred_row['opponent_strength_factor'] = opponent_strength_factor
            pred_row['recent_goals'] = goals
            pred_row['recent_assists'] = assists
            pred_row['bonus_trend'] = bonus
            
            # Add additional features if needed in the model
            pred_row['goals_per_90'] = 90 * goals / minutes if minutes > 0 else 0
            pred_row['assists_per_90'] = 90 * assists / minutes if minutes > 0 else 0
            pred_row['goal_involvement_per_90'] = pred_row['goals_per_90'] + pred_row['assists_per_90']
            
            # Try to get creativity and threat if available
            pred_row['recent_creativity'] = float(midfielder.get('creativity', 0))
            pred_row['recent_threat'] = float(midfielder.get('threat', 0))
        else:
            # Use historical data for features
            for feature in training_features:
                if feature in player_history.columns:
                    pred_row[feature] = player_history[feature].iloc[0]
        
        # Add fixture-specific features
        pred_row['opponent_strength_factor'] = opponent_strength_factor
        pred_row['next_3_avg_difficulty'] = next_3_avg_difficulty
        pred_row['next_3_attacking_potential'] = next_3_attacking_potential
        
        # Add row to prediction data
        prediction_data.append(pred_row)

if not prediction_data:
    print("No midfielders with sufficient data found. Check your data sources.")
    exit(1)

# Create DataFrame for predictions
pred_df = pd.DataFrame(prediction_data)
print(f"Created prediction data for {len(pred_df)} midfielders")

# Ensure all required features exist
for feature in training_features:
    if feature not in pred_df.columns:
        print(f"Warning: Feature '{feature}' not found in prediction data. Adding with default values.")
        pred_df[feature] = 0

# Try to load the model
try:
    with open('midfielder_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Loaded midfielder model successfully")
except FileNotFoundError:
    print("Model file 'midfielder_model.pkl' not found. Please run the model training script first.")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Handle potential pipeline vs. direct model
is_pipeline = hasattr(model, 'steps')

# Make predictions
try:
    X_pred = pred_df[training_features]
    
    # Make prediction
    if is_pipeline:
        pred_df['predicted_points'] = model.predict(X_pred)
    else:
        # If direct model, we need to handle imputation ourselves
        X_imputed = X_pred.fillna(X_pred.mean())
        pred_df['predicted_points'] = model.predict(X_imputed)
        
    print("Generated predictions successfully")
except Exception as e:
    print(f"Error making predictions: {e}")
    print("Feature shapes:", {f: X_pred[f].shape for f in training_features})
    exit(1)

# Show top midfielder picks
top_midfielders = pred_df.sort_values('predicted_points', ascending=False).head(10)
print("\nTop 10 Midfielder Picks for Upcoming Gameweek:")
print(top_midfielders[['player_name', 'team_name', 'cost', 'predicted_points', 'form']].to_string(index=False))

# Show best midfielder by price range
print("\nBest Midfielder by Price Range:")
price_ranges = [(0, 5.0), (5.0, 6.5), (6.5, 8.0), (8.0, 15.0)]
for low, high in price_ranges:
    price_range_df = pred_df[(pred_df['cost'] >= low) & (pred_df['cost'] < high)]
    if not price_range_df.empty:
        best_in_range = price_range_df.sort_values('predicted_points', ascending=False).iloc[0]
        print(f"£{low}-£{high}: {best_in_range['player_name']} ({best_in_range['team_name']}) - £{best_in_range['cost']}m, Predicted: {best_in_range['predicted_points']:.2f}")

# Select top 5 midfielders (for a 3-5-2 formation, assuming we need 5)
selected_midfielders = pred_df.sort_values('predicted_points', ascending=False).head(5)
print("\nSelected Midfielders for 3-5-2 formation:")
print(selected_midfielders[['player_name', 'team_name', 'cost', 'predicted_points']].to_string(index=False))
print(f"Total Cost: £{selected_midfielders['cost'].sum():.1f}m")

# Save predictions to CSV for further analysis
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
pred_df.to_csv(f'midfielder_predictions_gw{current_gw}_{timestamp}.csv', index=False)
print(f"\nPredictions saved to 'midfielder_predictions_gw{current_gw}_{timestamp}.csv'")

# Long-term planning insights
print("\nLong-term Planning Insights (Next 3 Gameweeks):")
if 'next_3_attacking_potential' in pred_df.columns:
    long_term_picks = pred_df.sort_values('next_3_attacking_potential', ascending=False).head(5)
    print("Midfielders with Best Upcoming Attacking Fixtures:")
    print(long_term_picks[['player_name', 'team_name', 'next_3_attacking_potential', 'predicted_points']].to_string(index=False))
else:
    long_term_picks = pred_df.sort_values('next_3_avg_difficulty').head(5)
    print("Midfielders with Easiest Upcoming Fixtures:")
    print(long_term_picks[['player_name', 'team_name', 'next_3_avg_difficulty', 'predicted_points']].to_string(index=False))

# Team diversity check (avoid overloading from one team)
team_counts = selected_midfielders['team_id'].value_counts()
if team_counts.max() > 2:  # Allow up to 2 midfielders from same team
    print("\nWarning: Multiple midfielders selected from the same team:")
    for team_id, count in team_counts.items():
        if count > 2:
            team_name = team_map.get(team_id, f"Team ID {team_id}")
            print(f"  {count} midfielders from {team_name}")
    
    # Alternative selection with team diversity
    diverse_selection = []
    teams_selected = {}
    
    for _, midfielder in pred_df.sort_values('predicted_points', ascending=False).iterrows():
        team_id = midfielder['team_id']
        if teams_selected.get(team_id, 0) < 2 and len(diverse_selection) < 5:
            diverse_selection.append(midfielder)
            teams_selected[team_id] = teams_selected.get(team_id, 0) + 1
    
    if len(diverse_selection) == 5:
        diverse_df = pd.DataFrame(diverse_selection)
        print("\nAlternative Selection (with team diversity - max 2 per team):")
        print(diverse_df[['player_name', 'team_name', 'cost', 'predicted_points']].to_string(index=False))
        print(f"Total Cost: £{diverse_df['cost'].sum():.1f}m")

# Premium/Budget pairing suggestions
print("\nSuggested Premium & Budget Midfielder Pairings:")
premium_mids = pred_df[pred_df['cost'] >= 8.0].sort_values('predicted_points', ascending=False).head(3)
budget_mids = pred_df[pred_df['cost'] < 6.5].sort_values('predicted_points', ascending=False).head(3)

if not premium_mids.empty and not budget_mids.empty:
    for _, premium in premium_mids.iterrows():
        for _, budget in budget_mids.iterrows():
            if premium['team_id'] != budget['team_id']:  # Avoid same team
                print(f"{premium['player_name']} (£{premium['cost']}m) + {budget['player_name']} (£{budget['cost']}m)")
                print(f"  Combined cost: £{premium['cost'] + budget['cost']:.1f}m, Predicted points: {premium['predicted_points'] + budget['predicted_points']:.2f}")
                break  # Just show one budget option per premium

print("\nPrediction completed successfully!")