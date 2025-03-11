import requests
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

print("Starting Enhanced Defender Predictor...")

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

# Get current defender data
defenders = [player for player in fpl_data['elements'] if player['element_type'] == 2]
defenders_df = pd.DataFrame(defenders)
print(f"Found {len(defenders)} defenders in current FPL data")

# Load historical data
try:
    history_df = pd.read_csv('defender_processed.csv')
    print(f"Loaded processed data with {len(history_df)} records")
except FileNotFoundError:
    print("Processed defender data not found. Please run the feature engineering script first.")
    exit(1)

# Check if we have the feature list used for training
if os.path.exists('defender_model_features.txt'):
    with open('defender_model_features.txt', 'r') as f:
        training_features = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(training_features)} features from feature list file")
else:
    # Default to basic features if file not found
    training_features = [
        'minutes_trend', 'rolling_points', 'value', 'avg_points_home', 'avg_points_away',
        'attacking_returns', 'clean_sheet_points', 'was_home', 'opponent_strength_factor'
    ]
    print("Feature list file not found, using default features")

# Create prediction data for each defender
prediction_data = []

for defender in defenders:
    player_id = defender['id']
    team_id = defender['team']
    
    # Find this player's fixture for current gameweek
    player_fixture = next((f for f in upcoming_fixtures 
                       if f['team_h'] == team_id or f['team_a'] == team_id), None)
    
    if player_fixture:
        is_home = player_fixture['team_h'] == team_id
        opponent_id = player_fixture['team_a'] if is_home else player_fixture['team_h']
        
        # Get recent data for features
        player_history = history_df[history_df['player_id'] == player_id]
        if 'round' in player_history.columns:
            player_history = player_history.sort_values('round', ascending=False)
        
        # If no historical data, try to get data from element-summary API
        if player_history.empty:
            try:
                player_response = requests.get(f'https://fantasy.premierleague.com/api/element-summary/{player_id}/')
                player_data = json.loads(player_response.text)
                
                if 'history' in player_data and player_data['history']:
                    print(f"Fetched recent history for {defender['web_name']} from API")
                else:
                    print(f"No history found for {defender['web_name']} (ID: {player_id})")
                    # Skip to next defender if no data
                    continue
            except Exception as e:
                print(f"Error fetching history for {defender['web_name']} (ID: {player_id}): {e}")
                continue
        
        # Get future fixtures for this player (for multi-gameweek planning)
        future_player_fixtures = [f for f in future_fixtures 
                                if f['team_h'] == team_id or f['team_a'] == team_id]
        
        # Calculate difficulty of future fixtures
        future_difficulty = []
        for f in future_player_fixtures:
            gw = f['event']
            is_f_home = f['team_h'] == team_id
            opp_id = f['team_a'] if is_f_home else f['team_h']
            
            # Calculate fixture difficulty based on opponent's attack strength vs team's defense
            if is_f_home:
                opp_attack = team_strength[opp_id]['attack_away']
                team_defense = team_strength[team_id]['defence_home']
            else:
                opp_attack = team_strength[opp_id]['attack_home']
                team_defense = team_strength[team_id]['defence_away']
            
            # Lower values are better for defenders (less likely to concede)
            fixture_diff = opp_attack / team_defense if team_defense > 0 else 1
            
            future_difficulty.append({
                'gw': gw,
                'difficulty': fixture_diff,
                'is_home': is_f_home
            })
        
        # Calculate fixture difficulty metrics for next 3 gameweeks
        next_3_avg_difficulty = np.mean([f['difficulty'] for f in future_difficulty[:3]]) if future_difficulty else 1
        next_3_min_difficulty = min([f['difficulty'] for f in future_difficulty[:3]]) if future_difficulty else 1
        next_3_max_difficulty = max([f['difficulty'] for f in future_difficulty[:3]]) if future_difficulty else 1
        
        # Calculate opponent strength metrics
        opponent_attack = team_strength[opponent_id]['attack_away'] if is_home else team_strength[opponent_id]['attack_home']
        team_defense = team_strength[team_id]['defence_home'] if is_home else team_strength[team_id]['defence_away']
        fixture_difficulty = opponent_attack / team_defense if team_defense > 0 else 1
        opponent_strength_factor = team_strength[opponent_id]['overall'] / 1200  # Normalize
        
        # Extract core metrics from player data
        minutes = defender.get('minutes', 0)
        total_points = defender.get('total_points', 0)
        cost = defender.get('now_cost', 0) / 10  # Convert to millions
        
        # Build prediction row
        pred_row = {
            'player_id': player_id,
            'player_name': defender['web_name'],
            'team_id': team_id,
            'team_name': team_map.get(team_id, "Unknown"),
            'cost': cost,
            'total_points': total_points,
            'minutes': minutes,
            'selected_by_percent': float(defender.get('selected_by_percent', 0)),
            'transfers_in': defender.get('transfers_in', 0),
            'transfers_out': defender.get('transfers_out', 0),
            'was_home': is_home,
            'opponent_team': opponent_id,
        }
        
        # Process historical data if available
        if not player_history.empty:
            # Add all available metrics from player history
            for feature in training_features:
                if feature in player_history.columns:
                    pred_row[feature] = player_history[feature].iloc[0]
        
        # Add values for fixture-specific features that aren't in history
        pred_row['opponent_strength_factor'] = opponent_strength_factor
        pred_row['fixture_difficulty'] = fixture_difficulty
        pred_row['next_3_avg_difficulty'] = next_3_avg_difficulty
        pred_row['next_3_min_difficulty'] = next_3_min_difficulty
        pred_row['next_3_max_difficulty'] = next_3_max_difficulty
        
        # Add row to prediction data
        prediction_data.append(pred_row)

if not prediction_data:
    print("No defenders with sufficient data found. Check your data sources.")
    exit(1)

# Create DataFrame for predictions
pred_df = pd.DataFrame(prediction_data)
print(f"Created prediction data for {len(pred_df)} defenders")

# Ensure all required features exist
for feature in training_features:
    if feature not in pred_df.columns:
        print(f"Warning: Feature '{feature}' not found in prediction data. Adding with default values.")
        pred_df[feature] = 0

# Try to load the model
try:
    model_file = 'defender_model_enhanced.pkl' if os.path.exists('defender_model_enhanced.pkl') else 'defender_model.pkl'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded defender model from {model_file}")
except FileNotFoundError:
    print("Model file not found. Please run the model training script first.")
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

# Show top defender picks
top_defenders = pred_df.sort_values('predicted_points', ascending=False).head(10)
print("\nTop 10 Defender Picks for Upcoming Gameweek:")
print(top_defenders[['player_name', 'team_name', 'cost', 'predicted_points', 'next_3_avg_difficulty']].to_string(index=False))

# Show best defender by price range
print("\nBest Defender by Price Range:")
price_ranges = [(0, 4.5), (4.5, 5.5), (5.5, 7.0), (7.0, 15.0)]
for low, high in price_ranges:
    price_range_df = pred_df[(pred_df['cost'] >= low) & (pred_df['cost'] < high)]
    if not price_range_df.empty:
        best_in_range = price_range_df.sort_values('predicted_points', ascending=False).iloc[0]
        print(f"£{low}-£{high}: {best_in_range['player_name']} ({best_in_range['team_name']}) - £{best_in_range['cost']}m, Predicted: {best_in_range['predicted_points']:.2f}")

# Select top 3 defenders (for a 3-4-3 formation)
selected_defenders = pred_df.sort_values('predicted_points', ascending=False).head(3)
print("\nSelected Defenders for 3-4-3 formation:")
print(selected_defenders[['player_name', 'team_name', 'cost', 'predicted_points']].to_string(index=False))
print(f"Total Cost: £{selected_defenders['cost'].sum():.1f}m")

# Save predictions to CSV for further analysis
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
pred_df.to_csv(f'defender_predictions_gw{current_gw}_{timestamp}.csv', index=False)
print(f"\nPredictions saved to 'defender_predictions_gw{current_gw}_{timestamp}.csv'")

# Long-term planning insights
print("\nLong-term Planning Insights (Next 3 Gameweeks):")
long_term_picks = pred_df.sort_values('next_3_avg_difficulty').head(5)
print("Defenders with Best Upcoming Fixtures:")
print(long_term_picks[['player_name', 'team_name', 'next_3_avg_difficulty', 'predicted_points']].to_string(index=False))

# Team diversity check (avoid overloading from one team)
team_counts = selected_defenders['team_id'].value_counts()
if team_counts.max() > 1:
    print("\nWarning: Multiple defenders selected from the same team:")
    for team_id, count in team_counts.items():
        if count > 1:
            team_name = team_map.get(team_id, f"Team ID {team_id}")
            print(f"  {count} defenders from {team_name}")
    
    # Alternative selection with team diversity
    diverse_selection = []
    teams_selected = set()
    
    for _, defender in pred_df.sort_values('predicted_points', ascending=False).iterrows():
        if defender['team_id'] not in teams_selected and len(diverse_selection) < 3:
            diverse_selection.append(defender)
            teams_selected.add(defender['team_id'])
    
    if len(diverse_selection) == 3:
        diverse_df = pd.DataFrame(diverse_selection)
        print("\nAlternative Selection (with team diversity):")
        print(diverse_df[['player_name', 'team_name', 'cost', 'predicted_points']].to_string(index=False))
        print(f"Total Cost: £{diverse_df['cost'].sum():.1f}m")

# Rotation potential - find defender pairs that have complementary fixtures
print("\nDefender Rotation Pairs (Budget Options):")
budget_defenders = pred_df[pred_df['cost'] < 5.0].sort_values('predicted_points', ascending=False).head(10)

if len(budget_defenders) >= 2:
    rotation_pairs = []
    
    for i, def1 in budget_defenders.iterrows():
        for j, def2 in budget_defenders.iterrows():
            if i < j:  # avoid duplicate pairs
                # Check if they have complementary fixtures over next few weeks
                complementary_score = 0
                
                # Simple heuristic: different teams increase complementary score
                if def1['team_id'] != def2['team_id']:
                    complementary_score += 2
                
                # Total predicted points
                combined_points = def1['predicted_points'] + def2['predicted_points']
                
                rotation_pairs.append({
                    'defender1': def1['player_name'],
                    'team1': def1['team_name'],
                    'defender2': def2['player_name'],
                    'team2': def2['team_name'],
                    'combined_cost': def1['cost'] + def2['cost'],
                    'combined_points': combined_points,
                    'complementary_score': complementary_score
                })
    
    # Show top rotation pairs
    rotation_df = pd.DataFrame(rotation_pairs)
    top_pairs = rotation_df.sort_values(['complementary_score', 'combined_points'], ascending=[False, False]).head(3)
    
    for _, pair in top_pairs.iterrows():
        print(f"{pair['defender1']} ({pair['team1']}) + {pair['defender2']} ({pair['team2']})")
        print(f"  Combined cost: £{pair['combined_cost']:.1f}m, Predicted points: {pair['combined_points']:.2f}")

print("\nPrediction completed successfully!")