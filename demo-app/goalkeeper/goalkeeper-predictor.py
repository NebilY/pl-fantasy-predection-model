import requests
import json
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

print("Starting Goalkeeper Predictor...")

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

# Get current goalkeeper data
goalkeepers = [player for player in fpl_data['elements'] if player['element_type'] == 1]
goalkeepers_df = pd.DataFrame(goalkeepers)
print(f"Found {len(goalkeepers)} goalkeepers in current FPL data")

# Load historical data if available
try:
    history_df = pd.read_csv('goalkeeper_processed.csv')
    print(f"Loaded processed data with {len(history_df)} records")
except FileNotFoundError:
    history_df = pd.DataFrame()  # Empty DataFrame if file not found
    print("No processed goalkeeper data found. Will rely on current season data only.")

# Check if we have the feature list used for training
if os.path.exists('goalkeeper_model_features.txt'):
    with open('goalkeeper_model_features.txt', 'r') as f:
        training_features = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(training_features)} features from feature list file")
else:
    # Default to core features if file not found
    training_features = [
        'minutes_trend', 'rolling_points', 'value', 'avg_points_home', 'avg_points_away',
        'clean_sheet_points', 'save_points', 'goals_conceded_points', 'bonus_points', 
        'was_home', 'opponent_strength_factor', 'clean_sheet_rate'
    ]
    print("Feature list file not found, using default features")

# Create prediction data for each goalkeeper
prediction_data = []

for goalkeeper in goalkeepers:
    player_id = goalkeeper['id']
    team_id = goalkeeper['team']
    
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
                    print(f"Fetched recent history for {goalkeeper['web_name']} from API")
                    # For now, we'll use current season stats
                else:
                    print(f"No history found for {goalkeeper['web_name']} (ID: {player_id})")
            except Exception as e:
                print(f"Error fetching history for {goalkeeper['web_name']} (ID: {player_id}): {e}")
        
        # Get future fixtures for this player (for multi-gameweek planning)
        future_player_fixtures = [f for f in future_fixtures 
                                if f['team_h'] == team_id or f['team_a'] == team_id]
        
        # Calculate difficulty of future fixtures
        future_difficulty = []
        future_cs_potential = []
        future_save_potential = []
        for f in future_player_fixtures:
            gw = f['event']
            is_f_home = f['team_h'] == team_id
            opp_id = f['team_a'] if is_f_home else f['team_h']
            
            # Calculate fixture metrics based on opponent's attack vs team's defense
            if is_f_home:
                opp_attack = team_strength[opp_id]['attack_away']
                team_defense = team_strength[team_id]['defence_home']
            else:
                opp_attack = team_strength[opp_id]['attack_home']
                team_defense = team_strength[team_id]['defence_away']
            
            # Higher values are better for clean sheets (less likely to concede)
            cs_potential = team_defense / opp_attack if opp_attack > 0 else 1
            # Higher values mean more potential saves (facing strong attack)
            save_potential = opp_attack / 1200
            
            future_difficulty.append({
                'gw': gw,
                'difficulty': f.get('difficulty', 3),
                'cs_potential': cs_potential,
                'save_potential': save_potential,
                'is_home': is_f_home
            })
        
        # Calculate fixture metrics for next 3 gameweeks
        next_3_avg_difficulty = np.mean([f['difficulty'] for f in future_difficulty[:3]]) if future_difficulty else 3
        next_3_cs_potential = np.mean([f['cs_potential'] for f in future_difficulty[:3]]) if future_difficulty else 1
        next_3_save_potential = np.mean([f['save_potential'] for f in future_difficulty[:3]]) if future_difficulty else 0.5
        
        # Calculate current fixture metrics
        if is_home:
            opp_attack = team_strength[opponent_id]['attack_away']
            team_defense = team_strength[team_id]['defence_home']
        else:
            opp_attack = team_strength[opponent_id]['attack_home']
            team_defense = team_strength[team_id]['defence_away']
        
        cs_potential = team_defense / opp_attack if opp_attack > 0 else 1
        opponent_strength_factor = team_strength[opponent_id]['overall'] / 1200  # Normalize
        
        # Extract core metrics from player data
        minutes = goalkeeper.get('minutes', 0)
        total_points = goalkeeper.get('total_points', 0)
        cost = goalkeeper.get('now_cost', 0) / 10  # Convert to millions
        
        # Build basic prediction row with current season stats
        pred_row = {
            'player_id': player_id,
            'player_name': goalkeeper['web_name'],
            'team_id': team_id,
            'team_name': team_map.get(team_id, "Unknown"),
            'cost': cost,
            'total_points': total_points,
            'minutes': minutes,
            'selected_by_percent': float(goalkeeper.get('selected_by_percent', 0)),
            'transfers_in': goalkeeper.get('transfers_in', 0),
            'transfers_out': goalkeeper.get('transfers_out', 0),
            'was_home': is_home,
            'form': float(goalkeeper.get('form', 0)),
        }
        
        # Add current season stats for feature calculation
        clean_sheets = goalkeeper.get('clean_sheets', 0)
        saves = goalkeeper.get('saves', 0)
        goals_conceded = goalkeeper.get('goals_conceded', 0)
        bonus = goalkeeper.get('bonus', 0)
        
        # Calculate basic features from current season stats if history not available
        if player_history.empty:
            # Standard point components
            pred_row['clean_sheet_points'] = clean_sheets * 4
            pred_row['save_points'] = int(saves / 3)
            pred_row['goals_conceded_points'] = -1 * int(goals_conceded / 2)
            pred_row['bonus_points'] = bonus
            
            # Form metrics
            pred_row['minutes_trend'] = minutes / goalkeeper.get('starts', 1) if goalkeeper.get('starts', 0) > 0 else 0
            pred_row['rolling_points'] = float(goalkeeper.get('form', 0))
            pred_row['value'] = total_points / cost if cost > 0 else 0
            pred_row['avg_points_home'] = float(goalkeeper.get('points_per_game', 0))
            pred_row['avg_points_away'] = float(goalkeeper.get('points_per_game', 0))
            
            # GK specific metrics
            pred_row['clean_sheet_rate'] = clean_sheets / goalkeeper.get('starts', 1) if goalkeeper.get('starts', 0) > 0 else 0
            pred_row['opponent_strength_factor'] = opponent_strength_factor
            
            # Additional features if needed
            pred_row['saves_per_game'] = saves / goalkeeper.get('starts', 1) if goalkeeper.get('starts', 0) > 0 else 0
            pred_row['save_points_per_game'] = pred_row['save_points'] / goalkeeper.get('starts', 1) if goalkeeper.get('starts', 0) > 0 else 0
            pred_row['goals_conceded_per_game'] = goals_conceded / goalkeeper.get('starts', 1) if goalkeeper.get('starts', 0) > 0 else 0
        else:
            # Use historical data for features
            for feature in training_features:
                if feature in player_history.columns:
                    pred_row[feature] = player_history[feature].iloc[0]
        
        # Add fixture-specific features
        pred_row['opponent_strength_factor'] = opponent_strength_factor
        pred_row['next_3_avg_difficulty'] = next_3_avg_difficulty
        pred_row['next_3_cs_potential'] = next_3_cs_potential
        pred_row['next_3_save_potential'] = next_3_save_potential
        
        # Ensure all training features exist (with defaults if needed)
        for feature in training_features:
            if feature not in pred_row:
                pred_row[feature] = 0
        
        # Add row to prediction data
        prediction_data.append(pred_row)

if not prediction_data:
    print("No goalkeepers with sufficient data found. Check your data sources.")
    exit(1)

# Create DataFrame for predictions
pred_df = pd.DataFrame(prediction_data)
print(f"Created prediction data for {len(pred_df)} goalkeepers")

# Try to load the model
try:
    with open('goalkeeper_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Loaded goalkeeper model successfully")
except FileNotFoundError:
    print("Model file 'goalkeeper_model.pkl' not found. Please run the model training script first.")
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

# Show top goalkeeper picks
top_gks = pred_df.sort_values('predicted_points', ascending=False).head(10)
print("\nTop 10 Goalkeeper Picks for Upcoming Gameweek:")
print(top_gks[['player_name', 'team_name', 'cost', 'predicted_points', 'form']].to_string(index=False))

# Show best goalkeeper by price range
print("\nBest Goalkeeper by Price Range:")
price_ranges = [(0, 4.5), (4.5, 5.5), (5.5, 15.0)]
for low, high in price_ranges:
    price_range_df = pred_df[(pred_df['cost'] >= low) & (pred_df['cost'] < high)]
    if not price_range_df.empty:
        best_in_range = price_range_df.sort_values('predicted_points', ascending=False).iloc[0]
        print(f"£{low}-£{high}: {best_in_range['player_name']} ({best_in_range['team_name']}) - £{best_in_range['cost']}m, Predicted: {best_in_range['predicted_points']:.2f}")

# Select top 2 goalkeepers (starter + bench)
selected_gks = pred_df.sort_values('predicted_points', ascending=False).head(2)
print("\nRecommended Goalkeeper Pair:")
starter = selected_gks.iloc[0]
backup = selected_gks.iloc[1] if len(selected_gks) > 1 else None
print(f"Starter: {starter['player_name']} ({starter['team_name']}) - £{starter['cost']}m, Predicted: {starter['predicted_points']:.2f}")
if backup is not None:
    print(f"Backup: {backup['player_name']} ({backup['team_name']}) - £{backup['cost']}m, Predicted: {backup['predicted_points']:.2f}")
    print(f"Total Cost: £{starter['cost'] + backup['cost']:.1f}m")

# Save predictions to CSV for further analysis
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
pred_df.to_csv(f'goalkeeper_predictions_gw{current_gw}_{timestamp}.csv', index=False)
print(f"\nPredictions saved to 'goalkeeper_predictions_gw{current_gw}_{timestamp}.csv'")

# Long-term planning insights
print("\nLong-term Planning Insights (Next 3 Gameweeks):")
if 'next_3_cs_potential' in pred_df.columns:
    long_term_picks = pred_df.sort_values('next_3_cs_potential', ascending=False).head(5)
    print("Goalkeepers with Best Clean Sheet Potential:")
    print(long_term_picks[['player_name', 'team_name', 'next_3_cs_potential', 'predicted_points']].to_string(index=False))

# Rotation pairing analysis
print("\nGoalkeeper Rotation Pairs:")
# Look at pairs of cheap goalkeepers with complementary fixtures
budget_gks = pred_df[pred_df['cost'] <= 4.8].sort_values('predicted_points', ascending=False).head(5)

if len(budget_gks) >= 2:
    rotation_pairs = []
    
    for i, gk1 in budget_gks.iterrows():
        for j, gk2 in budget_gks.iterrows():
            if i < j and gk1['team_id'] != gk2['team_id']:  # Different teams
                # Calculate combined metrics
                combined_cost = gk1['cost'] + gk2['cost']
                combined_cs_potential = (gk1['next_3_cs_potential'] + gk2['next_3_cs_potential']) / 2
                fixture_complementarity = 1  # Placeholder for fixture complementarity score
                
                rotation_pairs.append({
                    'gk1_name': gk1['player_name'],
                    'gk1_team': gk1['team_name'],
                    'gk2_name': gk2['player_name'],
                    'gk2_team': gk2['team_name'],
                    'combined_cost': combined_cost,
                    'combined_cs_potential': combined_cs_potential,
                    'fixture_complementarity': fixture_complementarity
                })
    
    # Show top rotation pairs
    if rotation_pairs:
        rotation_df = pd.DataFrame(rotation_pairs)
        top_pairs = rotation_df.sort_values(['combined_cs_potential', 'fixture_complementarity'], ascending=[False, False]).head(3)
        
        for _, pair in top_pairs.iterrows():
            print(f"{pair['gk1_name']} ({pair['gk1_team']}) + {pair['gk2_name']} ({pair['gk2_team']})")
            print(f"  Combined cost: £{pair['combined_cost']:.1f}m, Clean sheet potential: {pair['combined_cs_potential']:.2f}")

print("\nPrediction completed successfully!")