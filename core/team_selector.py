# core/team_selector.py
import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from config.settings import (
    MAX_BUDGET, 
    MAX_PLAYERS_PER_TEAM, 
    SQUAD_SIZE, 
    VALID_FORMATIONS
)

class TeamSelector:
    """
    Optimizes Fantasy Premier League team selection 
    using linear programming and intelligent constraints.
    """
    
    def __init__(self, predictions: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize TeamSelector with position-specific predictions.
        
        Args:
            predictions (Dict): Predictions for each player position
        """
        self.predictions = predictions
        self.all_players = self._combine_predictions()
    
    def _combine_predictions(self) -> pd.DataFrame:
        """
        Combine predictions from all positions into a single DataFrame.
        
        Returns:
            pd.DataFrame: Consolidated player predictions
        """
        players_data = []
        
        for position, position_predictions in self.predictions.items():
            for player in position_predictions:
                player_entry = {
                    'player_id': player['player_id'],
                    'player_name': player['player_name'],
                    'team_name': player['team_name'],
                    'position': position,
                    'predicted_points': player['predicted_points'],
                    'price': player['price']
                }
                players_data.append(player_entry)
        
        return pd.DataFrame(players_data)
    
    def optimize_squad(self) -> Dict[str, Any]:
        """
        Optimize 15-player squad using linear programming.
        
        Returns:
            Dict: Optimized squad details
        """
        # Create the optimization problem
        prob = pulp.LpProblem("FPL_Team_Selection", pulp.LpMaximize)
        
        # Decision variables: binary for each player
        player_vars = {
            player['player_id']: pulp.LpVariable(
                f"player_{player['player_id']}", 
                cat='Binary'
            ) 
            for _, player in self.all_players.iterrows()
        }
        
        # Objective: Maximize total predicted points
        prob += pulp.lpSum([
            player_vars[player['player_id']] * player['predicted_points'] 
            for _, player in self.all_players.iterrows()
        ])
        
        # Constraints
        
        # Budget constraint
        prob += pulp.lpSum([
            player_vars[player['player_id']] * player['price'] 
            for _, player in self.all_players.iterrows()
        ]) <= MAX_BUDGET
        
        # Position count constraints
        for position, count in SQUAD_SIZE.items():
            if position != 'TOTAL':
                position_players = self.all_players[
                    self.all_players['position'] == position
                ]
                prob += pulp.lpSum([
                    player_vars[player['player_id']] 
                    for _, player in position_players.iterrows()
                ]) == count
        
        # Team representation constraint
        team_counts = {}
        for _, player in self.all_players.iterrows():
            team = player['team_name']
            if team not in team_counts:
                team_counts[team] = 0
            team_counts[team] += 1
        
        for team, team_total in team_counts.items():
            team_players = self.all_players[
                self.all_players['team_name'] == team
            ]
            prob += pulp.lpSum([
                player_vars[player['player_id']] 
                for _, player in team_players.iterrows()
            ]) <= MAX_PLAYERS_PER_TEAM
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract selected players
        selected_players = [
            player for _, player in self.all_players.iterrows() 
            if player_vars[player['player_id']].value() > 0.5
        ]
        
        return {
            'squad': selected_players,
            'total_points': pulp.value(prob.objective),
            'total_cost': sum(player['price'] for player in selected_players)
        }
    
    def select_starting_xi(self, squad: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best starting XI and formation.
        
        Args:
            squad (List): Full 15-player squad
        
        Returns:
            Dict: Starting XI and formation details
        """
        # Group squad by position
        squad_by_position = {
            'goalkeeper': [p for p in squad if p['position'] == 'goalkeeper'],
            'defender': [p for p in squad if p['position'] == 'defender'],
            'midfielder': [p for p in squad if p['position'] == 'midfielder'],
            'forward': [p for p in squad if p['position'] == 'forward']
        }
        
        # Sort each position by predicted points
        for position in squad_by_position:
            squad_by_position[position].sort(
                key=lambda x: x['predicted_points'], 
                reverse=True
            )
        
        # Find optimal formation
        best_formation = None
        best_formation_points = 0
        
        for formation, (def_count, mid_count, fwd_count) in VALID_FORMATIONS.items():
            # Check if we have enough players in each position
            if (len(squad_by_position['defender']) >= def_count and
                len(squad_by_position['midfielder']) >= mid_count and
                len(squad_by_position['forward']) >= fwd_count):
                
                # Calculate formation points
                formation_points = (
                    squad_by_position['goalkeeper'][0]['predicted_points'] +
                    sum(p['predicted_points'] for p in squad_by_position['defender'][:def_count]) +
                    sum(p['predicted_points'] for p in squad_by_position['midfielder'][:mid_count]) +
                    sum(p['predicted_points'] for p in squad_by_position['forward'][:fwd_count])
                )
                
                if formation_points > best_formation_points:
                    best_formation = formation
                    best_formation_points = formation_points
        
        # If no formation found, default to first valid formation
        if not best_formation:
            best_formation = list(VALID_FORMATIONS.keys())[0]
        
        # Unpack formation
        def_count, mid_count, fwd_count = VALID_FORMATIONS[best_formation]
        
        # Build starting XI
        starting_xi = (
            [squad_by_position['goalkeeper'][0]] +
            squad_by_position['defender'][:def_count] +
            squad_by_position['midfielder'][:mid_count] +
            squad_by_position['forward'][:fwd_count]
        )
        
        # Build bench
        bench = (
            [squad_by_position['goalkeeper'][1]] +
            squad_by_position['defender'][def_count:] +
            squad_by_position['midfielder'][mid_count:] +
            squad_by_position['forward'][fwd_count:]
        )
        
        return {
            'formation': best_formation,
            'starting_xi': starting_xi,
            'bench': bench,
            'total_starting_points': sum(p['predicted_points'] for p in starting_xi)
        }
    
    def select_captain(self, starting_xi: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select captain and vice-captain from starting XI.
        
        Args:
            starting_xi (List): Starting XI players
        
        Returns:
            Dict: Captain and vice-captain selections
        """
        # Sort starting XI by predicted points
        sorted_players = sorted(
            starting_xi, 
            key=lambda x: x['predicted_points'], 
            reverse=True
        )
        
        # Select top two players
        captain = sorted_players[0]
        vice_captain = sorted_players[1]
        
        return {
            'captain': captain,
            'vice_captain': vice_captain
        }
    
    def optimize_team(self) -> Dict[str, Any]:
        """
        Complete team optimization process.
        
        Returns:
            Dict: Comprehensive team optimization results
        """
        # Optimize squad
        squad_result = self.optimize_squad()
        
        # Select starting XI and formation
        lineup_result = self.select_starting_xi(squad_result['squad'])
        
        # Select captain
        captain_result = self.select_captain(lineup_result['starting_xi'])
        
        # Combine results
        return {
            'full_squad': squad_result['squad'],
            'formation': lineup_result['formation'],
            'starting_xi': lineup_result['starting_xi'],
            'bench': lineup_result['bench'],
            'captain': captain_result['captain'],
            'vice_captain': captain_result['vice_captain'],
            'total_points': lineup_result['total_starting_points'],
            'total_cost': squad_result['total_cost']
        }

# Example usage
if __name__ == "__main__":
    # Simulated predictions (in a real scenario, this would come from PredictionManager)
    sample_predictions = {
        'goalkeeper': [
            {'player_id': 'gk1', 'player_name': 'Keeper One', 'team_name': 'Team A', 'predicted_points': 6.5, 'price': 5.5, 'position': 'goalkeeper'},
            # More goalkeeper predictions...
        ],
        'defender': [
            {'player_id': 'def1', 'player_name': 'Defender One', 'team_name': 'Team B', 'predicted_points': 7.0, 'price': 6.0, 'position': 'defender'},
            # More defender predictions...
        ],
        # Similar entries for midfielders and forwards
    }
    
    team_selector = TeamSelector(sample_predictions)
    optimized_team = team_selector.optimize_team()
    
    print("Optimized Team Details:")
    print(f"Formation: {optimized_team['formation']}")
    print("\nStarting XI:")
    for player in optimized_team['starting_xi']:
        print(f"{player['position'].capitalize()}: {player['player_name']} - {player['predicted_points']} points")
    
    print(f"\nCaptain: {optimized_team['captain']['player_name']}")
    print(f"Vice-Captain: {optimized_team['vice_captain']['player_name']}")
    print(f"Total Points: {optimized_team['total_points']}")
    print(f"Total Cost: £{optimized_team['total_cost']}m")