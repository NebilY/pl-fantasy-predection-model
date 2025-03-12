# core/transfer_planner.py
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from config.settings import MAX_FREE_TRANSFERS, MAX_TRANSFERS_STORED

class TransferPlanner:
    """
    Manages transfer strategies for Fantasy Premier League teams,
    considering long-term performance and point potential.
    """
    
    def __init__(self, 
                 current_team: Dict[str, Any], 
                 predictions: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize TransferPlanner with current team and predictions.
        
        Args:
            current_team (Dict): Current team composition
            predictions (Dict): Player predictions for upcoming gameweek
        """
        self.current_team = current_team
        self.predictions = predictions
        self.available_transfers = MAX_FREE_TRANSFERS
        self.transfer_history = []
    
    def evaluate_transfer_potential(self, 
                                    current_player: Dict[str, Any], 
                                    potential_replacement: Dict[str, Any]) -> float:
        """
        Calculate the potential point gain from a transfer.
        
        Args:
            current_player (Dict): Player being replaced
            potential_replacement (Dict): Potential new player
        
        Returns:
            float: Expected point improvement
        """
        # Compare predicted points
        current_points = current_player.get('predicted_points', 0)
        replacement_points = potential_replacement.get('predicted_points', 0)
        
        # Calculate point difference
        point_difference = replacement_points - current_points
        
        # Consider additional factors
        price_difference = potential_replacement.get('price', 0) - current_player.get('price', 0)
        
        # Scoring formula (can be refined)
        transfer_score = point_difference - (price_difference * 0.5)
        
        return transfer_score
    
    def find_best_transfers(self, 
                             num_transfers: int = 1) -> List[Dict[str, Any]]:
        """
        Identify the most beneficial transfers for the team.
        
        Args:
            num_transfers (int): Number of transfers to recommend
        
        Returns:
            List of recommended transfers
        """
        recommended_transfers = []
        
        # Combine predictions across positions
        all_players = []
        for position_predictions in self.predictions.values():
            all_players.extend(position_predictions)
        
        # Sort players by potential point gain
        for player_to_replace in self.current_team['squad']:
            potential_replacements = [
                p for p in all_players 
                if p['position'] == player_to_replace['position'] and 
                   p['player_id'] != player_to_replace['player_id']
            ]
            
            # Evaluate potential transfers
            transfer_options = []
            for replacement in potential_replacements:
                transfer_score = self.evaluate_transfer_potential(
                    player_to_replace, 
                    replacement
                )
                transfer_options.append({
                    'current_player': player_to_replace,
                    'replacement_player': replacement,
                    'transfer_score': transfer_score
                })
            
            # Sort by transfer score
            transfer_options.sort(key=lambda x: x['transfer_score'], reverse=True)
            
            # Add top transfer to recommendations
            if transfer_options:
                recommended_transfers.append(transfer_options[0])
            
            # Limit transfers
            if len(recommended_transfers) >= num_transfers:
                break
        
        return recommended_transfers
    
    def plan_transfers(self) -> Dict[str, Any]:
        """
        Generate a comprehensive transfer strategy.
        
        Returns:
            Dict containing transfer recommendations and details
        """
        # Determine number of transfers
        transfers_to_make = min(self.available_transfers, MAX_TRANSFERS_STORED)
        
        # Find best transfers
        recommended_transfers = self.find_best_transfers(transfers_to_make)
        
        # Calculate total point gain
        total_point_gain = sum(
            transfer['transfer_score'] for transfer in recommended_transfers
        )
        
        # Prepare transfer details
        transfer_details = {
            'recommended_transfers': [
                {
                    'out': transfer['current_player'],
                    'in': transfer['replacement_player']
                } for transfer in recommended_transfers
            ],
            'total_point_gain': total_point_gain,
            'transfers_available': self.available_transfers,
            'budget_impact': sum(
                transfer['replacement_player'].get('price', 0) - 
                transfer['current_player'].get('price', 0) 
                for transfer in recommended_transfers
            )
        }
        
        # Update transfer history
        self.transfer_history.append(transfer_details)
        
        # Update available transfers
        self.available_transfers = max(
            0, 
            self.available_transfers - len(recommended_transfers)
        )
        
        return transfer_details
    
    def analyze_long_term_strategy(self) -> Dict[str, Any]:
        """
        Provide insights into long-term transfer strategy.
        
        Returns:
            Dict with strategic insights
        """
        # Analyze transfer history
        if not self.transfer_history:
            return {"status": "No transfer history available"}
        
        # Calculate transfer effectiveness
        total_point_gains = [
            transfer_week['total_point_gain'] 
            for transfer_week in self.transfer_history
        ]
        
        insights = {
            "total_transfers_made": len(self.transfer_history),
            "average_point_gain_per_transfer_week": np.mean(total_point_gains) if total_point_gains else 0,
            "max_point_gain": max(total_point_gains) if total_point_gains else 0,
            "transfer_efficiency": sum(total_point_gains) / len(self.transfer_history) if self.transfer_history else 0
        }
        
        return insights