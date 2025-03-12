# core/prediction_manager.py
# core/prediction_manager.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from core.model_manager import ModelManager
from core.data_manager import DataManager
from config.settings import PROCESSED_DATA_DIR, CACHE_PREDICTIONS
import os
from datetime import datetime, timedelta

class PredictionManager:
    """
    Manages the generation of predictions for all player positions,
    handling data preparation, model inference, and result formatting.
    """
    
    def __init__(self):
        """
        Initialize PredictionManager with model and data managers.
        """
        self.model_manager = ModelManager()
        self.data_manager = DataManager()
        
        # Prediction cache
        self.prediction_cache: Dict[str, Dict] = {}
        
    def prepare_prediction_data(self, position: str, gameweek: int) -> pd.DataFrame:
        """
        Prepare prediction data for a specific position and gameweek.
        
        Args:
            position (str): Player position (goalkeeper, defender, midfielder, forward)
            gameweek (int): Gameweek number for predictions
        
        Returns:
            pd.DataFrame: Prepared features for prediction
        """
        # Load processed data for the position
        processed_file = os.path.join(PROCESSED_DATA_DIR, f"{position}_processed.csv")
        
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed data not found for {position}")
        
        df = pd.read_csv(processed_file)
        
        # Filter relevant features for the model
        model_features = self.model_manager.get_model_features(position)
        
        if not model_features:
            raise ValueError(f"No features found for {position} model")
        
        # Ensure all required features are present
        missing_features = [feat for feat in model_features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features for {position}: {missing_features}")
        
        # Prepare prediction dataset
        prediction_df = df[model_features].copy()
        
        return prediction_df
    
    def generate_predictions(self, gameweek: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate predictions for all positions for a specific gameweek.
        
        Args:
            gameweek (int): Gameweek number for predictions
        
        Returns:
            Dict containing predictions for each position
        """
        # Check cache first if enabled
        if CACHE_PREDICTIONS:
            cached_prediction = self._check_prediction_cache(gameweek)
            if cached_prediction:
                return cached_prediction
        
        predictions = {}
        positions = ['goalkeeper', 'defender', 'midfielder', 'forward']
        
        for position in positions:
            try:
                # Prepare data
                prediction_data = self.prepare_prediction_data(position, gameweek)
                
                # Generate predictions
                predicted_points = self.model_manager.predict(position, prediction_data)
                
                # Format predictions
                position_predictions = self._format_predictions(
                    prediction_data, 
                    predicted_points, 
                    position
                )
                
                predictions[position] = position_predictions
            
            except Exception as e:
                print(f"Error generating predictions for {position}: {e}")
                predictions[position] = []
        
        # Cache predictions if enabled
        if CACHE_PREDICTIONS:
            self._cache_predictions(gameweek, predictions)
        
        return predictions
    
    def _format_predictions(self, 
                             data: pd.DataFrame, 
                             predictions: np.ndarray, 
                             position: str) -> List[Dict[str, Any]]:
        """
        Format predictions into a standardized structure.
        
        Args:
            data (pd.DataFrame): Original prediction data
            predictions (np.ndarray): Predicted points
            position (str): Player position
        
        Returns:
            List of formatted player predictions
        """
        formatted_predictions = []
        
        for index, (_, row) in enumerate(data.iterrows()):
            prediction_entry = {
                'player_id': row.get('player_id', 'N/A'),
                'player_name': row.get('player_name', 'Unknown'),
                'team_name': row.get('team_name', 'N/A'),
                'position': position,
                'predicted_points': float(predictions[index]),
                'price': row.get('cost', 0),
                'additional_features': {
                    'minutes_trend': row.get('minutes_trend', 0),
                    'recent_form': row.get('rolling_points', 0),
                    # Add more relevant features
                }
            }
            
            formatted_predictions.append(prediction_entry)
        
        # Sort predictions by predicted points in descending order
        return sorted(
            formatted_predictions, 
            key=lambda x: x['predicted_points'], 
            reverse=True
        )
    
    def _check_prediction_cache(self, gameweek: int) -> Optional[Dict]:
        """
        Check if cached predictions exist for a gameweek.
        
        Args:
            gameweek (int): Gameweek number
        
        Returns:
            Cached predictions or None
        """
        if not CACHE_PREDICTIONS:
            return None
        
        cached = self.prediction_cache.get(str(gameweek))
        
        # Check cache expiration (1 hour)
        if cached and (datetime.now() - cached['timestamp']) < timedelta(hours=1):
            return cached['predictions']
        
        return None
    
    def _cache_predictions(self, gameweek: int, predictions: Dict) -> None:
        """
        Cache predictions for a gameweek.
        
        Args:
            gameweek (int): Gameweek number
            predictions (Dict): Predictions to cache
        """
        if not CACHE_PREDICTIONS:
            return
        
        self.prediction_cache[str(gameweek)] = {
            'predictions': predictions,
            'timestamp': datetime.now()
        }

# Example usage and testing
if __name__ == "__main__":
    prediction_manager = PredictionManager()
    
    # Generate predictions for current gameweek
    current_gameweek = 28  # This would typically be dynamically determined
    predictions = prediction_manager.generate_predictions(current_gameweek)
    
    # Print top 5 predictions for each position
    for position, player_predictions in predictions.items():
        print(f"\nTop 5 {position.capitalize()} Predictions:")
        for player in player_predictions[:5]:
            print(f"{player['player_name']} - {player['predicted_points']:.2f} points")