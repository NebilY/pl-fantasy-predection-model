# core/model_manager.py
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

from config.settings import MODELS_DIR, MODEL_FEATURES_DIR

class ModelManager:
    """
    Manages machine learning models for different player positions,
    handling model loading, saving, and version tracking.
    """
    
    SUPPORTED_POSITIONS = ['goalkeeper', 'defender', 'midfielder', 'forward']
    
    def __init__(self):
        """
        Initialize the ModelManager with default settings.
        """
        self.models: Dict[str, Any] = {}
        self.feature_sets: Dict[str, list] = {}
        self.model_versions: Dict[str, str] = {}
    
    def create_default_model(self, position: str) -> BaseEstimator:
        """
        Create a default Gradient Boosting Regressor for a given position.
        
        Args:
            position (str): Player position
        
        Returns:
            BaseEstimator: Initialized machine learning model
        """
        if position not in self.SUPPORTED_POSITIONS:
            raise ValueError(f"Unsupported position: {position}")
        
        # Position-specific model configuration could be added here
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    
    def save_model(self, 
                   model: BaseEstimator, 
                   position: str, 
                   features: Optional[list] = None,
                   version: Optional[str] = None) -> None:
        """
        Save a trained model for a specific position.
        
        Args:
            model (BaseEstimator): Trained machine learning model
            position (str): Player position
            features (list, optional): Features used to train the model
            version (str, optional): Model version identifier
        """
        if position not in self.SUPPORTED_POSITIONS:
            raise ValueError(f"Unsupported position: {position}")
        
        # Ensure model directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(MODEL_FEATURES_DIR, exist_ok=True)
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version()
        
        # Save model
        model_path = os.path.join(MODELS_DIR, f"{position}_model_{version}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save features
        if features:
            feature_path = os.path.join(MODEL_FEATURES_DIR, f"{position}_features_{version}.pkl")
            with open(feature_path, 'wb') as f:
                pickle.dump(features, f)
        
        # Update internal tracking
        self.models[position] = model
        self.feature_sets[position] = features
        self.model_versions[position] = version
        
        print(f"Model for {position} saved: {model_path}")
    
    def load_latest_model(self, position: str) -> BaseEstimator:
        """
        Load the latest model for a specific position.
        
        Args:
            position (str): Player position
        
        Returns:
            BaseEstimator: Loaded machine learning model
        """
        if position not in self.SUPPORTED_POSITIONS:
            raise ValueError(f"Unsupported position: {position}")
        
        # Find the latest model file
        model_files = [
            f for f in os.listdir(MODELS_DIR) 
            if f.startswith(f"{position}_model_") and f.endswith('.pkl')
        ]
        
        if not model_files:
            # Create and return a default model if no model exists
            default_model = self.create_default_model(position)
            return default_model
        
        # Sort and get the latest model
        latest_model_file = sorted(model_files)[-1]
        model_path = os.path.join(MODELS_DIR, latest_model_file)
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load corresponding features
        feature_file = latest_model_file.replace('model', 'features')
        feature_path = os.path.join(MODEL_FEATURES_DIR, feature_file)
        
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                features = pickle.load(f)
            self.feature_sets[position] = features
        
        # Update internal tracking
        self.models[position] = model
        self.model_versions[position] = latest_model_file.split('_')[-1].split('.')[0]
        
        return model
    
    def _generate_version(self) -> str:
        """
        Generate a unique version identifier for models.
        
        Returns:
            str: Version identifier
        """
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_model_features(self, position: str) -> Optional[list]:
        """
        Retrieve the features used for a specific position's model.
        
        Args:
            position (str): Player position
        
        Returns:
            Optional[list]: List of features or None
        """
        return self.feature_sets.get(position)
    
    def predict(self, position: str, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the latest model for a position.
        
        Args:
            position (str): Player position
            X (pd.DataFrame): Input features
        
        Returns:
            np.ndarray: Predicted values
        """
        model = self.load_latest_model(position)
        
        # Ensure features match those used during training
        model_features = self.get_model_features(position)
        if model_features:
            X = X[model_features]
        
        return model.predict(X)

# Example usage
if __name__ == "__main__":
    model_manager = ModelManager()
    
    # Load or create a model for defenders
    defender_model = model_manager.load_latest_model('defender')
    
    # Later, when you have training data
    # model_manager.save_model(trained_model, 'defender', features)