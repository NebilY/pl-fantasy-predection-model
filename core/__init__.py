"""
Core module for FPL Prediction System.

This module contains the core components of the FPL prediction system,
including data collection, feature engineering, prediction generation,
team selection, and transfer planning.
"""

# Version information
__version__ = '0.1.0'

# Import key classes for easier access
from .data_manager import DataManager
from .feature_engineering import FeatureEngineering
from .model_manager import ModelManager
from .prediction_manager import PredictionManager
from .team_selector import TeamSelector
from .transfer_planner import TransferPlanner

# Export main classes
__all__ = [
    'DataManager',
    'FeatureEngineering',
    'ModelManager',
    'PredictionManager',
    'TeamSelector',
    'TransferPlanner',
]