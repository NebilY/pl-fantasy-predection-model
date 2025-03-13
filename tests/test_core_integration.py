# tests/test_core_integration.py
import os
import sys
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your modules
from core.data_manager import DataManager
from core.feature_engineering import FeatureEngineering
from core.model_manager import ModelManager
from core.prediction_manager import PredictionManager
from core.team_selector import TeamSelector
from core.transfer_planner import TransferPlanner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integration_test")

def test_integration():
    logger.info("Starting integration test")
    
    # Initialize components
    data_manager = DataManager()
    feature_eng = FeatureEngineering(data_manager)
    model_manager = ModelManager()
    prediction_manager = PredictionManager()
    
    # Get current gameweek
    current_gw = data_manager.get_current_gameweek()
    logger.info(f"Current gameweek: {current_gw}")
    
    # Generate predictions for current gameweek
    logger.info("Generating predictions")
    predictions = prediction_manager.generate_predictions(current_gw)
    
    # Use predictions for team selection
    logger.info("Optimizing team selection")
    team_selector = TeamSelector(predictions)
    optimized_team = team_selector.optimize_team()
    
    # Print optimized team details
    logger.info(f"Optimized team formation: {optimized_team['formation']}")
    logger.info(f"Captain: {optimized_team['captain']['player_name']}")
    logger.info(f"Total predicted points: {optimized_team['total_points']}")
    
    # Sample transfer planning
    logger.info("Planning transfers")
    transfer_planner = TransferPlanner(optimized_team, predictions)
    transfer_plan = transfer_planner.plan_transfers()
    
    if transfer_plan['recommended_transfers']:
        logger.info(f"Recommended transfers: {len(transfer_plan['recommended_transfers'])}")
        for transfer in transfer_plan['recommended_transfers']:
            logger.info(f"OUT: {transfer['out']['player_name']} - IN: {transfer['in']['player_name']}")
    
    logger.info("Integration test completed")
    return {
        'optimized_team': optimized_team,
        'transfer_plan': transfer_plan
    }

if __name__ == "__main__":
    results = test_integration()