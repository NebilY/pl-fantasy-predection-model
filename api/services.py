"""
Service layer for the FPL Prediction System API.
This module connects API routes with core functionality.
"""

import logging
from typing import Dict, List, Any, Optional
from functools import lru_cache

# Import core components
from core.prediction_manager import PredictionManager
from core.team_selector import TeamSelector
from core.transfer_planner import TransferPlanner
from api.exceptions import (
    DataNotFoundException, 
    PredictionException,
    OptimizationException,
    TransferPlanningException
)

# Set up logging
logger = logging.getLogger("api.services")


class PredictionService:
    """Service for handling player predictions."""
    
    def __init__(self):
        """Initialize prediction service with manager."""
        try:
            self.prediction_manager = PredictionManager()
        except Exception as e:
            logger.error(f"Error initializing prediction service: {e}")
            raise PredictionException(detail=f"Failed to initialize prediction service: {str(e)}")
    
    async def get_player_predictions(self, gameweek: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get player predictions for a specific gameweek.
        
        Args:
            gameweek: Optional gameweek number, defaults to current gameweek
            
        Returns:
            Dictionary of player predictions by position
        """
        try:
            # Use current gameweek if none provided
            if gameweek is None:
                gameweek = self.prediction_manager.data_manager.get_current_gameweek()
                
            # Generate predictions
            predictions = self.prediction_manager.generate_predictions(gameweek)
            
            if not predictions:
                raise DataNotFoundException(detail=f"No predictions found for gameweek {gameweek}")
                
            return predictions
            
        except DataNotFoundException as e:
            # Re-raise data not found errors
            raise
        except Exception as e:
            # Log and wrap other errors
            logger.error(f"Error generating predictions for gameweek {gameweek}: {e}")
            raise PredictionException(detail=f"Failed to generate predictions: {str(e)}")


class TeamService:
    """Service for handling team optimization."""
    
    def __init__(self, prediction_service: Optional[PredictionService] = None):
        """Initialize team service."""
        self.prediction_service = prediction_service or PredictionService()
    
    async def optimize_team(
        self, 
        budget: float = 100.0, 
        gameweek: Optional[int] = None,
        existing_team: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize team selection based on predictions.
        
        Args:
            budget: Available budget in millions
            gameweek: Optional gameweek number
            existing_team: Optional existing team to consider
            
        Returns:
            Optimized team data
        """
        try:
            # Get predictions for the gameweek
            predictions = await self.prediction_service.get_player_predictions(gameweek)
            
            # Create team selector
            team_selector = TeamSelector(predictions)
            
            # Optimize team
            if existing_team:
                # TODO: Implement team optimization with existing team
                optimized_team = team_selector.optimize_team()
            else:
                optimized_team = team_selector.optimize_team()
            
            return optimized_team
            
        except Exception as e:
            logger.error(f"Error optimizing team: {e}")
            raise OptimizationException(detail=f"Failed to optimize team: {str(e)}")


class TransferService:
    """Service for handling transfer planning."""
    
    def __init__(
        self, 
        prediction_service: Optional[PredictionService] = None,
        team_service: Optional[TeamService] = None
    ):
        """Initialize transfer service."""
        self.prediction_service = prediction_service or PredictionService()
        self.team_service = team_service or TeamService(self.prediction_service)
    
    async def plan_transfers(
        self,
        current_team: Dict[str, Any],
        transfers_available: int = 1,
        budget_available: Optional[float] = None,
        gameweek: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate transfer recommendations.
        
        Args:
            current_team: Current team composition
            transfers_available: Number of available transfers
            budget_available: Available budget for transfers
            gameweek: Optional gameweek number
            
        Returns:
            Transfer recommendations
        """
        try:
            # Get predictions for the gameweek
            predictions = await self.prediction_service.get_player_predictions(gameweek)
            
            # Create transfer planner
            transfer_planner = TransferPlanner(current_team, predictions)
            transfer_planner.available_transfers = transfers_available
            
            # Generate transfer plan
            transfer_plan = transfer_planner.plan_transfers()
            
            return transfer_plan
            
        except Exception as e:
            logger.error(f"Error planning transfers: {e}")
            raise TransferPlanningException(detail=f"Failed to plan transfers: {str(e)}")


# Factory functions for dependency injection
@lru_cache()
def get_prediction_service() -> PredictionService:
    """Get or create a prediction service instance."""
    return PredictionService()


@lru_cache()
def get_team_service() -> TeamService:
    """Get or create a team service instance."""
    prediction_service = get_prediction_service()
    return TeamService(prediction_service)


@lru_cache()
def get_transfer_service() -> TransferService:
    """Get or create a transfer service instance."""
    prediction_service = get_prediction_service()
    team_service = get_team_service()
    return TransferService(prediction_service, team_service)