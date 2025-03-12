"""
API routes for the FPL Prediction System.
This module defines the endpoints for team optimization, transfers, and predictions.
"""

from fastapi import APIRouter, Depends, Query, Path, HTTPException, status
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Import models and services
from api.models import (
    PlayerPrediction, TeamOptimizationRequest, TeamOptimizationResponse,
    TransferPlanRequest, TransferPlanResponse, APIResponse
)
from api.services import get_prediction_service, get_team_service, get_transfer_service
from api.auth import auth_router

# Create router
router = APIRouter()

# Include auth router
router.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# Set up logging
logger = logging.getLogger("api.routes")

# Temporary placeholder response
@router.get("/")
async def api_root():
    """Root endpoint for the API."""
    return {"message": "FPL Prediction API is running"}

# Predictions endpoints
@router.get("/predictions/players", response_model=Dict[str, List[PlayerPrediction]])
async def get_player_predictions(
    gameweek: Optional[int] = Query(None, description="Gameweek number for predictions"),
    prediction_service = Depends(get_prediction_service)
):
    """Get player predictions for a specific gameweek."""
    try:
        predictions = await prediction_service.get_player_predictions(gameweek)
        return predictions
    except Exception as e:
        logger.exception(f"Error getting predictions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate predictions: {str(e)}"
        )

@router.get("/predictions/gameweek")
async def get_current_gameweek(
    prediction_service = Depends(get_prediction_service)
):
    """Get the current gameweek information."""
    try:
        current_gw = prediction_service.prediction_manager.data_manager.get_current_gameweek()
        return {
            "status": "success",
            "gameweek": current_gw,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.exception(f"Error getting current gameweek: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current gameweek: {str(e)}"
        )

# Team optimization endpoints
@router.get("/optimize/team", response_model=TeamResponse)
async def optimize_team(
    budget: float = Query(100.0, description="Available budget in millions"),
    gameweek: Optional[int] = Query(None, description="Gameweek number for predictions"),
    team_service = Depends(get_team_service)
):
    """Optimize team selection based on predictions."""
    try:
        optimized_team = await team_service.optimize_team(budget, gameweek)
        
        return {
            "status": "success", 
            "message": f"Team optimization for {budget}M budget",
            "timestamp": datetime.now(),
            "data": optimized_team
        }
    except Exception as e:
        logger.exception(f"Error optimizing team: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize team: {str(e)}"
        )

@router.post("/optimize/team", response_model=TeamResponse)
async def optimize_team_with_constraints(
    request: TeamOptimizationRequest,
    team_service = Depends(get_team_service)
):
    """Optimize team selection with custom constraints."""
    try:
        # Extract parameters from request
        optimized_team = await team_service.optimize_team(
            budget=request.budget,
            gameweek=request.gameweek,
            existing_team=request.existing_team
        )
        
        return {
            "status": "success", 
            "message": "Team optimization with constraints",
            "timestamp": datetime.now(),
            "data": optimized_team
        }
    except Exception as e:
        logger.exception(f"Error optimizing team with constraints: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize team: {str(e)}"
        )

# Transfer planning endpoints
@router.post("/transfers/plan", response_model=TransferResponse)
async def plan_transfers(
    request: TransferPlanRequest,
    transfer_service = Depends(get_transfer_service)
):
    """Generate transfer recommendations."""
    try:
        # Extract parameters from request
        transfer_plan = await transfer_service.plan_transfers(
            current_team=request.current_team,
            transfers_available=request.transfers_available,
            budget_available=request.budget_available,
            gameweek=request.gameweek
        )
        
        return {
            "status": "success",
            "message": f"Transfer plan with {request.transfers_available} available transfers",
            "timestamp": datetime.now(),
            "data": transfer_plan
        }
    except Exception as e:
        logger.exception(f"Error planning transfers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to plan transfers: {str(e)}"
        )

@router.get("/transfers/plan", response_model=TransferResponse)
async def plan_transfers_simple(
    transfers_available: int = Query(1, ge=0, le=2, description="Number of available transfers"),
    gameweek: Optional[int] = Query(None, description="Gameweek number for predictions"),
    team_id: Optional[str] = Query(None, description="Team ID to load existing team"),
    transfer_service = Depends(get_transfer_service)
):
    """Generate simple transfer recommendations."""
    try:
        # Get team data (simplified for now)
        current_team = {"squad": [], "team_id": team_id}
        
        # Generate transfer plan
        transfer_plan = await transfer_service.plan_transfers(
            current_team=current_team,
            transfers_available=transfers_available,
            gameweek=gameweek
        )
        
        return {
            "status": "success",
            "message": f"Transfer plan with {transfers_available} available transfers",
            "timestamp": datetime.now(),
            "data": transfer_plan
        }
    except Exception as e:
        logger.exception(f"Error planning transfers: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to plan transfers: {str(e)}"
        )

# Player analysis endpoints
@router.get("/players/{player_id}")
async def get_player_analysis(
    player_id: str = Path(..., description="Player ID to analyze"),
    gameweeks: int = Query(5, description="Number of gameweeks to analyze"),
    prediction_service = Depends(get_prediction_service)
):
    """Get detailed analysis for a specific player."""
    try:
        # Placeholder for player analysis logic
        player_data = {
            "player_id": player_id,
            "analysis_period": f"Last {gameweeks} gameweeks",
            "form": 7.8,
            "expected_points": 5.4,
            "fixtures": [
                {"opponent": "MUN (H)", "difficulty": 4, "expected_points": 4.2},
                {"opponent": "CHE (A)", "difficulty": 3, "expected_points": 3.8}
            ]
        }
        
        return {
            "status": "success",
            "message": f"Player analysis for ID {player_id}",
            "data": player_data
        }
    except Exception as e:
        logger.exception(f"Error analyzing player {player_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze player: {str(e)}"
        )

# Fixtures endpoints
@router.get("/fixtures")
async def get_fixtures(
    gameweek: Optional[int] = Query(None, description="Gameweek number for fixtures"),
    prediction_service = Depends(get_prediction_service)
):
    """Get fixtures for a specific gameweek."""
    try:
        # Placeholder for fixtures logic
        fixtures_data = [
            {"home_team": "ARS", "away_team": "MUN", "kickoff_time": "2024-03-15T15:00:00Z", "difficulty": 4},
            {"home_team": "CHE", "away_team": "LIV", "kickoff_time": "2024-03-15T17:30:00Z", "difficulty": 4}
        ]
        
        return {
            "status": "success",
            "message": f"Fixtures for gameweek {gameweek if gameweek else 'current'}",
            "data": fixtures_data
        }
    except Exception as e:
        logger.exception(f"Error getting fixtures: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fixtures: {str(e)}"
        )

# Health check for this router
@router.get("/health")
async def router_health():
    """Health check endpoint for the router."""
    return {"status": "healthy", "router": "api.routes"}