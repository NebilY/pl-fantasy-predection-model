"""
Pydantic models for the FPL Prediction System API.
This module defines the request and response models used in the API.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


# Player Models
class PlayerBase(BaseModel):
    """Base player information."""
    player_id: str
    player_name: str
    team_name: str
    position: str
    price: float = Field(..., description="Player price in millions")
    

class PlayerPrediction(PlayerBase):
    """Player with prediction information."""
    predicted_points: float
    additional_features: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Team Models
class TeamPlayer(PlayerBase):
    """Player in a team context."""
    is_captain: bool = False
    is_vice_captain: bool = False
    predicted_points: Optional[float] = None


class TeamOptimizationRequest(BaseModel):
    """Request model for team optimization."""
    budget: float = Field(100.0, description="Available budget in millions")
    existing_team: Optional[List[TeamPlayer]] = None
    gameweek: Optional[int] = None
    constraints: Optional[Dict[str, Any]] = None


class TeamOptimizationResponse(BaseModel):
    """Response model for team optimization."""
    formation: str
    starting_xi: List[PlayerPrediction]
    bench: List[PlayerPrediction]
    captain: PlayerPrediction
    vice_captain: PlayerPrediction
    total_points: float
    total_cost: float


# Transfer Models
class TransferAction(BaseModel):
    """A single transfer action."""
    player_out: PlayerBase
    player_in: PlayerBase
    point_impact: float


class TransferPlanRequest(BaseModel):
    """Request model for transfer planning."""
    current_team: List[TeamPlayer]
    transfers_available: int = Field(1, ge=0, le=2)
    budget_available: float
    gameweek: Optional[int] = None


class TransferPlanResponse(BaseModel):
    """Response model for transfer recommendations."""
    recommended_transfers: List[TransferAction]
    total_point_gain: float
    transfers_available: int
    budget_impact: float


# API Response Models
class APIResponse(BaseModel):
    """Base API response model."""
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionResponse(APIResponse):
    """Response for prediction endpoints."""
    data: List[PlayerPrediction]
    gameweek: int


class TeamResponse(APIResponse):
    """Response for team optimization endpoints."""
    data: TeamOptimizationResponse


class TransferResponse(APIResponse):
    """Response for transfer planning endpoints."""
    data: TransferPlanResponse


# Authentication Models
class UserCredentials(BaseModel):
    """User login credentials."""
    username: str
    password: str


class Token(BaseModel):
    """Authentication token."""
    access_token: str
    token_type: str
    expires_at: datetime


class UserProfile(BaseModel):
    """User profile information."""
    user_id: str
    username: str
    email: Optional[str] = None
    created_at: datetime
    is_premium: bool = False