"""
Utility functions for the FPL Prediction System API.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from pathlib import Path

# Import core utilities
from core.utils import LoggerManager


def format_player_data(player_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format raw player data for API responses.
    
    Args:
        player_data: Raw player data
        
    Returns:
        Formatted player data
    """
    # Format float values to 2 decimal places
    for key, value in player_data.items():
        if isinstance(value, float):
            player_data[key] = round(value, 2)
    
    return player_data


def parse_gameweek(gameweek_param: Optional[str]) -> Optional[int]:
    """
    Parse gameweek parameter from request.
    
    Args:
        gameweek_param: Gameweek parameter string
        
    Returns:
        Parsed gameweek number or None
    """
    if not gameweek_param:
        return None
        
    try:
        gameweek = int(gameweek_param)
        if gameweek < 1 or gameweek > 38:
            raise ValueError("Gameweek must be between 1 and 38")
        return gameweek
    except ValueError:
        if gameweek_param.lower() == "current":
            return None  # Use current gameweek
        raise ValueError(f"Invalid gameweek parameter: {gameweek_param}")


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    
    def default(self, obj):
        """Convert datetime objects to ISO format strings."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def save_request_data(data: Dict[str, Any], endpoint: str) -> str:
    """
    Save API request data for debugging and analysis.
    
    Args:
        data: Request data
        endpoint: API endpoint
        
    Returns:
        Path to saved file
    """
    try:
        logger = LoggerManager.get_logger("api.utils")
        
        # Create log directory if it doesn't exist
        log_dir = Path("logs/api_requests")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{endpoint.replace('/', '_')}_{timestamp}.json"
        file_path = log_dir / filename
        
        # Save data
        with open(file_path, "w") as f:
            json.dump(data, f, cls=JSONEncoder, indent=2)
            
        logger.info(f"Request data saved to {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger = LoggerManager.get_logger("api.utils")
        logger.error(f"Error saving request data: {e}")
        return "Failed to save request data"