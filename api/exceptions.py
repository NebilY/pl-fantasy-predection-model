"""
Custom exceptions for the FPL Prediction System API.
This module defines exception classes to handle various error scenarios.
"""

from fastapi import HTTPException
from typing import Any, Dict, Optional


class FPLPredictionException(HTTPException):
    """Base exception for FPL prediction system errors."""
    
    def __init__(
        self,
        status_code: int = 500,
        detail: str = "An error occurred in the FPL prediction system",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class DataNotFoundException(FPLPredictionException):
    """Exception raised when required data is not found."""
    
    def __init__(
        self,
        detail: str = "Required data not found",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=404, detail=detail, headers=headers)


class ValidationException(FPLPredictionException):
    """Exception raised when request validation fails."""
    
    def __init__(
        self,
        detail: str = "Validation error in request data",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=400, detail=detail, headers=headers)


class PredictionException(FPLPredictionException):
    """Exception raised when prediction generation fails."""
    
    def __init__(
        self,
        detail: str = "Error generating predictions",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=500, detail=detail, headers=headers)


class OptimizationException(FPLPredictionException):
    """Exception raised when team optimization fails."""
    
    def __init__(
        self,
        detail: str = "Error optimizing team selection",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=500, detail=detail, headers=headers)


class TransferPlanningException(FPLPredictionException):
    """Exception raised when transfer planning fails."""
    
    def __init__(
        self,
        detail: str = "Error planning transfers",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=500, detail=detail, headers=headers)


class APIRateLimitException(FPLPredictionException):
    """Exception raised when API rate limits are exceeded."""
    
    def __init__(
        self,
        detail: str = "API rate limit exceeded",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=429, detail=detail, headers=headers)


class AuthenticationException(FPLPredictionException):
    """Exception raised for authentication errors."""
    
    def __init__(
        self,
        detail: str = "Authentication failed",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=401, detail=detail, headers=headers)


class AuthorizationException(FPLPredictionException):
    """Exception raised for authorization errors."""
    
    def __init__(
        self,
        detail: str = "Not authorized to access this resource",
        headers: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=403, detail=detail, headers=headers)