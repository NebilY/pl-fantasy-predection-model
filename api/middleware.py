"""
Middleware components for the FPL Prediction System API.
This module contains middleware for authentication, logging, and other concerns.
"""

import time
from datetime import datetime
import logging
from typing import Dict, Callable, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Import custom exceptions
from api.exceptions import (
    APIRateLimitException, 
    AuthenticationException
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging request and response details."""
    
    def __init__(self, app: ASGIApp, logger: Optional[logging.Logger] = None):
        """
        Initialize middleware with app and logger.
        
        Args:
            app: The ASGI application
            logger: Optional logger instance
        """
        super().__init__(app)
        self.logger = logger or logging.getLogger("api.middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log details.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Get request info
        start_time = time.time()
        request_id = f"{int(time.time() * 1000)}-{id(request) % 10000}"
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = str(request.query_params)
        
        # Log request
        self.logger.info(
            f"Request {request_id}: {method} {path} - "
            f"Client: {client_ip} - Params: {query_params}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            self.logger.info(
                f"Response {request_id}: {method} {path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.4f}s"
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log exceptions
            process_time = time.time() - start_time
            self.logger.error(
                f"Error {request_id}: {method} {path} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.4f}s"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for API rate limiting."""
    
    def __init__(
        self, 
        app: ASGIApp, 
        requests_limit: int = 100,
        window_seconds: int = 60,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize middleware with rate limit settings.
        
        Args:
            app: The ASGI application
            requests_limit: Maximum requests per window
            window_seconds: Time window in seconds
            logger: Optional logger instance
        """
        super().__init__(app)
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.logger = logger or logging.getLogger("api.middleware")
        
        # Store rate limit data: {client_ip: [(timestamp, count), ...]}
        self.rate_limit_data: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request with rate limiting.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Skip rate limiting for certain paths
        if request.url.path in ["/", "/docs", "/redoc", "/health"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check and update rate limit
        current_time = time.time()
        
        # Initialize client data if not exists
        if client_ip not in self.rate_limit_data:
            self.rate_limit_data[client_ip] = []
        
        # Remove old timestamps
        window_start = current_time - self.window_seconds
        self.rate_limit_data[client_ip] = [
            (ts, count) for ts, count in self.rate_limit_data[client_ip]
            if ts >= window_start
        ]
        
        # Count requests in the current window
        total_requests = sum(count for _, count in self.rate_limit_data[client_ip])
        
        # Check if rate limit exceeded
        if total_requests >= self.requests_limit:
            self.logger.warning(f"Rate limit exceeded for client {client_ip}")
            raise APIRateLimitException(
                detail=f"Rate limit of {self.requests_limit} requests per {self.window_seconds} seconds exceeded"
            )
        
        # Add current request
        if not self.rate_limit_data[client_ip]:
            self.rate_limit_data[client_ip].append((current_time, 1))
        else:
            # Increment last timestamp if it's within a small window
            last_ts, last_count = self.rate_limit_data[client_ip][-1]
            if current_time - last_ts < 1:  # Group within 1 second
                self.rate_limit_data[client_ip][-1] = (last_ts, last_count + 1)
            else:
                self.rate_limit_data[client_ip].append((current_time, 1))
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.requests_limit - total_requests - 1
        reset_time = int(window_start + self.window_seconds)
        
        response.headers["X-RateLimit-Limit"] = str(self.requests_limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API authentication."""
    
    def __init__(
        self, 
        app: ASGIApp,
        exclude_paths: Optional[list] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize middleware with authentication settings.
        
        Args:
            app: The ASGI application
            exclude_paths: Paths to exclude from authentication
            logger: Optional logger instance
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/", "/docs", "/redoc", "/health", "/api/health",
            "/api/auth/login", "/api/auth/register"
        ]
        self.logger = logger or logging.getLogger("api.middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request with authentication.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Get authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            self.logger.warning(f"Missing Authorization header for {request.url.path}")
            raise AuthenticationException(detail="Authentication required")
        
        # Basic token validation - in a real system, you'd use JWT or OAuth
        if not auth_header.startswith("Bearer "):
            self.logger.warning(f"Invalid Authorization format for {request.url.path}")
            raise AuthenticationException(detail="Invalid authorization format. Use 'Bearer {token}'")
        
        token = auth_header.split(" ")[1]
        
        # Validate token (placeholder - implement real validation)
        if not self._validate_token(token):
            self.logger.warning(f"Invalid token for {request.url.path}")
            raise AuthenticationException(detail="Invalid or expired token")
        
        # Process request
        return await call_next(request)
    
    def _validate_token(self, token: str) -> bool:
        """
        Validate authentication token.
        
        Args:
            token: Authentication token
            
        Returns:
            Whether the token is valid
        """
        # Placeholder - implement real token validation
        # In a real system, you'd verify JWT signature, check expiration, etc.
        # For development, accept any non-empty token
        return bool(token)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling."""
    
    def __init__(
        self, 
        app: ASGIApp,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize middleware with error handling settings.
        
        Args:
            app: The ASGI application
            logger: Optional logger instance
        """
        super().__init__(app)
        self.logger = logger or logging.getLogger("api.middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request with error handling.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        try:
            return await call_next(request)
        except Exception as e:
            # Log exception
            self.logger.error(f"Unhandled exception: {str(e)}")
            
            # Create error response
            from fastapi.responses import JSONResponse
            
            status_code = 500
            detail = "Internal server error"
            
            # Format based on exception type
            if hasattr(e, "status_code"):
                status_code = e.status_code
            
            if hasattr(e, "detail"):
                detail = e.detail
            
            return JSONResponse(
                status_code=status_code,
                content={
                    "status": "error",
                    "message": detail,
                    "timestamp": datetime.now().isoformat()
                }
            )