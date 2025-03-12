"""
Simplified application entry point for the FPL Prediction System API.
This version eliminates complex dependencies to get the server running.
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

# Import simplified routes
from api.simplified_routes import router

# Create and configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Create FastAPI application
app = FastAPI(
    title="Fantasy Premier League Prediction API",
    description="API for FPL team optimization, transfer planning, and player predictions",
    version="1.0.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request information and timing."""
    start_time = time.time()
    
    # Get client IP and requested path
    client_ip = request.client.host
    request_path = request.url.path
    
    logger.info(f"Request received: {request.method} {request_path} from {client_ip}")
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response info
        logger.info(f"Request completed: {request.method} {request_path} - Status: {response.status_code} - Time: {process_time:.4f}s")
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        # Log exceptions
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request_path} - Error: {str(e)} - Time: {process_time:.4f}s")
        raise

# Include API routes
app.include_router(router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Fantasy Premier League Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    
    # Log startup information
    logger.info(f"Starting FPL Prediction API")
    
    # Start the server
    uvicorn.run(
        "api.simplified_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )