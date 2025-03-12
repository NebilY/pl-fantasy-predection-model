"""
Configuration settings for the FPL Prediction System.
This file centralizes all configuration variables used throughout the application.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FEATURES_DIR = os.path.join(MODELS_DIR, "features")

# FPL API URLs
FPL_API_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_API_BASE_URL}/bootstrap-static/"
FPL_FIXTURES_URL = f"{FPL_API_BASE_URL}/fixtures/"
FPL_PLAYER_SUMMARY_URL = f"{FPL_API_BASE_URL}/element-summary/"

# Team selection constraints
MAX_BUDGET = 100.0  # £100M budget
MAX_PLAYERS_PER_TEAM = 3
SQUAD_SIZE = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
    "TOTAL": 15
}

# Valid formations (DEF-MID-FWD)
VALID_FORMATIONS = {
    "3-4-3": (3, 4, 3),
    "3-5-2": (3, 5, 2),
    "4-3-3": (4, 3, 3),
    "4-4-2": (4, 4, 2),
    "4-5-1": (4, 5, 1),
    "5-3-2": (5, 3, 2),
    "5-4-1": (5, 4, 1)
}

# Transfer rules
MAX_FREE_TRANSFERS = 2  # Maximum free transfers that can be accumulated
MAX_TRANSFERS_STORED = 5  # FPL specific rule

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"

# Cache settings
CACHE_PREDICTIONS = True
PREDICTION_CACHE_TIMEOUT = 3600  # 1 hour in seconds

# Create required directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODEL_FEATURES_DIR, exist_ok=True)