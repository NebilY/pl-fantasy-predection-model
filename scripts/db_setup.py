"""
Database Connection and Initial Setup Script

This script helps verify database connection and performs initial setup
for the FPL Prediction System.
"""

import os
import sys
import logging
from typing import Dict

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import dependencies after path is set
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_database_connection():
    """
    Test database connection and basic functionality.
    """
    try:
        from config.database import (
            get_connection_pool, 
            execute_query, 
            get_connection, 
            release_connection
        )
        
        # Test connection pool creation
        logger.info("Testing database connection pool...")
        connection_pool = get_connection_pool()
        
        # Test getting a connection
        logger.info("Attempting to get a database connection...")
        connection = get_connection()
        
        # Test basic query
        logger.info("Executing test query...")
        test_query = "SELECT current_database()"
        result = execute_query(test_query)
        
        logger.info(f"Connected to database: {result[0][0]}")
        
        # Release connection
        release_connection(connection)
        
        return True
    
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False

def initialize_database():
    """
    Initialize database tables and initial setup.
    """
    try:
        from core.data_manager import DataManager
        
        # Create DataManager instance
        data_manager = DataManager()
        
        # Setup database tables
        logger.info("Setting up database tables...")
        data_manager.setup_database_tables()
        
        # Fetch initial bootstrap data
        logger.info("Fetching initial bootstrap data...")
        bootstrap_data = data_manager.fetch_bootstrap_static()
        
        # Fetch initial fixtures
        logger.info("Fetching initial fixtures...")
        fixtures_data = data_manager.fetch_fixtures()
        
        logger.info("Initial data collection completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

def collect_player_data():
    """
    Collect initial player data for all positions.
    """
    try:
        from core.data_manager import DataManager
        
        data_manager = DataManager()
        
        # Collect data for all positions
        logger.info("Collecting player data for all positions...")
        all_position_data = data_manager.collect_all_position_data()
        
        logger.info("Player data collection summary:")
        for position, data in all_position_data.items():
            logger.info(f"{position}: {len(data)} records")
        
        return True
    
    except Exception as e:
        logger.error(f"Player data collection error: {e}")
        return False

def main():
    """
    Main script execution flow.
    """
    logger.info("Starting FPL Prediction System Database Setup")
    
    # Test database connection
    if not test_database_connection():
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)
    
    # Initialize database tables
    if not initialize_database():
        logger.error("Database initialization failed. Exiting.")
        sys.exit(1)
    
    # Collect player data
    if not collect_player_data():
        logger.error("Player data collection failed.")
        # Continue even if data collection fails partially
    
    logger.info("FPL Prediction System Database Setup Complete")

if __name__ == "__main__":
    main()