#!/usr/bin/env python
"""
Script to fix database tables for FPL Prediction System.
This script recreates the problematic bootstrap_static and fixtures tables.
"""

import os
import sys
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import database connection
from config.database import execute_query

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_tables")

def recreate_tables():
    """Drop and recreate problematic tables."""
    try:
        # Drop tables if they exist
        drop_queries = [
            "DROP TABLE IF EXISTS bootstrap_static",
            "DROP TABLE IF EXISTS fixtures"
        ]
        
        for query in drop_queries:
            logger.info(f"Executing: {query}")
            execute_query(query, fetch=False)
        
        # Create tables with proper structure
        create_queries = [
            """
            CREATE TABLE bootstrap_static (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                season VARCHAR(9) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(season)
            )
            """,
            """
            CREATE TABLE fixtures (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                season VARCHAR(9) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(season)
            )
            """
        ]
        
        for query in create_queries:
            logger.info(f"Creating table...")
            execute_query(query, fetch=False)
        
        logger.info("Tables recreated successfully")
        return True
    except Exception as e:
        logger.error(f"Error recreating tables: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting database table fix")
    success = recreate_tables()
    
    if success:
        logger.info("Database tables fixed successfully")
    else:
        logger.error("Failed to fix database tables")
        sys.exit(1)