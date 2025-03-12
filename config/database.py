"""
Database connection module for FPL Prediction System.
Handles connections to PostgreSQL using connection pooling.
"""

import os
import logging
from typing import Any
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
    'dbname': os.getenv('DB_NAME')
}

# Connection pool
connection_pool = None

def get_connection_pool(min_conn=1, max_conn=10):
    """
    Get or create a database connection pool.
    
    Args:
        min_conn: Minimum number of connections in the pool
        max_conn: Maximum number of connections in the pool
        
    Returns:
        psycopg2.pool.ThreadedConnectionPool: The connection pool
    """
    global connection_pool
    
    if connection_pool is None:
        try:
            logger.info("Creating database connection pool")
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                dbname=DB_CONFIG['dbname']
            )
        except Exception as e:
            logger.error(f"Error creating connection pool: {e}")
            raise
    
    return connection_pool

def get_connection():
    """
    Get a connection from the pool.
    
    Returns:
        connection: Database connection
    """
    pool = get_connection_pool()
    try:
        return pool.getconn()
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        raise

def release_connection(connection):
    """
    Release a connection back to the pool.
    
    Args:
        connection: Database connection to release
    """
    pool = get_connection_pool()
    try:
        pool.putconn(connection)
    except Exception as e:
        logger.error(f"Error releasing connection to pool: {e}")
        raise

def execute_query(query, params=None, fetch=True):
    """
    Execute a SQL query and optionally fetch results.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query
        fetch: Whether to fetch results
        
    Returns:
        list: Query results (if fetch=True)
    """
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(query, params)
        
        if fetch:
            results = cursor.fetchall()
            return results
        else:
            connection.commit()
            return None
            
    except Exception as e:
        if connection:
            connection.rollback()
        logger.error(f"Error executing query: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if connection:
            release_connection(connection)

def close_all_connections():
    """Close all connections in the pool."""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        connection_pool = None
        logger.info("All database connections closed")