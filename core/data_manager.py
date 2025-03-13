import requests
import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# SQLAlchemy imports
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError

# Import settings and database connection
from config import settings
from config.database import execute_query, get_connection, release_connection, DB_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

class DataManager:
    """
    Handles the collection, storage, and retrieval of FPL data.
    """
    
    def __init__(self):
        """Initialize the Data Manager."""
        self.current_gw = None
        self.bootstrap_data = None
        self.fixtures_data = None
        
        # Create required directories for any file-based outputs
        os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
    
    def get_current_season(self) -> str:
        """
        Determine the current FPL season.
        
        Returns:
            str: Current season in format '2023/2024'
        """
        current_date = datetime.now()
        current_year = current_date.year
        
        # FPL season typically starts in August and ends in May
        if current_date.month >= 8:
            return f"{current_year}/{current_year + 1}"
        else:
            return f"{current_year - 1}/{current_year}"
    
    def _load_bootstrap_data_from_db(self) -> Optional[Dict]:
        """
        Load bootstrap data from database.
        
        Returns:
            Optional[Dict]: Bootstrap data or None if not found
        """
        try:
            query = """
            SELECT data 
            FROM bootstrap_static 
            WHERE season = %s
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            # Get current season
            current_season = self.get_current_season()
            
            result = execute_query(query, (current_season,))
            
            if result and result[0] and result[0][0]:
                logger.info("Bootstrap data loaded from database")
                
                # Check if the data is already a dictionary (happens with certain drivers)
                if isinstance(result[0][0], dict):
                    return result[0][0]
                
                # Otherwise parse the JSON string
                return json.loads(result[0][0])
                
            return None
            
        except Exception as e:
            logger.error(f"Error loading bootstrap data from database: {e}")
            return None
    
    def setup_database_tables(self) -> bool:
        """
        Create necessary database tables if they don't exist.
        
        Returns:
            bool: Whether table creation was successful
        """
        try:
            # SQL statements for creating tables
            create_table_queries = [
                """
                CREATE TABLE IF NOT EXISTS goalkeeper_history (
                    id SERIAL PRIMARY KEY,
                    season VARCHAR(9) NOT NULL,
                    gameweek INTEGER NOT NULL,
                    player_id INTEGER NOT NULL,
                    total_points NUMERIC,
                    minutes NUMERIC,
                    clean_sheets NUMERIC,
                    goals_conceded NUMERIC,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, gameweek)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS defender_history (
                    id SERIAL PRIMARY KEY,
                    season VARCHAR(9) NOT NULL,
                    gameweek INTEGER NOT NULL,
                    player_id INTEGER NOT NULL,
                    total_points NUMERIC,
                    minutes NUMERIC,
                    clean_sheets NUMERIC,
                    goals_scored NUMERIC,
                    assists NUMERIC,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, gameweek)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS midfielder_history (
                    id SERIAL PRIMARY KEY,
                    season VARCHAR(9) NOT NULL,
                    gameweek INTEGER NOT NULL,
                    player_id INTEGER NOT NULL,
                    total_points NUMERIC,
                    minutes NUMERIC,
                    goals_scored NUMERIC,
                    assists NUMERIC,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, gameweek)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS forward_history (
                    id SERIAL PRIMARY KEY,
                    season VARCHAR(9) NOT NULL,
                    gameweek INTEGER NOT NULL,
                    player_id INTEGER NOT NULL,
                    total_points NUMERIC,
                    minutes NUMERIC,
                    goals_scored NUMERIC,
                    assists NUMERIC,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, gameweek)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS player_gameweek_history (
                    id SERIAL PRIMARY KEY,
                    player_id INTEGER NOT NULL,
                    season VARCHAR(9) NOT NULL,
                    gameweek INTEGER NOT NULL,
                    total_points INTEGER,
                    minutes_played INTEGER,
                    goals_scored INTEGER,
                    assists INTEGER,
                    clean_sheet BOOLEAN,
                    bonus_points INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player_id, season, gameweek)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS bootstrap_static (
                    id SERIAL PRIMARY KEY,
                    data JSONB NOT NULL,
                    season VARCHAR(9) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    UNIQUE(season)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS fixtures (
                    id SERIAL PRIMARY KEY,
                    data JSONB NOT NULL,
                    season VARCHAR(9) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    UNIQUE(season)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS player_data (
                    id SERIAL PRIMARY KEY,
                    player_id INTEGER NOT NULL,
                    season VARCHAR(9) NOT NULL,
                    data JSONB NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    UNIQUE(player_id, season)
                )
                """
            ]
            
            # Execute each create table query
            for query in create_table_queries:
                execute_query(query, fetch=False)
            
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            return False
    
    def store_player_gameweek_history(self, player_data: List[Dict[str, Any]]) -> None:
        """
        Store detailed player gameweek performance.
        
        Args:
            player_data: List of player performance dictionaries
        """
        try:
            # Get current season
            current_season = self.get_current_season()
            
            # Create SQLAlchemy engine
            engine = sa.create_engine(
                f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
            )
            
            # Prepare data for bulk insert with upsert logic
            processed_data = []
            for entry in player_data:
                processed_entry = {
                    'player_id': entry.get('player_id'),
                    'season': current_season,
                    'gameweek': entry.get('round'),
                    'total_points': entry.get('total_points'),
                    'minutes_played': entry.get('minutes'),
                    'goals_scored': entry.get('goals_scored', 0),
                    'assists': entry.get('assists', 0),
                    'clean_sheet': entry.get('clean_sheets', 0) > 0,
                    'bonus_points': entry.get('bonus', 0)
                }
                processed_data.append(processed_entry)
            
            # Create table if not exists
            metadata = sa.MetaData()
            table = sa.Table(
                'player_gameweek_history', 
                metadata,
                sa.Column('id', sa.Integer, primary_key=True),
                sa.Column('player_id', sa.Integer, nullable=False),
                sa.Column('season', sa.String(9), nullable=False),
                sa.Column('gameweek', sa.Integer, nullable=False),
                sa.Column('total_points', sa.Integer),
                sa.Column('minutes_played', sa.Integer),
                sa.Column('goals_scored', sa.Integer),
                sa.Column('assists', sa.Integer),
                sa.Column('clean_sheet', sa.Boolean),
                sa.Column('bonus_points', sa.Integer),
                sa.Column('timestamp', sa.DateTime, default=sa.func.now()),
                sa.UniqueConstraint('player_id', 'season', 'gameweek')
            )
            
            # Create table if not exists
            metadata.create_all(engine)
            
            # Upsert operation
            with engine.connect() as conn:
                for entry in processed_data:
                    # Create an INSERT statement
                    stmt = insert(table).values(entry)
                    
                    # Create an ON CONFLICT (UPSERT) statement
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=['player_id', 'season', 'gameweek'],
                        set_={
                            'total_points': stmt.excluded.total_points,
                            'minutes_played': stmt.excluded.minutes_played,
                            'goals_scored': stmt.excluded.goals_scored,
                            'assists': stmt.excluded.assists,
                            'clean_sheet': stmt.excluded.clean_sheet,
                            'bonus_points': stmt.excluded.bonus_points
                        }
                    )
                    
                    # Execute the upsert
                    conn.execute(upsert_stmt)
            
            logger.info(f"Stored {len(processed_data)} player gameweek records")
        
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error storing player gameweek history: {e}")
        except Exception as e:
            logger.error(f"Error storing player gameweek history: {e}")
    
    def _store_position_data(self, position: str, data: pd.DataFrame) -> None:
        """
        Store position data in database with seasonal context.
        
        Args:
            position: Position code (GK, DEF, MID, FWD)
            data: DataFrame of player data
        """
        try:
            # Determine current season
            current_season = self.get_current_season()
            
            # Create SQLAlchemy engine
            engine = sa.create_engine(
                f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
            )
            
            # Prepare data for bulk insert
            table_name = f"{position.lower()}_history"
            
            # Add seasonal and gameweek context
            data['season'] = current_season
            data['timestamp'] = datetime.now()
            
            try:
                # Write DataFrame to PostgreSQL with upsert logic
                with engine.begin() as conn:
                    # Use pandas to_sql with if_exists='append'
                    data.to_sql(
                        table_name, 
                        conn, 
                        if_exists='append',  # Append new data
                        index=False,
                        method='multi'  # Batch insert for performance
                    )
                
                logger.info(f"{position} position data stored in database")
                
            except Exception as e:
                logger.error(f"Error storing {position} data to database: {e}")
                # Fall back to file storage
                file_path = os.path.join(settings.PROCESSED_DATA_DIR, f'{position.lower()}_history.csv')
                data.to_csv(file_path, index=False)
                logger.info(f"{position} data stored in file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error setting up database storage for {position}: {e}")
            # Fall back to file storage
            file_path = os.path.join(settings.PROCESSED_DATA_DIR, f'{position.lower()}_history.csv')
            data.to_csv(file_path, index=False)
            logger.info(f"{position} data stored in file: {file_path}")
    
    def fetch_bootstrap_static(self) -> Dict:
        """
        Fetch the main FPL API data including teams, players, and gameweeks.
        
        Returns:
            Dict: The bootstrap static data
        """
        try:
            logger.info("Fetching bootstrap static data from FPL API")
            response = requests.get(settings.FPL_BOOTSTRAP_URL)
            response.raise_for_status()
            self.bootstrap_data = response.json()
            
            # Determine current gameweek
            self._determine_current_gameweek()
            
            # Store bootstrap data
            self._store_bootstrap_data()
            
            return self.bootstrap_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching bootstrap static: {e}")
            raise

    def fetch_fixtures(self) -> List[Dict]:
        """
        Fetch all fixtures data from the FPL API.
        
        Returns:
            List[Dict]: List of fixture data
        """
        try:
            logger.info("Fetching fixtures data from FPL API")
            response = requests.get(settings.FPL_FIXTURES_URL)
            response.raise_for_status()
            self.fixtures_data = response.json()
            
            # Store fixtures data
            self._store_fixtures_data()
            
            return self.fixtures_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching fixtures: {e}")
            raise

    def _store_bootstrap_data(self) -> None:
        """
        Store bootstrap static data in the database.
        """
        try:
            # Get current season
            current_season = self.get_current_season()
            
            # Convert to JSON string
            data_json = json.dumps(self.bootstrap_data)
            
            # Use direct SQL for insert to avoid metadata issues
            insert_query = """
            INSERT INTO bootstrap_static (data, season, timestamp) 
            VALUES (%s, %s, NOW()) 
            ON CONFLICT (season) 
            DO UPDATE SET data = EXCLUDED.data, timestamp = NOW()
            """
            
            execute_query(insert_query, (data_json, current_season), fetch=False)
            
            logger.info("Bootstrap data stored in database")
            
        except Exception as e:
            logger.error(f"Error storing bootstrap data: {e}")
            # Fall back to file storage
            file_path = os.path.join(settings.RAW_DATA_DIR, 'bootstrap_static.json')
            with open(file_path, 'w') as f:
                json.dump(self.bootstrap_data, f, indent=2)
            logger.info(f"Bootstrap data stored in file: {file_path}")

    def _store_fixtures_data(self) -> None:
        """
        Store fixtures data in the database.
        """
        try:
            # Get current season
            current_season = self.get_current_season()
            
            # Convert to JSON string
            data_json = json.dumps(self.fixtures_data)
            
            # Use direct SQL for insert to avoid metadata issues
            insert_query = """
            INSERT INTO fixtures (data, season, timestamp) 
            VALUES (%s, %s, NOW()) 
            ON CONFLICT (season) 
            DO UPDATE SET data = EXCLUDED.data, timestamp = NOW()
            """
            
            execute_query(insert_query, (data_json, current_season), fetch=False)
            
            logger.info("Fixtures data stored in database")
            
        except Exception as e:
            logger.error(f"Error storing fixtures data: {e}")
            # Fall back to file storage
            file_path = os.path.join(settings.RAW_DATA_DIR, 'fixtures.json')
            with open(file_path, 'w') as f:
                json.dump(self.fixtures_data, f, indent=2)
            logger.info(f"Fixtures data stored in file: {file_path}")

    def _determine_current_gameweek(self) -> int:
        """
        Determine the current or next gameweek from bootstrap data.
        """
        if not self.bootstrap_data:
            return None
            
        events = self.bootstrap_data['events']
        
        # First try to find the current gameweek
        for event in events:
            if event['is_current']:
                self.current_gw = event['id']
                logger.info(f"Current gameweek: {self.current_gw}")
                return self.current_gw
        
        # If no current gameweek, find the next one
        for event in events:
            if event['is_next']:
                self.current_gw = event['id']
                logger.info(f"Next gameweek: {self.current_gw}")
                return self.current_gw
        
        # If neither found, use the first gameweek
        self.current_gw = events[0]['id'] if events else 1
        logger.info(f"Using gameweek: {self.current_gw}")
        return self.current_gw
    
    def get_current_gameweek(self) -> int:
        """
        Get the current gameweek.
        
        Returns:
            int: Current gameweek number
        """
        if self.current_gw is None:
            if self.bootstrap_data is None:
                self.fetch_bootstrap_static()
            else:
                self._determine_current_gameweek()
                
        return self.current_gw
    
    def collect_player_data_by_position(self, position_id: int) -> pd.DataFrame:
        """
        Collect player data for a specific position.
        
        Args:
            position_id: Position ID (1=GK, 2=DEF, 3=MID, 4=FWD)
            
        Returns:
            pd.DataFrame: Player data for the position
        """
        try:
            # Make sure bootstrap data is loaded
            if self.bootstrap_data is None:
                self.fetch_bootstrap_static()
                
            # Filter players by position
            position_players = [
                player for player in self.bootstrap_data['elements'] 
                if player['element_type'] == position_id
            ]
            
            if not position_players:
                logger.warning(f"No players found for position ID {position_id}")
                return pd.DataFrame()
                
            # Convert to DataFrame
            position_df = pd.DataFrame(position_players)
            
            # Add team information
            teams_dict = {
                team['id']: team['name'] 
                for team in self.bootstrap_data['teams']
            }
            
            position_df['team_name'] = position_df['team'].map(teams_dict)
            
            # Rename columns for clarity
            position_df = position_df.rename(columns={
                'id': 'player_id',
                'web_name': 'player_name',
                'now_cost': 'cost'
            })
            
            # Convert cost to actual value (in millions)
            position_df['cost'] = position_df['cost'] / 10.0
            
            logger.info(f"Collected data for {len(position_df)} players in position {position_id}")
            return position_df
            
        except Exception as e:
            logger.error(f"Error collecting player data for position {position_id}: {e}")
            return pd.DataFrame()
    
    def collect_all_position_data(self) -> Dict[str, pd.DataFrame]:
        """
        Collect data for all positions.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of position data
        """
        # First fetch bootstrap if not already done
        if self.bootstrap_data is None:
            db_data = self._load_bootstrap_data_from_db()
            if db_data:
                self.bootstrap_data = db_data
            else:
                self.fetch_bootstrap_static()
            
        # Fetch fixtures if not already done
        if self.fixtures_data is None:
            self.fetch_fixtures()
            
        # Collect data for each position
        position_data = {}
        position_ids = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        for position_id, position_name in position_ids.items():
            logger.info(f"Collecting data for {position_name} position")
            position_df = self.collect_player_data_by_position(position_id)
            
            # Store position-specific data in database
            if not position_df.empty:
                self._store_position_data(position_name, position_df)
            
            position_data[position_name] = position_df
            
        return position_data