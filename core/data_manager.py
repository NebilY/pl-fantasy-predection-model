"""
Data Manager for FPL Prediction System.

This module handles the collection and storage of Fantasy Premier League data,
with support for database storage and caching mechanisms.
"""

import requests
import pandas as pd
import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

# Import settings and database connection
from config import settings
from config.database import execute_query, get_connection, release_connection

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
            
            # Store data in database
            self._store_bootstrap_data()
            
            # Determine current gameweek
            self._determine_current_gameweek()
            
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
            
            # Store fixtures in database
            self._store_fixtures_data()
            
            return self.fixtures_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching fixtures: {e}")
            raise
    
    def fetch_player_history(self, player_id: int) -> Dict:
        """
        Fetch detailed history data for a specific player.
        
        Args:
            player_id: The FPL ID of the player
            
        Returns:
            Dict: Player history data
        """
        try:
            logger.info(f"Fetching history for player {player_id}")
            url = f"{settings.FPL_PLAYER_SUMMARY_URL}{player_id}/"
            response = requests.get(url)
            response.raise_for_status()
            player_data = response.json()
            
            # Store player data in database
            self._store_player_data(player_id, player_data)
            
            return player_data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching player {player_id} history: {e}")
            raise
    
    def get_players_by_position(self, position_id: int) -> pd.DataFrame:
        """
        Get all players of a specific position.
        
        Args:
            position_id: Position ID (1=GK, 2=DEF, 3=MID, 4=FWD)
            
        Returns:
            pd.DataFrame: DataFrame of players
        """
        if self.bootstrap_data is None:
            # Try to fetch from database first
            db_data = self._load_bootstrap_data_from_db()
            if db_data:
                self.bootstrap_data = db_data
            else:
                # If not in database, fetch from API
                self.fetch_bootstrap_static()
        
        # Filter players by position
        players = [p for p in self.bootstrap_data['elements'] if p['element_type'] == position_id]
        return pd.DataFrame(players)
    
    def collect_player_data_by_position(self, position_id: int) -> pd.DataFrame:
        """
        Collect detailed data for all players in a position.
        
        Args:
            position_id: Position ID (1=GK, 2=DEF, 3=MID, 4=FWD)
            
        Returns:
            pd.DataFrame: Complete player data with history
        """
        # Check if data already exists in database
        position_names = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        position_name = position_names.get(position_id, 'UNK')
        
        # Check if we already have recent data for this position
        existing_data = self._load_position_data_from_db(position_name)
        if existing_data is not None:
            logger.info(f"Using existing {position_name} data from database")
            return existing_data
        
        # Get players of this position
        players_df = self.get_players_by_position(position_id)
        
        # Team mapping
        team_map = {team['id']: team['name'] for team in self.bootstrap_data['teams']}
        
        # Collect history for each player
        player_histories = []
        
        for _, player in players_df.iterrows():
            player_id = player['id']
            
            try:
                # Check if player data exists in database
                db_player_data = self._load_player_data_from_db(player_id)
                
                if db_player_data:
                    player_data = db_player_data
                else:
                    # Get player history from API
                    player_data = self.fetch_player_history(player_id)
                
                # Process current season history
                if 'history' in player_data and player_data['history']:
                    for match in player_data['history']:
                        match_data = match.copy()
                        # Add player details
                        match_data['player_id'] = player_id
                        match_data['player_name'] = player['web_name']
                        match_data['team_id'] = player['team']
                        match_data['team_name'] = team_map.get(player['team'], "Unknown")
                        match_data['position'] = position_name
                        match_data['cost'] = player['now_cost'] / 10
                        
                        player_histories.append(match_data)
            
            except Exception as e:
                logger.error(f"Error processing player {player_id}: {e}")
                continue
        
        # Convert to DataFrame
        if player_histories:
            history_df = pd.DataFrame(player_histories)
            
            # Store processed data in database
            self._store_position_data(position_name, history_df)
            
            return history_df
        else:
            logger.warning(f"No history data collected for position {position_name}")
            return pd.DataFrame()
    
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
    
    def _store_bootstrap_data(self) -> None:
        """Store bootstrap data in database."""
        try:
            # Convert to JSON string
            data_json = json.dumps(self.bootstrap_data)
            
            # Check if data already exists
            check_query = "SELECT COUNT(*) FROM bootstrap_static WHERE timestamp::date = CURRENT_DATE"
            result = execute_query(check_query)
            
            if result[0][0] == 0:
                # Insert new data
                insert_query = """
                INSERT INTO bootstrap_static (data, timestamp)
                VALUES (%s, %s)
                """
                execute_query(insert_query, (data_json, datetime.now()), fetch=False)
                logger.info("Bootstrap data stored in database")
            else:
                # Update existing data
                update_query = """
                UPDATE bootstrap_static 
                SET data = %s, timestamp = %s
                WHERE timestamp::date = CURRENT_DATE
                """
                execute_query(update_query, (data_json, datetime.now()), fetch=False)
                logger.info("Bootstrap data updated in database")
                
        except Exception as e:
            logger.error(f"Error storing bootstrap data: {e}")
            # Fall back to file storage
            file_path = os.path.join(settings.RAW_DATA_DIR, 'bootstrap_static.json')
            with open(file_path, 'w') as f:
                json.dump(self.bootstrap_data, f, indent=2)
            logger.info(f"Bootstrap data stored in file: {file_path}")
    
    def _store_fixtures_data(self) -> None:
        """Store fixtures data in database."""
        try:
            # Convert to JSON string
            data_json = json.dumps(self.fixtures_data)
            
            # Check if data already exists
            check_query = "SELECT COUNT(*) FROM fixtures WHERE timestamp::date = CURRENT_DATE"
            result = execute_query(check_query)
            
            if result[0][0] == 0:
                # Insert new data
                insert_query = """
                INSERT INTO fixtures (data, timestamp)
                VALUES (%s, %s)
                """
                execute_query(insert_query, (data_json, datetime.now()), fetch=False)
                logger.info("Fixtures data stored in database")
            else:
                # Update existing data
                update_query = """
                UPDATE fixtures 
                SET data = %s, timestamp = %s
                WHERE timestamp::date = CURRENT_DATE
                """
                execute_query(update_query, (data_json, datetime.now()), fetch=False)
                logger.info("Fixtures data updated in database")
                
        except Exception as e:
            logger.error(f"Error storing fixtures data: {e}")
            # Fall back to file storage
            file_path = os.path.join(settings.RAW_DATA_DIR, 'fixtures.json')
            with open(file_path, 'w') as f:
                json.dump(self.fixtures_data, f, indent=2)
            logger.info(f"Fixtures data stored in file: {file_path}")
    
    def _store_player_data(self, player_id: int, player_data: Dict) -> None:
        """
        Store player data in database.
        
        Args:
            player_id: Player ID
            player_data: Player data dictionary
        """
        try:
            # Convert to JSON string
            data_json = json.dumps(player_data)
            
            # Check if data already exists
            check_query = "SELECT COUNT(*) FROM player_data WHERE player_id = %s AND timestamp::date = CURRENT_DATE"
            result = execute_query(check_query, (player_id,))
            
            if result[0][0] == 0:
                # Insert new data
                insert_query = """
                INSERT INTO player_data (player_id, data, timestamp)
                VALUES (%s, %s, %s)
                """
                execute_query(insert_query, (player_id, data_json, datetime.now()), fetch=False)
                logger.info(f"Player {player_id} data stored in database")
            else:
                # Update existing data
                update_query = """
                UPDATE player_data 
                SET data = %s, timestamp = %s
                WHERE player_id = %s AND timestamp::date = CURRENT_DATE
                """
                execute_query(update_query, (data_json, datetime.now(), player_id), fetch=False)
                logger.info(f"Player {player_id} data updated in database")
                
        except Exception as e:
            logger.error(f"Error storing player {player_id} data: {e}")
            # Fall back to file storage
            file_path = os.path.join(settings.RAW_DATA_DIR, f'player_{player_id}.json')
            with open(file_path, 'w') as f:
                json.dump(player_data, f, indent=2)
            logger.info(f"Player {player_id} data stored in file: {file_path}")
    
    def _store_position_data(self, position: str, data: pd.DataFrame) -> None:
        """
        Store position data in database.
        
        Args:
            position: Position code (GK, DEF, MID, FWD)
            data: DataFrame of player data
        """
        try:
            conn = get_connection()
            try:
                # Create temporary table
                temp_table = f"temp_{position.lower()}_history"
                table_name = f"{position.lower()}_history"
                
                # Drop temp table if exists
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
                
                # Write DataFrame to temporary table
                data.to_sql(temp_table, conn, index=False, if_exists='replace')
                
                # Check if main table exists
                cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = '{table_name}'
                )
                """)
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    # Create main table from temp
                    cursor.execute(f"""
                    CREATE TABLE {table_name} AS
                    SELECT *, CURRENT_TIMESTAMP AS timestamp
                    FROM {temp_table}
                    """)
                else:
                    # Delete current day's data
                    cursor.execute(f"""
                    DELETE FROM {table_name}
                    WHERE timestamp::date = CURRENT_DATE
                    """)
                    
                    # Insert new data
                    cursor.execute(f"""
                    INSERT INTO {table_name}
                    SELECT *, CURRENT_TIMESTAMP AS timestamp
                    FROM {temp_table}
                    """)
                
                # Drop temp table
                cursor.execute(f"DROP TABLE IF EXISTS {temp_table}")
                
                # Commit changes
                conn.commit()
                logger.info(f"{position} position data stored in database")
                
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                release_connection(conn)
                
        except Exception as e:
            logger.error(f"Error storing {position} data: {e}")
            # Fall back to file storage
            file_path = os.path.join(settings.PROCESSED_DATA_DIR, f'{position.lower()}_history.csv')
            data.to_csv(file_path, index=False)
            logger.info(f"{position} data stored in file: {file_path}")
    
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
            WHERE timestamp::date = CURRENT_DATE
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            result = execute_query(query)
            
            if result and result[0]:
                logger.info("Bootstrap data loaded from database")
                return json.loads(result[0][0])
            return None
            
        except Exception as e:
            logger.error(f"Error loading bootstrap data from database: {e}")
            return None
    
    def _load_player_data_from_db(self, player_id: int) -> Optional[Dict]:
        """
        Load player data from database.
        
        Args:
            player_id: Player ID
            
        Returns:
            Optional[Dict]: Player data or None if not found
        """
        try:
            query = """
            SELECT data 
            FROM player_data 
            WHERE player_id = %s AND timestamp::date = CURRENT_DATE
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            result = execute_query(query, (player_id,))
            
            if result and result[0]:
                logger.info(f"Player {player_id} data loaded from database")
                return json.loads(result[0][0])
            return None
            
        except Exception as e:
            logger.error(f"Error loading player data from database: {e}")
            return None
    
    def _load_position_data_from_db(self, position: str) -> Optional[pd.DataFrame]:
        """
        Load position data from database.
        
        Args:
            position: Position code (GK, DEF, MID, FWD)
            
        Returns:
            Optional[pd.DataFrame]: Position data or None if not found
        """
        try:
            conn = get_connection()
            table_name = f"{position.lower()}_history"
            
            # Check if table exists and has today's data
            check_query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            )
            """
            
            result = execute_query(check_query)
            
            if result[0][0]:
                # Table exists, check for today's data
                check_today_query = f"""
                SELECT COUNT(*) 
                FROM {table_name}
                WHERE timestamp::date = CURRENT_DATE
                """
                
                today_result = execute_query(check_today_query)
                
                if today_result[0][0] > 0:
                    # Data exists for today
                    query = f"""
                    SELECT * 
                    FROM {table_name}
                    WHERE timestamp::date = CURRENT_DATE
                    """
                    
                    # Use pandas to read from database
                    df = pd.read_sql(query, conn)
                    logger.info(f"{position} data loaded from database")
                    return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading {position} data from database: {e}")
            return None
        finally:
            release_connection(conn)
    
    def get_current_gameweek(self) -> int:
        """
        Get the current gameweek number.
        
        Returns:
            int: Current gameweek number
        """
        if self.current_gw is None:
            if self.bootstrap_data is None:
                db_data = self._load_bootstrap_data_from_db()
                if db_data:
                    self.bootstrap_data = db_data
                else:
                    self.fetch_bootstrap_static()
            self._determine_current_gameweek()
            
        return self.current_gw
    
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
            position_data[position_name] = self.collect_player_data_by_position(position_id)
            
        return position_data
    
    def setup_database_tables(self):
        """Create necessary database tables if they don't exist."""
        try:
            # Create bootstrap_static table
            bootstrap_query = """
            CREATE TABLE IF NOT EXISTS bootstrap_static (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
            """
            execute_query(bootstrap_query, fetch=False)
            
            # Create fixtures table
            fixtures_query = """
            CREATE TABLE IF NOT EXISTS fixtures (
                id SERIAL PRIMARY KEY,
                data JSONB NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
            """
            execute_query(fixtures_query, fetch=False)
            
            # Create player_data table
            player_query = """
            CREATE TABLE IF NOT EXISTS player_data (
                id SERIAL PRIMARY KEY,
                player_id INTEGER NOT NULL,
                data JSONB NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
            """
            execute_query(player_query, fetch=False)
            
            logger.info("Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            return False