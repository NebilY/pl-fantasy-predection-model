"""
Feature Engineering for FPL Prediction System.

This module processes raw Fantasy Premier League data into features 
for machine learning models, with position-specific processing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any
import os
from datetime import datetime

# Import settings and other modules
from config import settings
from core.data_manager import DataManager

# Set up logging
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Processes raw FPL data into features for prediction models.
    Creates position-specific features for GK, DEF, MID, and FWD.
    """
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Initialize the Feature Engineering module.
        
        Args:
            data_manager: Optional DataManager instance to use for data
        """
        self.data_manager = data_manager if data_manager else DataManager()
        self.position_data = {}
        self.processed_data = {}
    
    def load_position_data(self, position: str, data: Optional[pd.DataFrame] = None) -> bool:
        """
        Load data for a specific position, either directly or via data manager.
        
        Args:
            position: Position code ('GK', 'DEF', 'MID', 'FWD')
            data: Optional DataFrame to use instead of loading from data manager
            
        Returns:
            bool: Success status
        """
        try:
            if position not in ['GK', 'DEF', 'MID', 'FWD']:
                logger.error(f"Invalid position: {position}")
                return False
                
            if data is not None:
                self.position_data[position] = data
                logger.info(f"Loaded {len(data)} {position} records from provided DataFrame")
                return True
                
            # Position ID mapping
            position_ids = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
            
            # Load data from data manager
            position_data = self.data_manager.collect_player_data_by_position(position_ids[position])
            
            if position_data.empty:
                logger.warning(f"No data found for position {position}")
                return False
                
            self.position_data[position] = position_data
            logger.info(f"Loaded {len(position_data)} {position} records from data manager")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {position} data: {e}")
            return False
    
    def create_goalkeeper_features(self, save: bool = True) -> pd.DataFrame:
        """
        Create goalkeeper-specific features.
        
        Args:
            save: Whether to save the processed data
            
        Returns:
            pd.DataFrame: Processed goalkeeper data
        """
        if 'GK' not in self.position_data:
            logger.warning("Goalkeeper data not loaded. Please load data first.")
            return pd.DataFrame()
        
        try:
            logger.info("Creating goalkeeper features...")
            
            # Make a copy to avoid modifying the original
            history_df = self.position_data['GK'].copy()
            
            # ------------------ BASIC FEATURES ------------------
            # Standard point scoring components for goalkeepers
            history_df['clean_sheet_points'] = history_df['clean_sheets'] * 4  # 4 points per clean sheet
            history_df['save_points'] = (history_df['saves'] / 3).apply(np.floor).astype(int)  # 1 point per 3 saves
            history_df['penalty_save_points'] = history_df['penalties_saved'] * 5 if 'penalties_saved' in history_df.columns else 0
            history_df['goals_conceded_points'] = -1 * (history_df['goals_conceded'] / 2).apply(np.floor).astype(int)  # -1 point per 2 goals conceded
            history_df['bonus_points'] = history_df['bonus']

            # ------------------ FORM FEATURES ------------------
            # Keep original names for compatibility
            history_df['minutes_trend'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            history_df['rolling_points'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())

            # Additional form windows
            history_df['rolling_points_5'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
            history_df['minutes_trend_5'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

            # Calculate minutes consistency (higher value = more consistent minutes)
            history_df['minutes_rolling_std'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).std())
            history_df['minutes_rolling_mean'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).mean())
            history_df['minutes_consistency'] = 1.0
            mask = history_df['minutes_rolling_mean'] > 0
            history_df.loc[mask, 'minutes_consistency'] = 1 - (history_df.loc[mask, 'minutes_rolling_std'] / 90)
            history_df['minutes_consistency'] = history_df['minutes_consistency'].clip(0, 1)

            # Value metrics
            history_df['value'] = history_df['total_points'] / history_df['cost']
            history_df['recent_value'] = history_df['rolling_points_5'] / history_df['cost']

            # ------------------ HOME/AWAY FEATURES ------------------
            # Home/Away performance
            history_df_home = history_df[history_df['was_home'] == True].groupby('player_id')['total_points'].mean().reset_index()
            history_df_home.columns = ['player_id', 'avg_points_home']
            history_df_away = history_df[history_df['was_home'] == False].groupby('player_id')['total_points'].mean().reset_index()
            history_df_away.columns = ['player_id', 'avg_points_away']

            # Home/Away clean sheet tendency
            history_df_home_cs = history_df[history_df['was_home'] == True].groupby('player_id')['clean_sheets'].mean().reset_index()
            history_df_home_cs.columns = ['player_id', 'home_clean_sheet_rate']
            history_df_away_cs = history_df[history_df['was_home'] == False].groupby('player_id')['clean_sheets'].mean().reset_index()
            history_df_away_cs.columns = ['player_id', 'away_clean_sheet_rate']

            # Home/Away saves
            history_df_home_saves = history_df[history_df['was_home'] == True].groupby('player_id')['saves'].mean().reset_index()
            history_df_home_saves.columns = ['player_id', 'home_saves_per_game']
            history_df_away_saves = history_df[history_df['was_home'] == False].groupby('player_id')['saves'].mean().reset_index()
            history_df_away_saves.columns = ['player_id', 'away_saves_per_game']

            # Join home/away features
            history_df = pd.merge(history_df, history_df_home, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_home_cs, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away_cs, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_home_saves, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away_saves, on='player_id', how='left')

            # ------------------ GOALKEEPER-SPECIFIC FEATURES ------------------
            # Save metrics
            history_df['saves_per_game'] = history_df['saves'].astype(float)
            history_df['recent_saves'] = history_df.groupby('player_id')['saves'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

            # Save points rate (how often they get save points)
            history_df['save_points_per_game'] = history_df['save_points'].astype(float)
            history_df['recent_save_points'] = history_df.groupby('player_id')['save_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

            # Clean sheet metrics
            history_df['recent_clean_sheets'] = history_df.groupby('player_id')['clean_sheets'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
            history_df['clean_sheet_rate'] = history_df.groupby('player_id')['clean_sheets'].transform(
                lambda x: x.mean())

            # Goals conceded metrics
            history_df['goals_conceded_per_game'] = history_df['goals_conceded'].astype(float)
            history_df['recent_goals_conceded'] = history_df.groupby('player_id')['goals_conceded'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

            # Penalty save metrics (if available)
            if 'penalties_saved' in history_df.columns:
                history_df['penalties_saved_per_game'] = history_df['penalties_saved'].astype(float)
                history_df['has_saved_penalty'] = (history_df['penalties_saved'] > 0).astype(int)
            else:
                history_df['penalties_saved_per_game'] = 0
                history_df['has_saved_penalty'] = 0

            # Bonus point metrics
            history_df['bonus_per_game'] = history_df['bonus'].astype(float)
            history_df['recent_bonus'] = history_df.groupby('player_id')['bonus'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())

            # Clean sheet streaks (consecutive clean sheets)
            if 'round' in history_df.columns:
                # Initialize streak column
                history_df['clean_sheet_streak'] = 0
                
                # Process each player separately
                for player_id in history_df['player_id'].unique():
                    player_data = history_df[history_df['player_id'] == player_id].sort_values('round')
                    
                    # Calculate streaks
                    streak = 0
                    for i, row in player_data.iterrows():
                        if row['clean_sheets'] > 0:
                            streak += 1
                        else:
                            streak = 0
                        history_df.at[i, 'clean_sheet_streak'] = streak

            # Estimate shots faced from saves and goals conceded
            history_df['estimated_shots_faced'] = history_df['saves'] + history_df['goals_conceded']
            history_df['save_percentage'] = history_df.apply(
                lambda row: row['saves'] / row['estimated_shots_faced'] if row['estimated_shots_faced'] > 0 else 0, axis=1)

            # Recent save percentage
            history_df['recent_save_percentage'] = history_df.groupby('player_id')['save_percentage'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

            # Points per 90 minutes
            history_df['points_per_90'] = 90 * history_df['total_points'] / history_df['minutes'].clip(lower=1)

            # ------------------ TEAM DEFENSIVE FEATURES ------------------
            # Create team performance metrics
            team_defense_metrics = {}
            if 'round' in history_df.columns:
                for team_id in history_df['team_id'].unique():
                    team_matches = history_df[history_df['team_id'] == team_id]
                    if 'round' in team_matches.columns:
                        team_matches = team_matches.sort_values('round')
                    
                    if not team_matches.empty:
                        team_defense_metrics[team_id] = {}
                        
                        # Recent clean sheets
                        team_defense_metrics[team_id]['recent_clean_sheets'] = team_matches['clean_sheets'].tail(5).sum() if len(team_matches) >= 5 else 0
                        
                        # Recent goals conceded
                        team_defense_metrics[team_id]['recent_goals_conceded'] = team_matches['goals_conceded'].tail(5).sum() if len(team_matches) >= 5 else 0

            # Add team metrics to player data
            if team_defense_metrics:
                history_df['team_recent_clean_sheets'] = history_df['team_id'].map({k: v.get('recent_clean_sheets', 0) for k, v in team_defense_metrics.items()})
                history_df['team_recent_goals_conceded'] = history_df['team_id'].map({k: v.get('recent_goals_conceded', 0) for k, v in team_defense_metrics.items()})
            else:
                history_df['team_recent_clean_sheets'] = 0
                history_df['team_recent_goals_conceded'] = 0

            # ------------------ SAVE PROCESSED DATA ------------------
            # Fill NaN values with appropriate defaults
            history_df = history_df.fillna({
                'avg_points_home': history_df['total_points'].mean(),
                'avg_points_away': history_df['total_points'].mean(),
                'home_clean_sheet_rate': 0.3,  # Conservative defaults
                'away_clean_sheet_rate': 0.2,
                'home_saves_per_game': 3,
                'away_saves_per_game': 3,
                'minutes_consistency': 0,
                'team_recent_clean_sheets': 1,
                'team_recent_goals_conceded': 5,
                'save_percentage': 0.66,  # Average save percentage
                'recent_save_percentage': 0.66,
                'clean_sheet_streak': 0,
                'has_saved_penalty': 0
            })
            
            # Store processed data
            self.processed_data['GK'] = history_df
            
            # Save to file if requested
            if save:
                output_path = os.path.join(settings.PROCESSED_DATA_DIR, 'goalkeeper_processed.csv')
                history_df.to_csv(output_path, index=False)
                logger.info(f"Saved processed goalkeeper data to {output_path}")
            
            logger.info(f"Created {len(history_df.columns)} goalkeeper features")
            return history_df
            
        except Exception as e:
            logger.error(f"Error creating goalkeeper features: {e}")
            return pd.DataFrame()
    
    def create_defender_features(self, save: bool = True) -> pd.DataFrame:
        """
        Create defender-specific features.
        
        Args:
            save: Whether to save the processed data
            
        Returns:
            pd.DataFrame: Processed defender data
        """
        if 'DEF' not in self.position_data:
            logger.warning("Defender data not loaded. Please load data first.")
            return pd.DataFrame()
        
        try:
            logger.info("Creating defender features...")
            
            # Make a copy to avoid modifying the original
            history_df = self.position_data['DEF'].copy()
            
            # ------------------ BASE FEATURES ------------------
            # Standard point scoring components
            history_df['clean_sheet_points'] = history_df['clean_sheets'] * 4
            history_df['goals_points'] = history_df['goals_scored'] * 6
            history_df['assists_points'] = history_df['assists'] * 3
            history_df['bonus_points'] = history_df['bonus']

            # ------------------ FORM FEATURES ------------------
            # Recent form (3, 5, and 10 gameweek windows)
            history_df['minutes_trend'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            history_df['rolling_points'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            history_df['rolling_points_5'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
            history_df['minutes_trend_5'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

            # Calculate minutes consistency (higher value = more consistent minutes)
            history_df['minutes_rolling_std'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).std())
            history_df['minutes_rolling_mean'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).mean())
            history_df['minutes_consistency'] = 1.0
            mask = history_df['minutes_rolling_mean'] > 0
            history_df.loc[mask, 'minutes_consistency'] = 1 - (history_df.loc[mask, 'minutes_rolling_std'] / 90)
            history_df['minutes_consistency'] = history_df['minutes_consistency'].clip(0, 1)

            # Value metrics
            history_df['value'] = history_df['total_points'] / history_df['cost']
            history_df['recent_value'] = history_df['rolling_points_5'] / history_df['cost']

            # ------------------ HOME/AWAY FEATURES ------------------
            # Home/Away performance
            history_df_home = history_df[history_df['was_home'] == True].groupby('player_id')['total_points'].mean().reset_index()
            history_df_home.columns = ['player_id', 'avg_points_home']
            history_df_away = history_df[history_df['was_home'] == False].groupby('player_id')['total_points'].mean().reset_index()
            history_df_away.columns = ['player_id', 'avg_points_away']

            # Home/Away clean sheet tendency
            history_df_home_cs = history_df[history_df['was_home'] == True].groupby('player_id')['clean_sheets'].mean().reset_index()
            history_df_home_cs.columns = ['player_id', 'home_clean_sheet_rate']
            history_df_away_cs = history_df[history_df['was_home'] == False].groupby('player_id')['clean_sheets'].mean().reset_index()
            history_df_away_cs.columns = ['player_id', 'away_clean_sheet_rate']

            # Join home/away features
            history_df = pd.merge(history_df, history_df_home, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_home_cs, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away_cs, on='player_id', how='left')

            # ------------------ ATTACKING FEATURES ------------------
            # Basic attacking returns - recent goals and assists
            history_df['recent_goals'] = history_df.groupby('player_id')['goals_scored'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
            history_df['recent_assists'] = history_df.groupby('player_id')['assists'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
            history_df['attacking_returns'] = history_df['recent_goals'] + history_df['recent_assists']

            # Consistency of attacking returns
            history_df['goals_per_90'] = 90 * history_df['goals_scored'] / history_df['minutes'].clip(lower=1)
            history_df['assists_per_90'] = 90 * history_df['assists'] / history_df['minutes'].clip(lower=1)
            history_df['goal_involvement_per_90'] = history_df['goals_per_90'] + history_df['assists_per_90']

            # Bonus point trend
            history_df['bonus_trend'] = history_df.groupby('player_id')['bonus'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())

            # Fill NaN values with appropriate defaults
            history_df = history_df.fillna({
                'avg_points_home': history_df['total_points'].mean(),
                'avg_points_away': history_df['total_points'].mean(),
                'home_clean_sheet_rate': 0.3,  # Average clean sheet rate
                'away_clean_sheet_rate': 0.2,
                'minutes_consistency': 0,
            })
            
            # Store processed data
            self.processed_data['DEF'] = history_df
            
            # Save to file if requested
            if save:
                output_path = os.path.join(settings.PROCESSED_DATA_DIR, 'defender_processed.csv')
                history_df.to_csv(output_path, index=False)
                logger.info(f"Saved processed defender data to {output_path}")
            
            logger.info(f"Created {len(history_df.columns)} defender features")
            return history_df
            
        except Exception as e:
            logger.error(f"Error creating defender features: {e}")
            return pd.DataFrame()
    
    def create_midfielder_features(self, save: bool = True) -> pd.DataFrame:
        """
        Create midfielder-specific features.
        
        Args:
            save: Whether to save the processed data
            
        Returns:
            pd.DataFrame: Processed midfielder data
        """
        if 'MID' not in self.position_data:
            logger.warning("Midfielder data not loaded. Please load data first.")
            return pd.DataFrame()
        
        try:
            logger.info("Creating midfielder features...")
            
            # Make a copy to avoid modifying the original
            history_df = self.position_data['MID'].copy()
            
            # ------------------ BASE FEATURES ------------------
            # Standard point scoring components for midfielders
            history_df['goals_points'] = history_df['goals_scored'] * 5  # Midfielders get 5 points per goal
            history_df['assists_points'] = history_df['assists'] * 3
            history_df['clean_sheet_points'] = history_df['clean_sheets'] * 1  # Midfielders get 1 point per clean sheet
            history_df['bonus_points'] = history_df['bonus']

            # ------------------ FORM FEATURES ------------------
            # Recent form (3, 5, and 10 gameweek windows)
            history_df['minutes_trend'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            history_df['rolling_points'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            history_df['rolling_points_5'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
            history_df['minutes_trend_5'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

            # Calculate minutes consistency (higher value = more consistent minutes)
            history_df['minutes_rolling_std'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).std())
            history_df['minutes_rolling_mean'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).mean())
            history_df['minutes_consistency'] = 1.0
            mask = history_df['minutes_rolling_mean'] > 0
            history_df.loc[mask, 'minutes_consistency'] = 1 - (history_df.loc[mask, 'minutes_rolling_std'] / 90)
            history_df['minutes_consistency'] = history_df['minutes_consistency'].clip(0, 1)

            # Value metrics
            history_df['value'] = history_df['total_points'] / history_df['cost']
            history_df['recent_value'] = history_df['rolling_points_5'] / history_df['cost']

            # ------------------ HOME/AWAY FEATURES ------------------
            # Home/Away performance
            history_df_home = history_df[history_df['was_home'] == True].groupby('player_id')['total_points'].mean().reset_index()
            history_df_home.columns = ['player_id', 'avg_points_home']
            history_df_away = history_df[history_df['was_home'] == False].groupby('player_id')['total_points'].mean().reset_index()
            history_df_away.columns = ['player_id', 'avg_points_away']

            # Join home/away features
            history_df = pd.merge(history_df, history_df_home, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away, on='player_id', how='left')

            # ------------------ ATTACKING FEATURES ------------------
            # Basic attacking returns - recent goals and assists
            history_df['recent_goals'] = history_df.groupby('player_id')['goals_scored'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
            history_df['recent_assists'] = history_df.groupby('player_id')['assists'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
            history_df['attacking_returns'] = history_df['recent_goals'] + history_df['recent_assists']

            # Consistency of attacking returns
            history_df['goals_per_90'] = 90 * history_df['goals_scored'] / history_df['minutes'].clip(lower=1)
            history_df['assists_per_90'] = 90 * history_df['assists'] / history_df['minutes'].clip(lower=1)
            history_df['goal_involvement_per_90'] = history_df['goals_per_90'] + history_df['assists_per_90']

            # Add creativity and threat data if available (FPL-specific metrics)
            if 'creativity' in history_df.columns:
                history_df['recent_creativity'] = history_df.groupby('player_id')['creativity'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean())
            else:
                history_df['recent_creativity'] = 0

            if 'threat' in history_df.columns:
                history_df['recent_threat'] = history_df.groupby('player_id')['threat'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean())
            else:
                history_df['recent_threat'] = 0

            # Bonus point trend
            history_df['bonus_trend'] = history_df.groupby('player_id')['bonus'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())

            # Fill NaN values with appropriate defaults
            history_df = history_df.fillna({
                'avg_points_home': history_df['total_points'].mean(),
                'avg_points_away': history_df['total_points'].mean(),
                'minutes_consistency': 0,
                'recent_creativity': 0,
                'recent_threat': 0
            })
            
            # Store processed data
            self.processed_data['MID'] = history_df
            
            # Save to file if requested
            if save:
                output_path = os.path.join(settings.PROCESSED_DATA_DIR, 'midfielder_processed.csv')
                history_df.to_csv(output_path, index=False)
                logger.info(f"Saved processed midfielder data to {output_path}")
            
            logger.info(f"Created {len(history_df.columns)} midfielder features")
            return history_df
            
        except Exception as e:
            logger.error(f"Error creating midfielder features: {e}")
            return pd.DataFrame()
    
    def create_forward_features(self, save: bool = True) -> pd.DataFrame:
        """
        Create forward-specific features.
        
        Args:
            save: Whether to save the processed data
            
        Returns:
            pd.DataFrame: Processed forward data
        """
        if 'FWD' not in self.position_data:
            logger.warning("Forward data not loaded. Please load data first.")
            return pd.DataFrame()
        
        try:
            logger.info("Creating forward features...")
            
            # Make a copy to avoid modifying the original
            history_df = self.position_data['FWD'].copy()
            
            # ------------------ BASE FEATURES ------------------
            # Standard point scoring components for forwards
            history_df['goals_points'] = history_df['goals_scored'] * 4  # Forwards get 4 points per goal
            history_df['assists_points'] = history_df['assists'] * 3
            history_df['bonus_points'] = history_df['bonus']

            # Forward-specific penalties
            if 'penalties_scored' in history_df.columns:
                # Note: penalties already counted in goals, this is just for analysis
                history_df['penalties_scored_points'] = history_df['penalties_scored'] * 4
            else:
                history_df['penalties_scored_points'] = 0

            if 'penalties_missed' in history_df.columns:
                history_df['penalties_missed_points'] = history_df['penalties_missed'] * (-2)
            else:
                history_df['penalties_missed_points'] = 0

                    # ------------------ FORM FEATURES ------------------
               # Recent form (3, 5, and 10 gameweek windows)
            history_df['minutes_trend'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            history_df['rolling_points'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())
            history_df['rolling_points_5'] = history_df.groupby('player_id')['total_points'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
            history_df['minutes_trend_5'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
        
               # Calculate minutes consistency (higher value = more consistent minutes)
            history_df['minutes_rolling_std'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).std())
            history_df['minutes_rolling_mean'] = history_df.groupby('player_id')['minutes'].transform(
                lambda x: x.rolling(5, min_periods=3).mean())
            history_df['minutes_consistency'] = 1.0
            mask = history_df['minutes_rolling_mean'] > 0
            history_df.loc[mask, 'minutes_consistency'] = 1 - (history_df.loc[mask, 'minutes_rolling_std'] / 90)
            history_df['minutes_consistency'] = history_df['minutes_consistency'].clip(0, 1)
        
               # Value metrics
            history_df['value'] = history_df['total_points'] / history_df['cost']
            history_df['recent_value'] = history_df['rolling_points_5'] / history_df['cost']
        
               # ------------------ HOME/AWAY FEATURES ------------------
               # Home/Away performance
            history_df_home = history_df[history_df['was_home'] == True].groupby('player_id')['total_points'].mean().reset_index()
            history_df_home.columns = ['player_id', 'avg_points_home']
            history_df_away = history_df[history_df['was_home'] == False].groupby('player_id')['total_points'].mean().reset_index()
            history_df_away.columns = ['player_id', 'avg_points_away']
        
               # Home/Away goal scoring tendency
            history_df_home_goals = history_df[history_df['was_home'] == True].groupby('player_id')['goals_scored'].mean().reset_index()
            history_df_home_goals.columns = ['player_id', 'home_goals_per_game']
            history_df_away_goals = history_df[history_df['was_home'] == False].groupby('player_id')['goals_scored'].mean().reset_index()
            history_df_away_goals.columns = ['player_id', 'away_goals_per_game']
        
               # Join home/away features
            history_df = pd.merge(history_df, history_df_home, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_home_goals, on='player_id', how='left')
            history_df = pd.merge(history_df, history_df_away_goals, on='player_id', how='left')
        
               # ------------------ ATTACKING FEATURES ------------------
               # Basic attacking returns - recent goals and assists
            history_df['recent_goals'] = history_df.groupby('player_id')['goals_scored'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
            history_df['recent_assists'] = history_df.groupby('player_id')['assists'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
            history_df['attacking_returns'] = history_df['recent_goals'] + history_df['recent_assists']
        
               # Goal scoring streaks (consecutive games with goals)
               # This requires sorting data properly
            if 'round' in history_df.columns:
                # Initialize streak column
                history_df['goal_streak'] = 0
                   
                # Process each player separately
                for player_id in history_df['player_id'].unique():
                    player_data = history_df[history_df['player_id'] == player_id].sort_values('round')
                       
                    # Calculate streaks
                    streak = 0
                    for i, row in player_data.iterrows():
                        if row['goals_scored'] > 0:
                            streak += 1
                        else:
                            streak = 0
                        history_df.at[i, 'goal_streak'] = streak
        
               # Consistency of attacking returns
            history_df['goals_per_90'] = 90 * history_df['goals_scored'] / history_df['minutes'].clip(lower=1)
            history_df['assists_per_90'] = 90 * history_df['assists'] / history_df['minutes'].clip(lower=1)
            history_df['goal_involvement_per_90'] = history_df['goals_per_90'] + history_df['assists_per_90']
        
            # Add shots data if available
            if 'shots' in history_df.columns:
                history_df['shots_per_90'] = 90 * history_df['shots'] / history_df['minutes'].clip(lower=1)
                # Shot conversion rate
                history_df['shot_conversion'] = history_df.apply(
                    lambda row: row['goals_scored'] / row['shots'] if row['shots'] > 0 else 0, axis=1)
                # Recent shot metrics
                history_df['recent_shots'] = history_df.groupby('player_id')['shots'].transform(
                    lambda x: x.rolling(5, min_periods=1).sum())
                history_df['recent_shot_conversion'] = history_df.apply(
                    lambda row: row['recent_goals'] / row['recent_shots'] if row['recent_shots'] > 0 else 0, axis=1)
            else:
                history_df['shots_per_90'] = 0
                history_df['shot_conversion'] = 0
                history_df['recent_shots'] = 0
                history_df['recent_shot_conversion'] = 0
        
            # Add creativity and threat data if available (FPL-specific metrics)
            if 'creativity' in history_df.columns:
                history_df['recent_creativity'] = history_df.groupby('player_id')['creativity'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean())
            else:
                history_df['recent_creativity'] = 0
        
            if 'threat' in history_df.columns:
                history_df['recent_threat'] = history_df.groupby('player_id')['threat'].transform(
                    lambda x: x.rolling(5, min_periods=1).mean())
            else:
                history_df['recent_threat'] = 0
        
               # Bonus point trend
            history_df['bonus_trend'] = history_df.groupby('player_id')['bonus'].transform(
                lambda x: x.rolling(5, min_periods=1).sum())
        
            # Penalty duty
            if 'penalties_taken' in history_df.columns:
                history_df['penalty_taker'] = (history_df['penalties_taken'] > 0).astype(int)
            else:
                history_df['penalty_taker'] = 0
        
            # ------------------ ADVANCED METRICS ------------------
            # Expected goals and assists (if available)
            if 'expected_goals' in history_df.columns:
                history_df['xG_per_90'] = 90 * history_df['expected_goals'] / history_df['minutes'].clip(lower=1)
                history_df['goals_minus_xG'] = history_df['goals_scored'] - history_df['expected_goals']
                   # Recent xG
                history_df['recent_xG'] = history_df.groupby('player_id')['expected_goals'].transform(
                    lambda x: x.rolling(5, min_periods=1).sum())
            else:
                history_df['xG_per_90'] = 0
                history_df['goals_minus_xG'] = 0
                history_df['recent_xG'] = 0
        
            if 'expected_assists' in history_df.columns:
                history_df['xA_per_90'] = 90 * history_df['expected_assists'] / history_df['minutes'].clip(lower=1)
                history_df['assists_minus_xA'] = history_df['assists'] - history_df['expected_assists']
                # Recent xA
                history_df['recent_xA'] = history_df.groupby('player_id')['expected_assists'].transform(
                    lambda x: x.rolling(5, min_periods=1).sum())
            else:
                history_df['xA_per_90'] = 0
                history_df['assists_minus_xA'] = 0
                history_df['recent_xA'] = 0
        
            # Fill NaN values with appropriate defaults
            history_df = history_df.fillna({
                'avg_points_home': history_df['total_points'].mean(),
                'avg_points_away': history_df['total_points'].mean(),
                'home_goals_per_game': 0.2,  # Conservative defaults
                'away_goals_per_game': 0.15, 
                'minutes_consistency': 0,
                'shot_conversion': 0,
                'recent_shot_conversion': 0,
                'xG_per_90': 0,
                'xA_per_90': 0,
                'goals_minus_xG': 0,
                'assists_minus_xA': 0,
                'recent_xG': 0,
                'recent_xA': 0,
                'shots_per_90': 0,
                'recent_creativity': 0,
                'recent_threat': 0,
                'penalty_taker': 0,
                'goal_streak': 0
            })
               
            # Store processed data
            self.processed_data['FWD'] = history_df
               
               # Save to file if requested
            if save:
                output_path = os.path.join(settings.PROCESSED_DATA_DIR, 'forward_processed.csv')
                history_df.to_csv(output_path, index=False)
                logger.info(f"Saved processed forward data to {output_path}")
               
            logger.info(f"Created {len(history_df.columns)} forward features")
            return history_df
               
        except Exception as e:
            logger.error(f"Error creating forward features: {e}")
            return pd.DataFrame()
        
    def process_all_positions(self, save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Process all positions at once.
           
        Args:
            save: Whether to save the processed data
               
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of processed data by position
        """
        try:
            logger.info("Processing all positions...")
            result = {}
               
               # Load data for all positions if not already loaded
            if not self.position_data:
                   # Get data from data manager
                all_position_data = self.data_manager.collect_all_position_data()
                if not all_position_data:
                    logger.error("Failed to collect position data")
                    return {}
                   
                # Load each position
                for position, data in all_position_data.items():
                    self.load_position_data(position, data)
               
            # Process each position
            if 'GK' in self.position_data:
                result['GK'] = self.create_goalkeeper_features(save)
               
            if 'DEF' in self.position_data:
                result['DEF'] = self.create_defender_features(save)
               
            if 'MID' in self.position_data:
                result['MID'] = self.create_midfielder_features(save)
               
            if 'FWD' in self.position_data:
                result['FWD'] = self.create_forward_features(save)
               
            logger.info(f"Processed {len(result)} positions")
            return result
           
        except Exception as e:
            logger.error(f"Error processing all positions: {e}")
            return {}