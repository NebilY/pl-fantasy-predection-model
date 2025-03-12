# core/utils.py
import os
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional

class LoggerManager:
    """
    Centralized logging management for the FPL prediction system.
    """
    @staticmethod
    def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
        """
        Create a configured logger for different components.
        
        Args:
            name (str): Name of the logger
            log_level (int): Logging level
        
        Returns:
            logging.Logger: Configured logger
        """
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create file handler
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        )
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger

class ConfigManager:
    """
    Manages configuration loading and validation.
    """
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML file.
        
        Args:
            config_path (str, optional): Path to config file
        
        Returns:
            Dict: Loaded configuration
        """
        # Default to config.json in project root if no path provided
        if not config_path:
            config_path = os.path.join(os.getcwd(), 'config.json')
        
        try:
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            # Return default configuration if file not found
            return {
                'debug': False,
                'log_level': 'INFO',
                'api_endpoints': {
                    'fpl_base': 'https://fantasy.premierleague.com/api'
                }
            }
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON configuration in {config_path}")

class DataValidator:
    """
    Provides data validation utilities.
    """
    @staticmethod
    def validate_player_data(player_data: Dict[str, Any]) -> bool:
        """
        Validate player data dictionary.
        
        Args:
            player_data (Dict): Player data to validate
        
        Returns:
            bool: Whether data is valid
        """
        required_fields = [
            'player_id', 
            'player_name', 
            'position', 
            'predicted_points'
        ]
        
        return all(
            field in player_data and player_data[field] is not None 
            for field in required_fields
        )

class FileManager:
    """
    Utility functions for file management.
    """
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory (str): Path to directory
        """
        os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def save_json(data: Any, filepath: str) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data (Any): Data to save
            filepath (str): Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    
    @staticmethod
    def load_json(filepath: str) -> Any:
        """
        Load data from a JSON file.
        
        Args:
            filepath (str): Path to JSON file
        
        Returns:
            Any: Loaded data
        """
        with open(filepath, 'r') as f:
            return json.load(f)

class PerformanceTimer:
    """
    Utility for timing and profiling code execution.
    """
    @staticmethod
    def time_function(func):
        """
        Decorator to time function execution.
        
        Args:
            func (callable): Function to time
        
        Returns:
            callable: Wrapped function with timing
        """
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            
            logger = LoggerManager.get_logger('performance')
            logger.info(
                f"Function {func.__name__} execution time: "
                f"{(end_time - start_time).total_seconds()} seconds"
            )
            
            return result
        return wrapper

# Example usage
if __name__ == "__main__":
    # Demonstrate logger usage
    logger = LoggerManager.get_logger('utils_demo')
    logger.info("Utility module demonstration")
    
    # Demonstrate config loading
    config = ConfigManager.load_config()
    print("Loaded Configuration:", config)
    
    # Demonstrate performance timing
    @PerformanceTimer.time_function
    def example_function():
        import time
        time.sleep(2)  # Simulate some work
    
    example_function()