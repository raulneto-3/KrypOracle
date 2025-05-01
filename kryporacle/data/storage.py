import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Union, List

from ..config import settings

class DataStorage:
    """Class for storing and retrieving cryptocurrency market data."""
    
    def __init__(self, base_path: str = None):
        """Initialize the data storage.
        
        Args:
            base_path (str, optional): Base directory path for data storage.
                                       Defaults to the path specified in settings.
        """
        self.base_path = Path(base_path or settings.DATA_STORAGE_PATH)
        self._ensure_storage_path()
    
    def _ensure_storage_path(self):
        """Create storage directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_data_path(self, exchange_id: str, symbol: str, timeframe: str) -> Path:
        """Get the path for storing data for a specific exchange, symbol and timeframe.
        
        Args:
            exchange_id (str): The exchange identifier
            symbol (str): The trading pair symbol
            timeframe (str): The data timeframe
            
        Returns:
            Path: Path object for the data file
        """
        # Convert symbol to a filename-safe format (e.g., BTC_USDT)
        safe_symbol = symbol.replace('/', '_')
        
        # Create directory structure
        directory = self.base_path / exchange_id / safe_symbol
        directory.mkdir(parents=True, exist_ok=True)
        
        return directory / f"{timeframe}.csv"
    
    def save_market_data(self, exchange_id: str, data: Dict):
        """Save market data to storage.
        
        Args:
            exchange_id (str): The exchange identifier
            data (Dict): Market data dictionary with symbol, timeframe, and data
        """
        symbol = data['symbol']
        timeframe = data['timeframe']
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data['data'])
        
        # Ensure timestamp is a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        file_path = self._get_data_path(exchange_id, symbol, timeframe)
        
        # If file exists, try to update it rather than overwrite
        if file_path.exists():
            existing_df = pd.read_csv(file_path)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['timestamp'])
            combined_df = combined_df.sort_values('timestamp')
            combined_df.to_csv(file_path, index=False)
        else:
            # New file
            df.sort_values('timestamp').to_csv(file_path, index=False)
    
    def load_market_data(self, exchange_id: str, symbol: str, timeframe: str,
                        start_time: pd.Timestamp = None, 
                        end_time: pd.Timestamp = None) -> pd.DataFrame:
        """Load market data from storage.
        
        Args:
            exchange_id (str): The exchange identifier
            symbol (str): The trading pair symbol
            timeframe (str): The data timeframe
            start_time (pd.Timestamp, optional): Filter data starting from this time
            end_time (pd.Timestamp, optional): Filter data ending at this time
            
        Returns:
            pd.DataFrame: DataFrame containing the requested data
        """
        file_path = self._get_data_path(exchange_id, symbol, timeframe)
        
        if not file_path.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply time filters if specified
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        
        return df