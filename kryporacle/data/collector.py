import logging
from typing import Dict, List, Optional
import requests
import time
from datetime import datetime

from ..config import settings

logger = logging.getLogger(__name__)

class DataCollector:
    """Base class for collecting cryptocurrency market data."""
    
    def __init__(self, exchange_id: str):
        """Initialize data collector for a specific exchange.
        
        Args:
            exchange_id (str): The ID of the exchange to collect data from.
        """
        self.exchange_id = exchange_id
        self.config = settings.EXCHANGE_CONFIGS.get(exchange_id)
        
        if not self.config:
            raise ValueError(f"Configuration for exchange '{exchange_id}' not found")
        
        if not self.config.get("enabled"):
            logger.warning(f"Exchange {exchange_id} is disabled in configuration")
    
    def fetch_historical_data(self, symbol: str, timeframe: str, 
                             start_time: Optional[datetime] = None, 
                             end_time: Optional[datetime] = None) -> Dict:
        """Fetch historical market data for a specific symbol and timeframe.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Candle timeframe (e.g., '1m', '1h', '1d')
            start_time (datetime, optional): Start time for data collection
            end_time (datetime, optional): End time for data collection
            
        Returns:
            Dict: Dictionary containing the historical market data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data for a specific symbol.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dict: Dictionary containing the current ticker data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @staticmethod
    def _make_request(url: str, params: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, auth=None) -> Dict:
        """Helper method to make API requests with error handling.
        
        Args:
            url (str): The API endpoint URL
            params (Dict, optional): Query parameters for the request
            headers (Dict, optional): HTTP headers for the request
            auth (object, optional): Authentication handler
            
        Returns:
            Dict: The JSON response from the API
        """
        try:
            response = requests.get(url, params=params, headers=headers, auth=auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {e}")
            raise