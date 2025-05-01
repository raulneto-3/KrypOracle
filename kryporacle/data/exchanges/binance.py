import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, Optional, List
import urllib.parse

from ..collector import DataCollector
from ...config import settings

class BinanceCollector(DataCollector):
    """Data collector for the Binance exchange."""
    
    def __init__(self):
        """Initialize the Binance data collector."""
        super().__init__("binance")
        self.base_url = self.config["base_url"]
        self.api_key = self.config["api_key"]
        self.api_secret = self.config["api_secret"]
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate a signature for authenticated requests.
        
        Args:
            query_string (str): The query string to sign
            
        Returns:
            str: The HMAC SHA256 signature
        """
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def fetch_historical_data(self, symbol: str, timeframe: str,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict:
        """Fetch historical kline/candlestick data from Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Candle timeframe (e.g., '1m', '1h', '1d')
            start_time (datetime, optional): Start time for data collection
            end_time (datetime, optional): End time for data collection
            
        Returns:
            Dict: Dictionary containing the historical market data
        """
        # Convert symbol format from 'BTC/USDT' to 'BTCUSDT' for Binance
        formatted_symbol = symbol.replace('/', '')
        
        # Map timeframe to Binance interval format
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
        }
        
        if timeframe not in interval_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        endpoint = f"{self.base_url}/api/v3/klines"
        
        params = {
            'symbol': formatted_symbol,
            'interval': interval_map[timeframe],
            'limit': 1000  # Maximum allowed by Binance
        }
        
        # Add optional parameters if provided
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        # Make the request
        response = self._make_request(endpoint, params)
        
        # Process the response - Binance returns an array of arrays
        processed_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': [
                {
                    'timestamp': entry[0],
                    'open': float(entry[1]),
                    'high': float(entry[2]),
                    'low': float(entry[3]),
                    'close': float(entry[4]),
                    'volume': float(entry[5]),
                    'close_time': entry[6],
                    'quote_asset_volume': float(entry[7]),
                    'trades': int(entry[8]),
                    'taker_buy_base': float(entry[9]),
                    'taker_buy_quote': float(entry[10])
                } for entry in response
            ]
        }
        
        return processed_data
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data for a specific symbol from Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dict: Dictionary containing the current ticker data
        """
        # Convert symbol format
        formatted_symbol = symbol.replace('/', '')
        
        endpoint = f"{self.base_url}/api/v3/ticker/24hr"
        params = {'symbol': formatted_symbol}
        
        response = self._make_request(endpoint, params)
        
        return {
            'symbol': symbol,
            'last_price': float(response['lastPrice']),
            'price_change': float(response['priceChange']),
            'price_change_percent': float(response['priceChangePercent']),
            'high_24h': float(response['highPrice']),
            'low_24h': float(response['lowPrice']),
            'volume_24h': float(response['volume']),
            'timestamp': int(time.time() * 1000)
        }