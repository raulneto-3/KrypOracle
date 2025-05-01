import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

from ..data.storage import DataStorage

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing cryptocurrency market data."""
    
    def __init__(self, storage: Optional[DataStorage] = None):
        """Initialize the data processor.
        
        Args:
            storage (DataStorage, optional): Data storage instance for loading/saving data
        """
        self.storage = storage or DataStorage()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean market data by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): DataFrame containing market data
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # 1. Handle missing values
        # For OHLC data, forward fill is often appropriate (carry last known price)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(method='ffill')
                # If there are still NaNs at the beginning, backfill them
                df[col] = df[col].fillna(method='bfill')
        
        # For volume data, fill missing values with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # 2. Handle outliers
        # Apply a rolling median filter to detect and correct outliers
        # Define an outlier as a value that deviates more than 3 standard deviations
        # from a rolling median with window size 5
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                # Calculate rolling median and standard deviation
                rolling_median = df[col].rolling(window=5, center=True).median()
                rolling_std = df[col].rolling(window=5, center=True).std()
                
                # Fill NaNs in rolling calculations
                rolling_median = rolling_median.fillna(df[col])
                rolling_std = rolling_std.fillna(df[col].std())
                
                # Identify outliers
                is_outlier = (df[col] - rolling_median).abs() > 3 * rolling_std
                
                # Replace outliers with rolling median
                if is_outlier.any():
                    outlier_count = is_outlier.sum()
                    logger.info(f"Detected {outlier_count} outliers in '{col}' column")
                    df.loc[is_outlier, col] = rolling_median[is_outlier]
        
        # 3. Ensure data consistency (e.g., high >= low, high >= open/close, low <= open/close)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Make sure high is the highest value
            df['high'] = df[['high', 'open', 'close']].max(axis=1)
            # Make sure low is the lowest value
            df['low'] = df[['low', 'open', 'close']].min(axis=1)
            
        return df
    
    def normalize_timeframe(self, df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
        """Normalize data from source timeframe to target timeframe.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            source_tf (str): Source timeframe (e.g., '1m', '5m')
            target_tf (str): Target timeframe (e.g., '1h', '4h')
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        if df.empty:
            return df
            
        # Parse timeframes to pandas offset strings
        tf_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W'
        }
        
        source = tf_map.get(source_tf)
        target = tf_map.get(target_tf)
        
        if not source or not target:
            raise ValueError(f"Unsupported timeframe format: {source_tf} or {target_tf}")
        
        # Check if we're upsampling or downsampling
        if self._is_upsampling(source_tf, target_tf):
            return self._upsample_data(df, target)
        else:
            return self._downsample_data(df, target)
    
    def _is_upsampling(self, source_tf: str, target_tf: str) -> bool:
        """Determine if conversion is upsampling (source interval > target interval)."""
        # Map of timeframes to minutes for easy comparison
        minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }
        return minutes.get(source_tf, 0) > minutes.get(target_tf, float('inf'))
    
    def _upsample_data(self, df: pd.DataFrame, target_offset: str) -> pd.DataFrame:
        """Upsample data to a higher frequency timeframe."""
        # For upsampling, we typically use forward fill (repeat last value)
        df = df.set_index('timestamp')
        upsampled = df.resample(target_offset).ffill()
        return upsampled.reset_index()
    
    def _downsample_data(self, df: pd.DataFrame, target_offset: str) -> pd.DataFrame:
        """Downsample data to a lower frequency timeframe using OHLCV aggregation."""
        df = df.set_index('timestamp')
        
        # For OHLCV data, use specific aggregation methods
        aggregation = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Only apply aggregation to columns that exist in the dataframe
        agg_dict = {col: aggregation[col] for col in aggregation if col in df.columns}
        
        # Apply other aggregations for any additional columns
        for col in df.columns:
            if col not in agg_dict:
                # For unknown columns, use last value as default
                agg_dict[col] = 'last'
        
        # Perform resampling with the aggregation functions
        downsampled = df.resample(target_offset).agg(agg_dict)
        return downsampled.reset_index()
    
    def aggregate_data(self, df: pd.DataFrame, window: int, method: str = 'mean') -> pd.DataFrame:
        """Aggregate data using a rolling window.
        
        Args:
            df (pd.DataFrame): DataFrame with market data
            window (int): Size of the rolling window
            method (str): Aggregation method ('mean', 'sum', 'median', etc.)
            
        Returns:
            pd.DataFrame: Aggregated DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original dataframe
        result = df.copy()
        
        # Apply rolling aggregation to price and volume columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if method == 'mean':
                result[f'{col}_rolling_{window}'] = df[col].rolling(window=window).mean()
            elif method == 'sum':
                result[f'{col}_rolling_{window}'] = df[col].rolling(window=window).sum()
            elif method == 'median':
                result[f'{col}_rolling_{window}'] = df[col].rolling(window=window).median()
            elif method == 'std':
                result[f'{col}_rolling_{window}'] = df[col].rolling(window=window).std()
            else:
                raise ValueError(f"Unsupported aggregation method: {method}")
        
        return result
    
    def process_market_data(self, exchange_id: str, symbol: str, timeframe: str,
                           start_time: Optional[pd.Timestamp] = None,
                           end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Complete processing pipeline for market data.
        
        Args:
            exchange_id (str): The exchange identifier
            symbol (str): The trading pair symbol
            timeframe (str): The data timeframe
            start_time (pd.Timestamp, optional): Start time for data processing
            end_time (pd.Timestamp, optional): End time for data processing
            
        Returns:
            pd.DataFrame: Processed market data
        """
        # 1. Load data from storage
        df = self.storage.load_market_data(
            exchange_id=exchange_id, 
            symbol=symbol, 
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            logger.warning(f"No data found for {exchange_id}/{symbol}/{timeframe}")
            return df
        
        # 2. Clean the data
        df = self.clean_data(df)
        
        # 3. Additional processing can be added here
        
        return df