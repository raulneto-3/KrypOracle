import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Map of timeframe strings to pandas resample offset strings
TIMEFRAME_MAP = {
    '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
    '1h': '1H', '4h': '4H', '6h': '6H', '12h': '12H',
    '1d': '1D', '3d': '3D', '1w': '1W'
}

# Map of timeframes to minutes for comparing intervals
TIMEFRAME_MINUTES = {
    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '4h': 240, '6h': 360, '12h': 720,
    '1d': 1440, '3d': 4320, '1w': 10080
}

def standardize_data(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Standardize data (zero mean, unit variance) for machine learning tasks.
    
    Args:
        df (pd.DataFrame): DataFrame to standardize
        columns (List[str], optional): Columns to standardize.
                                       If None, standardizes all numeric columns
                                       
    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original dataframe
    result = df.copy()
    
    if not columns:
        # Use all numeric columns except timestamp-like columns
        columns = [col for col in result.select_dtypes(include=['number']).columns 
                  if 'time' not in col.lower() and 'date' not in col.lower()]
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
            
        mean = result[col].mean()
        std = result[col].std()
        
        if std == 0:  # Avoid division by zero
            logger.warning(f"Column '{col}' has zero standard deviation. Skipping standardization.")
            continue
            
        result[f'{col}_std'] = (result[col] - mean) / std
    
    return result

def normalize_data(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                  method: str = 'minmax') -> pd.DataFrame:
    """Normalize data to a specific range (e.g., 0-1).
    
    Args:
        df (pd.DataFrame): DataFrame to normalize
        columns (List[str], optional): Columns to normalize.
                                       If None, normalizes all numeric columns
        method (str): Normalization method ('minmax' or 'robust')
                                       
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original dataframe
    result = df.copy()
    
    if not columns:
        # Use all numeric columns except timestamp-like columns
        columns = [col for col in result.select_dtypes(include=['number']).columns 
                  if 'time' not in col.lower() and 'date' not in col.lower()]
    
    for col in columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
            
        if method == 'minmax':
            # Min-max normalization (0-1 range)
            min_val = result[col].min()
            max_val = result[col].max()
            
            if min_val == max_val:  # Avoid division by zero
                logger.warning(f"Column '{col}' has same min and max values. Setting normalized value to 0.5.")
                result[f'{col}_norm'] = 0.5
            else:
                result[f'{col}_norm'] = (result[col] - min_val) / (max_val - min_val)
                
        elif method == 'robust':
            # Robust normalization using percentiles (helps with outliers)
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:  # Avoid division by zero
                logger.warning(f"Column '{col}' has zero IQR. Setting normalized value to 0.5.")
                result[f'{col}_norm'] = 0.5
            else:
                result[f'{col}_norm'] = (result[col] - q1) / iqr
                
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    
    return result

def resample_data(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
    """Resample data from one timeframe to another.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        source_tf (str): Source timeframe (e.g., '1m', '5m')
        target_tf (str): Target timeframe (e.g., '1h', '4h')
        
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    if df.empty:
        return df
        
    # Ensure 'timestamp' column exists and is datetime type
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column for resampling")
    
    # Convert to pandas offset strings
    source = TIMEFRAME_MAP.get(source_tf)
    target = TIMEFRAME_MAP.get(target_tf)
    
    if not source or not target:
        raise ValueError(f"Unsupported timeframe format: {source_tf} or {target_tf}")
    
    # Check if we're upsampling or downsampling
    is_upsampling = TIMEFRAME_MINUTES.get(source_tf, 0) > TIMEFRAME_MINUTES.get(target_tf, float('inf'))
    
    # Set timestamp as index for resampling
    df_indexed = df.set_index('timestamp')
    
    if is_upsampling:
        # For upsampling (e.g., 1h -> 1m), we use forward fill
        resampled = df_indexed.resample(target).ffill()
    else:
        # For downsampling (e.g., 1m -> 1h), we use appropriate aggregations
        # Define aggregation functions for OHLCV data
        agg_dict = {}
        
        if 'open' in df.columns:
            agg_dict['open'] = 'first'
        if 'high' in df.columns:
            agg_dict['high'] = 'max'
        if 'low' in df.columns:
            agg_dict['low'] = 'min'
        if 'close' in df.columns:
            agg_dict['close'] = 'last'
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
            
        # For any remaining numeric columns, use mean
        for col in df.select_dtypes(include=['number']).columns:
            if col not in agg_dict:
                agg_dict[col] = 'mean'
                
        # For any other columns, use last
        for col in df.columns:
            if col not in agg_dict and col != 'timestamp':
                agg_dict[col] = 'last'
                
        resampled = df_indexed.resample(target).agg(agg_dict)
    
    # Reset index to get timestamp as a column again
    result = resampled.reset_index()
    
    return result