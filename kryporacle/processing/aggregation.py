import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)

def rolling_window_agg(df: pd.DataFrame, window: int, columns: Optional[List[str]] = None,
                      agg_func: Union[str, Dict[str, str], Callable] = 'mean') -> pd.DataFrame:
    """Apply rolling window aggregation to DataFrame columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        window (int): Size of the rolling window
        columns (List[str], optional): Columns to apply the rolling window.
                                      If None, applies to all numeric columns.
        agg_func: Aggregation function ('mean', 'sum', 'std', etc.) or a dictionary
                 mapping columns to aggregation functions, or a custom callable function
                 
    Returns:
        pd.DataFrame: DataFrame with rolling window aggregations added as new columns
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original dataframe
    result = df.copy()
    
    if not columns:
        # Use all numeric columns except timestamp-like columns
        columns = [col for col in result.select_dtypes(include=['number']).columns 
                  if 'time' not in col.lower() and 'date' not in col.lower()]
    
    # Handle different types of aggregation functions
    if isinstance(agg_func, str):
        # Same aggregation function for all columns
        for col in columns:
            if col in result.columns:
                result[f'{col}_rolling_{window}_{agg_func}'] = result[col].rolling(window=window).agg(agg_func)
    
    elif isinstance(agg_func, dict):
        # Different aggregation functions for different columns
        for col, func in agg_func.items():
            if col in result.columns and col in columns:
                result[f'{col}_rolling_{window}_{func}'] = result[col].rolling(window=window).agg(func)
    
    elif callable(agg_func):
        # Custom aggregation function
        for col in columns:
            if col in result.columns:
                result[f'{col}_rolling_{window}_custom'] = result[col].rolling(window=window).apply(agg_func)
    
    else:
        raise ValueError(f"Unsupported aggregation function type: {type(agg_func)}")
    
    return result

def time_based_agg(df: pd.DataFrame, freq: str, 
                  agg_funcs: Optional[Dict[str, Union[str, List[str]]]] = None) -> pd.DataFrame:
    """Aggregate data based on a time frequency.
    
    Args:
        df (pd.DataFrame): Input DataFrame with timestamp column
        freq (str): Frequency string for resampling (e.g., 'D' for daily, 'W' for weekly)
        agg_funcs (Dict): Dictionary mapping column names to aggregation functions
        
    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    if df.empty:
        return df
        
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    
    # Set timestamp as index for resampling
    df_indexed = df.set_index('timestamp')
    
    # Set default aggregation functions if not provided
    if not agg_funcs:
        agg_funcs = {}
        
        # Use common aggregations for OHLCV data
        if 'open' in df.columns:
            agg_funcs['open'] = 'first'
        if 'high' in df.columns:
            agg_funcs['high'] = 'max'
        if 'low' in df.columns:
            agg_funcs['low'] = 'min'
        if 'close' in df.columns:
            agg_funcs['close'] = 'last'
        if 'volume' in df.columns:
            agg_funcs['volume'] = 'sum'
            
        # For other numeric columns, use mean
        for col in df.select_dtypes(include=['number']).columns:
            if col not in agg_funcs:
                agg_funcs[col] = 'mean'
    
    # Apply resampling with aggregation functions
    resampled = df_indexed.resample(freq).agg(agg_funcs)
    
    # Reset index to get timestamp as a column again
    return resampled.reset_index()

def group_by_agg(df: pd.DataFrame, group_cols: List[str], 
                agg_funcs: Optional[Dict[str, Union[str, List[str]]]] = None) -> pd.DataFrame:
    """Aggregate data by grouping on specific columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        group_cols (List[str]): Columns to group by
        agg_funcs (Dict): Dictionary mapping column names to aggregation functions
        
    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    if df.empty:
        return df
        
    # Check if all group columns exist
    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Group columns not found in DataFrame: {missing_cols}")
    
    # Set default aggregation functions if not provided
    if not agg_funcs:
        agg_funcs = {}
        
        # Use appropriate aggregations for OHLCV data
        if 'open' in df.columns and 'open' not in group_cols:
            agg_funcs['open'] = 'first'
        if 'high' in df.columns and 'high' not in group_cols:
            agg_funcs['high'] = 'max'
        if 'low' in df.columns and 'low' not in group_cols:
            agg_funcs['low'] = 'min'
        if 'close' in df.columns and 'close' not in group_cols:
            agg_funcs['close'] = 'last'
        if 'volume' in df.columns and 'volume' not in group_cols:
            agg_funcs['volume'] = 'sum'
            
        # For other numeric columns, use mean
        for col in df.select_dtypes(include=['number']).columns:
            if col not in agg_funcs and col not in group_cols:
                agg_funcs[col] = 'mean'
    
    # Apply groupby with aggregation functions
    return df.groupby(group_cols).agg(agg_funcs).reset_index()

def calculate_returns(df: pd.DataFrame, price_col: str = 'close', 
                     periods: Optional[List[int]] = None) -> pd.DataFrame:
    """Calculate returns over different periods.
    
    Args:
        df (pd.DataFrame): Input DataFrame with price data
        price_col (str): Column name containing price data
        periods (List[int], optional): List of periods for return calculation
                                      If None, uses [1, 5, 20] (daily, weekly, monthly)
                                      
    Returns:
        pd.DataFrame: DataFrame with returns columns added
    """
    if df.empty:
        return df
        
    # Check if price column exists
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Make a copy to avoid modifying the original dataframe
    result = df.copy()
    
    # Set default periods if not provided
    if not periods:
        periods = [1, 5, 20]  # Common periods for daily data
    
    # Calculate returns for each period
    for period in periods:
        # Percentage change
        result[f'return_{period}'] = result[price_col].pct_change(periods=period)
        
        # Log return (useful for statistical analysis)
        result[f'log_return_{period}'] = np.log(result[price_col] / result[price_col].shift(period))
    
    return result