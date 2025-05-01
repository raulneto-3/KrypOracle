import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def bollinger_bands(df: pd.DataFrame, column: str = 'close',
                   window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands for a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate Bollinger Bands for (default: 'close')
        window (int): Window size for moving average (default: 20)
        std_dev (float): Number of standard deviations for bands (default: 2.0)
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands columns added:
                     - bb_middle: Middle band (SMA)
                     - bb_upper: Upper band (SMA + std_dev * standard deviation)
                     - bb_lower: Lower band (SMA - std_dev * standard deviation)
                     - bb_width: Band width ((upper - lower) / middle)
                     - bb_pct_b: %B indicator ((price - lower) / (upper - lower))
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Calculate middle band (SMA)
    result['bb_middle'] = result[column].rolling(window=window).mean()
    
    # Calculate standard deviation
    rolling_std = result[column].rolling(window=window).std()
    
    # Calculate upper and lower bands
    result['bb_upper'] = result['bb_middle'] + (std_dev * rolling_std)
    result['bb_lower'] = result['bb_middle'] - (std_dev * rolling_std)
    
    # Calculate band width
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
    
    # Calculate %B indicator
    result['bb_pct_b'] = (result[column] - result['bb_lower']) / \
                         (result['bb_upper'] - result['bb_lower'] + np.finfo(float).eps)  # Avoid division by zero
    
    return result


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate Average True Range (ATR) for a DataFrame.
    
    The True Range (TR) is defined as the greatest of:
    1. Current high - current low
    2. |Current high - previous close|
    3. |Current low - previous close|
    
    ATR is the moving average of the TR over the specified window.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        window (int): Window size for ATR calculation (default: 14)
        
    Returns:
        pd.DataFrame: DataFrame with ATR column added
    """
    if df.empty:
        return df
        
    # Verify that required columns exist
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    result = df.copy()
    
    # Calculate True Range (TR)
    high_low = result['high'] - result['low']
    high_prev_close = (result['high'] - result['close'].shift()).abs()
    low_prev_close = (result['low'] - result['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    result['true_range'] = tr
    
    # Calculate ATR
    result['atr'] = tr.rolling(window=window).mean()
    
    return result


def historical_volatility(df: pd.DataFrame, column: str = 'close',
                         window: int = 20, trading_periods: int = 365,
                         log_returns: bool = True) -> pd.DataFrame:
    """Calculate historical volatility for a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate volatility for (default: 'close')
        window (int): Window size for volatility calculation (default: 20)
        trading_periods (int): Number of trading periods in a year (default: 365)
        log_returns (bool): Whether to use logarithmic returns (default: True)
        
    Returns:
        pd.DataFrame: DataFrame with volatility column added
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Calculate returns
    if log_returns:
        # Logarithmic returns
        result['returns'] = np.log(result[column] / result[column].shift(1))
    else:
        # Simple returns
        result['returns'] = result[column].pct_change()
    
    # Calculate rolling standard deviation of returns
    result['volatility'] = result['returns'].rolling(window=window).std() * np.sqrt(trading_periods)
    
    # Drop the temporary returns column
    result = result.drop(columns=['returns'])
    
    return result


def relative_volatility_index(df: pd.DataFrame, column: str = 'close',
                            window: int = 10) -> pd.DataFrame:
    """Calculate Relative Volatility Index (RVI).
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate RVI for (default: 'close')
        window (int): Window size for RVI calculation (default: 10)
        
    Returns:
        pd.DataFrame: DataFrame with RVI column added
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Calculate price change
    result['price_change'] = result[column].diff()
    
    # Calculate standard deviation over a rolling window
    result['std_dev'] = result[column].rolling(window=window).std()
    
    # Calculate upward and downward volatility
    result['upward_vol'] = np.where(result['price_change'] > 0, result['std_dev'], 0)
    result['downward_vol'] = np.where(result['price_change'] < 0, result['std_dev'], 0)
    
    # Calculate smoothed upward and downward volatility
    result['smoothed_upward'] = result['upward_vol'].rolling(window=window).mean()
    result['smoothed_downward'] = result['downward_vol'].rolling(window=window).mean()
    
    # Calculate RVI
    result['rvi'] = (result['smoothed_upward'] / 
                    (result['smoothed_upward'] + result['smoothed_downward'] + np.finfo(float).eps) * 100)
    
    # Drop temporary columns
    result = result.drop(columns=['price_change', 'std_dev', 'upward_vol', 
                                 'downward_vol', 'smoothed_upward', 'smoothed_downward'])
    
    return result


def keltner_channels(df: pd.DataFrame, 
                    ema_window: int = 20, 
                    atr_window: int = 10,
                    atr_multiplier: float = 2.0) -> pd.DataFrame:
    """Calculate Keltner Channels.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        ema_window (int): Window size for EMA calculation (default: 20)
        atr_window (int): Window size for ATR calculation (default: 10)
        atr_multiplier (float): Multiplier for ATR (default: 2.0)
        
    Returns:
        pd.DataFrame: DataFrame with Keltner Channels columns added:
                     - kc_middle: Middle line (EMA of typical price)
                     - kc_upper: Upper band (middle + ATR * multiplier)
                     - kc_lower: Lower band (middle - ATR * multiplier)
    """
    if df.empty:
        return df
        
    # Verify that required columns exist
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    result = df.copy()
    
    # Calculate typical price (high + low + close) / 3
    result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
    
    # Calculate EMA of typical price
    result['kc_middle'] = result['typical_price'].ewm(span=ema_window, adjust=True).mean()
    
    # Calculate ATR
    atr_df = average_true_range(result, window=atr_window)
    result['atr'] = atr_df['atr']
    
    # Calculate upper and lower bands
    result['kc_upper'] = result['kc_middle'] + (result['atr'] * atr_multiplier)
    result['kc_lower'] = result['kc_middle'] - (result['atr'] * atr_multiplier)
    
    # Clean up intermediate columns
    result = result.drop(columns=['typical_price'])
    
    return result


def volatility_ratio(df: pd.DataFrame, column: str = 'close',
                    short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """Calculate the ratio of short-term volatility to long-term volatility.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate volatility for (default: 'close')
        short_window (int): Window size for short-term volatility (default: 5)
        long_window (int): Window size for long-term volatility (default: 20)
        
    Returns:
        pd.DataFrame: DataFrame with volatility ratio column added
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Calculate returns
    result['returns'] = np.log(result[column] / result[column].shift(1))
    
    # Calculate short-term and long-term volatility
    result['short_vol'] = result['returns'].rolling(window=short_window).std()
    result['long_vol'] = result['returns'].rolling(window=long_window).std()
    
    # Calculate volatility ratio
    result['volatility_ratio'] = result['short_vol'] / (result['long_vol'] + np.finfo(float).eps)
    
    # Drop temporary columns
    result = result.drop(columns=['returns', 'short_vol', 'long_vol'])
    
    return result


def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate multiple volatility indicators for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with all volatility indicators added
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    # Calculate Bollinger Bands
    result = bollinger_bands(result)
    
    # Calculate ATR
    result = average_true_range(result)
    
    # Calculate Historical Volatility
    result = historical_volatility(result)
    
    # Calculate RVI
    result = relative_volatility_index(result)
    
    # Calculate Keltner Channels
    result = keltner_channels(result)
    
    # Calculate Volatility Ratio
    result = volatility_ratio(result)
    
    return result