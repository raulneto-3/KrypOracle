import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def simple_moving_average(df: pd.DataFrame, column: str = 'close', 
                         window_sizes: Union[int, List[int]] = 20) -> pd.DataFrame:
    """Calculate Simple Moving Average (SMA) for a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate SMA for (default: 'close')
        window_sizes (Union[int, List[int]]): Window size(s) for SMA calculation
                                             Can be single integer or list of integers
                                             
    Returns:
        pd.DataFrame: DataFrame with SMA column(s) added
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Convert single window size to list for uniform processing
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]
    
    # Calculate SMA for each window size
    for window in window_sizes:
        result[f'sma_{window}'] = df[column].rolling(window=window).mean()
    
    return result


def exponential_moving_average(df: pd.DataFrame, column: str = 'close', 
                             window_sizes: Union[int, List[int]] = 20,
                             adjust: bool = True) -> pd.DataFrame:
    """Calculate Exponential Moving Average (EMA) for a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate EMA for (default: 'close')
        window_sizes (Union[int, List[int]]): Window size(s) for EMA calculation
                                             Can be single integer or list of integers
        adjust (bool): Whether to use adjusted EMA calculation (default: True)
                                             
    Returns:
        pd.DataFrame: DataFrame with EMA column(s) added
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Convert single window size to list for uniform processing
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]
    
    # Calculate EMA for each window size
    for window in window_sizes:
        result[f'ema_{window}'] = df[column].ewm(span=window, adjust=adjust).mean()
    
    return result


def macd(df: pd.DataFrame, column: str = 'close', 
        fast_period: int = 12, slow_period: int = 26, 
        signal_period: int = 9) -> pd.DataFrame:
    """Calculate Moving Average Convergence Divergence (MACD) for a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate MACD for (default: 'close')
        fast_period (int): Period for fast EMA (default: 12)
        slow_period (int): Period for slow EMA (default: 26)
        signal_period (int): Period for signal line (default: 9)
                                             
    Returns:
        pd.DataFrame: DataFrame with MACD columns added:
                     - macd_line: Fast EMA - Slow EMA
                     - macd_signal: Signal line (EMA of MACD line)
                     - macd_histogram: MACD line - Signal line
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Calculate fast and slow EMAs
    fast_ema = df[column].ewm(span=fast_period, adjust=True).mean()
    slow_ema = df[column].ewm(span=slow_period, adjust=True).mean()
    
    # Calculate MACD line (fast EMA - slow EMA)
    result['macd_line'] = fast_ema - slow_ema
    
    # Calculate signal line (EMA of MACD line)
    result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=True).mean()
    
    # Calculate MACD histogram (MACD line - signal line)
    result['macd_histogram'] = result['macd_line'] - result['macd_signal']
    
    return result


def rsi(df: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index (RSI) for a DataFrame column.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate RSI for (default: 'close')
        window (int): Look-back period for RSI calculation (default: 14)
                                             
    Returns:
        pd.DataFrame: DataFrame with RSI column added
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Calculate price changes
    delta = df[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and average loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    
    # Calculate RSI
    result['rsi'] = 100 - (100 / (1 + rs))
    
    return result


def detect_crossover(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.DataFrame:
    """Detect crossovers between two indicators (e.g., moving averages).
    
    Args:
        df (pd.DataFrame): DataFrame with indicator columns
        fast_col (str): Column name of the faster indicator
        slow_col (str): Column name of the slower indicator
        
    Returns:
        pd.DataFrame: DataFrame with crossover signals:
                     - crossover_signal: 1 for bullish crossover (fast crosses above slow)
                                        -1 for bearish crossover (fast crosses below slow)
                                         0 for no crossover
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Current state: fast > slow
    current_state = result[fast_col] > result[slow_col]
    
    # Previous state: fast > slow (shifted by 1)
    prev_state = current_state.shift(1)
    
    # Bullish crossover: previous state False, current state True
    bullish_crossover = (prev_state == False) & (current_state == True)
    
    # Bearish crossover: previous state True, current state False
    bearish_crossover = (prev_state == True) & (current_state == False)
    
    # Create crossover signal column
    result['crossover_signal'] = 0  # Default: no crossover
    result.loc[bullish_crossover, 'crossover_signal'] = 1   # Bullish crossover
    result.loc[bearish_crossover, 'crossover_signal'] = -1  # Bearish crossover
    
    return result


def golden_death_cross(df: pd.DataFrame, column: str = 'close',
                      fast_window: int = 50, slow_window: int = 200) -> pd.DataFrame:
    """Detect Golden Cross and Death Cross events.
    
    A Golden Cross occurs when a faster moving average crosses above a slower moving average,
    typically when the 50-day SMA crosses above the 200-day SMA.
    
    A Death Cross occurs when a faster moving average crosses below a slower moving average,
    typically when the 50-day SMA crosses below the 200-day SMA.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate moving averages for (default: 'close')
        fast_window (int): Period for faster moving average (default: 50)
        slow_window (int): Period for slower moving average (default: 200)
        
    Returns:
        pd.DataFrame: DataFrame with SMA columns and cross signals added:
                     - sma_{fast_window}: Fast SMA
                     - sma_{slow_window}: Slow SMA
                     - cross_signal: 1 for Golden Cross, -1 for Death Cross, 0 for no cross
    """
    if df.empty:
        return df
        
    # Calculate SMAs
    result = simple_moving_average(df, column, [fast_window, slow_window])
    
    # Detect crossovers
    fast_col = f'sma_{fast_window}'
    slow_col = f'sma_{slow_window}'
    result = detect_crossover(result, fast_col, slow_col)
    
    # Rename crossover_signal to cross_signal for clarity
    result = result.rename(columns={'crossover_signal': 'cross_signal'})
    
    return result


def find_support_resistance(df: pd.DataFrame, column: str = 'close',
                           window: int = 10, threshold: float = 0.02) -> Tuple[List[float], List[float]]:
    """Find support and resistance levels based on local minima and maxima.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to analyze (default: 'close')
        window (int): Window size for peak detection (default: 10)
        threshold (float): Minimum percentage difference between levels (default: 0.02)
        
    Returns:
        Tuple[List[float], List[float]]: Tuple containing list of support levels and list of resistance levels
    """
    if df.empty:
        return [], []
    
    prices = df[column].values
    
    # Find local minima (support levels)
    supports = []
    for i in range(window, len(prices) - window):
        if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] <= prices[i+j] for j in range(1, window+1)):
            supports.append(prices[i])
    
    # Find local maxima (resistance levels)
    resistances = []
    for i in range(window, len(prices) - window):
        if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
           all(prices[i] >= prices[i+j] for j in range(1, window+1)):
            resistances.append(prices[i])
    
    # Filter out levels that are too close to each other
    # (within threshold percentage)
    filtered_supports = []
    for s in supports:
        if not filtered_supports or all(abs(s - fs) / fs > threshold for fs in filtered_supports):
            filtered_supports.append(s)
    
    filtered_resistances = []
    for r in resistances:
        if not filtered_resistances or all(abs(r - fr) / fr > threshold for fr in filtered_resistances):
            filtered_resistances.append(r)
    
    # Sort levels
    filtered_supports.sort()
    filtered_resistances.sort()
    
    return filtered_supports, filtered_resistances


def identify_trend(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.DataFrame:
    """Identify market trend based on price movement and moving average.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to analyze (default: 'close')
        window (int): Window size for moving average (default: 20)
        
    Returns:
        pd.DataFrame: DataFrame with trend column added:
                     - trend: 1 for uptrend, -1 for downtrend, 0 for sideways/neutral
    """
    if df.empty:
        return df
        
    result = df.copy()
    
    # Calculate moving average
    result = simple_moving_average(result, column, window)
    ma_col = f'sma_{window}'
    
    # Calculate price position relative to moving average
    result['price_above_ma'] = result[column] > result[ma_col]
    
    # Calculate moving average slope (direction)
    result['ma_slope'] = result[ma_col].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Identify trend based on price position and moving average slope
    result['trend'] = 0  # Default: sideways/neutral
    
    # Uptrend: Price above MA and MA slope positive
    uptrend = (result['price_above_ma'] == True) & (result['ma_slope'] > 0)
    result.loc[uptrend, 'trend'] = 1
    
    # Downtrend: Price below MA and MA slope negative
    downtrend = (result['price_above_ma'] == False) & (result['ma_slope'] < 0)
    result.loc[downtrend, 'trend'] = -1
    
    # Clean up intermediate columns
    result = result.drop(columns=['price_above_ma', 'ma_slope'])
    
    return result


def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels.
    
    Args:
        high (float): Highest price point
        low (float): Lowest price point
        
    Returns:
        Dict[str, float]: Dictionary of Fibonacci retracement levels
    """
    diff = high - low
    
    return {
        "0.0": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.5": low + 0.5 * diff,
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1.0": high
    }


def calculate_indicators(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
    """Calculate multiple trend indicators for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
        column (str): Column name to calculate indicators for (default: 'close')
        
    Returns:
        pd.DataFrame: DataFrame with all trend indicators added
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    # Calculate SMAs
    result = simple_moving_average(result, column, [20, 50, 200])
    
    # Calculate EMAs
    result = exponential_moving_average(result, column, [9, 12, 26])
    
    # Calculate MACD
    result = macd(result, column)
    
    # Calculate RSI
    result = rsi(result, column)
    
    # Calculate trend
    result = identify_trend(result, column)
    
    # Calculate Golden/Death Cross
    cross_df = golden_death_cross(result, column)
    result['cross_signal'] = cross_df['cross_signal']
    
    return result