import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove duplicate rows from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        subset (List[str], optional): Columns to consider for identifying duplicates
                                     Defaults to using timestamp if present
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    if df.empty:
        return df
        
    if not subset and 'timestamp' in df.columns:
        subset = ['timestamp']
        
    return df.drop_duplicates(subset=subset)

def handle_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """Handle missing values in DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with missing values
        method (str): Method to fill missing values ('ffill', 'bfill', 'zero', or 'mean')
                      
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original dataframe
    result = df.copy()
    
    if method == 'ffill':
        result = result.fillna(method='ffill')
        # If there are still NAs at the beginning, backfill them
        result = result.fillna(method='bfill')
    elif method == 'bfill':
        result = result.fillna(method='bfill')
        # If there are still NAs at the end, forward fill them
        result = result.fillna(method='ffill')
    elif method == 'zero':
        result = result.fillna(0)
    elif method == 'mean':
        # Only apply mean to numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            result[col] = result[col].fillna(result[col].mean())
        # For any remaining NAs, use forward fill
        result = result.fillna(method='ffill').fillna(method='bfill')
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return result

def remove_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None, 
                   method: str = 'zscore', threshold: float = 3.0) -> pd.DataFrame:
    """Remove or replace outliers in specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        cols (List[str], optional): Columns to check for outliers.
                                   If None, uses all numeric columns.
        method (str): Method for outlier detection ('zscore' or 'iqr')
        threshold (float): Threshold for outlier detection
                          (e.g., z-score > threshold or IQR * threshold)
    
    Returns:
        pd.DataFrame: DataFrame with outliers handled
    """
    if df.empty:
        return df
        
    # Make a copy to avoid modifying the original dataframe
    result = df.copy()
    
    if not cols:
        # Use all numeric columns except timestamp-like columns
        cols = [col for col in result.select_dtypes(include=['number']).columns 
                if 'time' not in col.lower()]
    
    for col in cols:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
            
        if method == 'zscore':
            # Z-score method
            mean = result[col].mean()
            std = result[col].std()
            if std == 0:  # Avoid division by zero
                continue
                
            z_scores = np.abs((result[col] - mean) / std)
            outliers = z_scores > threshold
            
        elif method == 'iqr':
            # IQR method
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
            
        else:
            raise ValueError(f"Unsupported outlier detection method: {method}")
        
        # Replace outliers with median value
        if outliers.any():
            median_val = result[col].median()
            result.loc[outliers, col] = median_val
            logger.info(f"Replaced {outliers.sum()} outliers in column '{col}'")
    
    return result

def ensure_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLC data consistency.
    
    Makes sure that:
    - high >= max(open, close)
    - low <= min(open, close)
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with consistent OHLC values
    """
    if df.empty:
        return df
        
    # Check if dataframe has OHLC columns
    ohlc_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in ohlc_cols):
        return df
        
    # Make a copy to avoid modifying the original dataframe
    result = df.copy()
    
    # Ensure high is the highest value
    result['high'] = result[['high', 'open', 'close']].max(axis=1)
    
    # Ensure low is the lowest value
    result['low'] = result[['low', 'open', 'close']].min(axis=1)
    
    return result