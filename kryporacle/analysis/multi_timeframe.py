import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from ..processing.normalization import TIMEFRAME_MAP, TIMEFRAME_MINUTES

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Analyzer for multi-timeframe analysis.
    
    Provides methods to resample data to different timeframes and analyze
    trends across multiple timeframes for stronger confirmation signals.
    """
    
    @staticmethod
    def resample_to_higher_timeframe(df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
        """
        Resample data from a lower timeframe to a higher timeframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            source_tf (str): Source timeframe (e.g., '1m', '5m')
            target_tf (str): Target higher timeframe (e.g., '1h', '4h')
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Ensure timestamp is set as index
        if 'timestamp' in df.columns:
            df_indexed = df.set_index('timestamp')
        else:
            df_indexed = df.copy()
            if not isinstance(df_indexed.index, pd.DatetimeIndex):
                logger.warning("DataFrame index is not DatetimeIndex and no timestamp column found")
                return df
        
        # Get pandas resample frequency strings
        source_freq = TIMEFRAME_MAP.get(source_tf)
        target_freq = TIMEFRAME_MAP.get(target_tf)
        
        if not source_freq or not target_freq:
            logger.error(f"Unsupported timeframe format: {source_tf} or {target_tf}")
            return df
        
        # Check if target timeframe is higher than source
        if TIMEFRAME_MINUTES.get(target_tf, 0) <= TIMEFRAME_MINUTES.get(source_tf, 0):
            logger.warning(f"Target timeframe {target_tf} must be higher than source timeframe {source_tf}")
            return df
        
        # Resample OHLCV data
        resampled = df_indexed.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df_indexed.columns else None
        }).dropna()
        
        # Handle technical indicators by taking last value
        for col in df_indexed.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    # Try to resample the indicator by taking the last value
                    resampled[col] = df_indexed[col].resample(target_freq).last()
                except Exception as e:
                    logger.warning(f"Could not resample indicator {col}: {str(e)}")
        
        # Reset index to get timestamp as a column
        resampled = resampled.reset_index()
        
        return resampled
    
    @staticmethod
    def analyze_trend_alignment(lower_tf_df: pd.DataFrame, higher_tf_df: pd.DataFrame, 
                               indicator: str) -> pd.DataFrame:
        """
        Analyze if trends in two timeframes are aligned using the specified indicator.
        
        Args:
            lower_tf_df (pd.DataFrame): Lower timeframe DataFrame
            higher_tf_df (pd.DataFrame): Higher timeframe DataFrame
            indicator (str): Indicator column to use for trend analysis
            
        Returns:
            pd.DataFrame: Lower timeframe DataFrame with added trend alignment columns
        """
        result = lower_tf_df.copy()
        
        if indicator not in result.columns:
            logger.error(f"Indicator {indicator} not found in lower timeframe DataFrame")
            return result
        
        if indicator not in higher_tf_df.columns:
            logger.error(f"Indicator {indicator} not found in higher timeframe DataFrame")
            return result
        
        # Get higher timeframe trend direction
        higher_tf_trend = higher_tf_df[indicator].diff().fillna(0)
        higher_tf_df['trend_direction'] = np.where(higher_tf_trend > 0, 1, 
                                               np.where(higher_tf_trend < 0, -1, 0))
        
        # Merge higher timeframe trend to lower timeframe data
        # First convert timestamp to datetime if it's not
        if 'timestamp' in result.columns:
            result['timestamp'] = pd.to_datetime(result['timestamp'])
            higher_tf_df['timestamp'] = pd.to_datetime(higher_tf_df['timestamp'])
            
            # Create a date mapping to be able to join them
            result['date_key'] = result['timestamp'].dt.floor('D')
            higher_tf_df['date_key'] = higher_tf_df['timestamp'].dt.floor('D')
            
            # Map each lower timeframe bar to the most recent higher timeframe bar
            result['higher_tf_trend'] = np.nan
            
            for i in range(len(result)):
                current_time = result.iloc[i]['timestamp']
                # Find the latest higher timeframe bar that's earlier than or equal to the current bar
                matching_bars = higher_tf_df[higher_tf_df['timestamp'] <= current_time]
                if not matching_bars.empty:
                    latest_bar = matching_bars.iloc[-1]
                    result.loc[result.index[i], 'higher_tf_trend'] = latest_bar['trend_direction']
            
            # Clean up
            result = result.drop(columns=['date_key'])
        
        # Calculate trend alignment
        result['trend_aligned'] = (np.sign(result[indicator].diff().fillna(0)) == 
                                 result['higher_tf_trend'])
        
        return result
    
    @staticmethod
    def find_multi_timeframe_signals(df: pd.DataFrame, 
                                   primary_signal_col: str, 
                                   confirmation_col: str = 'trend_aligned',
                                   lookback: int = 3) -> pd.DataFrame:
        """
        Find signals that have confirmation from multiple timeframes.
        
        Args:
            df (pd.DataFrame): DataFrame with signal and confirmation columns
            primary_signal_col (str): Column name containing primary signals (1=buy, -1=sell)
            confirmation_col (str): Column name containing the trend alignment indicator
            lookback (int): Number of periods to check for confirmation
            
        Returns:
            pd.DataFrame: DataFrame with filtered signals based on multi-timeframe confirmation
        """
        result = df.copy()
        
        if primary_signal_col not in result.columns:
            logger.error(f"Signal column {primary_signal_col} not found")
            return result
        
        if confirmation_col not in result.columns:
            logger.error(f"Confirmation column {confirmation_col} not found")
            return result
        
        # Create confirmed signals column
        result['confirmed_signal'] = 0
        
        # Find and filter signals
        for i in range(lookback, len(result)):
            # If there's a primary signal
            if result.iloc[i][primary_signal_col] != 0:
                # Check if the last 'lookback' periods have confirmation
                confirmation_window = result.iloc[i-lookback:i+1][confirmation_col]
                
                # If most of the window is confirmed (aligned), accept the signal
                if confirmation_window.mean() >= 0.5:  # At least 50% confirmation
                    result.loc[result.index[i], 'confirmed_signal'] = result.iloc[i][primary_signal_col]
        
        return result