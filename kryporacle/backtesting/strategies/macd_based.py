import pandas as pd
import numpy as np
import logging

from .base import Strategy

logger = logging.getLogger(__name__)

class MACDStrategy(Strategy):
    """MACD-based trading strategy.
    
    Generates buy signals when MACD line crosses above signal line,
    and sell signals when MACD line crosses below signal line.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Initialize the MACD strategy.
        
        Args:
            fast_period (int): Fast period for MACD calculation
            slow_period (int): Slow period for MACD calculation
            signal_period (int): Signal line period for MACD calculation
        """
        super().__init__(name=f"MACD ({fast_period}/{slow_period}/{signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self._required_indicators = ['macd_line', 'macd_signal']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MACD crossovers.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        result = df.copy()
        
        # Generate MACD signals
        result['signal'] = 0  # Default: no signal
        
        # Buy signal: MACD line crosses above signal line
        buy_condition = (result['macd_line'].shift(1) <= result['macd_signal'].shift(1)) & \
                        (result['macd_line'] > result['macd_signal'])
        result.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: MACD line crosses below signal line
        sell_condition = (result['macd_line'].shift(1) >= result['macd_signal'].shift(1)) & \
                         (result['macd_line'] < result['macd_signal'])
        result.loc[sell_condition, 'signal'] = -1
        
        logger.info(f"Generated {buy_condition.sum()} buy signals and {sell_condition.sum()} sell signals")
        return result


class DualMACDStrategy(Strategy):
    """Dual MACD strategy combining two timeframes.
    
    Uses MACD signals from two different timeframes for confirmation
    before entering or exiting trades.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Initialize the Dual MACD strategy.
        
        Args:
            fast_period (int): Fast period for MACD calculation
            slow_period (int): Slow period for MACD calculation
            signal_period (int): Signal line period for MACD calculation
        """
        super().__init__(name=f"Dual MACD ({fast_period}/{slow_period}/{signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self._required_indicators = ['macd_line', 'macd_signal', 'macd_histogram']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Dual MACD strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        result = df.copy()
        
        # Generate dual timeframe signals
        result['signal'] = 0  # Default: no signal
        
        # Simple MACD crossover signals
        macd_cross_up = (result['macd_line'].shift(1) <= result['macd_signal'].shift(1)) & \
                        (result['macd_line'] > result['macd_signal'])
        
        macd_cross_down = (result['macd_line'].shift(1) >= result['macd_signal'].shift(1)) & \
                          (result['macd_line'] < result['macd_signal'])
        
        # Add histogram direction as confirmation
        histogram_increasing = result['macd_histogram'] > result['macd_histogram'].shift(1)
        histogram_decreasing = result['macd_histogram'] < result['macd_histogram'].shift(1)
        
        # Buy signal: MACD crosses above signal line AND histogram is increasing
        buy_condition = macd_cross_up & histogram_increasing
        
        # Sell signal: MACD crosses below signal line AND histogram is decreasing
        sell_condition = macd_cross_down & histogram_decreasing
        
        # Apply signals
        result.loc[buy_condition, 'signal'] = 1
        result.loc[sell_condition, 'signal'] = -1
        
        logger.info(f"Generated {buy_condition.sum()} buy signals and {sell_condition.sum()} sell signals")
        return result