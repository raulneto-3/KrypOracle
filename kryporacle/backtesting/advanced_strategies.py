import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

from .strategies import Strategy

logger = logging.getLogger(__name__)

class TrendFollowingStrategy(Strategy):
    """Trend Following Strategy that combines multiple indicators.
    
    Uses a combination of moving averages and ADX to identify trends
    and generate trading signals.
    """
    
    def __init__(self, ma_period: int = 20, trend_strength: int = 25):
        """Initialize the Trend Following strategy.
        
        Args:
            ma_period (int): Period for moving average
            trend_strength (int): ADX threshold to consider a strong trend
        """
        super().__init__(name=f"Trend Following ({ma_period}, {trend_strength})")
        self.ma_period = ma_period
        self.trend_strength = trend_strength
        self._required_indicators = [f'sma_{ma_period}', 'atr', 'volatility']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on trend following rules.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        result = df.copy()
        
        # Calculate ADX if not already present
        if 'adx' not in result.columns:
            # Simple proxy for ADX using ATR and volatility
            result['adx'] = result['atr'] / (result['close'] * result['volatility']) * 100
            result['adx'] = result['adx'].rolling(window=14).mean()
        
        ma_col = f'sma_{self.ma_period}'
        
        # Generate trend following signals
        result['signal'] = 0  # Default: no signal
        
        # Buy signal: Price above MA and strong uptrend (high ADX)
        buy_condition = (result['close'] > result[ma_col]) & \
                        (result['adx'] > self.trend_strength) & \
                        (result['close'].diff() > 0)  # Price increasing
        
        # Sell signal: Price below MA and strong downtrend or trend weakening
        sell_condition = (result['close'] < result[ma_col]) | \
                         ((result['adx'] < self.trend_strength) & (result['adx'].diff() < 0))
        
        # Apply signals only when not already in position
        result.loc[buy_condition, 'signal'] = 1
        
        # For selling, we'll only apply where we have a position
        # This will be handled by the backtest engine
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


class VolatilityBreakoutStrategy(Strategy):
    """Volatility Breakout trading strategy.
    
    Generates buy signals when price breaks out above recent volatility range,
    and sell signals when price falls below support levels.
    """
    
    def __init__(self, window: int = 20, volatility_factor: float = 1.5):
        """Initialize the Volatility Breakout strategy.
        
        Args:
            window (int): Lookback window for volatility calculation
            volatility_factor (float): Factor to multiply volatility for breakout threshold
        """
        super().__init__(name=f"Volatility Breakout ({window}, {volatility_factor})")
        self.window = window
        self.volatility_factor = volatility_factor
        self._required_indicators = ['atr']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on volatility breakouts.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        result = df.copy()
        
        # Calculate highest high and lowest low over the window
        result['highest_high'] = result['high'].rolling(window=self.window).max()
        result['lowest_low'] = result['low'].rolling(window=self.window).min()
        
        # Calculate breakout thresholds using ATR
        result['upper_threshold'] = result['highest_high'] + (result['atr'] * self.volatility_factor)
        result['lower_threshold'] = result['lowest_low'] - (result['atr'] * self.volatility_factor)
        
        # Generate breakout signals
        result['signal'] = 0  # Default: no signal
        
        # Buy signal: Price breaks above the upper threshold
        buy_condition = (result['close'].shift(1) <= result['upper_threshold'].shift(1)) & \
                        (result['close'] > result['upper_threshold'])
        
        # Sell signal: Price breaks below the lower threshold
        sell_condition = (result['close'].shift(1) >= result['lower_threshold'].shift(1)) & \
                         (result['close'] < result['lower_threshold'])
        
        # Apply signals
        result.loc[buy_condition, 'signal'] = 1
        result.loc[sell_condition, 'signal'] = -1
        
        # Clean up intermediate columns
        result = result.drop(columns=['highest_high', 'lowest_low', 'upper_threshold', 'lower_threshold'])
        
        logger.info(f"Generated {buy_condition.sum()} buy signals and {sell_condition.sum()} sell signals")
        return result