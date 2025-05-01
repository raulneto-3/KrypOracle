import pandas as pd
import numpy as np
import logging

from .base import Strategy

logger = logging.getLogger(__name__)

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