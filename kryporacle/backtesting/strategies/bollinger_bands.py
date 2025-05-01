import pandas as pd
import numpy as np
import logging

from .base import Strategy

logger = logging.getLogger(__name__)

class BollingerBandsStrategy(Strategy):
    """Bollinger Bands trading strategy.
    
    Generates buy signals when price crosses above the lower band,
    and sell signals when price crosses below the upper band.
    """
    
    def __init__(self, window: int = 20, std_dev: float = 2.0):
        """Initialize the Bollinger Bands strategy.
        
        Args:
            window (int): Window size for Bollinger Bands calculation
            std_dev (float): Number of standard deviations for bands
        """
        super().__init__(name=f"Bollinger Bands ({window}, {std_dev}σ)")
        self.window = window
        self.std_dev = std_dev
        self._required_indicators = ['bb_upper', 'bb_middle', 'bb_lower']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on Bollinger Bands.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        result = df.copy()
        
        # Generate Bollinger Bands signals
        result['signal'] = 0  # Default: no signal
        
        # Buy signal: Price crosses above the lower band
        buy_condition = (result['close'].shift(1) <= result['bb_lower'].shift(1)) & \
                        (result['close'] > result['bb_lower'])
        result.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: Price crosses below the upper band
        sell_condition = (result['close'].shift(1) >= result['bb_upper'].shift(1)) & \
                         (result['close'] < result['bb_upper'])
        result.loc[sell_condition, 'signal'] = -1
        
        logger.info(f"Generated {buy_condition.sum()} buy signals and {sell_condition.sum()} sell signals")
        return result