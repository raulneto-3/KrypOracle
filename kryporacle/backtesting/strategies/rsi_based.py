import pandas as pd
import numpy as np
import logging

from .base import Strategy

logger = logging.getLogger(__name__)

class RSIStrategy(Strategy):
    """RSI-based trading strategy.
    
    Generates buy signals when RSI crosses above oversold level,
    and sell signals when RSI crosses below overbought level.
    """
    
    def __init__(self, rsi_period: int = 14,
               overbought: int = 70, oversold: int = 30):
        """Initialize the RSI strategy.
        
        Args:
            rsi_period (int): Period for RSI calculation
            overbought (int): Overbought level (default: 70)
            oversold (int): Oversold level (default: 30)
        """
        super().__init__(name=f"RSI Strategy ({rsi_period})")
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self._required_indicators = ['rsi']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on RSI levels.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        result = df.copy()
        
        # Generate RSI signals
        result['signal'] = 0  # Default: no signal
        
        # Buy signal: RSI crosses above oversold level
        buy_condition = (result['rsi'].shift(1) <= self.oversold) & \
                        (result['rsi'] > self.oversold)
        result.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: RSI crosses below overbought level
        sell_condition = (result['rsi'].shift(1) >= self.overbought) & \
                         (result['rsi'] < self.overbought)
        result.loc[sell_condition, 'signal'] = -1
        
        logger.info(f"Generated {buy_condition.sum()} buy signals and {sell_condition.sum()} sell signals")
        return result