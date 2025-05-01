import pandas as pd
import numpy as np
import logging

from .base import Strategy

logger = logging.getLogger(__name__)

class MovingAverageCrossover(Strategy):
    """Moving Average Crossover strategy.
    
    Generates buy signals when the fast MA crosses above the slow MA,
    and sell signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """Initialize the Moving Average Crossover strategy.
        
        Args:
            fast_period (int): Period for the fast moving average
            slow_period (int): Period for the slow moving average
        """
        super().__init__(name=f"MA Crossover ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._required_indicators = [f'sma_{fast_period}', f'sma_{slow_period}']
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on MA crossover.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        result = df.copy()
        
        fast_ma = f'sma_{self.fast_period}'
        slow_ma = f'sma_{self.slow_period}'
        
        # Generate crossover signals
        result['signal'] = 0  # Default: no signal
        
        # Buy signal: fast MA crosses above slow MA
        buy_condition = (result[fast_ma].shift(1) <= result[slow_ma].shift(1)) & \
                        (result[fast_ma] > result[slow_ma])
        result.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA
        sell_condition = (result[fast_ma].shift(1) >= result[slow_ma].shift(1)) & \
                         (result[fast_ma] < result[slow_ma])
        result.loc[sell_condition, 'signal'] = -1
        
        logger.info(f"Generated {buy_condition.sum()} buy signals and {sell_condition.sum()} sell signals")
        return result