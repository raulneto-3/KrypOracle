import pandas as pd
import numpy as np
import logging

from .base import Strategy

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