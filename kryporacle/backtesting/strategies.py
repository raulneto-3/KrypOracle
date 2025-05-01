import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str = "Generic Strategy"):
        """Initialize the strategy.
        
        Args:
            name (str): Strategy name
        """
        self.name = name
        self._required_indicators = []
    
    @property
    def required_indicators(self) -> List[str]:
        """List of indicators required by the strategy."""
        return self._required_indicators
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals added:
                           1 = Buy signal
                           0 = Hold/No signal
                          -1 = Sell signal
        """
        pass


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