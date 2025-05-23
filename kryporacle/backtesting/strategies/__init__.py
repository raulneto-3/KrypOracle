# Import base strategy
from .base import Strategy

# Import basic strategies
from .moving_average import MovingAverageCrossover
from .rsi_based import RSIStrategy
from .bollinger_bands import BollingerBandsStrategy
from .macd_based import MACDStrategy, DualMACDStrategy

# Import advanced strategies
from .trend_following import TrendFollowingStrategy
from .volatility_based import VolatilityBreakoutStrategy
from .divergence_based import DivergenceStrategy
from .candlestick_strategy import CandlestickPatternStrategy, CandlestickFilterStrategy  
from .fibonacci import FibonacciRetracementStrategy
from .ichimoku import IchimokuStrategy
from .multi_timeframe import MultiTimeframeStrategy
from .volume_profile import VolumeProfileStrategy

# Expose all strategies at package level
__all__ = [
    'Strategy',
    # Basic strategies
    'MovingAverageCrossover',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'MACDStrategy',
    # Advanced strategies
    'DualMACDStrategy',
    'TrendFollowingStrategy',
    'VolatilityBreakoutStrategy',
    'DivergenceStrategy',
    'CandlestickPatternStrategy',  
    'CandlestickFilterStrategy',
    'FibonacciRetracementStrategy',
    'IchimokuStrategy',
    'MultiTimeframeStrategy',
    'VolumeProfileStrategy',

]