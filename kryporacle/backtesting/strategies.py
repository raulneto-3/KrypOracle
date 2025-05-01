"""
Este arquivo importa todas as estratégias básicas do subpacote 'strategies'
para manter a compatibilidade com código existente.
"""

from .strategies import (
    Strategy,
    MovingAverageCrossover,
    RSIStrategy,
    BollingerBandsStrategy,
    MACDStrategy
)

__all__ = [
    'Strategy',
    'MovingAverageCrossover',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'MACDStrategy',
]