"""
Este arquivo importa todas as estratégias avançadas do subpacote 'strategies'
para manter a compatibilidade com código existente.
"""

from .strategies import (
    Strategy,
    TrendFollowingStrategy, 
    DualMACDStrategy,
    VolatilityBreakoutStrategy,
    DivergenceStrategy
)

__all__ = [
    'Strategy',
    'TrendFollowingStrategy',
    'DualMACDStrategy',
    'VolatilityBreakoutStrategy',
    'DivergenceStrategy',
]