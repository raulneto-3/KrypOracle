import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class FibonacciAnalyzer:
    """
    Classe utilitária para cálculos de níveis de retração e extensão de Fibonacci.
    Fornece métodos para identificar pontos de swing e calcular níveis de retração.
    """
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """
        Calcula os níveis de retração de Fibonacci entre um ponto alto e um ponto baixo.
        
        Args:
            high (float): O ponto mais alto do preço
            low (float): O ponto mais baixo do preço
            
        Returns:
            Dict[str, float]: Dicionário com os níveis de retração de Fibonacci
        """
        diff = high - low
        
        return {
            "0.0": low,
            "0.236": low + 0.236 * diff,
            "0.382": low + 0.382 * diff,
            "0.5": low + 0.5 * diff,
            "0.618": low + 0.618 * diff,
            "0.786": low + 0.786 * diff,
            "1.0": high
        }
    
    @staticmethod
    def fibonacci_extension(high: float, low: float) -> Dict[str, float]:
        """
        Calcula os níveis de extensão de Fibonacci além do ponto alto.
        
        Args:
            high (float): O ponto mais alto do preço
            low (float): O ponto mais baixo do preço
            
        Returns:
            Dict[str, float]: Dicionário com os níveis de extensão de Fibonacci
        """
        diff = high - low
        
        return {
            "1.0": high,
            "1.236": high + 0.236 * diff,
            "1.382": high + 0.382 * diff,
            "1.5": high + 0.5 * diff,
            "1.618": high + 0.618 * diff,
            "1.786": high + 0.786 * diff,
            "2.0": high + diff,
            "2.618": high + 1.618 * diff
        }
    
    @staticmethod
    def detect_swing_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Detecta pontos de swing (máximos e mínimos locais) em uma série temporal.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLC
            window (int): Tamanho da janela para procurar máximos e mínimos locais
            
        Returns:
            pd.DataFrame: DataFrame com colunas adicionais para pontos de swing
        """
        result = df.copy()
        
        # Inicializar colunas para pontos de swing
        result['swing_high'] = False
        result['swing_low'] = False
        
        # Detectar máximos locais (swing highs)
        for i in range(window, len(result) - window):
            # Verificar se é um máximo local
            left_window = result['high'].iloc[i-window:i]
            right_window = result['high'].iloc[i+1:i+window+1]
            current = result['high'].iloc[i]
            
            if all(current >= left_window) and all(current >= right_window):
                result.loc[result.index[i], 'swing_high'] = True
        
        # Detectar mínimos locais (swing lows)
        for i in range(window, len(result) - window):
            # Verificar se é um mínimo local
            left_window = result['low'].iloc[i-window:i]
            right_window = result['low'].iloc[i+1:i+window+1]
            current = result['low'].iloc[i]
            
            if all(current <= left_window) and all(current <= right_window):
                result.loc[result.index[i], 'swing_low'] = True
        
        logger.info(f"Detected {result['swing_high'].sum()} swing highs and {result['swing_low'].sum()} swing lows")
        return result
    
    @staticmethod
    def get_recent_swing_points(df: pd.DataFrame, lookback: int = 100) -> Tuple[Optional[float], Optional[float]]:
        """
        Obtém os pontos de swing mais recentes (máximo e mínimo) dentro da janela de lookback.
        
        Args:
            df (pd.DataFrame): DataFrame com pontos de swing identificados
            lookback (int): Número de períodos a olhar para trás
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (Último swing high, Último swing low)
        """
        # Limitar a busca à janela de lookback
        window = df.iloc[-lookback:] if len(df) > lookback else df
        
        # Encontrar o último swing high
        swing_highs = window[window['swing_high'] == True]
        last_swing_high = swing_highs['high'].max() if not swing_highs.empty else None
        
        # Encontrar o último swing low
        swing_lows = window[window['swing_low'] == True]
        last_swing_low = swing_lows['low'].min() if not swing_lows.empty else None
        
        return last_swing_high, last_swing_low