"""
Módulo para identificação de padrões de candlestick relevantes para análise técnica em criptomoedas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class CandlestickPatterns:
    """
    Classe que fornece métodos para identificar padrões de candlestick em dados OHLC.
    Foca em padrões especialmente relevantes para o mercado de criptomoedas.
    """
    
    @staticmethod
    def is_doji(open_price: float, high_price: float, low_price: float, close_price: float,
               tolerance: float = 0.05) -> bool:
        """
        Identifica um padrão Doji (corpo muito pequeno).
        
        Args:
            open_price: Preço de abertura do candle
            high_price: Preço máximo do candle
            low_price: Preço mínimo do candle
            close_price: Preço de fechamento do candle
            tolerance: Tolerância para o tamanho do corpo em relação ao total do candle
        
        Returns:
            bool: True se for um Doji, False caso contrário
        """
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        # Evitar divisão por zero
        if total_range == 0:
            return False
        
        # Um Doji tem corpo muito pequeno em relação ao range total
        return body_size / total_range <= tolerance

    @staticmethod
    def is_hammer(open_price: float, high_price: float, low_price: float, close_price: float,
                 body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> bool:
        """
        Identifica um padrão Hammer (martelo).
        
        Args:
            open_price: Preço de abertura do candle
            high_price: Preço máximo do candle
            low_price: Preço mínimo do candle
            close_price: Preço de fechamento do candle
            body_ratio: Máxima proporção do corpo em relação ao range total
            shadow_ratio: Mínima proporção da sombra inferior em relação ao corpo
        
        Returns:
            bool: True se for um Hammer, False caso contrário
        """
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            return False
            
        # Identifica o topo e a base do corpo
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        # Calcula os tamanhos das sombras
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        
        # Condições para um Hammer:
        # 1. Corpo pequeno
        # 2. Sombra inferior longa
        # 3. Sombra superior pequena ou inexistente
        return (body_size / total_range <= body_ratio and 
                lower_shadow >= body_size * shadow_ratio and
                upper_shadow < body_size)

    @staticmethod
    def is_shooting_star(open_price: float, high_price: float, low_price: float, close_price: float,
                        body_ratio: float = 0.3, shadow_ratio: float = 2.0) -> bool:
        """
        Identifica um padrão Shooting Star (estrela cadente).
        
        Args:
            open_price: Preço de abertura do candle
            high_price: Preço máximo do candle
            low_price: Preço mínimo do candle
            close_price: Preço de fechamento do candle
            body_ratio: Máxima proporção do corpo em relação ao range total
            shadow_ratio: Mínima proporção da sombra superior em relação ao corpo
        
        Returns:
            bool: True se for um Shooting Star, False caso contrário
        """
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            return False
            
        # Identifica o topo e a base do corpo
        body_top = max(open_price, close_price)
        body_bottom = min(open_price, close_price)
        
        # Calcula os tamanhos das sombras
        upper_shadow = high_price - body_top
        lower_shadow = body_bottom - low_price
        
        # Condições para um Shooting Star:
        # 1. Corpo pequeno
        # 2. Sombra superior longa
        # 3. Sombra inferior pequena ou inexistente
        return (body_size / total_range <= body_ratio and 
                upper_shadow >= body_size * shadow_ratio and
                lower_shadow < body_size)

    @staticmethod
    def is_bullish_engulfing(current_open: float, current_close: float, 
                            prev_open: float, prev_close: float) -> bool:
        """
        Identifica um padrão Bullish Engulfing (engolfo de alta).
        
        Args:
            current_open: Preço de abertura do candle atual
            current_close: Preço de fechamento do candle atual
            prev_open: Preço de abertura do candle anterior
            prev_close: Preço de fechamento do candle anterior
        
        Returns:
            bool: True se for um Bullish Engulfing, False caso contrário
        """
        # Condições para um Bullish Engulfing:
        # 1. Candle anterior é de baixa (fechamento < abertura)
        # 2. Candle atual é de alta (fechamento > abertura)
        # 3. Abertura atual abaixo do fechamento anterior
        # 4. Fechamento atual acima da abertura anterior
        return (prev_close < prev_open and  # Candle anterior de baixa
                current_close > current_open and  # Candle atual de alta
                current_open < prev_close and  # Corpo atual "engole" o corpo anterior
                current_close > prev_open)

    @staticmethod
    def is_bearish_engulfing(current_open: float, current_close: float, 
                            prev_open: float, prev_close: float) -> bool:
        """
        Identifica um padrão Bearish Engulfing (engolfo de baixa).
        
        Args:
            current_open: Preço de abertura do candle atual
            current_close: Preço de fechamento do candle atual
            prev_open: Preço de abertura do candle anterior
            prev_close: Preço de fechamento do candle anterior
        
        Returns:
            bool: True se for um Bearish Engulfing, False caso contrário
        """
        # Condições para um Bearish Engulfing:
        # 1. Candle anterior é de alta (fechamento > abertura)
        # 2. Candle atual é de baixa (fechamento < abertura)
        # 3. Abertura atual acima do fechamento anterior
        # 4. Fechamento atual abaixo da abertura anterior
        return (prev_close > prev_open and  # Candle anterior de alta
                current_close < current_open and  # Candle atual de baixa
                current_open > prev_close and  # Corpo atual "engole" o corpo anterior
                current_close < prev_open)

    @staticmethod
    def is_morning_star(df: pd.DataFrame, idx: int, body_ratio: float = 0.3) -> bool:
        """
        Identifica um padrão Morning Star (estrela da manhã).
        
        Args:
            df: DataFrame com dados OHLC
            idx: Índice atual no DataFrame
            body_ratio: Máxima proporção do corpo para o candle do meio
        
        Returns:
            bool: True se for um Morning Star, False caso contrário
        """
        # Precisamos de pelo menos 3 candles
        if idx < 2:
            return False
        
        # Obter dados dos 3 candles
        candle1 = df.iloc[idx-2]  # Primeiro candle (bearish)
        candle2 = df.iloc[idx-1]  # Segundo candle (pequeno)
        candle3 = df.iloc[idx]    # Terceiro candle (bullish)
        
        # Calcular tamanhos dos corpos
        body1_size = abs(candle1['close'] - candle1['open'])
        body2_size = abs(candle2['close'] - candle2['open'])
        body3_size = abs(candle3['close'] - candle3['open'])
        
        total_range2 = candle2['high'] - candle2['low']
        if total_range2 == 0:
            return False
        
        # Condições para Morning Star:
        # 1. Primeiro candle é de baixa
        # 2. Segundo candle tem corpo pequeno (potencial doji)
        # 3. Terceiro candle é de alta
        # 4. Terceiro candle fecha acima da metade do primeiro candle
        return (candle1['close'] < candle1['open'] and  # Primeiro candle bearish
                body2_size / total_range2 <= body_ratio and  # Segundo candle pequeno
                candle3['close'] > candle3['open'] and  # Terceiro candle bullish
                candle3['close'] > (candle1['open'] + candle1['close']) / 2)  # Fecha acima da metade do primeiro

    @staticmethod
    def is_evening_star(df: pd.DataFrame, idx: int, body_ratio: float = 0.3) -> bool:
        """
        Identifica um padrão Evening Star (estrela da tarde).
        
        Args:
            df: DataFrame com dados OHLC
            idx: Índice atual no DataFrame
            body_ratio: Máxima proporção do corpo para o candle do meio
        
        Returns:
            bool: True se for um Evening Star, False caso contrário
        """
        # Precisamos de pelo menos 3 candles
        if idx < 2:
            return False
        
        # Obter dados dos 3 candles
        candle1 = df.iloc[idx-2]  # Primeiro candle (bullish)
        candle2 = df.iloc[idx-1]  # Segundo candle (pequeno)
        candle3 = df.iloc[idx]    # Terceiro candle (bearish)
        
        # Calcular tamanhos dos corpos
        body1_size = abs(candle1['close'] - candle1['open'])
        body2_size = abs(candle2['close'] - candle2['open'])
        body3_size = abs(candle3['close'] - candle3['open'])
        
        total_range2 = candle2['high'] - candle2['low']
        if total_range2 == 0:
            return False
        
        # Condições para Evening Star:
        # 1. Primeiro candle é de alta
        # 2. Segundo candle tem corpo pequeno (potencial doji)
        # 3. Terceiro candle é de baixa
        # 4. Terceiro candle fecha abaixo da metade do primeiro candle
        return (candle1['close'] > candle1['open'] and  # Primeiro candle bullish
                body2_size / total_range2 <= body_ratio and  # Segundo candle pequeno
                candle3['close'] < candle3['open'] and  # Terceiro candle bearish
                candle3['close'] < (candle1['open'] + candle1['close']) / 2)  # Fecha abaixo da metade do primeiro

    @staticmethod
    def is_three_white_soldiers(df: pd.DataFrame, idx: int, min_body_ratio: float = 0.6) -> bool:
        """
        Identifica padrão Three White Soldiers (três soldados brancos).
        
        Args:
            df: DataFrame com dados OHLC
            idx: Índice atual no DataFrame
            min_body_ratio: Proporção mínima do corpo em relação ao range total
        
        Returns:
            bool: True se for Three White Soldiers, False caso contrário
        """
        # Precisamos de pelo menos 3 candles
        if idx < 2:
            return False
        
        # Verificar três candles consecutivos de alta com corpos grandes
        for i in range(3):
            candle = df.iloc[idx-i]
            
            # Deve ser um candle de alta
            if candle['close'] <= candle['open']:
                return False
                
            # Corpo deve ser grande em relação ao range total
            body_size = candle['close'] - candle['open']
            total_range = candle['high'] - candle['low']
            
            if total_range == 0 or body_size / total_range < min_body_ratio:
                return False
                
            # Cada candle deve abrir dentro do corpo do anterior e fechar acima do anterior
            # (exceto o primeiro)
            if i > 0:
                prev_candle = df.iloc[idx-i+1]
                if (candle['open'] < prev_candle['open'] or 
                    candle['close'] <= prev_candle['close']):
                    return False
        
        return True

    @staticmethod
    def is_three_black_crows(df: pd.DataFrame, idx: int, min_body_ratio: float = 0.6) -> bool:
        """
        Identifica padrão Three Black Crows (três corvos pretos).
        
        Args:
            df: DataFrame com dados OHLC
            idx: Índice atual no DataFrame
            min_body_ratio: Proporção mínima do corpo em relação ao range total
        
        Returns:
            bool: True se for Three Black Crows, False caso contrário
        """
        # Precisamos de pelo menos 3 candles
        if idx < 2:
            return False
        
        # Verificar três candles consecutivos de baixa com corpos grandes
        for i in range(3):
            candle = df.iloc[idx-i]
            
            # Deve ser um candle de baixa
            if candle['close'] >= candle['open']:
                return False
                
            # Corpo deve ser grande em relação ao range total
            body_size = candle['open'] - candle['close']
            total_range = candle['high'] - candle['low']
            
            if total_range == 0 or body_size / total_range < min_body_ratio:
                return False
                
            # Cada candle deve abrir dentro do corpo do anterior e fechar abaixo do anterior
            # (exceto o primeiro)
            if i > 0:
                prev_candle = df.iloc[idx-i+1]
                if (candle['open'] > prev_candle['open'] or 
                    candle['close'] >= prev_candle['close']):
                    return False
        
        return True

    @staticmethod
    def identify_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifica todos os padrões de candlestick no DataFrame.
        
        Args:
            df: DataFrame com dados OHLC (open, high, low, close)
            
        Returns:
            DataFrame com colunas adicionais para cada padrão identificado
        """
        result = df.copy()
        
        # Inicializar colunas para cada padrão
        patterns = {
            'doji': False,
            'hammer': False,
            'shooting_star': False,
            'bullish_engulfing': False,
            'bearish_engulfing': False,
            'morning_star': False,
            'evening_star': False,
            'three_white_soldiers': False,
            'three_black_crows': False
        }
        
        for col in patterns.keys():
            result[f'pattern_{col}'] = False
        
        # Identificar padrões para cada candle
        for i in range(len(result)):
            # Padrões que exigem apenas o candle atual
            if i >= 0:
                candle = result.iloc[i]
                result.at[result.index[i], 'pattern_doji'] = CandlestickPatterns.is_doji(
                    candle['open'], candle['high'], candle['low'], candle['close']
                )
                result.at[result.index[i], 'pattern_hammer'] = CandlestickPatterns.is_hammer(
                    candle['open'], candle['high'], candle['low'], candle['close']
                )
                result.at[result.index[i], 'pattern_shooting_star'] = CandlestickPatterns.is_shooting_star(
                    candle['open'], candle['high'], candle['low'], candle['close']
                )
            
            # Padrões que exigem o candle atual e o anterior
            if i >= 1:
                current = result.iloc[i]
                prev = result.iloc[i-1]
                
                result.at[result.index[i], 'pattern_bullish_engulfing'] = CandlestickPatterns.is_bullish_engulfing(
                    current['open'], current['close'], prev['open'], prev['close']
                )
                result.at[result.index[i], 'pattern_bearish_engulfing'] = CandlestickPatterns.is_bearish_engulfing(
                    current['open'], current['close'], prev['open'], prev['close']
                )
            
            # Padrões que exigem três candles
            if i >= 2:
                result.at[result.index[i], 'pattern_morning_star'] = CandlestickPatterns.is_morning_star(
                    result, i
                )
                result.at[result.index[i], 'pattern_evening_star'] = CandlestickPatterns.is_evening_star(
                    result, i
                )
                result.at[result.index[i], 'pattern_three_white_soldiers'] = CandlestickPatterns.is_three_white_soldiers(
                    result, i
                )
                result.at[result.index[i], 'pattern_three_black_crows'] = CandlestickPatterns.is_three_black_crows(
                    result, i
                )
        
        return result