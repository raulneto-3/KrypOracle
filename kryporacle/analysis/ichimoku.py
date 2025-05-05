import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class IchimokuAnalyzer:
    """
    Classe para calcular e analisar os componentes do Ichimoku Kinko Hyo (Ichimoku Cloud).
    
    O Ichimoku Cloud é um sistema de indicadores técnicos japonês composto por cinco linhas:
    - Tenkan-sen (Linha de Conversão)
    - Kijun-sen (Linha Base)
    - Senkou Span A (Leading Span A)
    - Senkou Span B (Leading Span B)
    - Chikou Span (Lagging Span)
    
    As linhas Senkou Span A e Senkou Span B formam a "nuvem" (cloud).
    """
    
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame, 
                           tenkan_period: int = 9,
                           kijun_period: int = 26,
                           senkou_b_period: int = 52,
                           displacement: int = 26) -> pd.DataFrame:
        """
        Calcula todos os componentes do Ichimoku Cloud.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLC
            tenkan_period (int): Período para o cálculo do Tenkan-sen (linha de conversão)
            kijun_period (int): Período para o cálculo do Kijun-sen (linha base)
            senkou_b_period (int): Período para o cálculo do Senkou Span B
            displacement (int): Deslocamento para o futuro (geralmente 26 períodos)
            
        Returns:
            pd.DataFrame: DataFrame com as componentes do Ichimoku Cloud adicionadas
        """
        result = df.copy()
        
        # Verifica se as colunas necessárias existem
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in result.columns:
                raise ValueError(f"Coluna {col} não encontrada no DataFrame")
        
        # Tenkan-sen (Linha de Conversão): (máximo(9) + mínimo(9)) / 2
        tenkan_high = result['high'].rolling(window=tenkan_period).max()
        tenkan_low = result['low'].rolling(window=tenkan_period).min()
        result['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Linha Base): (máximo(26) + mínimo(26)) / 2
        kijun_high = result['high'].rolling(window=kijun_period).max()
        kijun_low = result['low'].rolling(window=kijun_period).min()
        result['kijun_sen'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, deslocado 26 períodos para o futuro
        result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B): (máximo(52) + mínimo(52)) / 2, deslocado 26 períodos para o futuro
        senkou_b_high = result['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = result['low'].rolling(window=senkou_b_period).min()
        result['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span): preço de fechamento, deslocado 26 períodos para trás
        result['chikou_span'] = result['close'].shift(-displacement)
        
        # Adicionar a diferença entre Senkou Span A e Senkou Span B (útil para análises)
        result['cloud_thickness'] = result['senkou_span_a'] - result['senkou_span_b']
        
        # Adicionar a cor da nuvem (1: bullish, -1: bearish)
        result['cloud_color'] = np.where(result['senkou_span_a'] > result['senkou_span_b'], 1, -1)
        
        logger.info("Ichimoku Cloud components calculated successfully")
        return result
    
    @staticmethod
    def analyze_cloud_position(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analisa a posição do preço em relação à nuvem Ichimoku.
        
        Args:
            df (pd.DataFrame): DataFrame com componentes do Ichimoku já calculados
            
        Returns:
            pd.DataFrame: DataFrame com análise de posição adicionada
        """
        result = df.copy()
        
        # Verifica se as colunas necessárias existem
        required_cols = ['close', 'senkou_span_a', 'senkou_span_b']
        for col in required_cols:
            if col not in result.columns:
                raise ValueError(f"Coluna {col} não encontrada. Execute calculate_ichimoku primeiro.")
        
        # Adicionar posição do preço em relação à nuvem
        # 1: acima da nuvem (bullish)
        # 0: dentro da nuvem (neutro)
        # -1: abaixo da nuvem (bearish)
        result['price_position'] = 0
        
        # Preço acima da nuvem
        result.loc[(result['close'] > result['senkou_span_a']) & 
                  (result['close'] > result['senkou_span_b']), 'price_position'] = 1
        
        # Preço abaixo da nuvem
        result.loc[(result['close'] < result['senkou_span_a']) & 
                  (result['close'] < result['senkou_span_b']), 'price_position'] = -1
        
        # Adicionar condição para verificar se o preço está cruzando a nuvem
        result['crossing_cloud'] = False
        
        # Cruzando de baixo para cima (potencial sinal de compra)
        result.loc[(result['close'] > result['senkou_span_a']) & 
                  (result['close'] > result['senkou_span_b']) & 
                  (result['close'].shift(1) < result['senkou_span_a'].shift(1)) | 
                  (result['close'].shift(1) < result['senkou_span_b'].shift(1)), 'crossing_cloud'] = True
        
        # Cruzando de cima para baixo (potencial sinal de venda)
        result.loc[(result['close'] < result['senkou_span_a']) & 
                  (result['close'] < result['senkou_span_b']) & 
                  (result['close'].shift(1) > result['senkou_span_a'].shift(1)) | 
                  (result['close'].shift(1) > result['senkou_span_b'].shift(1)), 'crossing_cloud'] = True
        
        return result
    
    @staticmethod
    def identify_signals(df: pd.DataFrame, 
                        use_cloud_filter: bool = True,
                        use_chikou_filter: bool = True) -> pd.DataFrame:
        """
        Identifica sinais de trading baseados no Ichimoku Cloud.
        
        Args:
            df (pd.DataFrame): DataFrame com componentes do Ichimoku já calculados
            use_cloud_filter (bool): Se True, filtra sinais baseados na posição do preço em relação à nuvem
            use_chikou_filter (bool): Se True, usa o Chikou Span como filtro adicional
            
        Returns:
            pd.DataFrame: DataFrame com sinais de trading identificados
        """
        result = df.copy()
        
        # Verificar cruzamento do Tenkan-sen e Kijun-sen (TK Cross)
        result['tk_cross'] = 0
        
        # TK Cross de baixo para cima (potencial sinal de compra)
        tk_cross_up = (result['tenkan_sen'].shift(1) <= result['kijun_sen'].shift(1)) & \
                      (result['tenkan_sen'] > result['kijun_sen'])
        result.loc[tk_cross_up, 'tk_cross'] = 1
        
        # TK Cross de cima para baixo (potencial sinal de venda)
        tk_cross_down = (result['tenkan_sen'].shift(1) >= result['kijun_sen'].shift(1)) & \
                       (result['tenkan_sen'] < result['kijun_sen'])
        result.loc[tk_cross_down, 'tk_cross'] = -1
        
        # Verificar cruzamento do preço com o Kijun-sen (Price-Kijun Cross)
        result['price_kijun_cross'] = 0
        
        # Preço cruza Kijun-sen de baixo para cima (potencial sinal de compra)
        price_kijun_up = (result['close'].shift(1) <= result['kijun_sen'].shift(1)) & \
                         (result['close'] > result['kijun_sen'])
        result.loc[price_kijun_up, 'price_kijun_cross'] = 1
        
        # Preço cruza Kijun-sen de cima para baixo (potencial sinal de venda)
        price_kijun_down = (result['close'].shift(1) >= result['kijun_sen'].shift(1)) & \
                          (result['close'] < result['kijun_sen'])
        result.loc[price_kijun_down, 'price_kijun_cross'] = -1
        
        # Aplicar filtros
        if use_cloud_filter:
            # Análise de posição em relação à nuvem (caso ainda não esteja calculada)
            if 'price_position' not in result.columns:
                result = IchimokuAnalyzer.analyze_cloud_position(result)
                
        if use_chikou_filter:
            # Verifica se o Chikou Span está acima/abaixo do preço de 26 períodos atrás
            # Isso só é válido para os últimos dados do DataFrame que têm o Chikou Span completo
            valid_chikou = ~result['chikou_span'].isna()
            result.loc[valid_chikou, 'chikou_confirmation'] = np.where(
                result.loc[valid_chikou, 'chikou_span'] > result.loc[valid_chikou, 'close'],
                1, -1
            )
        
        return result