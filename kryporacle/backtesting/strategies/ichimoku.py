import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union

from .base import Strategy
from ...analysis.ichimoku import IchimokuAnalyzer

logger = logging.getLogger(__name__)

class IchimokuStrategy(Strategy):
    """
    Estratégia de trading baseada no Ichimoku Kinko Hyo (Ichimoku Cloud).
    
    Utiliza os cinco componentes do Ichimoku Cloud para identificar tendências, 
    suportes, resistências e possíveis sinais de entrada e saída. A estratégia 
    pode ser configurada para usar diferentes componentes e filtros para gerar sinais.
    """
    
    def __init__(self, 
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_b_period: int = 52,
                 displacement: int = 26,
                 use_tk_cross: bool = True,
                 use_price_kijun_cross: bool = True,
                 use_cloud_filter: bool = True,
                 use_chikou_filter: bool = True,
                 signal_mode: str = 'combined'):
        """
        Inicializa a estratégia Ichimoku.
        
        Args:
            tenkan_period (int): Período para o cálculo do Tenkan-sen (padrão: 9)
            kijun_period (int): Período para o cálculo do Kijun-sen (padrão: 26)
            senkou_b_period (int): Período para o cálculo do Senkou Span B (padrão: 52)
            displacement (int): Deslocamento para o futuro/passado (padrão: 26)
            use_tk_cross (bool): Se True, usa cruzamentos Tenkan-Kijun para sinais
            use_price_kijun_cross (bool): Se True, usa cruzamentos do preço com Kijun para sinais
            use_cloud_filter (bool): Se True, filtra sinais baseados na posição do preço em relação à nuvem
            use_chikou_filter (bool): Se True, usa confirmação do Chikou Span
            signal_mode (str): Modo de combinação de sinais ('tk_only', 'price_kijun_only', 'combined', 'either')
        """
        super().__init__(name=f"Ichimoku ({tenkan_period}/{kijun_period}/{senkou_b_period})")
        
        # Parâmetros para cálculo do Ichimoku
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
        
        # Parâmetros para geração de sinais
        self.use_tk_cross = use_tk_cross
        self.use_price_kijun_cross = use_price_kijun_cross
        self.use_cloud_filter = use_cloud_filter
        self.use_chikou_filter = use_chikou_filter
        self.signal_mode = signal_mode
        
        # Não precisa de indicadores adicionais além dos componentes do Ichimoku
        self._required_indicators = []
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais de trading baseados no Ichimoku Cloud.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLC
            
        Returns:
            pd.DataFrame: DataFrame com sinais de trading (1 = compra, -1 = venda, 0 = sem ação)
        """
        # Copiar o DataFrame para não modificar o original
        result = df.copy()
        
        # Calcular componentes do Ichimoku
        logger.info("Calculando componentes do Ichimoku Cloud")
        result = IchimokuAnalyzer.calculate_ichimoku(
            result, 
            tenkan_period=self.tenkan_period,
            kijun_period=self.kijun_period,
            senkou_b_period=self.senkou_b_period,
            displacement=self.displacement
        )
        
        # Analisar posição em relação à nuvem
        result = IchimokuAnalyzer.analyze_cloud_position(result)
        
        # Identificar sinais básicos
        result = IchimokuAnalyzer.identify_signals(
            result,
            use_cloud_filter=self.use_cloud_filter,
            use_chikou_filter=self.use_chikou_filter
        )
        
        # Inicializar coluna de sinal final
        result['signal'] = 0
        
        # Aplicar lógica de sinal de acordo com o modo selecionado
        if self.signal_mode == 'tk_only':
            # Apenas sinais de cruzamento TK
            if self.use_tk_cross:
                # Aplicar filtro da nuvem se necessário
                if self.use_cloud_filter:
                    # Sinais de compra apenas quando o preço está acima da nuvem ou cruzando para cima
                    buy_condition = (result['tk_cross'] == 1) & \
                                  ((result['price_position'] == 1) | result['crossing_cloud'])
                    
                    # Sinais de venda apenas quando o preço está abaixo da nuvem ou cruzando para baixo
                    sell_condition = (result['tk_cross'] == -1) & \
                                   ((result['price_position'] == -1) | result['crossing_cloud'])
                else:
                    buy_condition = (result['tk_cross'] == 1)
                    sell_condition = (result['tk_cross'] == -1)
                
                # Aplicar filtro do Chikou se necessário
                if self.use_chikou_filter:
                    valid_chikou = ~result['chikou_confirmation'].isna()
                    buy_condition = buy_condition & \
                                  (result['chikou_confirmation'] == 1 | ~valid_chikou)
                    sell_condition = sell_condition & \
                                   (result['chikou_confirmation'] == -1 | ~valid_chikou)
                
                result.loc[buy_condition, 'signal'] = 1
                result.loc[sell_condition, 'signal'] = -1
                
        elif self.signal_mode == 'price_kijun_only':
            # Apenas sinais de cruzamento preço-Kijun
            if self.use_price_kijun_cross:
                # Aplicar filtro da nuvem se necessário
                if self.use_cloud_filter:
                    buy_condition = (result['price_kijun_cross'] == 1) & \
                                  ((result['price_position'] == 1) | result['crossing_cloud'])
                    sell_condition = (result['price_kijun_cross'] == -1) & \
                                   ((result['price_position'] == -1) | result['crossing_cloud'])
                else:
                    buy_condition = (result['price_kijun_cross'] == 1)
                    sell_condition = (result['price_kijun_cross'] == -1)
                
                # Aplicar filtro do Chikou se necessário
                if self.use_chikou_filter:
                    valid_chikou = ~result['chikou_confirmation'].isna()
                    buy_condition = buy_condition & \
                                  (result['chikou_confirmation'] == 1 | ~valid_chikou)
                    sell_condition = sell_condition & \
                                   (result['chikou_confirmation'] == -1 | ~valid_chikou)
                
                result.loc[buy_condition, 'signal'] = 1
                result.loc[sell_condition, 'signal'] = -1
                
        elif self.signal_mode == 'combined':
            # Requer ambos os sinais para maior confiabilidade
            if self.use_tk_cross and self.use_price_kijun_cross:
                # Sinal de compra quando ambos TK e preço-Kijun são positivos
                buy_condition = (result['tk_cross'] == 1) & \
                              (result['price_kijun_cross'] == 1)
                
                # Sinal de venda quando ambos TK e preço-Kijun são negativos
                sell_condition = (result['tk_cross'] == -1) & \
                               (result['price_kijun_cross'] == -1)
                
                # Aplicar filtro da nuvem se necessário
                if self.use_cloud_filter:
                    buy_condition = buy_condition & \
                                  ((result['price_position'] == 1) | result['crossing_cloud'])
                    sell_condition = sell_condition & \
                                   ((result['price_position'] == -1) | result['crossing_cloud'])
                
                # Aplicar filtro do Chikou se necessário
                if self.use_chikou_filter:
                    valid_chikou = ~result['chikou_confirmation'].isna()
                    buy_condition = buy_condition & \
                                  (result['chikou_confirmation'] == 1 | ~valid_chikou)
                    sell_condition = sell_condition & \
                                   (result['chikou_confirmation'] == -1 | ~valid_chikou)
                
                result.loc[buy_condition, 'signal'] = 1
                result.loc[sell_condition, 'signal'] = -1
                
        elif self.signal_mode == 'either':
            # Qualquer um dos sinais é suficiente (mais sensível)
            buy_condition = False
            sell_condition = False
            
            if self.use_tk_cross:
                buy_condition = buy_condition | (result['tk_cross'] == 1)
                sell_condition = sell_condition | (result['tk_cross'] == -1)
                
            if self.use_price_kijun_cross:
                buy_condition = buy_condition | (result['price_kijun_cross'] == 1)
                sell_condition = sell_condition | (result['price_kijun_cross'] == -1)
                
            # Aplicar filtro da nuvem se necessário
            if self.use_cloud_filter:
                buy_condition = buy_condition & \
                              ((result['price_position'] == 1) | result['crossing_cloud'])
                sell_condition = sell_condition & \
                               ((result['price_position'] == -1) | result['crossing_cloud'])
            
            # Aplicar filtro do Chikou se necessário
            if self.use_chikou_filter:
                valid_chikou = ~result['chikou_confirmation'].isna()
                buy_condition = buy_condition & \
                              (result['chikou_confirmation'] == 1 | ~valid_chikou)
                sell_condition = sell_condition & \
                               (result['chikou_confirmation'] == -1 | ~valid_chikou)
            
            result.loc[buy_condition, 'signal'] = 1
            result.loc[sell_condition, 'signal'] = -1
        
        # Contabilizar sinais gerados
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return result
    
    def __str__(self) -> str:
        """Representação em string da estratégia para logging."""
        return (f"{self.name} - "
                f"TK Cross: {self.use_tk_cross}, "
                f"Price-Kijun: {self.use_price_kijun_cross}, "
                f"Cloud Filter: {self.use_cloud_filter}, "
                f"Chikou Filter: {self.use_chikou_filter}, "
                f"Signal Mode: {self.signal_mode}")