import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union

from .base import Strategy
from ...analysis.volume_profile import VolumeProfileAnalyzer

logger = logging.getLogger(__name__)

class VolumeProfileStrategy(Strategy):
    """
    Estratégia baseada na análise do perfil de volume em diferentes níveis de preço.
    
    Identifica zonas de valor, acumulação/distribuição e gera sinais baseados
    em divergências entre volume e preço e áreas de interesse.
    """
    
    def __init__(self, 
                 price_bins: int = 10,
                 value_area_threshold: float = 70.0,
                 divergence_window: int = 14,
                 use_vwap: bool = True,
                 volume_profile_window: Optional[int] = 100,
                 cmf_window: int = 20,
                 cmf_threshold: float = 0.05,
                 reset_period: Optional[str] = 'D'):
        """
        Inicializa a estratégia de perfil de volume.
        
        Args:
            price_bins (int): Número de faixas de preço para análise de volume
            value_area_threshold (float): Percentual de volume para definir a área de valor
            divergence_window (int): Janela para detecção de divergências
            use_vwap (bool): Se True, usa VWAP para confirmação
            volume_profile_window (int): Número de períodos para calcular perfil de volume
            cmf_window (int): Janela para Chaikin Money Flow
            cmf_threshold (float): Limiar para zonas de acumulação/distribuição
            reset_period (str): Período para resetar cálculos ('D', 'W', 'M', None)
        """
        super().__init__(
            name=f"Volume Profile ({price_bins}, VA-{value_area_threshold}%, CMF-{cmf_window})"
        )
        
        self.price_bins = price_bins
        self.value_area_threshold = value_area_threshold
        self.divergence_window = divergence_window
        self.use_vwap = use_vwap
        self.volume_profile_window = volume_profile_window
        self.cmf_window = cmf_window
        self.cmf_threshold = cmf_threshold
        self.reset_period = reset_period
        
        # Indicadores necessários
        self._required_indicators = []
        
        logger.info(f"VolumeProfileStrategy inicializada: {self.name}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais de trading baseados na análise de perfil de volume.
        
        Args:
            df: DataFrame com dados OHLC e indicadores
            
        Returns:
            DataFrame com sinais de trading (1 = compra, -1 = venda, 0 = sem ação)
        """
        result = df.copy()
        
        # Verificar se temos coluna de volume
        if 'volume' not in result.columns:
            logger.error("Dados de volume não encontrados. Esta estratégia requer dados de volume.")
            return result
        
        # Inicializar coluna de sinal
        result['signal'] = 0
        
        # Verificar se temos dados suficientes
        min_periods = max(self.divergence_window, self.cmf_window, 
                          self.volume_profile_window or 0)
        
        if len(result) < min_periods:
            logger.warning(
                f"Dados insuficientes para análise de perfil de volume. "
                f"Necessário pelo menos {min_periods} candles."
            )
            return result
        
        # 1. Calcular perfil de volume
        result = VolumeProfileAnalyzer.calculate_volume_profile(
            result,
            n_bins=self.price_bins,
            window=self.volume_profile_window
        )
        
        # 2. Identificar áreas de valor
        result = VolumeProfileAnalyzer.identify_value_areas(
            result,
            threshold_pct=self.value_area_threshold
        )
        
        # 3. Detectar divergências entre preço e volume
        result = VolumeProfileAnalyzer.detect_volume_divergence(
            result,
            window=self.divergence_window
        )
        
        # 4. Identificar zonas de acumulação e distribuição
        result = VolumeProfileAnalyzer.identify_accumulation_distribution(
            result,
            window=self.cmf_window
        )
        
        # 5. Calcular VWAP se requisitado
        if self.use_vwap:
            result = VolumeProfileAnalyzer.calculate_vwap(
                result,
                reset_period=self.reset_period
            )
        
        # 6. Gerar sinais de trading
        
        # 6.1 Sinais de compra
        buy_condition = False
        
        # Divergência positiva (preço em baixa, volume em alta)
        buy_condition |= result['bullish_divergence']
        
        # Em zona de acumulação (CMF positivo acima do limiar)
        buy_condition |= result['accumulation_zone']
        
        # Preço próximo da value area low (suporte)
        if 'val' in result.columns and 'low' in result.columns:
            # Preço próximo do suporte (dentro de 1%)
            val_proximity = abs(result['low'] - result['val']) / result['val'] < 0.01
            buy_condition |= val_proximity
        
        # Confirmação com VWAP (preço cruzando acima do VWAP)
        if self.use_vwap and 'vwap' in result.columns:
            vwap_crossover = (result['close'].shift(1) < result['vwap'].shift(1)) & \
                            (result['close'] > result['vwap'])
            buy_condition &= vwap_crossover
        
        # 6.2 Sinais de venda
        sell_condition = False
        
        # Divergência negativa (preço em alta, volume em baixa)
        sell_condition |= result['bearish_divergence']
        
        # Em zona de distribuição (CMF negativo abaixo do limiar negativo)
        sell_condition |= result['distribution_zone']
        
        # Preço próximo da value area high (resistência)
        if 'vah' in result.columns and 'high' in result.columns:
            # Preço próximo da resistência (dentro de 1%)
            vah_proximity = abs(result['high'] - result['vah']) / result['vah'] < 0.01
            sell_condition |= vah_proximity
        
        # Confirmação com VWAP (preço cruzando abaixo do VWAP)
        if self.use_vwap and 'vwap' in result.columns:
            vwap_crossunder = (result['close'].shift(1) > result['vwap'].shift(1)) & \
                             (result['close'] < result['vwap'])
            sell_condition &= vwap_crossunder
        
        # Aplicar sinais
        result.loc[buy_condition, 'signal'] = 1
        result.loc[sell_condition, 'signal'] = -1
        
        # Adicionar coluna com tipo de sinal para análise
        result['signal_type'] = 'none'
        result.loc[result['bullish_divergence'] & (result['signal'] == 1), 'signal_type'] = 'bullish_divergence'
        result.loc[result['bearish_divergence'] & (result['signal'] == -1), 'signal_type'] = 'bearish_divergence'
        result.loc[result['accumulation_zone'] & (result['signal'] == 1), 'signal_type'] = 'accumulation'
        result.loc[result['distribution_zone'] & (result['signal'] == -1), 'signal_type'] = 'distribution'
        
        # Contar e registrar sinais gerados
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        logger.info(f"Gerados {buy_signals} sinais de compra e {sell_signals} sinais de venda")
        
        return result
    
    def __str__(self) -> str:
        """Representação em string da estratégia para logging."""
        return (f"{self.name} - "
                f"Bins: {self.price_bins}, "
                f"Value Area: {self.value_area_threshold}%, "
                f"Div. Window: {self.divergence_window}, "
                f"CMF ({self.cmf_window}, {self.cmf_threshold}), "
                f"VWAP: {self.use_vwap}")