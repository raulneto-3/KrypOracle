import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union

from .base import Strategy
from ...analysis.multi_timeframe import MultiTimeframeAnalyzer
from ...processing.normalization import TIMEFRAME_MINUTES

logger = logging.getLogger(__name__)

class MultiTimeframeStrategy(Strategy):
    """
    Analisa múltiplos timeframes simultaneamente.
    
    Gera sinais quando há confirmação da tendência em diferentes períodos
    de tempo, aumentando a confiabilidade do sinal.
    """
    
    def __init__(self, 
                 primary_tf: str = '1h',
                 higher_tf: str = '4h',
                 ma_period: int = 20,
                 base_strategy: Optional[Strategy] = None,
                 confirmation_periods: int = 3,
                 require_alignment: bool = True):
        """
        Inicializa a estratégia de múltiplos timeframes.
        
        Args:
            primary_tf (str): Timeframe primário para trading (ex: '1h')
            higher_tf (str): Timeframe superior para confirmação (ex: '4h', '1d')
            ma_period (int): Período da média móvel para análise de tendência
            base_strategy (Strategy): Estratégia base opcional para gerar sinais primários
            confirmation_periods (int): Número de períodos para confirmação de tendência
            require_alignment (bool): Se True, só gera sinais quando timeframes estão alinhados
        """
        super().__init__(name=f"Multi-Timeframe ({primary_tf}/{higher_tf}, MA{ma_period})")
        
        self.primary_tf = primary_tf
        self.higher_tf = higher_tf
        self.ma_period = ma_period
        self.base_strategy = base_strategy
        self.confirmation_periods = confirmation_periods
        self.require_alignment = require_alignment
        
        # Indicadores necessários
        self._required_indicators = [f'sma_{ma_period}']
        
        # Se temos uma estratégia base, adicionar seus indicadores
        if self.base_strategy:
            self._required_indicators.extend(self.base_strategy._required_indicators)
            
        logger.info(f"MultiTimeframeStrategy inicializada: {self.name}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais de trading baseados em análise de múltiplos timeframes.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLC e indicadores
            
        Returns:
            pd.DataFrame: DataFrame com sinais de trading
        """
        result = df.copy()
        
        # Inicializar coluna de sinal
        result['signal'] = 0
        
        # Se temos dados insuficientes, retornar sem sinais
        if len(result) < self.ma_period * 2:
            logger.warning(
                f"Dados insuficientes para análise multi-timeframe. "
                f"Necessário pelo menos {self.ma_period * 2} candles."
            )
            return result
        
        # Verificar se estamos recebendo dados com timestamp
        has_timestamp = 'timestamp' in result.columns
        if not has_timestamp and not isinstance(result.index, pd.DatetimeIndex):
            logger.warning("DataFrame não tem coluna timestamp nem DatetimeIndex")
            # Tentar criar uma coluna timestamp básica
            result['timestamp'] = pd.date_range(start='2022-01-01', periods=len(result), freq='H')
            
        # 1. Resample para o timeframe superior
        try:
            higher_tf_df = MultiTimeframeAnalyzer.resample_to_higher_timeframe(
                result, self.primary_tf, self.higher_tf
            )
        except Exception as e:
            logger.error(f"Erro ao resample para timeframe superior: {str(e)}")
            return result
        
        # 2. Analisar tendência no timeframe superior
        ma_col = f'sma_{self.ma_period}'
        if ma_col in higher_tf_df.columns:
            higher_tf_df['trend'] = np.where(higher_tf_df['close'] > higher_tf_df[ma_col], 1, 
                                          np.where(higher_tf_df['close'] < higher_tf_df[ma_col], -1, 0))
        else:
            logger.warning(f"Coluna {ma_col} não encontrada no DataFrame de timeframe superior")
            return result
        
        # 3. Gerar sinais primários
        if self.base_strategy:
            # Usar estratégia base para gerar sinais primários
            try:
                result = self.base_strategy.generate_signals(result)
                primary_signal_col = 'signal'
            except Exception as e:
                logger.error(f"Erro ao gerar sinais com estratégia base: {str(e)}")
                return result
        else:
            # Gerar sinais simples baseados em cruzamentos de média móvel
            result['primary_signal'] = 0
            # Cruzamento de preço acima da média móvel
            crossover_up = (result['close'].shift(1) <= result[ma_col].shift(1)) & \
                          (result['close'] > result[ma_col])
            # Cruzamento de preço abaixo da média móvel
            crossover_down = (result['close'].shift(1) >= result[ma_col].shift(1)) & \
                            (result['close'] < result[ma_col])
            
            result.loc[crossover_up, 'primary_signal'] = 1
            result.loc[crossover_down, 'primary_signal'] = -1
            primary_signal_col = 'primary_signal'
        
        # 4. Analisar alinhamento de tendências
        result = MultiTimeframeAnalyzer.analyze_trend_alignment(
            result, higher_tf_df, ma_col
        )
        
        # 5. Filtrar sinais com base na confirmação de múltiplos timeframes
        if self.require_alignment:
            result = MultiTimeframeAnalyzer.find_multi_timeframe_signals(
                result, 
                primary_signal_col=primary_signal_col,
                confirmation_col='trend_aligned',
                lookback=self.confirmation_periods
            )
            # Usar sinais confirmados
            result['signal'] = result['confirmed_signal']
        else:
            # Usar sinais primários diretamente
            result['signal'] = result[primary_signal_col]
        
        # Contar e registrar sinais gerados
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        logger.info(f"Gerados {buy_signals} sinais de compra e {sell_signals} sinais de venda")
        
        return result
    
    def __str__(self) -> str:
        """Representação em string da estratégia para logging."""
        base_str = f"{self.name}"
        if self.base_strategy:
            base_str += f" com {self.base_strategy.__class__.__name__}"
        return (f"{base_str} - "
                f"Primário: {self.primary_tf}, Superior: {self.higher_tf}, "
                f"MA: {self.ma_period}, Confirmação: {self.confirmation_periods}, "
                f"Alinhamento: {self.require_alignment}")