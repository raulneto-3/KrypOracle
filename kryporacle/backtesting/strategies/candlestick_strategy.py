"""
Estratégia de trading baseada em padrões de candlestick para o mercado de criptomoedas.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any

from .base import Strategy
from ...analysis.candlestick_patterns import CandlestickPatterns

logger = logging.getLogger(__name__)

class CandlestickPatternStrategy(Strategy):
    """
    Estratégia que identifica e gera sinais com base em padrões de candlestick.
    
    Esta estratégia utiliza a classe CandlestickPatterns para identificar diversos
    padrões de candlestick e gera sinais de compra e venda com base nos padrões
    identificados e seus respectivos pesos.
    """
    
    def __init__(self, 
                 bullish_patterns: Optional[List[str]] = None,
                 bearish_patterns: Optional[List[str]] = None,
                 pattern_weights: Optional[Dict[str, float]] = None,
                 confirmation_period: int = 1,
                 signal_threshold: float = 0.5):
        """
        Inicializa a estratégia de padrões de candlestick.
        
        Args:
            bullish_patterns: Lista de padrões de alta para procurar
            bearish_patterns: Lista de padrões de baixa para procurar
            pattern_weights: Dicionário com o peso de cada padrão
            confirmation_period: Número de períodos para confirmar um sinal
            signal_threshold: Limiar de pontuação para gerar um sinal
        """
        super().__init__(name=f"Candlestick Pattern Strategy")
        
        # Padrões de alta padrão
        self.bullish_patterns = bullish_patterns or [
            'doji',
            'hammer',
            'bullish_engulfing',
            'morning_star',
            'three_white_soldiers'
        ]
        
        # Padrões de baixa padrão
        self.bearish_patterns = bearish_patterns or [
            'shooting_star',
            'bearish_engulfing',
            'evening_star',
            'three_black_crows'
        ]
        
        # Pesos para cada padrão (influência no sinal final)
        self.pattern_weights = pattern_weights or {
            # Padrões de alta (valores positivos)
            'doji': 0.3,           # Padrão de indecisão com leve viés para alta após tendência de baixa
            'hammer': 1.5,         # Forte sinal de reversão de baixa para alta
            'bullish_engulfing': 2.0,  # Forte sinal de reversão de baixa para alta
            'morning_star': 2.5,   # Sinal muito forte de reversão de baixa para alta
            'three_white_soldiers': 3.0,  # Sinal extremamente forte de continuação de alta
            
            # Padrões de baixa (valores negativos)
            'shooting_star': -1.5,      # Forte sinal de reversão de alta para baixa
            'bearish_engulfing': -2.0,  # Forte sinal de reversão de alta para baixa
            'evening_star': -2.5,       # Sinal muito forte de reversão de alta para baixa
            'three_black_crows': -3.0,  # Sinal extremamente forte de continuação de baixa
        }
        
        # Parâmetros de filtragem
        self.confirmation_period = confirmation_period
        self.signal_threshold = signal_threshold
        
        # Não precisa de indicadores específicos, apenas dados OHLC
        self._required_indicators = []
        
        logger.info(f"CandlestickPatternStrategy inicializada com {len(self.bullish_patterns)} padrões de alta e {len(self.bearish_patterns)} padrões de baixa")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais de trading baseados em padrões de candlestick.
        
        Args:
            df: DataFrame com dados OHLC
            
        Returns:
            DataFrame com sinais de trading (1 = compra, -1 = venda, 0 = sem ação)
        """
        # Copiar o DataFrame para não modificar o original
        result = df.copy()
        
        # Identificar todos os padrões usando CandlestickPatterns
        logger.info("Identificando padrões de candlestick nos dados...")
        result = CandlestickPatterns.identify_patterns(result)
        
        # Criar coluna para pontuação de padrões
        result['pattern_score'] = 0.0
        
        # Calcular pontuação de padrão para cada linha
        for i in range(len(result)):
            score = 0.0
            
            # Analisar todos os padrões e calcular a pontuação
            for pattern_name in self.bullish_patterns + self.bearish_patterns:
                pattern_col = f'pattern_{pattern_name}'
                if pattern_col in result.columns and result.iloc[i][pattern_col]:
                    weight = self.pattern_weights.get(pattern_name, 0.0)
                    score += weight
                    
                    # Registrar padrões encontrados
                    pattern_type = "bullish" if weight > 0 else "bearish"
                    logger.debug(f"Padrão {pattern_name} ({pattern_type}) encontrado em {result.index[i]}, peso: {weight}")
            
            # Atribuir pontuação
            result.iat[i, result.columns.get_loc('pattern_score')] = score
        
        # Inicializar coluna de sinal
        result['signal'] = 0
        
        # Gerar sinais com base na pontuação e aplicar confirmação
        for i in range(self.confirmation_period, len(result)):
            # Calcular média da pontuação nos períodos de confirmação
            confirmation_window = result['pattern_score'].iloc[i-self.confirmation_period:i+1]
            avg_score = confirmation_window.mean()
            
            # Aplicar limiar para filtrar sinais fracos
            if avg_score > self.signal_threshold:
                result.iat[i, result.columns.get_loc('signal')] = 1  # Sinal de compra
            elif avg_score < -self.signal_threshold:
                result.iat[i, result.columns.get_loc('signal')] = -1  # Sinal de venda
        
        # Contabilizar sinais gerados
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        logger.info(f"Gerados {buy_signals} sinais de compra e {sell_signals} sinais de venda")
        
        # Adicionar coluna com tendência atual baseada nos padrões
        result['trend'] = pd.Series(np.nan, index=result.index)
        result.loc[result['pattern_score'] > 0, 'trend'] = 'bullish'
        result.loc[result['pattern_score'] < 0, 'trend'] = 'bearish'
        result.loc[result['pattern_score'] == 0, 'trend'] = 'neutral'
        
        # Remover colunas de padrões individuais para manter o DataFrame limpo
        pattern_cols = [col for col in result.columns if col.startswith('pattern_') and col != 'pattern_score']
        result = result.drop(columns=pattern_cols)
        
        return result

    def __str__(self) -> str:
        """Representação em string da estratégia para logging."""
        patterns = [f"{p} ({self.pattern_weights.get(p, 0):.1f})" for p in self.bullish_patterns + self.bearish_patterns]
        return (f"{self.name} - Bullish: {len(self.bullish_patterns)}, "
                f"Bearish: {len(self.bearish_patterns)}, "
                f"Threshold: {self.signal_threshold}, "
                f"Patterns: {', '.join(patterns)}")

class CandlestickFilterStrategy(CandlestickPatternStrategy):
    """
    Versão especializada da estratégia de padrões de candlestick que filtra por tendência.
    
    Esta estratégia permite filtrar padrões com base na tendência atual,
    usando médias móveis ou outros indicadores para aumentar a precisão dos sinais.
    """
    
    def __init__(self, 
                 bullish_patterns: Optional[List[str]] = None,
                 bearish_patterns: Optional[List[str]] = None,
                 pattern_weights: Optional[Dict[str, float]] = None,
                 confirmation_period: int = 1,
                 signal_threshold: float = 0.5,
                 trend_ma_period: int = 50,
                 require_trend_alignment: bool = True):
        """
        Inicializa a estratégia de padrões de candlestick com filtro de tendência.
        
        Args:
            bullish_patterns: Lista de padrões de alta para procurar
            bearish_patterns: Lista de padrões de baixa para procurar
            pattern_weights: Dicionário com o peso de cada padrão
            confirmation_period: Número de períodos para confirmar um sinal
            signal_threshold: Limiar de pontuação para gerar um sinal
            trend_ma_period: Período para cálculo da média móvel de tendência
            require_trend_alignment: Se True, os sinais devem estar alinhados com a tendência
        """
        super().__init__(
            bullish_patterns=bullish_patterns,
            bearish_patterns=bearish_patterns,
            pattern_weights=pattern_weights,
            confirmation_period=confirmation_period,
            signal_threshold=signal_threshold
        )
        
        self.name = f"Candlestick Pattern Filter Strategy (MA{trend_ma_period})"
        self.trend_ma_period = trend_ma_period
        self.require_trend_alignment = require_trend_alignment
        
        # Adicionar indicador de tendência aos requisitos
        self._required_indicators.append(f'sma_{trend_ma_period}')
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais de trading baseados em padrões de candlestick filtrados por tendência.
        
        Args:
            df: DataFrame com dados OHLC e indicadores
            
        Returns:
            DataFrame com sinais de trading (1 = compra, -1 = venda, 0 = sem ação)
        """
        # Usar a implementação base para identificar padrões
        result = super().generate_signals(df)
        
        # Se não precisamos alinhar com a tendência, retornar os sinais como estão
        if not self.require_trend_alignment:
            return result
        
        # Identificar coluna de média móvel
        ma_col = f'sma_{self.trend_ma_period}'
        if ma_col not in result.columns:
            logger.warning(f"Coluna de média móvel {ma_col} não encontrada. Filtro de tendência não será aplicado.")
            return result
        
        # Determinar tendência com base na média móvel
        result['ma_trend'] = 0
        result.loc[result['close'] > result[ma_col], 'ma_trend'] = 1  # Tendência de alta
        result.loc[result['close'] < result[ma_col], 'ma_trend'] = -1  # Tendência de baixa
        
        # Filtrar sinais que não estão alinhados com a tendência da MA
        filtered_signals = result['signal'].copy()
        
        # Quando tendência é de alta, manter apenas sinais de compra
        filtered_signals[(result['ma_trend'] == 1) & (result['signal'] == -1)] = 0
        
        # Quando tendência é de baixa, manter apenas sinais de venda
        filtered_signals[(result['ma_trend'] == -1) & (result['signal'] == 1)] = 0
        
        # Atualizar coluna de sinal
        result['signal'] = filtered_signals
        
        # Contabilizar sinais após filtragem
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        logger.info(f"Após filtro de tendência: {buy_signals} sinais de compra e {sell_signals} sinais de venda")
        
        return result