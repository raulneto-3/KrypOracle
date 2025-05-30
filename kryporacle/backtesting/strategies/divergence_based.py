import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

from .base import Strategy

logger = logging.getLogger(__name__)

class DivergenceStrategy(Strategy):
    """Estratégia que identifica divergências entre preço e osciladores."""
    
    def __init__(self, 
                 indicator: str = 'rsi',
                 lookback_period: int = 14, 
                 divergence_window: int = 5,
                 signal_threshold: float = 0.0):
        """Inicializa a estratégia de divergência."""
        super().__init__(name=f"Divergence ({indicator}, {lookback_period}, {divergence_window})")
        self.indicator = indicator
        self.lookback_period = lookback_period
        self.divergence_window = divergence_window
        self.signal_threshold = signal_threshold
        self._required_indicators = [indicator]
    
    def _find_local_extrema(self, series: pd.Series, window: int) -> Tuple[List[int], List[int]]:
        """Encontra mínimos e máximos locais em uma série."""
        local_min_indices = []
        local_max_indices = []
        
        for i in range(window, len(series) - window):
            # Janela à esquerda e à direita do ponto atual
            left_window = series.iloc[i-window:i]
            right_window = series.iloc[i+1:i+window+1]
            
            # Verificar se é um mínimo local
            if series.iloc[i] < left_window.min() and series.iloc[i] < right_window.min():
                local_min_indices.append(i)
            
            # Verificar se é um máximo local
            elif series.iloc[i] > left_window.max() and series.iloc[i] > right_window.max():
                local_max_indices.append(i)
                
        return local_min_indices, local_max_indices
    
    def _detect_positive_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Detecta divergências positivas (preço em baixa, indicador subindo)."""
        # Encontrar mínimos locais no preço e no indicador
        price_min_idx, _ = self._find_local_extrema(df['low'], self.divergence_window)
        indicator_min_idx, _ = self._find_local_extrema(df[self.indicator], self.divergence_window)
        
        divergence = pd.Series(False, index=df.index)
        
        # Verificar divergências recentes
        for i in range(len(price_min_idx) - 1):
            curr_price_idx = price_min_idx[i]
            next_price_idx = price_min_idx[i+1]
            
            # Encontrar mínimos correspondentes no indicador
            indicator_mins_in_range = [idx for idx in indicator_min_idx 
                                       if curr_price_idx <= idx <= next_price_idx]
            
            if not indicator_mins_in_range:
                continue
                
            # Verificar se o preço fez mínimos mais baixos, mas o indicador não
            price_lower_low = df['low'].iloc[next_price_idx] < df['low'].iloc[curr_price_idx]
            
            # Pegar o primeiro e último mínimo do indicador no intervalo
            first_ind_idx = indicator_mins_in_range[0]
            last_ind_idx = indicator_mins_in_range[-1]
            indicator_higher_low = df[self.indicator].iloc[last_ind_idx] > df[self.indicator].iloc[first_ind_idx]
            
            # Se o preço fez mínimos mais baixos mas o indicador fez mínimos mais altos
            if price_lower_low and indicator_higher_low:
                # Marcar o ponto como divergência positiva
                divergence.iloc[next_price_idx] = True
        
        return divergence
    
    def _detect_negative_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Detecta divergências negativas (preço em alta, indicador caindo)."""
        # Encontrar máximos locais no preço e no indicador
        _, price_max_idx = self._find_local_extrema(df['high'], self.divergence_window)
        _, indicator_max_idx = self._find_local_extrema(df[self.indicator], self.divergence_window)
        
        divergence = pd.Series(False, index=df.index)
        
        # Verificar divergências recentes
        for i in range(len(price_max_idx) - 1):
            curr_price_idx = price_max_idx[i]
            next_price_idx = price_max_idx[i+1]
            
            # Encontrar máximos correspondentes no indicador
            indicator_maxs_in_range = [idx for idx in indicator_max_idx 
                                       if curr_price_idx <= idx <= next_price_idx]
            
            if not indicator_maxs_in_range:
                continue
                
            # Verificar se o preço fez máximos mais altos, mas o indicador não
            price_higher_high = df['high'].iloc[next_price_idx] > df['high'].iloc[curr_price_idx]
            
            # Pegar o primeiro e último máximo do indicador no intervalo
            first_ind_idx = indicator_maxs_in_range[0]
            last_ind_idx = indicator_maxs_in_range[-1]
            indicator_lower_high = df[self.indicator].iloc[last_ind_idx] < df[self.indicator].iloc[first_ind_idx]
            
            # Se o preço fez máximos mais altos mas o indicador fez máximos mais baixos
            if price_higher_high and indicator_lower_high:
                # Marcar o ponto como divergência negativa
                divergence.iloc[next_price_idx] = True
        
        return divergence
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera sinais de trading baseados em divergências entre preço e indicador."""
        result = df.copy()
        
        # Inicializar coluna de sinais
        result['signal'] = 0
        
        # Verificar se temos dados suficientes
        if len(result) < (self.lookback_period + 2 * self.divergence_window):
            logger.warning(
                f"Dados insuficientes para detectar divergências. "
                f"Necessário pelo menos {self.lookback_period + 2 * self.divergence_window} candles."
            )
            return result
        
        # Detectar divergências positivas e negativas
        positive_div = self._detect_positive_divergence(result)
        negative_div = self._detect_negative_divergence(result)
        
        # Aplicar filtro de força do sinal se necessário
        if self.signal_threshold > 0:
            # Aqui podemos implementar um filtro baseado na magnitude da divergência
            pass
        
        # Aplicar sinais - usamos .loc para acessar pelo índice
        result.loc[positive_div.index[positive_div], 'signal'] = 1  # Sinal de compra
        result.loc[negative_div.index[negative_div], 'signal'] = -1  # Sinal de venda
        
        # Contar e registrar sinais gerados
        buy_signals = positive_div.sum()
        sell_signals = negative_div.sum()
        logger.info(f"Gerados {buy_signals} sinais de compra e {sell_signals} sinais de venda")
        
        return result