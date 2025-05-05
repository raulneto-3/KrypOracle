import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union

from .base import Strategy
from ...analysis.fibonacci import FibonacciAnalyzer

logger = logging.getLogger(__name__)

class FibonacciRetracementStrategy(Strategy):
    """
    Estratégia que utiliza níveis de retração de Fibonacci para identificar pontos de entrada e saída.
    
    Esta estratégia identifica swing points (máximos e mínimos locais) e calcula níveis de retração
    de Fibonacci. Gera sinais de compra quando o preço atinge níveis de suporte (níveis de retração
    em tendência de baixa) e sinais de venda quando atinge níveis de resistência (níveis de retração
    em tendência de alta).
    """
    
    def __init__(self, 
                 swing_window: int = 10,
                 lookback_period: int = 100,
                 support_levels: List[str] = None,
                 resistance_levels: List[str] = None,
                 confirmation_period: int = 2,
                 use_volume: bool = True,
                 price_bounce_pct: float = 0.005):
        """
        Inicializa a estratégia de retração de Fibonacci.
        
        Args:
            swing_window (int): Tamanho da janela para detectar pontos de swing
            lookback_period (int): Período para procurar pontos de swing
            support_levels (List[str]): Níveis de suporte a monitorar (ex: ["0.618", "0.786"])
            resistance_levels (List[str]): Níveis de resistência a monitorar (ex: ["0.382", "0.5"])
            confirmation_period (int): Períodos para confirmar um sinal
            use_volume (bool): Se True, considera aumento de volume como confirmação
            price_bounce_pct (float): % mínima de movimento do preço para confirmação (0.005 = 0.5%)
        """
        super().__init__(name=f"Fibonacci Retracement ({swing_window}, {lookback_period})")
        
        # Parâmetros da estratégia
        self.swing_window = swing_window
        self.lookback_period = lookback_period
        self.support_levels = support_levels or ["0.618", "0.786"]
        self.resistance_levels = resistance_levels or ["0.382", "0.5"]
        self.confirmation_period = confirmation_period
        self.use_volume = use_volume
        self.price_bounce_pct = price_bounce_pct
        
        # Indicadores necessários
        self._required_indicators = ['atr']
        if use_volume:
            self._required_indicators.append('volume')
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera sinais de trading baseados em níveis de retração de Fibonacci.
        
        Args:
            df (pd.DataFrame): DataFrame com dados OHLC e indicadores
            
        Returns:
            pd.DataFrame: DataFrame com sinais de trading (1 = compra, -1 = venda, 0 = sem ação)
        """
        # Copiar o DataFrame para não modificar o original
        result = df.copy()
        
        # Adicionar pontos de swing e níveis de Fibonacci
        result = FibonacciAnalyzer.detect_swing_points(result, window=self.swing_window)
        
        # Inicializar colunas para níveis de Fibonacci
        for level in self.support_levels + self.resistance_levels:
            result[f'fib_level_{level}'] = np.nan
        
        # Inicializar coluna de sinal
        result['signal'] = 0
        
        # Inicializar coluna para armazenar nível Fibonacci ativo
        result['active_fib_level'] = np.nan
        
        # Para cada ponto no DataFrame
        for i in range(self.lookback_period, len(result)):
            # Pegar uma janela de lookback para análise
            window = result.iloc[i-self.lookback_period:i+1]
            
            # Encontrar o último swing high e low significativos dentro da janela
            swing_highs = window[window['swing_high'] == True]
            swing_lows = window[window['swing_low'] == True]
            
            if not swing_highs.empty and not swing_lows.empty:
                # Pegar o último swing high e swing low
                last_swing_high_idx = swing_highs.index[-1]
                last_swing_low_idx = swing_lows.index[-1]
                
                last_swing_high = result.loc[last_swing_high_idx, 'high']
                last_swing_low = result.loc[last_swing_low_idx, 'low']
                
                # Determinar a direção da tendência
                downtrend = last_swing_high_idx > last_swing_low_idx  # Último ponto alto depois do último ponto baixo
                uptrend = last_swing_low_idx > last_swing_high_idx    # Último ponto baixo depois do último ponto alto
                
                current_price = result.iloc[i]['close']
                current_low = result.iloc[i]['low']
                current_high = result.iloc[i]['high']
                
                # Calcular níveis de Fibonacci para a tendência atual
                if downtrend:
                    fib_levels = FibonacciAnalyzer.fibonacci_retracement(last_swing_high, last_swing_low)
                    
                    # Verificar sinais de compra em suportes
                    for level in self.support_levels:
                        if level in fib_levels:
                            support = fib_levels[level]
                            
                            # Armazenar o nível ativo
                            result.loc[result.index[i], f'fib_level_{level}'] = support
                            
                            # Verificar se o preço tocou no suporte (dentro de 1 ATR)
                            atr_value = result.iloc[i]['atr']
                            price_near_support = (current_low <= support + 0.5 * atr_value) and (current_price >= support - 0.5 * atr_value)
                            
                            # Verificar uma confirmação de movimento para cima a partir do suporte
                            price_bounced_up = False
                            if i >= self.confirmation_period:
                                # Preço subiu pelo menos price_bounce_pct% do nível de suporte
                                bounce = (current_price - support) / support
                                price_bounced_up = bounce > self.price_bounce_pct
                            
                            # Verificar volume aumentando (opcional)
                            volume_increased = True
                            if self.use_volume and 'volume' in result.columns:
                                avg_volume = result['volume'].iloc[i-5:i].mean()
                                current_volume = result.iloc[i]['volume']
                                volume_increased = current_volume > avg_volume * 1.2  # 20% acima da média
                            
                            # Se todas as condições forem atendidas, gerar sinal de compra
                            if price_near_support and price_bounced_up and volume_increased:
                                result.loc[result.index[i], 'signal'] = 1
                                result.loc[result.index[i], 'active_fib_level'] = level
                                logger.info(f"Buy signal generated at {result.index[i]} near Fibonacci level {level} ({support:.2f})")
                
                elif uptrend:
                    fib_levels = FibonacciAnalyzer.fibonacci_retracement(last_swing_high, last_swing_low)
                    
                    # Verificar sinais de venda em resistências
                    for level in self.resistance_levels:
                        if level in fib_levels:
                            resistance = fib_levels[level]
                            
                            # Armazenar o nível ativo
                            result.loc[result.index[i], f'fib_level_{level}'] = resistance
                            
                            # Verificar se o preço tocou na resistência (dentro de 1 ATR)
                            atr_value = result.iloc[i]['atr']
                            price_near_resistance = (current_high >= resistance - 0.5 * atr_value) and (current_price <= resistance + 0.5 * atr_value)
                            
                            # Verificar uma confirmação de movimento para baixo a partir da resistência
                            price_bounced_down = False
                            if i >= self.confirmation_period:
                                # Preço caiu pelo menos price_bounce_pct% do nível de resistência
                                bounce = (resistance - current_price) / resistance
                                price_bounced_down = bounce > self.price_bounce_pct
                            
                            # Verificar volume aumentando (opcional)
                            volume_increased = True
                            if self.use_volume and 'volume' in result.columns:
                                avg_volume = result['volume'].iloc[i-5:i].mean()
                                current_volume = result.iloc[i]['volume']
                                volume_increased = current_volume > avg_volume * 1.2  # 20% acima da média
                            
                            # Se todas as condições forem atendidas, gerar sinal de venda
                            if price_near_resistance and price_bounced_down and volume_increased:
                                result.loc[result.index[i], 'signal'] = -1
                                result.loc[result.index[i], 'active_fib_level'] = level
                                logger.info(f"Sell signal generated at {result.index[i]} near Fibonacci level {level} ({resistance:.2f})")
        
        # Contabilizar sinais gerados
        buy_signals = (result['signal'] == 1).sum()
        sell_signals = (result['signal'] == -1).sum()
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return result
        
    def __str__(self) -> str:
        """Representação em string da estratégia para logging."""
        return (f"{self.name} - Support levels: {', '.join(self.support_levels)}, "
                f"Resistance levels: {', '.join(self.resistance_levels)}, "
                f"Confirmation: {self.confirmation_period}, "
                f"Use volume: {self.use_volume}")