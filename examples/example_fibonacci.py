"""
Exemplo de uso da estratégia de retração de Fibonacci.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys

# Configuração de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Adicionar raiz do projeto ao path
sys.path.append(".")

# Importações necessárias
from kryporacle.data.exchanges.binance import BinanceCollector
from kryporacle.data.storage import DataStorage
from kryporacle.backtesting.engine import BacktestEngine
from kryporacle.backtesting.strategies.fibonacci import FibonacciRetracementStrategy
from kryporacle.backtesting.metrics import calculate_metrics
from kryporacle.backtesting.utils import get_backtest_dir, get_backtest_path
from kryporacle.visualization.charts import OHLCChart
from kryporacle.visualization.json_generator import generate_strategy_backtest_json

def main():
    """
    Executa backtest da estratégia de retração de Fibonacci em dados reais de criptomoedas.
    """
    # Inicializar componentes
    collector = BinanceCollector()
    storage = DataStorage()
    
    # Configurar parâmetros do backtest
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "4h"  # 4 horas é bom para identificar pontos de swing
    initial_capital = 10000.0  # $10,000 USD
    
    # Obter dados dos últimos 180 dias (6 meses)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    
    print(f"Buscando dados de {symbol} de {start_time} a {end_time}...")
    print(f"Os resultados do backtest serão salvos em: {get_backtest_dir()}")
    
    try:
        # Buscar dados históricos
        data = collector.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Salvar os dados
        storage.save_market_data(exchange_id, data)
        print(f"Salvos {len(data['data'])} registros no armazenamento")
        
        # Inicializar motor de backtest
        engine = BacktestEngine(initial_capital=initial_capital, 
                              commission=0.001)  # 0.1% comissão
        
        # Criar e configurar a estratégia Fibonacci
        print("\nExecutando estratégia de retração de Fibonacci...")
        fib_strategy = FibonacciRetracementStrategy(
            swing_window=10,  # Tamanho da janela para detectar pontos de swing
            lookback_period=50,  # Período para buscar pontos de swing
            support_levels=["0.618", "0.786"],  # Níveis de suporte clássicos
            resistance_levels=["0.382", "0.5"],  # Níveis de resistência clássicos
            confirmation_period=2,  # Períodos para confirmar um sinal
            use_volume=True,  # Usar volume como confirmação
            price_bounce_pct=0.005  # 0.5% de movimento mínimo para confirmação
        )
        
        # Executar o backtest
        result_df = engine.run(
            strategy=fib_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Calcular métricas de desempenho
        if not result_df.empty:
            print("\nResultados para Estratégia de Retração de Fibonacci:")
            metrics = calculate_metrics(result_df, timeframe=timeframe)
            print_metrics(metrics)
            
            # Visualizar resultados
            plot_strategy_results(
                result_df,
                strategy_name="Fibonacci Retracement",
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Gerar JSON para a estratégia
            generate_strategy_backtest_json(
                df=result_df,
                strategy_name="Fibonacci Retracement",
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                parameters={
                    "swing_window": fib_strategy.swing_window,
                    "lookback_period": fib_strategy.lookback_period,
                    "support_levels": fib_strategy.support_levels,
                    "resistance_levels": fib_strategy.resistance_levels
                },
                metrics=metrics,
                filename=get_backtest_path("fibonacci_retracement_strategy.json")
            )
            
    except Exception as e:
        logging.error(f"Erro durante execução: {e}")
        raise

def print_metrics(metrics: dict):
    """Imprime as métricas de desempenho do backtest."""
    print(f"Retorno Total: {metrics['total_return']:.2%}")
    print(f"Retorno Anual: {metrics['annual_return']:.2%}")
    print(f"Índice Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Drawdown Máximo: {metrics['max_drawdown']:.2%}")
    print(f"Total de Operações: {metrics['total_trades']}")
    print(f"Taxa de Acerto: {metrics['win_rate']:.2%}")

def plot_strategy_results(df: pd.DataFrame, strategy_name: str, symbol: str, timeframe: str):
    """
    Visualiza os resultados do backtest usando o módulo charts.py
    """
    # Preparar DataFrame para visualização
    chart_df = df.copy()
    
    # Verificar se o índice já é um DatetimeIndex
    if isinstance(chart_df.index, pd.DatetimeIndex):
        # Se já for um DatetimeIndex, criar coluna timestamp sem resetar o índice
        chart_df['timestamp'] = chart_df.index
    else:
        # Se não for DatetimeIndex, verificar se já existe coluna timestamp
        if 'timestamp' not in chart_df.columns:
            # Resetar o índice apenas se necessário
            chart_df = chart_df.reset_index().rename(columns={'index': 'timestamp'})
    
    # Garantir que timestamp seja datetime
    chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
    
    # Inicializar chart plotter
    ohlc_chart = OHLCChart(figsize=(16, 12))
    
    # Criar título
    print(f"Gerando gráfico para estratégia {strategy_name}...")
    title = f"{strategy_name} - {symbol} ({timeframe})"
    
    # Criar OHLC chart com volume
    fig, axes = ohlc_chart.plot_ohlc(
        chart_df, 
        title=title,
        volume=True if 'volume' in chart_df.columns else False
    )
    
    # Plotar o valor da carteira
    ax_portfolio = axes[0].twinx()
    ax_portfolio.plot(chart_df['timestamp'], chart_df['portfolio_value'], 
                    color='purple', linewidth=1.5, label='Valor da Carteira')
    ax_portfolio.set_ylabel('Valor da Carteira ($)', color='purple')
    ax_portfolio.tick_params(axis='y', labelcolor='purple')
    
    # Adicionar pontos de compra e venda
    buy_signals = chart_df[chart_df['signal'] == 1]
    sell_signals = chart_df[chart_df['signal'] == -1]
    
    if not buy_signals.empty:
        axes[0].scatter(buy_signals['timestamp'], buy_signals['low'] * 0.995, 
                      marker='^', color='green', s=100, label='Compra')
        
    if not sell_signals.empty:
        axes[0].scatter(sell_signals['timestamp'], sell_signals['high'] * 1.005, 
                      marker='v', color='red', s=100, label='Venda')
    
    # Plotar níveis de fibonacci se disponíveis
    fib_levels = [col for col in chart_df.columns if col.startswith('fib_level_')]
    if fib_levels:
        for level in fib_levels:
            # Filtrar apenas pontos onde o nível está definido
            level_data = chart_df[chart_df[level].notna()]
            if not level_data.empty:
                # Extrair o valor do nível (ex: 0.618, 0.786)
                level_value = level.split('_')[-1]
                axes[0].plot(level_data['timestamp'], level_data[level], 
                           linestyle='--', alpha=0.7, 
                           label=f'Fib {level_value}')
    
    # Adicionar legenda
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax_portfolio.get_legend_handles_labels()
    axes[0].legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    # Salvar gráfico
    filename = f"fibonacci_retracement_strategy"
    ohlc_chart.save_figure(fig, filename, directory=get_backtest_dir())
    print(f"Gráfico salvo em {get_backtest_dir()}/{filename}.png")

if __name__ == "__main__":
    main()