"""
Exemplo de uso das estratégias de padrões de candlestick.
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
from kryporacle.backtesting.strategies import CandlestickPatternStrategy, CandlestickFilterStrategy
from kryporacle.backtesting.metrics import calculate_metrics
from kryporacle.backtesting.utils import get_backtest_dir, get_backtest_path
from kryporacle.visualization.charts import OHLCChart
from kryporacle.visualization.json_generator import generate_strategy_backtest_json, generate_strategy_comparison_json

def main():
    """
    Executa backtests das estratégias de padrões de candlestick em dados reais de criptomoedas.
    """
    # Inicializar componentes
    collector = BinanceCollector()
    storage = DataStorage()
    
    # Configurar parâmetros do backtest
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "4h"  # 4 horas é bom para padrões de candlestick
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
        
        # Rodar backtests com diferentes configurações
        results = {}
        
        # Estratégia 1: Todos os padrões de candlestick
        print("\nExecutando estratégia básica de padrões de candlestick...")
        basic_strategy = CandlestickPatternStrategy(
            confirmation_period=1,
            signal_threshold=0.5
        )
        basic_result = engine.run(
            strategy=basic_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Candlestick Básica"] = basic_result
        
        # Estratégia 2: Apenas padrões fortes (maior threshold)
        print("\nExecutando estratégia com padrões fortes (threshold alto)...")
        strong_patterns_strategy = CandlestickPatternStrategy(
            bullish_patterns=['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers'],
            bearish_patterns=['shooting_star', 'bearish_engulfing', 'evening_star', 'three_black_crows'],
            confirmation_period=2,  # Requer confirmação por 2 períodos
            signal_threshold=1.5    # Threshold mais alto
        )
        strong_result = engine.run(
            strategy=strong_patterns_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Padrões Fortes"] = strong_result
        
        # Estratégia 3: Padrões filtrados por tendência
        print("\nExecutando estratégia com filtro de tendência...")
        trend_filter_strategy = CandlestickFilterStrategy(
            bullish_patterns=['hammer', 'bullish_engulfing', 'morning_star', 'three_white_soldiers'],
            bearish_patterns=['shooting_star', 'bearish_engulfing', 'evening_star', 'three_black_crows'],
            confirmation_period=1,
            signal_threshold=1.0,
            trend_ma_period=50,
            require_trend_alignment=True
        )
        trend_filter_result = engine.run(
            strategy=trend_filter_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Filtro de Tendência"] = trend_filter_result
        
        # Estratégia 4: Padrões de reversão
        print("\nExecutando estratégia com padrões de reversão...")
        reversal_strategy = CandlestickPatternStrategy(
            bullish_patterns=['hammer', 'bullish_engulfing', 'morning_star'],
            bearish_patterns=['shooting_star', 'bearish_engulfing', 'evening_star'],
            pattern_weights={
                'hammer': 2.0,
                'bullish_engulfing': 2.5,
                'morning_star': 3.0,
                'shooting_star': -2.0,
                'bearish_engulfing': -2.5,
                'evening_star': -3.0
            },
            confirmation_period=1,
            signal_threshold=2.0  # Threshold mais alto para mais precisão
        )
        reversal_result = engine.run(
            strategy=reversal_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Padrões de Reversão"] = reversal_result
        
        # Calcular métricas e visualizar resultados
        metrics_dict = {}
        for strategy_name, result_df in results.items():
            if not result_df.empty:
                print(f"\nResultados para {strategy_name}:")
                metrics = calculate_metrics(result_df, timeframe=timeframe)
                metrics_dict[strategy_name] = metrics
                print_metrics(metrics)
                
                # Visualizar resultados com charts
                plot_strategy_results(
                    result_df,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe
                )
                
                # Gerar JSON para a estratégia individual
                generate_strategy_backtest_json(
                    df=result_df,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    parameters={},  # Parâmetros seriam extraídos da estratégia
                    metrics=metrics,
                    filename=get_backtest_path(f"candlestick_{strategy_name.replace(' ', '_').lower()}.json")
                )
        
        # Gerar JSON de comparação de estratégias
        generate_strategy_comparison_json(
            results=results,
            metrics=metrics_dict,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            filename=get_backtest_path("candlestick_comparison.json")
        )
        
        # Comparar estratégias
        compare_strategies(results, timeframe)
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback
        traceback.print_exc()

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
    ohlc_chart = OHLCChart(figsize=(16, 10))
    
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
    
    # Adicionar legenda
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax_portfolio.get_legend_handles_labels()
    axes[0].legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    # Plotar pontuação de padrões se disponível
    if 'pattern_score' in chart_df.columns:
        if len(axes) > 1:
            # Usar o painel de volume para mostrar a pontuação
            ax_score = axes[1].twinx()
            ax_score.plot(chart_df['timestamp'], chart_df['pattern_score'], 
                         color='orange', linewidth=1.0, label='Pontuação de Padrões')
            ax_score.set_ylabel('Pontuação', color='orange')
            ax_score.tick_params(axis='y', labelcolor='orange')
            ax_score.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Legenda para pontuação
            handles3, labels3 = ax_score.get_legend_handles_labels()
            axes[1].legend(handles3, labels3, loc='upper right')
    
    # Salvar gráfico
    filename = f"candlestick_{strategy_name.replace(' ', '_').lower()}"
    ohlc_chart.save_figure(fig, filename, directory=get_backtest_dir())
    print(f"Gráfico salvo em {get_backtest_dir()}/{filename}.png")

def print_metrics(metrics: dict):
    """Exibe métricas de desempenho formatadas."""
    print(f"  Capital Inicial: ${metrics['initial_value']:.2f}")
    print(f"  Capital Final: ${metrics['final_value']:.2f}")
    print(f"  Retorno Total: {metrics['total_return']:.2%}")
    print(f"  Retorno Anualizado: {metrics['annual_return']:.2%}")
    print(f"  Volatilidade: {metrics['volatility']:.2%}")
    print(f"  Índice Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Drawdown Máximo: {metrics['max_drawdown']:.2%}")
    print(f"  Total de Operações: {metrics['total_trades']}")
    print(f"  Taxa de Acerto: {metrics['win_rate']:.2%}")

def compare_strategies(results: dict, timeframe: str):
    """Compara o desempenho de diferentes estratégias."""
    print("\nComparação de Estratégias:")
    
    # Criar figura usando OHLCChart
    chart = OHLCChart(figsize=(15, 8))
    fig, ax = chart.create_figure()
    
    # Plotar valor da carteira para cada estratégia
    for strategy_name, result_df in results.items():
        if not result_df.empty:
            # Criar coluna timestamp sem resetar o índice
            chart_df = result_df.copy()
            chart_df['timestamp'] = chart_df.index
            
            # Garantir que timestamp seja datetime
            chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
            
            ax[0].plot(chart_df['timestamp'], chart_df['portfolio_value'], label=strategy_name)
    
    ax[0].set_title('Comparação de Estratégias de Padrões de Candlestick', fontsize=14)
    ax[0].set_xlabel('Data', fontsize=12)
    ax[0].set_ylabel('Valor da Carteira ($)', fontsize=12)
    ax[0].legend(fontsize=10)
    ax[0].grid(True, alpha=0.3)
    
    # Ajustar formato de data
    chart.adjust_date_format(ax[0])
    
    # Salvar gráfico
    chart.save_figure(fig, "candlestick_strategies_comparison", directory=get_backtest_dir())
    print(f"Gráfico de comparação salvo em {get_backtest_dir()}/candlestick_strategies_comparison.png")
    
    # Criar tabela resumo
    summary = []
    for strategy_name, result_df in results.items():
        if not result_df.empty:
            metrics = calculate_metrics(result_df, timeframe=timeframe)
            summary.append({
                'Estratégia': strategy_name,
                'Retorno Total': f"{metrics['total_return']:.2%}",
                'Retorno Anual': f"{metrics['annual_return']:.2%}",
                'Índice Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Drawdown Máximo': f"{metrics['max_drawdown']:.2%}",
                'Operações': metrics['total_trades'],
                'Taxa de Acerto': f"{metrics['win_rate']:.2%}"
            })
    
    # Imprimir tabela resumo
    if summary:
        print("\nResumo de Desempenho:")
        summary_df = pd.DataFrame(summary)
        # Ordenar por retorno anual (do maior para o menor)
        summary_df['Sort_Value'] = summary_df['Retorno Anual'].str.rstrip('%').astype(float)
        summary_df = summary_df.sort_values('Sort_Value', ascending=False).drop('Sort_Value', axis=1)
        print(summary_df.to_string(index=False))
        
        # Salvar tabela resumo como CSV
        summary_df.to_csv(get_backtest_path("candlestick_performance_summary.csv"), index=False)
        print(f"Resumo de desempenho salvo em {get_backtest_path('candlestick_performance_summary.csv')}")

if __name__ == "__main__":
    main()