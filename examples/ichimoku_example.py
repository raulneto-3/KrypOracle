"""
Exemplo de uso da estratégia Ichimoku Cloud.
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
from kryporacle.backtesting.strategies.ichimoku import IchimokuStrategy
from kryporacle.backtesting.metrics import calculate_metrics
from kryporacle.backtesting.utils import get_backtest_dir, get_backtest_path
from kryporacle.visualization.charts import OHLCChart
from kryporacle.visualization.json_generator import generate_strategy_backtest_json

def main():
    """
    Executa backtest da estratégia Ichimoku Cloud em dados reais de criptomoedas.
    """
    # Inicializar componentes
    collector = BinanceCollector()
    storage = DataStorage()
    
    # Configurar parâmetros do backtest
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "4h"  # 4 horas é bom para Ichimoku
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
        
        # Criar e executar diferentes configurações da estratégia Ichimoku
        strategies = {
            "Ichimoku Padrão": IchimokuStrategy(
                tenkan_period=9,
                kijun_period=26,
                senkou_b_period=52,
                use_tk_cross=True,
                use_price_kijun_cross=False,
                use_cloud_filter=True,
                use_chikou_filter=True,
                signal_mode='tk_only'
            ),
            "Ichimoku Rápido": IchimokuStrategy(
                tenkan_period=7,
                kijun_period=22,
                senkou_b_period=44,
                use_tk_cross=True,
                use_price_kijun_cross=True,
                use_cloud_filter=True,
                use_chikou_filter=False,
                signal_mode='either'
            ),
            "Ichimoku Conservador": IchimokuStrategy(
                tenkan_period=9,
                kijun_period=26,
                senkou_b_period=52,
                use_tk_cross=True,
                use_price_kijun_cross=True,
                use_cloud_filter=True,
                use_chikou_filter=True,
                signal_mode='combined'
            )
        }
        
        results = {}
        
        for name, strategy in strategies.items():
            print(f"\nExecutando estratégia {name}...")
            result_df = engine.run(
                strategy=strategy,
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            results[name] = result_df
            
            # Calcular métricas de desempenho
            if not result_df.empty:
                print(f"\nResultados para {name}:")
                metrics = calculate_metrics(result_df, timeframe=timeframe)
                print_metrics(metrics)
                
                # Visualizar resultados
                plot_strategy_results(
                    result_df,
                    strategy_name=name,
                    symbol=symbol,
                    timeframe=timeframe
                )
                
                # Gerar JSON para a estratégia
                generate_strategy_backtest_json(
                    df=result_df,
                    strategy_name=name,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    parameters=strategy_parameters_to_dict(strategy),
                    metrics=metrics,
                    filename=get_backtest_path(f"ichimoku_{name.lower().replace(' ', '_')}.json")
                )
                
        # Comparar as diferentes estratégias
        compare_strategies(results, timeframe)
            
    except Exception as e:
        logging.error(f"Erro durante execução: {e}")
        raise

def strategy_parameters_to_dict(strategy: IchimokuStrategy) -> dict:
    """Converte os parâmetros da estratégia para um dicionário."""
    return {
        "tenkan_period": strategy.tenkan_period,
        "kijun_period": strategy.kijun_period,
        "senkou_b_period": strategy.senkou_b_period,
        "displacement": strategy.displacement,
        "use_tk_cross": strategy.use_tk_cross,
        "use_price_kijun_cross": strategy.use_price_kijun_cross,
        "use_cloud_filter": strategy.use_cloud_filter,
        "use_chikou_filter": strategy.use_chikou_filter,
        "signal_mode": strategy.signal_mode
    }

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
    
    # Plotar componentes do Ichimoku
    ichimoku_components = [col for col in chart_df.columns if col in 
                         ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']]
    
    colors = {
        'tenkan_sen': 'red',
        'kijun_sen': 'blue',
        'senkou_span_a': 'green',
        'senkou_span_b': 'red',
        'chikou_span': 'purple'
    }
    
    labels = {
        'tenkan_sen': 'Tenkan-sen',
        'kijun_sen': 'Kijun-sen',
        'senkou_span_a': 'Senkou Span A',
        'senkou_span_b': 'Senkou Span B',
        'chikou_span': 'Chikou Span'
    }
    
    for component in ichimoku_components:
        # Filtrar valores NaN
        plot_data = chart_df[~chart_df[component].isna()]
        if not plot_data.empty:
            axes[0].plot(plot_data['timestamp'], plot_data[component], 
                       color=colors.get(component, 'gray'), 
                       alpha=0.7, 
                       linewidth=1.0,
                       label=labels.get(component, component))
    
    # Preencher a nuvem entre Senkou Span A e B
    if 'senkou_span_a' in chart_df.columns and 'senkou_span_b' in chart_df.columns:
        # Filtrar valores NaN
        cloud_data = chart_df[~chart_df['senkou_span_a'].isna() & ~chart_df['senkou_span_b'].isna()]
        if not cloud_data.empty:
            # Nuvem verde quando Senkou Span A > Senkou Span B (bullish)
            axes[0].fill_between(cloud_data['timestamp'], 
                              cloud_data['senkou_span_a'], 
                              cloud_data['senkou_span_b'], 
                              where=cloud_data['senkou_span_a'] >= cloud_data['senkou_span_b'],
                              color='green', alpha=0.2)
            
            # Nuvem vermelha quando Senkou Span B > Senkou Span A (bearish)
            axes[0].fill_between(cloud_data['timestamp'], 
                              cloud_data['senkou_span_a'], 
                              cloud_data['senkou_span_b'], 
                              where=cloud_data['senkou_span_a'] < cloud_data['senkou_span_b'],
                              color='red', alpha=0.2)
    
    # Adicionar legenda
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax_portfolio.get_legend_handles_labels()
    axes[0].legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    # Salvar gráfico
    filename = f"ichimoku_{strategy_name.lower().replace(' ', '_')}"
    ohlc_chart.save_figure(fig, filename, directory=get_backtest_dir())
    print(f"Gráfico salvo em {get_backtest_dir()}/{filename}.png")

def compare_strategies(results: dict, timeframe: str):
    """Compara o desempenho de diferentes estratégias Ichimoku."""
    print("\nComparação de Estratégias Ichimoku:")
    
    # Criar figura usando OHLCChart
    chart = OHLCChart(figsize=(15, 8))
    fig, axes = chart.create_figure(1, 1)  # Sem subplot para volume
    
    # Acessar o primeiro elemento da lista de eixos
    ax = axes[0]  # Correção: acessar o primeiro elemento da lista
    
    # Plotar valor da carteira para cada estratégia
    for strategy_name, result_df in results.items():
        if not result_df.empty:
            # Criar coluna timestamp sem resetar o índice
            chart_df = result_df.copy()
            chart_df['timestamp'] = chart_df.index
            
            # Garantir que timestamp seja datetime
            chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
            
            ax.plot(chart_df['timestamp'], chart_df['portfolio_value'], label=strategy_name)
    
    ax.set_title('Comparação de Estratégias Ichimoku', fontsize=14)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Valor da Carteira ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Ajustar formato de data
    chart.adjust_date_format(ax)
    
    # Salvar gráfico
    chart.save_figure(fig, "ichimoku_strategies_comparison", directory=get_backtest_dir())
    print(f"Gráfico de comparação salvo em {get_backtest_dir()}/ichimoku_strategies_comparison.png")
    
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
        summary_df.to_csv(get_backtest_path("ichimoku_performance_summary.csv"), index=False)
        print(f"Resumo de desempenho salvo em {get_backtest_path('ichimoku_performance_summary.csv')}")

if __name__ == "__main__":
    main()