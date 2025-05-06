"""
Exemplo de uso da estratégia de múltiplos timeframes.

Este exemplo demonstra como combinar análises de diferentes períodos
para obter sinais de trading mais confiáveis e reduzir falsos positivos.
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
from kryporacle.backtesting.strategies import (
    MultiTimeframeStrategy, 
    MovingAverageCrossover, 
    RSIStrategy,
    BollingerBandsStrategy
)
from kryporacle.backtesting.metrics import calculate_metrics
from kryporacle.backtesting.utils import get_backtest_dir, get_backtest_path
from kryporacle.visualization.charts import OHLCChart
from kryporacle.visualization.json_generator import generate_strategy_backtest_json, generate_strategy_comparison_json


def main():
    """
    Executa backtests da estratégia de múltiplos timeframes em dados reais de criptomoedas.
    """
    # Inicializar componentes
    collector = BinanceCollector()
    storage = DataStorage()
    
    # Configurar parâmetros do backtest
    exchange_id = "binance"
    symbol = "BTC/USDT"
    primary_timeframe = "1h"   # Timeframe primário para trading
    higher_timeframe = "4h"    # Timeframe superior para confirmação
    initial_capital = 10000.0  # $10,000 USD
    
    # Obter dados dos últimos 180 dias (6 meses)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)
    
    print(f"Buscando dados de {symbol} de {start_time} a {end_time}...")
    print(f"Os resultados do backtest serão salvos em: {get_backtest_dir()}")
    
    try:
        # Buscar dados históricos para o timeframe primário
        print(f"Buscando dados para timeframe primário: {primary_timeframe}...")
        primary_data = collector.fetch_historical_data(
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Buscar dados históricos para o timeframe superior
        print(f"Buscando dados para timeframe superior: {higher_timeframe}...")
        higher_data = collector.fetch_historical_data(
            symbol=symbol,
            timeframe=higher_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Salvar os dados
        storage.save_market_data(exchange_id, primary_data)
        print(f"Salvos {len(primary_data['data'])} registros do timeframe primário")
        
        storage.save_market_data(exchange_id, higher_data)
        print(f"Salvos {len(higher_data['data'])} registros do timeframe superior")
        
        # Inicializar motor de backtest
        engine = BacktestEngine(initial_capital=initial_capital, 
                              commission=0.001)  # 0.1% comissão
        
        # Criar e executar diferentes configurações da estratégia Multi-Timeframe
        results = {}
        
        # 1. Primeiro, executar estratégias básicas para referência
        
        # Estratégia de referência: MA Crossover em timeframe único
        print("\nExecutando estratégia de referência: MA Crossover (20/50)...")
        ma_strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
        ma_results = engine.run(
            strategy=ma_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["MA Crossover (20/50)"] = ma_results
        
        # Estratégia de referência: RSI
        print("\nExecutando estratégia de referência: RSI (14)...")
        rsi_strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
        rsi_results = engine.run(
            strategy=rsi_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["RSI (14)"] = rsi_results
        
        # 2. Agora, executar variações da estratégia multi-timeframe
        
        # Multi-Timeframe Básico - Usando médias móveis simples
        print("\nExecutando estratégia Multi-Timeframe Básica...")
        mtf_basic_strategy = MultiTimeframeStrategy(
            primary_tf=primary_timeframe,
            higher_tf=higher_timeframe,
            ma_period=20,
            confirmation_periods=3,
            require_alignment=True
        )
        mtf_basic_results = engine.run(
            strategy=mtf_basic_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Multi-Timeframe Básico"] = mtf_basic_results
        
        # Multi-Timeframe com MA Crossover como estratégia base
        print("\nExecutando estratégia Multi-Timeframe com MA Crossover...")
        mtf_ma_strategy = MultiTimeframeStrategy(
            primary_tf=primary_timeframe,
            higher_tf=higher_timeframe,
            ma_period=20,
            base_strategy=MovingAverageCrossover(fast_period=20, slow_period=50),
            confirmation_periods=3,
            require_alignment=True
        )
        mtf_ma_results = engine.run(
            strategy=mtf_ma_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Multi-Timeframe com MA"] = mtf_ma_results
        
        # Multi-Timeframe com RSI como estratégia base
        print("\nExecutando estratégia Multi-Timeframe com RSI...")
        mtf_rsi_strategy = MultiTimeframeStrategy(
            primary_tf=primary_timeframe,
            higher_tf=higher_timeframe,
            ma_period=20,
            base_strategy=RSIStrategy(rsi_period=14, oversold=30, overbought=70),
            confirmation_periods=2,
            require_alignment=True
        )
        mtf_rsi_results = engine.run(
            strategy=mtf_rsi_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Multi-Timeframe com RSI"] = mtf_rsi_results
        
        # Multi-Timeframe com menos confirmações (mais sinais, menos conservador)
        print("\nExecutando estratégia Multi-Timeframe Agressiva...")
        mtf_aggressive_strategy = MultiTimeframeStrategy(
            primary_tf=primary_timeframe,
            higher_tf=higher_timeframe,
            ma_period=20,
            confirmation_periods=1,  # Menos confirmações
            require_alignment=False  # Não requer alinhamento estrito
        )
        mtf_aggressive_results = engine.run(
            strategy=mtf_aggressive_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Multi-Timeframe Agressivo"] = mtf_aggressive_results
        
        # Multi-Timeframe com mais confirmações (menos sinais, mais conservador)
        print("\nExecutando estratégia Multi-Timeframe Conservadora...")
        mtf_conservative_strategy = MultiTimeframeStrategy(
            primary_tf=primary_timeframe,
            higher_tf=higher_timeframe,
            ma_period=20,
            confirmation_periods=5,  # Mais confirmações
            require_alignment=True   # Requer alinhamento estrito
        )
        mtf_conservative_results = engine.run(
            strategy=mtf_conservative_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results["Multi-Timeframe Conservador"] = mtf_conservative_results
        
        # Calcular métricas e visualizar resultados
        metrics_dict = {}
        for strategy_name, result_df in results.items():
            if not result_df.empty:
                print(f"\nResultados para {strategy_name}:")
                metrics = calculate_metrics(result_df, timeframe=primary_timeframe)
                metrics_dict[strategy_name] = metrics
                print_metrics(metrics)
                
                # Visualizar resultados com charts
                plot_strategy_results(
                    result_df,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=primary_timeframe
                )
                
                # Gerar JSON para a estratégia individual
                generate_strategy_backtest_json(
                    df=result_df,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=primary_timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    parameters={},  # Parâmetros seriam extraídos da estratégia
                    metrics=metrics,
                    filename=get_backtest_path(f"mtf_{strategy_name.replace(' ', '_').replace('/', '_').lower()}.json")
                )
        
        # Gerar JSON de comparação de estratégias
        generate_strategy_comparison_json(
            results=results,
            metrics=metrics_dict,
            symbol=symbol,
            timeframe=primary_timeframe,
            start_time=start_time,
            end_time=end_time,
            filename=get_backtest_path("multi_timeframe_comparison.json")
        )
        
        # Comparar estratégias
        compare_strategies(results, primary_timeframe)
        
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
    
    # Plotar tendência de timeframe superior se disponível
    if 'higher_tf_trend' in chart_df.columns:
        # Criar novo eixo para indicadores
        ax_trend = axes[0].twinx()
        ax_trend.spines['right'].set_position(('outward', 60))  # Deslocar eixo para direita
        
        # Filtrando para mostrar apenas mudanças de tendência
        trend_changes = chart_df[chart_df['higher_tf_trend'] != chart_df['higher_tf_trend'].shift(1)]
        
        # Plotar áreas de tendência
        for i in range(len(trend_changes) - 1):
            start_idx = trend_changes.index[i]
            end_idx = trend_changes.index[i+1]
            trend = trend_changes.iloc[i]['higher_tf_trend']
            
            if trend == 1:  # Tendência de alta
                axes[0].axvspan(chart_df.loc[start_idx, 'timestamp'], 
                              chart_df.loc[end_idx, 'timestamp'],
                              alpha=0.1, color='green', label='Alta TF' if i == 0 else "")
            elif trend == -1:  # Tendência de baixa
                axes[0].axvspan(chart_df.loc[start_idx, 'timestamp'], 
                              chart_df.loc[end_idx, 'timestamp'],
                              alpha=0.1, color='red', label='Baixa TF' if i == 0 else "")
    
    # Adicionar legenda
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax_portfolio.get_legend_handles_labels()
    axes[0].legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    # Plotar indicadores de alinhamento se disponíveis
    if 'trend_aligned' in chart_df.columns:
        if len(axes) > 1:
            # Usar o painel de volume para mostrar o alinhamento de tendência
            ax_align = axes[1].twinx()
            aligned = chart_df['trend_aligned'].astype(int)
            ax_align.plot(chart_df['timestamp'], aligned, 
                         color='orange', linewidth=1.0, label='Alinhamento de Tendência')
            ax_align.set_ylabel('Alinhado', color='orange')
            ax_align.set_yticks([0, 1])
            ax_align.set_yticklabels(['Não', 'Sim'])
            ax_align.tick_params(axis='y', labelcolor='orange')
            
            # Legenda para alinhamento
            handles3, labels3 = ax_align.get_legend_handles_labels()
            axes[1].legend(handles3, labels3, loc='upper right')
    
    # Salvar gráfico
    filename = f"multi_timeframe_{strategy_name.replace(' ', '_').lower()}"
    ohlc_chart.save_figure(fig, filename, directory=get_backtest_dir())
    print(f"Gráfico salvo em {get_backtest_dir()}/{filename}.png")


def print_metrics(metrics: dict):
    """Exibe métricas de desempenho formatadas."""
    print(f"  Retorno Total: {metrics['total_return']:.2%}")
    print(f"  Retorno Anual: {metrics['annual_return']:.2%}")
    print(f"  Índice Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Drawdown Máximo: {metrics['max_drawdown']:.2%}")
    print(f"  Operações: {metrics['total_trades']}")
    print(f"  Taxa de Acerto: {metrics['win_rate']:.2%}")
    print(f"  Relação Lucro/Prejuízo: {metrics['profit_factor']:.2f}")


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
    
    ax[0].set_title('Comparação de Estratégias Multi-Timeframe', fontsize=14)
    ax[0].set_xlabel('Data', fontsize=12)
    ax[0].set_ylabel('Valor da Carteira ($)', fontsize=12)
    ax[0].legend(fontsize=10)
    ax[0].grid(True, alpha=0.3)
    
    # Ajustar formato de data
    chart.adjust_date_format(ax[0])
    
    # Salvar gráfico
    chart.save_figure(fig, "multi_timeframe_strategies_comparison", directory=get_backtest_dir())
    print(f"Gráfico de comparação salvo em {get_backtest_dir()}/multi_timeframe_strategies_comparison.png")
    
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
        summary_df.to_csv(get_backtest_path("multi_timeframe_performance_summary.csv"), index=False)
        print(f"Resumo de desempenho salvo em {get_backtest_path('multi_timeframe_performance_summary.csv')}")
        
        # Criar gráfico de barras comparando métricas principais
        plt.figure(figsize=(14, 8))
        
        # Extrair dados para gráfico
        strategies = summary_df['Estratégia']
        returns = summary_df['Retorno Total'].str.rstrip('%').astype(float)
        sharpe = summary_df['Índice Sharpe'].astype(float)
        win_rate = summary_df['Taxa de Acerto'].str.rstrip('%').astype(float)
        
        # Criar posições das barras
        x = range(len(strategies))
        width = 0.25
        
        # Plotar barras agrupadas
        plt.bar([i - width for i in x], returns, width=width, label='Retorno Total (%)', color='green')
        plt.bar(x, sharpe, width=width, label='Índice Sharpe', color='blue')
        plt.bar([i + width for i in x], win_rate, width=width, label='Taxa de Acerto (%)', color='orange')
        
        # Adicionar rótulos e legendas
        plt.xlabel('Estratégia')
        plt.ylabel('Valor')
        plt.title('Comparação de Métricas de Desempenho')
        plt.xticks(x, strategies, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Salvar gráfico
        plt.savefig(get_backtest_path('multi_timeframe_metrics_comparison.png'))
        print(f"Gráfico de métricas salvo em {get_backtest_path('multi_timeframe_metrics_comparison.png')}")


if __name__ == "__main__":
    main()