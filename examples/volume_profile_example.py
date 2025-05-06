"""
Exemplo de uso da estratégia de perfil de volume (Volume Profile).

Este exemplo demonstra como analisar a distribuição de volume em diferentes níveis de preço
para identificar zonas de suporte/resistência, detectar divergências entre preço e volume,
e gerar sinais de trading baseados nessas análises.
"""

import os
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
from kryporacle.backtesting.strategies.volume_profile import VolumeProfileStrategy
from kryporacle.analysis.volume_profile import VolumeProfileAnalyzer
from kryporacle.backtesting.metrics import calculate_metrics
from kryporacle.backtesting.utils import get_backtest_dir, get_backtest_path
from kryporacle.visualization.charts import OHLCChart
from kryporacle.visualization.json_generator import generate_strategy_backtest_json, generate_strategy_comparison_json

# Update the BacktestEngine initialization in main() function

def main():
    """
    Executa backtest da estratégia de perfil de volume em dados reais de criptomoedas.
    """
    # Parâmetros de backtest
    symbol = "BTC/USDT"
    timeframes = ["4h"]  # Múltiplos timeframes para testar
    start_date = datetime.now() - timedelta(days=180)
    initial_capital = 10000.0
    
    # Instanciar coletor de dados e storage
    collector = BinanceCollector()
    storage = DataStorage()
    
    for timeframe in timeframes:
        print(f"\n{'-'*80}")
        print(f"Executando estratégia de perfil de volume para {symbol} em {timeframe}")
        print(f"{'-'*80}")
        
        # Obter dados
        data_dict = collector.fetch_historical_data(symbol, timeframe, start_date)
        
        # Verificar se temos dados
        if not data_dict or not data_dict.get('data'):
            print(f"Nenhum dado encontrado para {symbol} em {timeframe}")
            continue
        
        # Converter o dicionário para DataFrame
        data = pd.DataFrame(data_dict['data'])
        
        # Converter timestamp para datetime e definir como índice
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        
        # Armazenar dados
        market_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data.reset_index().to_dict('records')
        }
        storage.save_market_data('binance', market_data)
        
        # Inicializar engine de backtest - CORREÇÃO: Adaptar para a API correta
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.001,  # Usar commission em vez de maker_fee/taker_fee
            slippage=0.0005  # Adicionar slippage estimado
        )
        
        # Definir variações da estratégia para testar
        strategies = {
            "Volume Profile Padrão": VolumeProfileStrategy(
                price_bins=10,
                value_area_threshold=70.0,
                divergence_window=14,
                use_vwap=True,
                volume_profile_window=100,
                cmf_window=20,
                cmf_threshold=0.05,
                reset_period='D'
            ),
            "VP Alta Sensibilidade": VolumeProfileStrategy(
                price_bins=20,
                value_area_threshold=60.0,
                divergence_window=10,
                use_vwap=True,
                volume_profile_window=50,
                cmf_window=14,
                cmf_threshold=0.03,
                reset_period='D'
            ),
            "VP Conservador": VolumeProfileStrategy(
                price_bins=8,
                value_area_threshold=80.0,
                divergence_window=20,
                use_vwap=True,
                volume_profile_window=150,
                cmf_window=30,
                cmf_threshold=0.07,
                reset_period='D'
            ),
            "VP Sem VWAP": VolumeProfileStrategy(
                price_bins=10,
                value_area_threshold=70.0,
                divergence_window=14,
                use_vwap=False,
                volume_profile_window=100,
                cmf_window=20,
                cmf_threshold=0.05,
                reset_period=None
            )
        }
        
        results = {}
        
        for name, strategy in strategies.items():
            print(f"\nExecutando estratégia {name}...")
            # CORREÇÃO: Adaptar chamada do método run para a API correta
            result_df = engine.run(
                strategy=strategy,
                exchange_id='binance',
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=None,
                calculate_indicators=True
            )
            
            if result_df.empty:
                print(f"Nenhum sinal gerado pela estratégia {name}")
                continue
            
            results[name] = result_df
            
            # Calcular métricas
            metrics = calculate_metrics(result_df, timeframe=timeframe)
            print_metrics(metrics)
            
            # Visualizar resultados
            plot_strategy_results(result_df, name, symbol, timeframe)
            
            # Gerar JSON para visualização interativa
            json_file = generate_strategy_backtest_json(
                result_df, 
                strategy_name=name,
                symbol=symbol, 
                timeframe=timeframe,
                start_time=start_date,        
                end_time=datetime.now(),      
                metrics=metrics,
                parameters=strategy_parameters_to_dict(strategy),
                filename=os.path.join(get_backtest_dir(), f"volume_profile_{name.lower().replace(' ', '_')}.json")
            )
            print(f"Visualização interativa salva em {json_file}")
        
        # Comparar estratégias
        if len(results) > 1:
            compare_strategies(results, timeframe)
            
            # Gerar JSON para comparação interativa
            json_file = generate_strategy_comparison_json(
                results,
                metrics={name: calculate_metrics(df, timeframe=timeframe) for name, df in results.items()},
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_date,
                end_time=datetime.now(),
                filename=os.path.join(get_backtest_dir(), "volume_profile_comparison.json")
            )
            print(f"Comparativo interativo salvo em {json_file}")
            
def strategy_parameters_to_dict(strategy: VolumeProfileStrategy) -> dict:
    """Converte os parâmetros da estratégia para um dicionário."""
    return {
        "price_bins": strategy.price_bins,
        "value_area_threshold": strategy.value_area_threshold,
        "divergence_window": strategy.divergence_window,
        "use_vwap": strategy.use_vwap,
        "volume_profile_window": strategy.volume_profile_window,
        "cmf_window": strategy.cmf_window,
        "cmf_threshold": strategy.cmf_threshold,
        "reset_period": strategy.reset_period
    }

def print_metrics(metrics: dict):
    """Imprime as métricas de desempenho do backtest."""
    print("\nMétricas de Desempenho:")
    print(f"Retorno Total: {metrics['total_return']:.2%}")
    print(f"Retorno Anualizado: {metrics['annual_return']:.2%}")
    print(f"Índice de Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"Drawdown Máximo: {metrics['max_drawdown']:.2%}")
    print(f"Total de Operações: {metrics['total_trades']}")
    print(f"Taxa de Acerto: {metrics['win_rate']:.2%}")
    print(f"Razão Lucro/Perda: {metrics['profit_factor']:.2f}")

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
    
    # Plotar VWAP se disponível
    if 'vwap' in chart_df.columns:
        axes[0].plot(chart_df['timestamp'], chart_df['vwap'], 
                    color='blue', linewidth=1.0, label='VWAP')
    
    # Plotar áreas de valor (VAH, VAL, POC) se disponíveis
    if 'vah' in chart_df.columns and 'val' in chart_df.columns:
        # Filtrar valores NaN
        value_data = chart_df[~chart_df['vah'].isna() & ~chart_df['val'].isna()]
        if not value_data.empty:
            # Value Area High (resistência)
            axes[0].plot(value_data['timestamp'], value_data['vah'], 
                       color='red', linestyle='--', alpha=0.7, linewidth=1.0,
                       label='Value Area High')
            
            # Value Area Low (suporte)
            axes[0].plot(value_data['timestamp'], value_data['val'], 
                       color='green', linestyle='--', alpha=0.7, linewidth=1.0,
                       label='Value Area Low')
            
            # Point of Control (nível de maior volume)
            if 'poc' in value_data.columns:
                axes[0].plot(value_data['timestamp'], value_data['poc'], 
                           color='blue', linestyle='--', alpha=0.7, linewidth=1.0,
                           label='Point of Control')
    
    # Plotar indicadores de acumulação/distribuição no painel inferior
    if 'cmf' in chart_df.columns and len(axes) > 1:
        ax_cmf = axes[1].twinx()
        ax_cmf.plot(chart_df['timestamp'], chart_df['cmf'], 
                   color='orange', linewidth=1.0, label='Chaikin Money Flow')
        ax_cmf.set_ylabel('CMF', color='orange')
        ax_cmf.tick_params(axis='y', labelcolor='orange')
        ax_cmf.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_cmf.axhline(y=0.05, color='green', linestyle=':', alpha=0.5)
        ax_cmf.axhline(y=-0.05, color='red', linestyle=':', alpha=0.5)
        
        # Legenda para CMF
        handles3, labels3 = ax_cmf.get_legend_handles_labels()
        axes[1].legend(handles3, labels3, loc='upper right')
    
    # Adicionar legenda
    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = ax_portfolio.get_legend_handles_labels()
    axes[0].legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    # Salvar gráfico
    filename = f"volume_profile_{strategy_name.lower().replace(' ', '_')}"
    ohlc_chart.save_figure(fig, filename, directory=get_backtest_dir())
    print(f"Gráfico salvo em {get_backtest_dir()}/{filename}.png")

def compare_strategies(results: dict, timeframe: str):
    """Compara o desempenho de diferentes estratégias de perfil de volume."""
    print("\nComparação de Estratégias de Perfil de Volume:")
    
    # Criar figura usando OHLCChart
    chart = OHLCChart(figsize=(15, 8))
    fig, ax = chart.create_figure(1, 1)  # Sem subplot para volume
    
    # Plotar valor da carteira para cada estratégia
    for strategy_name, result_df in results.items():
        if not result_df.empty:
            # Criar coluna timestamp sem resetar o índice
            chart_df = result_df.copy()
            chart_df['timestamp'] = chart_df.index
            
            # Garantir que timestamp seja datetime
            chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
            
            ax[0].plot(chart_df['timestamp'], chart_df['portfolio_value'], label=strategy_name)
    
    ax[0].set_title('Comparação de Estratégias de Perfil de Volume', fontsize=14)
    ax[0].set_xlabel('Data', fontsize=12)
    ax[0].set_ylabel('Valor da Carteira ($)', fontsize=12)
    ax[0].legend(fontsize=10)
    ax[0].grid(True, alpha=0.3)
    
    # Ajustar formato de data
    chart.adjust_date_format(ax[0])
    
    # Salvar gráfico
    chart.save_figure(fig, "volume_profile_strategies_comparison", directory=get_backtest_dir())
    print(f"Gráfico de comparação salvo em {get_backtest_dir()}/volume_profile_strategies_comparison.png")
    
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
        summary_df.to_csv(get_backtest_path("volume_profile_performance_summary.csv"), index=False)
        print(f"Resumo de desempenho salvo em {get_backtest_path('volume_profile_performance_summary.csv')}")
        
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
        plt.savefig(get_backtest_path('volume_profile_metrics_comparison.png'))
        print(f"Gráfico de métricas salvo em {get_backtest_path('volume_profile_metrics_comparison.png')}")

if __name__ == "__main__":
    main()