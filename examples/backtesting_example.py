import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to path for imports
sys.path.append(".")

from kryporacle.data.exchanges.binance import BinanceCollector
from kryporacle.data.storage import DataStorage
from kryporacle.backtesting.engine import BacktestEngine
from kryporacle.backtesting.strategies import (
    MovingAverageCrossover, 
    RSIStrategy, 
    BollingerBandsStrategy,
    MACDStrategy
)
from kryporacle.backtesting.advanced_strategies import (
    TrendFollowingStrategy,
    DualMACDStrategy,
    VolatilityBreakoutStrategy,
    DivergenceStrategy  # Adicionar importação aqui
)
from kryporacle.backtesting.metrics import calculate_metrics
from kryporacle.backtesting.visualizer import plot_backtest_results
from kryporacle.backtesting.utils import get_backtest_dir, get_backtest_path

def main():
    """Example script demonstrating the advanced backtesting functionality."""
    # Initialize components
    collector = BinanceCollector()
    storage = DataStorage()
    
    # Set up backtest parameters
    exchange_id = "binance"
    symbol = "BTC/USDT"
    timeframe = "4h"
    initial_capital = 10000.0  # Start with $10,000
    
    # Get data for the last 365 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)
    # start_time = end_time - timedelta(days=365)
    
    print(f"Fetching {symbol} data from {start_time} to {end_time}...")
    print(f"Backtest results will be saved to: {get_backtest_dir()}")
    try:
        # First, ensure we have historical data
        data = collector.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Save the data
        storage.save_market_data(exchange_id, data)
        print(f"Saved {len(data['data'])} records to storage")
        
        # Initialize backtest engine
        engine = BacktestEngine(initial_capital=initial_capital, 
                               commission=0.001)  # 0.1% commission
        
        # Run multiple backtests with different strategies
        results = {}
        
        #------------------------------------------------
        # Estratégias Básicas
        #------------------------------------------------
        
        # Strategy 1: Moving Average Crossover (50/200)
        print("\nRunning Moving Average Crossover (50/200) strategy...")
        ma_strategy = MovingAverageCrossover(fast_period=50, slow_period=200)
        ma_results = engine.run(
            strategy=ma_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['MA Crossover (50/200)'] = ma_results
        
        # Strategy 2: RSI (14) with 30/70 thresholds
        print("\nRunning RSI strategy...")
        rsi_strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
        rsi_results = engine.run(
            strategy=rsi_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['RSI (14)'] = rsi_results
        
        # Strategy 3: Bollinger Bands
        print("\nRunning Bollinger Bands strategy...")
        bb_strategy = BollingerBandsStrategy(window=20, std_dev=2.0)
        bb_results = engine.run(
            strategy=bb_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['Bollinger Bands'] = bb_results
        
        # Strategy 4: MACD
        print("\nRunning MACD strategy...")
        macd_strategy = MACDStrategy()
        macd_results = engine.run(
            strategy=macd_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['MACD'] = macd_results
        
        #------------------------------------------------
        # Estratégias Avançadas
        #------------------------------------------------
        
        # Strategy 5: Trend Following
        print("\nRunning Trend Following strategy...")
        trend_strategy = TrendFollowingStrategy(ma_period=20, trend_strength=25)
        trend_results = engine.run(
            strategy=trend_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['Trend Following'] = trend_results
        
        # Strategy 6: Dual MACD
        print("\nRunning Dual MACD strategy...")
        dual_macd_strategy = DualMACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        dual_macd_results = engine.run(
            strategy=dual_macd_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['Dual MACD'] = dual_macd_results
        
        # Strategy 7: Volatility Breakout
        print("\nRunning Volatility Breakout strategy...")
        breakout_strategy = VolatilityBreakoutStrategy(window=20, volatility_factor=1.5)
        breakout_results = engine.run(
            strategy=breakout_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['Volatility Breakout'] = breakout_results
        
        # Strategy 8: Divergence Strategy
        print("\nRunning Divergence Strategy...")
        divergence_strategy = DivergenceStrategy(
            indicator='rsi',           # Indicador para detectar divergências
            lookback_period=14,        # Período do RSI
            divergence_window=5,       # Janela para detectar extremos locais
            signal_threshold=0.0       # Sem filtro de força do sinal
        )
        divergence_results = engine.run(
            strategy=divergence_strategy,
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        results['RSI Divergence'] = divergence_results
        
        # Visualize each backtest
        for strategy_name, result_df in results.items():
            if not result_df.empty:
                print(f"\nResults for {strategy_name}:")
                metrics = calculate_metrics(result_df, timeframe=timeframe)
                print_metrics(metrics)
                
                # Create and save plot
                plot_backtest_results(
                    result_df, 
                    strategy_name,
                    timeframe=timeframe,
                    filename=f"backtest_{strategy_name.replace(' ', '_').lower()}.png"
                )
                
        # Compare all strategies
        compare_strategies(results, timeframe)

        create_strategy_comparison_heatmap(results, timeframe)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def print_metrics(metrics: dict):
    """Exibe métricas de desempenho formatadas."""
    print(f"  Valor Inicial: R${metrics['initial_value']:.2f}")
    print(f"  Valor Final: R${metrics['final_value']:.2f}")
    print(f"  Retorno Total: {metrics['total_return']:.2%}")
    print(f"  Retorno Anual: {metrics['annual_return']:.2%}")
    print(f"  Volatilidade: {metrics['volatility']:.2%}")
    print(f"  Índice Sharpe: {metrics['sharpe_ratio']:.2f}")
    print(f"  Drawdown Máximo: {metrics['max_drawdown']:.2%}")
    print(f"  Total de Operações: {metrics['total_trades']}")
    print(f"  Taxa de Acerto: {metrics['win_rate']:.2%}")

def compare_strategies(results: dict, timeframe: str):
    """Compara o desempenho de diferentes estratégias."""
    print("\nComparação de Estratégias:")
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot portfolio value for each strategy
    for strategy_name, result_df in results.items():
        if not result_df.empty:
            plt.plot(result_df.index, result_df['portfolio_value'], label=strategy_name)
    
    plt.title('Comparação de Desempenho das Estratégias', fontsize=14)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Valor da Carteira (R$)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(get_backtest_path("all_strategies_comparison.png"))
    print(f"Strategy comparison plot saved to {get_backtest_path('all_strategies_comparison.png')}")
    
    # Create a summary table
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
    
    # Print summary table
    if summary:
        print("\nResumo de Desempenho:")
        summary_df = pd.DataFrame(summary)
        # Ordenar por retorno anual (do maior para o menor)
        summary_df['Sort_Value'] = summary_df['Retorno Anual'].str.rstrip('%').astype(float)
        summary_df = summary_df.sort_values('Sort_Value', ascending=False).drop('Sort_Value', axis=1)
        print(summary_df.to_string(index=False))
        
        # Salvar tabela de resumo como CSV
        summary_df.to_csv(get_backtest_path("strategy_performance_summary.csv"), index=False)
        print(f"Strategy performance summary saved to {get_backtest_path('strategy_performance_summary.csv')}")

def create_strategy_comparison_heatmap(results: dict, timeframe: str):
    """Cria um mapa de calor comparando diferentes aspectos de desempenho das estratégias."""
    if not results:
        return
    
    # Extract metrics for each strategy
    metrics_data = {}
    for strategy_name, result_df in results.items():
        if not result_df.empty:
            metrics = calculate_metrics(result_df, timeframe=timeframe)
            metrics_data[strategy_name] = {
                'Retorno Total': metrics['total_return'],
                'Retorno Anual': metrics['annual_return'],
                'Índice Sharpe': metrics['sharpe_ratio'],
                'Drawdown Máximo': metrics['max_drawdown'],
                'Taxa de Acerto': metrics['win_rate']
            }
    
    if not metrics_data:
        return
    
    # Create DataFrame for heatmap
    metrics_df = pd.DataFrame(metrics_data).T
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize data for better visualization
    normalized_df = metrics_df.copy()
    for col in normalized_df.columns:
        if col == 'Drawdown Máximo':  # Smaller is better for drawdown
            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
            normalized_df[col] = 1 - normalized_df[col]  # Invert so higher is better
        else:  # Higher is better for other metrics
            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
    
    # Plot heatmap
    import seaborn as sns
    sns.heatmap(normalized_df, annot=metrics_df, fmt='.2%', cmap='RdYlGn', linewidths=0.5)
    plt.title('Comparação de Desempenho das Estratégias (Pontuações Normalizadas)', fontsize=14)
    plt.tight_layout()
    plt.savefig(get_backtest_path("strategy_heatmap_comparison.png"))
    print(f"Strategy heatmap comparison saved to {get_backtest_path('strategy_heatmap_comparison.png')}")


if __name__ == "__main__":
    main()