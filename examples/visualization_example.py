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
from kryporacle.processing.processor import DataProcessor
from kryporacle.analysis import trends, volatility
from kryporacle.visualization.charts import OHLCChart
from kryporacle.visualization.indicators import IndicatorVisualizer
from kryporacle.visualization.dashboard import Dashboard

def main():
    """Example script demonstrating the visualization functionality."""
    # Initialize components
    collector = BinanceCollector()
    storage = DataStorage()
    processor = DataProcessor(storage)
    
    # Fetch historical data for BTC/USDT (last 30 days of 4-hour data)
    symbol = "BTC/USDT"
    timeframe = "4h"
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Fetching {symbol} data from {start_time} to {end_time}...")
    try:
        data = collector.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Save the data
        storage.save_market_data("binance", data)
        print(f"Saved {len(data['data'])} records to storage")
        
        # Load and process the data
        df = processor.process_market_data("binance", symbol, timeframe)
        
        if df.empty:
            print("No data found in storage")
            return
            
        print(f"\nData loaded: {df.shape[0]} records")
        
        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        df = trends.calculate_indicators(df)
        df = volatility.calculate_volatility_indicators(df)
        
        # Create visualizations
        create_basic_charts(df)
        create_indicator_charts(df)
        create_dashboard(df)
        
    except Exception as e:
        print(f"Error: {e}")

def create_basic_charts(df):
    """Create and save basic charts."""
    print("\nCreating basic charts...")
    
    # Initialize chart object
    ohlc_chart = OHLCChart()
    
    # 1. Create candlestick chart
    fig, ax = ohlc_chart.plot_candlestick(df, title="BTC/USDT Candlestick Chart")
    ohlc_chart.save_figure(fig, "candlestick_chart.png")
    
    # 2. Create OHLC chart with volume
    fig, axes = ohlc_chart.plot_ohlc(df, title="BTC/USDT OHLC Chart", volume=True)
    ohlc_chart.save_figure(fig, "ohlc_with_volume.png")
    
    # 3. Create a simple line chart of closing prices
    fig, ax = ohlc_chart.plot_line(df, 'close', title="BTC/USDT Closing Prices")
    ohlc_chart.save_figure(fig, "closing_prices.png")
    
    print("Basic charts saved.")

def create_indicator_charts(df):
    """Create and save charts with technical indicators."""
    print("\nCreating technical indicator charts...")
    
    # Initialize indicator visualizer
    indicator_viz = IndicatorVisualizer()
    
    # 1. Create price chart with SMA
    fig, ax = indicator_viz.plot_with_sma(df, title="BTC/USDT with Simple Moving Averages")
    indicator_viz.save_figure(fig, "price_with_sma.png")
    
    # 2. Create price chart with EMA
    fig, ax = indicator_viz.plot_with_ema(df, title="BTC/USDT with Exponential Moving Averages")
    indicator_viz.save_figure(fig, "price_with_ema.png")
    
    # 3. Create MACD chart
    fig, ax = indicator_viz.plot_macd(df, title="MACD Indicator")
    indicator_viz.save_figure(fig, "macd_indicator.png")
    
    # 4. Create RSI chart
    fig, ax = indicator_viz.plot_rsi(df, title="Relative Strength Index")
    indicator_viz.save_figure(fig, "rsi_indicator.png")
    
    # 5. Create Bollinger Bands chart
    fig, ax = indicator_viz.plot_bollinger_bands(df, title="Bollinger Bands")
    indicator_viz.save_figure(fig, "bollinger_bands.png")
    
    # 6. Create multi-indicator chart
    fig, axes = indicator_viz.plot_multi_indicator_chart(
        df, 
        indicators=['candlestick', 'sma', 'volume', 'macd', 'rsi'],
        figsize=(12, 15)
    )
    indicator_viz.save_figure(fig, "multi_indicator_chart.png")
    
    print("Technical indicator charts saved.")

def create_dashboard(df):
    """Create and save market analysis dashboard."""
    print("\nCreating market analysis dashboard...")
    
    # Initialize dashboard
    dashboard = Dashboard(figsize=(15, 10))
    
    # Create comprehensive market dashboard
    fig, axes_dict = dashboard.create_market_dashboard(df, title="BTC/USDT Market Analysis")
    dashboard.save_figure(fig, "market_dashboard.png")
    
    # Create multi-asset dashboard if we had more assets
    # For demo purposes, let's create a dummy second asset by shifting the data
    df2 = df.copy()
    df2['close'] = df2['close'] * 0.95  # 5% lower
    df2['open'] = df2['open'] * 0.95
    df2['high'] = df2['high'] * 0.95
    df2['low'] = df2['low'] * 0.95
    
    # Create multi-asset dashboard
    fig = dashboard.create_multi_asset_dashboard({
        'BTC/USDT': df,
        'ETH/USDT': df2  # This is just a dummy for demo purposes
    })
    dashboard.save_figure(fig, "multi_asset_dashboard.png")
    
    print("Dashboards saved.")

if __name__ == "__main__":
    main()



# # Import needed components
# from kryporacle.visualization.charts import OHLCChart
# from kryporacle.visualization.indicators import IndicatorVisualizer
# from kryporacle.visualization.dashboard import Dashboard

# # 1. Create basic candlestick chart
# ohlc_chart = OHLCChart()
# fig, ax = ohlc_chart.plot_candlestick(df, title="BTC/USDT Price")
# ohlc_chart.save_figure(fig, "btc_price.png")

# # 2. Create chart with technical indicators
# indicator_viz = IndicatorVisualizer()
# fig, ax = indicator_viz.plot_bollinger_bands(df, title="Bollinger Bands")
# indicator_viz.save_figure(fig, "bollinger.png")

# # 3. Create comprehensive dashboard
# dashboard = Dashboard()
# fig, axes = dashboard.create_market_dashboard(df, title="Market Analysis")
# dashboard.save_figure(fig, "market_dashboard.png")