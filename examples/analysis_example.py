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
from kryporacle.analysis.trends import (
    simple_moving_average, 
    exponential_moving_average, 
    macd, 
    rsi, 
    golden_death_cross
)
from kryporacle.analysis.volatility import (
    bollinger_bands,
    average_true_range,
    historical_volatility
)

def main():
    """Example script demonstrating the analysis functionality."""
    # Initialize components
    collector = BinanceCollector()
    storage = DataStorage()
    processor = DataProcessor(storage)
    
    # Fetch some historical data (last 30 days of 4-hour BTC/USDT)
    symbol = "BTC/USDT"
    timeframe = "4h"
    
    # Get data for the last 30 days
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
        
        # Calculate trend indicators
        print("\nCalculating trend indicators...")
        df = simple_moving_average(df, window_sizes=[20, 50])
        df = exponential_moving_average(df, window_sizes=[12, 26])
        df = macd(df)
        df = rsi(df)
        df = golden_death_cross(df)
        
        # Calculate volatility indicators
        print("Calculating volatility indicators...")
        df = bollinger_bands(df)
        df = average_true_range(df)
        df = historical_volatility(df)
        
        # Display results
        print("\nTrend and volatility indicators:")
        indicator_columns = [
            'timestamp', 'close', 'sma_20', 'ema_12', 
            'rsi', 'macd_line', 'macd_signal',
            'bb_upper', 'bb_lower', 'atr', 'volatility'
        ]
        print(df[indicator_columns].tail())
        
        # Create some basic plots
        plot_data(df)
        
    except Exception as e:
        print(f"Error: {e}")

def plot_data(df):
    """Create plots of the data with indicators."""
    # Create a figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Price with SMA and EMA
    axes[0].set_title(f"Price with Moving Averages")
    axes[0].plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    axes[0].plot(df['timestamp'], df['sma_20'], label='SMA(20)', color='red')
    axes[0].plot(df['timestamp'], df['sma_50'], label='SMA(50)', color='green')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Bollinger Bands
    axes[1].set_title(f"Bollinger Bands")
    axes[1].plot(df['timestamp'], df['close'], label='Close Price', color='blue')
    axes[1].plot(df['timestamp'], df['bb_upper'], label='Upper Band', color='red')
    axes[1].plot(df['timestamp'], df['bb_middle'], label='Middle Band', color='green')
    axes[1].plot(df['timestamp'], df['bb_lower'], label='Lower Band', color='red')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: RSI
    axes[2].set_title(f"RSI")
    axes[2].plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
    axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.5)
    axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('RSI')
    axes[2].set_xlabel('Date')
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('crypto_analysis.png')
    print("Chart saved as 'crypto_analysis.png'")
    
if __name__ == "__main__":
    main()