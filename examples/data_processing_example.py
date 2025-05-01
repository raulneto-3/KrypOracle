import pandas as pd
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
from kryporacle.processing import (
    normalize_data,
    calculate_returns,
    rolling_window_agg
)

def main():
    """Example script demonstrating the data processing pipeline."""
    # Initialize components
    collector = BinanceCollector()
    storage = DataStorage()
    processor = DataProcessor(storage)
    
    # Fetch some historical data (last 7 days of hourly BTC/USDT)
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # Get data for the last 7 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
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
            
        print("\nOriginal data sample:")
        print(df.head())
        print(f"Data shape: {df.shape}")
        
        # Calculate returns
        df_returns = calculate_returns(df, price_col='close', periods=[1, 24, 168])  # 1 hour, 1 day, 1 week returns
        print("\nData with returns:")
        print(df_returns[['timestamp', 'close', 'return_1', 'return_24', 'return_168']].tail())
        
        # Add rolling window metrics
        df_rolling = rolling_window_agg(df, window=24, columns=['close', 'volume'], 
                                       agg_func={'close': 'mean', 'volume': 'sum'})
        print("\nData with rolling metrics:")
        print(df_rolling[['timestamp', 'close', 'close_rolling_24_mean', 'volume', 'volume_rolling_24_sum']].tail())
        
        # Normalize data
        df_norm = normalize_data(df, columns=['close', 'volume'], method='minmax')
        print("\nNormalized data:")
        print(df_norm[['timestamp', 'close', 'close_norm', 'volume', 'volume_norm']].tail())
        
        print("\nProcessing pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()