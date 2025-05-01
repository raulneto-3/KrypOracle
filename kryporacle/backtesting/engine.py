import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import logging

from ..data.storage import DataStorage
from ..processing.processor import DataProcessor

logger = logging.getLogger(__name__)

class BacktestEngine:
    """Engine for backtesting trading strategies on historical data."""
    
    def __init__(self, initial_capital: float = 10000.0, 
                commission: float = 0.001, slippage: float = 0.0):
        """Initialize the backtesting engine.
        
        Args:
            initial_capital (float): Starting capital for backtesting
            commission (float): Commission rate per trade (e.g., 0.001 = 0.1%)
            slippage (float): Slippage assumption per trade (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.storage = DataStorage()
        self.processor = DataProcessor(self.storage)
        
    def run(self, strategy, exchange_id: str, symbol: str, timeframe: str,
           start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, 
           calculate_indicators: bool = True) -> pd.DataFrame:
        """Run backtesting with a strategy on historical data.
        
        Args:
            strategy: Strategy instance to backtest
            exchange_id (str): Exchange identifier (e.g., 'binance')
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
            timeframe (str): Timeframe for the data (e.g., '1h', '1d')
            start_time (datetime, optional): Start time for backtesting
            end_time (datetime, optional): End time for backtesting
            calculate_indicators (bool): Whether to calculate indicators before running the strategy
            
        Returns:
            pd.DataFrame: DataFrame with backtest results
        """
        logger.info(f"Starting backtest for {symbol} on {exchange_id} ({timeframe})")
        
        # Load and process historical data
        df = self.processor.process_market_data(
            exchange_id=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        if df.empty:
            logger.warning(f"No data found for {exchange_id}/{symbol}/{timeframe}")
            return pd.DataFrame()
        
        # Calculate indicators if needed
        if calculate_indicators:
            logger.info("Calculating technical indicators")
            from ..analysis import trends, volatility
            df = trends.calculate_indicators(df)
            df = volatility.calculate_volatility_indicators(df)
        
        # Check if all required indicators for the strategy are present
        required_indicators = strategy.required_indicators
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        
        if missing_indicators:
            logger.error(f"Missing indicators required by strategy: {missing_indicators}")
            return pd.DataFrame()
        
        # Initialize backtest DataFrame with position and portfolio columns
        df['position'] = 0
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['portfolio_value'] = self.initial_capital
        df['cash'] = self.initial_capital
        df['holdings'] = 0.0
        df['trade_count'] = 0
        
        # Run strategy to generate signals
        df = strategy.generate_signals(df)
        
        # Apply signals to simulate trading
        df = self._apply_signals(df, symbol)
        
        # Calculate performance metrics
        df = self._calculate_performance(df)
        
        logger.info(f"Backtest completed for {symbol} on {exchange_id} ({timeframe})")
        return df
    
    def _apply_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Apply trade signals to simulate trading.
        
        Args:
            df (pd.DataFrame): DataFrame with trade signals
            symbol (str): Trading pair symbol
            
        Returns:
            pd.DataFrame: DataFrame with trade results
        """
        # Make sure we have the required columns
        if 'signal' not in df.columns:
            logger.error("DataFrame must contain 'signal' column from strategy")
            return df
        
        # Get base and quote currencies from symbol
        base_currency, quote_currency = symbol.split('/')
        
        # Set up initial state
        position = 0  # 0 = no position, 1 = long, -1 = short (if supported)
        cash = self.initial_capital
        holdings = 0.0
        trade_count = 0
        entry_price = 0.0
        
        # Iterate through the DataFrame to apply trading logic
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            signal = df.iloc[i]['signal']
            
            # Determine if we need to change position
            if signal == 1 and position == 0:  # Buy signal
                # Apply slippage (worse price when buying)
                adjusted_price = current_price * (1 + self.slippage)
                
                # Calculate how much to buy
                amount_to_buy = cash * (1 - self.commission) / adjusted_price
                holdings = amount_to_buy
                
                # Update cash
                cash = 0
                
                position = 1
                trade_count += 1
                entry_price = adjusted_price
                
                # Update DataFrame
                df.at[df.index[i], 'position'] = position
                df.at[df.index[i], 'entry_price'] = entry_price
                
            elif signal == -1 and position == 1:  # Sell signal
                # Apply slippage (worse price when selling)
                adjusted_price = current_price * (1 - self.slippage)
                
                # Calculate proceeds from selling
                cash = holdings * adjusted_price * (1 - self.commission)
                holdings = 0
                
                position = 0
                df.at[df.index[i], 'position'] = position
                df.at[df.index[i], 'exit_price'] = adjusted_price
            
            # Calculate portfolio value
            portfolio_value = cash + holdings * current_price
            df.at[df.index[i], 'portfolio_value'] = portfolio_value
            df.at[df.index[i], 'cash'] = cash
            df.at[df.index[i], 'holdings'] = holdings
            df.at[df.index[i], 'trade_count'] = trade_count
        
        return df
    
    def _calculate_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics for a backtest.
        
        Args:
            df (pd.DataFrame): DataFrame with trade history
            
        Returns:
            pd.DataFrame: DataFrame with performance metrics added
        """
        # Calculate daily returns
        df['daily_returns'] = df['portfolio_value'].pct_change()
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['daily_returns']).cumprod() - 1
        
        # Calculate drawdowns
        df['peak_value'] = df['portfolio_value'].cummax()
        df['drawdown'] = (df['portfolio_value'] - df['peak_value']) / df['peak_value']
        
        return df