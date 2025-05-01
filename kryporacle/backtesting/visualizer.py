import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import logging
import os
from pathlib import Path

from .utils import get_backtest_path, sanitize_filename

logger = logging.getLogger(__name__)

def plot_backtest_results(df: pd.DataFrame, strategy_name: str, 
                         timeframe: str = '1d',
                         figsize: Tuple[float, float] = (15, 10),
                         filename: Optional[str] = None) -> plt.Figure:
    """Visualize backtest results with price, signals and portfolio value.
    
    Args:
        df (pd.DataFrame): DataFrame with backtest results
        strategy_name (str): Name of the strategy
        figsize (Tuple[float, float]): Figure size
        filename (str, optional): File name to save the plot
        
    Returns:
        plt.Figure: Matplotlib figure with the plot
"""
    if df.empty:
        logger.error("DataFrame is empty, cannot generate plot")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Define grid for subplots
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.1)
    
    # PLOT 1: Price with signals
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title(f"Backtest Results - {strategy_name}", fontsize=14)
    
    # Plot price
    ax1.plot(df.index, df['close'], label='Price', color='blue', alpha=0.7)
    
    # Plot buy signals
    buy_signals = df[df['signal'] == 1]
    ax1.scatter(buy_signals.index, buy_signals['close'], 
               color='green', marker='^', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = df[df['signal'] == -1]
    ax1.scatter(sell_signals.index, sell_signals['close'], 
               color='red', marker='v', s=100, label='Sell Signal')
    
    # Add any available technical indicators
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        ax1.plot(df.index, df['sma_20'], color='orange', linestyle='--', alpha=0.7, label='SMA 20')
        ax1.plot(df.index, df['sma_50'], color='purple', linestyle='--', alpha=0.7, label='SMA 50')
    
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'bb_middle' in df.columns:
        ax1.plot(df.index, df['bb_upper'], color='grey', linestyle='--', alpha=0.4, label='BB Upper')
        ax1.plot(df.index, df['bb_lower'], color='grey', linestyle='--', alpha=0.4, label='BB Lower')
        ax1.plot(df.index, df['bb_middle'], color='grey', linestyle='-', alpha=0.4, label='BB Middle')
        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], color='grey', alpha=0.1)
    
    # Add legend and grid
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Price')
    
    # PLOT 2: Portfolio value
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df['portfolio_value'], label='Portfolio Value', color='green')
    ax2.set_ylabel('Portfolio Value')
    ax2.grid(True, alpha=0.3)
    
    # PLOT 3: Drawdown
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(df.index, 0, df['drawdown'], color='red', alpha=0.3, label='Drawdown')
    ax3.set_ylabel('Drawdown')
    ax3.set_ylim(-1, 0.05)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis with dates
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Only show x labels for the bottom plot
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # Add performance metrics as text
    from .metrics import calculate_metrics
    metrics = calculate_metrics(df, timeframe=timeframe)
    
    if metrics:
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}\n"
            f"Annual Return: {metrics['annual_return']:.2%}\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
            f"Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']:.2%}"
        )
        
        plt.figtext(0.01, 0.01, metrics_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure if filename is provided
    if filename:
        # Sanitize the filename
        sanitized_filename = sanitize_filename(filename)
        
        # Get full path in the backtests directory using the environment variable
        full_path = get_backtest_path(sanitized_filename)
        
        # Create directory if it doesn't exist (for subdirectories)
        directory = os.path.dirname(full_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        plt.savefig(full_path)
        logger.info(f"Plot saved to {full_path}")
    
    return fig