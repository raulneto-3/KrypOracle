import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path

from .charts import ChartBase, OHLCChart
from .indicators import IndicatorVisualizer

logger = logging.getLogger(__name__)

class Dashboard(ChartBase):
    """Class for creating dashboards with multiple charts and indicators."""
    
    def __init__(self, **kwargs):
        """Initialize the dashboard."""
        super().__init__(**kwargs)
        self.ohlc_chart = OHLCChart(**kwargs)
        self.indicator_viz = IndicatorVisualizer(**kwargs)
        
    def create_market_dashboard(self, df: pd.DataFrame,
                              title: str = "Market Analysis Dashboard",
                              figsize: Tuple[float, float] = (16, 12)) -> Tuple[Figure, Dict[str, Axes]]:
        """Create a comprehensive market analysis dashboard.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            title (str): Dashboard title
            figsize (Tuple[float, float]): Figure size
            
        Returns:
            Tuple[Figure, Dict[str, Axes]]: Figure and dictionary of named axes
        """
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        fig.suptitle(title, fontsize=16)
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[3, 1, 1], width_ratios=[3, 1])
        
        # Create axes for different panels
        ax_ohlc = fig.add_subplot(gs[0, 0])  # Main OHLC chart
        ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_ohlc)  # Volume chart
        ax_macd = fig.add_subplot(gs[2, 0], sharex=ax_ohlc)  # MACD
        ax_rsi = fig.add_subplot(gs[0, 1])  # RSI
        ax_atr = fig.add_subplot(gs[1, 1])  # ATR
        ax_stats = fig.add_subplot(gs[2, 1])  # Statistics text
        
        axes_dict = {
            'ohlc': ax_ohlc,
            'volume': ax_volume,
            'macd': ax_macd,
            'rsi': ax_rsi,
            'atr': ax_atr,
            'stats': ax_stats
        }
        
        # 1. Plot OHLC with Bollinger Bands
        self.ohlc_chart.plot_candlestick(df, ax=ax_ohlc, title="Price Chart")
        
        # Add Bollinger Bands overlay
        df_bb = df.copy()
        if not all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            from ..analysis.volatility import bollinger_bands
            df_bb = bollinger_bands(df)
        
        ax_ohlc.plot(df['timestamp'], df_bb['bb_upper'], 'r--', alpha=0.7, label='Upper Band')
        ax_ohlc.plot(df['timestamp'], df_bb['bb_middle'], 'b-', alpha=0.7, label='Middle Band')
        ax_ohlc.plot(df['timestamp'], df_bb['bb_lower'], 'g--', alpha=0.7, label='Lower Band')
        ax_ohlc.fill_between(df['timestamp'], df_bb['bb_upper'], df_bb['bb_lower'], color='gray', alpha=0.1)
        ax_ohlc.legend()
        
        # 2. Plot Volume
        if 'volume' in df.columns:
            colors = ['green' if close >= open else 'red' 
                     for open, close in zip(df['open'], df['close'])]
            ax_volume.bar(df['timestamp'], df['volume'], color=colors, alpha=0.6)
            ax_volume.set_title('Volume')
            ax_volume.set_ylabel('Volume')
        
        # 3. Plot MACD
        self.indicator_viz.plot_macd(df, ax=ax_macd)
        
        # 4. Plot RSI
        self.indicator_viz.plot_rsi(df, ax=ax_rsi)
        
        # 5. Plot ATR
        self.indicator_viz.plot_atr(df, ax=ax_atr)
        
        # 6. Add statistics text panel
        self._add_statistics_panel(df, ax_stats)
        
        # Hide x-axis labels for all but the bottom row
        plt.setp(ax_ohlc.get_xticklabels(), visible=False)
        plt.setp(ax_volume.get_xticklabels(), visible=False)
        
        # Format date axis on bottom panel
        self.adjust_date_format(ax_macd)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.95, hspace=0.3)
        
        return fig, axes_dict
    
    def _add_statistics_panel(self, df: pd.DataFrame, ax: Axes):
        """Add statistics text panel to dashboard.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            ax (Axes): Matplotlib axes to use
        """
        # Turn off axis
        ax.axis('off')
        
        # Calculate statistics
        current_price = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[0]
        pct_change = (price_change / df['close'].iloc[0]) * 100
        
        price_high = df['high'].max()
        price_low = df['low'].min()
        
        # Calculate volatility if not present
        if 'volatility' not in df.columns:
            from ..analysis.volatility import historical_volatility
            df_vol = historical_volatility(df)
            volatility = df_vol['volatility'].iloc[-1] * 100
        else:
            volatility = df['volatility'].iloc[-1] * 100
            
        # Calculate average volume
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
        
        # Create statistics text
        stats_text = (
            f"Current Price: ${current_price:.2f}\n"
            f"Change: {price_change:.2f} ({pct_change:.2f}%)\n\n"
            f"Period High: ${price_high:.2f}\n"
            f"Period Low: ${price_low:.2f}\n\n"
            f"Volatility: {volatility:.2f}%\n"
            f"Avg Volume: {avg_volume:.0f}"
        )
        
        # Add text to axes
        ax.text(0.5, 0.5, stats_text, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=10)
        
        # Add title
        ax.set_title('Statistics')
    
    def create_multi_asset_dashboard(self, dataframes: Dict[str, pd.DataFrame],
                                   figsize: Tuple[float, float] = (16, 12)) -> Figure:
        """Create a dashboard comparing multiple assets.
        
        Args:
            dataframes (Dict[str, pd.DataFrame]): Dict mapping asset names to DataFrames
            figsize (Tuple[float, float]): Figure size
            
        Returns:
            Figure: Figure with the dashboard
        """
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        fig.suptitle("Multi-Asset Comparison", fontsize=16)
        
        # Determine grid layout based on number of assets
        num_assets = len(dataframes)
        cols = min(2, num_assets)
        rows = (num_assets + cols - 1) // cols  # Ceiling division
        
        # Create grid
        gs = gridspec.GridSpec(rows, cols, figure=fig)
        
        # Plot each asset
        for i, (asset_name, df) in enumerate(dataframes.items()):
            row, col = divmod(i, cols)
            ax = fig.add_subplot(gs[row, col])
            
            # Plot candlestick chart
            self.ohlc_chart.plot_candlestick(df, ax=ax, title=f"{asset_name} Price")
            
            # Add 20-day SMA
            from ..analysis.trends import simple_moving_average
            df_sma = simple_moving_average(df, window_sizes=20)
            ax.plot(df['timestamp'], df_sma['sma_20'], 'r-', label='SMA(20)')
            
            # Format date axis
            self.adjust_date_format(ax)
            
            # Add legend
            ax.legend()
        
        # Add correlation heatmap if more than one asset
        if num_assets > 1:
            # Create an additional subplot for correlation heatmap
            ax_corr = fig.add_subplot(gs[-1, -1])
            
            # Calculate correlations between closing prices
            close_prices = {}
            
            # Find common date range
            min_dates = []
            max_dates = []
            for name, df in dataframes.items():
                if 'timestamp' in df.columns:
                    min_dates.append(df['timestamp'].min())
                    max_dates.append(df['timestamp'].max())
            
            if min_dates and max_dates:
                start_date = max(min_dates)
                end_date = min(max_dates)
                
                # Extract close prices in common date range
                for name, df in dataframes.items():
                    if 'timestamp' in df.columns and 'close' in df.columns:
                        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                        close_prices[name] = df.loc[mask, 'close'].values
                
                # If we have enough data points, create correlation matrix
                if all(len(prices) > 1 for prices in close_prices.values()):
                    # Convert to DataFrame
                    price_df = pd.DataFrame(close_prices)
                    
                    # Calculate correlation
                    corr_matrix = price_df.corr()
                    
                    # Plot heatmap
                    im = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax_corr)
                    
                    # Add labels
                    ax_corr.set_xticks(np.arange(len(corr_matrix.columns)))
                    ax_corr.set_yticks(np.arange(len(corr_matrix.columns)))
                    ax_corr.set_xticklabels(corr_matrix.columns)
                    ax_corr.set_yticklabels(corr_matrix.columns)
                    
                    # Rotate x-axis labels
                    plt.setp(ax_corr.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Loop over data dimensions and create text annotations
                    for i in range(len(corr_matrix.columns)):
                        for j in range(len(corr_matrix.columns)):
                            ax_corr.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                       ha="center", va="center", color="black")
                    
                    ax_corr.set_title("Price Correlation")
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.95)
        
        return fig