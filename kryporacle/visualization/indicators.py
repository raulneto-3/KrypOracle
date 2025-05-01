import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

from .charts import ChartBase, OHLCChart
from ..analysis import trends, volatility

logger = logging.getLogger(__name__)

class IndicatorVisualizer(ChartBase):
    """Visualizer for technical indicators."""
    
    def __init__(self, **kwargs):
        """Initialize the indicator visualizer."""
        super().__init__(**kwargs)
        self.ohlc_chart = OHLCChart(**kwargs)
    
    def plot_with_sma(self, df: pd.DataFrame, sma_periods: List[int] = [20, 50, 200],
                      ax: Optional[Axes] = None, title: str = "Price with SMA") -> Tuple[Figure, Axes]:
        """Plot price with Simple Moving Averages.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            sma_periods (List[int]): List of SMA periods to plot
            ax (Axes, optional): Matplotlib axes to plot on
            title (str): Chart title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the chart
        """
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
            
        # Calculate SMAs if needed
        df_with_sma = trends.simple_moving_average(df, window_sizes=sma_periods)
        
        # Plot price
        ax.plot(df['timestamp'], df['close'], label='Close', color='black', alpha=0.75)
        
        # Plot SMAs
        colors = ['red', 'blue', 'green', 'purple', 'orange']  # Cycle through these colors
        for i, period in enumerate(sma_periods):
            col_name = f'sma_{period}'
            if col_name in df_with_sma.columns:
                color_idx = i % len(colors)
                ax.plot(df['timestamp'], df_with_sma[col_name], 
                       label=f'SMA({period})', 
                       color=colors[color_idx])
        
        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.legend()
        
        # Format date axis
        self.adjust_date_format(ax)
        
        return fig, ax
    
    def plot_with_ema(self, df: pd.DataFrame, ema_periods: List[int] = [9, 20, 50],
                     ax: Optional[Axes] = None, title: str = "Price with EMA") -> Tuple[Figure, Axes]:
        """Plot price with Exponential Moving Averages.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            ema_periods (List[int]): List of EMA periods to plot
            ax (Axes, optional): Matplotlib axes to plot on
            title (str): Chart title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the chart
        """
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
            
        # Calculate EMAs if needed
        df_with_ema = trends.exponential_moving_average(df, window_sizes=ema_periods)
        
        # Plot price
        ax.plot(df['timestamp'], df['close'], label='Close', color='black', alpha=0.75)
        
        # Plot EMAs
        colors = ['red', 'blue', 'green', 'purple', 'orange']  # Cycle through these colors
        for i, period in enumerate(ema_periods):
            col_name = f'ema_{period}'
            if col_name in df_with_ema.columns:
                color_idx = i % len(colors)
                ax.plot(df['timestamp'], df_with_ema[col_name], 
                       label=f'EMA({period})', 
                       color=colors[color_idx],
                       linestyle='--')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.legend()
        
        # Format date axis
        self.adjust_date_format(ax)
        
        return fig, ax
    
    def plot_macd(self, df: pd.DataFrame, ax: Optional[Axes] = None, 
                 fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 title: str = "MACD Indicator") -> Tuple[Figure, Axes]:
        """Plot MACD indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            ax (Axes, optional): Matplotlib axes to plot on
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            title (str): Chart title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the chart
        """
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
            
        # Calculate MACD if needed
        if not all(col in df.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
            df_macd = trends.macd(df, fast_period=fast_period, 
                                 slow_period=slow_period, 
                                 signal_period=signal_period)
        else:
            df_macd = df
        
        # Plot MACD line and signal line
        ax.plot(df['timestamp'], df_macd['macd_line'], label='MACD Line', color='blue')
        ax.plot(df['timestamp'], df_macd['macd_signal'], label='Signal Line', color='red')
        
        # Plot MACD histogram as bar chart
        bar_width = 0.7 * (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds() / (24 * 3600)
        
        # Color bars based on direction (above/below zero)
        for i, (timestamp, value) in enumerate(zip(df['timestamp'], df_macd['macd_histogram'])):
            color = 'green' if value >= 0 else 'red'
            ax.bar(timestamp, value, width=bar_width, color=color, alpha=0.5)
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel('MACD Value')
        ax.legend()
        
        # Format date axis
        self.adjust_date_format(ax)
        
        return fig, ax
    
    def plot_rsi(self, df: pd.DataFrame, ax: Optional[Axes] = None, 
               period: int = 14, title: str = "RSI Indicator",
               overbought: int = 70, oversold: int = 30) -> Tuple[Figure, Axes]:
        """Plot RSI indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            ax (Axes, optional): Matplotlib axes to plot on
            period (int): RSI calculation period
            title (str): Chart title
            overbought (int): Overbought level
            oversold (int): Oversold level
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the chart
        """
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
            
        # Calculate RSI if needed
        if 'rsi' not in df.columns:
            df_rsi = trends.rsi(df, window=period)
        else:
            df_rsi = df
        
        # Plot RSI line
        ax.plot(df['timestamp'], df_rsi['rsi'], label='RSI', color='purple')
        
        # Add overbought and oversold lines
        ax.axhline(y=overbought, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=oversold, color='green', linestyle='--', alpha=0.5)
        
        # Add center line
        ax.axhline(y=50, color='black', linestyle='-', alpha=0.2)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel('RSI Value')
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Add legend
        ax.legend()
        
        # Format date axis
        self.adjust_date_format(ax)
        
        return fig, ax
    
    def plot_bollinger_bands(self, df: pd.DataFrame, ax: Optional[Axes] = None,
                           window: int = 20, std_dev: float = 2.0,
                           title: str = "Bollinger Bands") -> Tuple[Figure, Axes]:
        """Plot Bollinger Bands.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            ax (Axes, optional): Matplotlib axes to plot on
            window (int): Window size for moving average
            std_dev (float): Number of standard deviations for bands
            title (str): Chart title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the chart
        """
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
            
        # Calculate Bollinger Bands if needed
        band_cols = ['bb_upper', 'bb_middle', 'bb_lower']
        if not all(col in df.columns for col in band_cols):
            df_bb = volatility.bollinger_bands(df, window=window, std_dev=std_dev)
        else:
            df_bb = df
        
        # Plot price
        ax.plot(df['timestamp'], df['close'], label='Close', color='black', alpha=0.75)
        
        # Plot Bollinger Bands
        ax.plot(df['timestamp'], df_bb['bb_upper'], label=f'Upper Band ({std_dev}σ)', 
               color='red', linestyle='--', alpha=0.7)
        ax.plot(df['timestamp'], df_bb['bb_middle'], label='Middle Band (SMA)', 
               color='blue', alpha=0.7)
        ax.plot(df['timestamp'], df_bb['bb_lower'], label=f'Lower Band ({std_dev}σ)', 
               color='green', linestyle='--', alpha=0.7)
        
        # Fill between bands
        ax.fill_between(df['timestamp'], df_bb['bb_upper'], df_bb['bb_lower'], 
                       color='gray', alpha=0.1)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.legend()
        
        # Format date axis
        self.adjust_date_format(ax)
        
        return fig, ax
    
    def plot_atr(self, df: pd.DataFrame, ax: Optional[Axes] = None,
                window: int = 14, title: str = "Average True Range") -> Tuple[Figure, Axes]:
        """Plot Average True Range (ATR) indicator.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            ax (Axes, optional): Matplotlib axes to plot on
            window (int): ATR calculation period
            title (str): Chart title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the chart
        """
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
            
        # Calculate ATR if needed
        if 'atr' not in df.columns:
            df_atr = volatility.average_true_range(df, window=window)
        else:
            df_atr = df
        
        # Plot ATR
        ax.plot(df['timestamp'], df_atr['atr'], label='ATR', color='purple')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel('ATR Value')
        ax.legend()
        
        # Format date axis
        self.adjust_date_format(ax)
        
        return fig, ax
    
    def plot_multi_indicator_chart(self, df: pd.DataFrame, 
                                 indicators: List[str] = ['candlestick', 'sma', 'volume', 'macd', 'rsi'],
                                 figsize: Optional[Tuple[float, float]] = None,
                                 **kwargs) -> Tuple[Figure, List[Axes]]:
        """Create a multi-panel chart with different indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            indicators (List[str]): List of indicators to include
                Options: 'candlestick', 'sma', 'ema', 'volume', 'macd', 'rsi', 'bollinger', 'atr'
            figsize (Tuple[float, float]): Figure size override
            **kwargs: Additional arguments for individual indicator functions
            
        Returns:
            Tuple[Figure, List[Axes]]: Figure and list of axes with the charts
        """
        # Determine the number of panels needed
        # Some indicators can share panels, others need their own
        panel_mapping = {
            'candlestick': 0,  # Main price panel
            'sma': 0,          # Overlaid on price
            'ema': 0,          # Overlaid on price
            'bollinger': 0,    # Overlaid on price
            'volume': 1,       # Volume panel
            'macd': 2,         # MACD panel
            'rsi': 3,          # RSI panel
            'atr': 4           # ATR panel
        }
        
        # Filter out invalid indicators
        valid_indicators = [ind for ind in indicators if ind in panel_mapping]
        if not valid_indicators:
            raise ValueError(f"No valid indicators specified. Options are: {list(panel_mapping.keys())}")
        
        # Determine required panels
        required_panels = set(panel_mapping[ind] for ind in valid_indicators)
        num_panels = max(required_panels) + 1
        
        # Set panel heights (main price panel is larger)
        height_ratios = [3 if i == 0 else 1 for i in range(num_panels)]
        
        # Create figure with GridSpec
        if figsize is None:
            figsize = (self.figsize[0], self.figsize[1] * num_panels / 3)
            
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = plt.GridSpec(num_panels, 1, height_ratios=height_ratios, figure=fig)
        
        # Create axes for each panel
        axes = [fig.add_subplot(gs[i]) for i in range(num_panels)]
        
        # Share x-axis for all panels
        for i in range(1, num_panels):
            axes[i].sharex(axes[0])
        
        # Plot indicators on appropriate panels
        for indicator in valid_indicators:
            panel_idx = panel_mapping[indicator]
            ax = axes[panel_idx]
            
            if indicator == 'candlestick':
                self.ohlc_chart.plot_candlestick(df, ax=ax, **kwargs)
                
            elif indicator == 'sma':
                sma_periods = kwargs.get('sma_periods', [20, 50, 200])
                df_with_sma = trends.simple_moving_average(df, window_sizes=sma_periods)
                
                for i, period in enumerate(sma_periods):
                    col_name = f'sma_{period}'
                    if col_name in df_with_sma.columns:
                        color = ['red', 'blue', 'green'][i % 3]
                        ax.plot(df['timestamp'], df_with_sma[col_name],
                               label=f'SMA({period})',
                               color=color)
                ax.legend()
                
            elif indicator == 'ema':
                ema_periods = kwargs.get('ema_periods', [9, 20, 50])
                df_with_ema = trends.exponential_moving_average(df, window_sizes=ema_periods)
                
                for i, period in enumerate(ema_periods):
                    col_name = f'ema_{period}'
                    if col_name in df_with_ema.columns:
                        color = ['orange', 'purple', 'brown'][i % 3]
                        ax.plot(df['timestamp'], df_with_ema[col_name],
                               label=f'EMA({period})',
                               color=color,
                               linestyle='--')
                ax.legend()
                
            elif indicator == 'bollinger':
                window = kwargs.get('bb_window', 20)
                std_dev = kwargs.get('bb_std_dev', 2.0)
                
                # Calculate Bollinger Bands
                df_bb = volatility.bollinger_bands(df, window=window, std_dev=std_dev)
                
                # Plot Bollinger Bands
                ax.plot(df['timestamp'], df_bb['bb_upper'], 
                       label=f'Upper Band ({std_dev}σ)', 
                       color='red', linestyle='--')
                ax.plot(df['timestamp'], df_bb['bb_middle'], 
                       label='Middle Band', 
                       color='blue')
                ax.plot(df['timestamp'], df_bb['bb_lower'], 
                       label=f'Lower Band ({std_dev}σ)', 
                       color='green', linestyle='--')
                
                # Fill between bands
                ax.fill_between(df['timestamp'], df_bb['bb_upper'], df_bb['bb_lower'],
                               color='gray', alpha=0.1)
                ax.legend()
                
            elif indicator == 'volume':
                if 'volume' in df.columns:
                    # Color volume bars based on price direction
                    colors = ['green' if close >= open else 'red' 
                             for open, close in zip(df['open'], df['close'])]
                    
                    ax.bar(df['timestamp'], df['volume'], width=0.7, color=colors, alpha=0.5)
                    ax.set_ylabel('Volume')
                    ax.set_title('Trading Volume')
                    
            elif indicator == 'macd':
                fast = kwargs.get('macd_fast', 12)
                slow = kwargs.get('macd_slow', 26)
                signal = kwargs.get('macd_signal', 9)
                
                # Calculate MACD
                df_macd = trends.macd(df, fast_period=fast, slow_period=slow, signal_period=signal)
                
                # Plot MACD line and signal
                ax.plot(df['timestamp'], df_macd['macd_line'], label='MACD Line', color='blue')
                ax.plot(df['timestamp'], df_macd['macd_signal'], label='Signal Line', color='red')
                
                # Plot histogram as bars
                for i, (time, value) in enumerate(zip(df['timestamp'], df_macd['macd_histogram'])):
                    color = 'green' if value >= 0 else 'red'
                    ax.bar(time, value, width=0.7, color=color, alpha=0.5)
                    
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title('MACD Indicator')
                ax.legend()
                
            elif indicator == 'rsi':
                period = kwargs.get('rsi_period', 14)
                
                # Calculate RSI
                df_rsi = trends.rsi(df, window=period)
                
                # Plot RSI
                ax.plot(df['timestamp'], df_rsi['rsi'], color='purple')
                ax.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                ax.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                ax.axhline(y=50, color='black', linestyle='-', alpha=0.2)
                
                ax.set_ylim(0, 100)
                ax.set_title(f'RSI({period})')
                
            elif indicator == 'atr':
                period = kwargs.get('atr_period', 14)
                
                # Calculate ATR
                df_atr = volatility.average_true_range(df, window=period)
                
                # Plot ATR
                ax.plot(df['timestamp'], df_atr['atr'], color='purple')
                ax.set_title(f'ATR({period})')
        
        # Format date axis on all subplots
        for ax in axes:
            self.adjust_date_format(ax)
            
        # Hide x-axis labels for all but the bottom panel
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
            
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        
        return fig, axes