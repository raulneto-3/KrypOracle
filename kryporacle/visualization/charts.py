import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ChartBase:
    """Base class for all chart types."""
    
    def __init__(self, figsize: Tuple[float, float] = (12, 8), 
                style: str = 'seaborn-v0_8-darkgrid', 
                dpi: int = 100):
        """Initialize the chart with figure settings.
        
        Args:
            figsize (Tuple[float, float]): Figure width and height in inches
            style (str): Matplotlib style
            dpi (int): Dots per inch resolution
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        
        # Set the style
        plt.style.use(self.style)
        
    def create_figure(self, rows: int = 1, cols: int = 1, 
                     sharex: bool = True) -> Tuple[Figure, List[Axes]]:
        """Create a new matplotlib figure and axes.
        
        Args:
            rows (int): Number of subplot rows
            cols (int): Number of subplot columns
            sharex (bool): Whether to share the x-axis among subplots
            
        Returns:
            Tuple[Figure, List[Axes]]: Figure and list of axes
        """
        fig, axes = plt.subplots(
            rows, cols, 
            figsize=self.figsize, 
            sharex=sharex, 
            dpi=self.dpi
        )
        
        # If there's only one subplot, wrap axes in a list for consistent handling
        if rows * cols == 1:
            axes = [axes]
        elif rows > 1:
            # If we have multiple rows, flatten the axes array for easier indexing
            axes = axes.flatten()
            
        return fig, axes
    
    def save_figure(self, fig: Figure, filename: str, directory: Optional[str] = None) -> str:
        """Save the figure to a file.
        
        Args:
            fig (Figure): The figure to save
            filename (str): Filename without extension
            directory (str, optional): Directory to save the file
            
        Returns:
            str: Path to the saved file
        """
        # Ensure filename has an extension
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            filename = f"{filename}.png"
            
        if directory:
            # Create directory if it doesn't exist
            save_dir = Path(directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / filename
        else:
            file_path = Path(filename)
            
        fig.savefig(file_path, bbox_inches='tight')
        logger.info(f"Figure saved to {file_path}")
        
        return str(file_path)
    
    def adjust_date_format(self, ax: Axes, date_format: str = '%Y-%m-%d'):
        """Adjust the date formatting on the x-axis.
        
        Args:
            ax (Axes): The axes to adjust
            date_format (str): Date format string
        """
        # Format the date on the x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        
        # Rotate and align the tick labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add more space at the bottom
        plt.subplots_adjust(bottom=0.2)


class OHLCChart(ChartBase):
    """Chart class for OHLC (Open, High, Low, Close) price data."""
    
    def plot_candlestick(self, df: pd.DataFrame, ax: Optional[Axes] = None, 
                        title: str = "Candlestick Chart") -> Tuple[Figure, Axes]:
        """Create a candlestick chart from OHLC data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            ax (Axes, optional): Matplotlib axes to plot on
            title (str): Chart title
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the candlestick chart
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
        
        # Ensure timestamp is datetime type
        df = df.copy()
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate width of candlestick body
        # This ensures candlesticks look good regardless of data density
        data_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        num_points = len(df)
        width = 0.7 * data_range / (num_points * 24 * 3600)  # 70% of the average time between points
            
        # Plot candlesticks
        for i, row in df.iterrows():
            # Candlestick body
            if row['close'] >= row['open']:
                # Bullish (green) candle
                color = 'green'
                bottom = row['open']
                height = row['close'] - row['open']
            else:
                # Bearish (red) candle
                color = 'red'
                bottom = row['close']
                height = row['open'] - row['close']
            
            # Plot candle body (rectangle)
            rect = plt.Rectangle(
                (mdates.date2num(row['timestamp']) - width/2, bottom),
                width, height, 
                edgecolor='black',
                facecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Plot high and low wicks
            ax.plot(
                [mdates.date2num(row['timestamp']), mdates.date2num(row['timestamp'])],
                [row['low'], bottom],
                color='black', linewidth=1
            )
            ax.plot(
                [mdates.date2num(row['timestamp']), mdates.date2num(row['timestamp'])],
                [bottom + height, row['high']],
                color='black', linewidth=1
            )
        
        # Set title and labels
        ax.set_title(title)
        ax.set_ylabel('Price')
        
        # Format date axis
        self.adjust_date_format(ax)
        
        # Auto-adjust y-axis limits
        buffer = 0.05 * (df['high'].max() - df['low'].min())
        ax.set_ylim(df['low'].min() - buffer, df['high'].max() + buffer)
        
        # Set x-axis limits
        date_range = pd.date_range(df['timestamp'].min(), df['timestamp'].max())
        ax.set_xlim(date_range[0], date_range[-1])
        
        return fig, ax

    def plot_ohlc(self, df: pd.DataFrame, ax: Optional[Axes] = None,
                 title: str = "OHLC Chart", volume: bool = True) -> Tuple[Figure, Axes]:
        """Create an OHLC chart with optional volume panel.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data and timestamp column
            ax (Axes, optional): Matplotlib axes to plot on
            title (str): Chart title
            volume (bool): Whether to include volume panel
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the OHLC chart
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        if volume and 'volume' not in df.columns:
            logger.warning("Volume column not found. Volume panel will not be shown.")
            volume = False
        
        # Create figure with GridSpec for price and volume panels
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create GridSpec with optional volume panel
        if volume:
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
            ax_price = fig.add_subplot(gs[0])
            ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
            axes = [ax_price, ax_volume]
        else:
            gs = gridspec.GridSpec(1, 1, figure=fig)
            ax_price = fig.add_subplot(gs[0])
            axes = [ax_price]
        
        # Plot candlestick chart on price panel
        self.plot_candlestick(df, ax=ax_price, title=title)
        
        # Plot volume if requested
        if volume:
            # Normalize timestamp for plotting
            timestamps = mdates.date2num(df['timestamp'])
            
            # Calculate width for volume bars
            width = 0.7 * (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else 0.7
            
            # Color volume bars based on price direction
            colors = ['green' if close >= open else 'red' 
                     for open, close in zip(df['open'], df['close'])]
            
            # Plot volume bars
            ax_volume.bar(timestamps, df['volume'], width=width, color=colors, alpha=0.5)
            ax_volume.set_ylabel('Volume')
            
            # Format date axis
            self.adjust_date_format(ax_volume)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        return fig, axes

    def plot_line(self, df: pd.DataFrame, y_column: str, ax: Optional[Axes] = None,
                 title: str = None, color: str = 'blue', label: Optional[str] = None) -> Tuple[Figure, Axes]:
        """Create a line chart.
        
        Args:
            df (pd.DataFrame): DataFrame with time series data and timestamp column
            y_column (str): Column name to plot on y-axis
            ax (Axes, optional): Matplotlib axes to plot on
            title (str, optional): Chart title
            color (str): Line color
            label (str, optional): Line label for legend
            
        Returns:
            Tuple[Figure, Axes]: Figure and axes with the line chart
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        if y_column not in df.columns:
            raise ValueError(f"Column '{y_column}' not found in DataFrame")
        
        # Create figure and axes if not provided
        if ax is None:
            fig, axes = self.create_figure()
            ax = axes[0]
        else:
            fig = ax.figure
        
        # Ensure timestamp is datetime type
        df_plot = df.copy()
        if not pd.api.types.is_datetime64_dtype(df_plot['timestamp']):
            df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
        
        # Plot the line
        ax.plot(df_plot['timestamp'], df_plot[y_column], color=color, label=label or y_column)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        ax.set_ylabel(y_column)
        
        # Add legend if label is provided
        if label:
            ax.legend()
        
        # Format date axis
        self.adjust_date_format(ax)
        
        return fig, ax