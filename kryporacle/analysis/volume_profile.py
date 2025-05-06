import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class VolumeProfileAnalyzer:
    """
    Analyzer for Volume Profile techniques.
    
    Provides methods to calculate and analyze volume distribution across price levels,
    identify value areas, and detect volume-price divergences.
    """
    
    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, 
                              n_bins: int = 10, 
                              price_col: str = 'close', 
                              volume_col: str = 'volume',
                              window: int = None) -> pd.DataFrame:
        """
        Calculate volume distribution across price levels.
        
        Args:
            df (pd.DataFrame): DataFrame with price and volume data
            n_bins (int): Number of price bins to divide the range into
            price_col (str): Column name for price data
            volume_col (str): Column name for volume data
            window (int): Optional rolling window to calculate for recent periods only
            
        Returns:
            pd.DataFrame: DataFrame with volume profile data
        """
        result = df.copy()
        
        if window is not None:
            # Use rolling window for recent data only
            data = result.iloc[-window:]
        else:
            data = result
        
        if volume_col not in data.columns:
            logger.warning(f"Volume column '{volume_col}' not found in DataFrame")
            return result
        
        if price_col not in data.columns:
            logger.warning(f"Price column '{price_col}' not found in DataFrame")
            return result
        
        # Calculate price range and bin edges
        price_min = data[price_col].min()
        price_max = data[price_col].max()
        bin_edges = np.linspace(price_min, price_max, n_bins + 1)
        
        # Assign each price to a bin
        bin_indices = np.digitize(data[price_col], bin_edges)
        
        # Calculate volume per bin
        volume_profile = {}
        
        for i in range(1, n_bins + 1):  # np.digitize bins start at 1
            # Get volume for this price bin
            bin_volume = data[bin_indices == i][volume_col].sum()
            # Calculate price level for this bin (mid-point)
            price_level = (bin_edges[i-1] + bin_edges[i]) / 2
            volume_profile[price_level] = bin_volume
        
        # Store results in new DataFrame columns
        vp_levels = np.array(list(volume_profile.keys()))
        vp_volumes = np.array(list(volume_profile.values()))
        
        # Identify high volume nodes
        mean_volume = np.mean(vp_volumes)
        high_volume_threshold = mean_volume * 1.5
        value_area_mask = vp_volumes >= high_volume_threshold
        
        # Save value areas
        value_areas = vp_levels[value_area_mask]
        
        # Store in original dataframe
        result['vp_price_levels'] = [vp_levels] * len(result)
        result['vp_volumes'] = [vp_volumes] * len(result)
        result['vp_value_areas'] = [value_areas] * len(result)
        
        return result
    
    @staticmethod
    def identify_value_areas(df: pd.DataFrame, 
                            threshold_pct: float = 70,
                            price_level_col: str = 'vp_price_levels',
                            volume_col: str = 'vp_volumes') -> pd.DataFrame:
        """
        Identify value areas (price zones with high trading volume).
        
        Args:
            df (pd.DataFrame): DataFrame with volume profile data
            threshold_pct (float): Percentage of total volume to include in value area
            price_level_col (str): Column name for price levels
            volume_col (str): Column name for volume values
            
        Returns:
            pd.DataFrame: DataFrame with value area data
        """
        result = df.copy()
        
        if price_level_col not in result.columns or volume_col not in result.columns:
            logger.warning(f"Required columns not found: {price_level_col}, {volume_col}")
            return result
        
        # Use the last row for the calculation (which contains the arrays)
        price_levels = result.iloc[-1][price_level_col]
        volumes = result.iloc[-1][volume_col]
        
        # Create a list of (price, volume) tuples and sort by volume (descending)
        price_volume_pairs = sorted(zip(price_levels, volumes), key=lambda x: x[1], reverse=True)
        
        total_volume = sum(volumes)
        threshold_volume = total_volume * threshold_pct / 100
        
        # Collect price levels until threshold is reached
        cumulative_volume = 0
        value_area_prices = []
        
        for price, volume in price_volume_pairs:
            cumulative_volume += volume
            value_area_prices.append(price)
            if cumulative_volume >= threshold_volume:
                break
        
        # Calculate value area high and low
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        # Store in DataFrame
        result['vah'] = value_area_high
        result['val'] = value_area_low
        result['poc'] = price_volume_pairs[0][0]  # Point of Control (highest volume price)
        
        return result
    
    @staticmethod
    def detect_volume_divergence(df: pd.DataFrame,
                               price_col: str = 'close',
                               volume_col: str = 'volume',
                               window: int = 14) -> pd.DataFrame:
        """
        Detect divergences between price and volume.
        
        Args:
            df (pd.DataFrame): DataFrame with price and volume data
            price_col (str): Column name for price data
            volume_col (str): Column name for volume data
            window (int): Window size for divergence detection
            
        Returns:
            pd.DataFrame: DataFrame with divergence indicators
        """
        result = df.copy()
        
        if price_col not in result.columns or volume_col not in result.columns:
            logger.warning(f"Required columns not found: {price_col}, {volume_col}")
            return result
        
        # Calculate price and volume changes
        result['price_change'] = result[price_col].pct_change()
        result['volume_change'] = result[volume_col].pct_change()
        
        # Calculate rolling correlation between price and volume changes
        result['price_volume_corr'] = result['price_change'].rolling(window=window).corr(
            result['volume_change']
        )
        
        # Detect divergences
        result['bullish_divergence'] = False
        result['bearish_divergence'] = False
        
        # Bullish divergence: price making lower lows but volume not confirming
        for i in range(window, len(result)):
            if (result[price_col].iloc[i] < result[price_col].iloc[i-window:i].min() and
                result[volume_col].iloc[i] > result[volume_col].iloc[i-window:i].min()):
                result.loc[result.index[i], 'bullish_divergence'] = True
        
        # Bearish divergence: price making higher highs but volume not confirming
        for i in range(window, len(result)):
            if (result[price_col].iloc[i] > result[price_col].iloc[i-window:i].max() and
                result[volume_col].iloc[i] < result[volume_col].iloc[i-window:i].max()):
                result.loc[result.index[i], 'bearish_divergence'] = True
        
        return result
    
    @staticmethod
    def identify_accumulation_distribution(df: pd.DataFrame,
                                        price_col: str = 'close',
                                        volume_col: str = 'volume',
                                        window: int = 20) -> pd.DataFrame:
        """
        Identify accumulation and distribution zones using volume and price.
        
        Args:
            df (pd.DataFrame): DataFrame with price and volume data
            price_col (str): Column name for price data
            volume_col (str): Column name for volume data
            window (int): Window size for calculation
            
        Returns:
            pd.DataFrame: DataFrame with accumulation/distribution indicators
        """
        result = df.copy()
        
        if price_col not in result.columns or volume_col not in result.columns:
            logger.warning(f"Required columns not found: {price_col}, {volume_col}")
            return result
        
        # Calculate typical price
        result['typical_price'] = (result['high'] + result['low'] + result[price_col]) / 3
        
        # Calculate money flow
        result['money_flow'] = result['typical_price'] * result[volume_col]
        
        # Calculate money flow volume (positive or negative based on price direction)
        result['price_direction'] = np.where(
            result['typical_price'] > result['typical_price'].shift(1), 1,
            np.where(result['typical_price'] < result['typical_price'].shift(1), -1, 0)
        )
        
        result['mf_volume'] = result['money_flow'] * result['price_direction']
        
        # Calculate accumulation/distribution line
        result['ad_line'] = result['mf_volume'].cumsum()
        
        # Calculate Chaikin Money Flow (CMF)
        result['cmf'] = result['mf_volume'].rolling(window=window).sum() / result[volume_col].rolling(window=window).sum()
        
        # Identify accumulation and distribution zones
        result['accumulation_zone'] = result['cmf'] > 0.05
        result['distribution_zone'] = result['cmf'] < -0.05
        
        return result
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame,
                     price_col: str = 'close',
                     volume_col: str = 'volume',
                     reset_period: str = None) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            df (pd.DataFrame): DataFrame with price and volume data
            price_col (str): Column name for price data
            volume_col (str): Column name for volume data
            reset_period (str): Period to reset VWAP calculation (e.g. 'D' for daily)
            
        Returns:
            pd.DataFrame: DataFrame with VWAP
        """
        result = df.copy()
        
        if price_col not in result.columns or volume_col not in result.columns:
            logger.warning(f"Required columns not found: {price_col}, {volume_col}")
            return result
        
        # Calculate typical price
        result['typical_price'] = (result['high'] + result['low'] + result[price_col]) / 3
        
        # Calculate VWAP components
        result['tp_volume'] = result['typical_price'] * result[volume_col]
        
        if reset_period is not None and 'timestamp' in result.columns:
            # Convert to datetime if not already
            result['timestamp'] = pd.to_datetime(result['timestamp'])
            
            # Group by reset period
            if reset_period == 'D':
                result['date'] = result['timestamp'].dt.date
            elif reset_period == 'W':
                result['date'] = result['timestamp'].dt.isocalendar().week
            elif reset_period == 'M':
                result['date'] = result['timestamp'].dt.month
            
            # Calculate cumulative sum for each group
            result['cum_tp_volume'] = result.groupby('date')['tp_volume'].cumsum()
            result['cum_volume'] = result.groupby('date')[volume_col].cumsum()
            
            # Calculate VWAP
            result['vwap'] = result['cum_tp_volume'] / result['cum_volume']
            
            # Clean up
            result = result.drop(columns=['date'])
        else:
            # Calculate running VWAP
            result['cum_tp_volume'] = result['tp_volume'].cumsum()
            result['cum_volume'] = result[volume_col].cumsum()
            result['vwap'] = result['cum_tp_volume'] / result['cum_volume']
        
        # Clean up intermediate columns
        result = result.drop(columns=['typical_price', 'tp_volume', 'cum_tp_volume', 'cum_volume'])
        
        return result