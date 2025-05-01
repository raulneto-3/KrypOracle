import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)

def calculate_metrics(df: pd.DataFrame, risk_free_rate: float = 0.0, timeframe: str = '1d') -> Dict:
    """Calculate performance metrics for a backtest.
    
    Args:
        df (pd.DataFrame): DataFrame with backtest results
        risk_free_rate (float): Risk-free rate used for Sharpe ratio calculation
        timeframe (str): Timeframe of the data (e.g., '5m', '1h', '1d')
        
    Returns:
        Dict: Dictionary containing performance metrics
    """
    # Validate input
    required_cols = ['portfolio_value', 'daily_returns', 'drawdown']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns for metrics calculation: {[col for col in required_cols if col not in df.columns]}")
        return {}
    
    # Initial and final portfolio value
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    
    # Total return
    total_return = (final_value / initial_value) - 1
    
    # Determine periods per year based on timeframe
    periods_per_year = get_periods_per_year(timeframe)
    
    # Annualized return calculation based on timeframe
    if pd.api.types.is_datetime64_dtype(df.index):
        # Se o índice é datetime
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            # Anualização baseada em dias corridos
            annual_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            # Se período muito curto, usar a quantidade de períodos
            num_periods = len(df)
            if num_periods > 1:
                annual_return = ((1 + total_return) ** (periods_per_year / num_periods)) - 1
            else:
                annual_return = 0
            logger.warning(f"Very short period (<1 day). Using period count for annualization with {periods_per_year} periods per year.")
    elif 'timestamp' in df.columns and pd.api.types.is_datetime64_dtype(df['timestamp']):
        # Se há uma coluna timestamp, usamos ela
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        if days > 0:
            annual_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            # Se período muito curto, usar a quantidade de períodos
            num_periods = len(df)
            if num_periods > 1:
                annual_return = ((1 + total_return) ** (periods_per_year / num_periods)) - 1
            else:
                annual_return = 0
            logger.warning(f"Very short period (<1 day). Using period count for annualization with {periods_per_year} periods per year.")
    else:
        # Se não temos informações de data, usamos o número de períodos
        num_periods = len(df)
        if num_periods > 1:
            annual_return = ((1 + total_return) ** (periods_per_year / num_periods)) - 1
        else:
            annual_return = 0
        logger.warning(f"No datetime index found. Using number of periods with {periods_per_year} periods per year.")
    
    # Volatility (annualized standard deviation of returns)
    volatility = df['daily_returns'].std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    if volatility > 0:
        sharpe_ratio = (annual_return - risk_free_rate) / volatility
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    max_drawdown = df['drawdown'].min()
    
    # Win rate
    if 'trade_count' in df.columns and df['trade_count'].iloc[-1] > 0:
        total_trades = df['trade_count'].iloc[-1]
        profitable_trades = df[df['daily_returns'] > 0].shape[0]
        win_rate = profitable_trades / total_trades
    else:
        total_trades = 0
        win_rate = 0
    
    # Return metrics
    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate
    }

def get_periods_per_year(timeframe: str) -> int:
    """Determine how many periods of the given timeframe are in a trading year.
    
    Args:
        timeframe (str): Timeframe string (e.g., '5m', '1h', '1d')
        
    Returns:
        int: Number of periods per year
    """
    # Parse the timeframe string to extract number and unit
    match = re.match(r'(\d+)([mhdwMy])', timeframe)
    if not match:
        logger.warning(f"Could not parse timeframe '{timeframe}'. Using daily (365) as default.")
        return 365
    
    number, unit = int(match.group(1)), match.group(2)
    
    # Calculate periods per year based on the unit
    if unit == 'm':  # Minutes
        # Aproximadamente 365 dias de negociação × 6.5 horas por dia × 60 minutos por hora
        minutes_per_year = 365 * 6.5 * 60
        return int(minutes_per_year / number)
    elif unit == 'h':  # Hours
        # Aproximadamente 365 dias de negociação × 6.5 horas por dia
        hours_per_year = 365 * 6.5
        return int(hours_per_year / number)
    elif unit == 'd':  # Days
        # Aproximadamente 365 dias de negociação por ano
        return int(365 / number)
    elif unit == 'w':  # Weeks
        # Aproximadamente 52 semanas por ano
        return int(52 / number)
    elif unit == 'M':  # Months
        # 12 meses por ano
        return int(12 / number)
    elif unit == 'y':  # Years
        # 1 ano por ano
        return int(1 / number)
    else:
        logger.warning(f"Unknown timeframe unit '{unit}'. Using daily (365) as default.")
        return 365