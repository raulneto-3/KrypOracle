import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Any

logger = logging.getLogger(__name__)

class Strategy:
    """Base class for all trading strategies.
    
    All strategies must inherit from this class and implement the 
    generate_signals method.
    """
    
    def __init__(self, name: str = "Generic Strategy"):
        """Initialize the strategy.
        
        Args:
            name (str): Name of the strategy
        """
        self.name = name
        self._required_indicators = []
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on strategy rules.
        
        Must be implemented by all strategy subclasses.
        
        Args:
            df (pd.DataFrame): DataFrame with market data and indicators
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        raise NotImplementedError("Strategy subclasses must implement generate_signals method")
    
    @property
    def required_indicators(self) -> List[str]:
        """Get list of required indicators for this strategy.
        
        Returns:
            List[str]: List of indicator names required by this strategy
        """
        return self._required_indicators