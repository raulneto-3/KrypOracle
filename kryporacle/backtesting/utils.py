import os
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

def get_backtest_dir() -> str:
    """Get or create the backtest output directory from environment variable.
    
    Returns:
        str: Path to the backtest directory
    """
    # Get path from environment variable or config setting
    from ..config.settings import BACKTESTS_PATH
    backtest_dir = Path(BACKTESTS_PATH)
    
    # Create directory if it doesn't exist
    if not backtest_dir.exists():
        try:
            backtest_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created backtest directory at {backtest_dir}")
        except Exception as e:
            logger.error(f"Error creating backtest directory: {e}")
    
    return str(backtest_dir)

def get_backtest_path(filename: str) -> str:
    """Get full path for a backtest output file.
    
    Args:
        filename (str): Name of the file
        
    Returns:
        str: Full path to the file in the backtest directory
    """
    backtest_dir = get_backtest_dir()
    return os.path.join(backtest_dir, filename)

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid filename characters
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)