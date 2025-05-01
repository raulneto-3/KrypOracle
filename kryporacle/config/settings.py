import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH", str(BASE_DIR / "data" / "storage"))
BACKTESTS_PATH = os.getenv("BACKTESTS_PATH", str(BASE_DIR / "data" / "backtests"))

# Exchange configuration
EXCHANGE_CONFIGS = {
    "binance": {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_API_SECRET", ""),
        "base_url": "https://api.binance.com",
        "enabled": True,
    },
    "coinbase": {
        "api_key": os.getenv("COINBASE_API_KEY", ""),
        "api_secret": os.getenv("COINBASE_API_SECRET", ""),
        "base_url": "https://api.exchange.coinbase.com",
        "enabled": False,  # Default to disabled until configured
    },
}

# Default timeframes for data collection
DEFAULT_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Default cryptocurrency symbols to track
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]