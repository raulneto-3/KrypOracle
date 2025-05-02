import pandas as pd
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import os
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        return super().default(obj)

def generate_strategy_backtest_json(
    df: pd.DataFrame, 
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    parameters: dict,
    metrics: dict,
    filename: Optional[str] = None
) -> dict:
    """
    Generate JSON for an individual strategy backtest that can be used with Plotly
    and for AI training data.
    
    Args:
        df: DataFrame with backtest results
        strategy_name: Name of the strategy
        symbol: Trading pair symbol
        timeframe: Data timeframe
        start_time: Backtest start time
        end_time: Backtest end time
        parameters: Strategy parameters
        metrics: Performance metrics
        filename: Optional filename to save JSON
        
    Returns:
        dict: JSON-compatible dictionary
    """
    # Create structure
    result = {
        "metadata": {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": "binance",
            "start_date": start_time.isoformat(),
            "end_date": end_time.isoformat(),
            "parameters": parameters
        },
        "time_series_data": {
            "timestamps": df.index.astype(str).tolist(),
            "price": {
                "open": df["open"].tolist(),
                "high": df["high"].tolist(),
                "low": df["low"].tolist(),
                "close": df["close"].tolist()
            },
            "indicators": {},
            "trading": {
                "position": df["position"].tolist(),
                "portfolio_value": df["portfolio_value"].tolist(),
                "cash": df["cash"].tolist() if "cash" in df.columns else [],
                "holdings": df["holdings"].tolist() if "holdings" in df.columns else [],
                "drawdown": df["drawdown"].tolist()
            }
        },
        "performance_metrics": metrics
    }
    
    # Add volume if present
    if "volume" in df.columns:
        result["time_series_data"]["price"]["volume"] = df["volume"].tolist()
    
    # Add technical indicators based on what's in the DataFrame
    indicators_map = {
        "sma": ["sma_20", "sma_50", "sma_200"],
        "ema": ["ema_9", "ema_20", "ema_50"],
        "bollinger_bands": ["bb_upper", "bb_middle", "bb_lower"],
        "macd": ["macd_line", "macd_signal", "macd_hist"],
        "rsi": ["rsi"]
    }
    
    # Process indicators
    for indicator_group, columns in indicators_map.items():
        available_columns = [col for col in columns if col in df.columns]
        
        if available_columns:
            if indicator_group == "bollinger_bands":
                result["time_series_data"]["indicators"]["bollinger_bands"] = {
                    "upper": df["bb_upper"].tolist() if "bb_upper" in df.columns else [],
                    "middle": df["bb_middle"].tolist() if "bb_middle" in df.columns else [],
                    "lower": df["bb_lower"].tolist() if "bb_lower" in df.columns else []
                }
            elif indicator_group == "macd":
                result["time_series_data"]["indicators"]["macd"] = {
                    "line": df["macd_line"].tolist() if "macd_line" in df.columns else [],
                    "signal": df["macd_signal"].tolist() if "macd_signal" in df.columns else [],
                    "histogram": df["macd_hist"].tolist() if "macd_hist" in df.columns else []
                }
            else:
                for col in available_columns:
                    result["time_series_data"]["indicators"][col] = df[col].tolist()
    
    # Trade signals
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]
    
    if not buy_signals.empty:
        result["time_series_data"]["trading"]["signals_buy"] = {
            "timestamps": buy_signals.index.astype(str).tolist(),
            "prices": buy_signals["close"].tolist()
        }
        
    if not sell_signals.empty:
        result["time_series_data"]["trading"]["signals_sell"] = {
            "timestamps": sell_signals.index.astype(str).tolist(),
            "prices": sell_signals["close"].tolist()
        }
    
    # Add plotly visualization config
    result["plotly_visualization"] = create_plotly_config(df, strategy_name)
    
    # Save to file if requested
    if filename:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        print(f"Backtest JSON saved to: {filename}")
    
    return result


def generate_strategy_comparison_json(
    results: Dict[str, pd.DataFrame],
    metrics: Dict[str, dict],
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    filename: Optional[str] = None
) -> dict:
    """
    Generate a JSON for strategy comparison that can be used with Plotly.
    
    Args:
        results: Dict mapping strategy names to their dataframes
        metrics: Dict mapping strategy names to performance metrics
        symbol: Trading pair symbol
        timeframe: Data timeframe
        start_time: Backtest start time
        end_time: Backtest end time
        filename: Optional filename to save JSON
        
    Returns:
        dict: JSON-compatible dictionary
    """
    # Create structure
    strategy_names = list(results.keys())
    
    # Use the first dataframe to get timestamps
    first_df = next(iter(results.values()))
    timestamps = first_df.index.astype(str).tolist()
    
    result = {
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "exchange": "binance",
            "start_date": start_time.isoformat(),
            "end_date": end_time.isoformat(),
            "strategies": strategy_names
        },
        "time_series_data": {
            "timestamps": timestamps,
            "strategies": {}
        },
        "performance_metrics": metrics
    }
    
    # Fill in each strategy's data
    for strategy_name, df in results.items():
        result["time_series_data"]["strategies"][strategy_name] = {
            "portfolio_value": df["portfolio_value"].tolist(),
            "drawdown": df["drawdown"].tolist(),
            "position": df["position"].tolist()
        }
    
    # Create visualization config for portfolio comparison
    result["plotly_visualization"] = {
        "portfolio_chart": {
            "layout": {
                "title": "Comparação de Desempenho das Estratégias",
                "xaxis": {"title": "Data"},
                "yaxis": {"title": "Valor da Carteira (R$)"},
                "height": 600,
                "width": 1000,
                "template": "plotly_white",
                "legend": {"orientation": "h", "y": 1.1}
            },
            "data": []
        }
    }
    
    # Add each strategy to the visualization
    colors = ["blue", "red", "green", "purple", "orange", "teal", "pink", "brown", "gray", "black"]
    for i, strategy_name in enumerate(strategy_names):
        color_idx = i % len(colors)
        result["plotly_visualization"]["portfolio_chart"]["data"].append({
            "type": "scatter",
            "mode": "lines",
            "name": strategy_name,
            "x": "timestamps",
            "y": f"strategies.{strategy_name}.portfolio_value",
            "line": {"width": 2, "color": colors[color_idx]}
        })
    
    # Create heatmap
    # Extract metrics into a format suitable for heatmap
    metrics_for_heatmap = {
        'Retorno Total': [], 
        'Retorno Anual': [],
        'Índice Sharpe': [],
        'Drawdown Máximo': [],
        'Taxa de Acerto': []
    }
    
    raw_values = {col: [] for col in metrics_for_heatmap}
    
    strategy_order = []
    for strategy_name in strategy_names:
        if strategy_name in metrics:
            strategy_order.append(strategy_name)
            raw_values['Retorno Total'].append(metrics[strategy_name]['total_return'])
            raw_values['Retorno Anual'].append(metrics[strategy_name]['annual_return'])
            raw_values['Índice Sharpe'].append(metrics[strategy_name]['sharpe_ratio'])
            raw_values['Drawdown Máximo'].append(metrics[strategy_name]['max_drawdown'])
            raw_values['Taxa de Acerto'].append(metrics[strategy_name]['win_rate'])
    
    # Format the text values for display
    text_values = []
    for i, strategy in enumerate(strategy_order):
        row = []
        row.append(f"{raw_values['Retorno Total'][i]:.2%}")
        row.append(f"{raw_values['Retorno Anual'][i]:.2%}")
        row.append(f"{raw_values['Índice Sharpe'][i]:.2f}")
        row.append(f"{raw_values['Drawdown Máximo'][i]:.2%}")
        row.append(f"{raw_values['Taxa de Acerto'][i]:.2%}")
        text_values.append(row)
    
    # Normalize values for the heatmap coloring
    z_values = []
    for i in range(len(strategy_order)):
        row = []
        for j, col in enumerate(metrics_for_heatmap.keys()):
            val = raw_values[col][i]
            # Normalize between 0 and 1
            all_vals = raw_values[col]
            min_val = min(all_vals)
            max_val = max(all_vals)
            
            if max_val == min_val:
                normalized = 0.5
            else:
                normalized = (val - min_val) / (max_val - min_val)
            
            # Invert drawdown so lower is better (higher value)
            if col == 'Drawdown Máximo':
                normalized = 1 - normalized
                
            row.append(normalized)
        z_values.append(row)
    
    # Add heatmap to visualization
    result["plotly_visualization"]["heatmap_chart"] = {
        "layout": {
            "title": "Comparação de Desempenho das Estratégias (Pontuações Normalizadas)",
            "height": 600,
            "width": 900
        },
        "data": {
            "type": "heatmap",
            "z": z_values,
            "x": list(metrics_for_heatmap.keys()),
            "y": strategy_order,
            "text": text_values,
            "colorscale": "RdYlGn"
        }
    }
    
    # Save to file if requested
    if filename:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        print(f"Strategy comparison JSON saved to: {filename}")
    
    return result


def create_plotly_config(df: pd.DataFrame, strategy_name: str) -> dict:
    """Create Plotly visualization configuration for a single backtest."""
    config = {
        "layout": {
            "title": f"{strategy_name} Backtest Results",
            "height": 900,
            "width": 1200,
            "template": "plotly_dark",
            "grid": {"rows": 3, "columns": 1, "rowHeights": [0.6, 0.2, 0.2]},
            "legend": {"orientation": "h", "y": 1.02}
        },
        "charts": [
            {
                "name": "price_chart",
                "type": "candlestick",
                "row": 1,
                "col": 1,
                "data": {
                    "x": "timestamps",
                    "open": "price.open",
                    "high": "price.high",
                    "low": "price.low",
                    "close": "price.close"
                },
                "overlays": []
            },
            {
                "name": "portfolio_chart",
                "type": "scatter",
                "row": 2,
                "col": 1,
                "data": {
                    "x": "timestamps",
                    "y": "trading.portfolio_value",
                    "name": "Portfolio Value",
                    "line": {"color": "green", "width": 1.5}
                }
            },
            {
                "name": "drawdown_chart",
                "type": "scatter",
                "row": 3,
                "col": 1,
                "data": {
                    "x": "timestamps",
                    "y": "trading.drawdown",
                    "name": "Drawdown",
                    "fill": "tozeroy",
                    "line": {"color": "red"}
                }
            }
        ]
    }
    
    # Add indicators as overlays if they exist
    indicator_overlay_config = {
        "sma_20": {"name": "SMA 20", "color": "orange", "width": 1.5},
        "sma_50": {"name": "SMA 50", "color": "purple", "width": 1.5},
        "sma_200": {"name": "SMA 200", "color": "blue", "width": 1.5},
        "ema_9": {"name": "EMA 9", "color": "yellow", "width": 1.5},
        "ema_20": {"name": "EMA 20", "color": "cyan", "width": 1.5},
        "bb_upper": {"name": "BB Upper", "color": "grey", "width": 1, "dash": "dash"},
        "bb_middle": {"name": "BB Middle", "color": "grey", "width": 1},
        "bb_lower": {"name": "BB Lower", "color": "grey", "width": 1, "dash": "dash"}
    }
    
    for indicator, config_info in indicator_overlay_config.items():
        if indicator in df.columns:
            config["charts"][0]["overlays"].append({
                "type": "scatter",
                "name": config_info["name"],
                "data": f"indicators.{indicator}",
                "line": {
                    "color": config_info["color"], 
                    "width": config_info["width"],
                    "dash": config_info.get("dash", "solid")
                }
            })
    
    # Add buy/sell signals
    buy_signals = df[df["signal"] == 1]
    if not buy_signals.empty:
        config["charts"][0]["overlays"].append({
            "type": "scatter",
            "mode": "markers",
            "name": "Buy Signals",
            "data": {
                "x": "trading.signals_buy.timestamps",
                "y": "trading.signals_buy.prices"
            },
            "marker": {"color": "green", "size": 10, "symbol": "triangle-up"}
        })
    
    sell_signals = df[df["signal"] == -1]
    if not sell_signals.empty:
        config["charts"][0]["overlays"].append({
            "type": "scatter",
            "mode": "markers",
            "name": "Sell Signals",
            "data": {
                "x": "trading.signals_sell.timestamps",
                "y": "trading.signals_sell.prices"
            },
            "marker": {"color": "red", "size": 10, "symbol": "triangle-down"}
        })
    
    return config