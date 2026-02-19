"""Resample 1-minute bars to 5-minute OHLCV."""

import pandas as pd

from src.data.loader import BarData


def resample_to_5min(data: BarData) -> BarData:
    """Resample 1-minute BarData to 5-minute using standard OHLCV aggregation."""
    df = data.df.copy()

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Include optional columns if present
    if "num_trades" in df.columns:
        agg_rules["num_trades"] = "sum"
    if "bid_vol" in df.columns:
        agg_rules["bid_vol"] = "sum"
    if "ask_vol" in df.columns:
        agg_rules["ask_vol"] = "sum"

    # Only aggregate columns that exist
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    resampled = df.resample("5min").agg(agg_rules)

    # Drop bars where close is NaN (incomplete periods)
    resampled = resampled.dropna(subset=["close"])

    return BarData(instrument=data.instrument, timeframe="5min", df=resampled)
