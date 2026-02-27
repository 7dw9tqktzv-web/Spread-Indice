"""Resample 1-minute bars to target frequency OHLCV."""


from src.data.loader import BarData


def resample_bardata(data: BarData, target_freq: str) -> BarData:
    """Resample BarData to target frequency using standard OHLCV aggregation.

    Parameters
    ----------
    data : BarData
        Source data (typically 1-minute bars).
    target_freq : str
        Pandas frequency string: "1min", "3min", "5min", "15min", etc.
    """
    if data.timeframe == target_freq:
        return data  # No-op

    df = data.df.copy()

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Include optional columns if present
    for col in ("num_trades", "bid_vol", "ask_vol"):
        if col in df.columns:
            agg_rules[col] = "sum"

    # Only aggregate columns that exist
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

    resampled = df.resample(target_freq).agg(agg_rules)

    # Drop bars where close is NaN (incomplete periods)
    resampled = resampled.dropna(subset=["close"])

    return BarData(instrument=data.instrument, timeframe=target_freq, df=resampled)


def resample_to_5min(data: BarData) -> BarData:
    """Resample 1-minute BarData to 5-minute (backward compatible)."""
    return resample_bardata(data, "5min")
