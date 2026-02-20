"""Session filtering, gap handling, and outlier removal."""

import pandas as pd

from src.data.loader import BarData
from src.utils.time_utils import SessionConfig, filter_session, filter_trading_window


def clean(data: BarData, session: SessionConfig) -> BarData:
    """Clean raw bar data: session filter, remove zero-volume, forward-fill small gaps."""
    df = data.df.copy()

    # 1. Filter to Globex session (with buffer exclusion)
    df = filter_session(df, session)

    # 2. Remove zero-volume bars
    if "volume" in df.columns:
        df = df.loc[df["volume"] > 0]

    # 3. Forward-fill small gaps (up to 3 consecutive missing bars)
    # Reindex to full 1-minute frequency within existing range, then ffill with limit
    if len(df) > 1:
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1min")
        full_idx = full_idx[full_idx.dayofweek < 5]  # exclude Saturday (5) and Sunday (6)
        df = df.reindex(full_idx)
        df = df.ffill(limit=3)
        df = df.dropna(subset=["close"])

    # 4. Re-apply session filter on reindexed data
    df = filter_session(df, session)

    return BarData(instrument=data.instrument, timeframe=data.timeframe, df=df)
