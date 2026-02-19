"""Multi-instrument time alignment for spread pairs."""

from dataclasses import dataclass

import pandas as pd

from src.data.loader import BarData
from src.spread.pair import SpreadPair


@dataclass
class AlignedPair:
    """Time-aligned pair of instruments."""
    pair: SpreadPair
    df: pd.DataFrame  # columns: close_a, close_b, volume_a, volume_b
    timeframe: str


def align_pair(a: BarData, b: BarData, pair: SpreadPair) -> AlignedPair:
    """Inner-join two BarData on timestamp to produce aligned pair data."""
    df_a = a.df[["close", "volume"]].rename(columns={"close": "close_a", "volume": "volume_a"})
    df_b = b.df[["close", "volume"]].rename(columns={"close": "close_b", "volume": "volume_b"})

    merged = df_a.join(df_b, how="inner")

    # Verify no NaN in output
    assert not merged.isna().any().any(), "NaN values found after alignment"

    return AlignedPair(pair=pair, df=merged, timeframe=a.timeframe)
