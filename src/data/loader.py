"""CSV parsing for Sierra Charts bar data exports."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.constants import Instrument, SIERRA_COLUMNS


@dataclass
class BarData:
    """Immutable container for a single instrument's bar data."""
    instrument: Instrument
    timeframe: str
    df: pd.DataFrame  # DatetimeIndex (CT), columns: open, high, low, close, volume, ...


def load_sierra_csv(path: Path, instrument: Instrument) -> BarData:
    """Load a Sierra Charts CSV export into a BarData container.

    Expected format: Date, Time, Open, High, Low, Last, Volume, NumberOfTrades, BidVolume, AskVolume
    Timestamps are assumed to be in Chicago Time (CT).
    """
    df = pd.read_csv(
        path,
        skipinitialspace=True,
    )

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Build datetime index from Date + Time columns
    df["datetime"] = pd.to_datetime(
        df["Date"].str.strip() + " " + df["Time"].str.strip(),
        format="%Y/%m/%d %H:%M:%S",
    )
    df = df.set_index("datetime").drop(columns=["Date", "Time"])

    # Rename columns to standard names
    rename_map = {k: v for k, v in SIERRA_COLUMNS.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Enforce float64 for price/volume columns
    numeric_cols = ["open", "high", "low", "close", "volume", "num_trades", "bid_vol", "ask_vol"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()

    return BarData(instrument=instrument, timeframe="1min", df=df)
