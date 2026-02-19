"""Parquet read/write cache layer for processed data."""

from pathlib import Path

import pandas as pd

from src.data.alignment import AlignedPair
from src.data.loader import BarData
from src.spread.pair import SpreadPair

CACHE_DIR = Path("output/cache")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --- BarData cache ---

def cache_bardata(data: BarData, cache_dir: Path = CACHE_DIR) -> Path:
    """Save BarData to Parquet. Returns the file path."""
    _ensure_dir(cache_dir)
    filename = f"{data.instrument.value}_{data.timeframe}.parquet"
    path = cache_dir / filename
    data.df.to_parquet(path)
    return path


def load_bardata_cache(instrument_name: str, timeframe: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame | None:
    """Load cached BarData DataFrame, or None if not found."""
    path = cache_dir / f"{instrument_name}_{timeframe}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


# --- AlignedPair cache ---

def cache_aligned_pair(aligned: AlignedPair, cache_dir: Path = CACHE_DIR) -> Path:
    """Save AlignedPair to Parquet."""
    _ensure_dir(cache_dir)
    pair_name = f"{aligned.pair.leg_a.value}_{aligned.pair.leg_b.value}"
    filename = f"aligned_{pair_name}_{aligned.timeframe}.parquet"
    path = cache_dir / filename
    aligned.df.to_parquet(path)
    return path


def load_aligned_pair_cache(
    pair: SpreadPair,
    timeframe: str,
    cache_dir: Path = CACHE_DIR,
) -> AlignedPair | None:
    """Load cached AlignedPair, or None if not found."""
    pair_name = f"{pair.leg_a.value}_{pair.leg_b.value}"
    path = cache_dir / f"aligned_{pair_name}_{timeframe}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return AlignedPair(pair=pair, df=df, timeframe=timeframe)
    return None
