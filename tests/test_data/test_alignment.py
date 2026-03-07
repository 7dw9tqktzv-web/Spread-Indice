"""Unit tests for pair alignment."""

import numpy as np
import pandas as pd
import pytest

from src.data.alignment import AlignedPair, align_pair
from src.data.loader import BarData
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument


def _make_bardata(instrument, prices, start="2024-01-02 18:00", freq="5min"):
    """Create a BarData with given close prices."""
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame({
        "close": prices,
        "volume": np.ones(n) * 1000.0,
    }, index=idx)
    return BarData(instrument=instrument, timeframe=freq, df=df)


class TestAlignPair:

    def test_basic_alignment(self):
        a = _make_bardata(Instrument.NQ, [100.0, 101.0, 102.0])
        b = _make_bardata(Instrument.YM, [200.0, 201.0, 202.0])
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)

        aligned = align_pair(a, b, pair)
        assert isinstance(aligned, AlignedPair)
        assert len(aligned.df) == 3
        assert list(aligned.df.columns) == ["close_a", "volume_a", "close_b", "volume_b"]

    def test_mismatched_indices_inner_join(self):
        """Only overlapping timestamps should survive."""
        idx_a = pd.date_range("2024-01-02 18:00", periods=5, freq="5min")
        idx_b = pd.date_range("2024-01-02 18:10", periods=5, freq="5min")
        a = BarData(Instrument.NQ, "5min", pd.DataFrame(
            {"close": [100.0] * 5, "volume": [1000.0] * 5}, index=idx_a
        ))
        b = BarData(Instrument.YM, "5min", pd.DataFrame(
            {"close": [200.0] * 5, "volume": [1000.0] * 5}, index=idx_b
        ))
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
        aligned = align_pair(a, b, pair)
        # Overlap: 18:10, 18:15, 18:20 = 3 bars
        assert len(aligned.df) == 3

    def test_nan_raises_error(self):
        """NaN in close prices after alignment should raise ValueError."""
        idx = pd.date_range("2024-01-02 18:00", periods=3, freq="5min")
        a = BarData(Instrument.NQ, "5min", pd.DataFrame(
            {"close": [100.0, np.nan, 102.0], "volume": [1000.0] * 3}, index=idx
        ))
        b = BarData(Instrument.YM, "5min", pd.DataFrame(
            {"close": [200.0, 201.0, 202.0], "volume": [1000.0] * 3}, index=idx
        ))
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
        with pytest.raises(ValueError, match="NaN"):
            align_pair(a, b, pair)

    def test_non_positive_prices_raise_error(self):
        """Zero or negative prices should raise ValueError (log guard)."""
        a = _make_bardata(Instrument.NQ, [100.0, 0.0, 102.0])
        b = _make_bardata(Instrument.YM, [200.0, 201.0, 202.0])
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
        with pytest.raises(ValueError, match="Non-positive prices"):
            align_pair(a, b, pair)

    def test_negative_prices_raise_error(self):
        a = _make_bardata(Instrument.NQ, [100.0, -5.0, 102.0])
        b = _make_bardata(Instrument.YM, [200.0, 201.0, 202.0])
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
        with pytest.raises(ValueError, match="Non-positive prices"):
            align_pair(a, b, pair)
