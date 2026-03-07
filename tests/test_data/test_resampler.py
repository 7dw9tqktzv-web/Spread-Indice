"""Unit tests for bar data resampler."""

import numpy as np
import pandas as pd

from src.data.loader import BarData
from src.data.resampler import resample_bardata, resample_to_5min
from src.utils.constants import Instrument


def _make_1min_data(n=10):
    """Create synthetic 1-min BarData."""
    idx = pd.date_range("2024-01-02 18:00", periods=n, freq="1min")
    df = pd.DataFrame({
        "open": np.arange(100.0, 100.0 + n),
        "high": np.arange(101.0, 101.0 + n),
        "low": np.arange(99.0, 99.0 + n),
        "close": np.arange(100.5, 100.5 + n),
        "volume": np.ones(n) * 100.0,
    }, index=idx)
    return BarData(instrument=Instrument.NQ, timeframe="1min", df=df)


class TestResampleBardata:

    def test_noop_same_freq(self):
        data = _make_1min_data(5)
        data_tf = BarData(Instrument.NQ, "5min", data.df.copy())
        result = resample_bardata(data_tf, "5min")
        assert result is data_tf  # same object, no-op

    def test_1min_to_5min(self):
        data = _make_1min_data(10)
        result = resample_bardata(data, "5min")
        assert result.timeframe == "5min"
        assert len(result.df) == 2  # 10 bars / 5 = 2

    def test_ohlcv_aggregation(self):
        data = _make_1min_data(5)
        result = resample_bardata(data, "5min")
        row = result.df.iloc[0]
        assert row["open"] == 100.0      # first
        assert row["high"] == 105.0      # max
        assert row["low"] == 99.0        # min
        assert row["close"] == 104.5     # last
        assert row["volume"] == 500.0    # sum

    def test_resample_to_5min_helper(self):
        data = _make_1min_data(10)
        result = resample_to_5min(data)
        assert result.timeframe == "5min"
        assert len(result.df) == 2
