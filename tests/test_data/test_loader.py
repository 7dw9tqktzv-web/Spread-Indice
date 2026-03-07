"""Unit tests for data loader (CSV parsing)."""

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import BarData, load_sierra_csv
from src.utils.constants import Instrument


SAMPLE_CSV = """\
Date, Time, Open, High, Low, Last, Volume, NumberOfTrades, BidVolume, AskVolume
2024/01/02, 18:00:00, 100.00, 100.50, 99.50, 100.25, 1500, 50, 700, 800
2024/01/02, 18:01:00, 100.25, 100.75, 100.00, 100.50, 1200, 45, 600, 600
2024/01/02, 18:02:00, 100.50, 101.00, 100.25, 100.75, 1800, 60, 900, 900
"""


@pytest.fixture
def csv_path(tmp_path):
    """Write sample CSV to temp file and return path."""
    p = tmp_path / "NQ_test.txt"
    p.write_text(SAMPLE_CSV)
    return p


class TestLoadSierraCSV:

    def test_basic_load(self, csv_path):
        data = load_sierra_csv(csv_path, Instrument.NQ)
        assert isinstance(data, BarData)
        assert data.instrument == Instrument.NQ
        assert data.timeframe == "1min"
        assert len(data.df) == 3

    def test_columns_renamed(self, csv_path):
        data = load_sierra_csv(csv_path, Instrument.NQ)
        assert "close" in data.df.columns
        assert "volume" in data.df.columns
        assert "Last" not in data.df.columns

    def test_datetime_index(self, csv_path):
        data = load_sierra_csv(csv_path, Instrument.NQ)
        assert isinstance(data.df.index, pd.DatetimeIndex)
        assert data.df.index[0] == pd.Timestamp("2024-01-02 18:00:00")

    def test_prices_are_numeric(self, csv_path):
        data = load_sierra_csv(csv_path, Instrument.NQ)
        assert data.df["close"].dtype == float
        assert pd.api.types.is_numeric_dtype(data.df["volume"])

    def test_sorted_index(self, csv_path):
        data = load_sierra_csv(csv_path, Instrument.NQ)
        assert data.df.index.is_monotonic_increasing

    def test_whitespace_in_columns(self, tmp_path):
        """CSV with extra whitespace in column names should still work."""
        csv = "Date , Time , Open , High , Low , Last , Volume , NumberOfTrades, BidVolume, AskVolume\n"
        csv += "2024/01/02, 18:00:00, 100, 101, 99, 100.5, 1000, 40, 500, 500\n"
        p = tmp_path / "test.txt"
        p.write_text(csv)
        data = load_sierra_csv(p, Instrument.NQ)
        assert "close" in data.df.columns
        assert len(data.df) == 1
