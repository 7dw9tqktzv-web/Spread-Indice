"""Unit tests for regime and trading window filters."""

import numpy as np
import pandas as pd
import pytest

from src.signals.filters import (
    RegimeFilterConfig,
    apply_regime_filter,
    apply_trading_window_filter,
    compute_regime_mask,
)
from src.utils.time_utils import SessionConfig


def _make_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 18:00", periods=n, freq="5min")


class TestComputeRegimeMask:
    def test_all_favorable(self):
        idx = _make_index(5)
        metrics = pd.DataFrame({
            "hurst": [0.3] * 5,
            "correlation": [0.9] * 5,
            "adf_stat": [-3.5] * 5,
            "half_life": [20] * 5,
        }, index=idx)
        mask = compute_regime_mask(metrics, RegimeFilterConfig())
        assert mask.all()

    def test_hurst_too_high(self):
        idx = _make_index(3)
        metrics = pd.DataFrame({
            "hurst": [0.3, 0.6, 0.3],       # bar 1 unfavorable
            "correlation": [0.9] * 3,
            "adf_stat": [-3.5] * 3,
            "half_life": [20] * 3,
        }, index=idx)
        mask = compute_regime_mask(metrics, RegimeFilterConfig())
        assert mask.iloc[0] == True
        assert mask.iloc[1] == False  # hurst > 0.45
        assert mask.iloc[2] == True

    def test_nan_metric_is_false(self):
        idx = _make_index(2)
        metrics = pd.DataFrame({
            "hurst": [0.3, np.nan],
            "correlation": [0.9, 0.9],
            "adf_stat": [-3.5, -3.5],
            "half_life": [20, 20],
        }, index=idx)
        mask = compute_regime_mask(metrics, RegimeFilterConfig())
        assert mask.iloc[0] == True
        assert mask.iloc[1] == False  # NaN → not OK


class TestApplyRegimeFilter:
    def test_entry_blocked_by_bad_regime(self):
        idx = _make_index(5)
        # Raw signals: flat, flat, entry long, hold, exit
        # Bad regime at bars 2 AND 3 so re-entry at bar 3 is also blocked
        signals = pd.Series([0, 0, 1, 1, 0], index=idx, name="signal")
        metrics = pd.DataFrame({
            "hurst": [0.3, 0.3, 0.6, 0.6, 0.3],  # bars 2-3: hurst bad
            "correlation": [0.9] * 5,
            "adf_stat": [-3.5] * 5,
            "half_life": [20] * 5,
        }, index=idx)
        filtered = apply_regime_filter(signals, metrics)
        assert filtered.iloc[2] == 0  # entry blocked
        assert filtered.iloc[3] == 0  # re-entry also blocked (bad regime)

    def test_exit_not_blocked(self):
        idx = _make_index(5)
        # Raw signals: flat, entry long, hold, exit, flat
        signals = pd.Series([0, 1, 1, 0, 0], index=idx, name="signal")
        metrics = pd.DataFrame({
            "hurst": [0.3, 0.3, 0.3, 0.6, 0.3],  # bar 3: bad regime at exit
            "correlation": [0.9] * 5,
            "adf_stat": [-3.5] * 5,
            "half_life": [20] * 5,
        }, index=idx)
        filtered = apply_regime_filter(signals, metrics)
        assert filtered.iloc[1] == 1  # entry allowed
        assert filtered.iloc[3] == 0  # exit preserved (not blocked)

    def test_hold_not_affected(self):
        idx = _make_index(4)
        # Already in position, regime goes bad → hold is preserved
        signals = pd.Series([0, 1, 1, 1], index=idx, name="signal")
        metrics = pd.DataFrame({
            "hurst": [0.3, 0.3, 0.6, 0.6],  # regime degrades while holding
            "correlation": [0.9] * 4,
            "adf_stat": [-3.5] * 4,
            "half_life": [20] * 4,
        }, index=idx)
        filtered = apply_regime_filter(signals, metrics)
        assert filtered.iloc[1] == 1  # entry
        assert filtered.iloc[2] == 1  # hold preserved despite bad hurst
        assert filtered.iloc[3] == 1  # hold preserved


class TestTradingWindowFilter:
    def test_signal_zeroed_outside_window(self):
        # 03:55 and 14:05 are outside [04:00, 14:00)
        idx = pd.to_datetime([
            "2024-01-02 03:55",  # before window
            "2024-01-02 04:00",  # window start (included)
            "2024-01-02 10:00",  # inside
            "2024-01-02 13:55",  # inside (last bar before end)
            "2024-01-02 14:00",  # window end (excluded)
            "2024-01-02 18:00",  # overnight
        ])
        signals = pd.Series([1, 1, -1, 1, -1, 1], index=idx, name="signal")
        filtered = apply_trading_window_filter(signals)

        assert filtered.iloc[0] == 0   # 03:55 → zeroed
        assert filtered.iloc[1] == 1   # 04:00 → kept
        assert filtered.iloc[2] == -1  # 10:00 → kept
        assert filtered.iloc[3] == 1   # 13:55 → kept
        assert filtered.iloc[4] == 0   # 14:00 → zeroed (strict <)
        assert filtered.iloc[5] == 0   # 18:00 → zeroed

    def test_flat_signals_unchanged(self):
        idx = pd.to_datetime(["2024-01-02 02:00", "2024-01-02 10:00"])
        signals = pd.Series([0, 0], index=idx, name="signal")
        filtered = apply_trading_window_filter(signals)
        assert (filtered == 0).all()

    def test_custom_session(self):
        session = SessionConfig(trading_start=pd.Timestamp("2000-01-01 05:00").time(),
                                trading_end=pd.Timestamp("2000-01-01 12:00").time())
        idx = pd.to_datetime(["2024-01-02 04:30", "2024-01-02 05:00", "2024-01-02 12:00"])
        signals = pd.Series([1, 1, 1], index=idx, name="signal")
        filtered = apply_trading_window_filter(signals, session=session)

        assert filtered.iloc[0] == 0  # 04:30 outside [05:00, 12:00)
        assert filtered.iloc[1] == 1  # 05:00 inside
        assert filtered.iloc[2] == 0  # 12:00 outside (strict <)
