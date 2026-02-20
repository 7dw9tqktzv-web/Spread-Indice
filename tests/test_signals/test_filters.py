"""Unit tests for regime filters."""

import numpy as np
import pandas as pd
import pytest

from src.signals.filters import RegimeFilterConfig, apply_regime_filter, compute_regime_mask


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
