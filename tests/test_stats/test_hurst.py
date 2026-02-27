"""Tests for Hurst exponent estimation (variance-ratio method)."""

import numpy as np
import pandas as pd

from src.stats.hurst import hurst_exponent, hurst_rolling


def _make_series(values, freq="5min"):
    """Helper: wrap array in a DatetimeIndex Series."""
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def test_random_walk_hurst_near_half():
    """Random walk (cumsum of normals) should have H close to 0.5."""
    rng = np.random.default_rng(42)
    n = 5000
    z = np.cumsum(rng.standard_normal(n))

    h = hurst_exponent(_make_series(z))
    assert 0.35 < h < 0.65, f"Expected H near 0.5 for random walk, got {h}"


def test_mean_reverting_ar1_hurst_below_half():
    """Mean-reverting AR(1) with b=0.5 should have H < 0.50."""
    rng = np.random.default_rng(42)
    n = 5000
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.5 * z[i - 1] + rng.standard_normal()

    h = hurst_exponent(_make_series(z))
    assert h < 0.50, f"Expected H < 0.50 for mean-reverting series, got {h}"


def test_insufficient_data_returns_nan():
    """With only 3 data points, hurst_exponent should return NaN."""
    spread = _make_series([1.0, 2.0, 1.5])
    h = hurst_exponent(spread)
    assert np.isnan(h), f"Expected NaN for insufficient data, got {h}"


def test_hurst_rolling_length_matches_input():
    """hurst_rolling output should have the same length as input."""
    rng = np.random.default_rng(42)
    n = 500
    z = np.cumsum(rng.standard_normal(n))
    spread = _make_series(z)

    result = hurst_rolling(spread, window=100, step=1)
    assert len(result) == n
    assert result.index.equals(spread.index)


def test_hurst_rolling_values_clipped():
    """hurst_rolling values should be clipped to [0.01, 0.99]."""
    rng = np.random.default_rng(42)
    n = 1000
    z = np.cumsum(rng.standard_normal(n))
    spread = _make_series(z)

    result = hurst_rolling(spread, window=100, step=1)
    valid = result.dropna()

    if len(valid) > 0:
        assert valid.min() >= 0.01, f"Min Hurst {valid.min()} below 0.01"
        assert valid.max() <= 0.99, f"Max Hurst {valid.max()} above 0.99"


def test_hurst_rolling_ffilled():
    """hurst_rolling should forward-fill computed values."""
    rng = np.random.default_rng(42)
    n = 500
    z = np.cumsum(rng.standard_normal(n))
    spread = _make_series(z)

    result = hurst_rolling(spread, window=100, step=1)

    # After the window period, values should be mostly non-NaN due to ffill
    tail = result.iloc[100:]
    non_nan_count = tail.notna().sum()
    assert non_nan_count == len(tail), "After window, all values should be filled"
