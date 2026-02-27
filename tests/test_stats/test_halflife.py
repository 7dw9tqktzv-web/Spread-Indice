"""Tests for half-life of mean reversion estimation."""

import numpy as np
import pandas as pd

from src.stats.halflife import half_life, half_life_rolling


def _make_series(values, freq="5min"):
    """Helper: wrap array in a DatetimeIndex Series."""
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def test_known_ar1_half_life():
    """AR(1) with b=0.9: theoretical half_life = -ln(2)/ln(0.9) ~ 6.58."""
    rng = np.random.default_rng(42)
    n = 10000
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.9 * z[i - 1] + rng.standard_normal()

    hl = half_life(_make_series(z))
    expected = -np.log(2) / np.log(0.9)  # ~6.58
    assert abs(hl - expected) < 2.0, f"Expected hl ~ {expected:.2f}, got {hl:.2f}"


def test_random_walk_nan_or_very_large():
    """Random walk (b ~ 1) should return NaN or a very large half-life.

    The function returns NaN when b >= 1.0 exactly. On finite samples,
    OLS may estimate b slightly below 1.0, yielding a very large hl.
    Either outcome is acceptable for a non-mean-reverting series.
    """
    rng = np.random.default_rng(42)
    n = 5000
    z = np.cumsum(rng.standard_normal(n))

    hl = half_life(_make_series(z))
    assert np.isnan(hl) or hl > 100, f"Expected NaN or hl > 100 for random walk, got {hl}"


def test_insufficient_data_returns_nan():
    """With fewer than 10 data points, should return NaN."""
    spread = _make_series([1.0, 2.0, 1.5, 2.5, 1.0])
    hl = half_life(spread)
    assert np.isnan(hl), f"Expected NaN for insufficient data, got {hl}"


def test_half_life_rolling_returns_series_with_ffill():
    """half_life_rolling should return a Series of same length, forward-filled."""
    rng = np.random.default_rng(42)
    n = 500
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.9 * z[i - 1] + rng.standard_normal()

    spread = _make_series(z)
    result = half_life_rolling(spread, window=50, step=1)

    assert isinstance(result, pd.Series)
    assert len(result) == n
    assert result.index.equals(spread.index)

    # After the window period, values should be filled (not all NaN)
    tail = result.iloc[50:]
    non_nan_count = tail.notna().sum()
    assert non_nan_count > len(tail) * 0.8, (
        f"Expected mostly non-NaN after window, got {non_nan_count}/{len(tail)}"
    )


def test_half_life_rolling_values_positive():
    """Valid half-life values should be positive (for 0 < b < 1)."""
    rng = np.random.default_rng(42)
    n = 500
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.8 * z[i - 1] + rng.standard_normal()

    spread = _make_series(z)
    result = half_life_rolling(spread, window=100, step=1)

    valid = result.dropna()
    if len(valid) > 0:
        assert (valid > 0).all(), "All valid half-life values should be positive"
