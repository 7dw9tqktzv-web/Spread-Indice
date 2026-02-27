"""Tests for simplified ADF statistic (Sierra Chart compatible)."""

import numpy as np
import pandas as pd

from src.stats.stationarity import adf_statistic_simple


def _make_series(values, freq="5min"):
    """Helper: wrap array in a DatetimeIndex Series."""
    idx = pd.date_range("2024-01-02 18:00", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def test_stationary_ar1_below_critical():
    """Stationary AR(1) with b=0.5 should produce ADF stat < -2.86."""
    rng = np.random.default_rng(42)
    n = 5000
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.5 * z[i - 1] + rng.standard_normal()

    spread = _make_series(z)
    result = adf_statistic_simple(spread, window=200, step=1)

    # Take the last computed value (most data available)
    last_valid = result.dropna().iloc[-1]
    assert last_valid < -2.86, f"Expected ADF stat < -2.86, got {last_valid}"


def test_random_walk_above_critical():
    """Random walk (cumsum of normals) should produce ADF stat > -2.86."""
    rng = np.random.default_rng(42)
    n = 5000
    z = np.cumsum(rng.standard_normal(n))

    spread = _make_series(z)
    result = adf_statistic_simple(spread, window=200, step=1)

    last_valid = result.dropna().iloc[-1]
    assert last_valid > -2.86, f"Expected ADF stat > -2.86 for random walk, got {last_valid}"


def test_insufficient_data_all_nan():
    """With fewer than 10 clean data points, all results should be NaN."""
    spread = _make_series([1.0, 2.0, 1.5, 2.5, 1.0])
    result = adf_statistic_simple(spread, window=20, step=1)

    assert result.isna().all(), "Expected all NaN for insufficient data"


def test_step_greater_than_one_ffills():
    """When step > 1, output should be forward-filled (not all NaN between steps)."""
    rng = np.random.default_rng(42)
    n = 500
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.5 * z[i - 1] + rng.standard_normal()

    spread = _make_series(z)
    adf_statistic_simple(spread, window=50, step=1)
    result_step5 = adf_statistic_simple(spread, window=50, step=5)

    # Step 5 should have forward-filled values (not NaN) at non-step positions
    # After the first computed point, there should be consecutive non-NaN values
    filled = result_step5.dropna()
    assert len(filled) > 0, "Step=5 should still produce some non-NaN values"

    # The ffill should create runs of identical values between step points
    tail = result_step5.iloc[50:100]
    non_nan = tail.dropna()
    assert len(non_nan) > len(tail) // 2, "Ffill should produce mostly non-NaN values"


def test_output_length_matches_input():
    """Output Series should have the same length and index as input."""
    rng = np.random.default_rng(42)
    n = 300
    spread = _make_series(rng.standard_normal(n))
    result = adf_statistic_simple(spread, window=50, step=1)

    assert len(result) == n
    assert result.index.equals(spread.index)
