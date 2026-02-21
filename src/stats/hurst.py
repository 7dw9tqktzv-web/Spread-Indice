"""Hurst exponent estimation via Variance-Ratio method (vectorized).

Applied directly on spread LEVELS (not differences).
tau(lag) = std(spread[t+lag] - spread[t]) for each lag.
H = slope of log(tau) vs log(lag).

H < 0.5: mean-reverting
H = 0.5: random walk
H > 0.5: trending

Vectorized implementation: rolling std computed by pandas (C-optimized),
only the polyfit loop remains in Python.

References:
    - Ernie Chan, "Algorithmic Trading" (variance-ratio on log-prices/spreads)
    - Weron (2002), "Estimating long range dependence: finite sample properties"
"""

import numpy as np
import pandas as pd


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Estimate Hurst exponent on spread levels via variance-ratio.

    Parameters
    ----------
    series : pd.Series
        Spread level series (e.g. OLS residual).
    max_lag : int
        Maximum lag for tau computation. Default 20.

    Returns
    -------
    float
        Hurst exponent H. NaN if insufficient data.
    """
    ts = series.dropna().values
    n = len(ts)
    if n < max_lag + 2:
        return np.nan

    upper = min(max_lag, n // 2)
    lags = range(2, upper + 1)

    tau = []
    valid_lags = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        s = np.std(diffs)
        if s > 1e-12:
            tau.append(s)
            valid_lags.append(lag)

    if len(valid_lags) < 3:
        return np.nan

    log_lags = np.log(valid_lags)
    log_tau = np.log(tau)
    H = np.polyfit(log_lags, log_tau, 1)[0]
    return float(np.clip(H, 0.01, 0.99))


def hurst_rolling(series: pd.Series, window: int = 256, step: int = 1) -> pd.Series:
    """Rolling Hurst exponent — vectorized implementation.

    Phase 1: precompute rolling std for each lag using pandas (C-optimized).
    Phase 2: loop over positions for polyfit only (lightweight).

    Parameters
    ----------
    series : pd.Series
        Spread level series.
    window : int
        Rolling window size in bars.
    step : int
        Compute every N bars (1 = every bar).

    Returns
    -------
    pd.Series
        Rolling Hurst values, forward-filled.
    """
    ts = series.values
    n = len(ts)
    max_lag = min(window // 4, 50)

    if n < window or max_lag < 3:
        return pd.Series(np.nan, index=series.index)

    lags = list(range(2, max_lag + 1))
    log_lags = np.log(lags)
    n_lags = len(lags)

    # Phase 1: precompute rolling std for each lag (pandas C code, fast)
    # For lag k: diffs[t] = ts[t] - ts[t-k], then rolling std over (window - k) bars
    tau_matrix = np.full((n, n_lags), np.nan)
    for j, lag in enumerate(lags):
        diffs = ts[lag:] - ts[:-lag]
        roll_std = pd.Series(diffs).rolling(window - lag, min_periods=window - lag).std().values
        tau_matrix[lag:, j] = roll_std

    # Precompute linear regression constants (log_lags is fixed)
    # slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    sum_x = log_lags.sum()
    sum_x2 = (log_lags ** 2).sum()

    # Phase 2: compute H at each position via analytical slope
    result = np.full(n, np.nan)
    for i in range(window, n, step):
        tau = tau_matrix[i]
        valid = ~np.isnan(tau) & (tau > 1e-12)

        n_valid = valid.sum()
        if n_valid < 3:
            continue

        if n_valid == n_lags:
            # Fast path: all lags valid, use precomputed sums
            log_tau = np.log(tau)
            sum_y = log_tau.sum()
            sum_xy = (log_lags * log_tau).sum()
            H = (n_lags * sum_xy - sum_x * sum_y) / (n_lags * sum_x2 - sum_x ** 2)
        else:
            # Slow path: subset of valid lags
            log_tau = np.log(tau[valid])
            lx = log_lags[valid]
            nv = n_valid
            sx = lx.sum()
            sy = log_tau.sum()
            sxy = (lx * log_tau).sum()
            sx2 = (lx ** 2).sum()
            H = (nv * sxy - sx * sy) / (nv * sx2 - sx ** 2)

        result[i] = np.clip(H, 0.01, 0.99)

    out = pd.Series(result, index=series.index)
    return out.ffill()
