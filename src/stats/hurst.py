"""Hurst exponent estimation via Rescaled Range (R/S) method."""

import numpy as np
import pandas as pd


def hurst_exponent(series: pd.Series) -> float:
    """Estimate Hurst exponent on full series using R/S method.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    ts = series.dropna().values
    n = len(ts)
    if n < 16:
        return np.nan

    max_k = min(n // 2, 512)
    lags = []
    rs_values = []

    for lag in [int(2 ** i) for i in np.arange(3, np.log2(max_k) + 0.1, 0.5)]:
        if lag > max_k or lag < 4:
            continue
        rs_list = []
        for start in range(0, n - lag, lag):
            chunk = ts[start : start + lag]
            mean_c = chunk.mean()
            deviations = np.cumsum(chunk - mean_c)
            R = deviations.max() - deviations.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))

    if len(lags) < 2:
        return np.nan

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    H = np.polyfit(log_lags, log_rs, 1)[0]
    return float(np.clip(H, 0.01, 0.99))


def hurst_rolling(series: pd.Series, window: int = 64, step: int = 1) -> pd.Series:
    """Rolling Hurst exponent computed every `step` bars."""
    result = pd.Series(np.nan, index=series.index)

    for i in range(window, len(series), step):
        chunk = series.iloc[i - window : i]
        result.iloc[i] = hurst_exponent(chunk)

    return result.ffill()
