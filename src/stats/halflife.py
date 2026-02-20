"""Half-life of mean reversion via AR(1) / Ornstein-Uhlenbeck estimation."""

import numpy as np
import pandas as pd


def half_life(spread: pd.Series) -> float:
    """Estimate half-life from AR(1) regression on full series.

    Z(t) = a + b*Z(t-1) + e(t)
    half_life = -ln(2) / ln(b)

    Returns NaN if b >= 1 (not mean-reverting) or insufficient data.
    """
    ts = spread.dropna()
    if len(ts) < 10:
        return np.nan

    y = ts.values[1:]
    x = ts.values[:-1]

    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        params = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan

    b = params[1]
    if b >= 1.0 or b <= 0.0:
        return np.nan

    return -np.log(2) / np.log(b)


def half_life_rolling(
    spread: pd.Series,
    window: int = 24,
    step: int = 1,
) -> pd.Series:
    """Rolling half-life computed every `step` bars."""
    result = pd.Series(np.nan, index=spread.index)

    for i in range(window, len(spread), step):
        chunk = spread.iloc[i - window : i]
        result.iloc[i] = half_life(chunk)

    return result.ffill()
