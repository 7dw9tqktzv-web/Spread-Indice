"""Half-life of mean reversion via AR(1) / Ornstein-Uhlenbeck estimation.

Vectorized: uses rolling covariance instead of per-bar lstsq.
"""

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
    """Rolling half-life — vectorized via rolling covariance.

    AR(1): Z(t) = a + b*Z(t-1)
    b = Cov(Z(t), Z(t-1)) / Var(Z(t-1))
    half_life = -ln(2) / ln(b)
    """
    y = spread.iloc[1:]             # Z(t)
    x = spread.iloc[:-1]            # Z(t-1)

    # Align indices
    y = y.copy()
    y.index = x.index

    # Rolling covariance and variance (pandas C-optimized)
    cov_xy = y.rolling(window - 1, min_periods=window - 1).cov(x)
    var_x = x.rolling(window - 1, min_periods=window - 1).var()

    # AR(1) coefficient b
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (cov_xy / var_x).values

    # Half-life = -ln(2) / ln(b), valid only for 0 < b < 1
    result = np.full(len(spread), np.nan)
    valid = (b > 0) & (b < 1) & np.isfinite(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl_vals = -np.log(2) / np.log(b)
    # Map back to spread index (x starts at index 0, result at index 1+)
    result[1:][valid] = hl_vals[valid]

    out = pd.Series(result, index=spread.index, name="half_life")
    if step > 1:
        # Subsample then ffill
        mask = np.zeros(len(out), dtype=bool)
        mask[window::step] = True
        out[~mask] = np.nan
    return out.ffill()
