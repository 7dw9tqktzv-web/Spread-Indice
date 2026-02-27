"""ADF stationarity tests for spread residuals.

Two implementations:
- Simplified (Sierra Chart compatible): ΔZ = μ + γ·Z(t-1), stat = γ/SE(γ)
- Full (statsmodels): augmented with lag terms, AIC lag selection
"""

import numpy as np
import pandas as pd
from numba import njit
from statsmodels.tsa.stattools import adfuller

# --- Simplified ADF (Sierra Chart compatible, numba-compiled) ---

@njit(cache=True)
def _adf_simple_numba(spread_vals, window, step):
    """Numba-compiled rolling simplified ADF statistic.

    Regression: ΔZ(t) = μ + γ·Z(t-1) + ε
    ADF statistic = γ / SE(γ)
    """
    n = len(spread_vals)
    result = np.full(n, np.nan)
    n_pts = window - 1

    for i in range(window, n, step):
        # Compute sums for OLS in one pass
        sum_d = 0.0
        sum_l = 0.0
        sum_dl = 0.0
        sum_ll = 0.0
        count = 0

        for j in range(i - n_pts, i):
            d_j = spread_vals[j] - spread_vals[j - 1]  # delta
            l_j = spread_vals[j - 1]  # lag
            if np.isnan(d_j) or np.isnan(l_j):
                continue
            sum_d += d_j
            sum_l += l_j
            sum_dl += d_j * l_j
            sum_ll += l_j * l_j
            count += 1

        if count < 10:
            continue

        mean_d = sum_d / count
        mean_l = sum_l / count

        # ss_l = sum((l - mean_l)^2) = sum_ll - count * mean_l^2
        ss_l = sum_ll - count * mean_l * mean_l
        if ss_l <= 0.0:
            continue

        # gamma = sum((l - mean_l)(d - mean_d)) / ss_l
        #       = (sum_dl - count * mean_l * mean_d) / ss_l
        ss_dl = sum_dl - count * mean_l * mean_d
        gamma = ss_dl / ss_l
        mu = mean_d - gamma * mean_l

        # Residual sum of squares (second pass)
        ssr = 0.0
        for j in range(i - n_pts, i):
            d_j = spread_vals[j] - spread_vals[j - 1]
            l_j = spread_vals[j - 1]
            if np.isnan(d_j) or np.isnan(l_j):
                continue
            resid = d_j - mu - gamma * l_j
            ssr += resid * resid

        variance = ssr / (count - 2) if count > 2 else 0.0
        if variance <= 0.0:
            continue

        se_gamma = np.sqrt(variance / ss_l)
        if se_gamma > 0.0:
            result[i] = gamma / se_gamma

    return result


def adf_statistic_simple(spread: pd.Series, window: int = 24, step: int = 1) -> pd.Series:
    """Rolling simplified ADF statistic (compatible Sierra Chart v1.5).

    Regression: ΔZ(t) = μ + γ·Z(t-1) + ε
    ADF statistic = γ / SE(γ)

    No augmentation terms (no lag differences).
    Critical value at 5%: -2.86 (reject H0 if stat < -2.86).

    Numba-compiled inner loop for ~20-50x speedup over pure Python.
    """
    spread_vals = np.ascontiguousarray(spread.values, dtype=np.float64)
    result = _adf_simple_numba(spread_vals, window, step)

    out = pd.Series(result, index=spread.index, name="adf_simple")
    if step > 1:
        out = out.ffill()
    return out


# --- Full ADF (statsmodels) ---

def adf_test(spread: pd.Series, maxlag: int | None = None) -> dict:
    """Run full ADF test on spread series (statsmodels).

    Returns dict with: statistic, pvalue, usedlag, nobs, critical_values, is_stationary.
    """
    clean = spread.dropna()
    result = adfuller(clean.values, maxlag=maxlag, autolag="AIC")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "usedlag": result[2],
        "nobs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def adf_rolling(
    spread: pd.Series,
    window: int = 24,
    step: int = 1,
) -> pd.Series:
    """Rolling ADF p-value (statsmodels) computed every `step` bars.

    Returns Series of p-values (NaN where not computed).
    """
    pvalues = pd.Series(np.nan, index=spread.index)

    for i in range(window, len(spread), step):
        chunk = spread.iloc[i - window : i].dropna()
        if len(chunk) < 10:
            continue
        try:
            result = adfuller(chunk.values, maxlag=None, autolag="AIC")
            pvalues.iloc[i] = result[1]
        except Exception:
            pass

    return pvalues.ffill()
