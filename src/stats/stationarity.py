"""ADF stationarity tests for spread residuals.

Two implementations:
- Simplified (Sierra Chart compatible): ΔZ = μ + γ·Z(t-1), stat = γ/SE(γ)
- Full (statsmodels): augmented with lag terms, AIC lag selection
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# --- Simplified ADF (Sierra Chart compatible) ---

def adf_statistic_simple(spread: pd.Series, window: int = 24) -> pd.Series:
    """Rolling simplified ADF statistic (compatible Sierra Chart v1.5).

    Regression: ΔZ(t) = μ + γ·Z(t-1) + ε
    ADF statistic = γ / SE(γ)

    No augmentation terms (no lag differences).
    Critical value at 5%: -2.86 (reject H0 if stat < -2.86).
    """
    spread_vals = spread.values
    delta = np.diff(spread_vals, prepend=np.nan)  # ΔZ
    lag = np.roll(spread_vals, 1)                  # Z(t-1)
    lag[0] = np.nan

    n = len(spread_vals)
    result = np.full(n, np.nan)
    n_pts = window - 1  # pairs (delta, lag) in each window

    for i in range(window, n):
        d = delta[i - n_pts:i]
        l = lag[i - n_pts:i]

        # Skip if any NaN
        mask = ~(np.isnan(d) | np.isnan(l))
        if mask.sum() < 10:
            continue

        d_clean = d[mask]
        l_clean = l[mask]
        n_clean = len(d_clean)

        # OLS: d = mu + gamma * l
        mean_l = l_clean.mean()
        mean_d = d_clean.mean()
        ss_l = np.sum((l_clean - mean_l) ** 2)

        if ss_l == 0:
            continue

        gamma = np.sum((l_clean - mean_l) * (d_clean - mean_d)) / ss_l
        mu = mean_d - gamma * mean_l

        # Residuals
        residuals = d_clean - mu - gamma * l_clean
        ssr = np.sum(residuals ** 2)
        variance = ssr / (n_clean - 2) if n_clean > 2 else 0

        if variance <= 0 or ss_l == 0:
            continue

        se_gamma = np.sqrt(variance / ss_l)
        if se_gamma > 0:
            result[i] = gamma / se_gamma

    return pd.Series(result, index=spread.index, name="adf_simple")


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
