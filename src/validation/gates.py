"""Binary statistical gates for entry filtering.

Three gates must pass simultaneously for entries to be allowed:
- ADF statistic < -2.86 (5% significance: spread is stationary)
- Hurst exponent < 0.50 (mean-reversion by definition)
- Correlation > 0.70 (strong pair relationship)

ADF has low power on short intraday windows (~10% pass rate at window=24),
but the diagnostic shows it filters noise effectively: PF degrades monotonically
as the threshold is relaxed. The 5% critical value (-2.86) is the best
compromise between selectivity and trade count.

Gates are NOT optimized -- thresholds come from statistical theory.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from src.stats.correlation import rolling_correlation
from src.stats.hurst import hurst_rolling
from src.stats.stationarity import adf_statistic_simple


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for binary gates (fixed by theory, NOT optimized)."""

    adf_threshold: float = -2.86  # 5% critical value
    hurst_threshold: float = 0.50  # mean-reversion definition
    corr_threshold: float = 0.70  # strong correlation

    # Rolling windows for computing gate metrics
    adf_window: int = 24
    hurst_window: int = 64
    corr_window: int = 24


def compute_gate_mask(
    spread: pd.Series,
    close_a: pd.Series,
    close_b: pd.Series,
    config: GateConfig,
) -> np.ndarray:
    """Compute binary gate mask: True where ALL three gates pass.

    Parameters
    ----------
    spread : pd.Series
        Spread residual (from HedgeResult.spread).
    close_a, close_b : pd.Series
        Raw close prices.
    config : GateConfig

    Returns
    -------
    np.ndarray of bool, same length as spread.
    """
    adf = adf_statistic_simple(spread, window=config.adf_window, step=1)
    hurst = hurst_rolling(spread, window=config.hurst_window, step=1)
    corr = rolling_correlation(close_a, close_b, window=config.corr_window)

    adf_vals = adf.values
    hurst_vals = hurst.values
    corr_vals = corr.values

    gate_adf = (adf_vals < config.adf_threshold) & ~np.isnan(adf_vals)
    gate_hurst = (hurst_vals < config.hurst_threshold) & ~np.isnan(hurst_vals)
    gate_corr = (corr_vals > config.corr_threshold) & ~np.isnan(corr_vals)

    return gate_adf & gate_hurst & gate_corr


@njit(cache=True)
def apply_gate_filter_numba(
    sig: np.ndarray, gate_mask: np.ndarray
) -> np.ndarray:
    """Block new entries where gate_mask is False. Never blocks exits.

    Same entry-only blocking logic as _apply_conf_filter_numba but with
    a boolean mask instead of a confidence score.

    Parameters
    ----------
    sig : np.ndarray of int8
        Signal array from generate_signals_numba.
    gate_mask : np.ndarray of bool
        True = all gates pass, False = at least one gate fails.

    Returns
    -------
    np.ndarray of int8
    """
    out = sig.copy()
    prev = np.int8(0)
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0 and not gate_mask[t]:
            out[t] = np.int8(0)
        prev = out[t]
    return out
