"""Regime filters that block entries when statistical conditions are unfavorable."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeFilterConfig:
    """Thresholds for regime filtering."""
    max_hurst: float = 0.45          # Hurst must be below (mean-reverting)
    min_correlation: float = 0.80    # minimum correlation between legs
    max_adf_stat: float = -2.86      # ADF stat must be below (more negative = more stationary)
    min_half_life: int = 5           # half-life minimum in bars
    max_half_life: int = 120         # half-life maximum in bars


def compute_regime_mask(
    metrics: pd.DataFrame,
    config: RegimeFilterConfig,
) -> pd.Series:
    """Compute boolean mask: True where regime is favorable for entry.

    Parameters
    ----------
    metrics : pd.DataFrame
        Output of dashboard.compute_all_metrics() with columns:
        adf_stat, hurst, half_life, correlation.
    config : RegimeFilterConfig
        Threshold values.

    Returns
    -------
    pd.Series of bool, indexed like metrics.
    """
    ok = pd.Series(True, index=metrics.index)

    if "hurst" in metrics.columns:
        ok = ok & (metrics["hurst"] < config.max_hurst)

    if "correlation" in metrics.columns:
        ok = ok & (metrics["correlation"] > config.min_correlation)

    if "adf_stat" in metrics.columns:
        ok = ok & (metrics["adf_stat"] < config.max_adf_stat)

    if "half_life" in metrics.columns:
        ok = ok & (metrics["half_life"] >= config.min_half_life)
        ok = ok & (metrics["half_life"] <= config.max_half_life)

    # NaN in any metric → regime not OK
    ok = ok.fillna(False)

    return ok


def apply_regime_filter(
    signals: pd.Series,
    metrics: pd.DataFrame,
    config: RegimeFilterConfig | None = None,
) -> pd.Series:
    """Filter signals: block entries when regime is unfavorable, never block exits.

    Parameters
    ----------
    signals : pd.Series
        Raw signals from SignalGenerator.generate() ({+1, 0, -1}).
    metrics : pd.DataFrame
        Output of dashboard.compute_all_metrics().
    config : RegimeFilterConfig
        Threshold values.

    Returns
    -------
    pd.Series of {+1, 0, -1} — filtered signals.
    """
    if config is None:
        config = RegimeFilterConfig()

    regime_ok = compute_regime_mask(metrics, config)
    sig = signals.values.copy()
    prev = 0  # track position state to detect entries vs holds

    for t in range(len(sig)):
        current = sig[t]

        # Detect entry: transition from flat to non-flat
        is_entry = (prev == 0) and (current != 0)

        if is_entry and not regime_ok.iloc[t]:
            sig[t] = 0  # block entry

        prev = sig[t]

    return pd.Series(sig, index=signals.index, name="signal")
