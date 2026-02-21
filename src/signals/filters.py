"""Regime and session filters that block entries when conditions are unfavorable."""

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from src.utils.time_utils import SessionConfig


# ---------------------------------------------------------------------------
# Legacy binary filter config (kept for backward compatibility with tests)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeFilterConfig:
    """Thresholds for regime filtering (binary mode)."""
    max_hurst: float = 0.45
    min_correlation: float = 0.80
    max_adf_stat: float = -2.86
    min_half_life: int = 5
    max_half_life: int = 120


def compute_regime_mask(
    metrics: pd.DataFrame,
    config: RegimeFilterConfig,
) -> pd.Series:
    """Compute boolean mask: True where regime is favorable for entry."""
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

    ok = ok.fillna(False)
    return ok


# ---------------------------------------------------------------------------
# Confidence scoring system
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceConfig:
    """Configuration for confidence-based regime scoring.

    Each metric produces a score [0, 1] via linear interpolation.
    Final confidence = weighted sum of scores × 100%.
    ADF is a mandatory gate: if adf_stat >= adf_gate → confidence = 0%.
    """
    # Gate
    adf_gate: float = -1.00           # ADF stat must be below this to enter scoring

    # Score bounds: (worst=0, best=1)
    adf_worst: float = -1.00          # score 0
    adf_best: float = -3.50           # score 1

    hurst_worst: float = 0.52         # score 0
    hurst_best: float = 0.38          # score 1

    corr_worst: float = 0.65          # score 0
    corr_best: float = 0.92           # score 1

    hl_sweet_low: float = 10.0        # start of sweet spot (score 1)
    hl_sweet_high: float = 60.0       # end of sweet spot (score 1)
    hl_min: float = 3.0               # below → score 0
    hl_max: float = 200.0             # above → score 0

    # Weights (must sum to 1.0)
    w_adf: float = 0.40
    w_hurst: float = 0.25
    w_corr: float = 0.20
    w_hl: float = 0.15

    # Minimum confidence to allow entry
    min_confidence: float = 40.0


def _score_linear(value: float, worst: float, best: float) -> float:
    """Linear interpolation: worst→0, best→1, clipped to [0, 1]."""
    if np.isnan(value):
        return 0.0
    span = best - worst
    if abs(span) < 1e-12:
        return 0.0
    return float(np.clip((value - worst) / span, 0.0, 1.0))


def _score_halflife(hl: float, cfg: ConfidenceConfig) -> float:
    """Half-life score: 0 outside [min, max], 1 in sweet spot, interpolated between."""
    if np.isnan(hl) or hl < cfg.hl_min or hl > cfg.hl_max:
        return 0.0
    if cfg.hl_sweet_low <= hl <= cfg.hl_sweet_high:
        return 1.0
    if hl < cfg.hl_sweet_low:
        return float((hl - cfg.hl_min) / (cfg.hl_sweet_low - cfg.hl_min))
    # hl > hl_sweet_high
    return float((cfg.hl_max - hl) / (cfg.hl_max - cfg.hl_sweet_high))


def compute_confidence(
    metrics: pd.DataFrame,
    config: ConfidenceConfig,
) -> pd.Series:
    """Compute confidence score (0-100%) for each bar.

    Parameters
    ----------
    metrics : pd.DataFrame
        Columns: adf_stat, hurst, half_life, correlation.
    config : ConfidenceConfig

    Returns
    -------
    pd.Series of float (0-100), indexed like metrics.
    """
    n = len(metrics)
    confidence = np.zeros(n)

    adf = metrics["adf_stat"].values if "adf_stat" in metrics.columns else np.full(n, np.nan)
    hurst = metrics["hurst"].values if "hurst" in metrics.columns else np.full(n, np.nan)
    corr = metrics["correlation"].values if "correlation" in metrics.columns else np.full(n, np.nan)
    hl = metrics["half_life"].values if "half_life" in metrics.columns else np.full(n, np.nan)

    for i in range(n):
        # Gate: ADF must pass
        if np.isnan(adf[i]) or adf[i] >= config.adf_gate:
            confidence[i] = 0.0
            continue

        s_adf = _score_linear(adf[i], config.adf_worst, config.adf_best)
        s_hurst = _score_linear(hurst[i], config.hurst_worst, config.hurst_best)
        s_corr = _score_linear(corr[i], config.corr_worst, config.corr_best)
        s_hl = _score_halflife(hl[i], config)

        confidence[i] = (
            config.w_adf * s_adf
            + config.w_hurst * s_hurst
            + config.w_corr * s_corr
            + config.w_hl * s_hl
        ) * 100.0

    return pd.Series(confidence, index=metrics.index, name="confidence")


def apply_confidence_filter(
    signals: pd.Series,
    metrics: pd.DataFrame,
    config: ConfidenceConfig | None = None,
) -> pd.Series:
    """Filter signals using confidence scoring. Block entries below min_confidence.

    Never blocks exits (signal → 0 transitions are always allowed).

    Parameters
    ----------
    signals : pd.Series
        Raw signals ({+1, 0, -1}).
    metrics : pd.DataFrame
        Output of dashboard.compute_all_metrics().
    config : ConfidenceConfig

    Returns
    -------
    pd.Series of {+1, 0, -1} — filtered signals.
    """
    if config is None:
        config = ConfidenceConfig()

    confidence = compute_confidence(metrics, config)
    sig = signals.values.copy()
    prev = 0

    for t in range(len(sig)):
        current = sig[t]
        is_entry = (prev == 0) and (current != 0)

        if is_entry and confidence.iloc[t] < config.min_confidence:
            sig[t] = 0

        prev = sig[t]

    return pd.Series(sig, index=signals.index, name="signal")


# ---------------------------------------------------------------------------
# Legacy binary filter (used by existing tests)
# ---------------------------------------------------------------------------

def apply_regime_filter(
    signals: pd.Series,
    metrics: pd.DataFrame,
    config: RegimeFilterConfig | None = None,
) -> pd.Series:
    """Filter signals: block entries when regime is unfavorable, never block exits."""
    if config is None:
        config = RegimeFilterConfig()

    regime_ok = compute_regime_mask(metrics, config)
    sig = signals.values.copy()
    prev = 0

    for t in range(len(sig)):
        current = sig[t]
        is_entry = (prev == 0) and (current != 0)

        if is_entry and not regime_ok.iloc[t]:
            sig[t] = 0

        prev = sig[t]

    return pd.Series(sig, index=signals.index, name="signal")


# ---------------------------------------------------------------------------
# Trading window filter
# ---------------------------------------------------------------------------

def apply_trading_window_filter(
    signals: pd.Series,
    session: SessionConfig | None = None,
) -> pd.Series:
    """Force signals to 0 outside the trading window [trading_start, trading_end)."""
    if session is None:
        session = SessionConfig()

    t_start = session.trading_start
    t_end = session.trading_end

    in_window = signals.index.map(lambda ts: t_start <= ts.time() < t_end)
    sig = signals.copy()
    sig[~in_window] = 0
    return sig
