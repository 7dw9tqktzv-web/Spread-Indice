"""Regime and session filters that block entries when conditions are unfavorable."""

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd
from numba import njit

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


# ---------------------------------------------------------------------------
# Vectorized confidence scoring (numpy, ~250x faster than loop)
# ---------------------------------------------------------------------------

def _score_halflife_vec(hl: np.ndarray, hl_min: float, hl_max: float,
                        sweet_low: float, sweet_high: float) -> np.ndarray:
    """Vectorized half-life scoring: trapezoid shape."""
    score = np.zeros_like(hl)
    valid = ~np.isnan(hl) & (hl >= hl_min) & (hl <= hl_max)

    # Sweet spot
    in_sweet = valid & (hl >= sweet_low) & (hl <= sweet_high)
    score[in_sweet] = 1.0

    # Below sweet spot
    below = valid & (hl < sweet_low) & (hl >= hl_min)
    denom_lo = sweet_low - hl_min
    if denom_lo > 0:
        score[below] = (hl[below] - hl_min) / denom_lo

    # Above sweet spot
    above = valid & (hl > sweet_high) & (hl <= hl_max)
    denom_hi = hl_max - sweet_high
    if denom_hi > 0:
        score[above] = (hl_max - hl[above]) / denom_hi

    return score


def compute_confidence(
    metrics: pd.DataFrame,
    config: ConfidenceConfig,
) -> pd.Series:
    """Compute confidence score (0-100%) for each bar — vectorized.

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

    adf = metrics["adf_stat"].values if "adf_stat" in metrics.columns else np.full(n, np.nan)
    hurst = metrics["hurst"].values if "hurst" in metrics.columns else np.full(n, np.nan)
    corr = metrics["correlation"].values if "correlation" in metrics.columns else np.full(n, np.nan)
    hl = metrics["half_life"].values if "half_life" in metrics.columns else np.full(n, np.nan)

    # ADF gate: NaN or >= gate → confidence = 0
    gate_fail = np.isnan(adf) | (adf >= config.adf_gate)

    # Linear scores (vectorized)
    adf_span = config.adf_best - config.adf_worst
    s_adf = np.clip((adf - config.adf_worst) / adf_span, 0.0, 1.0) if abs(adf_span) > 1e-12 else np.zeros(n)
    s_adf[np.isnan(adf)] = 0.0

    hurst_span = config.hurst_best - config.hurst_worst
    s_hurst = np.clip((hurst - config.hurst_worst) / hurst_span, 0.0, 1.0) if abs(hurst_span) > 1e-12 else np.zeros(n)
    s_hurst[np.isnan(hurst)] = 0.0

    corr_span = config.corr_best - config.corr_worst
    s_corr = np.clip((corr - config.corr_worst) / corr_span, 0.0, 1.0) if abs(corr_span) > 1e-12 else np.zeros(n)
    s_corr[np.isnan(corr)] = 0.0

    s_hl = _score_halflife_vec(hl, config.hl_min, config.hl_max, config.hl_sweet_low, config.hl_sweet_high)

    # Weighted sum
    confidence = (config.w_adf * s_adf + config.w_hurst * s_hurst
                  + config.w_corr * s_corr + config.w_hl * s_hl) * 100.0
    confidence[gate_fail] = 0.0

    return pd.Series(confidence, index=metrics.index, name="confidence")


# ---------------------------------------------------------------------------
# Numba-compiled confidence filter (entry blocker)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _apply_conf_filter_numba(sig: np.ndarray, confidence: np.ndarray,
                              min_conf: float) -> np.ndarray:
    """Block new entries where confidence < min_conf. Never blocks exits."""
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0 and confidence[t] < min_conf:
            out[t] = 0
        prev = out[t]
    return out


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
    sig = _apply_conf_filter_numba(
        signals.values.astype(np.int8),
        confidence.values,
        config.min_confidence,
    )
    return pd.Series(sig, index=signals.index, name="signal")


# ---------------------------------------------------------------------------
# Numba-compiled time stop filter
# ---------------------------------------------------------------------------

@njit(cache=True)
def apply_time_stop(sig: np.ndarray, max_bars: int) -> np.ndarray:
    """Force exit after max_bars in position. max_bars=0 means no limit.

    After forced exit, stays flat until the original signal naturally returns to 0
    (prevents immediate re-entry on the same signal block).
    """
    if max_bars <= 0:
        return sig
    out = sig.copy()
    bars_in_pos = 0
    forced_flat = False
    for t in range(len(out)):
        if out[t] != 0:
            if forced_flat:
                out[t] = 0
            else:
                bars_in_pos += 1
                if bars_in_pos > max_bars:
                    out[t] = 0
                    forced_flat = True
        else:
            bars_in_pos = 0
            forced_flat = False
    return out


# ---------------------------------------------------------------------------
# Numba-compiled entry window + flat EOD filter (for grid search)
# ---------------------------------------------------------------------------

@njit(cache=True)
def apply_window_filter_numba(sig: np.ndarray, minutes: np.ndarray,
                               entry_start_min: int, entry_end_min: int,
                               flat_min: int) -> np.ndarray:
    """Apply entry window + flat EOD on numpy arrays (numba-compiled)."""
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        m = minutes[t]
        curr = out[t]

        # Force flat at/after flat_time or before entry_start
        if m >= flat_min or m < entry_start_min:
            out[t] = 0
            prev = 0
            continue

        # Block new entries outside [entry_start, entry_end)
        if not (entry_start_min <= m < entry_end_min):
            if prev == 0 and curr != 0:
                out[t] = 0

        prev = out[t]
    return out


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
    """Force signals to 0 outside the trading window [trading_start, trading_end).

    Legacy behavior: blocks entries AND force-closes outside the window.
    For separate entry/flat control, use apply_entry_flat_filter().
    """
    if session is None:
        session = SessionConfig()

    t_start = session.trading_start
    t_end = session.trading_end

    in_window = signals.index.map(lambda ts: t_start <= ts.time() < t_end)
    sig = signals.copy()
    sig[~in_window] = 0
    return sig


def apply_entry_flat_filter(
    signals: pd.Series,
    entry_start: time = time(4, 0),
    entry_end: time = time(14, 0),
    flat_time: time = time(15, 30),
) -> pd.Series:
    """Block new entries outside [entry_start, entry_end), force flat at flat_time.

    Positions opened before entry_end can ride until flat_time.
    At flat_time, all positions are force-closed (signal -> 0).

    Parameters
    ----------
    signals : pd.Series
        Signal array {+1, 0, -1}.
    entry_start : time
        Earliest time for new entries (default 04:00 CT).
    entry_end : time
        Latest time for new entries (default 14:00 CT).
    flat_time : time
        Force all positions flat at this time (default 15:30 CT).
    """
    sig = signals.values.copy()
    idx = signals.index
    prev = 0

    for t in range(len(sig)):
        ts_time = idx[t].time()
        curr = sig[t]

        # Force flat at/after flat_time (and before next day entry_start)
        if ts_time >= flat_time or ts_time < entry_start:
            sig[t] = 0
            prev = 0
            continue

        # Block new entries outside entry window
        if not (entry_start <= ts_time < entry_end):
            # Outside entry window but before flat_time: allow existing positions to ride
            is_entry = (prev == 0) and (curr != 0)
            if is_entry:
                sig[t] = 0

        prev = sig[t]

    return pd.Series(sig, index=signals.index, name="signal")
