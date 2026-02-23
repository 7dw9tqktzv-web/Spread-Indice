"""Etape 1 — Exploration rapide NQ/RTY et ES/RTY (OLS + Kalman).

Runs multiple configs per pair and reports results + long/short analysis.

Usage:
    python scripts/explore_pairs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.generator import generate_signals_numba
from src.signals.filters import (
    ConfidenceConfig, compute_confidence,
    _apply_conf_filter_numba, apply_window_filter_numba,
)
from src.backtest.engine import run_backtest_vectorized


SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930  # 15:30 CT
CONF_CFG = ConfidenceConfig()

METRICS_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
}

# ======================================================================
# Configs to test
# ======================================================================

OLS_CONFIGS = [
    {"label": "OLS tight",   "window": 2640, "zw": 30, "ze": 3.0,  "zx": 1.5, "zs": 4.0, "conf": 50},
    {"label": "OLS medium",  "window": 2640, "zw": 30, "ze": 2.5,  "zx": 1.0, "zs": 3.5, "conf": 60},
    {"label": "OLS loose",   "window": 3960, "zw": 30, "ze": 3.0,  "zx": 1.0, "zs": 4.5, "conf": 50},
    {"label": "OLS wide",    "window": 1980, "zw": 36, "ze": 2.5,  "zx": 1.0, "zs": 3.5, "conf": 50},
    {"label": "OLS high_c",  "window": 2640, "zw": 30, "ze": 3.0,  "zx": 1.0, "zs": 4.0, "conf": 67},
    {"label": "OLS short_w", "window": 1320, "zw": 24, "ze": 2.5,  "zx": 1.0, "zs": 3.5, "conf": 50},
]

KALMAN_CONFIGS = [
    {"label": "K medium",    "alpha": 3e-7,  "ze": 1.5,  "zx": 0.5,  "zs": 2.5, "conf": 50},
    {"label": "K tight",     "alpha": 1e-7,  "ze": 2.0,  "zx": 0.5,  "zs": 2.75, "conf": 60},
    {"label": "K balanced",  "alpha": 3e-7,  "ze": 1.5,  "zx": 0.25, "zs": 2.75, "conf": 70},
    {"label": "K loose",     "alpha": 5e-7,  "ze": 1.25, "zx": 0.5,  "zs": 2.5,  "conf": 50},
    {"label": "K high_c",    "alpha": 3e-7,  "ze": 1.375,"zx": 0.25, "zs": 2.75, "conf": 75},
    {"label": "K sniper",    "alpha": 1.5e-7,"ze": 2.0,  "zx": 0.75, "zs": 2.75, "conf": 65},
]

PAIRS = [
    ("NQ_RTY", Instrument.NQ, Instrument.RTY, 20.0, 50.0, 0.25, 0.10),
    ("ES_RTY", Instrument.ES, Instrument.RTY, 50.0, 50.0, 0.25, 0.10),
]

# Trading window: 02:00-14:00 CT
ENTRY_START_MIN = 120
ENTRY_END_MIN = 840


def run_exploration(pair_name, leg_a, leg_b, mult_a, mult_b, tick_a, tick_b):
    """Run all configs for one pair."""
    print(f"\n{'='*120}")
    print(f" EXPLORATION: {pair_name}")
    print(f"{'='*120}")

    pair = SpreadPair(leg_a=leg_a, leg_b=leg_b)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print(f"  ERREUR: pas de cache pour {pair_name}")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years = (idx[-1] - idx[0]).days / 365.25

    header = (f"  {'Config':<16} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} "
              f"{'Avg$':>7} {'AvgD':>5} {'Trd/y':>5} | "
              f"{'Long$':>9} {'Short$':>9} {'L/S%':>6}")
    sep = "  " + "-" * 110

    # ---- OLS ----
    print(f"\n  --- OLS ---")
    print(header)
    print(sep)

    for cfg in OLS_CONFIGS:
        est = create_estimator("ols_rolling", window=cfg["window"], zscore_window=cfg["zw"])
        hr = est.estimate(aligned)
        spread = hr.spread
        beta = hr.beta.values

        # zscore
        mu = spread.rolling(cfg["zw"]).mean()
        sigma = spread.rolling(cfg["zw"]).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

        # metrics + confidence
        profile_cfg = METRICS_PROFILES["tres_court"]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, CONF_CFG).values

        # signals
        raw = generate_signals_numba(zscore, cfg["ze"], cfg["zx"], cfg["zs"])
        sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
        sig = apply_window_filter_numba(sig, minutes, ENTRY_START_MIN, ENTRY_END_MIN, FLAT_MIN)

        # backtest
        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            mult_a, mult_b, tick_a, tick_b,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        _print_result(cfg["label"], bt, years)

    # ---- Kalman ----
    print(f"\n  --- Kalman ---")
    print(header)
    print(sep)

    for cfg in KALMAN_CONFIGS:
        est = create_estimator("kalman", alpha_ratio=cfg["alpha"], warmup=200, gap_P_multiplier=5.0)
        hr = est.estimate(aligned)
        beta = hr.beta.values
        zscore = np.ascontiguousarray(
            np.nan_to_num(hr.zscore.values, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float64,
        )

        # metrics + confidence
        profile_cfg = METRICS_PROFILES["tres_court"]
        metrics = compute_all_metrics(hr.spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, CONF_CFG).values

        # signals
        raw = generate_signals_numba(zscore, cfg["ze"], cfg["zx"], cfg["zs"])
        sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
        sig = apply_window_filter_numba(sig, minutes, ENTRY_START_MIN, ENTRY_END_MIN, FLAT_MIN)

        # backtest
        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            mult_a, mult_b, tick_a, tick_b,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        _print_result(cfg["label"], bt, years)


def _print_result(label, bt, years):
    """Print one result line with long/short analysis."""
    n = bt["trades"]
    if n == 0:
        print(f"  {label:<16} {'0':>5} {'--':>6} {'--':>10} {'--':>6} {'--':>7} {'--':>5} {'--':>5} |")
        return

    pnl = bt["pnl"]
    pf = bt["profit_factor"]
    wr = bt["win_rate"]
    avg_pnl = bt["avg_pnl_trade"]
    avg_dur = bt["avg_duration_bars"]
    trd_y = n / years if years > 0 else 0

    # Long/short analysis from trade data
    trade_pnls = bt["trade_pnls"]
    trade_sides = bt.get("trade_sides", None)

    long_pnl = 0
    short_pnl = 0
    if trade_sides is not None:
        long_mask = trade_sides == 1
        short_mask = trade_sides == -1
        long_pnl = float(trade_pnls[long_mask].sum()) if long_mask.any() else 0
        short_pnl = float(trade_pnls[short_mask].sum()) if short_mask.any() else 0
    else:
        long_pnl = pnl / 2
        short_pnl = pnl / 2

    total_abs = abs(long_pnl) + abs(short_pnl)
    long_pct = (long_pnl / total_abs * 100) if total_abs > 0 else 50

    flag = ""
    if pnl > 0 and abs(long_pct) > 80:
        flag = " **BIAS"
    elif pnl > 0 and abs(long_pct) > 60:
        flag = " *bias"

    print(
        f"  {label:<16} {n:>5} {wr:>5.1f}% ${pnl:>9,.0f} {pf:>6.2f} "
        f"${avg_pnl:>6,.0f} {avg_dur:>5.1f} {trd_y:>5.1f} | "
        f"${long_pnl:>8,.0f} ${short_pnl:>8,.0f} {long_pct:>5.1f}%{flag}"
    )


if __name__ == "__main__":
    for pair_name, leg_a, leg_b, mult_a, mult_b, tick_a, tick_b in PAIRS:
        run_exploration(pair_name, leg_a, leg_b, mult_a, mult_b, tick_a, tick_b)
