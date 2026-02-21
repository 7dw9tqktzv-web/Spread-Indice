"""Compare Config E across standard vs micro contracts, multipliers, and dollar stops.

Usage:
    python scripts/compare_micro_sizing.py
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

# ======================================================================
# Config E parameters (fixed)
# ======================================================================
OLS_WINDOW = 3300
ZW_WINDOW = 30
Z_ENTRY = 3.15
Z_EXIT = 1.00
Z_STOP = 4.50
MIN_CONF = 67.0
ENTRY_START_MIN = 120   # 02:00 CT
ENTRY_END_MIN = 840     # 14:00 CT
FLAT_MIN = 930          # 15:30 CT
METRICS_CFG = MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6)
CONF_CFG = ConfidenceConfig()

# ======================================================================
# Instrument specs
# ======================================================================
CONFIGS = [
    {"name": "NQ/YM x1 (baseline)", "mult_a": 20.0, "mult_b": 5.0,  "tick_a": 0.25, "tick_b": 1.0, "comm": 2.50, "max_mult": 1, "dollar_stop": 0.0},
    {"name": "NQ/YM x2",            "mult_a": 20.0, "mult_b": 5.0,  "tick_a": 0.25, "tick_b": 1.0, "comm": 2.50, "max_mult": 2, "dollar_stop": 0.0},
    {"name": "NQ/YM x2 SL$300",     "mult_a": 20.0, "mult_b": 5.0,  "tick_a": 0.25, "tick_b": 1.0, "comm": 2.50, "max_mult": 2, "dollar_stop": 300.0},
    {"name": "NQ/YM x2 SL$500",     "mult_a": 20.0, "mult_b": 5.0,  "tick_a": 0.25, "tick_b": 1.0, "comm": 2.50, "max_mult": 2, "dollar_stop": 500.0},
    {"name": "NQ/YM x2 SL$800",     "mult_a": 20.0, "mult_b": 5.0,  "tick_a": 0.25, "tick_b": 1.0, "comm": 2.50, "max_mult": 2, "dollar_stop": 800.0},
    {"name": "NQ/YM x2 SL$1000",    "mult_a": 20.0, "mult_b": 5.0,  "tick_a": 0.25, "tick_b": 1.0, "comm": 2.50, "max_mult": 2, "dollar_stop": 1000.0},
    {"name": "NQ/YM x3",            "mult_a": 20.0, "mult_b": 5.0,  "tick_a": 0.25, "tick_b": 1.0, "comm": 2.50, "max_mult": 3, "dollar_stop": 0.0},
]


def main():
    # Load data (NQ/YM — same prices for micros)
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    print(f"Data: {n:,} bars, {idx[0]} -> {idx[-1]}")
    years = (idx[-1] - idx[0]).days / 365.25

    # Precompute minutes array
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # Hedge ratio
    est = create_estimator("ols_rolling", window=OLS_WINDOW, zscore_window=ZW_WINDOW)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    # Metrics + confidence
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], METRICS_CFG)
    confidence = compute_confidence(metrics, CONF_CFG).values

    # Z-score
    mu = spread.rolling(ZW_WINDOW).mean()
    sigma = spread.rolling(ZW_WINDOW).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    # Signals (same for all configs)
    raw_signals = generate_signals_numba(zscore, Z_ENTRY, Z_EXIT, Z_STOP)
    sig_conf = _apply_conf_filter_numba(raw_signals, confidence, MIN_CONF)
    sig_final = apply_window_filter_numba(sig_conf, minutes, ENTRY_START_MIN, ENTRY_END_MIN, FLAT_MIN)

    # Run each config
    results = []
    print(f"\n{'='*130}")
    print(f" COMPARAISON CONFIG E — Standard vs Micro vs Dollar Stop")
    print(f"{'='*130}\n")

    for cfg in CONFIGS:
        bt = run_backtest_vectorized(
            px_a, px_b, sig_final, beta,
            mult_a=cfg["mult_a"], mult_b=cfg["mult_b"],
            tick_a=cfg["tick_a"], tick_b=cfg["tick_b"],
            slippage_ticks=1, commission=cfg["comm"],
            initial_capital=100_000.0,
            max_multiplier=cfg["max_mult"],
            dollar_stop=cfg["dollar_stop"],
        )

        num = bt["trades"]
        if num == 0:
            results.append({"name": cfg["name"], "trades": 0, "wr": 0, "pnl": 0,
                           "pf": 0, "sharpe": 0, "calmar": 0, "max_dd": 0,
                           "max_dd_pct": 0, "avg_pnl": 0, "trd_an": 0, "avg_dur": 0})
            continue

        # Compute Sharpe/Calmar from equity curve
        eq = bt["equity"]
        bars_per_year = 264 * 252

        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.diff(eq) / eq[:-1]
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(bars_per_year) if np.std(rets) > 0 else 0.0

        running_max = np.maximum.accumulate(eq)
        dd = (eq - running_max)
        max_dd = float(dd.min())
        max_dd_pct = abs(float((dd / running_max).min())) * 100

        total_return = (eq[-1] - eq[0]) / eq[0] if eq[0] != 0 else 0.0
        ann_return = (1 + total_return) ** (bars_per_year / max(len(eq), 1)) - 1
        calmar = (ann_return * 100) / max_dd_pct if max_dd_pct > 0 else 0.0

        trd_an = num / years if years > 0 else 0

        results.append({
            "name": cfg["name"],
            "trades": num,
            "wr": bt["win_rate"],
            "pnl": bt["pnl"],
            "pf": bt["profit_factor"],
            "avg_pnl": bt["avg_pnl_trade"],
            "sharpe": float(sharpe),
            "calmar": float(calmar),
            "max_dd": max_dd,
            "max_dd_pct": max_dd_pct,
            "trd_an": round(trd_an, 1),
            "avg_dur": bt["avg_duration_bars"],
        })

    # Print results table
    header = (f" {'Config':<22} {'Trd':>5} {'WR%':>5} {'PnL':>10} {'PF':>5} "
              f"{'Avg$':>7} {'Sharpe':>6} {'Calmar':>6} {'MaxDD':>9} {'DD%':>5} "
              f"{'Trd/An':>6} {'Dur':>4}")
    print(header)
    print(" " + "-" * 125)

    for r in results:
        if r["trades"] == 0:
            print(f" {r['name']:<22}   --- no trades ---")
            continue
        print(
            f" {r['name']:<22} {r['trades']:>5} {r['wr']:>5.1f} "
            f"${r['pnl']:>9,.0f} {r['pf']:>5.2f} "
            f"${r['avg_pnl']:>6,.0f} {r['sharpe']:>6.2f} {r['calmar']:>6.2f} "
            f"${r['max_dd']:>8,.0f} {r['max_dd_pct']:>5.1f}% "
            f"{r['trd_an']:>6.1f} {r['avg_dur']:>4.1f}"
        )

    # Hedge error analysis for micro configs
    print(f"\n{'='*80}")
    print(" ANALYSE ERREUR HEDGE (micros)")
    print(f"{'='*80}\n")

    from src.sizing.position import find_optimal_multiplier

    # Compute n_b_raw at each entry
    entries_mask = np.diff(np.concatenate([[0], sig_final])) != 0
    entries_mask = entries_mask & (sig_final != 0)
    entry_bars = np.where(entries_mask)[0]

    if len(entry_bars) > 0:
        for max_m in [1, 2, 3]:
            errors = []
            for eb in entry_bars:
                b = abs(beta[eb]) if np.isfinite(beta[eb]) else 1.0
                # NQ/YM standard: n_b_raw
                nb_raw_std = (px_a[eb] * 20.0) / (px_b[eb] * 5.0) * b
                # Micro: n_b_raw
                nb_raw_micro = (px_a[eb] * 2.0) / (px_b[eb] * 0.5) * b

                _, _, _, err = find_optimal_multiplier(nb_raw_micro, max_m)
                errors.append(err)

            errors = np.array(errors)
            print(f" max_multiplier={max_m}: mean error={errors.mean():.2f}%, "
                  f"median={np.median(errors):.2f}%, max={errors.max():.2f}%")

    print(f"\n Resultats sauvegardes.")


if __name__ == "__main__":
    main()
