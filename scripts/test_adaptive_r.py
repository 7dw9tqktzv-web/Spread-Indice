"""Test adaptive R (EWMA) on top 5 Kalman configs.

Compares r_ewma_span x adaptive_Q combos vs fixed R baseline.
Focus: MaxDD reduction and 2023 yearly breakdown.

Usage:
    python scripts/test_adaptive_r.py
"""

import sys
import time as time_mod
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
# Constants (same as validate_kalman_top.py)
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930  # 15:30 CT

METRICS_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=66, hurst_window=132, halflife_window=66, correlation_window=66),
}

WINDOWS_MAP = {
    "03:00-12:00": (180, 720),
    "04:00-13:00": (240, 780),
    "05:00-12:00": (300, 720),
}

# ======================================================================
# Top 5 Kalman configs (from validate_kalman_top.py)
# ======================================================================

CONFIGS = {
    "K_Sniper": {
        "alpha": 3e-7, "profil": "court",
        "z_entry": 1.8125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 65.0, "window": "05:00-12:00",
    },
    "K_BestPnL": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.375, "z_exit": 0.25, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
    },
    "K_Balanced": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
    },
    "K_Quality": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "04:00-13:00",
    },
    "K_ShortWin": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "05:00-12:00",
    },
}

# ======================================================================
# Adaptive R combos to test
# ======================================================================

R_COMBOS = [
    # (r_ewma_span, adaptive_Q, label)
    (0,    False, "R_fixed (baseline)"),
    (200,  False, "EWMA=200 Q_fixed"),
    (200,  True,  "EWMA=200 Q_adapt"),
    (500,  False, "EWMA=500 Q_fixed"),
    (500,  True,  "EWMA=500 Q_adapt"),
    (1000, False, "EWMA=1000 Q_fixed"),
    (1000, True,  "EWMA=1000 Q_adapt"),
    (2000, False, "EWMA=2000 Q_fixed"),
    (2000, True,  "EWMA=2000 Q_adapt"),
    (5000, False, "EWMA=5000 Q_fixed"),
    (5000, True,  "EWMA=5000 Q_adapt"),
]


# ======================================================================
# Pipeline
# ======================================================================

def run_single(aligned, px_a, px_b, idx, minutes, cfg, r_span, adapt_q):
    """Run one backtest with given config + adaptive R params."""
    est = create_estimator(
        "kalman",
        alpha_ratio=cfg["alpha"],
        r_ewma_span=r_span,
        adaptive_Q=adapt_q,
    )
    hr = est.estimate(aligned)
    beta = hr.beta.values
    zscore = np.ascontiguousarray(hr.zscore.values, dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    metrics_cfg = METRICS_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(
        hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg
    )
    confidence = compute_confidence(metrics, CONF_CFG).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )

    # Compute MaxDD from equity curve
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity - running_max).min())

    # Yearly PnL breakdown
    yearly = {}
    if bt["trades"] > 0:
        entry_bars = bt["trade_entry_bars"]
        pnls = bt["trade_pnls"]
        entry_years = idx[entry_bars].year
        for y in sorted(entry_years.unique()):
            mask = entry_years == y
            yearly[int(y)] = {
                "trades": int(mask.sum()),
                "pnl": float(pnls[mask].sum()),
                "wr": float((pnls[mask] > 0).sum() / mask.sum() * 100) if mask.sum() > 0 else 0,
            }

    # R diagnostics
    r_init = hr.params["R_init"]
    r_final = hr.params["R_final"]
    r_history = hr.diagnostics["R_history"].dropna()
    r_min = float(r_history.min()) if len(r_history) > 0 else r_init
    r_max = float(r_history.max()) if len(r_history) > 0 else r_init
    r_ratio = r_max / r_min if r_min > 0 else 1.0

    return {
        "trades": bt["trades"],
        "pnl": bt["pnl"],
        "pf": bt["profit_factor"],
        "wr": bt["win_rate"],
        "max_dd": max_dd,
        "calmar": float(bt["pnl"] / abs(max_dd)) if max_dd < 0 else 0,
        "r_init": r_init,
        "r_final": r_final,
        "r_ratio": r_ratio,
        "yearly": yearly,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time_mod.time()

    print("Loading NQ_YM 5min data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years_span = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years_span:.1f} years")
    print(f"Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")

    # Track which configs improved MaxDD for yearly breakdown
    improved_combos = []

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n\n{'=' * 130}")
        print(f" {cfg_name} | alpha={cfg['alpha']:.0e} profil={cfg['profil']} "
              f"ze={cfg['z_entry']} zx={cfg['z_exit']} zs={cfg['z_stop']} "
              f"c={cfg['conf']}% win={cfg['window']}")
        print(f"{'=' * 130}")

        print(f"\n  {'R Config':<24} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} "
              f"{'MaxDD':>9} {'Calmar':>7} {'dPF%':>6} {'dDD%':>6} {'R ratio':>8}")
        print(f"  {'-' * 105}")

        baseline = None
        results_for_config = []

        for r_span, adapt_q, label in R_COMBOS:
            res = run_single(aligned, px_a, px_b, idx, minutes, cfg, r_span, adapt_q)
            results_for_config.append((label, r_span, adapt_q, res))

            if baseline is None:
                baseline = res
                dpf = ""
                ddd = ""
            else:
                dpf_val = (res["pf"] / baseline["pf"] - 1) * 100 if baseline["pf"] > 0 else 0
                ddd_val = (res["max_dd"] / baseline["max_dd"] - 1) * 100 if baseline["max_dd"] < 0 else 0
                dpf = f"{dpf_val:>+5.1f}%"
                ddd = f"{ddd_val:>+5.1f}%"
                # Track improvements (MaxDD closer to 0 = better, so ddd_val > 0 means improvement)
                if ddd_val > 2.0:  # at least 2% MaxDD improvement
                    improved_combos.append((cfg_name, label, r_span, adapt_q, res, baseline))

            print(f"  {label:<24} {res['trades']:>5} {res['wr']:>5.1f}% ${res['pnl']:>9,.0f} "
                  f"{res['pf']:>6.2f} ${res['max_dd']:>8,.0f} {res['calmar']:>7.2f} "
                  f"{dpf:>6} {ddd:>6} {res['r_ratio']:>8.1f}x")

    # ================================================================
    # Yearly breakdown for configs where MaxDD improved
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print(f" YEARLY BREAKDOWN — Configs with MaxDD improvement (>2%)")
    print(f"{'=' * 130}")

    if not improved_combos:
        print("\n  Aucune config avec amelioration du MaxDD > 2%.")
        print("  HYPOTHESE INVALIDEE: R adaptatif n'ameliore pas le MaxDD sur ces configs.")
    else:
        # Also show baseline yearly for comparison
        for cfg_name, label, r_span, adapt_q, res, bl in improved_combos:
            dd_improv = (res["max_dd"] / bl["max_dd"] - 1) * 100
            print(f"\n  {cfg_name} | {label} | MaxDD: ${bl['max_dd']:,.0f} -> ${res['max_dd']:,.0f} ({dd_improv:+.1f}%)")

            # Collect all years from both baseline and result
            all_years = sorted(set(list(bl["yearly"].keys()) + list(res["yearly"].keys())))

            print(f"  {'Year':>6}  {'--- Baseline ---':^28}  {'--- Adaptive R ---':^28}  {'Delta PnL':>10}")
            print(f"  {'':>6}  {'Trd':>5} {'PnL':>10} {'WR%':>6}    {'Trd':>5} {'PnL':>10} {'WR%':>6}  {'':>10}")
            print(f"  {'-' * 90}")

            for y in all_years:
                bl_y = bl["yearly"].get(y, {"trades": 0, "pnl": 0, "wr": 0})
                rs_y = res["yearly"].get(y, {"trades": 0, "pnl": 0, "wr": 0})
                delta = rs_y["pnl"] - bl_y["pnl"]
                flag = " <-- TARGET" if y == 2023 else ""
                print(f"  {y:>6}  {bl_y['trades']:>5} ${bl_y['pnl']:>9,.0f} {bl_y['wr']:>5.1f}%"
                      f"    {rs_y['trades']:>5} ${rs_y['pnl']:>9,.0f} {rs_y['wr']:>5.1f}%"
                      f"  ${delta:>+9,.0f}{flag}")

    # ================================================================
    # Even if no MaxDD improvement, show 2023 for all baselines
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print(f" REFERENCE: 2023 PnL across all configs (baseline R fixed)")
    print(f"{'=' * 130}")
    print(f"\n  {'Config':<14} {'2023 Trd':>9} {'2023 PnL':>10} {'2023 WR%':>9} {'Total PnL':>10} {'MaxDD':>9}")
    print(f"  {'-' * 65}")

    for cfg_name, cfg in CONFIGS.items():
        res = run_single(aligned, px_a, px_b, idx, minutes, cfg, 0, False)
        y2023 = res["yearly"].get(2023, {"trades": 0, "pnl": 0, "wr": 0})
        print(f"  {cfg_name:<14} {y2023['trades']:>9} ${y2023['pnl']:>9,.0f} {y2023['wr']:>8.1f}%"
              f" ${res['pnl']:>9,.0f} ${res['max_dd']:>8,.0f}")

    elapsed = time_mod.time() - t_start
    print(f"\n{'=' * 130}")
    print(f" COMPLETE en {elapsed:.0f}s ({len(CONFIGS) * len(R_COMBOS)} backtests)")
    print(f"{'=' * 130}")


if __name__ == "__main__":
    main()
