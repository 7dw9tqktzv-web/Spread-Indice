"""Phase 13c Surgical Grid: time_stop x delta_sl for Config D.

45 configs (9 time_stop x 5 delta_sl), then corr_w sweep on optimal.
For each config: trades, PnL, PF, WR, MaxDD, W/L ratio, worst trade,
max consecutive losses, CPCV, and EXIT TYPE BREAKDOWN.

Usage:
    python scripts/phase13c_surgical_grid.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.signals.filters import apply_time_stop, apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.validation.cpcv import CPCVConfig, run_cpcv
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask

# ======================================================================
# Constants (D's fixed parameters)
# ======================================================================

from src.config.instruments import get_pair_specs, DEFAULT_SLIPPAGE_TICKS

_NQ, _YM = get_pair_specs("NQ", "YM")
MULT_A, MULT_B = _NQ.multiplier, _YM.multiplier
TICK_A, TICK_B = _NQ.tick_size, _YM.tick_size
SLIPPAGE = DEFAULT_SLIPPAGE_TICKS
COMMISSION = _NQ.commission
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

# D's fixed params
OLS_WINDOW = 7000
ADF_WINDOW = 96
ZW = 30
Z_ENTRY = 3.25
DELTA_TP = 2.75   # z_exit = z_entry - delta_tp = 0.50
WINDOW = "02:00-14:00"
ENTRY_START, ENTRY_END = 120, 840

GATE_ADF_THRESH = -2.86
GATE_HURST_THRESH = 0.50
GATE_CORR_THRESH = 0.70
GATE_HURST_WINDOW = 64
GATE_CORR_WINDOW = 24

CPCV_CFG = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)

# ======================================================================
# Surgical grid axes
# ======================================================================

TIME_STOPS = [0, 12, 15, 18, 20, 25, 30, 40, 50]
DELTA_SLS = [1.00, 1.25, 1.50, 1.75, 2.00]
# z_stop = z_entry + delta_sl = [4.25, 4.50, 4.75, 5.00, 5.25]

# Phase 2: corr_w sweep (after picking optimal time_stop)
CORR_WINDOWS = [18, 20, 22, 24, 26, 28, 30]


# ======================================================================
# Helpers
# ======================================================================

def classify_exits(entries, exits, sides, zscore, z_exit, z_stop,
                    time_stop, minutes, flat_min, entry_start_min):
    """Classify each trade exit: Z_EXIT, Z_STOP, TIME_STOP, FLAT."""
    types = []
    for i in range(len(entries)):
        eb, xb = entries[i], exits[i]
        m = minutes[xb]
        if m >= flat_min or m < entry_start_min:
            types.append("FLAT")
            continue
        dur = xb - eb
        if time_stop > 0 and dur >= time_stop:
            types.append("TIME_STOP")
            continue
        z = zscore[xb] if xb < len(zscore) else np.nan
        if not np.isnan(z):
            if sides[i] == 1 and z < -z_stop:
                types.append("Z_STOP")
                continue
            if sides[i] == -1 and z > z_stop:
                types.append("Z_STOP")
                continue
        types.append("Z_EXIT")
    return types


def max_consecutive_losses(pnls):
    max_c = cur = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


def run_config(px_a, px_b, sig_base, beta, zscore, gate_mask, minutes, n,
               time_stop, delta_sl, corr_window=None, spread=None, close_a=None, close_b=None):
    """Run one surgical config. Returns metrics dict or None if < 10 trades."""
    z_exit = round(max(Z_ENTRY - DELTA_TP, 0.0), 4)
    z_stop = round(Z_ENTRY + delta_sl, 4)

    # Regenerate signals with this z_stop
    raw_signals = generate_signals_numba(zscore, Z_ENTRY, z_exit, z_stop)
    sig_ts = apply_time_stop(raw_signals, time_stop)

    # Use custom corr_window gate if specified
    if corr_window is not None and corr_window != GATE_CORR_WINDOW:
        gate_cfg = GateConfig(
            adf_threshold=GATE_ADF_THRESH,
            hurst_threshold=GATE_HURST_THRESH,
            corr_threshold=GATE_CORR_THRESH,
            adf_window=ADF_WINDOW,
            hurst_window=GATE_HURST_WINDOW,
            corr_window=corr_window,
        )
        gm = compute_gate_mask(spread, close_a, close_b, gate_cfg)
    else:
        gm = gate_mask

    sig_gated = apply_gate_filter_numba(sig_ts, gm)
    sig_final = apply_window_filter_numba(
        sig_gated, minutes, ENTRY_START, ENTRY_END, FLAT_MIN
    )

    bt = run_backtest_vectorized(
        px_a, px_b, sig_final, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )

    num = bt["trades"]
    if num < 10:
        return None

    pnls = bt["trade_pnls"]
    entries = bt["trade_entry_bars"]
    exits = bt["trade_exit_bars"]
    sides = bt["trade_sides"]

    # CPCV
    cpcv = run_cpcv(entries, exits, pnls, n, CPCV_CFG)

    # Max drawdown
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity - running_max).min())

    # W/L ratio
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    avg_win = float(winners.mean()) if len(winners) > 0 else 0
    avg_loss = float(losers.mean()) if len(losers) > 0 else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 99.9

    # Exit type breakdown
    exit_types = classify_exits(
        entries, exits, sides, zscore,
        z_exit, z_stop, time_stop,
        minutes, FLAT_MIN, ENTRY_START,
    )
    exit_counts = Counter(exit_types)

    exit_breakdown = {}
    for etype in ["Z_EXIT", "Z_STOP", "TIME_STOP", "FLAT"]:
        cnt = exit_counts.get(etype, 0)
        emask = np.array([et == etype for et in exit_types])
        epnl = pnls[emask]
        exit_breakdown[etype] = {
            "count": cnt,
            "pct": cnt / num * 100,
            "pnl": float(epnl.sum()) if cnt > 0 else 0,
            "avg_pnl": float(epnl.mean()) if cnt > 0 else 0,
            "wr": float((epnl > 0).sum() / cnt * 100) if cnt > 0 else 0,
        }

    return {
        "time_stop": time_stop,
        "delta_sl": delta_sl,
        "z_stop": z_stop,
        "corr_w": corr_window if corr_window else GATE_CORR_WINDOW,
        "trades": num,
        "pnl": round(float(pnls.sum()), 2),
        "pf": round(bt["profit_factor"], 3),
        "wr": round(bt["win_rate"], 1),
        "max_dd": round(max_dd, 2),
        "avg_pnl": round(float(pnls.mean()), 2),
        "wl_ratio": round(wl_ratio, 2),
        "worst_trade": round(float(pnls.min()), 2),
        "max_consec": max_consecutive_losses(pnls),
        "cpcv_med": round(cpcv["median_sharpe"], 4),
        "cpcv_pct_pos": round(cpcv["pct_positive"], 1),
        "exit_breakdown": exit_breakdown,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 110)
    print(" PHASE 13c SURGICAL GRID -- Config D")
    print("=" * 110)

    z_exit = round(max(Z_ENTRY - DELTA_TP, 0.0), 4)
    print(f"\n  Base config: OLS={OLS_WINDOW} ADF_w={ADF_WINDOW} ZW={ZW} {WINDOW}")
    print(f"  z_entry={Z_ENTRY} z_exit={z_exit} (fixed)")
    print(f"  Grid: {len(TIME_STOPS)} time_stops x {len(DELTA_SLS)} delta_sls = "
          f"{len(TIME_STOPS) * len(DELTA_SLS)} configs")
    print(f"  Then: corr_w sweep [{CORR_WINDOWS[0]}-{CORR_WINDOWS[-1]}] on optimal")

    # Load data
    print("\nLoading market data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    n = len(px_a)
    print(f"  {n:,} bars | {idx[0].date()} to {idx[-1].date()}")

    # Compute shared components (OLS + gate mask + zscore -- ONCE)
    print("\nComputing OLS hedge + gate mask + zscore...")
    est = create_estimator("ols_rolling", window=OLS_WINDOW, zscore_window=ZW)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    gate_cfg = GateConfig(
        adf_threshold=GATE_ADF_THRESH,
        hurst_threshold=GATE_HURST_THRESH,
        corr_threshold=GATE_CORR_THRESH,
        adf_window=ADF_WINDOW,
        hurst_window=GATE_HURST_WINDOW,
        corr_window=GATE_CORR_WINDOW,
    )
    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg
    )

    mu = spread.rolling(ZW).mean()
    sigma = spread.rolling(ZW).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    # ======================================================================
    # PHASE 1: time_stop x delta_sl grid (45 configs)
    # ======================================================================
    print("\n" + "=" * 110)
    print(" PHASE 1: TIME_STOP x DELTA_SL GRID")
    print("=" * 110)

    results = []
    for ts in TIME_STOPS:
        for dsl in DELTA_SLS:
            r = run_config(px_a, px_b, None, beta, zscore, gate_mask, minutes, n,
                           ts, dsl)
            if r:
                results.append(r)

    print(f"\n  {len(results)} configs with >= 10 trades\n")

    # Summary table
    header = (f"  {'ts':>3} {'dsl':>5} {'zs':>5} | {'#':>4} {'PnL':>9} {'PF':>6} "
              f"{'WR':>5} {'DD':>8} {'W/L':>5} {'Wrst':>7} {'MC':>3} | "
              f"{'CPCV':>6} {'P+%':>5} | "
              f"{'%ts':>4} {'ts$':>7} {'tsWR':>4}")
    print(header)
    print(f"  {'-' * len(header)}")

    # Reference row (D original: ts=0, dsl=1.50)
    ref = None

    for r in sorted(results, key=lambda x: (x["time_stop"], x["delta_sl"])):
        ts_info = r["exit_breakdown"].get("TIME_STOP", {"pct": 0, "avg_pnl": 0, "wr": 0})
        zs_info = r["exit_breakdown"].get("Z_STOP", {"pct": 0, "count": 0})

        marker = ""
        if r["time_stop"] == 0 and abs(r["delta_sl"] - 1.50) < 0.01:
            marker = " <-- D original"
            ref = r

        print(f"  {r['time_stop']:>3} {r['delta_sl']:>5.2f} {r['z_stop']:>5.2f} | "
              f"{r['trades']:>4} ${r['pnl']:>+8,.0f} {r['pf']:>6.2f} "
              f"{r['wr']:>4.0f}% ${r['max_dd']:>+7,.0f} {r['wl_ratio']:>5.2f} "
              f"${r['worst_trade']:>+6,.0f} {r['max_consec']:>3} | "
              f"{r['cpcv_med']:>6.3f} {r['cpcv_pct_pos']:>4.0f}% | "
              f"{ts_info['pct']:>3.0f}% ${ts_info['avg_pnl']:>+6,.0f} "
              f"{ts_info['wr']:>3.0f}%{marker}")

    # ======================================================================
    # Diagnostic: exit type detail for key configs
    # ======================================================================
    print("\n" + "=" * 110)
    print(" EXIT TYPE DETAIL (key configs)")
    print("=" * 110)

    key_configs = [r for r in results
                   if r["time_stop"] in [0, 12, 20, 25, 30]
                   and abs(r["delta_sl"] - 1.50) < 0.01]

    for r in sorted(key_configs, key=lambda x: x["time_stop"]):
        print(f"\n  ts={r['time_stop']} dsl={r['delta_sl']:.2f} zs={r['z_stop']:.2f} | "
              f"{r['trades']} trades, ${r['pnl']:+,.0f}, PF {r['pf']:.2f}")
        for etype in ["Z_EXIT", "Z_STOP", "TIME_STOP", "FLAT"]:
            info = r["exit_breakdown"].get(etype, {"count": 0})
            if info["count"] > 0:
                print(f"    {etype:>10}: {info['count']:>4} ({info['pct']:>5.1f}%)  "
                      f"PnL ${info['pnl']:>+9,.0f}  Avg ${info['avg_pnl']:>+6,.0f}  "
                      f"WR {info['wr']:.0f}%")

    # ======================================================================
    # Find optimal time_stop (best trade-off)
    # ======================================================================
    print("\n" + "=" * 110)
    print(" OPTIMAL TIME_STOP SELECTION")
    print("=" * 110)

    # Filter to delta_sl=1.50 (D's original stop) for clean comparison
    dsl_150 = [r for r in results if abs(r["delta_sl"] - 1.50) < 0.01]
    dsl_150.sort(key=lambda x: x["time_stop"])

    print(f"\n  Comparing time_stops at delta_sl=1.50 (z_stop=4.75):\n")
    print(f"  {'ts':>3} | {'#':>4} {'PF':>6} {'WR':>5} {'DD':>8} {'W/L':>5} "
          f"{'Worst':>7} {'MC':>3} | {'CPCV':>6} {'P+%':>5} | "
          f"{'%ts_exit':>8} {'ts_avg$':>8} {'ALERT':>10}")

    best_ts = 0
    best_score = -999

    for r in dsl_150:
        ts_info = r["exit_breakdown"].get("TIME_STOP", {"pct": 0, "avg_pnl": 0, "wr": 0, "count": 0})

        # Alert if time_stop is destructive (>15% triggers AND losing)
        alert = ""
        if ts_info["pct"] > 15 and ts_info["avg_pnl"] < -100:
            alert = "DESTRUCTIF"
        elif ts_info["pct"] > 15:
            alert = "FREQUENT"
        elif ts_info["count"] > 0 and ts_info["avg_pnl"] > 0:
            alert = "profitable!"

        # Simple score: PF * (1 - max_dd_penalty) * cpcv_bonus
        # Prefer configs where time_stop isn't destructive
        score = r["pf"] * (1 + r["cpcv_med"])
        if r["worst_trade"] < -2000:
            score *= 0.8
        if ts_info["pct"] > 15 and ts_info["avg_pnl"] < -200:
            score *= 0.7  # penalize destructive time_stop

        if score > best_score:
            best_score = score
            best_ts = r["time_stop"]

        print(f"  {r['time_stop']:>3} | {r['trades']:>4} {r['pf']:>6.2f} "
              f"{r['wr']:>4.0f}% ${r['max_dd']:>+7,.0f} {r['wl_ratio']:>5.2f} "
              f"${r['worst_trade']:>+6,.0f} {r['max_consec']:>3} | "
              f"{r['cpcv_med']:>6.3f} {r['cpcv_pct_pos']:>4.0f}% | "
              f"{ts_info['pct']:>7.1f}% ${ts_info['avg_pnl']:>+7,.0f} "
              f"{alert:>10}")

    print(f"\n  --> Best time_stop (heuristic): {best_ts}")

    # ======================================================================
    # PHASE 2: corr_w sweep at optimal time_stop + delta_sl=1.50
    # ======================================================================
    print("\n" + "=" * 110)
    print(f" PHASE 2: CORR_W SWEEP (time_stop={best_ts}, delta_sl=1.50)")
    print("=" * 110)

    corr_results = []
    for cw in CORR_WINDOWS:
        r = run_config(px_a, px_b, None, beta, zscore, gate_mask, minutes, n,
                       best_ts, 1.50,
                       corr_window=cw, spread=spread,
                       close_a=aligned.df["close_a"],
                       close_b=aligned.df["close_b"])
        if r:
            corr_results.append(r)

    print(f"\n  {'cw':>4} | {'#':>4} {'PnL':>9} {'PF':>6} {'WR':>5} {'DD':>8} "
          f"{'CPCV':>6} {'P+%':>5}")
    for r in corr_results:
        marker = " <-- default" if r["corr_w"] == 24 else ""
        print(f"  {r['corr_w']:>4} | {r['trades']:>4} ${r['pnl']:>+8,.0f} "
              f"{r['pf']:>6.2f} {r['wr']:>4.0f}% ${r['max_dd']:>+7,.0f} "
              f"{r['cpcv_med']:>6.3f} {r['cpcv_pct_pos']:>4.0f}%{marker}")

    # ======================================================================
    # FINAL RECOMMENDATION
    # ======================================================================
    print("\n" + "=" * 110)
    print(" FINAL CONFIG D (SURGICAL)")
    print("=" * 110)

    # Find best corr_w (highest CPCV with PF > 1.5)
    viable_corr = [r for r in corr_results if r["pf"] > 1.5]
    if viable_corr:
        best_corr = max(viable_corr, key=lambda x: x["cpcv_med"])
    else:
        best_corr = next((r for r in corr_results if r["corr_w"] == 24), corr_results[0])

    best = best_corr
    z_exit = round(max(Z_ENTRY - DELTA_TP, 0.0), 4)
    print(f"\n  OLS={OLS_WINDOW} ADF_w={ADF_WINDOW} ZW={ZW} {WINDOW}")
    print(f"  z_entry={Z_ENTRY} z_exit={z_exit} z_stop={best['z_stop']}")
    print(f"  time_stop={best['time_stop']} corr_w={best['corr_w']}")
    print(f"\n  {best['trades']} trades | ${best['pnl']:+,.0f} | PF {best['pf']:.2f} | "
          f"WR {best['wr']:.0f}% | DD ${best['max_dd']:+,.0f}")
    print(f"  W/L {best['wl_ratio']:.2f} | Worst ${best['worst_trade']:+,.0f} | "
          f"MaxConsec {best['max_consec']}")
    print(f"  CPCV median {best['cpcv_med']:.4f} | {best['cpcv_pct_pos']:.0f}% paths+")

    # Exit breakdown
    print(f"\n  Exit breakdown:")
    for etype in ["Z_EXIT", "Z_STOP", "TIME_STOP", "FLAT"]:
        info = best["exit_breakdown"].get(etype, {"count": 0})
        if info["count"] > 0:
            print(f"    {etype:>10}: {info['count']:>4} ({info['pct']:>5.1f}%)  "
                  f"Avg ${info['avg_pnl']:>+6,.0f}  WR {info['wr']:.0f}%")

    print("\n" + "=" * 110)
    print()


if __name__ == "__main__":
    main()
