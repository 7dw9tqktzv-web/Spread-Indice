"""Compare Config D results: 5min engine vs hybrid 1s engine.

Runs the same Config D parameters on both engines and produces a
side-by-side comparison of trades, PnL, PF, WR, MaxDD.
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.backtest.engine_hybrid import run_hybrid_backtest, warmup_hybrid
from src.data.cache import load_aligned_pair_cache
from src.data.loader_1s import load_1s_full
from src.hedge.factory import create_estimator
from src.signals.filters import apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask

# =========================================================================
# Config D parameters
# =========================================================================

CONFIG_D = {
    # OLS
    "ols_window": 7000,
    "zscore_window": 30,
    # Thresholds
    "z_entry": 3.25,
    "z_exit": 0.50,
    "z_stop": 4.75,
    "z_cooldown": 1.50,    # 5min z must return below 1.50 before re-entry
    # Window
    "entry_start_min": 120,    # 02:00
    "entry_end_min": 840,      # 14:00
    "flat_min": 930,           # 15:30
    # Gates
    "gate_adf": -2.86,
    "gate_hurst": 0.50,
    "gate_corr": 0.70,
    "gate_adf_window": 96,
    "gate_hurst_window": 64,
    "gate_corr_window": 24,
    # Instrument specs
    "mult_a": 20.0,
    "mult_b": 5.0,
    "tick_a": 0.25,
    "tick_b": 1.0,
    "slippage": 1,
    "commission": 2.50,
    # Dollar exits (disabled for baseline)
    "dollar_tp": 0,
    "dollar_sl": 0,
}


# =========================================================================
# Step 1: Run baseline 5min backtest
# =========================================================================

def run_baseline_5min(aligned):
    """Run Config D on the standard 5min vectorized engine."""
    print("\n" + "=" * 70)
    print("  BASELINE: 5min Engine (Config D)")
    print("=" * 70)

    t0 = time_mod.time()
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    est = create_estimator("ols_rolling", window=7000, zscore_window=30)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    gate_cfg = GateConfig(
        adf_threshold=-2.86, hurst_threshold=0.50, corr_threshold=0.70,
        adf_window=96, hurst_window=64, corr_window=24,
    )
    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg
    )

    mu = spread.rolling(30).mean()
    sigma = spread.rolling(30).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw = generate_signals_numba(zscore, 3.25, 0.50, 4.75)
    sig_gated = apply_gate_filter_numba(raw, gate_mask)
    sig_final = apply_window_filter_numba(sig_gated, minutes, 120, 840, 930)

    bt = run_backtest_vectorized(
        px_a, px_b, sig_final, beta,
        20.0, 5.0, 0.25, 1.0, 1, 2.50, 100_000.0,
    )

    elapsed = time_mod.time() - t0

    # Max drawdown on cumulative PnL
    pnl = bt["trade_pnls"]
    cum_pnl = np.cumsum(pnl) if len(pnl) > 0 else np.array([0.0])
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float((cum_pnl - peak).min()) if len(pnl) > 0 else 0.0

    print(f"  Trades:  {bt['trades']}")
    print(f"  WR:      {bt['win_rate']:.1f}%")
    print(f"  PnL:     ${bt['pnl']:+,.0f}")
    print(f"  PF:      {bt['profit_factor']:.2f}")
    print(f"  MaxDD:   ${max_dd:+,.0f}")
    print(f"  AvgPnL:  ${bt['avg_pnl_trade']:+,.0f}")
    print(f"  Time:    {elapsed:.1f}s")

    # Build trades list for comparison
    entries = bt["trade_entry_bars"]
    exits = bt["trade_exit_bars"]
    sides = bt["trade_sides"]
    pnls = bt["trade_pnls"]
    n = bt["trades"]

    trades_5min = []
    for k in range(n):
        trades_5min.append({
            "entry_time": idx[entries[k]],
            "exit_time": idx[exits[k]],
            "side": int(sides[k]),
            "pnl_net": float(pnls[k]),
            "duration_bars": int(exits[k] - entries[k]),
        })

    return bt, trades_5min, max_dd


# =========================================================================
# Step 2: Run hybrid 1s backtest
# =========================================================================

def run_hybrid_1s(aligned, data_1s):
    """Run Config D on the hybrid 1s engine."""
    print("\n" + "=" * 70)
    print("  HYBRID: 1s Engine (Config D)")
    print("=" * 70)

    # Warmup numba
    print("  Warming up numba kernel...", end="", flush=True)
    t0 = time_mod.time()
    warmup_hybrid()
    print(f" {time_mod.time() - t0:.1f}s")

    # Run
    t0 = time_mod.time()
    result = run_hybrid_backtest(aligned, data_1s, CONFIG_D)
    elapsed = time_mod.time() - t0

    print(f"  Trades:  {result['trades']}")
    print(f"  WR:      {result['win_rate']:.1f}%")
    print(f"  PnL:     ${result['pnl']:+,.0f}")
    print(f"  PF:      {result['profit_factor']:.2f}")
    print(f"  MaxDD:   ${result['max_drawdown']:+,.0f}")
    print(f"  AvgPnL:  ${result['avg_pnl_trade']:+,.0f}")
    print(f"  Time:    {elapsed:.1f}s")

    return result


# =========================================================================
# Step 3: Compare
# =========================================================================

def compare_results(bt_5min, trades_5min, dd_5min, result_1s):
    """Compare 5min vs 1s results."""
    print("\n" + "=" * 70)
    print("  COMPARISON: 5min vs 1s")
    print("=" * 70)

    df_1s = result_1s["trades_df"]

    # Summary table
    print(f"\n{'Metric':<20} {'5min':>12} {'1s':>12} {'Delta':>12}")
    print("-" * 56)

    metrics = [
        ("Trades", bt_5min["trades"], result_1s["trades"]),
        ("Win Rate %", bt_5min["win_rate"], result_1s["win_rate"]),
        ("PnL $", bt_5min["pnl"], result_1s["pnl"]),
        ("Profit Factor", bt_5min["profit_factor"], result_1s["profit_factor"]),
        ("Avg PnL/trade $", bt_5min["avg_pnl_trade"], result_1s["avg_pnl_trade"]),
        ("Max Drawdown $", dd_5min, result_1s["max_drawdown"]),
    ]

    for name, v5, v1 in metrics:
        delta = v1 - v5
        if isinstance(v5, int):
            print(f"  {name:<18} {v5:>12} {v1:>12} {delta:>+12}")
        elif abs(v5) >= 100:
            print(f"  {name:<18} {v5:>12,.0f} {v1:>12,.0f} {delta:>+12,.0f}")
        else:
            print(f"  {name:<18} {v5:>12.2f} {v1:>12.2f} {delta:>+12.2f}")

    # Exit reason distribution
    if len(df_1s) > 0:
        print(f"\n  Exit Reasons (1s engine):")
        for reason, count in df_1s["exit_reason"].value_counts().items():
            pct = count / len(df_1s) * 100
            avg_pnl = df_1s[df_1s["exit_reason"] == reason]["pnl_net"].mean()
            print(f"    {reason:<15} {count:>4} ({pct:>5.1f}%)  avg PnL ${avg_pnl:+,.0f}")

    # MFE/MAE stats
    if len(df_1s) > 0 and "mfe" in df_1s.columns:
        print(f"\n  MFE/MAE (1s engine):")
        print(f"    MFE median: ${df_1s['mfe'].median():+,.0f}")
        print(f"    MAE median: ${df_1s['mae'].median():+,.0f}")
        print(f"    MFE mean:   ${df_1s['mfe'].mean():+,.0f}")
        print(f"    MAE mean:   ${df_1s['mae'].mean():+,.0f}")

    # Trade-by-trade timing comparison
    if len(trades_5min) > 0 and len(df_1s) > 0:
        print(f"\n  Timing Analysis:")
        # Match trades by entry date (same day, same side)
        matched = 0
        earlier_entry = 0
        later_entry = 0
        entry_deltas = []

        for t5 in trades_5min:
            date_5 = t5["entry_time"].date()
            side_5 = t5["side"]
            # Find matching 1s trade (same date, same side, closest entry time)
            candidates = df_1s[
                (df_1s["entry_time"].dt.date == date_5)
                & (df_1s["side"] == side_5)
            ]
            if len(candidates) == 0:
                continue
            # Closest entry time
            deltas = (candidates["entry_time"] - t5["entry_time"]).abs()
            best_idx = deltas.idxmin()
            best = candidates.loc[best_idx]
            delta_sec = (best["entry_time"] - t5["entry_time"]).total_seconds()
            entry_deltas.append(delta_sec)
            matched += 1
            if delta_sec < -5:
                earlier_entry += 1
            elif delta_sec > 5:
                later_entry += 1

        if matched > 0:
            arr = np.array(entry_deltas)
            print(f"    Matched trades:   {matched}/{len(trades_5min)}")
            print(f"    1s enters EARLIER: {earlier_entry} trades")
            print(f"    1s enters LATER:   {later_entry} trades")
            print(f"    Entry delta (sec): median={np.median(arr):+.0f}s, "
                  f"mean={np.mean(arr):+.0f}s, std={np.std(arr):.0f}s")

    return


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("  Config D Comparison: 5min vs Hybrid 1s Engine")
    print("=" * 70)

    # Load data
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)

    print("\n  Loading 5min data...", flush=True)
    aligned = load_aligned_pair_cache(pair, "5min")
    print(f"  {len(aligned.df):,} bars")

    print("\n  Loading 1s data...", flush=True)
    data_1s = load_1s_full(pair)
    print(f"  {len(data_1s):,} rows")

    # Run both engines
    bt_5min, trades_5min, dd_5min = run_baseline_5min(aligned)
    result_1s = run_hybrid_1s(aligned, data_1s)

    # Compare
    compare_results(bt_5min, trades_5min, dd_5min, result_1s)

    # Export trades
    output_dir = Path("output/NQ_YM")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_1s = result_1s["trades_df"]
    if len(df_1s) > 0:
        csv_path = output_dir / "hybrid_1s_trades.csv"
        df_1s.to_csv(csv_path, index=False)
        print(f"\n  1s trades exported to {csv_path}")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
