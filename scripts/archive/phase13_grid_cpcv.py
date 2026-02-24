"""Phase 13 Grid: ~10,000 configs with binary gates + CPCV(10,2).

NQ_YM OLS, binary gates (ADF < -2.86, Hurst < 0.50, Corr > 0.70).
Each config: run_backtest_vectorized ONCE, then CPCV 45 paths via trade filtering.

Usage:
    python scripts/phase13_grid_cpcv.py --dry-run
    python scripts/phase13_grid_cpcv.py --workers 4
"""

import argparse
import sys
import time as time_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
from src.signals.generator import generate_signals_numba
from src.signals.filters import apply_time_stop, apply_window_filter_numba
from src.backtest.engine import run_backtest_vectorized
from src.validation.gates import GateConfig, compute_gate_mask, apply_gate_filter_numba
from src.validation.cpcv import CPCVConfig, run_cpcv

# ======================================================================
# Constants
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0  # NQ, YM
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930  # 15:30 CT

# Gate config (fixed by theory, NOT in grid)
GATE_CFG = GateConfig()  # adf=-2.86, hurst=0.50, corr=0.70, windows 24/64/24

# CPCV config
CPCV_CFG = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)

# ======================================================================
# Grid parameters
# ======================================================================

OLS_WINDOWS = [2000, 3000, 4000, 5000]
ZW_WINDOWS = [15, 20, 30, 40, 50]
Z_ENTRIES = [2.25, 2.50, 2.75, 3.00, 3.25]
Z_EXITS = [0.25, 0.50, 0.75, 1.00, 1.25]
Z_STOPS = [3.50, 4.00, 4.50, 5.00]
TIME_STOPS = [0, 6, 12, 20]

# Trading windows: (name, entry_start_min, entry_end_min)
WINDOWS = [
    ("02:00-14:00", 120, 840),
    ("06:00-14:00", 360, 840),
]


def print_grid_summary():
    """Print grid configuration summary and return total combo count."""
    n_ols = len(OLS_WINDOWS)
    n_zw = len(ZW_WINDOWS)
    n_entry = len(Z_ENTRIES)
    n_exit = len(Z_EXITS)
    n_stop = len(Z_STOPS)
    n_ts = len(TIME_STOPS)
    n_win = len(WINDOWS)

    valid_signal = 0
    for ze, zx, zs in product(Z_ENTRIES, Z_EXITS, Z_STOPS):
        if zx < ze and zs > ze:
            valid_signal += 1

    n_inner = valid_signal * n_ts * n_win
    n_pipeline = n_ols * n_zw
    total = n_pipeline * n_inner

    print("=" * 100)
    print(" PHASE 13 — BINARY GATES + CPCV(10,2) — NQ_YM 5min")
    print("=" * 100)
    print()
    print(f" OLS windows:     {OLS_WINDOWS}  ({n_ols})")
    print(f" ZW windows:      {ZW_WINDOWS}  ({n_zw})")
    print(f" z_entry:         {Z_ENTRIES}  ({n_entry})")
    print(f" z_exit:          {Z_EXITS}  ({n_exit})")
    print(f" z_stop:          {Z_STOPS}  ({n_stop})")
    print(f" time_stop:       {TIME_STOPS}  ({n_ts})")
    print(f" windows:         {[w[0] for w in WINDOWS]}  ({n_win})")
    print(f" flat:            15:30 CT")
    print()
    print(f" Gates (fixed):   ADF < {GATE_CFG.adf_threshold}, "
          f"Hurst < {GATE_CFG.hurst_threshold}, Corr > {GATE_CFG.corr_threshold}")
    print(f" Gate windows:    ADF={GATE_CFG.adf_window}, "
          f"Hurst={GATE_CFG.hurst_window}, Corr={GATE_CFG.corr_window}")
    print(f" CPCV:            {CPCV_CFG.n_folds} folds, {CPCV_CFG.n_test_folds} test, "
          f"purge={CPCV_CFG.purge_bars} bars, 45 paths")
    print()
    print(f" Pipeline combos (OLS x ZW):       {n_pipeline}")
    print(f" Valid signal combos (e x x x s):  {valid_signal} / {n_entry * n_exit * n_stop}")
    print(f" Inner combos per pipeline:        {n_inner}")
    print(f" TOTAL COMBOS:                     {total:,}")
    print("=" * 100)
    return total


# ======================================================================
# Worker function (one per OLS window)
# ======================================================================

def run_ols_batch(ols_window):
    """Process one OLS window: compute hedge + gate mask once, then sweep params."""
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # OLS hedge ratio (depends only on ols_window)
    est = create_estimator("ols_rolling", window=ols_window, zscore_window=ZW_WINDOWS[0])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    # Binary gate mask (computed ONCE per OLS window)
    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], GATE_CFG
    )

    years = (idx[-1] - idx[0]).days / 365.25
    results = []
    count = 0

    for zw in ZW_WINDOWS:
        # Recompute zscore for this ZW
        mu = spread.rolling(zw).mean()
        sigma = spread.rolling(zw).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(zscore, dtype=np.float64)

        for z_entry in Z_ENTRIES:
            for z_exit in Z_EXITS:
                if z_exit >= z_entry:
                    continue
                for z_stop in Z_STOPS:
                    if z_stop <= z_entry:
                        continue

                    raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)

                    for ts in TIME_STOPS:
                        sig_ts = apply_time_stop(raw_signals, ts)
                        sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)

                        for win_name, entry_start, entry_end in WINDOWS:
                            sig_final = apply_window_filter_numba(
                                sig_gated, minutes, entry_start, entry_end, FLAT_MIN
                            )

                            bt = run_backtest_vectorized(
                                px_a, px_b, sig_final, beta,
                                MULT_A, MULT_B, TICK_A, TICK_B,
                                SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
                            )

                            count += 1
                            num = bt["trades"]
                            if num < 10:
                                continue

                            # CPCV: 45 paths via trade filtering
                            cpcv = run_cpcv(
                                bt["trade_entry_bars"],
                                bt["trade_exit_bars"],
                                bt["trade_pnls"],
                                n,
                                CPCV_CFG,
                            )

                            # Max drawdown from equity
                            equity = bt["equity"]
                            running_max = np.maximum.accumulate(equity)
                            max_dd = float((equity - running_max).min())

                            # Full-sample Sharpe (for reference)
                            pnls = bt["trade_pnls"]
                            sharpe_full = 0.0
                            if pnls.std() > 1e-12:
                                sharpe_full = float(pnls.mean() / pnls.std())

                            results.append({
                                "ols": ols_window,
                                "zw": zw,
                                "window": win_name,
                                "z_entry": z_entry,
                                "z_exit": z_exit,
                                "z_stop": z_stop,
                                "time_stop": ts,
                                "trades": num,
                                "win_rate": bt["win_rate"],
                                "pnl": round(bt["pnl"], 2),
                                "pf": round(bt["profit_factor"], 3),
                                "avg_pnl": round(bt["avg_pnl_trade"], 2),
                                "avg_dur": round(bt["avg_duration_bars"], 1),
                                "max_dd": round(max_dd, 2),
                                "sharpe_full": round(sharpe_full, 4),
                                "cpcv_median_sharpe": round(cpcv["median_sharpe"], 4),
                                "cpcv_mean_sharpe": round(cpcv["mean_sharpe"], 4),
                                "cpcv_std_sharpe": round(cpcv["std_sharpe"], 4),
                                "cpcv_min_sharpe": round(cpcv["min_sharpe"], 4),
                                "cpcv_pct_positive": round(cpcv["pct_positive"], 1),
                                "cpcv_valid_paths": cpcv["n_valid_paths"],
                                "trd_an": round(num / years, 1) if years > 0 else 0,
                            })

    return ols_window, results, count


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 13 Grid + CPCV")
    parser.add_argument("--dry-run", action="store_true", help="Show grid summary only")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    total = print_grid_summary()

    if args.dry_run:
        print("\n  --dry-run: exiting without running.\n")
        return

    print(f"\nStarting grid with {args.workers} workers...\n")

    t0 = time_mod.time()
    all_results = []
    total_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_ols_batch, ols_w): ols_w for ols_w in OLS_WINDOWS}
        for future in as_completed(futures):
            ols_w = futures[future]
            try:
                ols_w_ret, results, count = future.result()
                total_count += count
                all_results.extend(results)
                elapsed = time_mod.time() - t0
                print(f"  OLS={ols_w_ret}: {count:,} combos evaluated, "
                      f"{len(results)} stored (trades>=10), {elapsed:.0f}s")
            except Exception as e:
                print(f"  OLS={ols_w} FAILED: {e}")

    elapsed = time_mod.time() - t0
    print(f"\nTotal: {total_count:,} combos in {elapsed:.0f}s "
          f"({total_count / elapsed:.0f} combos/s)")

    if not all_results:
        print("No results with trades >= 10.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values("cpcv_median_sharpe", ascending=False)

    output_path = PROJECT_ROOT / "output" / "NQ_YM" / "phase13_grid_cpcv.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} results to {output_path}")

    # Display top 20
    print(f"\n{'='*120}")
    print(f" TOP 20 BY CPCV MEDIAN SHARPE")
    print(f"{'='*120}")
    cols = ["ols", "zw", "window", "z_entry", "z_exit", "z_stop", "time_stop",
            "trades", "pnl", "pf", "max_dd", "sharpe_full",
            "cpcv_median_sharpe", "cpcv_pct_positive", "cpcv_valid_paths"]
    print(df[cols].head(20).to_string(index=False))
    print()

    # Summary stats
    n_positive_cpcv = (df["cpcv_median_sharpe"] > 0).sum()
    n_high_pct = (df["cpcv_pct_positive"] >= 60).sum()
    print(f"  Configs with CPCV median Sharpe > 0: {n_positive_cpcv} / {len(df)}")
    print(f"  Configs with >= 60% positive paths:  {n_high_pct} / {len(df)}")


if __name__ == "__main__":
    main()
