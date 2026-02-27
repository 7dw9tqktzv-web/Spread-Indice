"""Phase 13c Grid Massif: ~24.7M configs with ADF window in grid + delta sigma.

NQ_YM OLS, binary gates (ADF < -2.86, Hurst < 0.50, Corr > 0.70).
ADF window swept in grid to capture sensitivity.
Delta sigma: z_exit = max(z_entry - delta_tp, 0.0), z_stop = z_entry + delta_sl.

Usage:
    python scripts/phase13c_grid_massif.py --dry-run
    python scripts/phase13c_grid_massif.py --workers 20
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

from src.backtest.engine import run_backtest_grid

# ======================================================================
# Constants
# ======================================================================
from src.config.instruments import DEFAULT_SLIPPAGE_TICKS, get_pair_specs
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.signals.filters import apply_time_stop, apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.validation.cpcv import CPCVConfig, run_cpcv
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask

_NQ, _YM = get_pair_specs("NQ", "YM")
MULT_A, MULT_B = _NQ.multiplier, _YM.multiplier
TICK_A, TICK_B = _NQ.tick_size, _YM.tick_size
SLIPPAGE = DEFAULT_SLIPPAGE_TICKS
COMMISSION = _NQ.commission
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930  # 15:30 CT

# CPCV config
CPCV_CFG = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)

# ======================================================================
# Grid parameters
# ======================================================================

OLS_WINDOWS = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000]
ADF_WINDOWS = [12, 18, 24, 30, 36, 48, 64, 96, 128]
ZW_WINDOWS = [10, 15, 20, 25, 28, 30, 35, 40, 45, 50, 60]

Z_ENTRIES = [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75]
DELTA_TP = [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00]
DELTA_SL = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50]
TIME_STOPS = [0, 3, 6, 10, 12, 16, 20, 30, 40, 50]

# Trading windows: (name, entry_start_min, entry_end_min)
WINDOWS = [
    ("02:00-14:00", 120, 840),
    ("04:00-14:00", 240, 840),
    ("06:00-14:00", 360, 840),
]

# Gate thresholds (fixed by theory)
GATE_ADF_THRESH = -2.86
GATE_HURST_THRESH = 0.50
GATE_CORR_THRESH = 0.70
GATE_HURST_WINDOW = 64
GATE_CORR_WINDOW = 24


def print_grid_summary():
    """Print grid configuration summary and return total combo count."""
    n_ols = len(OLS_WINDOWS)
    n_adf = len(ADF_WINDOWS)
    n_zw = len(ZW_WINDOWS)
    n_entry = len(Z_ENTRIES)
    n_dtp = len(DELTA_TP)
    n_dsl = len(DELTA_SL)
    n_ts = len(TIME_STOPS)
    n_win = len(WINDOWS)

    n_pipeline = n_ols * n_adf
    n_inner = n_zw * n_entry * n_dtp * n_dsl * n_ts * n_win
    total = n_pipeline * n_inner

    print("=" * 100)
    print(" PHASE 13c -- GRID MASSIF -- ADF IN GRID + DELTA SIGMA -- NQ_YM 5min")
    print("=" * 100)
    print()
    print(f" OLS windows:     {OLS_WINDOWS}  ({n_ols})")
    print(f" ADF windows:     {ADF_WINDOWS}  ({n_adf})")
    print(f" ZW windows:      {ZW_WINDOWS}  ({n_zw})")
    print(f" z_entry:         {Z_ENTRIES}  ({n_entry})")
    print(f" delta_tp:        {DELTA_TP}  ({n_dtp})")
    print(f" delta_sl:        {DELTA_SL}  ({n_dsl})")
    print(f" time_stop:       {TIME_STOPS}  ({n_ts})")
    print(f" windows:         {[w[0] for w in WINDOWS]}  ({n_win})")
    print(" flat:            15:30 CT")
    print()
    print(f" Gates (fixed):   ADF < {GATE_ADF_THRESH}, "
          f"Hurst < {GATE_HURST_THRESH}, Corr > {GATE_CORR_THRESH}")
    print(f" Gate windows:    ADF=IN GRID {ADF_WINDOWS}, "
          f"Hurst={GATE_HURST_WINDOW}, Corr={GATE_CORR_WINDOW}")
    print(" Delta sigma:     z_exit = max(z_entry - delta_tp, 0.0)")
    print("                  z_stop = z_entry + delta_sl")
    print(f" CPCV:            {CPCV_CFG.n_folds} folds, {CPCV_CFG.n_test_folds} test, "
          f"purge={CPCV_CFG.purge_bars} bars, 45 paths")
    print()
    print(f" Pipeline combos (OLS x ADF):      {n_pipeline}")
    print(f" Inner combos per pipeline:        {n_inner:,}")
    print(f" TOTAL COMBOS:                     {total:,}")
    print("=" * 100)
    return total


# ======================================================================
# Worker function (one per OLS x ADF pair)
# ======================================================================

def run_ols_adf_batch(args):
    """Process one (OLS, ADF) pair: compute hedge + gate mask once, then sweep."""
    ols_window, adf_window = args

    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # OLS hedge ratio
    est = create_estimator("ols_rolling", window=ols_window, zscore_window=ZW_WINDOWS[0])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    # Binary gate mask with specific adf_window
    gate_cfg = GateConfig(
        adf_threshold=GATE_ADF_THRESH,
        hurst_threshold=GATE_HURST_THRESH,
        corr_threshold=GATE_CORR_THRESH,
        adf_window=adf_window,
        hurst_window=GATE_HURST_WINDOW,
        corr_window=GATE_CORR_WINDOW,
    )
    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg
    )

    years = (idx[-1] - idx[0]).days / 365.25
    results = []
    count = 0

    for zw in ZW_WINDOWS:
        mu = spread.rolling(zw).mean()
        sigma = spread.rolling(zw).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(zscore, dtype=np.float64)

        for z_entry in Z_ENTRIES:
            for dtp in DELTA_TP:
                z_exit = round(max(z_entry - dtp, 0.0), 4)

                for dsl in DELTA_SL:
                    z_stop = round(z_entry + dsl, 4)

                    raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)

                    for ts in TIME_STOPS:
                        sig_ts = apply_time_stop(raw_signals, ts)
                        sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)

                        for win_name, entry_start, entry_end in WINDOWS:
                            sig_final = apply_window_filter_numba(
                                sig_gated, minutes, entry_start, entry_end, FLAT_MIN
                            )

                            bt = run_backtest_grid(
                                px_a, px_b, sig_final, beta,
                                MULT_A, MULT_B, TICK_A, TICK_B,
                                SLIPPAGE, COMMISSION,
                            )

                            count += 1
                            num = bt["trades"]
                            if num < 10:
                                continue

                            cpcv = run_cpcv(
                                bt["trade_entry_bars"],
                                bt["trade_exit_bars"],
                                bt["trade_pnls"],
                                n,
                                CPCV_CFG,
                            )

                            max_dd = bt["max_dd"]

                            pnls = bt["trade_pnls"]
                            sharpe_full = 0.0
                            if pnls.std() > 1e-12:
                                sharpe_full = float(pnls.mean() / pnls.std())

                            results.append({
                                "ols": ols_window,
                                "adf_w": adf_window,
                                "zw": zw,
                                "window": win_name,
                                "z_entry": z_entry,
                                "delta_tp": dtp,
                                "delta_sl": dsl,
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

    return (ols_window, adf_window), results, count


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 13c Grid Massif + CPCV")
    parser.add_argument("--dry-run", action="store_true", help="Show grid summary only")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel workers")
    args = parser.parse_args()

    total = print_grid_summary()

    if args.dry_run:
        print("\n  --dry-run: exiting without running.\n")
        return

    print(f"\nStarting grid with {args.workers} workers...\n")

    t0 = time_mod.time()
    all_results = []
    total_count = 0

    jobs = list(product(OLS_WINDOWS, ADF_WINDOWS))
    n_jobs = len(jobs)
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_ols_adf_batch, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                key, results, count = future.result()
                total_count += count
                all_results.extend(results)
                done += 1
                elapsed = time_mod.time() - t0
                rate = total_count / elapsed if elapsed > 0 else 0
                eta = (total - total_count) / rate if rate > 0 else 0
                print(f"  [{done:>3}/{n_jobs}] OLS={key[0]:>5}, ADF_w={key[1]:>3}: "
                      f"{count:>7,} combos, {len(results):>6,} stored, "
                      f"{elapsed:>6.0f}s elapsed, ~{eta:>5.0f}s ETA")
            except Exception as e:
                done += 1
                print(f"  [{done:>3}/{n_jobs}] OLS={job[0]:>5}, ADF_w={job[1]:>3} FAILED: {e}")

    elapsed = time_mod.time() - t0
    rate = total_count / elapsed if elapsed > 0 else 0
    print(f"\nTotal: {total_count:,} combos in {elapsed:.0f}s ({rate:.0f} combos/s)")

    if not all_results:
        print("No results with trades >= 10.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values("cpcv_median_sharpe", ascending=False)

    output_path = PROJECT_ROOT / "output" / "NQ_YM" / "phase13c_grid_massif.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} results to {output_path}")

    # Display top 30
    print(f"\n{'='*150}")
    print(" TOP 30 BY CPCV MEDIAN SHARPE")
    print(f"{'='*150}")
    cols = ["ols", "adf_w", "zw", "window", "z_entry", "delta_tp", "delta_sl",
            "z_exit", "z_stop", "time_stop",
            "trades", "pnl", "pf", "max_dd", "sharpe_full",
            "cpcv_median_sharpe", "cpcv_pct_positive", "cpcv_valid_paths"]
    print(df[cols].head(30).to_string(index=False))
    print()

    # Summary stats
    n_positive_cpcv = (df["cpcv_median_sharpe"] > 0).sum()
    n_high_pct = (df["cpcv_pct_positive"] >= 60).sum()
    n_pf_above = (df["pf"] >= 1.30).sum()
    n_trades_150 = (df["trades"] >= 150).sum()
    n_dd_safe = (df["max_dd"] > -5000).sum()

    print(f"  Total configs stored (trades >= 10):     {len(df):,}")
    print(f"  CPCV median Sharpe > 0:                  {n_positive_cpcv:,}")
    print(f"  CPCV >= 60% positive paths:              {n_high_pct:,}")
    print(f"  PF >= 1.30:                              {n_pf_above:,}")
    print(f"  Trades >= 150:                           {n_trades_150:,}")
    print(f"  MaxDD > -$5,000 (propfirm safe):         {n_dd_safe:,}")

    # Tier 1 propfirm filter
    tier1 = df[
        (df["trades"] >= 300) &
        (df["pf"] >= 1.30) &
        (df["max_dd"] > -5000) &
        (df["cpcv_pct_positive"] >= 70)
    ]
    print(f"\n  TIER 1 (trades>=300, PF>=1.30, DD>-$5k, paths+>=70%): {len(tier1):,}")
    if len(tier1) > 0:
        print(tier1[cols].head(10).to_string(index=False))

    # Tier 2
    tier2 = df[
        (df["trades"] >= 200) &
        (df["pf"] >= 1.20) &
        (df["max_dd"] > -5500) &
        (df["cpcv_pct_positive"] >= 65)
    ]
    print(f"\n  TIER 2 (trades>=200, PF>=1.20, DD>-$5.5k, paths+>=65%): {len(tier2):,}")
    if len(tier2) > 0:
        print(tier2[cols].head(10).to_string(index=False))

    # Tier 3
    tier3 = df[
        (df["trades"] >= 150) &
        (df["pf"] >= 1.15) &
        (df["max_dd"] > -6000) &
        (df["cpcv_pct_positive"] >= 60)
    ]
    print(f"\n  TIER 3 (trades>=150, PF>=1.15, DD>-$6k, paths+>=60%): {len(tier3):,}")
    if len(tier3) > 0:
        print(tier3[cols].head(10).to_string(index=False))

    # Tier 4
    tier4 = df[
        (df["trades"] >= 100) &
        (df["pf"] >= 1.10) &
        (df["max_dd"] > -7000) &
        (df["cpcv_pct_positive"] >= 55)
    ]
    print(f"\n  TIER 4 (trades>=100, PF>=1.10, DD>-$7k, paths+>=55%): {len(tier4):,}")
    if len(tier4) > 0 and len(tier3) == 0:
        print(tier4[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
