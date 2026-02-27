"""Refined grid search: 120k+ combos with numba-optimized pipeline.

OLS × ZW × z_entry × z_exit × z_stop × confidence × time_stop × windows

Usage:
    python scripts/run_refined_grid.py
    python scripts/run_refined_grid.py --dry-run   # show grid summary only
    python scripts/run_refined_grid.py --workers 5  # multiprocessing
"""

import argparse
import sys
import time as time_mod
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_grid
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import (
    ConfidenceConfig,
    _apply_conf_filter_numba,
    apply_time_stop,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

# ======================================================================
# Constants
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0   # NQ, YM
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
METRICS_TRES_COURT = MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6)
CONF_CFG = ConfidenceConfig()

# ======================================================================
# Grid parameters
# ======================================================================

OLS_WINDOWS = [1980, 2310, 2640, 2970, 3300]
ZW_WINDOWS = [24, 30, 36, 42, 48]

Z_ENTRIES = [2.85, 2.90, 2.95, 3.00, 3.05, 3.10, 3.15, 3.20]
Z_EXITS = [1.00, 1.10, 1.20, 1.25, 1.30, 1.40]
Z_STOPS = [3.75, 4.00, 4.25, 4.50]
CONFS = [67.0, 68.0, 69.0, 70.0, 72.0]
TIME_STOPS = [0, 12, 18, 24, 36]  # 0 = no limit

# Trading windows: (name, entry_start_min, entry_end_min)
# All use flat_time = 15:30 CT = 930 min
FLAT_MIN = 930
WINDOWS = [
    ("02:00-12:00",  120, 720),
    ("02:00-13:00",  120, 780),
    ("02:00-14:00",  120, 840),
    ("02:00-15:00",  120, 900),
    ("03:00-13:00",  180, 780),
    ("03:00-14:00",  180, 840),
    ("04:00-13:00",  240, 780),
    ("04:00-14:00",  240, 840),
    ("04:00-15:00",  240, 900),
]


def print_grid_summary():
    """Print grid configuration summary."""
    n_ols = len(OLS_WINDOWS)
    n_zw = len(ZW_WINDOWS)
    n_entry = len(Z_ENTRIES)
    n_exit = len(Z_EXITS)
    n_stop = len(Z_STOPS)
    n_conf = len(CONFS)
    n_ts = len(TIME_STOPS)
    n_win = len(WINDOWS)

    # Count valid signal combos (z_exit < z_entry and z_stop > z_entry)
    valid_signal = 0
    for ze, zx, zs in product(Z_ENTRIES, Z_EXITS, Z_STOPS):
        if zx < ze and zs > ze:
            valid_signal += 1

    n_inner = valid_signal * n_conf * n_ts * n_win
    n_pipeline = n_ols * n_zw
    total = n_pipeline * n_inner

    print("=" * 100)
    print(" GRILLE AFFINEE — NQ_YM 5min")
    print("=" * 100)
    print()
    print(f" OLS windows:     {OLS_WINDOWS}  ({n_ols})")
    print(f" ZW windows:      {ZW_WINDOWS}  ({n_zw})")
    print(f" z_entry:         {Z_ENTRIES}  ({n_entry})")
    print(f" z_exit:          {Z_EXITS}  ({n_exit})")
    print(f" z_stop:          {Z_STOPS}  ({n_stop})")
    print(f" confidence:      {CONFS}  ({n_conf})")
    print(f" time_stop:       {TIME_STOPS}  ({n_ts})")
    print(f" windows:         {[w[0] for w in WINDOWS]}  ({n_win})")
    print(" flat:            15:30 CT")
    print()
    print(f" Pipeline combos (OLS × ZW):          {n_pipeline}")
    print(f" Valid signal combos (e×x×s):          {valid_signal} / {n_entry * n_exit * n_stop}")
    print(f" Inner combos per pipeline:            {n_inner}")
    print(f" TOTAL COMBOS:                         {total:,}")
    print("=" * 100)
    return total


# ======================================================================
# Worker function (one per OLS window)
# ======================================================================

def _load_data():
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    return load_aligned_pair_cache(pair, "5min")


def run_ols_batch(ols_window):
    """Process one OLS window: compute hedge + metrics once, then sweep ZW + signal params."""
    aligned = _load_data()
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    # Precompute minutes array for window filter
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # Compute hedge ratio (depends on OLS window only, ZW affects only zscore)
    est_base = create_estimator("ols_rolling", window=ols_window, zscore_window=ZW_WINDOWS[0])
    hr = est_base.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    # Compute metrics + confidence (independent of ZW)
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], METRICS_TRES_COURT)
    confidence = compute_confidence(metrics, CONF_CFG).values

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

                    # Generate raw signals (numba, ~2ms)
                    raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)

                    for ts in TIME_STOPS:
                        # Apply time stop
                        sig_ts = apply_time_stop(raw_signals, ts)

                        for min_conf in CONFS:
                            # Apply confidence filter (numba)
                            sig_conf = _apply_conf_filter_numba(sig_ts, confidence, min_conf)

                            for win_name, entry_start, entry_end in WINDOWS:
                                # Apply window filter (numba)
                                sig_final = apply_window_filter_numba(
                                    sig_conf, minutes, entry_start, entry_end, FLAT_MIN
                                )

                                # Grid backtest (no equity)
                                bt = run_backtest_grid(
                                    px_a, px_b, sig_final, beta,
                                    mult_a=MULT_A, mult_b=MULT_B,
                                    tick_a=TICK_A, tick_b=TICK_B,
                                    slippage_ticks=SLIPPAGE, commission=COMMISSION,
                                )

                                count += 1
                                num = bt["trades"]
                                if num < 10 or bt["pnl"] <= 0:
                                    continue

                                trd_an = num / years if years > 0 else 0
                                results.append({
                                    "ols": ols_window, "zw": zw, "window": win_name,
                                    "z_entry": z_entry, "z_exit": z_exit,
                                    "z_stop": z_stop, "conf": min_conf, "time_stop": ts,
                                    "trades": num, "win_rate": bt["win_rate"],
                                    "pnl": bt["pnl"], "pf": bt["profit_factor"],
                                    "avg_pnl": bt["avg_pnl_trade"],
                                    "avg_dur": bt["avg_duration_bars"],
                                    "trd_an": round(trd_an, 1),
                                })

    return ols_window, results, count


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Refined grid search NQ_YM")
    parser.add_argument("--dry-run", action="store_true", help="Show grid summary only")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    total = print_grid_summary()

    if args.dry_run:
        return

    print(f"\n Lancement... ({args.workers} worker(s))")
    sys.stdout.flush()
    t0 = time_mod.time()

    all_results = []
    total_count = 0

    if args.workers > 1:
        with Pool(args.workers) as pool:
            for ols_w, results, count in pool.imap_unordered(run_ols_batch, OLS_WINDOWS):
                total_count += count
                all_results.extend(results)
                elapsed = time_mod.time() - t0
                print(f"  OLS={ols_w} done: {count:,} combos, {len(results)} profitable, {elapsed:.0f}s")
                sys.stdout.flush()
    else:
        for ols_w in OLS_WINDOWS:
            ols_w, results, count = run_ols_batch(ols_w)
            total_count += count
            all_results.extend(results)
            elapsed = time_mod.time() - t0
            print(f"  OLS={ols_w} done: {count:,} combos, {len(results)} profitable, {elapsed:.0f}s")
            sys.stdout.flush()

    elapsed = time_mod.time() - t0
    print(f"\n Total: {total_count:,} combos en {elapsed:.0f}s ({total_count/max(elapsed,1):.0f} combos/s)")
    print(f" {len(all_results)} configs profitables (PnL > 0, >= 10 trades)")

    if not all_results:
        print(" Aucun resultat profitable.")
        return

    df = pd.DataFrame(all_results)

    # Compute composite score: PF * sqrt(trades) * (pnl/1000)
    df["score"] = df["pf"] * np.sqrt(df["trades"]) * (df["pnl"] / 1000.0).clip(lower=0.1)

    # ---- TOP 50 ----
    print(f"\n{'='*150}")
    print(" TOP 50 PAR SCORE COMPOSITE")
    print(f"{'='*150}")
    top = df.sort_values("score", ascending=False).head(50)

    header = (f" {'#':>3} {'OLS':>5} {'ZW':>3} {'Window':<12} {'Zent':>5} {'Zex':>4} {'Zst':>5} "
              f"{'Conf':>4} {'TStop':>5} {'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} "
              f"{'Avg':>7} {'Dur':>4} {'Trd/An':>6} {'Score':>7}")
    print(header)
    print(" " + "-" * 145)

    for i, (_, r) in enumerate(top.iterrows()):
        ts_label = f"{r['time_stop']:.0f}b" if r['time_stop'] > 0 else "none"
        print(
            f" {i+1:>3} {r['ols']:>5.0f} {r['zw']:>3.0f} {r['window']:<12} "
            f"{r['z_entry']:>5.2f} {r['z_exit']:>4.2f} {r['z_stop']:>5.2f} "
            f"{r['conf']:>4.0f} {ts_label:>5} {r['trades']:>5.0f} {r['win_rate']:>5.1f}% "
            f"${r['pnl']:>9,.0f} {r['pf']:>5.2f} ${r['avg_pnl']:>6,.0f} "
            f"{r['avg_dur']:>4.1f} {r['trd_an']:>6.1f} {r['score']:>7.1f}"
        )

    # ---- Analysis by dimension ----
    print(f"\n{'='*100}")
    print(" ANALYSE PAR DIMENSION (configs avec PF > 1.3)")
    print(f"{'='*100}")

    df_good = df[df["pf"] > 1.3].copy()
    if len(df_good) > 0:
        for dim, col in [("OLS", "ols"), ("ZW", "zw"), ("Window", "window"),
                          ("z_entry", "z_entry"), ("z_exit", "z_exit"),
                          ("z_stop", "z_stop"), ("Confidence", "conf"),
                          ("Time Stop", "time_stop")]:
            grp = df_good.groupby(col).agg(
                count=("pf", "size"),
                avg_pf=("pf", "mean"),
                avg_trades=("trades", "mean"),
                avg_pnl=("pnl", "mean"),
            ).round(2)
            print(f"\n {dim}:")
            for idx_val, row in grp.iterrows():
                print(f"   {idx_val}: {row['count']:>4} configs, PF moy={row['avg_pf']:.2f}, "
                      f"trades moy={row['avg_trades']:.0f}, PnL moy=${row['avg_pnl']:,.0f}")

    # Save to CSV
    output_path = PROJECT_ROOT / "output" / "NQ_YM" / "grid_refined_ols.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n Resultats complets sauvegardes: {output_path}")


if __name__ == "__main__":
    main()
