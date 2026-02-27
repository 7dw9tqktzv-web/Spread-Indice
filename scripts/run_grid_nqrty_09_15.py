"""Grid search OLS Rolling -- NQ_RTY 09:00-15:00 CT (Etape 1).

Signal brut uniquement (pas de filtre metrique).
Delta-sigma parametrization : z_exit = max(z_entry - delta_tp, 0), z_stop = z_entry + delta_sl.
Commission Phidias : $2.20/side.

Usage:
    python scripts/run_grid_nqrty_09_15.py --workers 10
    python scripts/run_grid_nqrty_09_15.py --dry-run
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_grid
from src.config.instruments import get_pair_specs
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.signals.filters import apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("grid_nqrty_09_15")

OUTPUT_DIR = PROJECT_ROOT / "output" / "NQ_RTY"

# ======================================================================
# Constants
# ======================================================================

PAIR = ("NQ", "RTY")
_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1                        # 1 tick/leg
COMMISSION = 2.20                   # Phidias $2.20/side

FLAT_MIN = 15 * 60 + 30             # 930

# Entry windows: (label, start_hour, start_min, end_hour, end_min)
ENTRY_WINDOWS = [
    ("02:00-07:00", 2, 0, 7, 0),
    ("02:00-15:00", 2, 0, 15, 0),
    ("06:00-15:00", 6, 0, 15, 0),
    ("07:00-15:00", 7, 0, 15, 0),
    ("09:00-15:00", 9, 0, 15, 0),
]

MIN_TRADES = 200                    # Filtre minimum

# ======================================================================
# Grid axes
# ======================================================================

# OLS windows (bars 5min) : 2j a 40j (264 bars/jour)
OLS_WINDOWS = [
    528,    # 2j
    1320,   # 5j
    2640,   # 10j
    3960,   # 15j
    5280,   # 20j
    6600,   # 25j
    7920,   # 30j
    9240,   # 35j
    10560,  # 40j
]

# Z-score windows
ZSCORE_WINDOWS = [10, 12, 15, 20, 24, 25, 30, 36, 40]

# z_entry
Z_ENTRIES = [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]

# delta_tp (z_exit = max(z_entry - delta_tp, 0))
DELTA_TPS = [1.00, 1.25, 1.50, 2.00, 2.25, 2.50, 3.00]

# delta_sl (z_stop = z_entry + delta_sl)
DELTA_SLS = [1.00, 1.25, 1.50, 1.75, 2.00]


# ======================================================================
# Job definition
# ======================================================================

@dataclass
class GridJob:
    ols_window: int
    zscore_window: int
    window_label: str
    entry_start_min: int
    entry_end_min: int


def run_job(job: GridJob) -> list[dict]:
    """Run one (OLS_window, ZW, window) triple, sweeping all z-combos."""
    try:
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": "No cache for NQ_RTY"}]

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        idx = aligned.df.index

        # OLS hedge ratio
        est = create_estimator(
            "ols_rolling",
            window=job.ols_window,
            zscore_window=job.zscore_window,
        )
        hr = est.estimate(aligned)
        spread = hr.spread
        beta = hr.beta.values

        # Z-score (rolling mean/std)
        mu = spread.rolling(job.zscore_window).mean()
        sigma = spread.rolling(job.zscore_window).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace(
                [np.inf, -np.inf], np.nan
            ).values
        zscore = np.ascontiguousarray(
            np.nan_to_num(zscore, nan=0.0), dtype=np.float64
        )

        # Minutes array for window filter
        minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

        results = []

        for z_entry in Z_ENTRIES:
            for delta_tp in DELTA_TPS:
                z_exit = max(z_entry - delta_tp, 0.0)
                for delta_sl in DELTA_SLS:
                    z_stop = z_entry + delta_sl

                    raw_signals = generate_signals_numba(
                        zscore, z_entry, z_exit, z_stop
                    )

                    sig = apply_window_filter_numba(
                        raw_signals.copy(),
                        minutes,
                        job.entry_start_min,
                        job.entry_end_min,
                        FLAT_MIN,
                    )

                    bt = run_backtest_grid(
                        px_a, px_b, sig, beta,
                        mult_a=MULT_A, mult_b=MULT_B,
                        tick_a=TICK_A, tick_b=TICK_B,
                        slippage_ticks=SLIPPAGE,
                        commission=COMMISSION,
                    )

                    results.append({
                        "window": job.window_label,
                        "ols_window": job.ols_window,
                        "zscore_window": job.zscore_window,
                        "z_entry": z_entry,
                        "z_exit": round(z_exit, 2),
                        "z_stop": round(z_stop, 2),
                        "delta_tp": delta_tp,
                        "delta_sl": delta_sl,
                        "trades": bt["trades"],
                        "win_rate": bt["win_rate"],
                        "pnl": bt["pnl"],
                        "profit_factor": bt["profit_factor"],
                        "avg_pnl_trade": bt["avg_pnl_trade"],
                        "avg_duration_bars": bt["avg_duration_bars"],
                    })

        return results

    except Exception as e:
        return [{"error": str(e), "ols_window": job.ols_window,
                 "zscore_window": job.zscore_window}]


# ======================================================================
# Analysis helpers
# ======================================================================

def print_axis_stability(df: pd.DataFrame, axis: str, metric: str = "profit_factor"):
    """Print median metric per axis value (stability analysis)."""
    grouped = df.groupby(axis)[metric].agg(["median", "mean", "count"])
    log.info(f"\n  Stability by {axis} ({metric}):")
    log.info(f"    {'Value':>8}  {'Median':>8}  {'Mean':>8}  {'Count':>6}")
    log.info(f"    {'-----':>8}  {'------':>8}  {'----':>8}  {'-----':>6}")
    for val, row in grouped.iterrows():
        log.info(f"    {val:>8}  {row['median']:>8.2f}  {row['mean']:>8.2f}  {row['count']:>6.0f}")


def print_heatmap(df: pd.DataFrame, row_axis: str, col_axis: str,
                  metric: str = "profit_factor"):
    """Print OLS x ZW heatmap (median metric)."""
    pivot = df.pivot_table(values=metric, index=row_axis, columns=col_axis,
                           aggfunc="median")
    log.info(f"\n  Heatmap {row_axis} x {col_axis} (median {metric}):")

    # Header
    header = f"    {'':>8}"
    for c in pivot.columns:
        header += f"  {c:>6}"
    log.info(header)
    log.info(f"    {'':>8}  " + "-" * (8 * len(pivot.columns)))

    for idx_val, row in pivot.iterrows():
        line = f"    {idx_val:>8}"
        for v in row.values:
            if np.isnan(v):
                line += f"  {'---':>6}"
            else:
                line += f"  {v:>6.2f}"
        log.info(line)


def print_duration_clusters(df: pd.DataFrame):
    """Analyze avg_duration_bars distribution."""
    dur = df["avg_duration_bars"]
    bins = [0, 3, 6, 12, 24, 48, 100, float("inf")]
    labels = ["<15min", "15-30m", "30m-1h", "1-2h", "2-4h", "4-8h", ">8h"]
    cats = pd.cut(dur, bins=bins, labels=labels, right=False)
    counts = cats.value_counts().sort_index()

    log.info("\n  Duration clusters (avg trade duration):")
    log.info(f"    {'Cluster':>10}  {'Count':>6}  {'% of total':>10}")
    log.info(f"    {'-------':>10}  {'-----':>6}  {'---------':>10}")
    total = len(df)
    for label, cnt in counts.items():
        pct = cnt / total * 100 if total > 0 else 0
        log.info(f"    {label:>10}  {cnt:>6}  {pct:>9.1f}%")

    # Median PF per duration cluster
    df_copy = df.copy()
    df_copy["dur_cluster"] = cats.values
    grp = df_copy.groupby("dur_cluster", observed=True)["profit_factor"].median()
    log.info("\n  Median PF per duration cluster:")
    for label, pf in grp.items():
        log.info(f"    {label:>10}  PF {pf:.2f}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Grid NQ_RTY 09:00-15:00 Step 1")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Count z-combos per job
    z_combos = len(Z_ENTRIES) * len(DELTA_TPS) * len(DELTA_SLS)
    n_jobs = len(OLS_WINDOWS) * len(ZSCORE_WINDOWS) * len(ENTRY_WINDOWS)
    total = n_jobs * z_combos

    window_labels = [w[0] for w in ENTRY_WINDOWS]
    log.info("Grid NQ_RTY multi-window -- Etape 1 (signal brut)")
    log.info(f"  OLS windows:     {len(OLS_WINDOWS)} ({OLS_WINDOWS[0]} to {OLS_WINDOWS[-1]})")
    log.info(f"  Z-score windows: {len(ZSCORE_WINDOWS)} ({ZSCORE_WINDOWS[0]} to {ZSCORE_WINDOWS[-1]})")
    log.info(f"  z_entry:         {len(Z_ENTRIES)} ({Z_ENTRIES[0]} to {Z_ENTRIES[-1]})")
    log.info(f"  delta_tp:        {len(DELTA_TPS)} ({DELTA_TPS[0]} to {DELTA_TPS[-1]})")
    log.info(f"  delta_sl:        {len(DELTA_SLS)} ({DELTA_SLS[0]} to {DELTA_SLS[-1]})")
    log.info(f"  Commission:      ${COMMISSION}/side (Phidias)")
    log.info(f"  Windows:         {window_labels}, flat 15:30")
    log.info(f"  Z-combos/job:    {z_combos}")
    log.info(f"  Jobs:            {n_jobs}")
    log.info(f"  TOTAL BACKTESTS: {total:,}")
    log.info(f"  Min trades:      {MIN_TRADES}")

    if args.dry_run:
        log.info("DRY RUN -- stopping here.")
        return

    # Build jobs (1 per OLS x ZW x window)
    jobs = []
    for ols_w in OLS_WINDOWS:
        for zw in ZSCORE_WINDOWS:
            for wlabel, sh, sm, eh, em in ENTRY_WINDOWS:
                jobs.append(GridJob(
                    ols_window=ols_w,
                    zscore_window=zw,
                    window_label=wlabel,
                    entry_start_min=sh * 60 + sm,
                    entry_end_min=eh * 60 + em,
                ))

    log.info(f"Launching {len(jobs)} jobs with {args.workers} workers...")
    t0 = time.time()

    all_results = []
    errors = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j): j for j in jobs}
        for future in as_completed(futures):
            completed += 1
            try:
                batch = future.result()
                for r in batch:
                    if "error" in r:
                        errors += 1
                        log.error(f"  Error: {r}")
                    else:
                        all_results.append(r)
            except Exception as e:
                errors += 1
                log.error(f"Job exception: {e}")

            if completed % 10 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - completed) / rate if rate > 0 else 0
                log.info(
                    f"  {completed}/{len(jobs)} jobs "
                    f"({len(all_results):,} results, {errors} errors, "
                    f"{elapsed:.0f}s, ETA {eta:.0f}s)"
                )

    elapsed = time.time() - t0
    log.info(f"Grid complete: {len(all_results):,} results in {elapsed:.0f}s, {errors} errors")

    # Save all results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_all = OUTPUT_DIR / "grid_nqrty_09_15_step1.csv"
    df.to_csv(csv_all, index=False)
    log.info(f"Saved {len(df):,} rows to {csv_all}")

    # =====================================================================
    # Analysis per window
    # =====================================================================

    for wlabel in df["window"].unique():
        log.info("\n" + "#" * 70)
        log.info(f"# WINDOW: {wlabel}")
        log.info("#" * 70)

        wdf = df[df["window"] == wlabel]
        viable = wdf[wdf["trades"] >= MIN_TRADES].copy()
        log.info(f"\nViable configs (trades >= {MIN_TRADES}): {len(viable):,} / {len(wdf):,} "
                 f"({len(viable)/max(len(wdf),1)*100:.1f}%)")

        if viable.empty:
            log.info("AUCUNE config viable. Skipping.")
            continue

        profitable = viable[viable["pnl"] > 0]
        log.info(f"Profitable (PnL > 0): {len(profitable):,} / {len(viable):,} "
                 f"({len(profitable)/max(len(viable),1)*100:.1f}%)")

        # --- Top 30 by PF ---
        top_pf = viable.nlargest(30, "profit_factor")
        log.info(f"\nTOP 30 BY PROFIT FACTOR (trades >= {MIN_TRADES}):")
        log.info(f"  {'OLS':>6} {'ZW':>4} {'ze':>5} {'zx':>5} {'zs':>5} "
                 f"{'dTP':>5} {'dSL':>5} | "
                 f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7} {'AvgDur':>6}")
        log.info("  " + "-" * 90)
        for _, r in top_pf.iterrows():
            log.info(
                f"  {r['ols_window']:>6} {r['zscore_window']:>4} "
                f"{r['z_entry']:>5.2f} {r['z_exit']:>5.2f} {r['z_stop']:>5.2f} "
                f"{r['delta_tp']:>5.2f} {r['delta_sl']:>5.2f} | "
                f"{r['trades']:>5.0f} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} "
                f"{r['profit_factor']:>6.2f} ${r['avg_pnl_trade']:>6,.0f} "
                f"{r['avg_duration_bars']:>5.1f}"
            )

        # --- Stability analysis per axis ---
        log.info("\n  " + "=" * 55)
        log.info(f"  STABILITY ANALYSIS -- {wlabel}")
        log.info("  " + "=" * 55)

        print_axis_stability(viable, "ols_window")
        print_axis_stability(viable, "zscore_window")
        print_axis_stability(viable, "z_entry")
        print_axis_stability(viable, "delta_tp")
        print_axis_stability(viable, "delta_sl")

        # --- Heatmap OLS x ZW ---
        log.info("\n  " + "=" * 55)
        log.info(f"  HEATMAP OLS x ZW -- {wlabel}")
        log.info("  " + "=" * 55)
        print_heatmap(viable, "ols_window", "zscore_window")

        # --- Duration clusters ---
        log.info("\n  " + "=" * 55)
        log.info(f"  DURATION CLUSTERS -- {wlabel}")
        log.info("  " + "=" * 55)
        print_duration_clusters(viable)

        # --- Sweet spot summary ---
        log.info("\n  " + "=" * 55)
        log.info(f"  SWEET SPOT -- {wlabel}")
        log.info("  " + "=" * 55)

        for axis in ["ols_window", "zscore_window", "z_entry", "delta_tp", "delta_sl"]:
            grp = viable.groupby(axis)["profit_factor"].median()
            best_val = grp.idxmax()
            best_pf = grp.max()
            log.info(f"    Best {axis}: {best_val} (median PF {best_pf:.2f})")

    # Save all + viable subset
    csv_viable = OUTPUT_DIR / "grid_nqrty_09_15_step1_viable.csv"
    all_viable = df[df["trades"] >= MIN_TRADES].sort_values("profit_factor", ascending=False)
    all_viable.to_csv(csv_viable, index=False)
    log.info(f"\nSaved {len(all_viable):,} viable rows to {csv_viable}")


if __name__ == "__main__":
    main()
