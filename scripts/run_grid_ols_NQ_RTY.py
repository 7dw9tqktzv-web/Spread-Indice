"""Grid search OLS Rolling — NQ_RTY exhaustif.

Gammes tres larges, aucun a priori de NQ/YM.
Teste toutes les combinaisons possibles pour trouver un edge OLS sur NQ/RTY.

Usage:
    python scripts/run_grid_ols_NQ_RTY.py --workers 10
    python scripts/run_grid_ols_NQ_RTY.py --dry-run
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
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import (
    ConfidenceConfig,
    _apply_conf_filter_numba,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("grid_ols_nq_rty")

OUTPUT_DIR = PROJECT_ROOT / "output"

# ======================================================================
# Grid parameters — NQ_RTY OLS — GAMMES TRES LARGES
# ======================================================================

PAIR = ("NQ", "RTY")
_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.50
FLAT_MIN = 930  # 15:30 CT

# OLS windows: 3j a 40j (264 bars/jour)
OLS_WINDOWS = [
    792,    # 3j
    1056,   # 4j
    1320,   # 5j
    1584,   # 6j
    1980,   # 7.5j
    2376,   # 9j
    2640,   # 10j
    3300,   # 12.5j
    3960,   # 15j
    5280,   # 20j
    6600,   # 25j
    7920,   # 30j
    10560,  # 40j
]

# Z-score windows: 6 a 60 bars (30min a 5h)
ZSCORE_WINDOWS = [6, 12, 18, 24, 30, 36, 42, 48, 60]

# z_entry: 1.5 a 4.0 (large gamme)
Z_ENTRIES = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]

# z_exit: 0.0 a 2.5
Z_EXITS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]

# z_stop: 2.0 a 6.0
Z_STOPS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# min_confidence: 0 (off) a 80%
MIN_CONFIDENCES = [0.0, 30.0, 40.0, 50.0, 55.0, 60.0, 65.0, 67.0, 70.0, 75.0, 80.0]

# Metric profiles
METRIC_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "moyen":      MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
}

# Entry windows
ENTRY_WINDOWS = [
    ("02:00-14:00", 2, 0, 14, 0),
    ("03:00-12:00", 3, 0, 12, 0),
    ("04:00-12:00", 4, 0, 12, 0),
    ("04:00-13:00", 4, 0, 13, 0),
    ("05:00-12:00", 5, 0, 12, 0),
    ("06:00-11:00", 6, 0, 11, 0),
    ("08:00-14:00", 8, 0, 14, 0),
]


@dataclass
class OLSJobNQRTY:
    ols_window: int
    zscore_window: int
    profile_name: str
    window_label: str
    entry_start_min: int
    entry_end_min: int


def run_ols_job(job: OLSJobNQRTY) -> list[dict]:
    """Run OLS hedge + all signal combos for one (ols_window, zscore_window, profile, entry_window)."""
    try:
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": "No cache for NQ_RTY"}]

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        idx = aligned.df.index

        # OLS hedge ratio
        est = create_estimator("ols_rolling", window=job.ols_window, zscore_window=job.zscore_window)
        hr = est.estimate(aligned)
        spread = hr.spread
        beta = hr.beta.values

        # OLS z-score (rolling mean/std)
        mu = spread.rolling(job.zscore_window).mean()
        sigma = spread.rolling(job.zscore_window).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

        # Metrics + confidence
        profile_cfg = METRIC_PROFILES[job.profile_name]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, ConfidenceConfig()).values

        # Minutes array
        minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

        results = []

        for z_entry in Z_ENTRIES:
            for z_exit in Z_EXITS:
                if z_exit >= z_entry:
                    continue
                for z_stop in Z_STOPS:
                    if z_stop <= z_entry:
                        continue

                    raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)

                    for min_conf in MIN_CONFIDENCES:
                        if min_conf > 0:
                            sig = _apply_conf_filter_numba(raw_signals.copy(), confidence, min_conf)
                        else:
                            sig = raw_signals.copy()

                        sig = apply_window_filter_numba(
                            sig, minutes,
                            job.entry_start_min, job.entry_end_min, FLAT_MIN,
                        )

                        bt = run_backtest_grid(
                            px_a, px_b, sig, beta,
                            mult_a=MULT_A, mult_b=MULT_B,
                            tick_a=TICK_A, tick_b=TICK_B,
                            slippage_ticks=SLIPPAGE, commission=COMMISSION,
                        )

                        results.append({
                            "ols_window": job.ols_window,
                            "zscore_window": job.zscore_window,
                            "profil": job.profile_name,
                            "window": job.window_label,
                            "z_entry": z_entry,
                            "z_exit": z_exit,
                            "z_stop": z_stop,
                            "min_confidence": min_conf,
                            "trades": bt["trades"],
                            "win_rate": bt["win_rate"],
                            "pnl": bt["pnl"],
                            "profit_factor": bt["profit_factor"],
                            "avg_pnl_trade": bt["avg_pnl_trade"],
                            "avg_duration_bars": bt["avg_duration_bars"],
                        })

        return results

    except Exception as e:
        return [{"error": str(e), "ols_window": job.ols_window}]


def main():
    parser = argparse.ArgumentParser(description="Grid search OLS NQ_RTY")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Count signal combos per job
    signal_combos = 0
    for ze in Z_ENTRIES:
        for zx in Z_EXITS:
            if zx >= ze:
                continue
            for zs in Z_STOPS:
                if zs <= ze:
                    continue
                signal_combos += 1
    signal_combos *= len(MIN_CONFIDENCES)

    # Jobs = OLS_WINDOWS x ZSCORE_WINDOWS x PROFILES x ENTRY_WINDOWS
    n_jobs = len(OLS_WINDOWS) * len(ZSCORE_WINDOWS) * len(METRIC_PROFILES) * len(ENTRY_WINDOWS)
    total = n_jobs * signal_combos

    log.info("Grid OLS NQ_RTY:")
    log.info(f"  OLS windows:     {len(OLS_WINDOWS)} ({OLS_WINDOWS[0]} to {OLS_WINDOWS[-1]})")
    log.info(f"  Z-score windows: {len(ZSCORE_WINDOWS)} ({ZSCORE_WINDOWS[0]} to {ZSCORE_WINDOWS[-1]})")
    log.info(f"  z_entry:         {len(Z_ENTRIES)} ({Z_ENTRIES[0]} to {Z_ENTRIES[-1]})")
    log.info(f"  z_exit:          {len(Z_EXITS)} ({Z_EXITS[0]} to {Z_EXITS[-1]})")
    log.info(f"  z_stop:          {len(Z_STOPS)} ({Z_STOPS[0]} to {Z_STOPS[-1]})")
    log.info(f"  min_confidence:  {len(MIN_CONFIDENCES)} ({MIN_CONFIDENCES[0]} to {MIN_CONFIDENCES[-1]})")
    log.info(f"  profiles:        {len(METRIC_PROFILES)} ({list(METRIC_PROFILES.keys())})")
    log.info(f"  entry windows:   {len(ENTRY_WINDOWS)}")
    log.info(f"  Signal combos/job: {signal_combos}")
    log.info(f"  Jobs: {n_jobs}")
    log.info(f"  TOTAL BACKTESTS: {total:,}")

    if args.dry_run:
        log.info("DRY RUN — stopping here.")
        return

    # Build jobs
    jobs = []
    for ols_w in OLS_WINDOWS:
        for zw in ZSCORE_WINDOWS:
            for prof_name in METRIC_PROFILES:
                for wlabel, sh, sm, eh, em in ENTRY_WINDOWS:
                    entry_start = sh * 60 + sm
                    entry_end = eh * 60 + em
                    jobs.append(OLSJobNQRTY(
                        ols_window=ols_w,
                        zscore_window=zw,
                        profile_name=prof_name,
                        window_label=wlabel,
                        entry_start_min=entry_start,
                        entry_end_min=entry_end,
                    ))

    log.info(f"Launching {len(jobs)} jobs with {args.workers} workers...")
    t0 = time.time()

    all_results = []
    errors = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_ols_job, j): j for j in jobs}
        for future in as_completed(futures):
            completed += 1
            try:
                batch = future.result()
                for r in batch:
                    if "error" in r:
                        errors += 1
                    else:
                        all_results.append(r)
            except Exception as e:
                errors += 1
                log.error(f"Job exception: {e}")

            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (len(jobs) - completed) / rate if rate > 0 else 0
                log.info(f"  {completed}/{len(jobs)} jobs ({len(all_results):,} results, "
                         f"{errors} errors, {elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    log.info(f"Grid complete: {len(all_results):,} results in {elapsed:.0f}s, {errors} errors")

    # Save all results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_all = OUTPUT_DIR / "NQ_RTY" / "grid_ols.csv"
    df.to_csv(csv_all, index=False)
    log.info(f"Saved {len(df):,} rows to {csv_all}")

    # Filter profitable
    profitable = df[df["pnl"] > 0]
    csv_filt = OUTPUT_DIR / "NQ_RTY" / "grid_ols_filtered.csv"
    profitable.to_csv(csv_filt, index=False)
    log.info(f"Profitable: {len(profitable):,} / {len(df):,} ({len(profitable)/max(len(df),1)*100:.1f}%)")

    # Quick top 20
    if not profitable.empty:
        top = profitable.nlargest(20, "pnl")
        log.info("\nTOP 20 BY PNL:")
        log.info(f"  {'OLS':>6} {'ZW':>4} {'Prof':<10} {'Window':<14} "
                 f"{'ze':>5} {'zx':>4} {'zs':>4} {'conf':>4} | "
                 f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7}")
        log.info("  " + "-" * 100)
        for _, r in top.iterrows():
            log.info(f"  {r['ols_window']:>6} {r['zscore_window']:>4} {r['profil']:<10} {r['window']:<14} "
                     f"{r['z_entry']:>5.2f} {r['z_exit']:>4.2f} {r['z_stop']:>4.1f} {r['min_confidence']:>4.0f} | "
                     f"{r['trades']:>5.0f} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} "
                     f"{r['profit_factor']:>6.2f} ${r['avg_pnl_trade']:>6,.0f}")

        # Top 20 by PF (min 30 trades)
        pf_pool = profitable[profitable["trades"] >= 30]
        if not pf_pool.empty:
            top_pf = pf_pool.nlargest(20, "profit_factor")
            log.info("\nTOP 20 BY PF (min 30 trades):")
            log.info(f"  {'OLS':>6} {'ZW':>4} {'Prof':<10} {'Window':<14} "
                     f"{'ze':>5} {'zx':>4} {'zs':>4} {'conf':>4} | "
                     f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7}")
            log.info("  " + "-" * 100)
            for _, r in top_pf.iterrows():
                log.info(f"  {r['ols_window']:>6} {r['zscore_window']:>4} {r['profil']:<10} {r['window']:<14} "
                         f"{r['z_entry']:>5.2f} {r['z_exit']:>4.2f} {r['z_stop']:>4.1f} {r['min_confidence']:>4.0f} | "
                         f"{r['trades']:>5.0f} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} "
                         f"{r['profit_factor']:>6.2f} ${r['avg_pnl_trade']:>6,.0f}")
    else:
        log.info("AUCUNE config profitable trouvee.")


if __name__ == "__main__":
    main()
