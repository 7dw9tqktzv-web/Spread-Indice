"""Grid search backtest for OLS rolling method across all pairs and parameter combinations.

Usage:
    python scripts/run_grid.py --workers 20
    python scripts/run_grid.py --workers 10 --dry-run
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_grid
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import (
    ConfidenceConfig, compute_confidence,
    _apply_conf_filter_numba, apply_window_filter_numba,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.utils.time_utils import parse_session_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("grid")

OUTPUT_DIR = PROJECT_ROOT / "output"

# ──────────────────────────────────────────────────────────────────────
# Grid definition
# ──────────────────────────────────────────────────────────────────────

PAIRS = [
    ("NQ", "ES"), ("NQ", "RTY"), ("NQ", "YM"),
    ("ES", "RTY"), ("ES", "YM"), ("RTY", "YM"),
]

OLS_WINDOWS = [1320, 2640, 3960, 5280, 6600, 7920]  # 5j->30j (264 bars/j)

ZSCORE_WINDOWS = [12, 20, 24, 36, 48]

# (z_entry, z_exit, z_stop)
SIGNAL_COMBOS = [
    (2.0, 1.5, 3.0), (2.0, 1.0, 3.0), (2.0, 0.5, 3.0), (2.0, 0.0, 3.0),
    (2.5, 2.0, 3.5), (2.5, 1.5, 3.5), (2.5, 1.0, 3.5), (2.5, 0.5, 3.5),
    (3.0, 2.5, 4.0), (3.0, 2.0, 4.0), (3.0, 1.5, 4.0), (3.0, 1.0, 4.0),
]

MIN_CONFIDENCES = [30.0, 40.0, 50.0, 60.0, 70.0]

METRIC_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "moyen":      MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "long":       MetricsConfig(adf_window=96, hurst_window=512, halflife_window=96, correlation_window=48),
}

# ──────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────

def load_instruments():
    with open(PROJECT_ROOT / "config" / "instruments.yaml") as f:
        return yaml.safe_load(f)


def load_session():
    with open(PROJECT_ROOT / "config" / "backtest.yaml") as f:
        cfg = yaml.safe_load(f)
    return parse_session_config(cfg["session"])


# ──────────────────────────────────────────────────────────────────────
# Full job: metrics + signal + backtest (all in one worker)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GridJob:
    pair_name: str
    leg_a: str
    leg_b: str
    ols_window: int
    profile_name: str
    mult_a: float
    mult_b: float
    tick_a: float
    tick_b: float
    entry_start_min: int   # trading window entry start (minutes from midnight)
    entry_end_min: int     # trading window entry end
    flat_min: int          # force-flat time (session end)


def run_full_job(job: GridJob) -> list[dict]:
    """Run hedge + metrics + all signal/backtest combos in a single worker.
    Returns list of summary dicts (small, no DataFrames serialized)."""
    try:
        pair = SpreadPair(leg_a=Instrument(job.leg_a), leg_b=Instrument(job.leg_b))
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": f"No cache for {job.pair_name}"}]

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        idx = aligned.df.index

        # --- Hedge ratio (once per job) ---
        est = create_estimator("ols_rolling", window=job.ols_window, zscore_window=ZSCORE_WINDOWS[0])
        hr = est.estimate(aligned)
        beta = hr.beta.values
        spread = hr.spread

        # --- Metrics (once per job) ---
        profile_cfg = METRIC_PROFILES[job.profile_name]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)

        # --- Confidence (once per job, then threshold per min_conf) ---
        base_conf = ConfidenceConfig()
        confidence = compute_confidence(metrics, base_conf).values

        # --- Minutes array for window filter (once per job) ---
        minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

        # --- Z-scores for each window (once per job) ---
        zscores = {}
        for zw in ZSCORE_WINDOWS:
            mu = spread.rolling(zw).mean()
            sigma = spread.rolling(zw).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                zs = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
            zs = np.ascontiguousarray(zs, dtype=np.float64)
            zscores[zw] = zs

        # --- Loop over signal combos ---
        results = []

        for zw in ZSCORE_WINDOWS:
            zscore = zscores[zw]

            for z_entry, z_exit, z_stop in SIGNAL_COMBOS:
                # Generate signals (numba-compiled)
                raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)

                for min_conf in MIN_CONFIDENCES:
                    # Apply confidence filter (numba-compiled)
                    sig_conf = _apply_conf_filter_numba(raw_signals, confidence, min_conf)

                    # Apply entry window + flat EOD (numba-compiled)
                    sig = apply_window_filter_numba(
                        sig_conf, minutes,
                        job.entry_start_min, job.entry_end_min, job.flat_min,
                    )

                    # Run grid-optimized backtest (no equity curve)
                    bt = run_backtest_grid(
                        px_a, px_b, sig, beta,
                        mult_a=job.mult_a, mult_b=job.mult_b,
                        tick_a=job.tick_a, tick_b=job.tick_b,
                        slippage_ticks=1, commission=2.50,
                    )

                    results.append({
                        "pair": job.pair_name,
                        "ols_window": job.ols_window,
                        "zscore_window": zw,
                        "z_entry": z_entry,
                        "z_exit": z_exit,
                        "z_stop": z_stop,
                        "min_confidence": min_conf,
                        "profil": job.profile_name,
                        "trades": bt["trades"],
                        "win_rate": bt["win_rate"],
                        "pnl": bt["pnl"],
                        "profit_factor": bt["profit_factor"],
                        "avg_pnl_trade": bt["avg_pnl_trade"],
                        "avg_duration_bars": bt["avg_duration_bars"],
                    })

        return results

    except Exception as e:
        return [{"error": str(e), "pair": job.pair_name, "ols_window": job.ols_window}]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid search OLS backtest")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show job count without running")
    args = parser.parse_args()

    instruments = load_instruments()
    session = load_session()

    # Precompute trading window as minutes
    tw_start_min = session.trading_start.hour * 60 + session.trading_start.minute
    tw_end_min = session.trading_end.hour * 60 + session.trading_end.minute
    flat_min = session.session_end.hour * 60 + session.session_end.minute

    # Build jobs
    jobs = []
    for (leg_a, leg_b), ols_w, profile in product(PAIRS, OLS_WINDOWS, METRIC_PROFILES.keys()):
        pair_name = f"{leg_a}_{leg_b}"
        spec_a = instruments[leg_a]
        spec_b = instruments[leg_b]
        jobs.append(GridJob(
            pair_name=pair_name, leg_a=leg_a, leg_b=leg_b,
            ols_window=ols_w, profile_name=profile,
            mult_a=spec_a["multiplier"], mult_b=spec_b["multiplier"],
            tick_a=spec_a["tick_size"], tick_b=spec_b["tick_size"],
            entry_start_min=tw_start_min, entry_end_min=tw_end_min,
            flat_min=flat_min,
        ))

    signal_combos_per_job = len(ZSCORE_WINDOWS) * len(SIGNAL_COMBOS) * len(MIN_CONFIDENCES)
    total_backtests = len(jobs) * signal_combos_per_job

    log.info(f"Grid search OLS")
    log.info(f"  Jobs: {len(jobs)} (parallel with {args.workers} workers)")
    log.info(f"  Signal combos per job: {signal_combos_per_job}")
    log.info(f"  Total backtests: {total_backtests:,}")
    log.info(f"  Pairs: {[f'{a}_{b}' for a,b in PAIRS]}")
    log.info(f"  OLS windows: {OLS_WINDOWS}")
    log.info(f"  Z-score windows: {ZSCORE_WINDOWS}")
    log.info(f"  Signal combos: {len(SIGNAL_COMBOS)}")
    log.info(f"  Min confidences: {MIN_CONFIDENCES}")
    log.info(f"  Metric profiles: {list(METRIC_PROFILES.keys())}")

    if args.dry_run:
        log.info("Dry run -- exiting.")
        return

    t0 = time.time()
    all_results = []
    completed = 0
    errors = 0

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_full_job, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            completed += 1

            try:
                results = future.result()
            except Exception as e:
                errors += 1
                log.error(f"[{completed}/{len(jobs)}] CRASH {job.pair_name} ols={job.ols_window} {job.profile_name}: {e}")
                continue

            if results and "error" in results[0]:
                errors += 1
                log.error(f"[{completed}/{len(jobs)}] ERROR {job.pair_name}: {results[0]['error']}")
                continue

            all_results.extend(results)

            elapsed = time.time() - t0
            eta = (elapsed / completed) * (len(jobs) - completed) if completed > 0 else 0
            log.info(
                f"[{completed}/{len(jobs)}] {job.pair_name} ols={job.ols_window} {job.profile_name} | "
                f"{len(results)} configs | elapsed={elapsed:.0f}s ETA={eta:.0f}s"
            )
            sys.stdout.flush()

    phase1_time = time.time() - t0
    log.info(f"Complete: {completed}/{len(jobs)} jobs, {errors} errors, {phase1_time:.0f}s")
    log.info(f"Total results: {len(all_results):,}")

    # Save raw CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "grid_results_ols.csv"

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        log.info(f"[OUTPUT] Raw results -> {csv_path} ({len(df):,} rows)")
        generate_reports(df)
    else:
        log.warning("No results to save!")

    total_time = time.time() - t0
    log.info(f"Total elapsed: {total_time:.0f}s ({total_time/60:.1f}min)")


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

def generate_reports(df: pd.DataFrame):
    """Generate filtered reports and top-10 rankings."""

    filtered = df[(df["trades"] > 100) & (df["pnl"] > 0) & (df["avg_pnl_trade"] > 30)].copy()
    log.info(f"[REPORT] Filtered: {len(filtered):,} / {len(df):,} configs pass (trades>100, pnl>0, avg>$30)")

    if len(filtered) == 0:
        log.warning("[REPORT] No configs pass the filter. Relaxing to trades>50, pnl>0...")
        filtered = df[(df["trades"] > 50) & (df["pnl"] > 0)].copy()
        log.info(f"[REPORT] Relaxed filter: {len(filtered):,} configs")

    if len(filtered) == 0:
        log.warning("[REPORT] Still no configs pass. Showing top 20 by PnL regardless.")
        filtered = df.nlargest(20, "pnl").copy()

    def dedup_top(subset, sort_col, ascending=False, n=10):
        sorted_df = subset.sort_values(sort_col, ascending=ascending)
        seen = set()
        top = []
        for _, row in sorted_df.iterrows():
            key = (row["pair"], row["ols_window"], row["zscore_window"], row["z_entry"])
            if key not in seen:
                seen.add(key)
                top.append(row)
                if len(top) >= n:
                    break
        return pd.DataFrame(top)

    def norm(series):
        mn, mx = series.min(), series.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    if len(filtered) > 0:
        f = filtered.copy()

        # Score equilibre (no equity-derived metrics in grid mode)
        f["score_equilibre"] = (
            0.40 * norm(f["profit_factor"])
            + 0.25 * norm(f["win_rate"])
            + 0.25 * norm(f["avg_pnl_trade"])
            + 0.10 * norm(f["trades"])
        )

        trading_days = 5 * 252
        f["daily_pnl"] = f["pnl"] / trading_days
        f["score_daily"] = (f["daily_pnl"] / 300).clip(0, 1)
        f["score_consistency"] = ((f["win_rate"] - 40) / 30).clip(0, 1)
        f["score_pf"] = ((f["profit_factor"] - 1.0) / 2.0).clip(0, 1)
        f["score_propfirm"] = (
            0.40 * f["score_daily"]
            + 0.30 * f["score_consistency"]
            + 0.30 * f["score_pf"]
        )

        filtered = f

    rankings = [
        ("TOP 10 PNL", "pnl", False),
        ("TOP 10 PROFIT FACTOR", "profit_factor", False),
        ("TOP 10 EQUILIBRE", "score_equilibre", False),
        ("TOP 10 PROPFIRM", "score_propfirm", False),
    ]

    for title, col, asc in rankings:
        if col not in filtered.columns:
            continue
        top = dedup_top(filtered, col, ascending=asc)
        log.info(f"\n{'='*90}")
        log.info(f" {title}")
        log.info(f"{'='*90}")
        log.info(f" {'Pair':<8} {'OLS':>5} {'ZW':>3} {'Zent':>4} {'Zex':>4} {'Conf':>4} {'Prof':<10} "
                 f"{'Trd':>4} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Avg$':>6} {'AvgD':>5}")
        log.info("-" * 90)
        for _, r in top.iterrows():
            log.info(
                f" {r['pair']:<8} {r['ols_window']:>5} {r['zscore_window']:>3} "
                f"{r['z_entry']:>4.1f} {r['z_exit']:>4.1f} {r['min_confidence']:>4.0f} "
                f"{r['profil']:<10} {r['trades']:>4} {r['win_rate']:>5.1f} "
                f"${r['pnl']:>9,.0f} {r['profit_factor']:>5.2f} "
                f"${r['avg_pnl_trade']:>5,.0f} {r['avg_duration_bars']:>5.0f}"
            )

    log.info(f"\n{'='*90}")
    log.info(f" BEST CONFIG PER PAIR (by PF)")
    log.info(f"{'='*90}")
    for pair_name in [f"{a}_{b}" for a, b in PAIRS]:
        pair_df = filtered[filtered["pair"] == pair_name]
        if len(pair_df) == 0:
            log.info(f" {pair_name:<8} -- no profitable config")
            continue
        best = pair_df.loc[pair_df["profit_factor"].idxmax()]
        log.info(
            f" {best['pair']:<8} OLS={best['ols_window']:>5} ZW={best['zscore_window']:>2} "
            f"Zent={best['z_entry']:.1f} Zex={best['z_exit']:.1f} Conf={best['min_confidence']:.0f} "
            f"Prof={best['profil']:<10} "
            f"Trd={best['trades']:>4} Win={best['win_rate']:.1f}% "
            f"PnL=${best['pnl']:>,.0f} PF={best['profit_factor']:.2f}"
        )

    csv_path = OUTPUT_DIR / "grid_results_ols_filtered.csv"
    filtered.to_csv(csv_path, index=False)
    log.info(f"\n[OUTPUT] Filtered results -> {csv_path} ({len(filtered):,} rows)")


if __name__ == "__main__":
    main()
