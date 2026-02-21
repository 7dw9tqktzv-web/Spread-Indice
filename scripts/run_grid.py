"""Grid search backtest for OLS rolling method across all pairs and parameter combinations.

Usage:
    python scripts/run_grid.py --workers 20
    python scripts/run_grid.py --workers 10 --dry-run
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import BacktestConfig, BacktestEngine, InstrumentSpec
from src.backtest.performance import compute_performance
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import ConfidenceConfig, apply_confidence_filter, apply_trading_window_filter
from src.signals.generator import SignalConfig, SignalGenerator
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.utils.time_utils import SessionConfig, parse_session_config

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

OLS_WINDOWS = [1320, 2640, 3960, 5280, 6600, 7920]  # 5j→30j (264 bars/j)

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


def build_spec(instruments, name):
    s = instruments[name]
    return InstrumentSpec(multiplier=s["multiplier"], tick_size=s["tick_size"], tick_value=s["tick_value"])


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Pre-compute metrics (slow, parallelized)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MetricJob:
    pair_name: str
    leg_a: str
    leg_b: str
    ols_window: int
    profile_name: str


def compute_metric_job(job: MetricJob) -> dict:
    """Compute hedge + metrics for one (pair, ols_window, profile) combo.
    Returns dict with all intermediate data needed for signal loop."""
    try:
        pair = SpreadPair(leg_a=Instrument(job.leg_a), leg_b=Instrument(job.leg_b))
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return {"error": f"No cache for {job.pair_name}", "job": job}

        close_a = aligned.df["close_a"]
        close_b = aligned.df["close_b"]

        # Compute hedge ONCE (beta/spread don't depend on zscore_window)
        est_base = create_estimator("ols_rolling", window=job.ols_window, zscore_window=ZSCORE_WINDOWS[0])
        hr_base = est_base.estimate(aligned)
        beta = hr_base.beta
        spread = hr_base.spread

        # Compute z-score for each zscore_window (reuse beta/spread)
        hedge_results = {}
        for zw in ZSCORE_WINDOWS:
            mu = spread.rolling(zw).mean()
            sigma = spread.rolling(zw).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan)
            hedge_results[zw] = {
                "beta": beta,
                "spread": spread,
                "zscore": zscore,
            }

        # Metrics (depends only on ols_window + profile, computed once)
        profile_cfg = METRIC_PROFILES[job.profile_name]
        metrics = compute_all_metrics(spread, close_a, close_b, profile_cfg)

        return {
            "job": job,
            "hedge_results": hedge_results,
            "metrics": metrics,
            "close_a": close_a,
            "close_b": close_b,
        }
    except Exception as e:
        return {"error": str(e), "job": job}


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Signal + backtest loop (fast, sequential per metric combo)
# ──────────────────────────────────────────────────────────────────────

def run_signal_loop(metric_data: dict, session: SessionConfig, instruments: dict) -> list[dict]:
    """Run all signal/confidence combos for one pre-computed metric result."""
    job = metric_data["job"]
    hedge_results = metric_data["hedge_results"]
    metrics_df = metric_data["metrics"]
    close_a = metric_data["close_a"]
    close_b = metric_data["close_b"]

    spec_a = build_spec(instruments, job.leg_a)
    spec_b = build_spec(instruments, job.leg_b)

    bt_config = BacktestConfig(
        initial_capital=100_000.0,
        commission_per_contract=2.50,
        slippage_ticks=1,
    )
    engine = BacktestEngine(config=bt_config)

    results = []

    for zw in ZSCORE_WINDOWS:
        hr = hedge_results[zw]
        beta = hr["beta"]
        zscore = hr["zscore"]

        for z_entry, z_exit, z_stop in SIGNAL_COMBOS:
            # Generate signals once per (zscore_window, signal_combo)
            sig_cfg = SignalConfig(z_entry=z_entry, z_exit=z_exit, z_stop=z_stop)
            gen = SignalGenerator(config=sig_cfg)
            raw_signals = gen.generate(zscore)

            for min_conf in MIN_CONFIDENCES:
                conf_cfg = ConfidenceConfig(min_confidence=min_conf)
                filtered = apply_confidence_filter(raw_signals, metrics_df, conf_cfg)
                final = apply_trading_window_filter(filtered, session)

                bt_result = engine.run(close_a, close_b, final, beta, spec_a, spec_b)
                perf = compute_performance(bt_result)

                # Trade durations
                if bt_result.trades:
                    durations = [(t.exit_bar - t.entry_bar) for t in bt_result.trades]
                    avg_dur = np.mean(durations)
                    max_dur = max(durations)
                else:
                    avg_dur = 0
                    max_dur = 0

                results.append({
                    "pair": job.pair_name,
                    "ols_window": job.ols_window,
                    "zscore_window": zw,
                    "z_entry": z_entry,
                    "z_exit": z_exit,
                    "z_stop": z_stop,
                    "min_confidence": min_conf,
                    "profil": job.profile_name,
                    "trades": perf.num_trades,
                    "win_rate": round(perf.win_rate, 1),
                    "pnl": round(perf.total_pnl, 2),
                    "profit_factor": round(perf.profit_factor, 2),
                    "avg_pnl_trade": round(perf.avg_pnl_per_trade, 2),
                    "sharpe": round(perf.sharpe_ratio, 2),
                    "calmar": round(perf.calmar_ratio, 2),
                    "max_dd_pct": round(perf.max_drawdown_pct, 2),
                    "max_dd_dollar": round(perf.max_drawdown_pct * 1000, 2),  # 100k capital
                    "avg_duration_bars": round(avg_dur, 1),
                    "max_duration_bars": int(max_dur),
                })

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid search OLS backtest")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show job count without running")
    args = parser.parse_args()

    # Build metric jobs
    jobs = []
    for (leg_a, leg_b), ols_w, profile in product(PAIRS, OLS_WINDOWS, METRIC_PROFILES.keys()):
        pair_name = f"{leg_a}_{leg_b}"
        jobs.append(MetricJob(pair_name=pair_name, leg_a=leg_a, leg_b=leg_b,
                              ols_window=ols_w, profile_name=profile))

    signal_combos_per_job = len(ZSCORE_WINDOWS) * len(SIGNAL_COMBOS) * len(MIN_CONFIDENCES)
    total_backtests = len(jobs) * signal_combos_per_job

    log.info(f"Grid search OLS")
    log.info(f"  Metric jobs: {len(jobs)} (parallel with {args.workers} workers)")
    log.info(f"  Signal combos per job: {signal_combos_per_job}")
    log.info(f"  Total backtests: {total_backtests:,}")
    log.info(f"  Pairs: {[f'{a}_{b}' for a,b in PAIRS]}")
    log.info(f"  OLS windows: {OLS_WINDOWS}")
    log.info(f"  Z-score windows: {ZSCORE_WINDOWS}")
    log.info(f"  Signal combos: {len(SIGNAL_COMBOS)}")
    log.info(f"  Min confidences: {MIN_CONFIDENCES}")
    log.info(f"  Metric profiles: {list(METRIC_PROFILES.keys())}")

    if args.dry_run:
        log.info("Dry run — exiting.")
        return

    instruments = load_instruments()
    session = load_session()

    # Phase 1: parallel metric computation
    log.info(f"[PHASE 1] Computing {len(jobs)} metric jobs with {args.workers} workers...")
    t0 = time.time()

    all_results = []
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compute_metric_job, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            completed += 1

            try:
                metric_data = future.result()
            except Exception as e:
                errors += 1
                log.error(f"[PHASE 1] {completed}/{len(jobs)} CRASH {job.pair_name} ols={job.ols_window} {job.profile_name}: {e}")
                continue

            if "error" in metric_data:
                errors += 1
                log.error(f"[PHASE 1] {completed}/{len(jobs)} ERROR {job.pair_name} ols={job.ols_window} {job.profile_name}: {metric_data['error']}")
                continue

            # Phase 2: signal loop (fast, in main process)
            results = run_signal_loop(metric_data, session, instruments)
            all_results.extend(results)

            elapsed = time.time() - t0
            eta_remaining = (elapsed / completed) * (len(jobs) - completed) if completed > 0 else 0
            log.info(
                f"[PHASE 1] {completed}/{len(jobs)} done | "
                f"{job.pair_name} ols={job.ols_window} {job.profile_name} | "
                f"{len(results)} configs | "
                f"elapsed={elapsed:.0f}s ETA={eta_remaining:.0f}s"
            )

    phase1_time = time.time() - t0
    log.info(f"[PHASE 1] Complete: {completed}/{len(jobs)} jobs, {errors} errors, {phase1_time:.0f}s")
    log.info(f"[PHASE 1] Total results: {len(all_results):,}")

    # Save raw CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "grid_results_ols.csv"

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        log.info(f"[OUTPUT] Raw results -> {csv_path} ({len(df):,} rows)")

        # Apply filters and generate reports
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

    # Filter: trades > 100, pnl > 0, avg_pnl_trade > 30
    filtered = df[(df["trades"] > 100) & (df["pnl"] > 0) & (df["avg_pnl_trade"] > 30)].copy()
    log.info(f"[REPORT] Filtered: {len(filtered):,} / {len(df):,} configs pass (trades>100, pnl>0, avg>$30)")

    if len(filtered) == 0:
        log.warning("[REPORT] No configs pass the filter. Relaxing to trades>50, pnl>0...")
        filtered = df[(df["trades"] > 50) & (df["pnl"] > 0)].copy()
        log.info(f"[REPORT] Relaxed filter: {len(filtered):,} configs")

    if len(filtered) == 0:
        log.warning("[REPORT] Still no configs pass. Showing top 10 by PnL regardless.")
        filtered = df.nlargest(20, "pnl").copy()

    # Deduplicate by pattern: group by (pair, ols_window, zscore_window, z_entry) and keep best min_confidence
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

    # Normalize helper
    def norm(series):
        mn, mx = series.min(), series.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    # Scoring columns
    if len(filtered) > 0:
        f = filtered.copy()

        # Équilibré score
        f["score_equilibre"] = (
            0.30 * norm(f["sharpe"])
            + 0.25 * norm(f["profit_factor"])
            + 0.20 * norm(f["calmar"])
            + 0.15 * norm(f["win_rate"])
            + 0.10 * norm(f["trades"])
        )

        # Propfirm score
        trading_days = 5 * 252  # ~5 years
        f["daily_pnl"] = f["pnl"] / trading_days
        f["score_daily"] = (f["daily_pnl"] / 300).clip(0, 1)
        f["score_dd"] = (1 - f["max_dd_dollar"] / 5000).clip(0, 1)
        f["score_consistency"] = ((f["win_rate"] - 40) / 30).clip(0, 1)
        f["score_pf"] = ((f["profit_factor"] - 1.0) / 2.0).clip(0, 1)
        f["score_propfirm"] = (
            0.30 * f["score_daily"]
            + 0.30 * f["score_dd"]
            + 0.20 * f["score_consistency"]
            + 0.20 * f["score_pf"]
        )

        filtered = f

    # Print rankings
    rankings = [
        ("TOP 10 PNL", "pnl", False),
        ("TOP 10 SHARPE", "sharpe", False),
        ("TOP 10 CALMAR", "calmar", False),
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
                 f"{'Trd':>4} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Shrp':>5} {'DD%':>5} {'AvgD':>5}")
        log.info("-" * 90)
        for _, r in top.iterrows():
            log.info(
                f" {r['pair']:<8} {r['ols_window']:>5} {r['zscore_window']:>3} "
                f"{r['z_entry']:>4.1f} {r['z_exit']:>4.1f} {r['min_confidence']:>4.0f} "
                f"{r['profil']:<10} {r['trades']:>4} {r['win_rate']:>5.1f} "
                f"${r['pnl']:>9,.0f} {r['profit_factor']:>5.2f} {r['sharpe']:>5.2f} "
                f"{r['max_dd_pct']:>5.1f} {r['avg_duration_bars']:>5.0f}"
            )

    # Best config per pair
    log.info(f"\n{'='*90}")
    log.info(f" BEST CONFIG PER PAIR (by Sharpe)")
    log.info(f"{'='*90}")
    for pair_name in [f"{a}_{b}" for a, b in PAIRS]:
        pair_df = filtered[filtered["pair"] == pair_name]
        if len(pair_df) == 0:
            log.info(f" {pair_name:<8} — no profitable config")
            continue
        best = pair_df.loc[pair_df["sharpe"].idxmax()]
        log.info(
            f" {best['pair']:<8} OLS={best['ols_window']:>5} ZW={best['zscore_window']:>2} "
            f"Zent={best['z_entry']:.1f} Zex={best['z_exit']:.1f} Conf={best['min_confidence']:.0f} "
            f"Prof={best['profil']:<10} "
            f"Trd={best['trades']:>4} Win={best['win_rate']:.1f}% "
            f"PnL=${best['pnl']:>,.0f} PF={best['profit_factor']:.2f} Shrp={best['sharpe']:.2f} "
            f"DD={best['max_dd_pct']:.1f}%"
        )

    # Save filtered CSV
    csv_path = OUTPUT_DIR / "grid_results_ols_filtered.csv"
    filtered.to_csv(csv_path, index=False)
    log.info(f"\n[OUTPUT] Filtered results -> {csv_path} ({len(filtered):,} rows)")


if __name__ == "__main__":
    main()
