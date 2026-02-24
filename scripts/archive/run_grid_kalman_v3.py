"""Grid search Kalman v3 — definitive with entry windows + finer z steps.

Based on v2 findings:
- warmup & gap_P_mult: ZERO impact -> fixed (200, 5.0)
- alpha sweet spot: 1e-7 to 5e-7, add 2.5e-7
- z_entry finer (0.0625 step), range narrowed to 1.25-2.25
- z_exit: add 0.125/0.375 intermediates
- z_stop: finer around 2.5-3.0
- min_confidence: add 64, 66 for transition resolution
- NEW: entry window as grid dimension (5 windows)
- 5 propfirm filtering profiles in report

Usage:
    python scripts/run_grid_kalman_v3.py --workers 10
    python scripts/run_grid_kalman_v3.py --dry-run
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

from src.backtest.engine import run_backtest_grid, run_backtest_vectorized
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
log = logging.getLogger("grid_kalman_v3")

OUTPUT_DIR = PROJECT_ROOT / "output"

# ──────────────────────────────────────────────────────────────────────
# Grid v3 — definitive parameters
# ──────────────────────────────────────────────────────────────────────

PAIR = ("NQ", "YM")

# alpha_ratio: sweet spot 1e-7 to 5e-7, add 2.5e-7
ALPHA_RATIOS = [1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7, 5e-7]

# warmup & gap_P_mult: FIXED (zero impact confirmed in v2)
FIXED_WARMUP = 200
FIXED_GAP_P_MULT = 5.0

# z_entry: finer steps 0.0625 in narrowed sweet spot 1.25-2.25
Z_ENTRIES = [1.25, 1.3125, 1.375, 1.4375, 1.5, 1.5625, 1.625, 1.6875,
             1.75, 1.8125, 1.875, 1.9375, 2.0, 2.0625, 2.125, 2.1875, 2.25]

# z_exit: add 0.125/0.375 intermediates
Z_EXITS = [0.0, 0.125, 0.25, 0.375, 0.50, 0.75, 1.00, 1.25, 1.50]

# z_stop: finer around 2.5-3.0
Z_STOPS = [2.25, 2.50, 2.625, 2.75, 2.875, 3.00, 3.25]

# min_confidence: add 64, 66 for transition resolution
MIN_CONFIDENCES = [50.0, 60.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 75.0]

# metric profiles
METRIC_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "moyen":      MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
}

# Entry windows: (label, start_hour, start_min, end_hour, end_min)
ENTRY_WINDOWS = [
    ("03:00-12:00", 3, 0, 12, 0),
    ("04:00-12:00", 4, 0, 12, 0),
    ("04:00-13:00", 4, 0, 13, 0),
    ("04:00-14:00", 4, 0, 14, 0),
    ("05:00-12:00", 5, 0, 12, 0),
]

# OLS Config E baseline
OLS_BASELINE = {
    "trades": 176, "win_rate": 68.2, "pnl": 23215, "profit_factor": 1.86,
    "avg_pnl_trade": 132, "avg_duration_bars": 5.9,
}


def load_instruments():
    with open(PROJECT_ROOT / "config" / "instruments.yaml") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────
# Job: 1 (alpha x profile x window) = 1 Kalman run + all signal combos
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KalmanJobV3:
    alpha_ratio: float
    profile_name: str
    window_label: str
    entry_start_min: int
    entry_end_min: int
    flat_min: int
    mult_a: float
    mult_b: float
    tick_a: float
    tick_b: float


def run_kalman_job_v3(job: KalmanJobV3) -> list[dict]:
    """Run Kalman hedge + all signal combos for one (alpha, profile, window)."""
    try:
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": "No cache for NQ_YM"}]

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        idx = aligned.df.index

        # --- Kalman hedge ratio ---
        est = create_estimator(
            "kalman",
            alpha_ratio=job.alpha_ratio,
            warmup=FIXED_WARMUP,
            gap_P_multiplier=FIXED_GAP_P_MULT,
        )
        hr = est.estimate(aligned)
        beta = hr.beta.values
        spread = hr.spread

        # Kalman z-score (innovation-based)
        zscore = hr.zscore.values
        zscore = np.ascontiguousarray(
            np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float64,
        )

        # --- Metrics + confidence ---
        profile_cfg = METRIC_PROFILES[job.profile_name]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        base_conf = ConfidenceConfig()
        confidence = compute_confidence(metrics, base_conf).values

        # --- Minutes array ---
        minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

        # --- Loop signal combos ---
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
                        sig_conf = _apply_conf_filter_numba(raw_signals, confidence, min_conf)
                        sig = apply_window_filter_numba(
                            sig_conf, minutes,
                            job.entry_start_min, job.entry_end_min, job.flat_min,
                        )

                        bt = run_backtest_grid(
                            px_a, px_b, sig, beta,
                            mult_a=job.mult_a, mult_b=job.mult_b,
                            tick_a=job.tick_a, tick_b=job.tick_b,
                            slippage_ticks=1, commission=2.50,
                        )

                        results.append({
                            "alpha_ratio": job.alpha_ratio,
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
        return [{"error": str(e), "alpha_ratio": job.alpha_ratio}]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grid search Kalman v3 NQ_YM (definitive)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show job count without running")
    args = parser.parse_args()

    instruments = load_instruments()
    spec_a = instruments["NQ"]
    spec_b = instruments["YM"]

    # Flat time is always 15:30 (session end)
    flat_min = 15 * 60 + 30

    # Build jobs: alpha x profile x window
    jobs = []
    for alpha, profile, (wlabel, wsh, wsm, weh, wem) in product(
        ALPHA_RATIOS, METRIC_PROFILES.keys(), ENTRY_WINDOWS
    ):
        jobs.append(KalmanJobV3(
            alpha_ratio=alpha,
            profile_name=profile,
            window_label=wlabel,
            entry_start_min=wsh * 60 + wsm,
            entry_end_min=weh * 60 + wem,
            flat_min=flat_min,
            mult_a=spec_a["multiplier"], mult_b=spec_b["multiplier"],
            tick_a=spec_a["tick_size"], tick_b=spec_b["tick_size"],
        ))

    # Count valid signal combos
    valid_combos = 0
    for ze in Z_ENTRIES:
        for zx in Z_EXITS:
            if zx >= ze:
                continue
            for zs in Z_STOPS:
                if zs <= ze:
                    continue
                valid_combos += 1
    combos_per_job = valid_combos * len(MIN_CONFIDENCES)
    total_backtests = len(jobs) * combos_per_job

    log.info("Grid search KALMAN v3 -- NQ_YM (definitive)")
    log.info(f"  Jobs: {len(jobs)} ({len(ALPHA_RATIOS)} alpha x {len(METRIC_PROFILES)} profil x {len(ENTRY_WINDOWS)} window)")
    log.info(f"  Workers: {args.workers}")
    log.info(f"  Fixed: warmup={FIXED_WARMUP}, gap_P_mult={FIXED_GAP_P_MULT}")
    log.info(f"  Valid signal combos per job: {combos_per_job}")
    log.info(f"  Total backtests: {total_backtests:,}")
    log.info(f"  Alpha ratios: {ALPHA_RATIOS}")
    log.info(f"  z_entry: {Z_ENTRIES[0]} -> {Z_ENTRIES[-1]} step 0.0625 ({len(Z_ENTRIES)} values)")
    log.info(f"  z_exit: {Z_EXITS} ({len(Z_EXITS)} values)")
    log.info(f"  z_stop: {Z_STOPS} ({len(Z_STOPS)} values)")
    log.info(f"  min_confidence: {MIN_CONFIDENCES} ({len(MIN_CONFIDENCES)} values)")
    log.info(f"  Entry windows: {[w[0] for w in ENTRY_WINDOWS]}")
    log.info(f"  OLS baseline: PF {OLS_BASELINE['profit_factor']}, ${OLS_BASELINE['pnl']:,}")

    if args.dry_run:
        log.info("Dry run -- exiting.")
        return

    t0 = time.time()
    all_results = []
    completed = 0
    errors = 0

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_kalman_job_v3, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            completed += 1

            try:
                results = future.result()
            except Exception as e:
                errors += 1
                log.error(f"[{completed}/{len(jobs)}] CRASH a={job.alpha_ratio:.1e} {job.profile_name} {job.window_label}: {e}")
                continue

            if results and "error" in results[0]:
                errors += 1
                log.error(f"[{completed}/{len(jobs)}] ERROR: {results[0]['error']}")
                continue

            all_results.extend(results)

            elapsed = time.time() - t0
            eta = (elapsed / completed) * (len(jobs) - completed) if completed > 0 else 0
            if completed % 10 == 0 or completed == len(jobs):
                log.info(
                    f"[{completed}/{len(jobs)}] a={job.alpha_ratio:.1e} "
                    f"{job.profile_name} {job.window_label} | {len(results):,} | "
                    f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
                )
                sys.stdout.flush()

    total_time = time.time() - t0
    log.info(f"Complete: {completed}/{len(jobs)} jobs, {errors} errors, {total_time:.0f}s")
    log.info(f"Total results: {len(all_results):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "NQ_YM" / "grid_kalman_v3.csv"

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        log.info(f"[OUTPUT] Raw results -> {csv_path} ({len(df):,} rows)")
        generate_reports(df)
    else:
        log.warning("No results to save!")

    log.info(f"Total elapsed: {total_time:.0f}s ({total_time/60:.1f}min)")


# ──────────────────────────────────────────────────────────────────────
# Propfirm profiles
# ──────────────────────────────────────────────────────────────────────

PROPFIRM_PROFILES = {
    "1_Sniper": {
        "label": "CONSERVATIVE SNIPER -- max quality, evaluation safety",
        "trades_min": 30, "trades_max": 999999,
        "pf_min": 2.5, "wr_min": 75.0, "pnl_min": 0,
        "avg_pnl_min": 0, "avg_dur_max": 999,
    },
    "2_Steady": {
        "label": "STEADY EARNER -- consistent income, payout eligible",
        "trades_min": 100, "trades_max": 300,
        "pf_min": 1.5, "wr_min": 65.0, "pnl_min": 0,
        "avg_pnl_min": 80, "avg_dur_max": 999,
    },
    "3_Volume": {
        "label": "VOLUME GRINDER -- max trades, statistical power",
        "trades_min": 200, "trades_max": 999999,
        "pf_min": 1.2, "wr_min": 60.0, "pnl_min": 0,
        "avg_pnl_min": 30, "avg_dur_max": 999,
    },
    "4_RiskAdj": {
        "label": "RISK-ADJUSTED CHAMPION -- best Sharpe/Calmar proxy",
        "trades_min": 50, "trades_max": 999999,
        "pf_min": 1.8, "wr_min": 65.0, "pnl_min": 0,
        "avg_pnl_min": 0, "avg_dur_max": 999,
    },
    "5_Balanced": {
        "label": "BALANCED COMPOSITE -- volume + quality + complementarity",
        "trades_min": 80, "trades_max": 250,
        "pf_min": 1.3, "wr_min": 65.0, "pnl_min": 10000,
        "avg_pnl_min": 50, "avg_dur_max": 999,
    },
}


def apply_profile(df, profile):
    """Filter DataFrame by propfirm profile criteria."""
    mask = (
        (df["trades"] >= profile["trades_min"]) &
        (df["trades"] <= profile["trades_max"]) &
        (df["profit_factor"] >= profile["pf_min"]) &
        (df["win_rate"] >= profile["wr_min"]) &
        (df["pnl"] >= profile["pnl_min"]) &
        (df["avg_pnl_trade"] >= profile["avg_pnl_min"]) &
        (df["avg_duration_bars"] <= profile["avg_dur_max"])
    )
    return df[mask].copy()


# ──────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────

def generate_reports(df: pd.DataFrame):
    """Reports with propfirm profiles and per-window analysis."""

    # Base filter: profitable
    base = df[(df["trades"] > 20) & (df["pnl"] > 0) & (df["profit_factor"] > 1.0)].copy()
    log.info(f"[REPORT] Base filter: {len(base):,} / {len(df):,} (trades>20, pnl>0, PF>1)")

    log.info(f"\n{'='*130}")
    log.info(f" OLS BASELINE: {OLS_BASELINE['trades']} trades, WR {OLS_BASELINE['win_rate']}%, "
             f"PnL ${OLS_BASELINE['pnl']:,}, PF {OLS_BASELINE['profit_factor']}")
    log.info(f"{'='*130}")

    beats_pf = base[base["profit_factor"] > OLS_BASELINE["profit_factor"]]
    log.info(f" Configs beating OLS PF: {len(beats_pf):,}")

    # ── Top 10 per window ──
    log.info(f"\n{'='*130}")
    log.info(f" TOP 10 PNL PER ENTRY WINDOW")
    log.info(f"{'='*130}")

    for wlabel, _, _, _, _ in ENTRY_WINDOWS:
        wsub = base[base["window"] == wlabel]
        if len(wsub) == 0:
            log.info(f"\n  [{wlabel}] -- no profitable configs")
            continue
        top = wsub.nlargest(10, "pnl")
        log.info(f"\n  [{wlabel}] {len(wsub):,} profitable configs")
        _print_table(top)

    # ── Top 10 PF overall ──
    log.info(f"\n{'='*130}")
    log.info(f" TOP 10 PROFIT FACTOR (all windows)")
    log.info(f"{'='*130}")
    top_pf = base[base["trades"] >= 30].nlargest(10, "profit_factor")
    _print_table(top_pf)

    # ── Top 10 PnL overall ──
    log.info(f"\n{'='*130}")
    log.info(f" TOP 10 PNL (all windows)")
    log.info(f"{'='*130}")
    top_pnl = base[base["trades"] >= 30].nlargest(10, "pnl")
    _print_table(top_pnl)

    # ── Best per alpha ──
    log.info(f"\n{'='*130}")
    log.info(f" BEST PER ALPHA (by PF, trades>=30)")
    log.info(f"{'='*130}")
    for alpha in ALPHA_RATIOS:
        sub = base[(base["alpha_ratio"] == alpha) & (base["trades"] >= 30)]
        if len(sub) == 0:
            log.info(f"  a={alpha:.1e} -- no configs")
            continue
        best = sub.loc[sub["profit_factor"].idxmax()]
        m = "*" if best["profit_factor"] > OLS_BASELINE["profit_factor"] else " "
        log.info(
            f" {m}a={alpha:.1e} {best['profil']:<10} {best['window']:<12} | "
            f"ze={best['z_entry']:.4f} zx={best['z_exit']:.3f} zs={best['z_stop']:.3f} "
            f"c={best['min_confidence']:.0f} | "
            f"Trd={best['trades']:>4} WR={best['win_rate']:.1f}% PF={best['profit_factor']:.2f} "
            f"PnL=${best['pnl']:>,.0f} Avg=${best['avg_pnl_trade']:.0f}"
        )

    # ── Best per window ──
    log.info(f"\n{'='*130}")
    log.info(f" BEST PER WINDOW (by PF, trades>=30)")
    log.info(f"{'='*130}")
    for wlabel, _, _, _, _ in ENTRY_WINDOWS:
        sub = base[(base["window"] == wlabel) & (base["trades"] >= 30)]
        if len(sub) == 0:
            log.info(f"  {wlabel} -- no configs")
            continue
        best = sub.loc[sub["profit_factor"].idxmax()]
        log.info(
            f"  {wlabel:<12} a={best['alpha_ratio']:.1e} {best['profil']:<10} | "
            f"ze={best['z_entry']:.4f} zx={best['z_exit']:.3f} zs={best['z_stop']:.3f} "
            f"c={best['min_confidence']:.0f} | "
            f"Trd={best['trades']:>4} WR={best['win_rate']:.1f}% PF={best['profit_factor']:.2f} "
            f"PnL=${best['pnl']:>,.0f}"
        )

    # ── Propfirm profiles ──
    log.info(f"\n{'='*130}")
    log.info(f" PROPFIRM PROFILES")
    log.info(f"{'='*130}")

    for pname, profile in PROPFIRM_PROFILES.items():
        filtered = apply_profile(base, profile)
        log.info(f"\n  --- {pname}: {profile['label']} ---")
        log.info(f"  Filters: trades[{profile['trades_min']}-{profile['trades_max']}] "
                 f"PF>={profile['pf_min']} WR>={profile['wr_min']}% "
                 f"Avg$>={profile['avg_pnl_min']}")
        log.info(f"  Matching configs: {len(filtered):,}")

        if len(filtered) == 0:
            log.info("  No matching configs.")
            continue

        # Top 10 by PnL, deduped by (alpha, z_entry, z_exit, window)
        seen = set()
        top = []
        for _, row in filtered.sort_values("pnl", ascending=False).iterrows():
            key = (row["alpha_ratio"], row["z_entry"], row["z_exit"], row["window"])
            if key not in seen:
                seen.add(key)
                top.append(row)
                if len(top) >= 10:
                    break
        if top:
            top_df = pd.DataFrame(top)
            _print_table(top_df)

    # Save filtered CSV
    csv_path = OUTPUT_DIR / "NQ_YM" / "grid_kalman_v3_filtered.csv"
    base.to_csv(csv_path, index=False)
    log.info(f"\n[OUTPUT] Filtered -> {csv_path} ({len(base):,} rows)")


def _print_table(df):
    """Print a standard table of results."""
    log.info(f"  {'Alpha':>8} {'Prof':<10} {'Window':<12} {'Zent':>6} {'Zex':>5} {'Zst':>5} "
             f"{'Conf':>4} {'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Avg$':>7} {'AvgD':>5}")
    log.info("  " + "-" * 118)
    for _, r in df.iterrows():
        m = "*" if r["profit_factor"] > OLS_BASELINE["profit_factor"] else " "
        log.info(
            f" {m}{r['alpha_ratio']:>8.1e} {r['profil']:<10} {r['window']:<12} "
            f"{r['z_entry']:>6.4f} {r['z_exit']:>5.3f} {r['z_stop']:>5.3f} {r['min_confidence']:>4.0f} "
            f"{r['trades']:>5} {r['win_rate']:>5.1f} "
            f"${r['pnl']:>9,.0f} {r['profit_factor']:>5.2f} "
            f"${r['avg_pnl_trade']:>6,.0f} {r['avg_duration_bars']:>5.1f}"
        )


if __name__ == "__main__":
    main()
