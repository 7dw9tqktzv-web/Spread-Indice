"""Grid search Kalman v2 -- NQ_RTY with CORRECT confidence weights.

Phase 9 grid used NQ_YM default weights (ADF 40%, Hurst 25%, Corr 20%, HL 15%).
Phase 11 ablation showed NQ_RTY needs: ADF 50%, Hurst 30%, Corr 20%, HL 0%.
This grid re-runs with correct weights + expanded profiles from Phase 11.

Goal: find a Kalman config for Sierra textbox (discretionary bias indicator).
Criteria: L/S symmetry, yearly stability, complementarity with OLS #8/#6.

Usage:
    python scripts/grid_kalman_v2_NQ_RTY.py --workers 10
    python scripts/grid_kalman_v2_NQ_RTY.py --dry-run
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("grid_kalman_v2_nq_rty")

OUTPUT_DIR = PROJECT_ROOT / "output" / "NQ_RTY"

# ======================================================================
# NQ_RTY CORRECT confidence weights (Phase 11 ablation)
# ======================================================================

NQ_RTY_CONF = ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00)

# ======================================================================
# Grid parameters -- focused from Phase 9 sweet spots
# ======================================================================

PAIR = ("NQ", "RTY")

# Alpha: Phase 9 sweet spot 1.5e-7 to 5e-7
ALPHA_RATIOS = [1e-7, 1.5e-7, 2e-7, 2.5e-7, 3e-7, 5e-7]

FIXED_WARMUP = 200
FIXED_GAP_P_MULT = 5.0

# z_entry: 1.25 to 2.25 step 0.125 (coarser than Phase 9 but sufficient)
Z_ENTRIES = [1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25]

# z_exit: focused
Z_EXITS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

# z_stop: focused
Z_STOPS = [2.25, 2.5, 2.75, 3.0, 3.25]

# min_confidence: with correct weights, test range
MIN_CONFIDENCES = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]

# Profiles: Phase 9 originals + Phase 11 proven profiles
METRIC_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "p16_80":     MetricsConfig(adf_window=16, hurst_window=80, halflife_window=16, correlation_window=8),
    "court":      MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "p28_144":    MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p36_96":     MetricsConfig(adf_window=36, hurst_window=96, halflife_window=36, correlation_window=9),
    "moyen":      MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "p48_128":    MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
}

# Windows: Phase 9 + 06:00-14:00 from Phase 11 OLS success
ENTRY_WINDOWS = [
    ("03:00-12:00", 3, 0, 12, 0),
    ("04:00-13:00", 4, 0, 13, 0),
    ("05:00-12:00", 5, 0, 12, 0),
    ("06:00-14:00", 6, 0, 14, 0),
]


def load_instruments():
    with open(PROJECT_ROOT / "config" / "instruments.yaml") as f:
        return yaml.safe_load(f)


# ======================================================================
# Phase 1: Grid search
# ======================================================================

@dataclass
class KalmanJob:
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


def run_kalman_job(job: KalmanJob) -> list[dict]:
    """Run Kalman hedge + all signal combos for one (alpha, profile, window)."""
    try:
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": "No cache for NQ_RTY"}]

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

        # --- Metrics + confidence with CORRECT NQ_RTY weights ---
        profile_cfg = METRIC_PROFILES[job.profile_name]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, NQ_RTY_CONF).values

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


# ======================================================================
# Phase 2: Full backtest on top candidates (L/S split, yearly, IS/OOS)
# ======================================================================

def run_full_analysis(top_configs: pd.DataFrame, n_top: int = 20):
    """Re-run top configs with full engine for L/S, yearly, IS/OOS."""
    log.info(f"\n{'='*120}")
    log.info(f" PHASE 2: FULL ANALYSIS ON TOP {n_top} CANDIDATES")
    log.info(f"{'='*120}")

    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    instruments = load_instruments()
    spec_a, spec_b = instruments["NQ"], instruments["RTY"]

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    flat_min = 15 * 60 + 30
    n_total = len(idx)
    # IS/OOS 60/40 split
    is_end = int(n_total * 0.60)

    results = []

    for i, (_, row) in enumerate(top_configs.head(n_top).iterrows()):
        alpha = row["alpha_ratio"]
        profile_name = row["profil"]
        window = row["window"]
        ze, zx, zs = row["z_entry"], row["z_exit"], row["z_stop"]
        min_conf = row["min_confidence"]

        # Parse window
        parts = window.split("-")
        sh, sm = int(parts[0].split(":")[0]), int(parts[0].split(":")[1])
        eh, em = int(parts[1].split(":")[0]), int(parts[1].split(":")[1])
        entry_start_min = sh * 60 + sm
        entry_end_min = eh * 60 + em

        # Kalman
        est = create_estimator("kalman", alpha_ratio=alpha, warmup=FIXED_WARMUP, gap_P_multiplier=FIXED_GAP_P_MULT)
        hr = est.estimate(aligned)
        beta = hr.beta.values
        spread = hr.spread
        zscore = hr.zscore.values
        zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

        # Metrics + confidence
        profile_cfg = METRIC_PROFILES[profile_name]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, NQ_RTY_CONF).values

        # Signals
        raw = generate_signals_numba(zscore, ze, zx, zs)
        sig_conf = _apply_conf_filter_numba(raw, confidence, min_conf)
        sig = apply_window_filter_numba(sig_conf, minutes, entry_start_min, entry_end_min, flat_min)

        # Full backtest
        bt_full = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            mult_a=spec_a["multiplier"], mult_b=spec_b["multiplier"],
            tick_a=spec_a["tick_size"], tick_b=spec_b["tick_size"],
            slippage_ticks=1, commission=2.50,
        )

        num_trades = bt_full["trades"]
        if num_trades == 0:
            continue

        # MaxDD from equity
        equity = bt_full["equity"]
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = float(dd.min())

        # L/S split
        trade_sides = bt_full["trade_sides"]
        trade_pnls = bt_full["trade_pnls"]
        trade_entry_bars = bt_full["trade_entry_bars"]

        long_mask = trade_sides > 0
        long_count = int(long_mask.sum())
        long_pct = long_count / num_trades * 100
        long_pnl = float(trade_pnls[long_mask].sum())
        short_pnl = float(trade_pnls[~long_mask].sum())

        # Yearly decomposition
        yearly = {}
        for j in range(num_trades):
            yr = idx[trade_entry_bars[j]].year
            if yr not in yearly:
                yearly[yr] = {"pnl": 0.0, "trades": 0, "wins": 0}
            yearly[yr]["pnl"] += trade_pnls[j]
            yearly[yr]["trades"] += 1
            if trade_pnls[j] > 0:
                yearly[yr]["wins"] += 1

        neg_years = [yr for yr, v in yearly.items() if v["pnl"] < 0]
        yearly_str = " | ".join(f"{yr}:${v['pnl']:,.0f}({v['trades']}t)" for yr, v in sorted(yearly.items()))

        # IS/OOS
        sig_is = sig[:is_end].copy()
        sig_oos = sig[is_end:].copy()
        bt_is = run_backtest_vectorized(
            px_a[:is_end], px_b[:is_end], sig_is, beta[:is_end],
            mult_a=spec_a["multiplier"], mult_b=spec_b["multiplier"],
            tick_a=spec_a["tick_size"], tick_b=spec_b["tick_size"],
            slippage_ticks=1, commission=2.50,
        )
        bt_oos = run_backtest_vectorized(
            px_a[is_end:], px_b[is_end:], sig_oos, beta[is_end:],
            mult_a=spec_a["multiplier"], mult_b=spec_b["multiplier"],
            tick_a=spec_a["tick_size"], tick_b=spec_b["tick_size"],
            slippage_ticks=1, commission=2.50,
        )

        is_pf = bt_is["profit_factor"] if bt_is["trades"] >= 10 else 0
        oos_pf = bt_oos["profit_factor"] if bt_oos["trades"] >= 10 else 0
        oos_go = "GO" if oos_pf > 1.0 and bt_oos["trades"] >= 10 else "FAIL"

        label = f"a={alpha:.1e}_{profile_name}_{window}_ze={ze}_zx={zx}_zs={zs}_c={min_conf:.0f}"

        result = {
            "label": label,
            "alpha": alpha,
            "profil": profile_name,
            "window": window,
            "z_entry": ze,
            "z_exit": zx,
            "z_stop": zs,
            "conf": min_conf,
            "trades": num_trades,
            "wr": bt_full["win_rate"],
            "pnl": bt_full["pnl"],
            "pf": bt_full["profit_factor"],
            "max_dd": max_dd,
            "long_pct": round(long_pct, 1),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "is_pf": round(is_pf, 2),
            "oos_pf": round(oos_pf, 2),
            "oos_go": oos_go,
            "neg_years": len(neg_years),
            "neg_years_list": neg_years,
            "yearly": yearly_str,
        }
        results.append(result)

        sym = "OK" if 35 <= long_pct <= 65 else "BIAS"
        log.info(
            f"  [{i+1:>2}] {profile_name:<10} {window:<12} a={alpha:.1e} "
            f"ze={ze:.3f} zx={zx:.3f} zs={zs:.3f} c={min_conf:.0f} | "
            f"Trd={num_trades:>3} WR={bt_full['win_rate']:.1f}% PF={bt_full['profit_factor']:.2f} "
            f"PnL=${bt_full['pnl']:>,.0f} MaxDD=${max_dd:>,.0f} | "
            f"L/S={long_pct:.0f}%/{100-long_pct:.0f}% [{sym}] | "
            f"IS={is_pf:.2f} OOS={oos_pf:.2f} [{oos_go}] | NegYr={len(neg_years)}"
        )

    return pd.DataFrame(results)


def extract_trades(sig, equity, idx):
    """Extract individual trades with side, PnL, timing."""
    trades = []
    in_trade = False
    entry_idx = 0
    entry_side = 0

    for i in range(1, len(sig)):
        if not in_trade and sig[i] != 0:
            in_trade = True
            entry_idx = i
            entry_side = sig[i]
        elif in_trade and (sig[i] == 0 or sig[i] != entry_side):
            pnl = equity[i] - equity[entry_idx]
            trades.append({
                "entry_time": idx[entry_idx],
                "exit_time": idx[i],
                "side": "LONG" if entry_side > 0 else "SHORT",
                "pnl": pnl,
                "duration": i - entry_idx,
            })
            in_trade = False
            if sig[i] != 0:
                in_trade = True
                entry_idx = i
                entry_side = sig[i]

    return trades


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Grid Kalman v2 NQ_RTY (correct weights)")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-grid", action="store_true", help="Skip grid, load existing CSV")
    args = parser.parse_args()

    instruments = load_instruments()
    spec_a, spec_b = instruments["NQ"], instruments["RTY"]
    flat_min = 15 * 60 + 30

    # ── Phase 1: Grid ──
    csv_path = OUTPUT_DIR / "grid_kalman_v2.csv"

    if args.skip_grid and csv_path.exists():
        log.info(f"[SKIP] Loading existing grid from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        jobs = []
        for alpha, profile, (wlabel, wsh, wsm, weh, wem) in product(
            ALPHA_RATIOS, METRIC_PROFILES.keys(), ENTRY_WINDOWS
        ):
            jobs.append(KalmanJob(
                alpha_ratio=alpha,
                profile_name=profile,
                window_label=wlabel,
                entry_start_min=wsh * 60 + wsm,
                entry_end_min=weh * 60 + wem,
                flat_min=flat_min,
                mult_a=spec_a["multiplier"], mult_b=spec_b["multiplier"],
                tick_a=spec_a["tick_size"], tick_b=spec_b["tick_size"],
            ))

        # Count combos
        valid_combos = sum(
            1 for ze in Z_ENTRIES for zx in Z_EXITS for zs in Z_STOPS
            if zx < ze and zs > ze
        )
        combos_per_job = valid_combos * len(MIN_CONFIDENCES)
        total = len(jobs) * combos_per_job

        log.info("=" * 80)
        log.info(" GRID KALMAN v2 -- NQ_RTY -- CORRECT CONFIDENCE WEIGHTS")
        log.info("=" * 80)
        log.info(f"  Confidence: ADF 50%, Hurst 30%, Corr 20%, HL 0% (Phase 11)")
        log.info(f"  Jobs: {len(jobs)} ({len(ALPHA_RATIOS)} alpha x {len(METRIC_PROFILES)} profiles x {len(ENTRY_WINDOWS)} windows)")
        log.info(f"  Combos/job: {combos_per_job}, Total: {total:,}")
        log.info(f"  Workers: {args.workers}")
        log.info(f"  Alpha: {ALPHA_RATIOS}")
        log.info(f"  z_entry: {Z_ENTRIES[0]} -> {Z_ENTRIES[-1]} ({len(Z_ENTRIES)} values)")
        log.info(f"  z_exit: {Z_EXITS}")
        log.info(f"  z_stop: {Z_STOPS}")
        log.info(f"  Profiles: {list(METRIC_PROFILES.keys())}")
        log.info(f"  Windows: {[w[0] for w in ENTRY_WINDOWS]}")

        if args.dry_run:
            log.info("Dry run -- exiting.")
            return

        t0 = time.time()
        all_results = []
        completed = 0

        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_kalman_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                completed += 1
                try:
                    results = future.result()
                except Exception as e:
                    log.error(f"[{completed}/{len(jobs)}] CRASH: {e}")
                    continue
                if results and "error" in results[0]:
                    log.error(f"[{completed}/{len(jobs)}] ERROR: {results[0]['error']}")
                    continue
                all_results.extend(results)
                if completed % 20 == 0 or completed == len(jobs):
                    elapsed = time.time() - t0
                    eta = (elapsed / completed) * (len(jobs) - completed)
                    log.info(f"[{completed}/{len(jobs)}] {len(all_results):,} results | {elapsed:.0f}s ETA={eta:.0f}s")

        elapsed = time.time() - t0
        log.info(f"Grid complete: {len(all_results):,} results in {elapsed:.0f}s ({elapsed/60:.1f}min)")

        df = pd.DataFrame(all_results)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        log.info(f"[OUTPUT] {csv_path}")

    # ── Filter ──
    log.info(f"\n{'='*80}")
    log.info(f" GRID RESULTS ANALYSIS")
    log.info(f"{'='*80}")
    log.info(f"Total configs: {len(df):,}")

    profitable = df[(df["trades"] >= 30) & (df["pnl"] > 0) & (df["profit_factor"] > 1.0)].copy()
    log.info(f"Profitable (trades>=30, PnL>0, PF>1): {len(profitable):,}")

    if len(profitable) == 0:
        log.warning("No profitable configs found!")
        return

    # Sweet spot analysis
    log.info(f"\n--- Sweet Spots ---")
    for dim in ["alpha_ratio", "profil", "window", "z_entry", "z_exit", "z_stop", "min_confidence"]:
        grp = profitable.groupby(dim).agg(
            count=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
            avg_pf=("profit_factor", "mean"),
            avg_wr=("win_rate", "mean"),
            avg_trades=("trades", "mean"),
        ).round(1)
        log.info(f"\n  {dim}:")
        for val, r in grp.iterrows():
            log.info(f"    {str(val):<12} n={r['count']:>5.0f}  PnL=${r['avg_pnl']:>8,.0f}  PF={r['avg_pf']:.2f}  WR={r['avg_wr']:.1f}%  Trd={r['avg_trades']:.0f}")

    # ── Select diverse top candidates for full analysis ──
    # Score: 40% PF + 25% PnL + 20% WR + 15% trades
    p = profitable.copy()
    p["score"] = (
        0.40 * (p["profit_factor"].clip(upper=5) / 5) +
        0.25 * (p["pnl"] / p["pnl"].max()) +
        0.20 * (p["win_rate"] / 100) +
        0.15 * (p["trades"] / p["trades"].max())
    )

    # Top 10 by score
    top_score = p.nlargest(10, "score")
    # Top 5 by PF (trades >= 50)
    top_pf = p[p["trades"] >= 50].nlargest(5, "profit_factor")
    # Top 5 by PnL
    top_pnl = p.nlargest(5, "pnl")

    candidates = pd.concat([top_score, top_pf, top_pnl]).drop_duplicates(
        subset=["alpha_ratio", "profil", "window", "z_entry", "z_exit", "z_stop", "min_confidence"]
    )

    log.info(f"\n--- Top candidates for full analysis: {len(candidates)} ---")
    for _, r in candidates.iterrows():
        log.info(
            f"  a={r['alpha_ratio']:.1e} {r['profil']:<10} {r['window']:<12} "
            f"ze={r['z_entry']:.3f} zx={r['z_exit']:.3f} zs={r['z_stop']:.3f} c={r['min_confidence']:.0f} | "
            f"Trd={r['trades']:>4} WR={r['win_rate']:.1f}% PF={r['profit_factor']:.2f} PnL=${r['pnl']:>,.0f}"
        )

    # ── Phase 2: Full analysis ──
    results_df = run_full_analysis(candidates, n_top=20)

    if len(results_df) > 0:
        # Filter: IS/OOS GO + L/S symmetry
        go = results_df[results_df["oos_go"] == "GO"].copy()
        symmetric = go[(go["long_pct"] >= 35) & (go["long_pct"] <= 65)].copy()

        log.info(f"\n{'='*80}")
        log.info(f" SUMMARY")
        log.info(f"{'='*80}")
        log.info(f"Total candidates analyzed: {len(results_df)}")
        log.info(f"IS/OOS GO: {len(go)}")
        log.info(f"IS/OOS GO + L/S symmetric: {len(symmetric)}")

        if len(symmetric) > 0:
            log.info(f"\n--- BEST SYMMETRIC + GO ---")
            for _, r in symmetric.sort_values("pf", ascending=False).iterrows():
                log.info(
                    f"  {r['profil']:<10} {r['window']:<12} a={r['alpha']:.1e} "
                    f"ze={r['z_entry']:.3f} zx={r['z_exit']:.3f} zs={r['z_stop']:.3f} c={r['conf']:.0f} | "
                    f"Trd={r['trades']:>3} WR={r['wr']:.1f}% PF={r['pf']:.2f} PnL=${r['pnl']:>,.0f} "
                    f"MaxDD=${r['max_dd']:>,.0f} | L/S={r['long_pct']:.0f}% | "
                    f"IS={r['is_pf']:.2f} OOS={r['oos_pf']:.2f} | NegYr={r['neg_years']}"
                )
                log.info(f"    Yearly: {r['yearly']}")
        elif len(go) > 0:
            log.info(f"\n--- BEST GO (no symmetric found) ---")
            for _, r in go.sort_values("pf", ascending=False).head(5).iterrows():
                log.info(
                    f"  {r['profil']:<10} {r['window']:<12} a={r['alpha']:.1e} "
                    f"ze={r['z_entry']:.3f} zx={r['z_exit']:.3f} zs={r['z_stop']:.3f} c={r['conf']:.0f} | "
                    f"Trd={r['trades']:>3} WR={r['wr']:.1f}% PF={r['pf']:.2f} PnL=${r['pnl']:>,.0f} "
                    f"MaxDD=${r['max_dd']:>,.0f} | L/S={r['long_pct']:.0f}% | "
                    f"IS={r['is_pf']:.2f} OOS={r['oos_pf']:.2f} | NegYr={r['neg_years']}"
                )
                log.info(f"    Yearly: {r['yearly']}")
        else:
            log.info("\nNo configs pass IS/OOS. Kalman NQ/RTY textbox may not be viable.")
            log.info("\n--- Best overall (for reference) ---")
            for _, r in results_df.sort_values("pf", ascending=False).head(5).iterrows():
                log.info(
                    f"  {r['profil']:<10} {r['window']:<12} a={r['alpha']:.1e} "
                    f"ze={r['z_entry']:.3f} zx={r['z_exit']:.3f} zs={r['z_stop']:.3f} c={r['conf']:.0f} | "
                    f"Trd={r['trades']:>3} WR={r['wr']:.1f}% PF={r['pf']:.2f} PnL=${r['pnl']:>,.0f} "
                    f"MaxDD=${r['max_dd']:>,.0f} | L/S={r['long_pct']:.0f}% | "
                    f"IS={r['is_pf']:.2f} OOS={r['oos_pf']:.2f} | NegYr={r['neg_years']}"
                )
                log.info(f"    Yearly: {r['yearly']}")

        # Save
        out_path = OUTPUT_DIR / "grid_kalman_v2_analysis.csv"
        results_df.to_csv(out_path, index=False)
        log.info(f"\n[OUTPUT] {out_path}")

    # Save filtered grid
    filtered_path = OUTPUT_DIR / "grid_kalman_v2_filtered.csv"
    profitable.to_csv(filtered_path, index=False)
    log.info(f"[OUTPUT] {filtered_path} ({len(profitable):,} rows)")


if __name__ == "__main__":
    main()
