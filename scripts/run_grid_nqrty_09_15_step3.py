"""Etape 3 : Grid affine ADF + Corr (seuls et combines) -- NQ_RTY 02:00-15:00.

Sweet spot identifie en Step 2 :
  - OLS : 5280, 6600, 7920
  - ZW : 10, 12, 15
  - ze : 2.50, 2.75, 3.00, 3.25
  - dTP : 2.00, 2.25, 2.50, 3.00
  - ADF seuil : -2.86, -3.00
  - Corr seuil : 0.65, 0.70
  - Gate mode : ADF seul, Corr seul, ADF+Corr
  - TS : 0, 5, 8 bars
  - HL_calc_window : 24, 48

Hurst elimine (non-discriminant pour NQ/RTY).

Usage:
    python scripts/run_grid_nqrty_09_15_step3.py --workers 10
    python scripts/run_grid_nqrty_09_15_step3.py --dry-run
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_grid
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.generator import generate_signals_numba
from src.signals.filters import apply_window_filter_numba, apply_time_stop
from src.spread.pair import SpreadPair
from src.stats.halflife import half_life_rolling
from src.utils.constants import Instrument
from src.config.instruments import get_pair_specs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("grid_nqrty_step3")

OUTPUT_DIR = PROJECT_ROOT / "output" / "NQ_RTY"

# ======================================================================
# Constants
# ======================================================================

_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.20

# Window 02:00-15:00 CT, flat 15:30
ENTRY_START_MIN = 2 * 60
ENTRY_END_MIN = 15 * 60
FLAT_MIN = 15 * 60 + 30

MIN_TRADES = 80  # Lower threshold for refined grid

# z_stop OFF (time stop only)
Z_STOP_DISABLED = 20.0

# ======================================================================
# Grid axes (refined sweet spot)
# ======================================================================

OLS_WINDOWS = [5280, 6600, 7920]
ZSCORE_WINDOWS = [10, 12, 15]
Z_ENTRIES = [2.50, 2.75, 3.00, 3.25]
DELTA_TPS = [2.00, 2.25, 2.50, 3.00]
TIME_STOPS = [0, 5, 8]  # bars (0 = off, 5 = ~25min, 8 = ~40min)
HL_CALC_WINDOWS = [24, 48]

# Gate configurations
ADF_THRESHOLDS = [-2.86, -3.00]
CORR_THRESHOLDS = [0.65, 0.70]

# MetricsConfig -- only 2 (matching HL_CALC_WINDOWS)
METRIC_CONFIGS = {
    24: MetricsConfig(adf_window=24, hurst_window=64, halflife_window=24, correlation_window=12),
    48: MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=24),
}

# Gate modes:
#   "none"     : no gate (baseline)
#   "adf"      : ADF only
#   "corr"     : Corr only
#   "adf+corr" : both must pass (AND)


# ======================================================================
# Gate filter
# ======================================================================

def apply_single_gate(sig: np.ndarray, metric_values: np.ndarray,
                      op: str, threshold: float) -> np.ndarray:
    """Block new entries where gate condition fails. Never blocks exits."""
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0:
            val = metric_values[t]
            if np.isnan(val):
                out[t] = 0
            elif op == "lt" and val >= threshold:
                out[t] = 0
            elif op == "gt" and val <= threshold:
                out[t] = 0
        prev = out[t]
    return out


def apply_double_gate(sig: np.ndarray,
                      adf_values: np.ndarray, adf_threshold: float,
                      corr_values: np.ndarray, corr_threshold: float) -> np.ndarray:
    """Block new entries where BOTH ADF AND Corr must pass. Never blocks exits."""
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0:
            adf_val = adf_values[t]
            corr_val = corr_values[t]
            # ADF must be < threshold (more negative = more stationary)
            adf_ok = not np.isnan(adf_val) and adf_val < adf_threshold
            # Corr must be > threshold
            corr_ok = not np.isnan(corr_val) and corr_val > corr_threshold
            if not (adf_ok and corr_ok):
                out[t] = 0
        prev = out[t]
    return out


# ======================================================================
# Job definition
# ======================================================================

@dataclass
class Step3Job:
    ols_window: int
    zscore_window: int
    hl_calc_window: int


def run_job(job: Step3Job) -> list[dict]:
    """Run one (OLS, ZW, HL_w) combo across all z/gate/TS combos."""
    try:
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": "No cache for NQ_RTY"}]

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        idx = aligned.df.index

        # OLS hedge ratio
        est = create_estimator("ols_rolling", window=job.ols_window,
                               zscore_window=job.zscore_window)
        hr = est.estimate(aligned)
        spread = hr.spread
        beta = hr.beta.values

        mu = spread.rolling(job.zscore_window).mean()
        sigma = spread.rolling(job.zscore_window).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace(
                [np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(
            np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

        # Half-life median
        hl_series = half_life_rolling(spread, window=job.hl_calc_window)
        hl_valid = hl_series.dropna()
        hl_valid = hl_valid[(hl_valid > 0) & (hl_valid < 500)]
        median_hl = float(hl_valid.median()) if len(hl_valid) > 0 else 20.0

        # Metrics
        mc = METRIC_CONFIGS[job.hl_calc_window]
        metrics = compute_all_metrics(spread, aligned.df["close_a"],
                                      aligned.df["close_b"], mc)
        adf_values = metrics["adf_stat"].values
        corr_values = metrics["correlation"].values

        # Minutes
        minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

        results = []

        for z_entry in Z_ENTRIES:
            for delta_tp in DELTA_TPS:
                z_exit = max(z_entry - delta_tp, 0.0)

                raw_signals = generate_signals_numba(
                    zscore, z_entry, z_exit, Z_STOP_DISABLED)

                for ts_bars in TIME_STOPS:
                    # Apply time stop
                    if ts_bars > 0:
                        sig_ts = apply_time_stop(raw_signals.copy(), ts_bars)
                    else:
                        sig_ts = raw_signals.copy()

                    # Apply window filter
                    sig_base = apply_window_filter_numba(
                        sig_ts.copy(), minutes,
                        ENTRY_START_MIN, ENTRY_END_MIN, FLAT_MIN)

                    # Build gate combos list
                    gate_combos = [
                        ("none", 0.0, 0.0),  # baseline
                    ]
                    for adf_t in ADF_THRESHOLDS:
                        gate_combos.append(("adf", adf_t, 0.0))
                    for corr_t in CORR_THRESHOLDS:
                        gate_combos.append(("corr", 0.0, corr_t))
                    for adf_t in ADF_THRESHOLDS:
                        for corr_t in CORR_THRESHOLDS:
                            gate_combos.append(("adf+corr", adf_t, corr_t))

                    for gate_mode, adf_t, corr_t in gate_combos:
                        if gate_mode == "none":
                            sig_gated = sig_base.copy()
                        elif gate_mode == "adf":
                            sig_gated = apply_single_gate(
                                sig_base.copy(), adf_values, "lt", adf_t)
                        elif gate_mode == "corr":
                            sig_gated = apply_single_gate(
                                sig_base.copy(), corr_values, "gt", corr_t)
                        elif gate_mode == "adf+corr":
                            sig_gated = apply_double_gate(
                                sig_base.copy(),
                                adf_values, adf_t,
                                corr_values, corr_t)

                        bt = run_backtest_grid(
                            px_a, px_b, sig_gated, beta,
                            mult_a=MULT_A, mult_b=MULT_B,
                            tick_a=TICK_A, tick_b=TICK_B,
                            slippage_ticks=SLIPPAGE, commission=COMMISSION)

                        results.append({
                            "ols_window": job.ols_window,
                            "zscore_window": job.zscore_window,
                            "hl_calc_window": job.hl_calc_window,
                            "median_hl": round(median_hl, 1),
                            "z_entry": z_entry,
                            "z_exit": round(z_exit, 2),
                            "delta_tp": delta_tp,
                            "time_stop_bars": ts_bars,
                            "gate": gate_mode,
                            "adf_threshold": adf_t,
                            "corr_threshold": corr_t,
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
# Analysis
# ======================================================================

def analyze_results(df: pd.DataFrame):
    """Full analysis of Step 3 results."""
    viable = df[df["trades"] >= MIN_TRADES].copy()
    if viable.empty:
        log.info("No viable configs found.")
        return

    log.info(f"\n{'='*80}")
    log.info(f"STEP 3 ANALYSIS -- {len(viable):,} viable configs (trades >= {MIN_TRADES})")
    log.info(f"{'='*80}")

    # --- Gate mode comparison ---
    log.info(f"\n--- GATE MODE COMPARISON ---")
    log.info(f"  {'Gate':>12}  {'Configs':>8}  {'Med PF':>8}  {'%Profit':>8}  "
             f"{'Med Trades':>10}  {'Med PnL':>10}  {'Med $/t':>10}")
    log.info("  " + "-" * 80)

    for gate in ["none", "adf", "corr", "adf+corr"]:
        sub = viable[viable["gate"] == gate]
        if sub.empty:
            continue
        prof = sub[sub["profit_factor"] > 1.0]
        log.info(f"  {gate:>12}  {len(sub):>8,}  {sub['profit_factor'].median():>8.3f}  "
                 f"{len(prof)/len(sub)*100:>7.1f}%  "
                 f"{sub['trades'].median():>10.0f}  "
                 f"${sub['pnl'].median():>9,.0f}  "
                 f"${sub['avg_pnl_trade'].median():>9,.0f}")

    # --- ADF+Corr threshold combos ---
    log.info(f"\n--- ADF+CORR THRESHOLD COMBOS ---")
    combo = viable[viable["gate"] == "adf+corr"]
    if not combo.empty:
        log.info(f"  {'ADF<':>8}  {'Corr>':>8}  {'Configs':>8}  {'Med PF':>8}  "
                 f"{'%Profit':>8}  {'Med Trades':>10}  {'Med PnL':>10}")
        log.info("  " + "-" * 70)
        for adf_t in ADF_THRESHOLDS:
            for corr_t in CORR_THRESHOLDS:
                sub = combo[(combo["adf_threshold"] == adf_t) &
                            (combo["corr_threshold"] == corr_t)]
                if sub.empty:
                    continue
                prof = sub[sub["profit_factor"] > 1.0]
                log.info(f"  {adf_t:>8.2f}  {corr_t:>8.2f}  {len(sub):>8}  "
                         f"{sub['profit_factor'].median():>8.3f}  "
                         f"{len(prof)/len(sub)*100:>7.1f}%  "
                         f"{sub['trades'].median():>10.0f}  "
                         f"${sub['pnl'].median():>9,.0f}")

    # --- ADF seul threshold ---
    log.info(f"\n--- ADF SEUL THRESHOLD ---")
    adf_only = viable[viable["gate"] == "adf"]
    if not adf_only.empty:
        for adf_t in ADF_THRESHOLDS:
            sub = adf_only[adf_only["adf_threshold"] == adf_t]
            if sub.empty:
                continue
            prof = sub[sub["profit_factor"] > 1.0]
            log.info(f"  ADF < {adf_t:.2f} : {len(sub):>6} configs, "
                     f"med PF={sub['profit_factor'].median():.3f}, "
                     f"profitable={len(prof)/len(sub)*100:.1f}%, "
                     f"med trades={sub['trades'].median():.0f}")

    # --- Time stop ---
    log.info(f"\n--- TIME STOP IMPACT ---")
    for ts in TIME_STOPS:
        sub = viable[viable["time_stop_bars"] == ts]
        prof = sub[sub["profit_factor"] > 1.0]
        log.info(f"  TS={ts:>2} bars : {len(sub):>6,} configs, "
                 f"med PF={sub['profit_factor'].median():.3f}, "
                 f"profitable={len(prof)/len(sub)*100:.1f}%")

    # --- TOP 30 ---
    log.info(f"\n{'='*80}")
    log.info(f"TOP 30 CONFIGS (trades >= {MIN_TRADES})")
    log.info(f"{'='*80}")
    top = viable.nlargest(30, "profit_factor")
    log.info(f"  {'OLS':>5} {'ZW':>3} {'ze':>5} {'dTP':>5} {'TS':>3} {'HLw':>3} "
             f"{'Gate':>10} {'ADF<':>6} {'C>':>5} | "
             f"{'Trd':>4} {'WR%':>6} {'PnL':>9} {'PF':>5} {'$/t':>6} {'Dur':>4}")
    log.info("  " + "-" * 95)
    for _, r in top.iterrows():
        log.info(
            f"  {int(r['ols_window']):>5} {int(r['zscore_window']):>3} "
            f"{r['z_entry']:>5.2f} {r['delta_tp']:>5.2f} "
            f"{int(r['time_stop_bars']):>3} {int(r['hl_calc_window']):>3} "
            f"{r['gate']:>10} {r['adf_threshold']:>6.2f} {r['corr_threshold']:>5.2f} | "
            f"{int(r['trades']):>4} {r['win_rate']:>5.1f}% "
            f"${r['pnl']:>8,.0f} {r['profit_factor']:>5.2f} "
            f"${r['avg_pnl_trade']:>5,.0f} "
            f"{r['avg_duration_bars']:>4.1f}"
        )

    # --- TOP 20 with trades >= 120 ---
    top120 = viable[viable["trades"] >= 120].nlargest(20, "profit_factor")
    if not top120.empty:
        log.info(f"\n{'='*80}")
        log.info(f"TOP 20 CONFIGS (trades >= 120)")
        log.info(f"{'='*80}")
        log.info(f"  {'OLS':>5} {'ZW':>3} {'ze':>5} {'dTP':>5} {'TS':>3} {'HLw':>3} "
                 f"{'Gate':>10} {'ADF<':>6} {'C>':>5} | "
                 f"{'Trd':>4} {'WR%':>6} {'PnL':>9} {'PF':>5} {'$/t':>6} {'Dur':>4}")
        log.info("  " + "-" * 95)
        for _, r in top120.iterrows():
            log.info(
                f"  {int(r['ols_window']):>5} {int(r['zscore_window']):>3} "
                f"{r['z_entry']:>5.2f} {r['delta_tp']:>5.2f} "
                f"{int(r['time_stop_bars']):>3} {int(r['hl_calc_window']):>3} "
                f"{r['gate']:>10} {r['adf_threshold']:>6.2f} {r['corr_threshold']:>5.2f} | "
                f"{int(r['trades']):>4} {r['win_rate']:>5.1f}% "
                f"${r['pnl']:>8,.0f} {r['profit_factor']:>5.2f} "
                f"${r['avg_pnl_trade']:>5,.0f} "
                f"{r['avg_duration_bars']:>4.1f}"
            )


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Grid NQ_RTY Step 3: ADF + Corr refined")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Count
    n_z = len(Z_ENTRIES) * len(DELTA_TPS)
    n_ts = len(TIME_STOPS)
    n_gates = 1 + len(ADF_THRESHOLDS) + len(CORR_THRESHOLDS) + len(ADF_THRESHOLDS) * len(CORR_THRESHOLDS)
    # 1 none + 2 adf + 2 corr + 4 adf+corr = 9
    n_jobs = len(OLS_WINDOWS) * len(ZSCORE_WINDOWS) * len(HL_CALC_WINDOWS)
    combos_per_job = n_z * n_ts * n_gates
    total = n_jobs * combos_per_job

    log.info("Grid NQ_RTY Step 3 -- ADF + Corr Refined")
    log.info(f"  OLS windows:    {OLS_WINDOWS}")
    log.info(f"  ZW:             {ZSCORE_WINDOWS}")
    log.info(f"  z_entry:        {Z_ENTRIES}")
    log.info(f"  delta_tp:       {DELTA_TPS}")
    log.info(f"  Time stops:     {TIME_STOPS} bars")
    log.info(f"  HL calc windows:{HL_CALC_WINDOWS}")
    log.info(f"  ADF thresholds: {ADF_THRESHOLDS}")
    log.info(f"  Corr thresholds:{CORR_THRESHOLDS}")
    log.info(f"  Gate modes:     none + {len(ADF_THRESHOLDS)} adf + {len(CORR_THRESHOLDS)} corr + {len(ADF_THRESHOLDS)*len(CORR_THRESHOLDS)} adf+corr = {n_gates}")
    log.info(f"  Jobs:           {n_jobs} ({len(OLS_WINDOWS)}x{len(ZSCORE_WINDOWS)}x{len(HL_CALC_WINDOWS)})")
    log.info(f"  Combos/job:     {combos_per_job}")
    log.info(f"  TOTAL BACKTESTS:{total:,}")
    log.info(f"  Min trades:     {MIN_TRADES}")

    if args.dry_run:
        log.info("DRY RUN -- stopping here.")
        return

    # Build jobs
    jobs = []
    for ols_w in OLS_WINDOWS:
        for zw in ZSCORE_WINDOWS:
            for hl_w in HL_CALC_WINDOWS:
                jobs.append(Step3Job(
                    ols_window=ols_w,
                    zscore_window=zw,
                    hl_calc_window=hl_w,
                ))

    log.info(f"\nLaunching {len(jobs)} jobs with {args.workers} workers...")
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

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - completed) / rate if rate > 0 else 0
            log.info(
                f"  [{completed}/{len(jobs)}] "
                f"{len(all_results):,} results, {errors} errors, "
                f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log.info(f"\nGrid complete: {len(all_results):,} results in {elapsed:.0f}s, {errors} errors")

    # Save all
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_all = OUTPUT_DIR / "grid_nqrty_step3.csv"
    df.to_csv(csv_all, index=False)
    log.info(f"Saved {len(df):,} rows to {csv_all}")

    # Analysis
    analyze_results(df)

    # Save viable
    viable = df[df["trades"] >= MIN_TRADES].sort_values("profit_factor", ascending=False)
    csv_viable = OUTPUT_DIR / "grid_nqrty_step3_viable.csv"
    viable.to_csv(csv_viable, index=False)
    log.info(f"\nSaved {len(viable):,} viable rows to {csv_viable}")


if __name__ == "__main__":
    main()
