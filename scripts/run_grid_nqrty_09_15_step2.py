"""Etape 2 : Contribution marginale metriques + time stop half-life -- NQ_RTY 02:00-15:00.

Pour chaque (OLS, ZW) :
  1. Compute half-life median du spread
  2. z_stop = OFF (tres large), remplace par time_stop = k * median_half_life
  3. Filtre mean-revertant : dTP >= 2.0 (z_exit <= 1.0)
  4. Test gates binaires individuelles (ADF seul, Hurst seul, Corr seul)
  5. Mesure delta PF/PnL vs baseline (sans gate)

Usage:
    python scripts/run_grid_nqrty_09_15_step2.py --workers 10
    python scripts/run_grid_nqrty_09_15_step2.py --dry-run
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
from src.signals.filters import apply_time_stop, apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.stats.halflife import half_life_rolling
from src.utils.constants import Instrument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("grid_nqrty_step2")

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

MIN_TRADES = 100  # Plus bas car gates reduisent le nombre de trades

# z_stop tres large (effectivement OFF, on utilise time_stop a la place)
Z_STOP_DISABLED = 20.0

# ======================================================================
# Grid axes
# ======================================================================

# OLS et ZW : concentres sur les sweet spots de Step 1
OLS_WINDOWS = [528, 1320, 2640, 3960, 5280, 6600, 7920, 9240, 10560]
ZSCORE_WINDOWS = [10, 12, 15, 20, 24, 25, 30]

# z_entry : focus mean-revertant
Z_ENTRIES = [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]

# dTP >= 2.0 uniquement (profil mean-revertant, z_exit <= 1.0)
DELTA_TPS = [2.00, 2.25, 2.50, 3.00]

# Time stop multipliers (k * median_half_life)
HL_MULTIPLIERS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]  # 0.0 = no time stop (baseline)

# Half-life window pour calcul median
HL_CALC_WINDOWS = [24, 48, 96]

# Binary gates individuelles (seuils a tester)
# Format : (gate_name, metric_column, operator, thresholds)
#   operator: "lt" = metric < threshold to pass, "gt" = metric > threshold to pass
GATES = {
    "adf": {
        "column": "adf_stat",
        "op": "lt",  # ADF stat must be < threshold (more negative = more stationary)
        "thresholds": [-2.00, -2.50, -2.86, -3.00, -3.50],
    },
    "hurst": {
        "column": "hurst",
        "op": "lt",  # Hurst must be < threshold (< 0.5 = mean-reverting)
        "thresholds": [0.55, 0.50, 0.48, 0.45, 0.42],
    },
    "corr": {
        "column": "correlation",
        "op": "gt",  # Correlation must be > threshold
        "thresholds": [0.60, 0.65, 0.70, 0.75, 0.80],
    },
}

# MetricsConfig windows to test
METRIC_CONFIGS = {
    "mc_24_64": MetricsConfig(adf_window=24, hurst_window=64, halflife_window=24, correlation_window=12),
    "mc_48_128": MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=24),
    "mc_96_256": MetricsConfig(adf_window=96, hurst_window=256, halflife_window=96, correlation_window=24),
}


# ======================================================================
# Gate filter (single metric, binary)
# ======================================================================

def apply_single_gate(sig: np.ndarray, metric_values: np.ndarray,
                      op: str, threshold: float) -> np.ndarray:
    """Block new entries where gate condition fails. Never blocks exits."""
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0:
            # New entry -- check gate
            val = metric_values[t]
            if np.isnan(val):
                out[t] = 0
            elif op == "lt" and val >= threshold:
                out[t] = 0
            elif op == "gt" and val <= threshold:
                out[t] = 0
        prev = out[t]
    return out


# ======================================================================
# Job definition
# ======================================================================

@dataclass
class Step2Job:
    ols_window: int
    zscore_window: int
    metric_config_name: str
    hl_calc_window: int


def run_job(job: Step2Job) -> list[dict]:
    """Run one (OLS, ZW, metric_config, hl_window) combo."""
    try:
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": "No cache for NQ_RTY"}]

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        idx = aligned.df.index

        # OLS hedge ratio + spread + zscore
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

        # Half-life median (for time stop calibration)
        hl_series = half_life_rolling(spread, window=job.hl_calc_window)
        hl_valid = hl_series.dropna()
        hl_valid = hl_valid[(hl_valid > 0) & (hl_valid < 500)]
        median_hl = float(hl_valid.median()) if len(hl_valid) > 0 else 20.0

        # Metrics for gates
        mc = METRIC_CONFIGS[job.metric_config_name]
        metrics = compute_all_metrics(spread, aligned.df["close_a"],
                                      aligned.df["close_b"], mc)

        # Minutes array
        minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

        results = []

        for z_entry in Z_ENTRIES:
            for delta_tp in DELTA_TPS:
                z_exit = max(z_entry - delta_tp, 0.0)

                # Generate raw signals (z_stop effectively OFF)
                raw_signals = generate_signals_numba(
                    zscore, z_entry, z_exit, Z_STOP_DISABLED)

                for hl_mult in HL_MULTIPLIERS:
                    time_stop_bars = int(round(median_hl * hl_mult)) if hl_mult > 0 else 0

                    # Apply time stop
                    if time_stop_bars > 0:
                        sig_ts = apply_time_stop(raw_signals.copy(), time_stop_bars)
                    else:
                        sig_ts = raw_signals.copy()

                    # Apply window filter
                    sig_base = apply_window_filter_numba(
                        sig_ts.copy(), minutes,
                        ENTRY_START_MIN, ENTRY_END_MIN, FLAT_MIN)

                    # --- Baseline (no gate) ---
                    bt = run_backtest_grid(
                        px_a, px_b, sig_base, beta,
                        mult_a=MULT_A, mult_b=MULT_B,
                        tick_a=TICK_A, tick_b=TICK_B,
                        slippage_ticks=SLIPPAGE, commission=COMMISSION)

                    results.append({
                        "ols_window": job.ols_window,
                        "zscore_window": job.zscore_window,
                        "metric_config": job.metric_config_name,
                        "hl_calc_window": job.hl_calc_window,
                        "median_hl": round(median_hl, 1),
                        "z_entry": z_entry,
                        "z_exit": round(z_exit, 2),
                        "delta_tp": delta_tp,
                        "hl_mult": hl_mult,
                        "time_stop_bars": time_stop_bars,
                        "gate": "none",
                        "gate_threshold": 0.0,
                        "trades": bt["trades"],
                        "win_rate": bt["win_rate"],
                        "pnl": bt["pnl"],
                        "profit_factor": bt["profit_factor"],
                        "avg_pnl_trade": bt["avg_pnl_trade"],
                        "avg_duration_bars": bt["avg_duration_bars"],
                    })

                    # --- Individual gates ---
                    for gate_name, gate_cfg in GATES.items():
                        col = gate_cfg["column"]
                        op = gate_cfg["op"]
                        metric_vals = metrics[col].values

                        for threshold in gate_cfg["thresholds"]:
                            sig_gated = apply_single_gate(
                                sig_base.copy(), metric_vals, op, threshold)

                            bt_g = run_backtest_grid(
                                px_a, px_b, sig_gated, beta,
                                mult_a=MULT_A, mult_b=MULT_B,
                                tick_a=TICK_A, tick_b=TICK_B,
                                slippage_ticks=SLIPPAGE, commission=COMMISSION)

                            results.append({
                                "ols_window": job.ols_window,
                                "zscore_window": job.zscore_window,
                                "metric_config": job.metric_config_name,
                                "hl_calc_window": job.hl_calc_window,
                                "median_hl": round(median_hl, 1),
                                "z_entry": z_entry,
                                "z_exit": round(z_exit, 2),
                                "delta_tp": delta_tp,
                                "hl_mult": hl_mult,
                                "time_stop_bars": time_stop_bars,
                                "gate": gate_name,
                                "gate_threshold": threshold,
                                "trades": bt_g["trades"],
                                "win_rate": bt_g["win_rate"],
                                "pnl": bt_g["pnl"],
                                "profit_factor": bt_g["profit_factor"],
                                "avg_pnl_trade": bt_g["avg_pnl_trade"],
                                "avg_duration_bars": bt_g["avg_duration_bars"],
                            })

        return results

    except Exception as e:
        return [{"error": str(e), "ols_window": job.ols_window,
                 "zscore_window": job.zscore_window}]


# ======================================================================
# Analysis
# ======================================================================

def analyze_gates(df: pd.DataFrame):
    """Analyze marginal contribution of each gate vs baseline."""
    viable = df[df["trades"] >= MIN_TRADES].copy()
    if viable.empty:
        log.info("No viable configs found.")
        return

    baseline = viable[viable["gate"] == "none"]
    log.info(f"\nBaseline (no gate): {len(baseline)} configs, "
             f"median PF {baseline['profit_factor'].median():.3f}")

    # --- Gate contribution ---
    log.info("\n" + "=" * 70)
    log.info("MARGINAL CONTRIBUTION PAR GATE (vs baseline, median PF)")
    log.info("=" * 70)

    for gate_name in GATES:
        gated = viable[viable["gate"] == gate_name]
        if gated.empty:
            continue

        log.info(f"\n  Gate: {gate_name.upper()}")
        log.info(f"    {'Threshold':>10}  {'Configs':>8}  {'Med PF':>8}  "
                 f"{'Med Trades':>10}  {'Delta PF':>10}  {'Med WR':>6}")
        log.info(f"    {'baseline':>10}  {len(baseline):>8}  "
                 f"{baseline['profit_factor'].median():>8.3f}  "
                 f"{baseline['trades'].median():>10.0f}  {'---':>10}  "
                 f"{baseline['win_rate'].median():>5.1f}%")

        for threshold in GATES[gate_name]["thresholds"]:
            subset = gated[gated["gate_threshold"] == threshold]
            if subset.empty:
                continue
            med_pf = subset["profit_factor"].median()
            delta_pf = med_pf - baseline["profit_factor"].median()
            sign = "+" if delta_pf >= 0 else ""
            log.info(f"    {threshold:>10.2f}  {len(subset):>8}  "
                     f"{med_pf:>8.3f}  {subset['trades'].median():>10.0f}  "
                     f"{sign}{delta_pf:>9.3f}  {subset['win_rate'].median():>5.1f}%")


def analyze_time_stop(df: pd.DataFrame):
    """Analyze time stop impact."""
    viable = df[(df["trades"] >= MIN_TRADES) & (df["gate"] == "none")].copy()
    if viable.empty:
        return

    log.info("\n" + "=" * 70)
    log.info("TIME STOP IMPACT (baseline, no gates)")
    log.info("=" * 70)

    log.info(f"\n  {'HL mult':>8}  {'TS bars':>8}  {'Configs':>8}  "
             f"{'Med PF':>8}  {'Med Trades':>10}  {'Med Dur':>8}  {'Med WR':>6}")
    log.info("  " + "-" * 70)

    for hl_mult in sorted(viable["hl_mult"].unique()):
        subset = viable[viable["hl_mult"] == hl_mult]
        ts_bars = subset["time_stop_bars"].median()
        log.info(f"  {hl_mult:>8.1f}  {ts_bars:>8.0f}  {len(subset):>8}  "
                 f"{subset['profit_factor'].median():>8.3f}  "
                 f"{subset['trades'].median():>10.0f}  "
                 f"{subset['avg_duration_bars'].median():>8.1f}  "
                 f"{subset['win_rate'].median():>5.1f}%")


def analyze_hl_calc_window(df: pd.DataFrame):
    """Analyze impact of half-life calculation window."""
    viable = df[(df["trades"] >= MIN_TRADES) & (df["gate"] == "none")].copy()
    if viable.empty:
        return

    log.info("\n" + "=" * 70)
    log.info("HALF-LIFE CALC WINDOW IMPACT")
    log.info("=" * 70)

    for hl_w in sorted(viable["hl_calc_window"].unique()):
        subset = viable[viable["hl_calc_window"] == hl_w]
        log.info(f"\n  HL calc window = {hl_w}:")
        log.info(f"    Median half-life: {subset['median_hl'].median():.1f} bars "
                 f"({subset['median_hl'].median() * 5:.0f} min)")

        for hl_mult in sorted(subset["hl_mult"].unique()):
            sub2 = subset[subset["hl_mult"] == hl_mult]
            if sub2.empty:
                continue
            ts = sub2["time_stop_bars"].median()
            log.info(f"    k={hl_mult:.1f} -> TS={ts:.0f} bars ({ts*5:.0f}min) | "
                     f"PF {sub2['profit_factor'].median():.3f}, "
                     f"trades {sub2['trades'].median():.0f}")


def analyze_best_combos(df: pd.DataFrame):
    """Show best gate + time stop combinations."""
    viable = df[df["trades"] >= MIN_TRADES].copy()
    if viable.empty:
        return

    log.info("\n" + "=" * 70)
    log.info(f"TOP 30 CONFIGS (all gates + time stops, trades >= {MIN_TRADES})")
    log.info("=" * 70)

    top = viable.nlargest(30, "profit_factor")
    log.info(f"  {'OLS':>6} {'ZW':>4} {'ze':>5} {'zx':>5} {'dTP':>5} "
             f"{'HLm':>4} {'TS':>4} {'Gate':>6} {'Thr':>6} | "
             f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7} {'Dur':>5}")
    log.info("  " + "-" * 100)
    for _, r in top.iterrows():
        log.info(
            f"  {r['ols_window']:>6} {r['zscore_window']:>4} "
            f"{r['z_entry']:>5.2f} {r['z_exit']:>5.2f} {r['delta_tp']:>5.2f} "
            f"{r['hl_mult']:>4.1f} {r['time_stop_bars']:>4.0f} "
            f"{r['gate']:>6} {r['gate_threshold']:>6.2f} | "
            f"{r['trades']:>5.0f} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} "
            f"{r['profit_factor']:>6.2f} ${r['avg_pnl_trade']:>6,.0f} "
            f"{r['avg_duration_bars']:>5.1f}"
        )


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Grid NQ_RTY Step 2: metrics + HL time stop")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Count combos per job
    n_z_combos = len(Z_ENTRIES) * len(DELTA_TPS)
    n_hl = len(HL_MULTIPLIERS)
    n_gates = 1 + sum(len(g["thresholds"]) for g in GATES.values())  # 1 baseline + gates
    combos_per_job = n_z_combos * n_hl * n_gates

    n_jobs = len(OLS_WINDOWS) * len(ZSCORE_WINDOWS) * len(METRIC_CONFIGS) * len(HL_CALC_WINDOWS)
    total = n_jobs * n_z_combos * n_hl * n_gates

    log.info("Grid NQ_RTY Step 2 -- Metriques + Half-Life Time Stop")
    log.info(f"  OLS windows:     {len(OLS_WINDOWS)}")
    log.info(f"  ZW:              {len(ZSCORE_WINDOWS)}")
    log.info(f"  z_entry:         {len(Z_ENTRIES)} ({Z_ENTRIES[0]} to {Z_ENTRIES[-1]})")
    log.info(f"  delta_tp:        {len(DELTA_TPS)} (mean-revertant only: {DELTA_TPS})")
    log.info(f"  HL multipliers:  {len(HL_MULTIPLIERS)} ({HL_MULTIPLIERS})")
    log.info(f"  HL calc windows: {len(HL_CALC_WINDOWS)} ({HL_CALC_WINDOWS})")
    log.info(f"  Metric configs:  {len(METRIC_CONFIGS)} ({list(METRIC_CONFIGS.keys())})")
    log.info(f"  Gates:           {n_gates} (1 baseline + {n_gates-1} gate combos)")
    log.info(f"  Combos/job:      {combos_per_job}")
    log.info(f"  Jobs:            {n_jobs}")
    log.info(f"  TOTAL BACKTESTS: {total:,}")
    log.info(f"  Min trades:      {MIN_TRADES}")

    if args.dry_run:
        log.info("DRY RUN -- stopping here.")
        return

    # Build jobs
    jobs = []
    for ols_w in OLS_WINDOWS:
        for zw in ZSCORE_WINDOWS:
            for mc_name in METRIC_CONFIGS:
                for hl_w in HL_CALC_WINDOWS:
                    jobs.append(Step2Job(
                        ols_window=ols_w,
                        zscore_window=zw,
                        metric_config_name=mc_name,
                        hl_calc_window=hl_w,
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

            if completed % 20 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - completed) / rate if rate > 0 else 0
                log.info(
                    f"  {completed}/{len(jobs)} jobs "
                    f"({len(all_results):,} results, {errors} errors, "
                    f"{elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    log.info(f"Grid complete: {len(all_results):,} results in {elapsed:.0f}s, {errors} errors")

    # Save all
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_all = OUTPUT_DIR / "grid_nqrty_step2.csv"
    df.to_csv(csv_all, index=False)
    log.info(f"Saved {len(df):,} rows to {csv_all}")

    # Analysis
    analyze_time_stop(df)
    analyze_hl_calc_window(df)
    analyze_gates(df)
    analyze_best_combos(df)

    # Save viable
    viable = df[df["trades"] >= MIN_TRADES].sort_values("profit_factor", ascending=False)
    csv_viable = OUTPUT_DIR / "grid_nqrty_step2_viable.csv"
    viable.to_csv(csv_viable, index=False)
    log.info(f"\nSaved {len(viable):,} viable rows to {csv_viable}")


if __name__ == "__main__":
    main()
