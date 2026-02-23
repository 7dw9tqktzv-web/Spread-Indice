"""Raffinement OLS_Balanced NQ/RTY — ablation metriques + grid affine.

Etape 1: Ablation des 4 metriques (ADF, Hurst, Corr, HL)
Etape 2: Grid affine autour de la config balanced pour augmenter les trades

Usage:
    python scripts/refine_ols_balanced.py --workers 10
    python scripts/refine_ols_balanced.py --dry-run
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

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
log = logging.getLogger("refine_balanced")

MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

OUTPUT_DIR = PROJECT_ROOT / "output"

# Base config (OLS_Balanced)
BASE = {
    "ols_window": 7920,
    "zscore_window": 48,
    "z_entry": 3.50,
    "z_exit": 0.50,
    "z_stop": 6.00,
    "min_confidence": 75.0,
    "entry_window": "08:00-14:00",
}

METRIC_PROFILES = {
    # --- Profils alignes (fenetres proportionnelles) ---
    "p12_64":    MetricsConfig(adf_window=12, hurst_window=64,  halflife_window=12, correlation_window=6),
    "p16_80":    MetricsConfig(adf_window=16, hurst_window=80,  halflife_window=16, correlation_window=8),
    "p18_96":    MetricsConfig(adf_window=18, hurst_window=96,  halflife_window=18, correlation_window=9),
    "p20_100":   MetricsConfig(adf_window=20, hurst_window=100, halflife_window=20, correlation_window=10),
    "p24_128":   MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "p28_144":   MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p30_160":   MetricsConfig(adf_window=30, hurst_window=160, halflife_window=30, correlation_window=15),
    "p36_192":   MetricsConfig(adf_window=36, hurst_window=192, halflife_window=36, correlation_window=18),
    "p42_224":   MetricsConfig(adf_window=42, hurst_window=224, halflife_window=42, correlation_window=21),
    "p48_256":   MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "p60_320":   MetricsConfig(adf_window=60, hurst_window=320, halflife_window=60, correlation_window=30),
    # --- Mixes : ADF rapide + Hurst lent ---
    "p18_192":   MetricsConfig(adf_window=18, hurst_window=192, halflife_window=18, correlation_window=18),
    "p24_256":   MetricsConfig(adf_window=24, hurst_window=256, halflife_window=24, correlation_window=24),
    "p30_256":   MetricsConfig(adf_window=30, hurst_window=256, halflife_window=30, correlation_window=24),
    # --- Mixes : ADF lent + Hurst rapide ---
    "p48_128":   MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
    "p48_96":    MetricsConfig(adf_window=48, hurst_window=96,  halflife_window=48, correlation_window=9),
    "p36_96":    MetricsConfig(adf_window=36, hurst_window=96,  halflife_window=36, correlation_window=9),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
    "08:00-14:00": (480, 840),
    "08:00-12:00": (480, 720),
    "06:00-12:00": (360, 720),
}

# Confidence weight presets
# HL retire du scoring (garde en textbox Sierra seulement)
# Ablation: no_hl = +155% trades, PnL quasi identique
CONF_PRESETS = {
    "no_hl":      ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00),
    "standard":   ConfidenceConfig(w_adf=0.40, w_hurst=0.25, w_corr=0.20, w_hl=0.15),
    "no_hurst":   ConfidenceConfig(w_adf=0.50, w_hurst=0.00, w_corr=0.30, w_hl=0.20),
    "no_corr":    ConfidenceConfig(w_adf=0.50, w_hurst=0.35, w_corr=0.00, w_hl=0.15),
    "no_adf_wt":  ConfidenceConfig(w_adf=0.00, w_hurst=0.40, w_corr=0.35, w_hl=0.25),
    "adf_only":   ConfidenceConfig(w_adf=1.00, w_hurst=0.00, w_corr=0.00, w_hl=0.00),
    "hurst_heavy": ConfidenceConfig(w_adf=0.25, w_hurst=0.45, w_corr=0.15, w_hl=0.15),
    "corr_heavy": ConfidenceConfig(w_adf=0.25, w_hurst=0.15, w_corr=0.45, w_hl=0.15),
}

# ======================================================================
# GRID PARAMETERS — raffine, HL retire, focus plus de trades
# ======================================================================

OLS_WINDOWS = [3960, 5280, 6600, 7920, 9240, 10560]
ZSCORE_WINDOWS = [12, 20, 24, 28, 32, 36, 48, 60]
Z_ENTRIES = [2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
Z_EXITS = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50]
Z_STOPS = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
MIN_CONFIDENCES = [0.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0]
PROFILES = list(METRIC_PROFILES.keys())  # 8 profils affines
ENTRY_WINDOWS = [
    "02:00-14:00", "04:00-14:00", "06:00-14:00",
    "08:00-14:00", "08:00-12:00", "06:00-12:00",
]
CONF_WEIGHT_NAMES = ["no_hl"]  # HL retire du scoring


@dataclass
class RefinedJob:
    ols_window: int
    zscore_window: int
    profile_name: str
    window_label: str
    entry_start_min: int
    entry_end_min: int
    conf_weight_name: str


def run_refined_job(job: RefinedJob) -> list[dict]:
    """Run one job = one (OLS, ZW, profile, window, conf_weights) combo x all signal params."""
    try:
        pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
        aligned = load_aligned_pair_cache(pair, "5min")
        if aligned is None:
            return [{"error": "No cache"}]

        px_a = aligned.df["close_a"].values
        px_b = aligned.df["close_b"].values
        idx = aligned.df.index

        # OLS hedge
        est = create_estimator("ols_rolling", window=job.ols_window, zscore_window=job.zscore_window)
        hr = est.estimate(aligned)
        spread = hr.spread
        beta = hr.beta.values

        mu = spread.rolling(job.zscore_window).mean()
        sigma = spread.rolling(job.zscore_window).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

        # Metrics + confidence with specific weights
        profile_cfg = METRIC_PROFILES[job.profile_name]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        conf_preset = CONF_PRESETS[job.conf_weight_name]
        confidence = compute_confidence(metrics, conf_preset).values

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
                            "conf_weights": job.conf_weight_name,
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


def run_ablation():
    """Quick ablation: test base config with each confidence preset."""
    print("=" * 100)
    print("  ETAPE 1 : ABLATION DES METRIQUES")
    print("=" * 100)

    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("  ERREUR: pas de cache")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    est = create_estimator("ols_rolling", window=BASE["ols_window"],
                           zscore_window=BASE["zscore_window"])
    hr = est.estimate(aligned)
    spread = hr.spread
    beta = hr.beta.values

    mu = spread.rolling(BASE["zscore_window"]).mean()
    sigma = spread.rolling(BASE["zscore_window"]).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

    profile_cfg = METRIC_PROFILES["p24_128"]  # = ancien "court"
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)

    raw = generate_signals_numba(zscore, BASE["z_entry"], BASE["z_exit"], BASE["z_stop"])
    entry_start, entry_end = WINDOWS_MAP[BASE["entry_window"]]

    print(f"\n  Config de base: OLS={BASE['ols_window']}, ZW={BASE['zscore_window']}, "
          f"ze={BASE['z_entry']}, zx={BASE['z_exit']}, zs={BASE['z_stop']}, "
          f"conf={BASE['min_confidence']}, window={BASE['entry_window']}")

    print(f"\n  {'Preset':<14} {'Weights (A/H/C/HL)':>22} | {'Trd':>5} {'WR%':>6} "
          f"{'PnL':>10} {'PF':>6} {'Avg$':>7}")
    print("  " + "-" * 80)

    # Also test conf=0 (no filter at all)
    sig_raw = apply_window_filter_numba(
        raw.copy(), minutes, entry_start, entry_end, FLAT_MIN,
    )
    bt_raw = run_backtest_grid(
        px_a, px_b, sig_raw, beta,
        mult_a=MULT_A, mult_b=MULT_B,
        tick_a=TICK_A, tick_b=TICK_B,
        slippage_ticks=SLIPPAGE, commission=COMMISSION,
    )
    print(f"  {'NO_FILTER':<14} {'---':>22} | {bt_raw['trades']:>5} {bt_raw['win_rate']:>5.1f}% "
          f"${bt_raw['pnl']:>9,.0f} {bt_raw['profit_factor']:>6.2f} "
          f"${bt_raw['avg_pnl_trade']:>6,.0f}")

    for preset_name, conf_cfg in CONF_PRESETS.items():
        confidence = compute_confidence(metrics, conf_cfg).values
        sig = _apply_conf_filter_numba(raw.copy(), confidence, BASE["min_confidence"])
        sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

        bt = run_backtest_grid(
            px_a, px_b, sig, beta,
            mult_a=MULT_A, mult_b=MULT_B,
            tick_a=TICK_A, tick_b=TICK_B,
            slippage_ticks=SLIPPAGE, commission=COMMISSION,
        )

        wt_str = f"{conf_cfg.w_adf:.0%}/{conf_cfg.w_hurst:.0%}/{conf_cfg.w_corr:.0%}/{conf_cfg.w_hl:.0%}"
        print(f"  {preset_name:<14} {wt_str:>22} | {bt['trades']:>5} {bt['win_rate']:>5.1f}% "
              f"${bt['pnl']:>9,.0f} {bt['profit_factor']:>6.2f} "
              f"${bt['avg_pnl_trade']:>6,.0f}")

    # Ablation per metric profile
    print(f"\n\n  ABLATION PAR PROFIL METRIQUE (memes poids standard, conf={BASE['min_confidence']}):")
    print(f"  {'Profile':<14} {'ADF/Hurst/HL/Corr win':>24} | {'Trd':>5} {'WR%':>6} "
          f"{'PnL':>10} {'PF':>6} {'Avg$':>7}")
    print("  " + "-" * 80)

    std_conf = CONF_PRESETS["standard"]
    for prof_name, prof_cfg in METRIC_PROFILES.items():
        metrics_p = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], prof_cfg)
        confidence_p = compute_confidence(metrics_p, std_conf).values
        sig = _apply_conf_filter_numba(raw.copy(), confidence_p, BASE["min_confidence"])
        sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

        bt = run_backtest_grid(
            px_a, px_b, sig, beta,
            mult_a=MULT_A, mult_b=MULT_B,
            tick_a=TICK_A, tick_b=TICK_B,
            slippage_ticks=SLIPPAGE, commission=COMMISSION,
        )

        win_str = f"{prof_cfg.adf_window}/{prof_cfg.hurst_window}/{prof_cfg.halflife_window}/{prof_cfg.correlation_window}"
        print(f"  {prof_name:<14} {win_str:>24} | {bt['trades']:>5} {bt['win_rate']:>5.1f}% "
              f"${bt['pnl']:>9,.0f} {bt['profit_factor']:>6.2f} "
              f"${bt['avg_pnl_trade']:>6,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Refine OLS Balanced NQ/RTY")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ================================================================
    # ETAPE 1: ABLATION
    # ================================================================
    run_ablation()

    # ================================================================
    # ETAPE 2: GRID AFFINE
    # ================================================================
    print(f"\n\n{'='*100}")
    print("  ETAPE 2 : GRID AFFINE")
    print("=" * 100)

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

    n_jobs = (len(OLS_WINDOWS) * len(ZSCORE_WINDOWS) * len(PROFILES)
              * len(ENTRY_WINDOWS) * len(CONF_WEIGHT_NAMES))
    total = n_jobs * signal_combos

    log.info(f"Grid affine NQ/RTY OLS:")
    log.info(f"  OLS windows:     {len(OLS_WINDOWS)} {OLS_WINDOWS}")
    log.info(f"  Z-score windows: {len(ZSCORE_WINDOWS)} {ZSCORE_WINDOWS}")
    log.info(f"  z_entry:         {len(Z_ENTRIES)} {Z_ENTRIES}")
    log.info(f"  z_exit:          {len(Z_EXITS)} {Z_EXITS}")
    log.info(f"  z_stop:          {len(Z_STOPS)} {Z_STOPS}")
    log.info(f"  min_confidence:  {len(MIN_CONFIDENCES)} {MIN_CONFIDENCES}")
    log.info(f"  profiles:        {len(PROFILES)} {PROFILES}")
    log.info(f"  entry windows:   {len(ENTRY_WINDOWS)} {ENTRY_WINDOWS}")
    log.info(f"  conf weights:    {len(CONF_WEIGHT_NAMES)} {CONF_WEIGHT_NAMES}")
    log.info(f"  Signal combos/job: {signal_combos}")
    log.info(f"  Jobs: {n_jobs}")
    log.info(f"  TOTAL BACKTESTS: {total:,}")

    if args.dry_run:
        log.info("DRY RUN -- stopping here.")
        return

    # Build jobs
    jobs = []
    for ols_w in OLS_WINDOWS:
        for zw in ZSCORE_WINDOWS:
            for prof_name in PROFILES:
                for wlabel in ENTRY_WINDOWS:
                    es, ee = WINDOWS_MAP[wlabel]
                    for cw_name in CONF_WEIGHT_NAMES:
                        jobs.append(RefinedJob(
                            ols_window=ols_w,
                            zscore_window=zw,
                            profile_name=prof_name,
                            window_label=wlabel,
                            entry_start_min=es,
                            entry_end_min=ee,
                            conf_weight_name=cw_name,
                        ))

    log.info(f"Launching {len(jobs)} jobs with {args.workers} workers...")
    t0 = time.time()

    all_results = []
    errors = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_refined_job, j): j for j in jobs}
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

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_all = OUTPUT_DIR / "NQ_RTY" / "grid_refined_ols.csv"
    df.to_csv(csv_all, index=False)
    log.info(f"Saved {len(df):,} rows to {csv_all}")

    # Filter: PnL > 0, PF > 1.3, trades > 150
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["profit_factor"])
    filtered = df_clean[(df_clean["pnl"] > 0) & (df_clean["profit_factor"] > 1.3) & (df_clean["trades"] > 150)]
    csv_filt = OUTPUT_DIR / "NQ_RTY" / "grid_refined_ols_filtered.csv"
    filtered.to_csv(csv_filt, index=False)
    profitable = df[df["pnl"] > 0]
    log.info(f"Profitable: {len(profitable):,} / {len(df):,} ({len(profitable)/max(len(df),1)*100:.1f}%)")
    log.info(f"Filtered (PnL>0, PF>1.3, trades>150): {len(filtered):,}")

    # ================================================================
    # TOP 20 from filtered set (PnL>0, PF>1.3, trades>150)
    # ================================================================
    if not filtered.empty:
        top_pnl = filtered.nlargest(20, "pnl")
        log.info(f"\nTOP 20 BY PNL (PF>1.3, trades>150):")
        log.info(f"  {'OLS':>6} {'ZW':>4} {'Prof':<10} {'Win':<12} "
                 f"{'ze':>5} {'zx':>4} {'zs':>4} {'conf':>4} | "
                 f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7}")
        log.info("  " + "-" * 100)
        for _, r in top_pnl.iterrows():
            log.info(f"  {r['ols_window']:>6} {r['zscore_window']:>4} {r['profil']:<10} "
                     f"{r['window']:<12} "
                     f"{r['z_entry']:>5.2f} {r['z_exit']:>4.2f} {r['z_stop']:>4.1f} "
                     f"{r['min_confidence']:>4.0f} | "
                     f"{r['trades']:>5.0f} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} "
                     f"{r['profit_factor']:>6.2f} ${r['avg_pnl_trade']:>6,.0f}")

        top_pf = filtered.nlargest(20, "profit_factor")
        log.info(f"\nTOP 20 BY PF (PF>1.3, trades>150):")
        log.info(f"  {'OLS':>6} {'ZW':>4} {'Prof':<10} {'Win':<12} "
                 f"{'ze':>5} {'zx':>4} {'zs':>4} {'conf':>4} | "
                 f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7}")
        log.info("  " + "-" * 100)
        for _, r in top_pf.iterrows():
            log.info(f"  {r['ols_window']:>6} {r['zscore_window']:>4} {r['profil']:<10} "
                     f"{r['window']:<12} "
                     f"{r['z_entry']:>5.2f} {r['z_exit']:>4.2f} {r['z_stop']:>4.1f} "
                     f"{r['min_confidence']:>4.0f} | "
                     f"{r['trades']:>5.0f} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} "
                     f"{r['profit_factor']:>6.2f} ${r['avg_pnl_trade']:>6,.0f}")
    else:
        log.info("AUCUNE config ne passe le filtre PnL>0 + PF>1.3 + trades>150")


if __name__ == "__main__":
    main()
