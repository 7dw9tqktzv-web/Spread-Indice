"""P2: Analyze Kalman diagnostics (P_trace, K_beta) vs trade outcomes.

Tests whether P_trace at entry predicts trade PnL.
If yes -> P3 (P_trace filter) has potential.
If no  -> skip P3, move to P4 (OU z-score).

Key analyses:
    1. P_trace mean at entry: winners vs losers
    2. P_trace percentile at entry vs trade PnL (Spearman rank correlation)
    3. P_trace by year (2023 vs rest)
    4. Top 10 worst trades: P_trace at entry
    5. K_beta stability (rolling std)

Usage:
    python scripts/analyze_kalman_diagnostics.py
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
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

# ======================================================================
# Constants (same as validate_kalman_top.py)
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930  # 15:30 CT

METRICS_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=66, hurst_window=132, halflife_window=66, correlation_window=66),
}

WINDOWS_MAP = {
    "03:00-12:00": (180, 720),
    "04:00-13:00": (240, 780),
    "05:00-12:00": (300, 720),
}

# ======================================================================
# Top 5 Kalman configs
# ======================================================================

CONFIGS = {
    "K_Sniper": {
        "alpha": 3e-7, "profil": "court",
        "z_entry": 1.8125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 65.0, "window": "05:00-12:00",
    },
    "K_BestPnL": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.375, "z_exit": 0.25, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
    },
    "K_Balanced": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
    },
    "K_Quality": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "04:00-13:00",
    },
    "K_ShortWin": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "05:00-12:00",
    },
}


# ======================================================================
# Pipeline
# ======================================================================

def run_analysis(aligned, px_a, px_b, idx, minutes, cfg):
    """Run backtest + extract diagnostics for one config."""
    est = create_estimator("kalman", alpha_ratio=cfg["alpha"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    zscore = np.ascontiguousarray(hr.zscore.values, dtype=np.float64)

    # Diagnostics
    p_trace = hr.diagnostics["P_trace"].values
    k_beta = hr.diagnostics["K_beta"].values

    # Signals
    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])
    metrics_cfg = METRICS_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(
        hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg
    )
    confidence = compute_confidence(metrics, CONF_CFG).values
    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    # Backtest
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )

    return bt, p_trace, k_beta


def analyze_config(cfg_name, cfg, bt, p_trace, k_beta, idx):
    """Analyze P_trace/K_beta vs trade outcomes for one config."""
    n_trades = bt["trades"]
    if n_trades == 0:
        print("  0 trades - skip")
        return

    entry_bars = bt["trade_entry_bars"]
    pnls = bt["trade_pnls"]
    entry_dates = idx[entry_bars]
    entry_years = entry_dates.year

    # P_trace at entry for each trade
    p_at_entry = p_trace[entry_bars]

    # ---- 1. Winners vs Losers ----
    winners = pnls > 0
    losers = pnls <= 0
    n_win = winners.sum()
    n_loss = losers.sum()

    p_win_mean = float(p_at_entry[winners].mean()) if n_win > 0 else 0
    p_loss_mean = float(p_at_entry[losers].mean()) if n_loss > 0 else 0
    p_all_mean = float(p_at_entry.mean())

    # Median too (more robust)
    p_win_med = float(np.median(p_at_entry[winners])) if n_win > 0 else 0
    p_loss_med = float(np.median(p_at_entry[losers])) if n_loss > 0 else 0

    print("\n  --- 1. P_trace at entry: Winners vs Losers ---")
    print(f"  {'':>20} {'Count':>6} {'P_mean':>14} {'P_median':>14}")
    print(f"  {'-' * 58}")
    print(f"  {'Winners':>20} {n_win:>6} {p_win_mean:>14.6e} {p_win_med:>14.6e}")
    print(f"  {'Losers':>20} {n_loss:>6} {p_loss_mean:>14.6e} {p_loss_med:>14.6e}")
    print(f"  {'All trades':>20} {n_trades:>6} {p_all_mean:>14.6e}")

    ratio = p_loss_mean / p_win_mean if p_win_mean > 0 else float('inf')
    print(f"  Ratio P_losers/P_winners: {ratio:.3f}x")

    # Mann-Whitney U test (non-parametric: is P_trace different for losers?)
    if n_win >= 5 and n_loss >= 5:
        u_stat, u_pval = stats.mannwhitneyu(
            p_at_entry[losers], p_at_entry[winners], alternative='greater'
        )
        print(f"  Mann-Whitney U (losers > winners): p={u_pval:.4f}"
              f" {'*** SIGNIFICANT' if u_pval < 0.05 else '(not significant)'}")

    # ---- 2. Spearman rank correlation: P_trace percentile vs PnL ----
    print("\n  --- 2. Spearman rank correlation: P_trace at entry vs Trade PnL ---")

    # Compute percentile rank of P_trace at entry (relative to full P_trace history)
    p_valid = p_trace[~np.isnan(p_trace)]
    p_percentiles = np.array([
        float(stats.percentileofscore(p_valid, p, kind='rank')) for p in p_at_entry
    ])

    # Spearman: P_trace percentile vs PnL
    rho_pct, pval_pct = stats.spearmanr(p_percentiles, pnls)
    print(f"  P_trace percentile vs PnL:  rho={rho_pct:+.4f}, p={pval_pct:.4f}"
          f" {'*** SIGNIFICANT' if pval_pct < 0.05 else '(not significant)'}")

    # Also raw P_trace vs PnL (in case percentile transform loses info)
    rho_raw, pval_raw = stats.spearmanr(p_at_entry, pnls)
    print(f"  P_trace raw vs PnL:         rho={rho_raw:+.4f}, p={pval_raw:.4f}"
          f" {'*** SIGNIFICANT' if pval_raw < 0.05 else '(not significant)'}")

    # ---- 2b. DECONFOUNDING: Spearman within each year ----
    print("\n  --- 2b. DECONFOUNDING: Spearman INTRA-ANNEE (retire le confound temporel) ---")
    print(f"  {'Year':>6} {'Trd':>5} {'rho':>8} {'p-value':>9} {'Sig?':>15}")
    print(f"  {'-' * 50}")

    intra_rhos = []
    intra_n = []
    for y in sorted(entry_years.unique()):
        mask = entry_years == y
        n_y = mask.sum()
        if n_y >= 10:  # need minimum trades for meaningful Spearman
            rho_y, pval_y = stats.spearmanr(p_at_entry[mask], pnls[mask])
            sig_y = "*** SIGNIFICANT" if pval_y < 0.05 else ""
            print(f"  {y:>6} {n_y:>5} {rho_y:>+8.4f} {pval_y:>9.4f} {sig_y:>15}")
            intra_rhos.append(rho_y)
            intra_n.append(n_y)
        else:
            print(f"  {y:>6} {n_y:>5}     (too few trades)")

    # Weighted average of intra-year rhos
    if intra_rhos:
        weights = np.array(intra_n, dtype=float)
        avg_rho = np.average(intra_rhos, weights=weights)
        print(f"  Weighted avg intra-year rho: {avg_rho:+.4f}")

    # ---- 2c. PARTIAL CORRELATION: P_trace vs PnL controlling for time ----
    print("\n  --- 2c. PARTIAL CORRELATION: P_trace vs PnL | controlling for entry_bar ---")
    # Partial Spearman: regress out time from both P_trace and PnL, then correlate residuals
    time_ranks = stats.rankdata(entry_bars)
    p_ranks = stats.rankdata(p_at_entry)
    pnl_ranks = stats.rankdata(pnls)

    # Residualize P_trace ranks on time ranks
    slope_p, intercept_p, _, _, _ = stats.linregress(time_ranks, p_ranks)
    p_resid = p_ranks - (slope_p * time_ranks + intercept_p)

    # Residualize PnL ranks on time ranks
    slope_pnl, intercept_pnl, _, _, _ = stats.linregress(time_ranks, pnl_ranks)
    pnl_resid = pnl_ranks - (slope_pnl * time_ranks + intercept_pnl)

    # Correlation of residuals = partial Spearman
    rho_partial, pval_partial = stats.spearmanr(p_resid, pnl_resid)
    print(f"  Partial rho (time removed): {rho_partial:+.4f}, p={pval_partial:.4f}"
          f" {'*** SIGNIFICANT' if pval_partial < 0.05 else '(not significant)'}")

    # Also: direct Spearman of entry_bar vs PnL (how much does time predict PnL?)
    rho_time, pval_time = stats.spearmanr(entry_bars, pnls)
    print(f"  Time vs PnL (confound):     rho={rho_time:+.4f}, p={pval_time:.4f}")
    rho_time_p, pval_time_p = stats.spearmanr(entry_bars, p_at_entry)
    print(f"  Time vs P_trace:            rho={rho_time_p:+.4f}, p={pval_time_p:.4f}")

    # Interpretation
    print("\n  --- INTERPRETATION ---")
    if len(intra_rhos) > 0 and avg_rho < -0.05 and pval_partial < 0.10:
        print("  >> P_trace a un pouvoir predictif REEL (intra-annee + partiel) -> P3 viable")
        verdict_p3 = "REEL"
    elif len(intra_rhos) > 0 and avg_rho > -0.05:
        print("  >> P_trace est un PROXY TEMPOREL (intra-annee rho ~0) -> P3 inutile")
        verdict_p3 = "PROXY"
    else:
        print("  >> Signal ambigu -> P3 a risque")
        verdict_p3 = "AMBIGU"

    # ---- 3. P_trace by year ----
    print("\n  --- 3. P_trace at entry by year ---")
    print(f"  {'Year':>6} {'Trd':>5} {'P_mean':>14} {'P_med':>14} {'PnL':>10} {'WR%':>6}")
    print(f"  {'-' * 60}")

    for y in sorted(entry_years.unique()):
        mask = entry_years == y
        p_y = p_at_entry[mask]
        pnl_y = pnls[mask]
        flag = " <-- TARGET" if y == 2023 else ""
        print(f"  {y:>6} {mask.sum():>5} {p_y.mean():>14.6e} {np.median(p_y):>14.6e}"
              f" ${pnl_y.sum():>9,.0f} {(pnl_y > 0).sum() / mask.sum() * 100:>5.1f}%{flag}")

    # ---- 4. P_trace quartile analysis ----
    print("\n  --- 4. P_trace quartile analysis ---")
    quartile_bounds = np.percentile(p_at_entry, [25, 50, 75])
    q_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    q_masks = [
        p_at_entry <= quartile_bounds[0],
        (p_at_entry > quartile_bounds[0]) & (p_at_entry <= quartile_bounds[1]),
        (p_at_entry > quartile_bounds[1]) & (p_at_entry <= quartile_bounds[2]),
        p_at_entry > quartile_bounds[2],
    ]
    print(f"  {'Quartile':<16} {'Trd':>5} {'Avg PnL':>10} {'WR%':>6} {'Total PnL':>11}")
    print(f"  {'-' * 55}")
    for label, qm in zip(q_labels, q_masks):
        n_q = qm.sum()
        if n_q > 0:
            avg_pnl = float(pnls[qm].mean())
            wr = float((pnls[qm] > 0).sum() / n_q * 100)
            total = float(pnls[qm].sum())
            print(f"  {label:<16} {n_q:>5} ${avg_pnl:>9,.0f} {wr:>5.1f}% ${total:>10,.0f}")

    # ---- 5. Top 10 worst trades ----
    print("\n  --- 5. Top 10 worst trades: P_trace at entry ---")
    worst_idx = np.argsort(pnls)[:10]
    print(f"  {'#':>3} {'Date':>12} {'PnL':>10} {'P_trace':>14} {'P_pctile':>9} {'K_beta':>12}")
    print(f"  {'-' * 65}")
    for rank, wi in enumerate(worst_idx, 1):
        eb = entry_bars[wi]
        pctile = float(stats.percentileofscore(p_valid, p_trace[eb], kind='rank'))
        print(f"  {rank:>3} {idx[eb].strftime('%Y-%m-%d'):>12} ${pnls[wi]:>9,.0f}"
              f" {p_trace[eb]:>14.6e} {pctile:>8.1f}% {k_beta[eb]:>12.6e}")

    # ---- 6. K_beta stability ----
    print("\n  --- 6. K_beta diagnostics ---")
    k_valid = k_beta[~np.isnan(k_beta)]
    # Split into halves
    half = len(k_valid) // 2
    k_first = k_valid[:half]
    k_last = k_valid[half:]
    print(f"  K_beta mean  : first_half={k_first.mean():.6e}, last_half={k_last.mean():.6e}")
    print(f"  K_beta std   : first_half={k_first.std():.6e}, last_half={k_last.std():.6e}")
    print(f"  K_beta range : [{k_valid.min():.6e}, {k_valid.max():.6e}]")

    # K_beta at entry: winners vs losers
    k_at_entry = k_beta[entry_bars]
    if n_win > 0 and n_loss > 0:
        print(f"  K_beta at entry: winners={k_at_entry[winners].mean():.6e},"
              f" losers={k_at_entry[losers].mean():.6e}")

    return {
        "rho_pct": rho_pct, "pval_pct": pval_pct,
        "rho_raw": rho_raw, "pval_raw": pval_raw,
        "p_ratio": ratio,
        "rho_partial": rho_partial, "pval_partial": pval_partial,
        "avg_intra_rho": avg_rho if intra_rhos else 0.0,
        "verdict_p3": verdict_p3,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time_mod.time()

    print("Loading NQ_YM 5min data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    print(f"Data: {len(px_a):,} bars, {(idx[-1] - idx[0]).days / 365.25:.1f} years")
    print(f"Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")

    # P_trace global stats (from K_Balanced as reference)
    ref_cfg = CONFIGS["K_Balanced"]
    est_ref = create_estimator("kalman", alpha_ratio=ref_cfg["alpha"])
    hr_ref = est_ref.estimate(aligned)
    p_global = hr_ref.diagnostics["P_trace"].dropna()
    print("\nP_trace global stats (K_Balanced reference):")
    print(f"  mean={p_global.mean():.6e}, std={p_global.std():.6e}")
    print(f"  min={p_global.min():.6e}, max={p_global.max():.6e}")
    print(f"  P25={p_global.quantile(0.25):.6e}, P50={p_global.quantile(0.50):.6e},"
          f" P75={p_global.quantile(0.75):.6e}, P95={p_global.quantile(0.95):.6e}")

    # Run analysis for each config
    summary = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n\n{'=' * 100}")
        print(f" {cfg_name} | alpha={cfg['alpha']:.0e} profil={cfg['profil']} "
              f"ze={cfg['z_entry']} zx={cfg['z_exit']} zs={cfg['z_stop']} "
              f"c={cfg['conf']}% win={cfg['window']}")
        print(f"{'=' * 100}")

        bt, p_trace, k_beta = run_analysis(aligned, px_a, px_b, idx, minutes, cfg)
        result = analyze_config(cfg_name, cfg, bt, p_trace, k_beta, idx)
        if result:
            summary[cfg_name] = result

    # ================================================================
    # Summary across all configs
    # ================================================================
    print(f"\n\n{'=' * 100}")
    print(" SUMMARY: P_trace as predictor of trade PnL")
    print(f"{'=' * 100}")
    print(f"\n  {'Config':<14} {'Global rho':>11} {'Intra-yr rho':>13} {'Partial rho':>12} {'p_partial':>10} {'Verdict':>12}")
    print(f"  {'-' * 80}")

    for cfg_name, res in summary.items():
        print(f"  {cfg_name:<14} {res['rho_pct']:>+11.4f} {res['avg_intra_rho']:>+13.4f}"
              f" {res['rho_partial']:>+12.4f} {res['pval_partial']:>10.4f} {res['verdict_p3']:>12}")

    # Global verdict
    n_real = sum(1 for r in summary.values() if r["verdict_p3"] == "REEL")
    n_proxy = sum(1 for r in summary.values() if r["verdict_p3"] == "PROXY")
    print(f"\n  P_trace REEL: {n_real}/{len(summary)}, PROXY temporel: {n_proxy}/{len(summary)}")
    if n_proxy >= 3:
        print("  >> VERDICT FINAL: P_trace est un PROXY TEMPOREL -> P3 INUTILE, passer a P4")
    elif n_real >= 3:
        print("  >> VERDICT FINAL: P_trace a un pouvoir predictif REEL -> P3 VIABLE")
    else:
        print("  >> VERDICT FINAL: Signal MIXTE -> P3 a risque, privilegier P4")

    elapsed = time_mod.time() - t_start
    print(f"\n{'=' * 100}")
    print(f" COMPLETE en {elapsed:.0f}s")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
