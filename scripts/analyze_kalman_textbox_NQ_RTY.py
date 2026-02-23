"""Analyse du modele Kalman NQ/RTY pour la textbox Sierra.

Montre les statistiques typiques du Kalman: beta, z-score, direction.
Compare avec NQ/YM K_Balanced pour reference.

Usage:
    python scripts/analyze_kalman_textbox_NQ_RTY.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator


def analyze_kalman(pair_name, leg_a, leg_b, alpha):
    """Analyze Kalman output for a pair."""
    print(f"\n{'='*100}")
    print(f"  KALMAN TEXTBOX: {pair_name} (alpha={alpha:.1e})")
    print(f"{'='*100}")

    pair = SpreadPair(leg_a=leg_a, leg_b=leg_b)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print(f"  ERREUR: pas de cache")
        return

    est = create_estimator("kalman", alpha_ratio=alpha, warmup=200, gap_P_multiplier=5.0)
    hr = est.estimate(aligned)

    idx = aligned.df.index
    beta = hr.beta.values
    zscore = hr.zscore.values
    spread = hr.spread.values

    # Filter out warmup NaN
    valid = ~np.isnan(beta) & ~np.isnan(zscore)
    beta_v = beta[valid]
    zscore_v = zscore[valid]
    spread_v = spread[valid] if not np.isnan(spread).all() else spread[valid]

    # === BETA ===
    print(f"\n  --- BETA (hedge ratio) ---")
    print(f"  Mean:   {np.mean(beta_v):.4f}")
    print(f"  Median: {np.median(beta_v):.4f}")
    print(f"  Std:    {np.std(beta_v):.4f}")
    print(f"  Min:    {np.min(beta_v):.4f}")
    print(f"  Max:    {np.max(beta_v):.4f}")
    print(f"  Range:  {np.max(beta_v) - np.min(beta_v):.4f}")

    # Beta stability (daily changes)
    beta_series = pd.Series(beta_v, index=idx[valid])
    daily_beta = beta_series.resample("D").last().dropna()
    daily_beta_change = daily_beta.diff().dropna()
    print(f"  Daily beta change: mean={daily_beta_change.mean():.6f}, std={daily_beta_change.std():.6f}")
    print(f"  Beta change > 1% of beta: {(daily_beta_change.abs() > abs(daily_beta.mean()) * 0.01).sum()} jours")

    # === Z-SCORE ===
    print(f"\n  --- Z-SCORE (innovation) ---")
    zscore_clean = zscore_v[~np.isinf(zscore_v)]
    print(f"  Mean:   {np.mean(zscore_clean):.4f}")
    print(f"  Std:    {np.std(zscore_clean):.4f}")
    print(f"  Min:    {np.min(zscore_clean):.4f}")
    print(f"  Max:    {np.max(zscore_clean):.4f}")
    print(f"  |z| > 1.0: {(np.abs(zscore_clean) > 1.0).sum()} bars ({(np.abs(zscore_clean) > 1.0).mean()*100:.1f}%)")
    print(f"  |z| > 1.5: {(np.abs(zscore_clean) > 1.5).sum()} bars ({(np.abs(zscore_clean) > 1.5).mean()*100:.1f}%)")
    print(f"  |z| > 2.0: {(np.abs(zscore_clean) > 2.0).sum()} bars ({(np.abs(zscore_clean) > 2.0).mean()*100:.1f}%)")
    print(f"  |z| > 2.5: {(np.abs(zscore_clean) > 2.5).sum()} bars ({(np.abs(zscore_clean) > 2.5).mean()*100:.1f}%)")

    # Z-score distribution check (should be ~N(0,1))
    from scipy.stats import jarque_bera, normaltest
    jb_stat, jb_p = jarque_bera(zscore_clean[:10000])  # subsample for speed
    print(f"  Jarque-Bera (N(0,1) test): stat={jb_stat:.1f}, p={jb_p:.4f}")

    # === DIRECTION SIGNAL ===
    print(f"\n  --- DIRECTION (z-score sign) ---")
    positive_z = (zscore_clean > 0).sum()
    negative_z = (zscore_clean < 0).sum()
    print(f"  z > 0 (short spread): {positive_z} bars ({positive_z/len(zscore_clean)*100:.1f}%)")
    print(f"  z < 0 (long spread):  {negative_z} bars ({negative_z/len(zscore_clean)*100:.1f}%)")

    # Direction by year
    zscore_series = pd.Series(zscore_clean, index=idx[valid][:len(zscore_clean)])
    yearly_direction = zscore_series.groupby(zscore_series.index.year).apply(
        lambda x: (x > 0).sum() / len(x) * 100
    )
    print(f"\n  Direction par annee (% z>0 = short bias):")
    for y, pct in yearly_direction.items():
        bias = "SHORT" if pct > 55 else ("LONG" if pct < 45 else "NEUTRAL")
        print(f"    {y}: {pct:.1f}% short | {100-pct:.1f}% long  [{bias}]")

    # === SPREAD ===
    print(f"\n  --- SPREAD (log_NQ - beta * log_RTY) ---")
    spread_clean = spread_v[~np.isnan(spread_v)]
    if len(spread_clean) > 0:
        print(f"  Mean:   {np.mean(spread_clean):.6f}")
        print(f"  Std:    {np.std(spread_clean):.6f}")
        print(f"  Min:    {np.min(spread_clean):.6f}")
        print(f"  Max:    {np.max(spread_clean):.6f}")

    # === DIAGNOSTICS (P_trace, K_beta) ===
    if hr.diagnostics is not None:
        print(f"\n  --- DIAGNOSTICS ---")
        diag = hr.diagnostics
        if isinstance(diag, dict):
            for k, v in diag.items():
                if hasattr(v, 'iloc') and len(v) > 0:
                    vals = v.dropna()
                    if len(vals) > 0:
                        print(f"  {k}: first={vals.iloc[0]:.6f}, last={vals.iloc[-1]:.6f}")
        elif hasattr(diag, 'columns'):
            if "P_trace" in diag.columns:
                pt = diag["P_trace"].dropna()
                print(f"  P_trace: first={pt.iloc[0]:.6f}, last={pt.iloc[-1]:.6f}")
            if "K_beta" in diag.columns:
                kb = diag["K_beta"].dropna()
                print(f"  K_beta (gain): mean={kb.mean():.6f}")

    # === TEXTBOX MOCKUP ===
    print(f"\n  --- TEXTBOX SIERRA (exemple derniere valeur) ---")
    last_i = len(beta) - 1
    last_beta = beta[last_i]
    last_z = zscore[last_i] if not np.isnan(zscore[last_i]) else 0
    last_date = idx[last_i]
    direction = "SHORT" if last_z > 0 else "LONG"

    print(f"  +----------------------------------+")
    print(f"  | {pair_name} Kalman a={alpha:.0e}      |")
    print(f"  | Beta: {last_beta:>8.4f}                |")
    print(f"  | Z-score: {last_z:>+7.3f}              |")
    print(f"  | Direction: {direction:<6}              |")
    print(f"  | {last_date.strftime('%Y-%m-%d %H:%M')}             |")
    print(f"  +----------------------------------+")

    return {
        "beta_mean": np.mean(beta_v),
        "beta_std": np.std(beta_v),
        "zscore_std": np.std(zscore_clean),
    }


def main():
    # NQ/RTY avec alpha de K_PropFirm
    res_nrt = analyze_kalman("NQ_RTY", Instrument.NQ, Instrument.RTY, alpha=2.5e-7)

    # NQ/YM reference (K_Balanced)
    res_nym = analyze_kalman("NQ_YM", Instrument.NQ, Instrument.YM, alpha=3e-7)

    # Comparison
    if res_nrt and res_nym:
        print(f"\n\n{'='*100}")
        print(f"  COMPARAISON NQ/RTY vs NQ/YM")
        print(f"{'='*100}")
        print(f"  {'Metrique':<25} {'NQ/RTY':>12} {'NQ/YM':>12}")
        print(f"  {'-'*50}")
        print(f"  {'Beta mean':<25} {res_nrt['beta_mean']:>12.4f} {res_nym['beta_mean']:>12.4f}")
        print(f"  {'Beta std':<25} {res_nrt['beta_std']:>12.4f} {res_nym['beta_std']:>12.4f}")
        print(f"  {'Z-score std':<25} {res_nrt['zscore_std']:>12.4f} {res_nym['zscore_std']:>12.4f}")


if __name__ == "__main__":
    main()
