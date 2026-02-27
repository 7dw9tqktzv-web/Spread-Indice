"""Etape 0 — Diagnostic du spread pour NQ/RTY et ES/RTY vs NQ/YM reference.

Usage:
    python scripts/diagnostic_spread.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.spread.pair import SpreadPair
from src.stats.correlation import rolling_correlation
from src.stats.halflife import half_life_rolling
from src.stats.hurst import hurst_rolling
from src.stats.stationarity import adf_statistic_simple
from src.utils.constants import Instrument

pairs_info = [
    ("NQ_YM", Instrument.NQ, Instrument.YM),
    ("NQ_RTY", Instrument.NQ, Instrument.RTY),
    ("ES_RTY", Instrument.ES, Instrument.RTY),
]

results = {}

for pair_name, leg_a, leg_b in pairs_info:
    print(f"\n{'='*80}")
    print(f" DIAGNOSTIC: {pair_name}")
    print(f"{'='*80}")

    pair = SpreadPair(leg_a=leg_a, leg_b=leg_b)
    aligned = load_aligned_pair_cache(pair, "5min")

    if aligned is None:
        print(f"  ERREUR: pas de cache pour {pair_name}")
        continue

    n = len(aligned.df)
    idx = aligned.df.index
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"  Data: {n:,} bars, {years:.1f} ans")
    print(f"  Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")

    # OLS spread (window=2640 = ~10j)
    est = create_estimator("ols_rolling", window=2640, zscore_window=30)
    hr = est.estimate(aligned)
    spread = hr.spread.dropna()
    beta = hr.beta.dropna()

    print(f"  Spread bars (non-NaN): {len(spread):,}")
    print(f"  Beta median: {beta.median():.4f}, mean: {beta.mean():.4f}, std: {beta.std():.4f}")

    # 1. Hurst (variance-ratio)
    hurst = hurst_rolling(spread, window=64)
    hurst_clean = hurst.dropna()
    hurst_median = float(hurst_clean.median())
    hurst_mean = float(hurst_clean.mean())

    # 2. ADF statistic (simple, rolling)
    # adf_statistic_simple returns a Series of rolling stats — take last value of each daily chunk
    adf_series = adf_statistic_simple(spread, window=264, step=1)
    adf_clean = adf_series.dropna()
    # Sample daily (every 264 bars)
    adf_sampled = adf_clean.iloc[::264]
    adf_pct_significant = float((adf_sampled < -2.86).mean() * 100)
    adf_median = float(adf_sampled.median())

    # 3. Correlation rolling
    corr = rolling_correlation(aligned.df["close_a"], aligned.df["close_b"], window=264)
    corr_clean = corr.dropna()
    corr_mean = float(corr_clean.mean())
    corr_std = float(corr_clean.std())
    corr_min = float(corr_clean.min())

    # 4. Spread drift annuel
    yearly_spread = spread.groupby(spread.index.year).agg(["first", "last"])
    drift_annual = []
    for _y, row in yearly_spread.iterrows():
        d = float(row["last"] - row["first"])
        drift_annual.append(d)
    drift_mean = float(np.mean(drift_annual)) if drift_annual else 0
    drift_abs_mean = float(np.mean(np.abs(drift_annual))) if drift_annual else 0

    # 5. ACF(1) du spread diff
    spread_diff = spread.diff().dropna()
    acf1 = float(spread_diff.autocorr(lag=1))

    # 6. Half-life
    hl = half_life_rolling(spread, window=264)
    hl_clean = hl.dropna()
    hl_clean = hl_clean[(hl_clean > 0) & (hl_clean < 500)]
    hl_median = float(hl_clean.median()) if len(hl_clean) > 0 else float("nan")

    results[pair_name] = {
        "hurst_median": hurst_median,
        "hurst_mean": hurst_mean,
        "adf_pct_sig": adf_pct_significant,
        "adf_median": adf_median,
        "corr_mean": corr_mean,
        "corr_std": corr_std,
        "corr_min": corr_min,
        "drift_mean": drift_mean,
        "drift_abs_mean": drift_abs_mean,
        "acf1": acf1,
        "hl_median": hl_median,
        "beta_median": float(beta.median()),
        "beta_std": float(beta.std()),
        "n_bars": n,
        "years": years,
    }

    print(f"  Hurst median: {hurst_median:.3f} (mean: {hurst_mean:.3f})")
    print(f"  ADF % significant: {adf_pct_significant:.1f}%")
    print(f"  ADF median stat: {adf_median:.3f}")
    print(f"  Correlation mean: {corr_mean:.3f} (std: {corr_std:.3f}, min: {corr_min:.3f})")
    print(f"  Spread drift mean: {drift_mean:.4f} (abs mean: {drift_abs_mean:.4f})")
    print(f"  ACF(1) spread diff: {acf1:.4f}")
    print(f"  Half-life median: {hl_median:.1f} bars")

# ================================================================
# Tableau comparatif
# ================================================================
print(f"\n\n{'='*100}")
print(" TABLEAU COMPARATIF -- DIAGNOSTIC SPREAD")
print(f"{'='*100}")
print(f"  {'Metrique':<25} {'NQ/YM (ref)':>15} {'NQ/RTY':>15} {'ES/RTY':>15}")
print(f"  {'-'*75}")

metrics_list = [
    ("Hurst median", "hurst_median", ".3f"),
    ("ADF % significant", "adf_pct_sig", ".1f"),
    ("ADF median stat", "adf_median", ".3f"),
    ("Correlation mean", "corr_mean", ".3f"),
    ("Correlation min", "corr_min", ".3f"),
    ("Drift annuel mean", "drift_mean", ".4f"),
    ("Drift abs mean", "drift_abs_mean", ".4f"),
    ("ACF(1) spread diff", "acf1", ".4f"),
    ("Half-life median", "hl_median", ".1f"),
    ("Beta median", "beta_median", ".4f"),
    ("Beta std", "beta_std", ".4f"),
]

for label, key, fmt in metrics_list:
    vals = []
    for pname in ["NQ_YM", "NQ_RTY", "ES_RTY"]:
        if pname in results:
            v = results[pname][key]
            vals.append(f"{v:{fmt}}")
        else:
            vals.append("N/A")
    print(f"  {label:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")


# ================================================================
# Yearly breakdown (drift + correlation)
# ================================================================
print(f"\n\n{'='*100}")
print(" DECOMPOSITION ANNUELLE -- DRIFT + CORRELATION")
print(f"{'='*100}")

for pair_name, leg_a, leg_b in pairs_info:
    pair = SpreadPair(leg_a=leg_a, leg_b=leg_b)
    aligned = load_aligned_pair_cache(pair, "5min")
    est = create_estimator("ols_rolling", window=2640, zscore_window=30)
    hr = est.estimate(aligned)
    spread = hr.spread.dropna()

    corr = rolling_correlation(aligned.df["close_a"], aligned.df["close_b"], window=264)

    print(f"\n  {pair_name}:")
    print(f"  {'Year':>6} {'Drift':>10} {'Corr mean':>10} {'Spread std':>10}")
    print(f"  {'-'*40}")

    for y in sorted(spread.index.year.unique()):
        mask = spread.index.year == y
        s_y = spread[mask]
        if len(s_y) < 100:
            continue
        drift_y = float(s_y.iloc[-1] - s_y.iloc[0])
        std_y = float(s_y.std())

        corr_y = corr[corr.index.year == y].dropna()
        corr_mean_y = float(corr_y.mean()) if len(corr_y) > 0 else 0

        print(f"  {y:>6} {drift_y:>10.4f} {corr_mean_y:>10.3f} {std_y:>10.4f}")


# ================================================================
# Verdicts
# ================================================================
print(f"\n\n{'='*100}")
print(" VERDICTS")
print(f"{'='*100}")
for pname in ["NQ_RTY", "ES_RTY"]:
    if pname not in results:
        continue
    r = results[pname]
    h = r["hurst_median"]
    adf_p = r["adf_pct_sig"]
    c = r["corr_mean"]

    if h > 0.55 and adf_p < 5:
        verdict = "NON VIABLE"
    elif h > 0.50 or adf_p < 20:
        verdict = "MARGINALE"
    elif h < 0.45:
        verdict = "PROMETTEUSE"
    else:
        verdict = "VIABLE"

    flags = []
    if h > 0.50:
        flags.append(f"Hurst eleve ({h:.3f})")
    if adf_p < 30:
        flags.append(f"ADF faible ({adf_p:.1f}%)")
    if c < 0.80:
        flags.append(f"Corr faible ({c:.3f})")

    warning = " | WARNINGS: " + ", ".join(flags) if flags else ""
    print(f"  {pname}: {verdict}{warning}")

    # Compare to NQ/YM reference
    ref = results["NQ_YM"]
    print(
        f"    vs NQ/YM: Hurst {h:.3f} vs {ref['hurst_median']:.3f}, "
        f"ADF {adf_p:.1f}% vs {ref['adf_pct_sig']:.1f}%, "
        f"Corr {c:.3f} vs {ref['corr_mean']:.3f}"
    )
