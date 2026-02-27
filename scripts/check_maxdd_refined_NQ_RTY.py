"""MaxDD check pour top candidats grid raffinee OLS NQ/RTY.

Selectionne ~15 configs diversifiees (top PnL, top PF, top volume),
calcule MaxDD + yearly breakdown via run_backtest_vectorized.

Usage:
    python scripts/check_maxdd_refined_NQ_RTY.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
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
from src.config.instruments import get_pair_specs

_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

METRIC_PROFILES = {
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
    "p18_192":   MetricsConfig(adf_window=18, hurst_window=192, halflife_window=18, correlation_window=18),
    "p24_256":   MetricsConfig(adf_window=24, hurst_window=256, halflife_window=24, correlation_window=24),
    "p30_256":   MetricsConfig(adf_window=30, hurst_window=256, halflife_window=30, correlation_window=24),
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

# HL retire du scoring (ablation: +155% trades, PnL quasi identique)
CONF_WEIGHTS = ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00)


def main():
    csv_path = PROJECT_ROOT / "output" / "NQ_RTY" / "grid_refined_ols_filtered.csv"
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["profit_factor"])
    df = df[df["profit_factor"] < 100]
    print(f"Total configs filtrees: {len(df):,}")

    # Select diversified candidates
    candidates = []

    # Top 5 by PnL (deduplicate by OLS+ZW cluster)
    top_pnl = df.copy()
    top_pnl["cluster"] = (top_pnl["ols_window"].astype(str) + "_" +
                           top_pnl["zscore_window"].astype(str))
    for _, r in top_pnl.sort_values("pnl", ascending=False).drop_duplicates("cluster").head(5).iterrows():
        candidates.append(("TopPnL", r))

    # Top 5 by PF (deduplicate by cluster)
    top_pf = df.copy()
    top_pf["cluster"] = (top_pf["ols_window"].astype(str) + "_" +
                          top_pf["zscore_window"].astype(str))
    for _, r in top_pf.sort_values("profit_factor", ascending=False).drop_duplicates("cluster").head(5).iterrows():
        candidates.append(("TopPF", r))

    # Top 5 by volume with PF > 1.5
    high_pf = df[df["profit_factor"] > 1.5].copy()
    high_pf["cluster"] = (high_pf["ols_window"].astype(str) + "_" +
                           high_pf["zscore_window"].astype(str))
    for _, r in high_pf.sort_values("trades", ascending=False).drop_duplicates("cluster").head(5).iterrows():
        candidates.append(("TopVol", r))

    # Deduplicate candidates by full config
    seen = set()
    unique_candidates = []
    for label, r in candidates:
        key = (int(r["ols_window"]), int(r["zscore_window"]), r["profil"], r["window"],
               r["z_entry"], r["z_exit"], r["z_stop"], r["min_confidence"])
        if key not in seen:
            seen.add(key)
            unique_candidates.append((label, r))

    print(f"Candidats uniques a tester: {len(unique_candidates)}")

    # Load data once
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # Cache hedge ratios per (ols_window, zscore_window)
    hedge_cache = {}

    results = []

    for i, (label, row) in enumerate(unique_candidates):
        ols_w = int(row["ols_window"])
        zw = int(row["zscore_window"])
        profile = row["profil"]
        window = row["window"]
        ze = row["z_entry"]
        zx = row["z_exit"]
        zs = row["z_stop"]
        conf = row["min_confidence"]

        print(f"  [{i+1}/{len(unique_candidates)}] {label}: OLS={ols_w} ZW={zw} {profile} {window} "
              f"ze={ze:.2f} zx={zx:.2f} zs={zs:.1f} conf={conf:.0f}", end=" ... ", flush=True)

        # Get hedge ratio (cached)
        hkey = (ols_w, zw)
        if hkey not in hedge_cache:
            est = create_estimator("ols_rolling", window=ols_w, zscore_window=zw)
            hr = est.estimate(aligned)
            spread = hr.spread
            beta = hr.beta.values
            mu = spread.rolling(zw).mean()
            sigma = spread.rolling(zw).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
            zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)
            hedge_cache[hkey] = (beta, zscore, spread)
        else:
            beta, zscore, spread = hedge_cache[hkey]

        # Metrics + confidence (HL weight = 0)
        profile_cfg = METRIC_PROFILES[profile]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, CONF_WEIGHTS).values

        raw = generate_signals_numba(zscore, ze, zx, zs)
        if conf > 0:
            sig = _apply_conf_filter_numba(raw, confidence, conf)
        else:
            sig = raw.copy()

        entry_start, entry_end = WINDOWS_MAP[window]
        sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        # MaxDD
        equity = bt["equity"]
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd = drawdown.min()

        # Max losing streak
        trade_pnls = bt["trade_pnls"]
        max_streak = 0
        current = 0
        for p in trade_pnls:
            if p < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        # Yearly
        entry_bars = bt["trade_entry_bars"]
        entry_dates = idx[entry_bars]
        trade_df = pd.DataFrame({"year": entry_dates.year, "pnl": trade_pnls})
        yearly = trade_df.groupby("year")["pnl"].sum()
        neg_years = (yearly < 0).sum()

        print(f"MaxDD=${max_dd:,.0f}", flush=True)

        results.append({
            "label": label,
            "ols_w": ols_w, "zw": zw, "profile": profile, "window": window,
            "ze": ze, "zx": zx, "zs": zs, "conf": conf,
            "trades": bt["trades"], "wr": bt["win_rate"], "pnl": bt["pnl"],
            "pf": bt["profit_factor"], "max_dd": max_dd, "avg_pnl": bt["avg_pnl_trade"],
            "max_streak": max_streak, "neg_years": neg_years,
            "yearly": yearly,
        })

    # ================================================================
    # DISPLAY RESULTS
    # ================================================================
    print(f"\n{'='*140}")
    print(f"  RESULTATS MAXDD — GRID RAFFINEE NQ/RTY (tries par PnL)")
    print(f"{'='*140}")

    # Sort by PnL
    results.sort(key=lambda x: -x["pnl"])

    # Safe configs (MaxDD < $8,000)
    safe = [r for r in results if abs(r["max_dd"]) < 8000]
    danger = [r for r in results if abs(r["max_dd"]) >= 8000]

    print(f"\n  {len(safe)} SAFE (MaxDD < $8,000) / {len(danger)} DANGER / {len(results)} total\n")

    header = (f"  {'Cat':<7} {'OLS':>6} {'ZW':>3} {'prof':<10} {'window':<14} "
              f"{'ze':>5} {'zx':>5} {'zs':>4} {'conf':>4} | "
              f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'MaxDD':>9} {'Avg$':>7} {'Strk':>4} {'Neg':>3}")
    print(header)
    print("  " + "-" * 130)

    for r in results:
        dd_flag = " <<<" if abs(r["max_dd"]) >= 8000 else ""
        safe_flag = " SAFE" if abs(r["max_dd"]) < 5000 else ""
        print(f"  {r['label']:<7} {r['ols_w']:>6} {r['zw']:>3} {r['profile']:<10} {r['window']:<14} "
              f"{r['ze']:>5.2f} {r['zx']:>5.2f} {r['zs']:>4.1f} {r['conf']:>4.0f} | "
              f"{r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>6.2f} "
              f"${r['max_dd']:>8,.0f} ${r['avg_pnl']:>6,.0f} {r['max_streak']:>4} {r['neg_years']:>3}"
              f"{dd_flag}{safe_flag}")
        # Yearly
        years_str = "          "
        for y, p in r["yearly"].items():
            flag = "***" if p < 0 else ""
            years_str += f"{y}=${p:+,.0f}{flag}  "
        print(years_str)
        print()


if __name__ == "__main__":
    main()
