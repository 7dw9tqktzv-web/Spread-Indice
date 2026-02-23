"""MaxDD check pour candidats OLS NQ/RTY filtres (PF>1.4, trades>200, MaxDD<8k).

Usage:
    python scripts/check_maxdd_ols_NQ_RTY_v2.py
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

MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

METRIC_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court": MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "moyen": MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "03:00-12:00": (180, 720),
    "04:00-12:00": (240, 720),
    "04:00-13:00": (240, 780),
    "05:00-12:00": (300, 720),
    "06:00-11:00": (360, 660),
    "08:00-14:00": (480, 840),
}


def main():
    # Load grid results and filter
    csv_path = PROJECT_ROOT / "output" / "NQ_RTY" / "grid_ols_filtered.csv"
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["profit_factor"])
    df = df[df["profit_factor"] < 100]

    filt = df[(df["profit_factor"] > 1.4) & (df["trades"] > 200)].copy()
    print(f"Candidats PF>1.4, trades>200: {len(filt)}")

    # Deduplicate: pick top PnL per unique (ols_window, zscore_window, profil) cluster
    filt["cluster"] = (filt["ols_window"].astype(str) + "_" +
                       filt["zscore_window"].astype(str) + "_" +
                       filt["profil"])
    top_per_cluster = filt.sort_values("pnl", ascending=False).drop_duplicates("cluster", keep="first")
    candidates = top_per_cluster.head(20)
    print(f"Clusters uniques (top 20): {len(candidates)}")

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

    for _, row in candidates.iterrows():
        ols_w = int(row["ols_window"])
        zw = int(row["zscore_window"])
        profile = row["profil"]
        window = row["window"]
        ze = row["z_entry"]
        zx = row["z_exit"]
        zs = row["z_stop"]
        conf = row["min_confidence"]

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

        # Metrics + confidence
        profile_cfg = METRIC_PROFILES[profile]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, ConfidenceConfig()).values

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

        results.append({
            "ols_w": ols_w, "zw": zw, "profile": profile, "window": window,
            "ze": ze, "zx": zx, "zs": zs, "conf": conf,
            "trades": bt["trades"], "wr": bt["win_rate"], "pnl": bt["pnl"],
            "pf": bt["profit_factor"], "max_dd": max_dd, "avg_pnl": bt["avg_pnl_trade"],
            "max_streak": max_streak, "neg_years": neg_years,
            "yearly": yearly,
        })

    # Filter MaxDD < 8000
    print(f"\n{'='*130}")
    print(f"  RESULTATS (tries par PnL, MaxDD < $8,000)")
    print(f"{'='*130}")

    safe = [r for r in results if abs(r["max_dd"]) < 8000]
    safe.sort(key=lambda x: -x["pnl"])

    print(f"\n  {len(safe)} configs avec MaxDD < $8,000 sur {len(results)} testees\n")

    print(f"  {'OLS':>6} {'ZW':>4} {'prof':<10} {'window':<14} {'ze':>5} {'zx':>4} {'zs':>4} {'conf':>4} | "
          f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'MaxDD':>9} {'Avg$':>7} {'Strk':>4} {'Neg':>3}")
    print("  " + "-" * 120)

    for r in safe:
        print(f"  {r['ols_w']:>6} {r['zw']:>4} {r['profile']:<10} {r['window']:<14} "
              f"{r['ze']:>5.2f} {r['zx']:>4.2f} {r['zs']:>4.1f} {r['conf']:>4.0f} | "
              f"{r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>6.2f} "
              f"${r['max_dd']:>8,.0f} ${r['avg_pnl']:>6,.0f} {r['max_streak']:>4} {r['neg_years']:>3}")
        # Yearly
        years_str = "    "
        for y, p in r["yearly"].items():
            flag = "***" if p < 0 else ""
            years_str += f"{y}=${p:+,.0f}{flag}  "
        print(years_str)
        print()

    # Also show those that fail MaxDD filter
    danger = [r for r in results if abs(r["max_dd"]) >= 8000]
    if danger:
        print(f"\n  --- DANGER (MaxDD >= $8,000) ---")
        for r in sorted(danger, key=lambda x: -x["pnl"]):
            print(f"  OLS={r['ols_w']} ZW={r['zw']} {r['profile']:<10} {r['window']:<14} "
                  f"ze={r['ze']:.2f} zx={r['zx']:.2f} zs={r['zs']:.1f} conf={r['conf']:.0f} | "
                  f"{r['trades']} trd, ${r['pnl']:,.0f}, PF {r['pf']:.2f}, MaxDD ${r['max_dd']:,.0f}")


if __name__ == "__main__":
    main()
