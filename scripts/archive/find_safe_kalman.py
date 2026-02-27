"""Find Kalman configs with MaxDD < $4,500 for propfirm safety.

Runs full vectorized backtest on promising candidates to compute MaxDD,
max losing streak, max consecutive DD dollars, and daily loss stats.

Usage:
    python scripts/find_safe_kalman.py
"""

import sys
import time as time_mod
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
    ConfidenceConfig,
    _apply_conf_filter_numba,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

# ── Constants ──
MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE, COMMISSION, CAPITAL = 1, 2.50, 100_000.0
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930

PROFILES_MAP = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=66, hurst_window=132, halflife_window=66, correlation_window=66),
    "moyen":      MetricsConfig(adf_window=264, hurst_window=528, halflife_window=264, correlation_window=264),
}
WINDOWS_MAP = {
    "03:00-12:00": (180, 720), "04:00-12:00": (240, 720),
    "04:00-13:00": (240, 780), "04:00-14:00": (240, 840),
    "05:00-12:00": (300, 720),
}

# Cache Kalman results per (alpha, profil) to avoid recomputation
_kalman_cache = {}


def get_kalman_data(aligned, alpha, profil):
    """Get or compute Kalman hedge + metrics + confidence for (alpha, profil)."""
    key = (alpha, profil)
    if key not in _kalman_cache:
        est = create_estimator("kalman", alpha_ratio=alpha)
        hr = est.estimate(aligned)
        beta = hr.beta.values
        zscore = np.ascontiguousarray(hr.zscore.values, dtype=np.float64)

        mcfg = PROFILES_MAP[profil]
        metrics = compute_all_metrics(
            hr.spread, aligned.df["close_a"], aligned.df["close_b"], mcfg
        )
        confidence = compute_confidence(metrics, CONF_CFG).values
        _kalman_cache[key] = (beta, zscore, confidence)
    return _kalman_cache[key]


def run_full_detailed(aligned, px_a, px_b, idx, minutes, row, years):
    """Run full backtest, return detailed metrics including DD analysis."""
    alpha = row["alpha_ratio"]
    profil = row["profil"]
    window = row["window"]

    beta, zscore, confidence = get_kalman_data(aligned, alpha, profil)

    raw = generate_signals_numba(zscore, row["z_entry"], row["z_exit"], row["z_stop"])
    sig = _apply_conf_filter_numba(raw, confidence, row["min_confidence"])
    es, ee = WINDOWS_MAP[window]
    sig = apply_window_filter_numba(sig, minutes, es, ee, FLAT_MIN)

    bt = run_backtest_vectorized(px_a, px_b, sig, beta, MULT_A, MULT_B, TICK_A, TICK_B,
                                  SLIPPAGE, COMMISSION, CAPITAL)

    n = bt["trades"]
    if n == 0:
        return None

    pnls = bt["trade_pnls"]
    equity = bt["equity"]
    entry_bars = bt["trade_entry_bars"]
    exit_bars = bt["trade_exit_bars"]

    # ── MaxDD from equity curve ──
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    max_dd = drawdowns.min()

    # ── Find the DD period (peak to trough) ──
    trough_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(equity[:trough_idx + 1]) if trough_idx > 0 else 0

    # ── Max losing streak (consecutive losing trades) ──
    streak = 0
    max_streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # ── Max consecutive DD in dollars (sum of consecutive losses) ──
    consec_dd = 0
    max_consec_dd = 0
    for p in pnls:
        if p < 0:
            consec_dd += p
            max_consec_dd = min(max_consec_dd, consec_dd)
        else:
            consec_dd = 0

    # ── Daily PnL analysis (worst day, worst week) ──
    entry_dates = idx[entry_bars].date
    trade_df = pd.DataFrame({"date": entry_dates, "pnl": pnls})
    daily_pnl = trade_df.groupby("date")["pnl"].sum()
    worst_day = daily_pnl.min() if len(daily_pnl) > 0 else 0
    best_day = daily_pnl.max() if len(daily_pnl) > 0 else 0

    # ── Weekly PnL ──
    entry_weeks = pd.Series(idx[entry_bars]).dt.isocalendar().apply(
        lambda x: f"{x.year}-W{x.week:02d}", axis=1
    ).values
    trade_df["week"] = entry_weeks
    weekly_pnl = trade_df.groupby("week")["pnl"].sum()
    worst_week = weekly_pnl.min() if len(weekly_pnl) > 0 else 0

    # ── Max DD duration in days ──
    if trough_idx > peak_idx:
        dd_duration_days = (idx[trough_idx] - idx[peak_idx]).days
    else:
        dd_duration_days = 0

    # ── Recovery: bars from trough to recovery ──
    recovery_idx = None
    for i in range(trough_idx, len(equity)):
        if equity[i] >= running_max[trough_idx]:
            recovery_idx = i
            break
    if recovery_idx and trough_idx > 0:
        recovery_days = (idx[recovery_idx] - idx[trough_idx]).days
    else:
        recovery_days = -1  # never recovered

    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(n)) if n > 1 and pnls.std() > 0 else 0
    calmar = float(bt["pnl"] / abs(max_dd)) if max_dd < 0 else 0

    return {
        "trades": n, "wr": bt["win_rate"], "pnl": bt["pnl"], "pf": bt["profit_factor"],
        "avg_pnl": bt["avg_pnl_trade"], "avg_dur": bt["avg_duration_bars"],
        "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
        "max_streak": max_streak, "max_consec_dd": max_consec_dd,
        "worst_day": worst_day, "best_day": best_day, "worst_week": worst_week,
        "max_loss": float(pnls.min()), "max_win": float(pnls.max()),
        "dd_duration_days": dd_duration_days, "recovery_days": recovery_days,
        "trades_yr": n / years, "pnl_yr": bt["pnl"] / years,
    }


def main():
    t_start = time_mod.time()

    print("Loading NQ_YM data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years:.1f} years\n")

    # ── Load grid CSV ──
    df = pd.read_csv("output/NQ_YM/grid_kalman_v3_filtered.csv")
    print(f"Loaded {len(df):,} filtered configs")

    # ── Pre-filter: configs most likely to have low MaxDD ──
    # Higher PF + higher WR = less deep drawdowns typically
    # Also higher z_entry = fewer trades = lower DD
    candidates = df[
        (df.profit_factor >= 1.3) &
        (df.win_rate >= 68) &
        (df.trades >= 20) &
        (df.trades <= 200) &  # high volume = high DD
        (df.avg_pnl_trade >= 50)
    ].copy()
    print(f"Pre-filtered candidates: {len(candidates):,}")

    # ── Further reduce: sample diverse configs ──
    # Sort by PF * WR to get highest quality first
    candidates["quality"] = candidates["profit_factor"] * candidates["win_rate"] / 100
    candidates = candidates.sort_values("quality", ascending=False)

    # Deduplicate: keep best per (alpha, profil, window, z_entry_bin, z_exit_bin)
    candidates["ze_bin"] = (candidates["z_entry"] * 4).round() / 4  # bin to 0.25
    candidates["zx_bin"] = (candidates["z_exit"] * 4).round() / 4
    deduped = candidates.groupby(
        ["alpha_ratio", "profil", "window", "ze_bin", "zx_bin"]
    ).first().reset_index()
    print(f"Deduped candidates to test: {len(deduped):,}")

    # ── Run full backtest on each ──
    print(f"\nRunning {len(deduped):,} full backtests...")
    results = []
    for i, (_, row) in enumerate(deduped.iterrows()):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(deduped)}]...")
        r = run_full_detailed(aligned, px_a, px_b, idx, minutes, row, years)
        if r is not None:
            r["alpha_ratio"] = row["alpha_ratio"]
            r["profil"] = row["profil"]
            r["window"] = row["window"]
            r["z_entry"] = row["z_entry"]
            r["z_exit"] = row["z_exit"]
            r["z_stop"] = row["z_stop"]
            r["min_confidence"] = row["min_confidence"]
            results.append(r)

    rdf = pd.DataFrame(results)
    print(f"\nTotal results: {len(rdf):,}")

    # ── Filter: MaxDD < $4,500 ──
    safe = rdf[rdf["max_dd"].abs() < 4500].copy()
    print(f"MaxDD < $4,500: {len(safe):,} configs")

    # ── Also show MaxDD < $6,000 for borderline ──
    borderline = rdf[(rdf["max_dd"].abs() >= 4500) & (rdf["max_dd"].abs() < 6000)].copy()
    print(f"MaxDD $4,500-$6,000 (borderline): {len(borderline):,} configs")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: SAFE CONFIGS (MaxDD < $4,500)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*160}")
    print(" CONFIGS PROPFIRM-SAFE : MaxDD < $4,500 (Topstep 150k trailing DD)")
    print(f"{'='*160}")

    if len(safe) > 0:
        safe = safe.sort_values("pnl", ascending=False)

        # Top 20 by PnL
        print("\n  Top 20 by PnL (MaxDD < $4,500):")
        print(f"  {'#':<3} {'Alpha':>8} {'Prof':<11} {'Window':<12} "
              f"{'Zent':>5} {'Zex':>5} {'Zst':>5} {'Conf':>4} | "
              f"{'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
              f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'WDay':>7} {'WWk':>8} "
              f"{'DDdur':>5} {'Recov':>5} {'Trd/y':>5}")
        print(f"  {'-'*158}")

        for i, (_, r) in enumerate(safe.head(20).iterrows()):
            recov = f"{r['recovery_days']:>4}d" if r['recovery_days'] >= 0 else " n/r"
            print(f"  {i+1:<3} {r['alpha_ratio']:>8.1e} {r['profil']:<11} {r['window']:<12} "
                  f"{r['z_entry']:>5.3f} {r['z_exit']:>5.3f} {r['z_stop']:>5.3f} {r['min_confidence']:>4.0f} | "
                  f"{r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} {r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} "
                  f"${r['max_dd']:>6,.0f} {r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} "
                  f"${r['worst_day']:>6,.0f} ${r['worst_week']:>7,.0f} "
                  f"{r['dd_duration_days']:>4}d {recov} {r['trades_yr']:>5.1f}")

        # Top 10 by PF (trades >= 30)
        safe_pf = safe[safe["trades"] >= 30].sort_values("pf", ascending=False)
        print("\n  Top 10 by PF (MaxDD < $4,500, trades >= 30):")
        print(f"  {'#':<3} {'Alpha':>8} {'Prof':<11} {'Window':<12} "
              f"{'Zent':>5} {'Zex':>5} {'Zst':>5} {'Conf':>4} | "
              f"{'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
              f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'WDay':>7} {'MaxLos':>7}")
        print(f"  {'-'*140}")

        for i, (_, r) in enumerate(safe_pf.head(10).iterrows()):
            print(f"  {i+1:<3} {r['alpha_ratio']:>8.1e} {r['profil']:<11} {r['window']:<12} "
                  f"{r['z_entry']:>5.3f} {r['z_exit']:>5.3f} {r['z_stop']:>5.3f} {r['min_confidence']:>4.0f} | "
                  f"{r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} {r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} "
                  f"${r['max_dd']:>6,.0f} {r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} "
                  f"${r['worst_day']:>6,.0f} ${r['max_loss']:>6,.0f}")

        # Best balanced: trades >= 50, PF >= 1.5
        safe_bal = safe[(safe["trades"] >= 50) & (safe["pf"] >= 1.5)].sort_values("pnl", ascending=False)
        print("\n  Top 10 BALANCED (MaxDD < $4,500, trades >= 50, PF >= 1.5):")
        if len(safe_bal) > 0:
            print(f"  {'#':<3} {'Alpha':>8} {'Prof':<11} {'Window':<12} "
                  f"{'Zent':>5} {'Zex':>5} {'Zst':>5} {'Conf':>4} | "
                  f"{'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
                  f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'WDay':>7} {'MaxLos':>7} {'Trd/y':>5}")
            print(f"  {'-'*148}")
            for i, (_, r) in enumerate(safe_bal.head(10).iterrows()):
                print(f"  {i+1:<3} {r['alpha_ratio']:>8.1e} {r['profil']:<11} {r['window']:<12} "
                      f"{r['z_entry']:>5.3f} {r['z_exit']:>5.3f} {r['z_stop']:>5.3f} {r['min_confidence']:>4.0f} | "
                      f"{r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} {r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} "
                      f"${r['max_dd']:>6,.0f} {r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} "
                      f"${r['worst_day']:>6,.0f} ${r['max_loss']:>6,.0f} {r['trades_yr']:>5.1f}")
        else:
            print("  Aucune config trouvee.")

    else:
        print("\n  AUCUNE CONFIG avec MaxDD < $4,500 trouvee!")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: BORDERLINE (MaxDD $4,500-$6,000)
    # ══════════════════════════════════════════════════════════════════
    if len(borderline) > 0:
        print(f"\n\n{'='*160}")
        print(" CONFIGS BORDERLINE : MaxDD $4,500 - $6,000")
        print(f"{'='*160}")

        borderline = borderline.sort_values("pnl", ascending=False)
        print("\n  Top 10 by PnL (borderline MaxDD):")
        print(f"  {'#':<3} {'Alpha':>8} {'Prof':<11} {'Window':<12} "
              f"{'Zent':>5} {'Zex':>5} {'Zst':>5} {'Conf':>4} | "
              f"{'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
              f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'WDay':>7} {'MaxLos':>7} {'Trd/y':>5}")
        print(f"  {'-'*148}")

        for i, (_, r) in enumerate(borderline.head(10).iterrows()):
            print(f"  {i+1:<3} {r['alpha_ratio']:>8.1e} {r['profil']:<11} {r['window']:<12} "
                  f"{r['z_entry']:>5.3f} {r['z_exit']:>5.3f} {r['z_stop']:>5.3f} {r['min_confidence']:>4.0f} | "
                  f"{r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} {r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} "
                  f"${r['max_dd']:>6,.0f} {r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} "
                  f"${r['worst_day']:>6,.0f} ${r['max_loss']:>6,.0f} {r['trades_yr']:>5.1f}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: Distribution MaxDD
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(" DISTRIBUTION MaxDD (toutes configs testees)")
    print(f"{'='*80}")

    bins = [0, 2000, 3000, 4000, 4500, 5000, 6000, 8000, 10000, 15000, 50000]
    labels = ["<$2k", "$2-3k", "$3-4k", "$4-4.5k", "$4.5-5k", "$5-6k", "$6-8k", "$8-10k", "$10-15k", ">$15k"]
    rdf["dd_bin"] = pd.cut(rdf["max_dd"].abs(), bins=bins, labels=labels, right=False)
    dist = rdf["dd_bin"].value_counts().sort_index()
    total = len(rdf)
    cumul = 0
    print(f"\n  {'MaxDD Range':<12} {'Count':>8} {'%':>6} {'Cumul%':>7}")
    print(f"  {'-'*35}")
    for label in labels:
        c = dist.get(label, 0)
        cumul += c
        print(f"  {label:<12} {c:>8} {c/total*100:>5.1f}% {cumul/total*100:>6.1f}%")

    elapsed = time_mod.time() - t_start
    print(f"\n\nComplete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
