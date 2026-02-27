"""Deep analysis OLS_Balanced NQ/RTY — trade-level diagnostics.

Usage:
    python scripts/deep_analysis_ols_balanced.py
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
    ConfidenceConfig,
    _apply_conf_filter_numba,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

# Config OLS_Balanced
OLS_WINDOW = 7920
ZSCORE_WINDOW = 48
PROFILE = "court"
WINDOW = "08:00-14:00"
Z_ENTRY = 3.50
Z_EXIT = 0.50
Z_STOP = 6.00
MIN_CONF = 75.0

MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
ENTRY_START = 480  # 08:00
ENTRY_END = 840    # 14:00

METRIC_PROFILES = {
    "court": MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
}


def main():
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # Hedge ratio + z-score
    est = create_estimator("ols_rolling", window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    hr = est.estimate(aligned)
    spread = hr.spread
    beta = hr.beta.values

    mu = spread.rolling(ZSCORE_WINDOW).mean()
    sigma = spread.rolling(ZSCORE_WINDOW).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

    # Metrics + confidence
    profile_cfg = METRIC_PROFILES[PROFILE]
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
    confidence = compute_confidence(metrics, ConfidenceConfig()).values

    # Signals
    raw = generate_signals_numba(zscore, Z_ENTRY, Z_EXIT, Z_STOP)
    sig = _apply_conf_filter_numba(raw, confidence, MIN_CONF)
    sig = apply_window_filter_numba(sig, minutes, ENTRY_START, ENTRY_END, FLAT_MIN)

    # Backtest
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )

    n_trades = bt["trades"]
    trade_pnls = bt["trade_pnls"]
    entry_bars = bt["trade_entry_bars"]
    exit_bars = bt["trade_exit_bars"]
    trade_sides = bt.get("trade_sides", np.zeros(n_trades))

    entry_dates = idx[entry_bars]
    exit_dates = idx[exit_bars]
    durations = exit_bars - entry_bars  # in bars (5min each)

    # ================================================================
    # BUILD TRADE DATAFRAME
    # ================================================================
    trades = pd.DataFrame({
        "entry_date": entry_dates,
        "exit_date": exit_dates,
        "entry_bar": entry_bars,
        "exit_bar": exit_bars,
        "pnl": trade_pnls,
        "side": trade_sides,
        "duration_bars": durations,
        "duration_min": durations * 5,
        "entry_hour": entry_dates.hour,
        "entry_minute": entry_dates.minute,
        "exit_hour": exit_dates.hour,
        "exit_minute": exit_dates.minute,
        "exit_minutes_ct": exit_dates.hour * 60 + exit_dates.minute,
        "entry_zscore": zscore[entry_bars],
        "exit_zscore": zscore[exit_bars],
    })

    # Classify exit type
    def classify_exit(row):
        exit_min = row["exit_minutes_ct"]
        ez = abs(row["exit_zscore"])
        # Flat = exit at 15:30 (930 min) or within 5 min
        if exit_min >= 925:
            return "FLAT_EOD"
        # Z-stop: exit z-score near or beyond z_stop
        if ez >= Z_STOP - 0.5:
            return "Z_STOP"
        # Z-exit: z-score reverted to near z_exit
        if ez <= Z_EXIT + 0.5:
            return "Z_EXIT"
        # Ambiguous
        return "OTHER"

    trades["exit_type"] = trades.apply(classify_exit, axis=1)

    # Win/Loss
    trades["result"] = np.where(trades["pnl"] > 0, "WIN", "LOSS")

    winners = trades[trades["pnl"] > 0]
    losers = trades[trades["pnl"] <= 0]

    # ================================================================
    # GLOBAL OVERVIEW
    # ================================================================
    print("=" * 100)
    print("  OLS_BALANCED NQ/RTY — DEEP TRADE ANALYSIS")
    print("=" * 100)

    print(f"\n  Total trades: {n_trades}")
    print(f"  Winners: {len(winners)} ({len(winners)/n_trades*100:.1f}%)")
    print(f"  Losers:  {len(losers)} ({len(losers)/n_trades*100:.1f}%)")
    print(f"  Total PnL: ${trade_pnls.sum():,.0f}")
    print(f"  Avg PnL/trade: ${trade_pnls.mean():,.0f}")

    # ================================================================
    # DURATION ANALYSIS
    # ================================================================
    print(f"\n{'='*100}")
    print("  DUREE DES TRADES (en minutes)")
    print(f"{'='*100}")

    print(f"\n  {'':>12} {'Count':>6} {'Mean':>8} {'Median':>8} {'Min':>8} {'Max':>8} {'Std':>8}")
    print("  " + "-" * 60)
    for label, subset in [("ALL", trades), ("WINNERS", winners), ("LOSERS", losers)]:
        d = subset["duration_min"]
        print(f"  {label:>12} {len(d):>6} {d.mean():>8.0f} {d.median():>8.0f} "
              f"{d.min():>8.0f} {d.max():>8.0f} {d.std():>8.0f}")

    # Duration distribution
    print("\n  Distribution des durees (tous trades):")
    bins = [0, 30, 60, 120, 240, 480, 1000, 9999]
    labels_d = ["<30min", "30-60m", "1-2h", "2-4h", "4-8h", "8h-16h", ">16h"]
    trades["dur_bin"] = pd.cut(trades["duration_min"], bins=bins, labels=labels_d)
    dur_dist = trades.groupby("dur_bin", observed=True).agg(
        count=("pnl", "size"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        wr=("pnl", lambda x: (x > 0).sum() / max(len(x), 1) * 100),
    )
    print(f"  {'Duree':<10} {'Count':>6} {'AvgPnL':>9} {'TotalPnL':>10} {'WR':>6}")
    print("  " + "-" * 45)
    for b, row in dur_dist.iterrows():
        print(f"  {str(b):<10} {row['count']:>6.0f} ${row['avg_pnl']:>8,.0f} "
              f"${row['total_pnl']:>9,.0f} {row['wr']:>5.1f}%")

    # ================================================================
    # EXIT TYPE ANALYSIS
    # ================================================================
    print(f"\n{'='*100}")
    print("  TYPE DE SORTIE")
    print(f"{'='*100}")

    exit_stats = trades.groupby("exit_type").agg(
        count=("pnl", "size"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
        wr=("pnl", lambda x: (x > 0).sum() / max(len(x), 1) * 100),
        avg_dur=("duration_min", "mean"),
    )
    exit_stats["pct"] = exit_stats["count"] / n_trades * 100

    print(f"\n  {'Type':<12} {'Count':>6} {'%':>6} {'AvgPnL':>9} {'TotalPnL':>10} {'WR':>6} {'AvgDur':>8}")
    print("  " + "-" * 65)
    for t, row in exit_stats.iterrows():
        print(f"  {t:<12} {row['count']:>6.0f} {row['pct']:>5.1f}% ${row['avg_pnl']:>8,.0f} "
              f"${row['total_pnl']:>9,.0f} {row['wr']:>5.1f}% {row['avg_dur']:>7.0f}m")

    # ================================================================
    # Z-SCORE AT EXIT
    # ================================================================
    print(f"\n{'='*100}")
    print("  Z-SCORE A L'ENTREE ET SORTIE")
    print(f"{'='*100}")

    print(f"\n  {'':>12} {'Entry Z mean':>14} {'Entry Z std':>12} {'Exit Z mean':>13} {'Exit Z std':>12}")
    print("  " + "-" * 60)
    for label, subset in [("ALL", trades), ("WINNERS", winners), ("LOSERS", losers)]:
        ez = subset["entry_zscore"].abs()
        xz = subset["exit_zscore"].abs()
        print(f"  {label:>12} {ez.mean():>14.3f} {ez.std():>12.3f} {xz.mean():>13.3f} {xz.std():>12.3f}")

    # ================================================================
    # SIDE ANALYSIS (LONG vs SHORT)
    # ================================================================
    print(f"\n{'='*100}")
    print("  LONG vs SHORT")
    print(f"{'='*100}")

    for side_val, side_name in [(1, "LONG"), (-1, "SHORT")]:
        sub = trades[trades["side"] == side_val]
        if len(sub) == 0:
            print(f"\n  {side_name}: 0 trades")
            continue
        w = sub[sub["pnl"] > 0]
        l = sub[sub["pnl"] <= 0]
        print(f"\n  {side_name}: {len(sub)} trades, WR {len(w)/len(sub)*100:.1f}%, "
              f"PnL ${sub['pnl'].sum():,.0f}, AvgPnL ${sub['pnl'].mean():,.0f}")
        print(f"    Winners: {len(w)}, avg ${w['pnl'].mean():,.0f}" if len(w) > 0 else "    Winners: 0")
        print(f"    Losers:  {len(l)}, avg ${l['pnl'].mean():,.0f}" if len(l) > 0 else "    Losers: 0")

    # ================================================================
    # LOSING TRADES DEEP DIVE
    # ================================================================
    print(f"\n{'='*100}")
    print("  AUTOPSIE DES TRADES PERDANTS")
    print(f"{'='*100}")

    print(f"\n  {len(losers)} trades perdants, perte totale: ${losers['pnl'].sum():,.0f}")
    print(f"  Pire trade: ${losers['pnl'].min():,.0f}")
    print(f"  Perte moyenne: ${losers['pnl'].mean():,.0f}")

    # Exit type of losers
    print("\n  Repartition des pertes par type de sortie:")
    loser_exit = losers.groupby("exit_type").agg(
        count=("pnl", "size"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        avg_dur=("duration_min", "mean"),
    )
    loser_exit["pct_loss"] = loser_exit["total_pnl"] / losers["pnl"].sum() * 100

    print(f"  {'Type':<12} {'Count':>6} {'TotalLoss':>10} {'%Loss':>7} {'AvgLoss':>9} {'AvgDur':>8}")
    print("  " + "-" * 55)
    for t, row in loser_exit.iterrows():
        print(f"  {t:<12} {row['count']:>6.0f} ${row['total_pnl']:>9,.0f} "
              f"{row['pct_loss']:>6.1f}% ${row['avg_pnl']:>8,.0f} {row['avg_dur']:>7.0f}m")

    # Side of losers
    print("\n  Pertes par direction:")
    for side_val, side_name in [(1, "LONG"), (-1, "SHORT")]:
        sub = losers[losers["side"] == side_val]
        if len(sub) > 0:
            print(f"    {side_name}: {len(sub)} trades, ${sub['pnl'].sum():,.0f} "
                  f"(avg ${sub['pnl'].mean():,.0f})")

    # Year of losers
    losers_copy = losers.copy()
    losers_copy["year"] = losers_copy["entry_date"].dt.year
    print("\n  Pertes par annee:")
    for y, grp in losers_copy.groupby("year"):
        print(f"    {y}: {len(grp)} trades, ${grp['pnl'].sum():,.0f} (avg ${grp['pnl'].mean():,.0f})")

    # Hour of losers
    print("\n  Pertes par heure d'entree:")
    for h, grp in losers.groupby("entry_hour"):
        print(f"    {h:02d}h: {len(grp)} trades, ${grp['pnl'].sum():,.0f}")

    # List all losing trades
    print("\n  Liste complete des trades perdants:")
    print(f"  {'#':>3} {'Entry':>18} {'Exit':>18} {'Side':>5} {'Dur':>6} {'PnL':>9} {'ExitType':<10} "
          f"{'EntryZ':>7} {'ExitZ':>7}")
    print("  " + "-" * 100)
    for i, (_, r) in enumerate(losers.sort_values("pnl").iterrows(), 1):
        side_str = "LONG" if r["side"] == 1 else "SHORT"
        print(f"  {i:>3} {str(r['entry_date'])[:16]:>18} {str(r['exit_date'])[:16]:>18} "
              f"{side_str:>5} {r['duration_min']:>5.0f}m ${r['pnl']:>8,.0f} {r['exit_type']:<10} "
              f"{r['entry_zscore']:>+7.3f} {r['exit_zscore']:>+7.3f}")


if __name__ == "__main__":
    main()
