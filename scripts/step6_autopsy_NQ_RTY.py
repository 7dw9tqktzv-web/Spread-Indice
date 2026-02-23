"""Etape 6 -- Autopsy des annees faibles NQ/RTY OLS (10 configs).

Analyse trade-by-trade des annees negatives ou faibles.
Decomposition mensuelle, regime du spread, heures d'entree.

Usage:
    python scripts/step6_autopsy_NQ_RTY.py
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
    ConfidenceConfig, compute_confidence,
    _apply_conf_filter_numba, apply_window_filter_numba,
    apply_time_stop,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

# ======================================================================
# Constants
# ======================================================================
MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

BEST_TIME_STOPS = {
    "#8_p36_96_9240_20_06:00-14:00": 0,
    "#6_p36_96_9240_20_04:00-14:00": 0,
    "#27_p36_96_7920_20_04:00-14:00": 0,
    "#23_p36_96_9240_36_04:00-14:00": 18,
    "#21_p16_80_9240_36_02:00-14:00": 18,
    "#2_p48_128_9240_32_02:00-14:00": 12,
    "#10_p16_80_3960_24_08:00-14:00": 0,
    "#12_p28_144_9240_28_08:00-12:00": 0,
    "#1_p16_80_10560_48_04:00-14:00": 0,
    "#32_p28_144_9240_48_08:00-14:00": 0,
}

METRIC_PROFILES = {
    "p16_80":  MetricsConfig(adf_window=16, hurst_window=80,  halflife_window=16, correlation_window=8),
    "p28_144": MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p36_96":  MetricsConfig(adf_window=36, hurst_window=96,  halflife_window=36, correlation_window=9),
    "p48_128": MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
    "08:00-14:00": (480, 840),
    "08:00-12:00": (480, 720),
}


def make_conf(min_conf):
    return ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00,
                            min_confidence=min_conf)


def build_signal(aligned, minutes, cfg, ts_bars):
    """Build full signal chain."""
    est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    zw = cfg["zw"]
    mu = spread.rolling(zw).mean()
    sigma = spread.rolling(zw).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    profile_cfg = METRIC_PROFILES[cfg["profile"]]
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"],
                                  profile_cfg)
    confidence = compute_confidence(metrics, make_conf(cfg["conf"])).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    if ts_bars > 0:
        sig = apply_time_stop(sig.copy(), ts_bars)

    return sig, beta, spread, zscore


def extract_trades(sig, px_a, px_b, beta, idx):
    """Extract individual trades with entry/exit details."""
    trades = []
    in_pos = False
    entry_bar = 0
    entry_dir = 0

    for i in range(1, len(sig)):
        if sig[i - 1] == 0 and sig[i] != 0:
            # Entry
            in_pos = True
            entry_bar = i
            entry_dir = int(sig[i])
        elif sig[i - 1] != 0 and sig[i] == 0 and in_pos:
            # Exit
            exit_bar = i
            # PnL calculation (simplified, matches engine logic)
            spread_entry = px_a[entry_bar] * MULT_A - beta[entry_bar] * px_b[entry_bar] * MULT_B
            spread_exit = px_a[exit_bar] * MULT_A - beta[exit_bar] * px_b[exit_bar] * MULT_B

            if entry_dir == 1:  # Long spread
                raw_pnl = spread_exit - spread_entry
            else:  # Short spread
                raw_pnl = spread_entry - spread_exit

            # Slippage + commission
            slip_cost = SLIPPAGE * (TICK_A * MULT_A + TICK_B * MULT_B) * 2
            net_pnl = raw_pnl - slip_cost - COMMISSION

            trades.append({
                "entry_dt": idx[entry_bar],
                "exit_dt": idx[exit_bar],
                "direction": "LONG" if entry_dir == 1 else "SHORT",
                "duration_bars": exit_bar - entry_bar,
                "entry_hour": idx[entry_bar].hour,
                "entry_month": idx[entry_bar].month,
                "entry_year": idx[entry_bar].year,
                "pnl": net_pnl,
                "px_a_entry": px_a[entry_bar],
                "px_b_entry": px_b[entry_bar],
                "beta_entry": beta[entry_bar],
            })
            in_pos = False

    return pd.DataFrame(trades)


def main():
    t_start = time_mod.time()

    print("=" * 130)
    print("  ETAPE 6 -- AUTOPSY DES ANNEES FAIBLES NQ/RTY OLS")
    print("=" * 130)

    # Load data
    print("\nLoading NQ/RTY data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # Load configs
    top10_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step3_top10.csv"
    configs_df = pd.read_csv(top10_path)

    configs = []
    for _, row in configs_df.iterrows():
        label = row["label"]
        configs.append({
            "label": label,
            "ols": int(row["ols"]),
            "zw": int(row["zw"]),
            "profile": row["profile"],
            "window": row["window"],
            "z_entry": row["z_entry"],
            "z_exit": row["z_exit"],
            "z_stop": row["z_stop"],
            "conf": row["conf"],
            "tier": row["tier"],
            "ts_bars": BEST_TIME_STOPS.get(label, 0),
        })

    # ==================================================================
    # SECTION A: Market regime analysis per year
    # ==================================================================
    print(f"\n{'='*130}")
    print("  SECTION A: REGIME DU MARCHE PAR ANNEE")
    print(f"{'='*130}")

    # Annual returns NQ and RTY
    close_a = aligned.df["close_a"]
    close_b = aligned.df["close_b"]

    years = sorted(set(idx.year))
    print(f"\n  {'Year':>6} {'NQ Start':>10} {'NQ End':>10} {'NQ %':>7} "
          f"{'RTY Start':>10} {'RTY End':>10} {'RTY %':>7} {'Divergence':>10}")
    print(f"  {'-'*85}")

    for y in years:
        mask = idx.year == y
        if mask.sum() < 100:
            continue
        nq_s = close_a[mask].iloc[0]
        nq_e = close_a[mask].iloc[-1]
        rty_s = close_b[mask].iloc[0]
        rty_e = close_b[mask].iloc[-1]
        nq_ret = (nq_e / nq_s - 1) * 100
        rty_ret = (rty_e / rty_s - 1) * 100
        div = nq_ret - rty_ret
        print(f"  {y:>6} {nq_s:>10,.0f} {nq_e:>10,.0f} {nq_ret:>+7.1f}% "
              f"{rty_s:>10,.0f} {rty_e:>10,.0f} {rty_ret:>+7.1f}% {div:>+10.1f}%")

    # Spread stats per year
    print(f"\n  --- Spread log-ratio stats par annee ---")
    log_ratio = np.log(close_a / close_b)
    print(f"  {'Year':>6} {'Mean':>8} {'Std':>8} {'Drift':>8} {'Range':>8}")
    print(f"  {'-'*45}")
    for y in years:
        mask = idx.year == y
        if mask.sum() < 100:
            continue
        lr = log_ratio[mask]
        drift = float(lr.iloc[-1] - lr.iloc[0])
        rng = float(lr.max() - lr.min())
        print(f"  {y:>6} {lr.mean():>8.4f} {lr.std():>8.4f} {drift:>+8.4f} {rng:>8.4f}")

    # ==================================================================
    # SECTION B: Per-config yearly PnL + monthly breakdown for weak years
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION B: DECOMPOSITION MENSUELLE DES ANNEES FAIBLES")
    print(f"{'='*130}")

    for cfg in configs:
        label = cfg["label"]
        tier = cfg["tier"]
        ts = cfg["ts_bars"]

        sig, beta, spread, zscore = build_signal(aligned, minutes, cfg, ts)

        # Extract trades
        df_trades = extract_trades(sig, px_a, px_b, beta, idx)
        if len(df_trades) == 0:
            print(f"\n  {label}: 0 trades, skip")
            continue

        # Yearly PnL
        yearly_pnl = df_trades.groupby("entry_year")["pnl"].sum()
        neg_years = [y for y, p in yearly_pnl.items() if p < 0]
        weak_years = [y for y, p in yearly_pnl.items() if p < 3000 and p >= 0]

        # Only analyze weak/negative years
        years_to_analyze = sorted(set(neg_years + weak_years))
        if not years_to_analyze:
            # No weak years, quick summary
            print(f"\n  --- {label} [{tier}]: Aucune annee faible (min=${yearly_pnl.min():,.0f}) ---")
            continue

        print(f"\n  {'='*100}")
        print(f"  {label} [{tier}] -- Annees a analyser: {years_to_analyze}")
        print(f"  {'='*100}")

        for year in years_to_analyze:
            year_trades = df_trades[df_trades["entry_year"] == year]
            total_pnl = year_trades["pnl"].sum()
            n_trades = len(year_trades)
            wins = (year_trades["pnl"] > 0).sum()
            wr = wins / n_trades * 100 if n_trades > 0 else 0

            flag = "NEGATIVE" if total_pnl < 0 else "FAIBLE"
            print(f"\n    --- {year} ({flag}: ${total_pnl:,.0f}, {n_trades} trades, WR {wr:.0f}%) ---")

            # Monthly breakdown
            monthly = year_trades.groupby("entry_month").agg(
                trades=("pnl", "count"),
                pnl=("pnl", "sum"),
                wins=("pnl", lambda x: (x > 0).sum()),
                avg_pnl=("pnl", "mean"),
                avg_dur=("duration_bars", "mean"),
            ).reset_index()

            print(f"    {'Month':>7} {'Trd':>4} {'WR%':>5} {'PnL':>9} {'AvgPnL':>8} {'AvgDur':>6}")
            print(f"    {'-'*48}")
            for _, m in monthly.iterrows():
                mwr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
                neg = " !!!" if m["pnl"] < -1000 else ""
                print(f"    {m['entry_month']:>5}   {m['trades']:>4} {mwr:>5.0f} ${m['pnl']:>8,.0f} "
                      f"${m['avg_pnl']:>7,.0f} {m['avg_dur']:>6.1f}{neg}")

            # Direction analysis
            dir_stats = year_trades.groupby("direction").agg(
                trades=("pnl", "count"),
                pnl=("pnl", "sum"),
                wins=("pnl", lambda x: (x > 0).sum()),
            ).reset_index()
            print(f"\n    Direction: ", end="")
            for _, d in dir_stats.iterrows():
                dwr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
                print(f"{d['direction']} {d['trades']}t/${d['pnl']:+,.0f} WR{dwr:.0f}%  ", end="")
            print()

            # Hour analysis
            hour_stats = year_trades.groupby("entry_hour").agg(
                trades=("pnl", "count"),
                pnl=("pnl", "sum"),
            ).reset_index()
            toxic_hours = hour_stats[hour_stats["pnl"] < -500]
            if len(toxic_hours) > 0:
                print(f"    Heures toxiques: ", end="")
                for _, h in toxic_hours.iterrows():
                    print(f"{h['entry_hour']}h (${h['pnl']:+,.0f})  ", end="")
                print()

            # Biggest losers
            worst = year_trades.nsmallest(3, "pnl")
            print(f"    Top 3 pertes:")
            for _, w in worst.iterrows():
                print(f"      {w['entry_dt'].strftime('%Y-%m-%d %H:%M')} {w['direction']} "
                      f"dur={w['duration_bars']}bars ${w['pnl']:+,.0f}")

    # ==================================================================
    # SECTION C: Cross-config pattern analysis
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION C: PATTERNS CROSS-CONFIG")
    print(f"{'='*130}")

    # Collect all negative year data
    neg_year_data = {}
    for cfg in configs:
        label = cfg["label"]
        ts = cfg["ts_bars"]
        sig, beta, spread, zscore = build_signal(aligned, minutes, cfg, ts)
        df_trades = extract_trades(sig, px_a, px_b, beta, idx)
        if len(df_trades) == 0:
            continue
        yearly = df_trades.groupby("entry_year")["pnl"].sum()
        for y, p in yearly.items():
            if p < 0:
                if y not in neg_year_data:
                    neg_year_data[y] = []
                neg_year_data[y].append((label, p))

    print(f"\n  Annees negatives par config:")
    for y in sorted(neg_year_data.keys()):
        configs_neg = neg_year_data[y]
        print(f"\n  {y}: {len(configs_neg)} configs negatives")
        for lbl, pnl in sorted(configs_neg, key=lambda x: x[1]):
            print(f"    {lbl:<48} ${pnl:>+8,.0f}")

    # Count configs negative per year
    print(f"\n  Resume:")
    print(f"  {'Year':>6} {'Neg configs':>12} {'Avg loss':>10}")
    print(f"  {'-'*35}")
    for y in sorted(neg_year_data.keys()):
        losses = [p for _, p in neg_year_data[y]]
        print(f"  {y:>6} {len(losses):>12} ${np.mean(losses):>9,.0f}")

    elapsed = time_mod.time() - t_start
    print(f"\n  Etape 6 complete en {elapsed:.0f}s")


if __name__ == "__main__":
    main()
