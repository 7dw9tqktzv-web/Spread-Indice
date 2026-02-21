"""Batch 1-3: Volume search for NQ_YM.

Batch 1: z_entry granulaire + fenetres de trading + stops
Batch 2: Analyse des trades perdants (exit reason, duree, heure, z-score)
Batch 3: Calibration 3min avec params propres (pas de scaling lineaire)

Usage:
    python scripts/batch_volume_search.py --batch1
    python scripts/batch_volume_search.py --batch2
    python scripts/batch_volume_search.py --batch3
    python scripts/batch_volume_search.py --all
"""

import argparse
import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.generator import SignalConfig, SignalGenerator
from src.signals.filters import ConfidenceConfig, compute_confidence
from src.backtest.engine import run_backtest_vectorized

# ======================================================================
# Shared constants
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0   # NQ, YM
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0

METRICS_TRES_COURT = MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6)
CONF_CFG = ConfidenceConfig()


# ======================================================================
# Helpers
# ======================================================================

def load_nq_ym(tf="5min"):
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    return load_aligned_pair_cache(pair, tf)


def compute_pipeline_arrays(aligned, ols_window, zscore_window, metrics_cfg):
    """Compute hedge, metrics, confidence, zscore arrays once."""
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index

    est = create_estimator("ols_rolling", window=ols_window, zscore_window=zscore_window)
    hr = est.estimate(aligned)
    beta = hr.beta.values

    metrics = compute_all_metrics(hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg)
    confidence = compute_confidence(metrics, CONF_CFG).values

    zscore = hr.zscore.values

    return px_a, px_b, idx, beta, confidence, zscore, metrics, hr


def apply_signal_filter(raw_signals, confidence, min_conf, idx, entry_start_min, entry_end_min, flat_min):
    """Apply confidence filter + entry window + flat EOD.

    entry_start_min/entry_end_min: minutes from midnight for entry window.
    flat_min: minutes from midnight for force flat.
    """
    n = len(raw_signals)
    sig = raw_signals.copy()
    minutes = idx.hour * 60 + idx.minute

    # Confidence filter (entry only)
    prev = 0
    for t in range(n):
        curr = sig[t]
        if (prev == 0) and (curr != 0) and (confidence[t] < min_conf):
            sig[t] = 0
        prev = sig[t]

    # Entry window + flat EOD
    prev = 0
    for t in range(n):
        m = minutes[t]
        curr = sig[t]

        # Force flat at/after flat_time or before entry_start
        if m >= flat_min or m < entry_start_min:
            sig[t] = 0
            prev = 0
            continue

        # Block new entries outside [entry_start, entry_end)
        if not (entry_start_min <= m < entry_end_min):
            is_entry = (prev == 0) and (curr != 0)
            if is_entry:
                sig[t] = 0

        prev = sig[t]

    return sig


def run_bt(px_a, px_b, sig, beta):
    return run_backtest_vectorized(
        px_a, px_b, sig, beta,
        mult_a=MULT_A, mult_b=MULT_B, tick_a=TICK_A, tick_b=TICK_B,
        slippage_ticks=SLIPPAGE, commission=COMMISSION, initial_capital=INITIAL_CAPITAL,
    )


def compute_dd(eq):
    running_max = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (running_max - eq) / running_max * 100
    return float(np.nan_to_num(dd).max())


def compute_sharpe(eq, bars_per_day=264):
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.diff(eq) / eq[:-1]
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    if len(ret) < 2 or np.std(ret) == 0:
        return 0.0
    return float((np.mean(ret) / np.std(ret)) * np.sqrt(bars_per_day * 252))


# ======================================================================
# BATCH 1: z_entry granulaire + fenetres + stops
# ======================================================================

def batch1(aligned):
    print("=" * 130)
    print(" BATCH 1 - GRILLE VOLUME: z_entry granulaire + fenetres + stops")
    print(" NQ_YM 5min | OLS=2640 | ZW=36 | profil tres_court | flat 15:30 CT")
    print("=" * 130)

    px_a, px_b, idx, beta, confidence, zscore, _, _ = compute_pipeline_arrays(
        aligned, ols_window=2640, zscore_window=36, metrics_cfg=METRICS_TRES_COURT
    )
    n = len(px_a)

    # Grid
    z_entries = [2.6, 2.7, 2.75, 2.8, 2.9, 3.0, 3.5]
    z_exits = [0.50, 1.00, 1.25, 1.50]
    z_stops = [3.5, 4.0, 4.5]
    confs = [67.0, 70.0]
    windows = [
        ("04:00-14:00", 240, 840),
        ("02:00-14:00", 120, 840),
        ("02:00-15:00", 120, 900),
    ]
    flat_min = 15 * 60 + 30  # 15:30 CT

    total = len(z_entries) * len(z_exits) * len(z_stops) * len(confs) * len(windows)
    print(f" Combinaisons: {total}")

    header = (f" {'Window':<14} {'Zent':>5} {'Zex':>4} {'Zst':>4} {'Conf':>4} "
              f"{'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Avg':>7} {'Shrp':>5} {'DD%':>5} {'Trd/An':>6}")
    print(header)
    print(" " + "-" * 120)

    results = []
    done = 0

    for win_name, entry_start, entry_end in windows:
        for z_entry in z_entries:
            for z_exit in z_exits:
                for z_stop in z_stops:
                    # Skip invalid combos
                    if z_exit >= z_entry or z_stop <= z_entry:
                        done += len(confs)
                        continue

                    gen = SignalGenerator(config=SignalConfig(z_entry=z_entry, z_exit=z_exit, z_stop=z_stop))
                    raw_signals = gen.generate(pd.Series(zscore, index=idx)).values

                    for min_conf in confs:
                        sig = apply_signal_filter(
                            raw_signals, confidence, min_conf, idx,
                            entry_start, entry_end, flat_min,
                        )
                        bt = run_bt(px_a, px_b, sig, beta)

                        done += 1
                        num = bt["trades"]
                        if num < 10:
                            continue

                        eq = bt["equity"]
                        dd = compute_dd(eq)
                        sharpe = compute_sharpe(eq)
                        years = (idx[-1] - idx[0]).days / 365.25
                        trd_an = num / years if years > 0 else 0

                        print(
                            f" {win_name:<14} {z_entry:>5.2f} {z_exit:>4.2f} {z_stop:>4.1f} {min_conf:>4.0f} "
                            f"{num:>5} {bt['win_rate']:>5.1f}% ${bt['pnl']:>9,.0f} {bt['profit_factor']:>5.2f} "
                            f"${bt['avg_pnl_trade']:>6,.0f} {sharpe:>5.2f} {dd:>5.1f}% {trd_an:>6.1f}"
                        )

                        results.append({
                            "window": win_name, "z_entry": z_entry, "z_exit": z_exit,
                            "z_stop": z_stop, "conf": min_conf, "trades": num,
                            "win_rate": bt["win_rate"], "pnl": bt["pnl"],
                            "pf": bt["profit_factor"], "avg_pnl": bt["avg_pnl_trade"],
                            "sharpe": sharpe, "dd_pct": dd, "trd_an": trd_an,
                        })

    # Top 20 by composite score
    print(f"\n {'='*130}")
    print(" TOP 20 PAR SCORE COMPOSITE (PF * Sharpe * sqrt(trades))")
    print(f" {'='*130}")
    if results:
        df = pd.DataFrame(results)
        df = df[df["pf"] > 1.0].copy()
        df["score"] = df["pf"] * df["sharpe"].clip(lower=0) * np.sqrt(df["trades"])
        df = df.sort_values("score", ascending=False).head(20)

        print(f" {'#':<3} {'Window':<14} {'Zent':>5} {'Zex':>4} {'Zst':>4} {'Conf':>4} "
              f"{'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Shrp':>5} {'DD%':>5} {'Trd/An':>6} {'Score':>6}")
        print(" " + "-" * 120)
        for i, (_, r) in enumerate(df.iterrows()):
            print(
                f" {i+1:<3} {r['window']:<14} {r['z_entry']:>5.2f} {r['z_exit']:>4.2f} {r['z_stop']:>4.1f} {r['conf']:>4.0f} "
                f"{r['trades']:>5} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>5.2f} "
                f"{r['sharpe']:>5.2f} {r['dd_pct']:>5.1f}% {r['trd_an']:>6.1f} {r['score']:>6.1f}"
            )

    print(f"\n {done} combinaisons testees, {len(results)} avec >= 10 trades")


# ======================================================================
# BATCH 2: Analyse des trades perdants
# ======================================================================

def batch2(aligned):
    print("\n" + "=" * 130)
    print(" BATCH 2 - ANALYSE DES TRADES PERDANTS")
    print(" Config ref: e=3.0 x=1.25 c=70 | flat 15:30 CT | entry [04:00-14:00)")
    print("=" * 130)

    px_a, px_b, idx, beta, confidence, zscore, metrics, hr = compute_pipeline_arrays(
        aligned, ols_window=2640, zscore_window=36, metrics_cfg=METRICS_TRES_COURT
    )
    n = len(px_a)

    # Generate signals with reference config
    gen = SignalGenerator(config=SignalConfig(z_entry=3.0, z_exit=1.25, z_stop=4.0))
    raw_signals = gen.generate(pd.Series(zscore, index=idx)).values

    sig = apply_signal_filter(
        raw_signals, confidence, 70.0, idx,
        entry_start_min=240, entry_end_min=840, flat_min=930,
    )
    bt = run_bt(px_a, px_b, sig, beta)

    num = bt["trades"]
    if num == 0:
        print(" 0 trades.")
        return

    te = bt["trade_entry_bars"]
    tx = bt["trade_exit_bars"]
    pnls = bt["trade_pnls"]
    sides = bt["trade_sides"]

    entry_times = idx[te]
    exit_times = idx[tx]
    durations = tx - te

    print(f"\n Total: {num} trades, Winners: {(pnls > 0).sum()}, Losers: {(pnls <= 0).sum()}")

    # --- Classify exit reason ---
    print("\n --- RAISON DE SORTIE ---")

    exit_reasons = []
    for i in range(num):
        eb, xb = te[i], tx[i]
        side = sides[i]
        z_at_exit = zscore[xb] if xb < len(zscore) else 0

        # Check z-score at exit
        exit_time = exit_times[i].time()
        duration_h = durations[i] * 5 / 60

        if abs(z_at_exit) >= 4.0:
            reason = "STOP_LOSS"
        elif exit_time >= time(15, 25):
            reason = "EOD_FLAT"
        elif (side == 1 and z_at_exit > -1.25) or (side == -1 and z_at_exit < 1.25):
            reason = "MEAN_REVERT"
        else:
            reason = "OTHER"

        exit_reasons.append({
            "trade": i, "pnl": float(pnls[i]), "side": side,
            "reason": reason, "z_entry": float(zscore[eb]),
            "z_exit": float(z_at_exit), "duration_bars": int(durations[i]),
            "duration_h": duration_h,
            "entry_hour": entry_times[i].hour,
            "exit_hour": exit_times[i].hour,
            "confidence_entry": float(confidence[eb]),
            "winner": pnls[i] > 0,
        })

    df = pd.DataFrame(exit_reasons)

    # Exit reason breakdown
    for reason in ["MEAN_REVERT", "STOP_LOSS", "EOD_FLAT", "OTHER"]:
        sub = df[df["reason"] == reason]
        if len(sub) == 0:
            continue
        wr = (sub["pnl"] > 0).mean() * 100
        print(f" {reason:<14} {len(sub):>4} trades  Win: {wr:>5.1f}%  "
              f"PnL total: ${sub['pnl'].sum():>8,.0f}  Avg: ${sub['pnl'].mean():>6,.0f}")

    # --- Winners vs Losers comparison ---
    print("\n --- WINNERS vs LOSERS ---")
    winners = df[df["winner"]]
    losers = df[~df["winner"]]

    for label, sub in [("WINNERS", winners), ("LOSERS", losers)]:
        if len(sub) == 0:
            continue
        print(f"\n {label} ({len(sub)} trades):")
        print(f"   Duree moyenne: {sub['duration_h'].mean():.1f}h ({sub['duration_bars'].mean():.0f} bars)")
        print(f"   Duree mediane: {np.median(sub['duration_h']):.1f}h")
        print(f"   |z| entree moyen: {sub['z_entry'].abs().mean():.2f}")
        print(f"   |z| sortie moyen: {sub['z_exit'].abs().mean():.2f}")
        print(f"   Confidence entree: {sub['confidence_entry'].mean():.1f}%")
        print(f"   PnL moyen: ${sub['pnl'].mean():,.0f}")
        print(f"   PnL median: ${np.median(sub['pnl']):,.0f}")
        print(f"   Pire trade: ${sub['pnl'].min():,.0f}")
        print(f"   Meilleur: ${sub['pnl'].max():,.0f}")

        # Heure distribution
        hour_counts = sub.groupby("entry_hour").size()
        print(f"   Heures entree: ", end="")
        for h in range(2, 15):
            if h in hour_counts.index:
                print(f"{h}h:{hour_counts[h]} ", end="")
        print()

        # Exit reason distribution
        reason_counts = sub.groupby("reason").size()
        print(f"   Exit reasons: ", end="")
        for r, c in reason_counts.items():
            print(f"{r}:{c} ", end="")
        print()

    # --- Impact du stop loss ---
    print("\n --- SIMULATION STOP LOSS ALTERNATIFS ---")
    print(f" {'Config':<25} {'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Avg':>7} {'Shrp':>5} {'DD%':>5}")
    print(" " + "-" * 80)

    for z_stop in [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
        gen_alt = SignalGenerator(config=SignalConfig(z_entry=3.0, z_exit=1.25, z_stop=z_stop))
        raw_alt = gen_alt.generate(pd.Series(zscore, index=idx)).values
        sig_alt = apply_signal_filter(
            raw_alt, confidence, 70.0, idx,
            entry_start_min=240, entry_end_min=840, flat_min=930,
        )
        bt_alt = run_bt(px_a, px_b, sig_alt, beta)
        if bt_alt["trades"] > 0:
            eq = bt_alt["equity"]
            dd = compute_dd(eq)
            sharpe = compute_sharpe(eq)
            print(
                f" z_stop={z_stop:.1f}                {bt_alt['trades']:>5} {bt_alt['win_rate']:>5.1f}% "
                f"${bt_alt['pnl']:>9,.0f} {bt_alt['profit_factor']:>5.2f} "
                f"${bt_alt['avg_pnl_trade']:>6,.0f} {sharpe:>5.2f} {dd:>5.1f}%"
            )

    # --- Duree et PnL ---
    print("\n --- PNL PAR TRANCHE DE DUREE ---")
    bins = [(0, 12, "<1h"), (12, 36, "1-3h"), (36, 72, "3-6h"), (72, 144, "6-12h"), (144, 9999, ">12h")]
    for lo, hi, label in bins:
        mask = (df["duration_bars"] >= lo) & (df["duration_bars"] < hi)
        sub = df[mask]
        if len(sub) == 0:
            continue
        wr = (sub["pnl"] > 0).mean() * 100
        print(f" {label:<8} {len(sub):>4} trades  Win: {wr:>5.1f}%  "
              f"PnL: ${sub['pnl'].sum():>8,.0f}  Avg: ${sub['pnl'].mean():>6,.0f}")


# ======================================================================
# BATCH 3: Calibration 3min params propres
# ======================================================================

def batch3():
    print("\n" + "=" * 130)
    print(" BATCH 3 - CALIBRATION 3min PARAMS PROPRES (pas de scaling lineaire)")
    print(" NQ_YM 3min | flat 15:30 CT | entry [04:00-14:00)")
    print("=" * 130)

    aligned_3min = load_nq_ym("3min")
    if aligned_3min is None:
        print(" Pas de cache 3min. Lancer d'abord la preparation des donnees.")
        return

    print(f" Data: {len(aligned_3min.df):,} barres 3min")

    # Independent parameter grid for 3min
    ols_windows = [1320, 1980, 2640, 3960]  # 3-12 jours (440 bars/jour en 3min)
    zw_windows = [20, 30, 40, 60]
    z_entries = [2.75, 3.0, 3.5]
    z_exits = [1.00, 1.25, 1.50]
    z_stops = [3.5, 4.0, 4.5]
    confs = [65.0, 67.0, 70.0]

    # Metric profiles for 3min (independent, not scaled)
    metrics_profiles = {
        "ultra_court": MetricsConfig(adf_window=10, hurst_window=40, halflife_window=10, correlation_window=5),
        "court":       MetricsConfig(adf_window=20, hurst_window=80, halflife_window=20, correlation_window=10),
        "moyen":       MetricsConfig(adf_window=40, hurst_window=160, halflife_window=40, correlation_window=20),
    }

    flat_min = 15 * 60 + 30  # 15:30
    entry_start = 240  # 04:00
    entry_end = 840    # 14:00

    total = len(ols_windows) * len(metrics_profiles) * len(zw_windows) * len(z_entries) * len(z_exits) * len(z_stops) * len(confs)
    print(f" Combinaisons: {total}")

    header = (f" {'OLS':>5} {'ZW':>3} {'Prof':<11} {'Zent':>5} {'Zex':>4} {'Zst':>4} {'Conf':>4} "
              f"{'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Avg':>7} {'Shrp':>5} {'DD%':>5} {'Trd/An':>6}")
    print(header)
    print(" " + "-" * 120)

    results = []
    done = 0

    for ols_w in ols_windows:
        for prof_name, prof_cfg in metrics_profiles.items():
            # Compute hedge + metrics + confidence (once per OLS+profile combo)
            try:
                px_a, px_b, idx, beta, confidence, zscore, _, _ = compute_pipeline_arrays(
                    aligned_3min, ols_window=ols_w, zscore_window=zw_windows[0], metrics_cfg=prof_cfg
                )
            except Exception as e:
                print(f" SKIP OLS={ols_w} {prof_name}: {e}")
                done += len(zw_windows) * len(z_entries) * len(z_exits) * len(z_stops) * len(confs)
                continue

            # Recompute zscore for each ZW
            spread_series = pd.Series(
                px_a * 0, index=idx  # placeholder, will recompute
            )
            # Actually need the spread from hedge result
            est = create_estimator("ols_rolling", window=ols_w, zscore_window=zw_windows[0])
            hr = est.estimate(aligned_3min)
            spread = hr.spread

            for zw in zw_windows:
                mu = spread.rolling(zw).mean()
                sigma = spread.rolling(zw).std()
                with np.errstate(divide="ignore", invalid="ignore"):
                    zs = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
                zscore_zw = zs

                for z_entry in z_entries:
                    for z_exit in z_exits:
                        for z_stop in z_stops:
                            if z_exit >= z_entry or z_stop <= z_entry:
                                done += len(confs)
                                continue

                            gen = SignalGenerator(config=SignalConfig(z_entry=z_entry, z_exit=z_exit, z_stop=z_stop))
                            raw_signals = gen.generate(pd.Series(zscore_zw, index=idx)).values

                            for min_conf in confs:
                                sig = apply_signal_filter(
                                    raw_signals, confidence, min_conf, idx,
                                    entry_start, entry_end, flat_min,
                                )
                                bt = run_bt(px_a, px_b, sig, beta)

                                done += 1
                                num = bt["trades"]
                                if num < 10:
                                    continue

                                eq = bt["equity"]
                                dd = compute_dd(eq)
                                sharpe = compute_sharpe(eq, bars_per_day=440)
                                years = (idx[-1] - idx[0]).days / 365.25
                                trd_an = num / years if years > 0 else 0

                                if bt["pnl"] > 0:
                                    print(
                                        f" {ols_w:>5} {zw:>3} {prof_name:<11} {z_entry:>5.2f} {z_exit:>4.2f} {z_stop:>4.1f} {min_conf:>4.0f} "
                                        f"{num:>5} {bt['win_rate']:>5.1f}% ${bt['pnl']:>9,.0f} {bt['profit_factor']:>5.2f} "
                                        f"${bt['avg_pnl_trade']:>6,.0f} {sharpe:>5.2f} {dd:>5.1f}% {trd_an:>6.1f}"
                                    )

                                results.append({
                                    "ols": ols_w, "zw": zw, "profile": prof_name,
                                    "z_entry": z_entry, "z_exit": z_exit, "z_stop": z_stop,
                                    "conf": min_conf, "trades": num, "win_rate": bt["win_rate"],
                                    "pnl": bt["pnl"], "pf": bt["profit_factor"],
                                    "avg_pnl": bt["avg_pnl_trade"], "sharpe": sharpe,
                                    "dd_pct": dd, "trd_an": trd_an,
                                })

    # Top 20
    print(f"\n {'='*130}")
    print(" TOP 20 CONFIGS 3min PAR SCORE COMPOSITE")
    print(f" {'='*130}")
    if results:
        df = pd.DataFrame(results)
        df_pos = df[df["pf"] > 1.0].copy()
        if len(df_pos) > 0:
            df_pos["score"] = df_pos["pf"] * df_pos["sharpe"].clip(lower=0) * np.sqrt(df_pos["trades"])
            df_pos = df_pos.sort_values("score", ascending=False).head(20)

            print(f" {'#':<3} {'OLS':>5} {'ZW':>3} {'Prof':<11} {'Zent':>5} {'Zex':>4} {'Zst':>4} {'Conf':>4} "
                  f"{'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Shrp':>5} {'DD%':>5} {'Trd/An':>6} {'Score':>6}")
            print(" " + "-" * 130)
            for i, (_, r) in enumerate(df_pos.iterrows()):
                print(
                    f" {i+1:<3} {r['ols']:>5} {r['zw']:>3} {r['profile']:<11} "
                    f"{r['z_entry']:>5.2f} {r['z_exit']:>4.2f} {r['z_stop']:>4.1f} {r['conf']:>4.0f} "
                    f"{r['trades']:>5} {r['win_rate']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>5.2f} "
                    f"{r['sharpe']:>5.2f} {r['dd_pct']:>5.1f}% {r['trd_an']:>6.1f} {r['score']:>6.1f}"
                )
        else:
            print(" Aucune config profitable en 3min.")

    print(f"\n {done} combinaisons testees, {len(results)} avec >= 10 trades")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Batch volume search NQ_YM")
    parser.add_argument("--batch1", action="store_true", help="Grid z_entry + fenetres + stops")
    parser.add_argument("--batch2", action="store_true", help="Analyse trades perdants")
    parser.add_argument("--batch3", action="store_true", help="Calibration 3min")
    parser.add_argument("--all", action="store_true", help="Tout lancer")
    args = parser.parse_args()

    run_all = args.all or not any([args.batch1, args.batch2, args.batch3])

    if args.batch1 or run_all:
        aligned = load_nq_ym("5min")
        batch1(aligned)

    if args.batch2 or run_all:
        aligned = load_nq_ym("5min")
        batch2(aligned)

    if args.batch3 or run_all:
        batch3()


if __name__ == "__main__":
    main()
