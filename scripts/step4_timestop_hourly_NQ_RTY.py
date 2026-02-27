"""Etape 4 -- Time Stop + Hourly Deep-Dive NQ/RTY OLS.

Section A: Time stop impact (0=off, 12, 18, 24, 36, 48 bars) on 10 configs.
Section B: Hourly PnL decomposition by entry hour (CT).

Usage:
    python scripts/step4_timestop_hourly_NQ_RTY.py
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
    apply_time_stop,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

# ======================================================================
# Constants NQ/RTY
# ======================================================================
MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
BARS_PER_DAY = 264
BARS_PER_YEAR = BARS_PER_DAY * 252

TIME_STOP_VALUES = [0, 12, 18, 24, 36, 48]  # bars (5min): 0=off, 1h, 1.5h, 2h, 3h, 4h

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


def build_pre_timestop(aligned, minutes, cfg):
    """Build signal up to (but not including) time stop.

    Returns (sig_no_ts, beta) where sig_no_ts has confidence + window filters
    but NO time stop applied yet.
    """
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

    return sig, beta


def run_bt(px_a, px_b, sig, beta):
    """Run backtest and return key metrics."""
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity - running_max).min())

    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    sharpe = float((np.mean(returns) / np.std(returns)) * np.sqrt(BARS_PER_YEAR)
                   if np.std(returns) > 0 else 0.0)

    return {
        "trades": bt["trades"],
        "wr": bt["win_rate"],
        "pnl": bt["pnl"],
        "pf": bt["profit_factor"],
        "sharpe": round(sharpe, 2),
        "max_dd": round(max_dd, 0),
        "avg_pnl": bt["avg_pnl_trade"],
        "avg_dur": bt.get("avg_duration_bars", 0),
        "trade_pnls": bt.get("trade_pnls", []),
        "equity": equity,
    }


def get_entry_indices(sig):
    """Find bar indices where entries happen (0 -> non-zero transition)."""
    entries = []
    for i in range(1, len(sig)):
        if sig[i - 1] == 0 and sig[i] != 0:
            entries.append(i)
    return entries


def main():
    t_start = time_mod.time()

    print("=" * 130)
    print("  ETAPE 4 -- TIME STOP + HOURLY DEEP-DIVE NQ/RTY OLS (10 configs)")
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
    hours_arr = idx.hour.values
    print(f"Data: {len(px_a):,} bars")

    # Load 10 configs from step 3
    top10_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step3_top10.csv"
    configs_df = pd.read_csv(top10_path)
    print(f"Configs: {len(configs_df)}")

    configs = []
    for _, row in configs_df.iterrows():
        configs.append({
            "label": row["label"],
            "ols": int(row["ols"]),
            "zw": int(row["zw"]),
            "profile": row["profile"],
            "window": row["window"],
            "z_entry": row["z_entry"],
            "z_exit": row["z_exit"],
            "z_stop": row["z_stop"],
            "conf": row["conf"],
            "tier": row["tier"],
        })

    # ==================================================================
    # Pre-compute base signals (without time stop)
    # ==================================================================
    print("\nPre-computing base signals for 10 configs...")
    base_signals = {}
    base_betas = {}
    for cfg in configs:
        sig, beta = build_pre_timestop(aligned, minutes, cfg)
        base_signals[cfg["label"]] = sig
        base_betas[cfg["label"]] = beta
        print(f"  {cfg['label']}: {np.sum(sig != 0):,} bars in position")

    # ==================================================================
    # SECTION A: TIME STOP IMPACT
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION A: TIME STOP IMPACT")
    print(f"  Test: {TIME_STOP_VALUES} bars (5min) = off, 1h, 1.5h, 2h, 3h, 4h")
    print(f"{'='*130}")

    ts_results = []

    for cfg in configs:
        label = cfg["label"]
        sig_base = base_signals[label]
        beta = base_betas[label]

        for ts in TIME_STOP_VALUES:
            if ts == 0:
                sig_final = sig_base.copy()
            else:
                sig_final = apply_time_stop(sig_base.copy(), ts)

            bt = run_bt(px_a, px_b, sig_final, beta)

            ts_results.append({
                "label": label,
                "tier": cfg["tier"],
                "time_stop": ts,
                "ts_label": f"{ts*5}min" if ts > 0 else "off",
                "trades": bt["trades"],
                "wr": bt["wr"],
                "pnl": bt["pnl"],
                "pf": bt["pf"],
                "sharpe": bt["sharpe"],
                "max_dd": bt["max_dd"],
                "avg_pnl": bt["avg_pnl"],
                "avg_dur": bt["avg_dur"],
            })

    df_ts = pd.DataFrame(ts_results)

    # Print per-config time stop comparison
    for cfg in configs:
        label = cfg["label"]
        tier = cfg["tier"]
        subset = df_ts[df_ts["label"] == label].sort_values("time_stop")

        # Baseline (no time stop)
        base = subset[subset["time_stop"] == 0].iloc[0]

        print(f"\n  --- {label} [{tier}] ---")
        print(f"  {'TS':>6} {'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Shrp':>5} "
              f"{'MaxDD':>9} {'AvgPnL':>7} {'AvgDur':>6} | {'dPF':>5} {'dDD':>7} {'dPnL':>8}")
        print(f"  {'-'*105}")

        for _, r in subset.iterrows():
            ts_lab = r["ts_label"]
            dpf = r["pf"] - base["pf"]
            ddd = r["max_dd"] - base["max_dd"]
            dpnl = r["pnl"] - base["pnl"]
            flag = " <<" if r["pf"] >= base["pf"] and r["max_dd"] >= base["max_dd"] else ""
            print(f"  {ts_lab:>6} {r['trades']:>4} {r['wr']:>5.1f} ${r['pnl']:>8,.0f} {r['pf']:>5.2f} "
                  f"{r['sharpe']:>5.2f} ${r['max_dd']:>8,.0f} ${r['avg_pnl']:>6,.0f} "
                  f"{r['avg_dur']:>6.1f} | {dpf:>+5.2f} ${ddd:>+7,.0f} ${dpnl:>+8,.0f}{flag}")

    # Global summary: best time stop per config
    print(f"\n\n  {'='*130}")
    print("  RESUME: MEILLEUR TIME STOP PAR CONFIG (critere: PF ameliore OU MaxDD ameliore sans casser PF)")
    print(f"  {'='*130}")

    print(f"\n  {'Label':<48} {'Tier':<7} {'Best TS':>7} {'PF_base':>7} {'PF_best':>7} "
          f"{'DD_base':>9} {'DD_best':>9} {'Verdict':>10}")
    print(f"  {'-'*115}")

    best_ts_per_config = {}
    for cfg in configs:
        label = cfg["label"]
        subset = df_ts[df_ts["label"] == label].copy()
        base_row = subset[subset["time_stop"] == 0].iloc[0]

        # Best = highest PF among those with MaxDD >= base MaxDD (less drawdown)
        # If none improve both, pick the one with best PF
        improved = subset[subset["max_dd"] >= base_row["max_dd"]]
        if len(improved) > 0:
            best = improved.sort_values("pf", ascending=False).iloc[0]
        else:
            best = subset.sort_values("pf", ascending=False).iloc[0]

        ts_val = int(best["time_stop"])
        best_ts_per_config[label] = ts_val

        verdict = "KEEP" if ts_val == 0 else f"USE {ts_val}"
        if best["pf"] > base_row["pf"] + 0.02 and best["max_dd"] >= base_row["max_dd"]:
            verdict += " ++"
        elif best["pf"] > base_row["pf"] + 0.02:
            verdict += " +"

        print(f"  {label:<48} {cfg['tier']:<7} {best['ts_label']:>7} "
              f"{base_row['pf']:>7.2f} {best['pf']:>7.2f} "
              f"${base_row['max_dd']:>8,.0f} ${best['max_dd']:>8,.0f} {verdict:>10}")

    # ==================================================================
    # SECTION B: HOURLY DEEP-DIVE
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION B: HOURLY DEEP-DIVE (PnL par heure d'entree CT)")
    print(f"{'='*130}")

    # Use base signals (no time stop) for hourly analysis
    for cfg in configs:
        label = cfg["label"]
        tier = cfg["tier"]
        sig = base_signals[label]
        beta = base_betas[label]

        # Get entry indices
        entry_indices = get_entry_indices(sig)

        # Run full backtest to get trade PnLs
        bt = run_bt(px_a, px_b, sig, beta)
        trade_pnls = bt["trade_pnls"]

        if len(entry_indices) != len(trade_pnls):
            print(f"\n  {label}: MISMATCH entries={len(entry_indices)} vs trades={len(trade_pnls)}, skip")
            continue

        # Map entry index -> hour -> PnL
        hourly_data = {}
        for eidx, tpnl in zip(entry_indices, trade_pnls):
            h = int(hours_arr[eidx])
            if h not in hourly_data:
                hourly_data[h] = {"pnl": 0.0, "trades": 0, "wins": 0}
            hourly_data[h]["pnl"] += tpnl
            hourly_data[h]["trades"] += 1
            if tpnl > 0:
                hourly_data[h]["wins"] += 1

        print(f"\n  --- {label} [{tier}] ({len(trade_pnls)} trades) ---")
        print(f"  {'Hour':>6} {'Trd':>4} {'WR%':>5} {'PnL':>9} {'AvgPnL':>8} {'CumPnL':>9} {'%Tot':>5}")
        print(f"  {'-'*55}")

        total_pnl = bt["pnl"]
        cum_pnl = 0.0
        for h in sorted(hourly_data.keys()):
            d = hourly_data[h]
            wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
            avg = d["pnl"] / d["trades"] if d["trades"] > 0 else 0
            cum_pnl += d["pnl"]
            pct = d["pnl"] / total_pnl * 100 if total_pnl != 0 else 0
            flag = " ***" if pct > 20 else (" **" if pct > 10 else (" *" if pct > 5 else ""))
            neg = " TOXIC" if d["pnl"] < -500 else ""
            print(f"  {h:>4}:00 {d['trades']:>4} {wr:>5.1f} ${d['pnl']:>8,.0f} ${avg:>7,.0f} "
                  f"${cum_pnl:>8,.0f} {pct:>5.1f}%{flag}{neg}")

    # ==================================================================
    # SECTION C: COMBINED SUMMARY
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION C: TABLE FINALE (baseline + best time stop + heures cles)")
    print(f"{'='*130}")

    print(f"\n  {'#':>2} {'Label':<42} {'Tier':<6} {'BestTS':>6} "
          f"{'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'MaxDD':>9} "
          f"{'BestHrs':>10} {'WorstHr':>10}")
    print(f"  {'-'*125}")

    for rank, cfg in enumerate(configs, 1):
        label = cfg["label"]
        tier = cfg["tier"]
        ts_val = best_ts_per_config[label]

        # Get metrics for best time stop
        row = df_ts[(df_ts["label"] == label) & (df_ts["time_stop"] == ts_val)].iloc[0]

        # Hourly data
        sig = base_signals[label]
        entry_indices = get_entry_indices(sig)
        bt_full = run_bt(px_a, px_b, sig, base_betas[label])
        trade_pnls = bt_full["trade_pnls"]

        hourly_data = {}
        if len(entry_indices) == len(trade_pnls):
            for eidx, tpnl in zip(entry_indices, trade_pnls):
                h = int(hours_arr[eidx])
                if h not in hourly_data:
                    hourly_data[h] = 0.0
                hourly_data[h] += tpnl

        if hourly_data:
            sorted_hrs = sorted(hourly_data.items(), key=lambda x: -x[1])
            best_hrs = ",".join(f"{h}h" for h, _ in sorted_hrs[:2])
            worst_hr = f"{sorted_hrs[-1][0]}h" if sorted_hrs[-1][1] < 0 else "none"
        else:
            best_hrs = "?"
            worst_hr = "?"

        ts_lab = f"{ts_val*5}m" if ts_val > 0 else "off"
        print(f"  {rank:>2} {label:<42} {tier:<6} {ts_lab:>6} "
              f"{row['trades']:>4} {row['wr']:>5.1f} ${row['pnl']:>8,.0f} {row['pf']:>5.2f} "
              f"${row['max_dd']:>8,.0f} {best_hrs:>10} {worst_hr:>10}")

    # Save results
    out_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step4_timestop.csv"
    df_ts.to_csv(out_path, index=False)
    print(f"\n  Sauvegarde: {out_path}")

    elapsed = time_mod.time() - t_start
    print(f"\n  Etape 4 complete en {elapsed:.0f}s")


if __name__ == "__main__":
    main()
