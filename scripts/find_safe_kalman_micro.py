"""Find Kalman configs viable for propfirm with MICRO contracts (MNQ/MYM).

Runs full backtest with micro multipliers, then scales to 1-5 contracts.
Also shows E-mini results for comparison.

Micro specs:
  MNQ: multiplier=2.0, tick_size=0.25, tick_value=0.50
  MYM: multiplier=0.5, tick_size=1.0, tick_value=0.50
  Commission: ~$0.62 per contract round-turn (NinjaTrader micro)

Usage:
    python scripts/find_safe_kalman_micro.py
"""

import sys
import time as time_mod
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.generator import generate_signals_numba
from src.signals.filters import (
    ConfidenceConfig, compute_confidence,
    _apply_conf_filter_numba, apply_window_filter_numba,
)
from src.backtest.engine import run_backtest_vectorized

# ── Constants ──
# E-mini
MULT_A_EMINI, MULT_B_EMINI = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
COMM_EMINI = 2.50
# Micro
MULT_A_MICRO, MULT_B_MICRO = 2.0, 0.5
COMM_MICRO = 0.62

SLIPPAGE = 1
CAPITAL = 100_000.0
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

_kalman_cache = {}


def get_kalman_data(aligned, alpha, profil):
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


def build_signal(aligned, minutes, row):
    beta, zscore, confidence = get_kalman_data(aligned, row["alpha_ratio"], row["profil"])
    raw = generate_signals_numba(zscore, row["z_entry"], row["z_exit"], row["z_stop"])
    sig = _apply_conf_filter_numba(raw, confidence, row["min_confidence"])
    es, ee = WINDOWS_MAP[row["window"]]
    sig = apply_window_filter_numba(sig, minutes, es, ee, FLAT_MIN)
    return sig, beta


def run_bt(px_a, px_b, sig, beta, mult_a, mult_b, commission):
    return run_backtest_vectorized(
        px_a, px_b, sig, beta,
        mult_a, mult_b, TICK_A, TICK_B,
        SLIPPAGE, commission, CAPITAL,
    )


def analyze_bt(bt, years, n_contracts=1):
    """Analyze backtest results, scaling PnL by n_contracts."""
    n = bt["trades"]
    if n == 0:
        return None
    pnls = bt["trade_pnls"] * n_contracts
    equity_base = bt["equity"]
    # Scale equity: capital + (equity - capital) * n_contracts
    equity = CAPITAL + (equity_base - CAPITAL) * n_contracts

    running_max = np.maximum.accumulate(equity)
    drawdowns = equity - running_max
    max_dd = drawdowns.min()

    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(n)) if n > 1 and pnls.std() > 0 else 0
    total_pnl = pnls.sum()
    calmar = float(total_pnl / abs(max_dd)) if max_dd < 0 else 0

    # Streaks
    streak = 0
    max_streak = 0
    consec_dd = 0
    max_consec_dd = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
            consec_dd += p
            max_consec_dd = min(max_consec_dd, consec_dd)
        else:
            streak = 0
            consec_dd = 0

    wr = (pnls > 0).sum() / n * 100
    pf = pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 999

    return {
        "trades": n, "wr": wr, "pnl": total_pnl, "pf": pf,
        "avg_pnl": total_pnl / n, "avg_dur": bt["avg_duration_bars"],
        "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
        "max_streak": max_streak, "max_consec_dd": max_consec_dd,
        "max_loss": float(pnls.min()), "max_win": float(pnls.max()),
        "trades_yr": n / years, "pnl_yr": total_pnl / years,
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
    df = pd.read_csv("output/grid_results_kalman_v3_nqym_filtered.csv")
    print(f"Loaded {len(df):,} filtered configs")

    # ── Pre-filter candidates ──
    candidates = df[
        (df.profit_factor >= 1.3) &
        (df.win_rate >= 65) &
        (df.trades >= 20) &
        (df.avg_pnl_trade >= 30)
    ].copy()
    candidates["quality"] = candidates["profit_factor"] * candidates["win_rate"] / 100
    candidates = candidates.sort_values("quality", ascending=False)

    # Deduplicate
    candidates["ze_bin"] = (candidates["z_entry"] * 4).round() / 4
    candidates["zx_bin"] = (candidates["z_exit"] * 4).round() / 4
    deduped = candidates.groupby(
        ["alpha_ratio", "profil", "window", "ze_bin", "zx_bin"]
    ).first().reset_index()
    print(f"Candidates to test: {len(deduped):,}")

    # ── Run micro backtests ──
    print(f"\nRunning {len(deduped):,} micro backtests...")
    results = []
    for i, (_, row) in enumerate(deduped.iterrows()):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(deduped)}]...", flush=True)

        sig, beta = build_signal(aligned, minutes, row)

        # Run with MICRO multipliers
        bt_micro = run_bt(px_a, px_b, sig, beta, MULT_A_MICRO, MULT_B_MICRO, COMM_MICRO)

        if bt_micro["trades"] == 0:
            continue

        # Analyze at 1, 2, 3 micro contracts
        for nc in [1, 2, 3]:
            r = analyze_bt(bt_micro, years, n_contracts=nc)
            if r is None:
                continue
            r["alpha_ratio"] = row["alpha_ratio"]
            r["profil"] = row["profil"]
            r["window"] = row["window"]
            r["z_entry"] = row["z_entry"]
            r["z_exit"] = row["z_exit"]
            r["z_stop"] = row["z_stop"]
            r["min_confidence"] = row["min_confidence"]
            r["n_contracts"] = nc
            results.append(r)

    rdf = pd.DataFrame(results)
    print(f"\nTotal results: {len(rdf):,} ({len(rdf)//3} configs x 3 scaling)")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: MaxDD < $6,500 — ALL SCALINGS
    # ══════════════════════════════════════════════════════════════════
    for nc in [1, 2, 3]:
        dd_limit = 6500
        sub = rdf[(rdf["n_contracts"] == nc) & (rdf["max_dd"].abs() < dd_limit)].copy()
        sub = sub.sort_values("pnl", ascending=False)

        print(f"\n\n{'='*160}")
        print(f" MICRO x{nc} — MaxDD < ${dd_limit:,} ({len(sub):,} configs)")
        print(f"{'='*160}")

        if len(sub) == 0:
            print("  Aucune config trouvee.")
            continue

        # ── Top 15 by PnL ──
        print(f"\n  Top 15 by PnL:")
        print(f"  {'#':<3} {'Alpha':>8} {'Prof':<11} {'Window':<12} "
              f"{'Zent':>5} {'Zex':>5} {'Zst':>5} {'Conf':>4} | "
              f"{'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
              f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'MaxLos':>7} "
              f"{'Trd/y':>5} {'PnL/y':>8} {'$/day':>6}")
        print(f"  {'-'*155}")

        for i, (_, r) in enumerate(sub.head(15).iterrows()):
            daily = r["pnl_yr"] / 252
            flag = ""
            if abs(r["max_dd"]) > 4500:
                flag = " [>4.5k]"
            print(f"  {i+1:<3} {r['alpha_ratio']:>8.1e} {r['profil']:<11} {r['window']:<12} "
                  f"{r['z_entry']:>5.3f} {r['z_exit']:>5.3f} {r['z_stop']:>5.3f} {r['min_confidence']:>4.0f} | "
                  f"{r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} {r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} "
                  f"${r['max_dd']:>6,.0f} {r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} "
                  f"${r['max_loss']:>6,.0f} "
                  f"{r['trades_yr']:>5.1f} ${r['pnl_yr']:>7,.0f} ${daily:>5,.0f}{flag}")

        # ── Top 10 BALANCED (trades >= 50, PF >= 1.5) ──
        bal = sub[(sub["trades"] >= 50) & (sub["pf"] >= 1.3)].sort_values("pnl", ascending=False)
        print(f"\n  Top 10 BALANCED (trades >= 50, PF >= 1.3):")
        if len(bal) > 0:
            print(f"  {'#':<3} {'Alpha':>8} {'Prof':<11} {'Window':<12} "
                  f"{'Zent':>5} {'Zex':>5} {'Zst':>5} {'Conf':>4} | "
                  f"{'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
                  f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'MaxLos':>7} "
                  f"{'Trd/y':>5} {'PnL/y':>8} {'$/day':>6}")
            print(f"  {'-'*155}")
            for i, (_, r) in enumerate(bal.head(10).iterrows()):
                daily = r["pnl_yr"] / 252
                flag = ""
                if abs(r["max_dd"]) > 4500:
                    flag = " [>4.5k]"
                print(f"  {i+1:<3} {r['alpha_ratio']:>8.1e} {r['profil']:<11} {r['window']:<12} "
                      f"{r['z_entry']:>5.3f} {r['z_exit']:>5.3f} {r['z_stop']:>5.3f} {r['min_confidence']:>4.0f} | "
                      f"{r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} {r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} "
                      f"${r['max_dd']:>6,.0f} {r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} "
                      f"${r['max_loss']:>6,.0f} "
                      f"{r['trades_yr']:>5.1f} ${r['pnl_yr']:>7,.0f} ${daily:>5,.0f}{flag}")
        else:
            print("  Aucune config trouvee.")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: Comparison table — same config at x1, x2, x3 micro + E-mini
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print(f" COMPARAISON SCALING : memes configs a x1, x2, x3 micro + E-mini")
    print(f"{'='*140}")

    # Pick 5 interesting configs from the micro x2 results
    micro2 = rdf[(rdf["n_contracts"] == 2) & (rdf["max_dd"].abs() < 6500)]
    micro2 = micro2.sort_values("pnl", ascending=False)

    # Diverse selection
    selected_keys = []
    seen_patterns = set()
    for _, row in micro2.iterrows():
        pattern = (row["alpha_ratio"], row["profil"], row["window"])
        z_key = (round(row["z_entry"], 2), round(row["z_exit"], 2))
        if pattern not in seen_patterns:
            seen_patterns.add(pattern)
            selected_keys.append(row)
        if len(selected_keys) >= 5:
            break

    for cfg in selected_keys:
        label = f"a={cfg['alpha_ratio']:.1e} {cfg['profil'][:5]} {cfg['window']} ze={cfg['z_entry']:.3f}"
        print(f"\n  {label}:")
        print(f"  {'Scale':<12} {'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
              f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'MaxLos':>7} {'PnL/y':>8} {'$/day':>6} {'Safe':>6}")
        print(f"  {'-'*100}")

        sig, beta = build_signal(aligned, minutes, cfg)

        # Micro x1, x2, x3
        bt_micro = run_bt(px_a, px_b, sig, beta, MULT_A_MICRO, MULT_B_MICRO, COMM_MICRO)
        for nc in [1, 2, 3]:
            r = analyze_bt(bt_micro, years, n_contracts=nc)
            if r is None:
                continue
            daily = r["pnl_yr"] / 252
            safe = "SAFE" if abs(r["max_dd"]) < 4500 else ("WARN" if abs(r["max_dd"]) < 6500 else "DANGER")
            print(f"  {'Micro x'+str(nc):<12} {r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} "
                  f"{r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} ${r['max_dd']:>6,.0f} "
                  f"{r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} ${r['max_loss']:>6,.0f} "
                  f"${r['pnl_yr']:>7,.0f} ${daily:>5,.0f} {safe:>6}")

        # E-mini x1
        bt_emini = run_bt(px_a, px_b, sig, beta, MULT_A_EMINI, MULT_B_EMINI, COMM_EMINI)
        r = analyze_bt(bt_emini, years, n_contracts=1)
        if r:
            daily = r["pnl_yr"] / 252
            safe = "SAFE" if abs(r["max_dd"]) < 4500 else ("WARN" if abs(r["max_dd"]) < 6500 else "DANGER")
            print(f"  {'E-mini x1':<12} {r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} "
                  f"{r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} ${r['max_dd']:>6,.0f} "
                  f"{r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} ${r['max_loss']:>6,.0f} "
                  f"${r['pnl_yr']:>7,.0f} ${daily:>5,.0f} {safe:>6}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: OLS ConfigE for comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print(f" OLS CONFIG E — MICRO vs E-MINI")
    print(f"{'='*140}")

    est_ols = create_estimator("ols_rolling", window=3300, zscore_window=30)
    hr_ols = est_ols.estimate(aligned)
    beta_ols = hr_ols.beta.values
    spread_ols = hr_ols.spread
    mu = spread_ols.rolling(30).mean()
    sigma = spread_ols.rolling(30).std()
    zs_ols = ((spread_ols - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zs_ols = np.ascontiguousarray(zs_ols, dtype=np.float64)
    raw_ols = generate_signals_numba(zs_ols, 3.15, 1.0, 4.5)
    mcfg_ols = PROFILES_MAP["tres_court"]
    met_ols = compute_all_metrics(spread_ols, aligned.df["close_a"], aligned.df["close_b"], mcfg_ols)
    conf_ols = compute_confidence(met_ols, CONF_CFG).values
    sig_ols = _apply_conf_filter_numba(raw_ols, conf_ols, 67.0)
    sig_ols = apply_window_filter_numba(sig_ols, minutes, 120, 840, FLAT_MIN)

    print(f"\n  {'Scale':<12} {'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} {'Avg$':>6} "
          f"{'MaxDD':>7} {'Strk':>4} {'ConsDD':>7} {'MaxLos':>7} {'PnL/y':>8} {'$/day':>6} {'Safe':>6}")
    print(f"  {'-'*100}")

    # OLS Micro
    bt_ols_micro = run_bt(px_a, px_b, sig_ols, beta_ols, MULT_A_MICRO, MULT_B_MICRO, COMM_MICRO)
    for nc in [1, 2, 3]:
        r = analyze_bt(bt_ols_micro, years, n_contracts=nc)
        if r:
            daily = r["pnl_yr"] / 252
            safe = "SAFE" if abs(r["max_dd"]) < 4500 else ("WARN" if abs(r["max_dd"]) < 6500 else "DANGER")
            print(f"  {'Micro x'+str(nc):<12} {r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} "
                  f"{r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} ${r['max_dd']:>6,.0f} "
                  f"{r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} ${r['max_loss']:>6,.0f} "
                  f"${r['pnl_yr']:>7,.0f} ${daily:>5,.0f} {safe:>6}")

    # OLS E-mini
    bt_ols_emini = run_bt(px_a, px_b, sig_ols, beta_ols, MULT_A_EMINI, MULT_B_EMINI, COMM_EMINI)
    r = analyze_bt(bt_ols_emini, years, n_contracts=1)
    if r:
        daily = r["pnl_yr"] / 252
        safe = "SAFE" if abs(r["max_dd"]) < 4500 else ("WARN" if abs(r["max_dd"]) < 6500 else "DANGER")
        print(f"  {'E-mini x1':<12} {r['trades']:>4} {r['wr']:>4.1f}% ${r['pnl']:>8,.0f} "
              f"{r['pf']:>5.2f} ${r['avg_pnl']:>5,.0f} ${r['max_dd']:>6,.0f} "
              f"{r['max_streak']:>4} ${r['max_consec_dd']:>6,.0f} ${r['max_loss']:>6,.0f} "
              f"${r['pnl_yr']:>7,.0f} ${daily:>5,.0f} {safe:>6}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: Distribution MaxDD micro x2
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print(f" DISTRIBUTION MaxDD — Micro x2")
    print(f"{'='*80}")

    micro2_all = rdf[rdf["n_contracts"] == 2].copy()
    bins = [0, 1000, 2000, 3000, 4000, 4500, 5000, 6000, 6500, 8000, 50000]
    labels = ["<$1k", "$1-2k", "$2-3k", "$3-4k", "$4-4.5k", "$4.5-5k", "$5-6k", "$6-6.5k", "$6.5-8k", ">$8k"]
    micro2_all["dd_bin"] = pd.cut(micro2_all["max_dd"].abs(), bins=bins, labels=labels, right=False)
    dist = micro2_all["dd_bin"].value_counts().sort_index()
    total = len(micro2_all)
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
