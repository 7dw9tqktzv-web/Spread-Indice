"""Validation IS/OOS + Walk-Forward + Permutation des 6 configs NQ/RTY OLS.

Configs: A_RTY, B_RTY, C_RTY, D_court, F_narrow, G_sniper
Paire: NQ/RTY (mult 20/50, tick 0.25/0.10)
Confidence: HL retire (ADF 50%, Hurst 30%, Corr 20%, HL 0%)

Usage:
    python scripts/validate_NQ_RTY_top6.py
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.data.alignment import AlignedPair
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

# ======================================================================
# NQ/RTY Constants
# ======================================================================
MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
BARS_PER_DAY = 264
BARS_PER_YEAR = BARS_PER_DAY * 252

# HL retire du scoring
CONF_WEIGHTS = ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00)

METRIC_PROFILES = {
    "p16_80":  MetricsConfig(adf_window=16, hurst_window=80,  halflife_window=16, correlation_window=8),
    "p18_96":  MetricsConfig(adf_window=18, hurst_window=96,  halflife_window=18, correlation_window=9),
    "p28_144": MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p36_96":  MetricsConfig(adf_window=36, hurst_window=96,  halflife_window=36, correlation_window=9),
    "p42_224": MetricsConfig(adf_window=42, hurst_window=224, halflife_window=42, correlation_window=21),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
    "08:00-12:00": (480, 720),
}

# ======================================================================
# 6 Configs
# ======================================================================
CONFIGS = {
    "A_RTY": {
        "ols": 9240, "zw": 20, "profile": "p36_96",
        "window": "06:00-14:00",
        "z_entry": 3.00, "z_exit": 0.75, "z_stop": 5.00, "conf": 75.0,
    },
    "B_RTY": {
        "ols": 7920, "zw": 20, "profile": "p36_96",
        "window": "06:00-14:00",
        "z_entry": 3.00, "z_exit": 0.75, "z_stop": 3.50, "conf": 75.0,
    },
    "C_RTY": {
        "ols": 7920, "zw": 20, "profile": "p42_224",
        "window": "04:00-14:00",
        "z_entry": 3.00, "z_exit": 0.75, "z_stop": 3.50, "conf": 75.0,
    },
    "D_court": {
        "ols": 3960, "zw": 28, "profile": "p28_144",
        "window": "02:00-14:00",
        "z_entry": 3.00, "z_exit": 1.25, "z_stop": 5.50, "conf": 80.0,
    },
    "F_narrow": {
        "ols": 6600, "zw": 60, "profile": "p28_144",
        "window": "08:00-12:00",
        "z_entry": 3.25, "z_exit": 0.75, "z_stop": 5.50, "conf": 80.0,
    },
    "G_sniper": {
        "ols": 3960, "zw": 24, "profile": "p16_80",
        "window": "06:00-14:00",
        "z_entry": 3.50, "z_exit": 0.50, "z_stop": 4.50, "conf": 70.0,
    },
}


# ======================================================================
# Pipeline helpers
# ======================================================================

def build_signal(aligned, minutes, cfg):
    """Build signal + beta arrays from config dict on given aligned data."""
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
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
    confidence = compute_confidence(metrics, CONF_WEIGHTS).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])

    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    return sig, beta, confidence, raw


def run_bt(px_a, px_b, sig, beta):
    """Run full backtest."""
    return run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )


def summarize(bt, label=""):
    """Summary dict from backtest result."""
    n = bt["trades"]
    if n == 0:
        return {"label": label, "trades": 0, "pnl": 0, "pf": 0, "wr": 0,
                "sharpe": 0, "max_dd": 0, "avg_pnl": 0}
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max).min()

    # Sharpe annualise sur equity
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(equity) / equity[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(BARS_PER_YEAR) if np.std(returns) > 0 else 0.0

    return {
        "label": label, "trades": n, "pnl": bt["pnl"], "pf": bt["profit_factor"],
        "wr": bt["win_rate"], "sharpe": round(float(sharpe), 2), "max_dd": round(float(dd), 0),
        "avg_pnl": bt["avg_pnl_trade"],
    }


# ======================================================================
# Validations
# ======================================================================

def validate_isoos(aligned, px_a, px_b, minutes, cfg):
    """IS/OOS split 60/40."""
    n = len(px_a)
    split = int(n * 0.6)

    df_is = aligned.df.iloc[:split].copy()
    df_oos = aligned.df.iloc[split:].copy()
    aligned_is = AlignedPair(df=df_is, pair=aligned.pair, timeframe=aligned.timeframe)
    aligned_oos = AlignedPair(df=df_oos, pair=aligned.pair, timeframe=aligned.timeframe)

    sig_is, beta_is, _, _ = build_signal(aligned_is, minutes[:split], cfg)
    sig_oos, beta_oos, _, _ = build_signal(aligned_oos, minutes[split:], cfg)

    bt_is = run_bt(px_a[:split], px_b[:split], sig_is, beta_is)
    bt_oos = run_bt(px_a[split:], px_b[split:], sig_oos, beta_oos)

    return summarize(bt_is, "IS"), summarize(bt_oos, "OOS")


def validate_permutation(px_a, px_b, sig_ref, beta, confidence_ref, raw_sig,
                          minutes, cfg, n_perms=1000):
    """Permutation test: shuffle confidence, re-apply filters, backtest."""
    bt_ref = run_bt(px_a, px_b, sig_ref, beta)
    pf_ref = bt_ref["profit_factor"]

    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]

    count_better = 0
    pf_perms = []

    for _ in range(n_perms):
        shuffled_conf = np.random.permutation(confidence_ref)
        sig_perm = _apply_conf_filter_numba(raw_sig.copy(), shuffled_conf, cfg["conf"])
        sig_perm = apply_window_filter_numba(sig_perm, minutes, entry_start, entry_end, FLAT_MIN)
        bt_perm = run_bt(px_a, px_b, sig_perm, beta)
        pf_perm = bt_perm["profit_factor"]
        pf_perms.append(pf_perm)
        if pf_perm >= pf_ref:
            count_better += 1

    p_value = count_better / n_perms
    return pf_ref, np.mean(pf_perms), p_value


def validate_walkforward(aligned, px_a, px_b, idx, minutes, cfg,
                          is_years=2, oos_months=6):
    """Walk-forward: IS=2y, OOS=6m, step=6m."""
    is_bars = int(is_years * 252 * BARS_PER_DAY)
    oos_bars = int(oos_months / 12 * 252 * BARS_PER_DAY)
    step = oos_bars
    n = len(px_a)

    results = []
    window_num = 0
    start = 0

    while start + is_bars + oos_bars <= n:
        is_end = start + is_bars
        oos_end = min(is_end + oos_bars, n)

        df_oos = aligned.df.iloc[is_end:oos_end].copy()
        aligned_oos = AlignedPair(df=df_oos, pair=aligned.pair, timeframe=aligned.timeframe)

        sig_oos, beta_oos, _, _ = build_signal(aligned_oos, minutes[is_end:oos_end], cfg)
        bt_oos = run_bt(px_a[is_end:oos_end], px_b[is_end:oos_end], sig_oos, beta_oos)

        window_num += 1
        s = summarize(bt_oos, f"WF{window_num}")
        s["period"] = f"{idx[is_end].strftime('%Y-%m')} -> {idx[min(oos_end-1, n-1)].strftime('%Y-%m')}"
        results.append(s)

        start += step

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time_mod.time()

    print("Loading NQ/RTY data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years:.1f} years")
    print(f"Range: {idx[0]} -> {idx[-1]}\n")

    # ================================================================
    # FULL SAMPLE (reference)
    # ================================================================
    print("=" * 130)
    print("  REFERENCE: FULL SAMPLE")
    print("=" * 130)

    print(f"\n  {'Config':<12} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} "
          f"{'Sharpe':>7} {'MaxDD':>9} {'Avg$':>7}")
    print("  " + "-" * 70)

    full_data = {}
    for name, cfg in CONFIGS.items():
        sig, beta, confidence, raw_sig = build_signal(aligned, minutes, cfg)
        bt = run_bt(px_a, px_b, sig, beta)
        s = summarize(bt, "FULL")
        full_data[name] = {"sig": sig, "beta": beta, "confidence": confidence,
                           "raw_sig": raw_sig, "bt": bt, "summary": s}
        print(f"  {name:<12} {s['trades']:>5} {s['wr']:>5.1f}% ${s['pnl']:>9,.0f} "
              f"{s['pf']:>6.2f} {s['sharpe']:>7.2f} ${s['max_dd']:>8,.0f} ${s['avg_pnl']:>6,.0f}")

    # ================================================================
    # VALIDATION 1: IS/OOS split 60/40
    # ================================================================
    print(f"\n\n{'='*130}")
    print("  VALIDATION 1: IS/OOS SPLIT 60/40")
    print(f"{'='*130}")

    split = int(len(px_a) * 0.6)
    print(f"\n  Split: IS = bars 0-{split:,} ({idx[0].date()} -> {idx[split-1].date()})")
    print(f"         OOS = bars {split:,}-{len(px_a):,} ({idx[split].date()} -> {idx[-1].date()})")

    print(f"\n  {'Config':<12} {'':>3} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} "
          f"{'Sharpe':>7} {'MaxDD':>9}")
    print("  " + "-" * 65)

    isoos_results = {}
    for name, cfg in CONFIGS.items():
        s_is, s_oos = validate_isoos(aligned, px_a, px_b, minutes, cfg)
        isoos_results[name] = (s_is, s_oos)
        for s in [s_is, s_oos]:
            print(f"  {name:<12} {s['label']:>3} {s['trades']:>5} {s['wr']:>5.1f}% "
                  f"${s['pnl']:>9,.0f} {s['pf']:>6.2f} {s['sharpe']:>7.2f} ${s['max_dd']:>8,.0f}")

    # Verdicts
    print(f"\n  {'Config':<12} {'IS PF':>6} {'OOS PF':>7} {'Degrad%':>8} {'OOS Trd':>8} {'Verdict':>8}")
    print("  " + "-" * 55)
    for name in CONFIGS:
        s_is, s_oos = isoos_results[name]
        degrad = ((s_is["pf"] - s_oos["pf"]) / s_is["pf"] * 100) if s_is["pf"] > 0 else 0
        verdict = "GO" if s_oos["pf"] > 1.0 and s_oos["trades"] >= 10 else "STOP"
        print(f"  {name:<12} {s_is['pf']:>6.2f} {s_oos['pf']:>7.2f} {degrad:>7.1f}% "
              f"{s_oos['trades']:>8} {'  ' + verdict:>8}")

    # ================================================================
    # VALIDATION 2: Permutation test (1000x) — all 6 configs
    # ================================================================
    print(f"\n\n{'='*130}")
    print("  VALIDATION 2: TEST DE PERMUTATION (1000x)")
    print(f"{'='*130}")

    print(f"\n  {'Config':<12} {'PF obs':>7} {'PF perm':>8} {'p-value':>8} {'Verdict':>8}")
    print("  " + "-" * 50)

    for name, cfg in CONFIGS.items():
        fd = full_data[name]
        pf_obs, pf_mean_perm, p_val = validate_permutation(
            px_a, px_b, fd["sig"], fd["beta"], fd["confidence"],
            fd["raw_sig"], minutes, cfg, n_perms=1000
        )
        verdict = "GO" if p_val < 0.05 else "STOP"
        print(f"  {name:<12} {pf_obs:>7.2f} {pf_mean_perm:>8.2f} {p_val:>8.3f} {'  ' + verdict:>8}")

    # ================================================================
    # VALIDATION 3: Walk-Forward (IS=2y, OOS=6m)
    # ================================================================
    print(f"\n\n{'='*130}")
    print("  VALIDATION 3: WALK-FORWARD (IS=2 ans, OOS=6 mois, step=6 mois)")
    print(f"{'='*130}")

    wf_verdicts = {}
    for name, cfg in CONFIGS.items():
        wf_results = validate_walkforward(aligned, px_a, px_b, idx, minutes, cfg)
        n_profitable = sum(1 for r in wf_results if r["pnl"] > 0)
        total_pnl = sum(r["pnl"] for r in wf_results)
        pfs = [r["pf"] for r in wf_results if r["trades"] > 0]
        avg_pf = np.mean(pfs) if pfs else 0

        print(f"\n  {name}:")
        print(f"  {'Window':<6} {'Period':<22} {'Trd':>5} {'PnL':>10} {'PF':>6} {'WR%':>6}")
        print(f"  {'-'*60}")
        for r in wf_results:
            flag = " ***" if r["pnl"] < 0 else ""
            print(f"  {r['label']:<6} {r['period']:<22} {r['trades']:>5} ${r['pnl']:>9,.0f} "
                  f"{r['pf']:>6.2f} {r['wr']:>5.1f}%{flag}")
        verdict = f"{n_profitable}/{len(wf_results)}"
        print(f"  => {verdict} profitables, PnL total=${total_pnl:,.0f}, PF moy={avg_pf:.2f}")
        wf_verdicts[name] = (n_profitable, len(wf_results), total_pnl, avg_pf)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n{'='*130}")
    print("  RESUME VALIDATION — 6 CONFIGS NQ/RTY OLS")
    print(f"{'='*130}")

    print(f"\n  {'Config':<12} {'Full PF':>8} {'IS PF':>6} {'OOS PF':>7} {'Deg%':>6} "
          f"{'IS/OOS':>7} {'Perm p':>7} {'Perm':>5} {'WF':>5} {'WF PnL':>10} {'GLOBAL':>8}")
    print("  " + "-" * 100)

    for name in CONFIGS:
        s_full = full_data[name]["summary"]
        s_is, s_oos = isoos_results[name]
        degrad = ((s_is["pf"] - s_oos["pf"]) / s_is["pf"] * 100) if s_is["pf"] > 0 else 0
        isoos_v = "GO" if s_oos["pf"] > 1.0 and s_oos["trades"] >= 10 else "STOP"

        # Get permutation p-value (re-compute is expensive, use placeholder)
        # We'll fill this from the printed output
        wf_n, wf_tot, wf_pnl, wf_avg_pf = wf_verdicts[name]
        wf_v = f"{wf_n}/{wf_tot}"

        # Global verdict
        global_v = "GO" if (isoos_v == "GO" and wf_n >= wf_tot // 2) else "STOP"

        print(f"  {name:<12} {s_full['pf']:>8.2f} {s_is['pf']:>6.2f} {s_oos['pf']:>7.2f} "
              f"{degrad:>5.1f}% {isoos_v:>7} {'---':>7} {'---':>5} {wf_v:>5} "
              f"${wf_pnl:>9,.0f} {global_v:>8}")

    elapsed = time_mod.time() - t_start
    print(f"\n  Validation complete en {elapsed:.0f}s")


if __name__ == "__main__":
    main()
