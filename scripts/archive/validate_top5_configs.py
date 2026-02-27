"""Validate top 5 configs from refined grid: IS/OOS, permutation, walk-forward.

Also tests time stop refinement (6-10 bars) and hourly entry filter.

Usage:
    python scripts/validate_top5_configs.py
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np

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
# Constants
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
METRICS_CFG = MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6)
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930

WINDOWS_MAP = {
    "02:00-15:00": (120, 900), "02:00-14:00": (120, 840),
    "03:00-14:00": (180, 840), "03:00-13:00": (180, 780),
    "04:00-14:00": (240, 840), "04:00-13:00": (240, 780),
}

# Top 5 configs
CONFIGS = {
    "A_Balanced": {
        "ols": 2640, "zw": 30, "z_entry": 3.15, "z_exit": 1.20,
        "z_stop": 4.50, "conf": 70.0, "ts": 18, "window": "03:00-14:00",
    },
    "B_Volume_PnL": {
        "ols": 2640, "zw": 48, "z_entry": 3.15, "z_exit": 1.00,
        "z_stop": 4.25, "conf": 68.0, "ts": 36, "window": "04:00-14:00",
    },
    "C_Max_Volume": {
        "ols": 2970, "zw": 42, "z_entry": 2.95, "z_exit": 1.25,
        "z_stop": 4.00, "conf": 67.0, "ts": 0, "window": "02:00-15:00",
    },
    "D_Sharpe_Alt": {
        "ols": 2640, "zw": 30, "z_entry": 3.05, "z_exit": 1.20,
        "z_stop": 4.50, "conf": 70.0, "ts": 18, "window": "04:00-13:00",
    },
    "E_PnL_PF": {
        "ols": 3300, "zw": 30, "z_entry": 3.15, "z_exit": 1.00,
        "z_stop": 4.50, "conf": 67.0, "ts": 0, "window": "02:00-14:00",
    },
}


# ======================================================================
# Pipeline helper
# ======================================================================

def build_signal(aligned, px_a, idx, minutes, cfg, ts_override=None, hour_filter=None):
    """Build signal array from config dict."""
    est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    mu = spread.rolling(cfg["zw"]).mean()
    sigma = spread.rolling(cfg["zw"]).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], METRICS_CFG)
    confidence = compute_confidence(metrics, CONF_CFG).values

    ts = ts_override if ts_override is not None else cfg["ts"]
    sig_pre_conf = apply_time_stop(raw, ts)
    sig = _apply_conf_filter_numba(sig_pre_conf, confidence, cfg["conf"])

    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    # Optional hourly filter: block entries outside [hour_start, hour_end) CT
    if hour_filter is not None:
        h_start, h_end = hour_filter
        hours = idx.hour
        prev = np.int8(0)
        for t in range(len(sig)):
            if sig[t] != 0 and prev == 0:
                h = hours[t]
                if h < h_start or h >= h_end:
                    sig[t] = 0
            prev = sig[t]

    return sig, beta, confidence, sig_pre_conf


def run_bt(px_a, px_b, sig, beta):
    """Run full backtest and return result dict."""
    return run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )


def summarize(bt, label=""):
    """Return summary dict from backtest result."""
    n = bt["trades"]
    if n == 0:
        return {"label": label, "trades": 0, "pnl": 0, "pf": 0, "wr": 0, "sharpe": 0, "max_dd": 0}
    pnls = bt["trade_pnls"]
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max).min()
    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(n)) if n > 1 and pnls.std() > 0 else 0
    return {
        "label": label, "trades": n, "pnl": bt["pnl"], "pf": bt["profit_factor"],
        "wr": bt["win_rate"], "sharpe": round(sharpe, 2), "max_dd": round(dd, 0),
        "avg_pnl": bt["avg_pnl_trade"], "avg_dur": bt["avg_duration_bars"],
    }


# ======================================================================
# Validations
# ======================================================================

def validate_isoos(aligned, px_a, px_b, idx, minutes, cfg, name):
    """IS/OOS split 60/40."""
    n = len(px_a)
    split = int(n * 0.6)

    # Create sub-aligned objects by slicing
    from src.data.alignment import AlignedPair
    df_is = aligned.df.iloc[:split].copy()
    df_oos = aligned.df.iloc[split:].copy()
    aligned_is = AlignedPair(df=df_is, pair=aligned.pair, timeframe=aligned.timeframe)
    aligned_oos = AlignedPair(df=df_oos, pair=aligned.pair, timeframe=aligned.timeframe)

    sig_is, beta_is, _, _ = build_signal(aligned_is, px_a[:split], idx[:split], minutes[:split], cfg)
    sig_oos, beta_oos, _, _ = build_signal(aligned_oos, px_a[split:], idx[split:], minutes[split:], cfg)

    bt_is = run_bt(px_a[:split], px_b[:split], sig_is, beta_is)
    bt_oos = run_bt(px_a[split:], px_b[split:], sig_oos, beta_oos)

    s_is = summarize(bt_is, "IS")
    s_oos = summarize(bt_oos, "OOS")

    return s_is, s_oos


def validate_permutation(px_a, px_b, sig_ref, beta, confidence_ref, cfg,
                          sig_pre_conf, minutes, n_perms=1000):
    """Permutation test: shuffle confidence, re-apply conf+window filter, backtest.

    Key: we shuffle confidence on the PRE-confidence signal (after time_stop but
    before conf filter), then re-apply conf filter + window filter.
    This tests whether the temporal alignment of confidence matters.
    """
    bt_ref = run_bt(px_a, px_b, sig_ref, beta)
    pf_ref = bt_ref["profit_factor"]

    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]

    count_better = 0
    pf_perms = []

    for _ in range(n_perms):
        shuffled_conf = np.random.permutation(confidence_ref)
        sig_perm = _apply_conf_filter_numba(sig_pre_conf.copy(), shuffled_conf, cfg["conf"])
        sig_perm = apply_window_filter_numba(sig_perm, minutes, entry_start, entry_end, FLAT_MIN)
        bt_perm = run_bt(px_a, px_b, sig_perm, beta)
        pf_perm = bt_perm["profit_factor"]
        pf_perms.append(pf_perm)
        if pf_perm >= pf_ref:
            count_better += 1

    p_value = count_better / n_perms
    return pf_ref, np.mean(pf_perms), p_value


def validate_walkforward(aligned, px_a, px_b, idx, minutes, cfg, is_years=2, oos_months=6):
    """Walk-forward: IS=2y, OOS=6m, step=6m."""
    from src.data.alignment import AlignedPair

    bars_per_day = 264
    is_bars = int(is_years * 252 * bars_per_day)
    oos_bars = int(oos_months / 12 * 252 * bars_per_day)
    step = oos_bars
    n = len(px_a)

    results = []
    window_num = 0

    start = 0
    while start + is_bars + oos_bars <= n:
        is_end = start + is_bars
        oos_end = min(is_end + oos_bars, n)

        # OOS only (params are fixed, no re-optimization)
        df_oos = aligned.df.iloc[is_end:oos_end].copy()
        aligned_oos = AlignedPair(df=df_oos, pair=aligned.pair, timeframe=aligned.timeframe)

        sig_oos, beta_oos, _, _ = build_signal(
            aligned_oos, px_a[is_end:oos_end], idx[is_end:oos_end],
            minutes[is_end:oos_end], cfg
        )
        bt_oos = run_bt(px_a[is_end:oos_end], px_b[is_end:oos_end], sig_oos, beta_oos)

        window_num += 1
        s = summarize(bt_oos, f"WF{window_num}")
        s["period"] = f"{idx[is_end].strftime('%Y-%m')} -> {idx[min(oos_end-1, n-1)].strftime('%Y-%m')}"
        results.append(s)

        start += step

    return results


# ======================================================================
# Time stop refinement + hourly filter test
# ======================================================================

def test_time_stop_refinement(aligned, px_a, px_b, idx, minutes, cfg, name):
    """Test time stops: 4, 6, 8, 10, 12, 14, 18, 24 bars."""
    results = []
    for ts in [4, 6, 8, 10, 12, 14, 18, 24, 36, 0]:
        sig, beta, _, _ = build_signal(aligned, px_a, idx, minutes, cfg, ts_override=ts)
        bt = run_bt(px_a, px_b, sig, beta)
        s = summarize(bt, f"ts={ts}")
        s["ts"] = ts
        results.append(s)
    return results


def test_hourly_filter(aligned, px_a, px_b, idx, minutes, cfg, name):
    """Test hourly entry filters."""
    filters = [
        ("No filter", None),
        ("7h-13h", (7, 13)),
        ("7h-12h", (7, 12)),
        ("7h-11h", (7, 11)),
        ("8h-12h", (8, 12)),
        ("8h-13h", (8, 13)),
        ("8h-11h", (8, 11)),
    ]
    results = []
    for label, hf in filters:
        sig, beta, _, _ = build_signal(aligned, px_a, idx, minutes, cfg, hour_filter=hf)
        bt = run_bt(px_a, px_b, sig, beta)
        s = summarize(bt, label)
        results.append(s)
    return results


# ======================================================================
# Main
# ======================================================================

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

    # ================================================================
    # VALIDATION 1: IS/OOS split 60/40
    # ================================================================
    print("=" * 120)
    print(" VALIDATION 1: IS/OOS SPLIT 60/40")
    print("=" * 120)

    print(f"\n  {'Config':<16} {'':>3} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'-'*65}")

    isoos_results = {}
    for name, cfg in CONFIGS.items():
        s_is, s_oos = validate_isoos(aligned, px_a, px_b, idx, minutes, cfg, name)
        isoos_results[name] = (s_is, s_oos)
        for s in [s_is, s_oos]:
            print(f"  {name:<16} {s['label']:>3} {s['trades']:>5} {s['wr']:>5.1f}% ${s['pnl']:>9,.0f} "
                  f"{s['pf']:>6.2f} {s['sharpe']:>7.2f} ${s['max_dd']:>7,.0f}")

    # Verdicts
    print(f"\n  {'Config':<16} {'IS PF':>6} {'OOS PF':>7} {'Degrad%':>8} {'Verdict':>8}")
    print(f"  {'-'*50}")
    for name in CONFIGS:
        s_is, s_oos = isoos_results[name]
        degrad = ((s_is["pf"] - s_oos["pf"]) / s_is["pf"] * 100) if s_is["pf"] > 0 else 0
        verdict = "GO" if s_oos["pf"] > 1.0 and s_oos["trades"] >= 10 else "STOP"
        print(f"  {name:<16} {s_is['pf']:>6.2f} {s_oos['pf']:>7.2f} {degrad:>7.1f}% {'  ' + verdict:>8}")

    # ================================================================
    # VALIDATION 2: Permutation test (1000x) — config A only (representative)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" VALIDATION 2: TEST DE PERMUTATION (1000x) — Config A")
    print("=" * 120)

    cfg_a = CONFIGS["A_Balanced"]
    sig_a, beta_a, conf_a, sig_pre_conf_a = build_signal(aligned, px_a, idx, minutes, cfg_a)
    pf_obs, pf_mean_perm, p_val = validate_permutation(
        px_a, px_b, sig_a, beta_a, conf_a, cfg_a,
        sig_pre_conf_a, minutes, n_perms=1000
    )
    print(f"\n  PF observe:        {pf_obs:.2f}")
    print(f"  PF moyen permut:   {pf_mean_perm:.2f}")
    print(f"  p-value:           {p_val:.3f}")
    print(f"  Verdict:           {'GO' if p_val < 0.05 else 'STOP'} (seuil 0.05)")

    # ================================================================
    # VALIDATION 3: Walk-Forward (6 fenetres)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" VALIDATION 3: WALK-FORWARD (IS=2 ans, OOS=6 mois)")
    print("=" * 120)

    for name, cfg in CONFIGS.items():
        wf_results = validate_walkforward(aligned, px_a, px_b, idx, minutes, cfg)
        n_profitable = sum(1 for r in wf_results if r["pnl"] > 0)
        total_pnl = sum(r["pnl"] for r in wf_results)
        avg_pf = np.mean([r["pf"] for r in wf_results if r["trades"] > 0])

        print(f"\n  {name}:")
        print(f"  {'Window':<6} {'Period':<22} {'Trd':>5} {'PnL':>10} {'PF':>6} {'WR%':>6}")
        print(f"  {'-'*60}")
        for r in wf_results:
            flag = " ***" if r["pnl"] < 0 else ""
            print(f"  {r['label']:<6} {r['period']:<22} {r['trades']:>5} ${r['pnl']:>9,.0f} "
                  f"{r['pf']:>6.2f} {r['wr']:>5.1f}%{flag}")
        print(f"  => {n_profitable}/{len(wf_results)} profitables, PnL total=${total_pnl:,.0f}, PF moy={avg_pf:.2f}")

    # ================================================================
    # TEST 4: Time Stop Refinement (config A)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" TEST 4: TIME STOP REFINEMENT — Config A")
    print("=" * 120)

    ts_results = test_time_stop_refinement(aligned, px_a, px_b, idx, minutes, cfg_a, "A_Balanced")
    print(f"\n  {'TS':>4} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'Avg$':>7} {'AvgDur':>6}")
    print(f"  {'-'*55}")
    for r in ts_results:
        ts_label = f"{r['ts']}b" if r['ts'] > 0 else "none"
        print(f"  {ts_label:>4} {r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} "
              f"{r['pf']:>6.2f} {r['sharpe']:>7.2f} ${r.get('avg_pnl', 0):>6,.0f} {r.get('avg_dur', 0):>5.1f}")

    # ================================================================
    # TEST 5: Hourly Entry Filter (config A)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" TEST 5: FILTRE HORAIRE D'ENTREE — Config A")
    print("=" * 120)

    hf_results = test_hourly_filter(aligned, px_a, px_b, idx, minutes, cfg_a, "A_Balanced")
    print(f"\n  {'Filter':<12} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'Avg$':>7}")
    print(f"  {'-'*55}")
    for r in hf_results:
        print(f"  {r['label']:<12} {r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} "
              f"{r['pf']:>6.2f} {r['sharpe']:>7.2f} ${r.get('avg_pnl', 0):>6,.0f}")

    # Also test on config C (max volume) since volume is the concern
    print("\n  --- Aussi sur Config C (Max Volume) ---")
    cfg_c = CONFIGS["C_Max_Volume"]
    hf_results_c = test_hourly_filter(aligned, px_a, px_b, idx, minutes, cfg_c, "C_Max_Volume")
    print(f"\n  {'Filter':<12} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'Avg$':>7}")
    print(f"  {'-'*55}")
    for r in hf_results_c:
        print(f"  {r['label']:<12} {r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} "
              f"{r['pf']:>6.2f} {r['sharpe']:>7.2f} ${r.get('avg_pnl', 0):>6,.0f}")

    # Also test time stop on config C
    print("\n\n  --- Time Stop Refinement — Config C ---")
    ts_results_c = test_time_stop_refinement(aligned, px_a, px_b, idx, minutes, cfg_c, "C_Max_Volume")
    print(f"\n  {'TS':>4} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'Avg$':>7} {'AvgDur':>6}")
    print(f"  {'-'*55}")
    for r in ts_results_c:
        ts_label = f"{r['ts']}b" if r['ts'] > 0 else "none"
        print(f"  {ts_label:>4} {r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} "
              f"{r['pf']:>6.2f} {r['sharpe']:>7.2f} ${r.get('avg_pnl', 0):>6,.0f} {r.get('avg_dur', 0):>5.1f}")

    elapsed = time_mod.time() - t_start
    print(f"\n\n{'=' * 120}")
    print(f" VALIDATION COMPLETE en {elapsed:.0f}s")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
