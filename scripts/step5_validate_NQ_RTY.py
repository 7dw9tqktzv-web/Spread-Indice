"""Etape 5 -- Validation complete NQ/RTY OLS (10 configs).

IS/OOS 60/40, Permutation 1000x, Walk-Forward 2y/6m.
Time stops recommandes de l'etape 4 appliques.

Usage:
    python scripts/step5_validate_NQ_RTY.py
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

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
MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
BARS_PER_DAY = 264
BARS_PER_YEAR = BARS_PER_DAY * 252

N_PERMS = 1000
IS_YEARS = 2
OOS_MONTHS = 6
WF_STEP_MONTHS = 6

# Time stops from step 4 (bars, 0=off)
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


def build_signal_on_subset(aligned_sub, cfg, ts_bars):
    """Build full signal chain on an AlignedPair subset."""
    idx = aligned_sub.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
    hr = est.estimate(aligned_sub)
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
    metrics = compute_all_metrics(spread, aligned_sub.df["close_a"],
                                  aligned_sub.df["close_b"], profile_cfg)
    confidence = compute_confidence(metrics, make_conf(cfg["conf"])).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])

    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    if ts_bars > 0:
        sig = apply_time_stop(sig.copy(), ts_bars)

    return sig, beta, confidence, raw


def run_bt(px_a, px_b, sig, beta):
    """Run backtest, return summary dict."""
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    max_dd = float((equity - running_max).min())

    return {
        "trades": bt["trades"],
        "wr": bt["win_rate"],
        "pnl": bt["pnl"],
        "pf": bt["profit_factor"],
        "max_dd": round(max_dd, 0),
        "equity": equity,
    }


def make_aligned_subset(aligned, start, end):
    """Create AlignedPair from index slice."""
    sub_df = aligned.df.iloc[start:end].copy()
    return AlignedPair(df=sub_df, pair=aligned.pair, timeframe=aligned.timeframe)


# ==================================================================
# Validation functions
# ==================================================================
def validate_isoos(aligned, cfg, ts_bars):
    """IS/OOS 60/40 split. Rebuild signals on each subset."""
    n = len(aligned.df)
    split = int(n * 0.60)

    aligned_is = make_aligned_subset(aligned, 0, split)
    aligned_oos = make_aligned_subset(aligned, split, n)

    sig_is, beta_is, _, _ = build_signal_on_subset(aligned_is, cfg, ts_bars)
    sig_oos, beta_oos, _, _ = build_signal_on_subset(aligned_oos, cfg, ts_bars)

    px_a_is = aligned_is.df["close_a"].values
    px_b_is = aligned_is.df["close_b"].values
    px_a_oos = aligned_oos.df["close_a"].values
    px_b_oos = aligned_oos.df["close_b"].values

    bt_is = run_bt(px_a_is, px_b_is, sig_is, beta_is)
    bt_oos = run_bt(px_a_oos, px_b_oos, sig_oos, beta_oos)

    verdict = "GO" if bt_oos["pf"] > 1.0 and bt_oos["trades"] >= 10 else "STOP"
    return bt_is, bt_oos, verdict


def validate_permutation(aligned, cfg, ts_bars, n_perms=N_PERMS):
    """Permutation test: shuffle confidence, re-filter, count PF >= observed."""
    sig_full, beta_full, conf_full, raw_full = build_signal_on_subset(aligned, cfg, ts_bars)
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]

    bt_ref = run_bt(px_a, px_b, sig_full, beta_full)
    ref_pf = bt_ref["pf"]

    count_ge = 0
    rng = np.random.default_rng(42)

    for _ in range(n_perms):
        conf_shuf = rng.permutation(conf_full)
        sig_perm = _apply_conf_filter_numba(raw_full.copy(), conf_shuf, cfg["conf"])
        sig_perm = apply_window_filter_numba(sig_perm, minutes, entry_start, entry_end, FLAT_MIN)
        if ts_bars > 0:
            sig_perm = apply_time_stop(sig_perm.copy(), ts_bars)

        bt_perm = run_bt(px_a, px_b, sig_perm, beta_full)
        if bt_perm["pf"] >= ref_pf:
            count_ge += 1

    p_val = count_ge / n_perms
    verdict = "GO" if p_val < 0.05 else "STOP"
    return ref_pf, p_val, verdict


def validate_walkforward(aligned, cfg, ts_bars):
    """Walk-forward: IS=2y, OOS=6m, step=6m."""
    n = len(aligned.df)
    is_bars = IS_YEARS * 252 * BARS_PER_DAY
    oos_bars = OOS_MONTHS * 21 * BARS_PER_DAY
    step_bars = WF_STEP_MONTHS * 21 * BARS_PER_DAY

    windows = []
    start = 0
    while start + is_bars + oos_bars <= n:
        oos_start = start + is_bars
        oos_end = min(oos_start + oos_bars, n)
        windows.append((oos_start, oos_end))
        start += step_bars

    if not windows:
        return [], 0, 0, "STOP"

    results = []
    for oos_s, oos_e in windows:
        aligned_oos = make_aligned_subset(aligned, oos_s, oos_e)
        sig_oos, beta_oos, _, _ = build_signal_on_subset(aligned_oos, cfg, ts_bars)

        px_a = aligned_oos.df["close_a"].values
        px_b = aligned_oos.df["close_b"].values
        bt = run_bt(px_a, px_b, sig_oos, beta_oos)

        idx_oos = aligned_oos.df.index
        period = f"{idx_oos[0].strftime('%Y-%m')} -> {idx_oos[-1].strftime('%Y-%m')}"

        results.append({
            "period": period,
            "trades": bt["trades"],
            "pnl": bt["pnl"],
            "pf": bt["pf"],
            "wr": bt["wr"],
            "max_dd": bt["max_dd"],
        })

    profitable = sum(1 for r in results if r["pnl"] > 0)
    total = len(results)
    verdict = "GO" if profitable >= total // 2 else "STOP"
    return results, profitable, total, verdict


def compute_yearly(equity, dates):
    """Yearly PnL from equity curve."""
    years_set = sorted(set(d.year for d in dates))
    yearly = {}
    for y in years_set:
        mask = np.array([d.year == y for d in dates])
        if mask.sum() < 10:
            continue
        eq = equity[mask]
        yearly[y] = float(eq[-1] - eq[0])
    return yearly


# ==================================================================
# Main
# ==================================================================
def main():
    t_start = time_mod.time()

    print("=" * 130)
    print("  ETAPE 5 -- VALIDATION COMPLETE NQ/RTY OLS (10 configs)")
    print("  IS/OOS 60/40 | Permutation 1000x | Walk-Forward 2y/6m")
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
    dates = idx.to_pydatetime()
    n_bars = len(px_a)
    split_60 = int(n_bars * 0.60)
    print(f"Data: {n_bars:,} bars, IS={split_60:,} ({idx[0].date()} -> {idx[split_60-1].date()}), "
          f"OOS={n_bars - split_60:,} ({idx[split_60].date()} -> {idx[-1].date()})")

    # Load configs
    top10_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step3_top10.csv"
    configs_df = pd.read_csv(top10_path)
    print(f"Configs: {len(configs_df)}")

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
    # Run validation for each config
    # ==================================================================
    all_results = []

    for i, cfg in enumerate(configs, 1):
        label = cfg["label"]
        tier = cfg["tier"]
        ts = cfg["ts_bars"]
        ts_lab = f"{ts*5}min" if ts > 0 else "off"

        print(f"\n\n{'='*130}")
        print(f"  CONFIG {i}/10: {label} [{tier}] (time_stop={ts_lab})")
        print(f"{'='*130}")

        # --- Full sample ---
        print("\n  [1/4] Full sample backtest...")
        sig_full, beta_full, _, _ = build_signal_on_subset(aligned, cfg, ts)
        bt_full = run_bt(px_a, px_b, sig_full, beta_full)
        yearly = compute_yearly(bt_full["equity"], dates)
        neg_years = [y for y, p in yearly.items() if p < 0]

        print(f"        Trades={bt_full['trades']}, WR={bt_full['wr']:.1f}%, "
              f"PnL=${bt_full['pnl']:,.0f}, PF={bt_full['pf']:.2f}, MaxDD=${bt_full['max_dd']:,.0f}")
        for y in sorted(yearly):
            flag = " *" if yearly[y] < 0 else ""
            print(f"          {y}: ${yearly[y]:>+9,.0f}{flag}")

        # --- IS/OOS ---
        print("\n  [2/4] IS/OOS 60/40...")
        bt_is, bt_oos, isoos_v = validate_isoos(aligned, cfg, ts)
        degrad = ((bt_oos["pf"] - bt_is["pf"]) / bt_is["pf"] * 100
                  if bt_is["pf"] > 0 else 0)
        print(f"        IS:  Trades={bt_is['trades']}, PF={bt_is['pf']:.2f}, "
              f"PnL=${bt_is['pnl']:,.0f}, MaxDD=${bt_is['max_dd']:,.0f}")
        print(f"        OOS: Trades={bt_oos['trades']}, PF={bt_oos['pf']:.2f}, "
              f"PnL=${bt_oos['pnl']:,.0f}, MaxDD=${bt_oos['max_dd']:,.0f}")
        print(f"        Degradation: {degrad:+.1f}%  ->  IS/OOS = {isoos_v}")

        # --- Permutation ---
        print("\n  [3/4] Permutation test (1000x)...")
        t_perm = time_mod.time()
        ref_pf, p_val, perm_v = validate_permutation(aligned, cfg, ts)
        perm_time = time_mod.time() - t_perm
        print(f"        Ref PF={ref_pf:.2f}, p-value={p_val:.3f} ({perm_time:.0f}s)  "
              f"->  Permutation = {perm_v}")

        # --- Walk-Forward ---
        print(f"\n  [4/4] Walk-Forward (IS={IS_YEARS}y, OOS={OOS_MONTHS}m, step={WF_STEP_MONTHS}m)...")
        wf_results, wf_profit, wf_total, wf_v = validate_walkforward(aligned, cfg, ts)

        for wr in wf_results:
            pf_str = f"{wr['pf']:.2f}" if wr['trades'] > 0 else "N/A"
            flag = " +" if wr["pnl"] > 0 else " -"
            print(f"        {wr['period']}: {wr['trades']:>3} trades, "
                  f"PF={pf_str:>5}, PnL=${wr['pnl']:>+8,.0f}, MaxDD=${wr['max_dd']:>7,.0f}{flag}")
        print(f"        Profitable: {wf_profit}/{wf_total}  ->  WF = {wf_v}")

        # --- Global verdict ---
        global_v = "GO" if (isoos_v == "GO" and perm_v == "GO"
                            and wf_profit >= wf_total // 2) else "STOP"
        print(f"\n  >>> VERDICT GLOBAL: {global_v} "
              f"(IS/OOS={isoos_v}, Perm={perm_v} p={p_val:.3f}, WF={wf_profit}/{wf_total}={wf_v})")

        all_results.append({
            "label": label,
            "tier": tier,
            "ts_bars": ts,
            "full_trades": bt_full["trades"],
            "full_wr": bt_full["wr"],
            "full_pnl": bt_full["pnl"],
            "full_pf": bt_full["pf"],
            "full_maxdd": bt_full["max_dd"],
            "is_trades": bt_is["trades"],
            "is_pf": bt_is["pf"],
            "is_pnl": bt_is["pnl"],
            "oos_trades": bt_oos["trades"],
            "oos_pf": bt_oos["pf"],
            "oos_pnl": bt_oos["pnl"],
            "oos_maxdd": bt_oos["max_dd"],
            "degradation_pct": round(degrad, 1),
            "isoos_verdict": isoos_v,
            "perm_pval": p_val,
            "perm_verdict": perm_v,
            "wf_profitable": wf_profit,
            "wf_total": wf_total,
            "wf_verdict": wf_v,
            "global_verdict": global_v,
            "neg_years": len(neg_years),
            "neg_years_list": str(neg_years),
        })

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    df_all = pd.DataFrame(all_results)

    print(f"\n\n{'='*130}")
    print("  TABLEAU RECAPITULATIF -- VALIDATION 10 CONFIGS NQ/RTY")
    print(f"{'='*130}")

    print(f"\n  {'#':>2} {'Label':<42} {'Tier':<6} {'TS':>4} "
          f"{'Full PF':>7} {'IS PF':>6} {'OOS PF':>6} {'Deg%':>5} {'IS/OOS':>6} "
          f"{'p-val':>6} {'Perm':>5} {'WF':>5} {'GLOBAL':>7}")
    print(f"  {'-'*125}")

    go_count = 0
    for _, r in df_all.iterrows():
        ts_lab = f"{r['ts_bars']*5}m" if r["ts_bars"] > 0 else "off"
        wf_str = f"{r['wf_profitable']}/{r['wf_total']}"
        g = r["global_verdict"]
        marker = " <<<" if g == "GO" else ""
        if g == "GO":
            go_count += 1
        print(f"  {_+1:>2} {r['label']:<42} {r['tier']:<6} {ts_lab:>4} "
              f"{r['full_pf']:>7.2f} {r['is_pf']:>6.2f} {r['oos_pf']:>6.2f} "
              f"{r['degradation_pct']:>+5.1f} {r['isoos_verdict']:>6} "
              f"{r['perm_pval']:>6.3f} {r['perm_verdict']:>5} {wf_str:>5} "
              f"{g:>7}{marker}")

    print(f"\n  TOTAL GO: {go_count}/10")

    # Tier breakdown
    print("\n  --- Par tier ---")
    for tier in ["SAFE", "WARN", "DANGER"]:
        tier_df = df_all[df_all["tier"] == tier]
        if len(tier_df) == 0:
            continue
        go_t = (tier_df["global_verdict"] == "GO").sum()
        print(f"  {tier:>7}: {go_t}/{len(tier_df)} GO")

    # OOS focus
    print("\n  --- Focus OOS ---")
    print(f"  {'#':>2} {'Label':<42} {'OOS Trd':>7} {'OOS PF':>6} {'OOS PnL':>9} {'OOS DD':>8}")
    print(f"  {'-'*85}")
    for _, r in df_all.iterrows():
        print(f"  {_+1:>2} {r['label']:<42} {r['oos_trades']:>7} {r['oos_pf']:>6.2f} "
              f"${r['oos_pnl']:>8,.0f} ${r['oos_maxdd']:>7,.0f}")

    # Save
    out_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step5_validation.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n  Sauvegarde: {out_path}")

    elapsed = time_mod.time() - t_start
    print(f"\n  Etape 5 complete en {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
