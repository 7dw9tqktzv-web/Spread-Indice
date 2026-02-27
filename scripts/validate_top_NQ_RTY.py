"""Validate top Kalman configs for NQ_RTY: IS/OOS, permutation, walk-forward + long/short analysis.

Top configs selected from grid search (856,800 combos).
OLS was tested and found non-viable for NQ/RTY.

Usage:
    python scripts/validate_top_NQ_RTY.py
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.config.instruments import get_pair_specs
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
# Constants — NQ/RTY
# ======================================================================

_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930  # 15:30 CT

METRICS_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "moyen":      MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "03:00-12:00": (180, 720),
    "04:00-12:00": (240, 720),
    "04:00-13:00": (240, 780),
    "05:00-12:00": (300, 720),
}

# ======================================================================
# Top configs from grid search — TO BE FILLED after grid completes
# ======================================================================

CONFIGS = {
    "K_Balanced": {
        "alpha": 1.5e-7, "profil": "moyen",
        "z_entry": 1.8125, "z_exit": 0.125, "z_stop": 3.25,
        "conf": 55.0, "window": "05:00-12:00",
        "desc": "PnL*PF champion, 171 trd, PF 1.75",
    },
    "K_Quality": {
        "alpha": 2.5e-7, "profil": "tres_court",
        "z_entry": 1.375, "z_exit": 1.25, "z_stop": 3.25,
        "conf": 70.0, "window": "03:00-12:00",
        "desc": "Max PF 3.10, 83 trd, WR 74.7%",
    },
    "K_Volume": {
        "alpha": 2.5e-7, "profil": "court",
        "z_entry": 1.3125, "z_exit": 0.25, "z_stop": 3.25,
        "conf": 50.0, "window": "05:00-12:00",
        "desc": "Max trades 355, PF 1.32, $88k",
    },
    "K_Sniper": {
        "alpha": 2.5e-7, "profil": "moyen",
        "z_entry": 1.875, "z_exit": 0.25, "z_stop": 3.25,
        "conf": 70.0, "window": "04:00-13:00",
        "desc": "Max avg $975/trade, 53 trd, PF 2.96",
    },
    "K_PropFirm": {
        "alpha": 2.5e-7, "profil": "tres_court",
        "z_entry": 1.5625, "z_exit": 1.5, "z_stop": 2.25,
        "conf": 60.0, "window": "05:00-12:00",
        "desc": "WR 90.2%, PF 4.54, 61 trd",
    },
}


# ======================================================================
# Pipeline helper
# ======================================================================

def build_signal_kalman(aligned, idx, minutes, cfg):
    """Build signal array from Kalman config dict.

    Returns (sig, beta, confidence, sig_pre_conf).
    """
    est = create_estimator("kalman", alpha_ratio=cfg["alpha"],
                           warmup=200, gap_P_multiplier=5.0)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    zscore = hr.zscore.values.copy()
    zscore = np.ascontiguousarray(
        np.nan_to_num(zscore, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float64,
    )

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    metrics_cfg = METRICS_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(
        hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg
    )
    confidence = compute_confidence(metrics, CONF_CFG).values

    sig_pre_conf = raw.copy()
    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])

    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

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
        return {"label": label, "trades": 0, "pnl": 0, "pf": 0, "wr": 0,
                "sharpe": 0, "max_dd": 0, "avg_pnl": 0, "avg_dur": 0,
                "calmar": 0, "long_pnl": 0, "short_pnl": 0}
    pnls = bt["trade_pnls"]
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max).min()
    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(n)) if n > 1 and pnls.std() > 0 else 0
    calmar = float(bt["pnl"] / abs(dd)) if dd < 0 else 0

    # Long/short PnL
    trade_sides = bt.get("trade_sides", None)
    long_pnl = short_pnl = 0.0
    if trade_sides is not None:
        long_mask = trade_sides == 1
        short_mask = trade_sides == -1
        long_pnl = float(pnls[long_mask].sum()) if long_mask.any() else 0
        short_pnl = float(pnls[short_mask].sum()) if short_mask.any() else 0

    return {
        "label": label, "trades": n, "pnl": bt["pnl"], "pf": bt["profit_factor"],
        "wr": bt["win_rate"], "sharpe": round(sharpe, 2), "max_dd": round(dd, 0),
        "avg_pnl": bt["avg_pnl_trade"], "avg_dur": bt["avg_duration_bars"],
        "calmar": round(calmar, 2),
        "long_pnl": long_pnl, "short_pnl": short_pnl,
    }


# ======================================================================
# Validations
# ======================================================================

def validate_full(aligned, px_a, px_b, idx, minutes, cfg, name):
    """Full-sample backtest."""
    sig, beta, _, _ = build_signal_kalman(aligned, idx, minutes, cfg)
    bt = run_bt(px_a, px_b, sig, beta)
    return summarize(bt, name)


def validate_isoos(aligned, px_a, px_b, idx, minutes, cfg, name):
    """IS/OOS split 60/40."""
    n = len(px_a)
    split = int(n * 0.6)

    df_is = aligned.df.iloc[:split].copy()
    df_oos = aligned.df.iloc[split:].copy()
    aligned_is = AlignedPair(df=df_is, pair=aligned.pair, timeframe=aligned.timeframe)
    aligned_oos = AlignedPair(df=df_oos, pair=aligned.pair, timeframe=aligned.timeframe)

    sig_is, beta_is, _, _ = build_signal_kalman(aligned_is, idx[:split], minutes[:split], cfg)
    sig_oos, beta_oos, _, _ = build_signal_kalman(aligned_oos, idx[split:], minutes[split:], cfg)

    bt_is = run_bt(px_a[:split], px_b[:split], sig_is, beta_is)
    bt_oos = run_bt(px_a[split:], px_b[split:], sig_oos, beta_oos)

    return summarize(bt_is, "IS"), summarize(bt_oos, "OOS")


def validate_permutation(px_a, px_b, sig_ref, beta, confidence_ref, cfg,
                          sig_pre_conf, minutes, n_perms=1000):
    """Permutation test: shuffle confidence, re-apply conf+window filter, backtest."""
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

        df_oos = aligned.df.iloc[is_end:oos_end].copy()
        aligned_oos = AlignedPair(df=df_oos, pair=aligned.pair, timeframe=aligned.timeframe)

        sig_oos, beta_oos, _, _ = build_signal_kalman(
            aligned_oos, idx[is_end:oos_end], minutes[is_end:oos_end], cfg
        )
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
    np.random.seed(42)

    print("Loading NQ_RTY data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years:.1f} years")
    print(f"Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}\n")

    # ================================================================
    # SECTION 0: Full-sample verification + Long/Short analysis
    # ================================================================
    print("=" * 130)
    print(" SECTION 0: FULL-SAMPLE VERIFICATION + LONG/SHORT ANALYSIS")
    print("=" * 130)
    print(f"\n  {'Config':<14} {'Desc':<40} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} "
          f"{'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} | {'Long$':>9} {'Short$':>9} {'L%':>5}")
    print(f"  {'-'*135}")

    for name, cfg in CONFIGS.items():
        s = validate_full(aligned, px_a, px_b, idx, minutes, cfg, name)
        total_abs = abs(s["long_pnl"]) + abs(s["short_pnl"])
        l_pct = (s["long_pnl"] / total_abs * 100) if total_abs > 0 else 50
        bias = " **BIAS" if abs(l_pct) > 80 else (" *bias" if abs(l_pct) > 60 else "")
        print(f"  {name:<14} {cfg['desc'][:40]:<40} {s['trades']:>5} {s['wr']:>5.1f}% ${s['pnl']:>9,.0f} "
              f"{s['pf']:>6.2f} {s['sharpe']:>7.2f} ${s['max_dd']:>7,.0f} {s['calmar']:>7.2f} | "
              f"${s['long_pnl']:>8,.0f} ${s['short_pnl']:>8,.0f} {l_pct:>4.0f}%{bias}")

    # ================================================================
    # SECTION 1: IS/OOS split 60/40
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print(" SECTION 1: IS/OOS SPLIT 60/40")
    print("=" * 130)

    split_idx = int(len(px_a) * 0.6)
    print(f"  IS: {idx[0].strftime('%Y-%m-%d')} to {idx[split_idx-1].strftime('%Y-%m-%d')}")
    print(f"  OOS: {idx[split_idx].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")

    print(f"\n  {'Config':<14} {'':>3} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'-'*65}")

    isoos_results = {}
    for name, cfg in CONFIGS.items():
        s_is, s_oos = validate_isoos(aligned, px_a, px_b, idx, minutes, cfg, name)
        isoos_results[name] = (s_is, s_oos)
        for s in [s_is, s_oos]:
            print(f"  {name:<14} {s['label']:>3} {s['trades']:>5} {s['wr']:>5.1f}% ${s['pnl']:>9,.0f} "
                  f"{s['pf']:>6.2f} {s['sharpe']:>7.2f} ${s['max_dd']:>7,.0f}")

    # Verdicts
    print(f"\n  {'Config':<14} {'IS PF':>6} {'OOS PF':>7} {'OOS Trd':>8} {'Degrad%':>8} {'Verdict':>8}")
    print(f"  {'-'*55}")
    for name in CONFIGS:
        s_is, s_oos = isoos_results[name]
        degrad = ((s_is["pf"] - s_oos["pf"]) / s_is["pf"] * 100) if s_is["pf"] > 0 else 0
        verdict = "GO" if s_oos["pf"] > 1.0 and s_oos["trades"] >= 10 else "STOP"
        print(f"  {name:<14} {s_is['pf']:>6.2f} {s_oos['pf']:>7.2f} {s_oos['trades']:>8} "
              f"{degrad:>7.1f}% {'  ' + verdict:>8}")

    # ================================================================
    # SECTION 2: Permutation test (1000x)
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print(" SECTION 2: TEST DE PERMUTATION (1000x)")
    print("=" * 130)

    perm_results = {}
    for perm_name in CONFIGS:
        cfg = CONFIGS[perm_name]
        sig, beta, conf, sig_pre_conf = build_signal_kalman(aligned, idx, minutes, cfg)
        pf_obs, pf_mean_perm, p_val = validate_permutation(
            px_a, px_b, sig, beta, conf, cfg,
            sig_pre_conf, minutes, n_perms=1000
        )
        print(f"\n  [{perm_name}]")
        print(f"  PF observe:        {pf_obs:.2f}")
        print(f"  PF moyen permut:   {pf_mean_perm:.2f}")
        print(f"  p-value:           {p_val:.3f}")
        verdict_perm = "GO" if p_val < 0.05 else "STOP"
        print(f"  Verdict:           {verdict_perm} (seuil 0.05)")
        perm_results[perm_name] = {"pf_obs": pf_obs, "p_val": p_val, "verdict": verdict_perm}

    # ================================================================
    # SECTION 3: Walk-Forward (IS=2y, OOS=6m)
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print(" SECTION 3: WALK-FORWARD (IS=2 ans, OOS=6 mois)")
    print("=" * 130)

    wf_summary = {}
    for name, cfg in CONFIGS.items():
        wf_results = validate_walkforward(aligned, px_a, px_b, idx, minutes, cfg)
        n_profitable = sum(1 for r in wf_results if r["pnl"] > 0)
        n_windows = len(wf_results)
        total_pnl = sum(r["pnl"] for r in wf_results)
        pf_values = [r["pf"] for r in wf_results if r["trades"] > 0]
        avg_pf = np.mean(pf_values) if pf_values else 0
        total_trades = sum(r["trades"] for r in wf_results)

        print(f"\n  {name} -- {cfg['desc']}:")
        print(f"  {'Window':<6} {'Period':<22} {'Trd':>5} {'PnL':>10} {'PF':>6} {'WR%':>6} {'MaxDD':>8}")
        print(f"  {'-'*70}")
        for r in wf_results:
            flag = " ***" if r["pnl"] < 0 else ""
            print(f"  {r['label']:<6} {r['period']:<22} {r['trades']:>5} ${r['pnl']:>9,.0f} "
                  f"{r['pf']:>6.2f} {r['wr']:>5.1f}% ${r['max_dd']:>7,.0f}{flag}")
        verdict = "GO" if n_profitable >= n_windows * 0.6 else "STOP"
        print(f"  => {n_profitable}/{n_windows} profitables, PnL total=${total_pnl:,.0f}, "
              f"PF moy={avg_pf:.2f}, Trades total={total_trades} | {verdict}")

        wf_summary[name] = {
            "n_prof": n_profitable, "n_win": n_windows, "total_pnl": total_pnl,
            "avg_pf": avg_pf, "total_trades": total_trades, "verdict": verdict,
        }

    # ================================================================
    # SECTION 4: Yearly breakdown (stability check)
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print(" SECTION 4: YEARLY BREAKDOWN")
    print("=" * 130)

    for name, cfg in CONFIGS.items():
        sig, beta, _, _ = build_signal_kalman(aligned, idx, minutes, cfg)
        bt = run_bt(px_a, px_b, sig, beta)
        if bt["trades"] == 0:
            continue

        entry_bars = bt["trade_entry_bars"]
        pnls = bt["trade_pnls"]
        trade_sides = bt.get("trade_sides", None)
        entry_years = idx[entry_bars].year

        print(f"\n  {name}:")
        print(f"  {'Year':>6} {'Trd':>5} {'PnL':>10} {'WR%':>6} {'Avg$':>7} | {'Long$':>9} {'Short$':>9}")
        print(f"  {'-'*65}")

        for y in sorted(entry_years.unique()):
            mask = entry_years == y
            n_y = mask.sum()
            pnl_y = pnls[mask].sum()
            wr_y = (pnls[mask] > 0).sum() / n_y * 100 if n_y > 0 else 0
            avg_y = pnl_y / n_y if n_y > 0 else 0

            long_y = short_y = 0
            if trade_sides is not None:
                long_mask_y = mask & (trade_sides == 1)
                short_mask_y = mask & (trade_sides == -1)
                long_y = pnls[long_mask_y].sum() if long_mask_y.any() else 0
                short_y = pnls[short_mask_y].sum() if short_mask_y.any() else 0

            flag = " ***" if pnl_y < 0 else ""
            print(f"  {y:>6} {n_y:>5} ${pnl_y:>9,.0f} {wr_y:>5.1f}% ${avg_y:>6,.0f} | "
                  f"${long_y:>8,.0f} ${short_y:>8,.0f}{flag}")

    # ================================================================
    # SECTION 5: Summary & Recommendations
    # ================================================================
    print(f"\n\n{'=' * 130}")
    print(" SECTION 5: RESUME FINAL NQ_RTY")
    print("=" * 130)

    print(f"\n  {'Config':<14} {'IS/OOS':>7} {'Perm':>6} {'WF':>8} {'WF PnL':>10} {'WF PF':>6} {'Recommandation'}")
    print(f"  {'-'*80}")
    for name in CONFIGS:
        s_is, s_oos = isoos_results[name]
        isoos_v = "GO" if s_oos["pf"] > 1.0 and s_oos["trades"] >= 10 else "STOP"
        wfs = wf_summary[name]
        wf_v = wfs["verdict"]
        perm_v = perm_results.get(name, {}).get("verdict", "?")
        p_val = perm_results.get(name, {}).get("p_val", -1)

        go_count = sum([isoos_v == "GO", perm_v == "GO", wf_v == "GO"])
        if go_count == 3:
            reco = "VALIDE"
        elif go_count >= 2:
            reco = "MARGINAL"
        else:
            reco = "REJETE"

        perm_str = f"{p_val:.3f}" if p_val >= 0 else "?"
        print(f"  {name:<14} {isoos_v:>7} {perm_str:>6} {wfs['n_prof']}/{wfs['n_win']:>3} "
              f"${wfs['total_pnl']:>9,.0f} {wfs['avg_pf']:>6.2f}  {reco}")

    elapsed = time_mod.time() - t_start
    print(f"\n\n{'=' * 130}")
    print(f" VALIDATION NQ_RTY COMPLETE en {elapsed:.0f}s")
    print(f"{'=' * 130}")


if __name__ == "__main__":
    main()
