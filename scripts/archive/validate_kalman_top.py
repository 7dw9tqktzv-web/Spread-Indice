"""Validate top Kalman configs from grid v3: IS/OOS, permutation, walk-forward.

Top configs selected from 5 propfirm profiles (1,009,800 combos).

Usage:
    python scripts/validate_kalman_top.py
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
# Constants
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930  # 15:30 CT

METRICS_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court":      MetricsConfig(adf_window=66, hurst_window=132, halflife_window=66, correlation_window=66),
    "moyen":      MetricsConfig(adf_window=264, hurst_window=528, halflife_window=264, correlation_window=264),
}

WINDOWS_MAP = {
    "03:00-12:00": (180, 720),
    "04:00-12:00": (240, 720),
    "04:00-13:00": (240, 780),
    "04:00-14:00": (240, 840),
    "05:00-12:00": (300, 720),
}

# ======================================================================
# Top configs from grid v3
# ======================================================================

CONFIGS = {
    "K_Sniper": {
        "alpha": 3e-7, "profil": "court",
        "z_entry": 1.8125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 65.0, "window": "05:00-12:00",
        "desc": "Sniper #1: 91 trades, PF 3.24, $64,990",
    },
    "K_BestPnL": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.375, "z_exit": 0.25, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
        "desc": "Best PnL: 238 trades, PF 1.84, $84,825",
    },
    "K_Balanced": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
        "desc": "Balanced PF>2: 218 trades, PF 2.09, $80,805",
    },
    "K_Quality": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "04:00-13:00",
        "desc": "Quality: 205 trades, PF 2.12, $80,255",
    },
    "K_ShortWin": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "05:00-12:00",
        "desc": "Short window: 183 trades, PF 2.16, $77,890",
    },
}


# ======================================================================
# Pipeline helper
# ======================================================================

def build_signal_kalman(aligned, idx, minutes, cfg):
    """Build signal array from Kalman config dict.

    Returns (sig, beta, confidence, sig_pre_conf).
    """
    est = create_estimator("kalman", alpha_ratio=cfg["alpha"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    zscore = hr.zscore.values.copy()
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    metrics_cfg = METRICS_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(
        hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg
    )
    confidence = compute_confidence(metrics, CONF_CFG).values

    # No time stop for Kalman (grid v3 didn't use it)
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
                "sharpe": 0, "max_dd": 0, "avg_pnl": 0, "avg_dur": 0}
    pnls = bt["trade_pnls"]
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max).min()
    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(n)) if n > 1 and pnls.std() > 0 else 0
    calmar = float(bt["pnl"] / abs(dd)) if dd < 0 else 0
    return {
        "label": label, "trades": n, "pnl": bt["pnl"], "pf": bt["profit_factor"],
        "wr": bt["win_rate"], "sharpe": round(sharpe, 2), "max_dd": round(dd, 0),
        "avg_pnl": bt["avg_pnl_trade"], "avg_dur": bt["avg_duration_bars"],
        "calmar": round(calmar, 2),
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

    print("Loading NQ_YM data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years:.1f} years")
    print(f"Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}\n")

    # ================================================================
    # SECTION 0: Full-sample verification
    # ================================================================
    print("=" * 120)
    print(" SECTION 0: FULL-SAMPLE VERIFICATION (should match grid results)")
    print("=" * 120)
    print(f"\n  {'Config':<14} {'Desc':<48} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7}")
    print(f"  {'-'*115}")

    for name, cfg in CONFIGS.items():
        s = validate_full(aligned, px_a, px_b, idx, minutes, cfg, name)
        print(f"  {name:<14} {cfg['desc']:<48} {s['trades']:>5} {s['wr']:>5.1f}% ${s['pnl']:>9,.0f} "
              f"{s['pf']:>6.2f} {s['sharpe']:>7.2f} ${s['max_dd']:>7,.0f} {s['calmar']:>7.2f}")

    # ================================================================
    # SECTION 1: IS/OOS split 60/40
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 1: IS/OOS SPLIT 60/40")
    print("=" * 120)

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
        print(f"  {name:<14} {s_is['pf']:>6.2f} {s_oos['pf']:>7.2f} {s_oos['trades']:>8} {degrad:>7.1f}% {'  ' + verdict:>8}")

    # ================================================================
    # SECTION 2: Permutation test (1000x) — on K_Balanced (representative)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 2: TEST DE PERMUTATION (1000x)")
    print("=" * 120)

    for perm_name in ["K_Balanced", "K_Sniper"]:
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
        print(f"  Verdict:           {'GO' if p_val < 0.05 else 'STOP'} (seuil 0.05)")

    # ================================================================
    # SECTION 3: Walk-Forward (IS=2y, OOS=6m)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 3: WALK-FORWARD (IS=2 ans, OOS=6 mois)")
    print("=" * 120)

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
    print(f"\n\n{'=' * 120}")
    print(" SECTION 4: YEARLY BREAKDOWN")
    print("=" * 120)

    for name, cfg in CONFIGS.items():
        sig, beta, _, _ = build_signal_kalman(aligned, idx, minutes, cfg)

        # Get trade entries/exits from backtest
        bt = run_bt(px_a, px_b, sig, beta)
        if bt["trades"] == 0:
            continue

        entry_bars = bt["trade_entry_bars"]
        pnls = bt["trade_pnls"]
        entry_years = idx[entry_bars].year

        print(f"\n  {name}:")
        print(f"  {'Year':>6} {'Trd':>5} {'PnL':>10} {'WR%':>6} {'Avg$':>7}")
        print(f"  {'-'*40}")

        for y in sorted(entry_years.unique()):
            mask = entry_years == y
            n_y = mask.sum()
            pnl_y = pnls[mask].sum()
            wr_y = (pnls[mask] > 0).sum() / n_y * 100 if n_y > 0 else 0
            avg_y = pnl_y / n_y if n_y > 0 else 0
            flag = " ***" if pnl_y < 0 else ""
            print(f"  {y:>6} {n_y:>5} ${pnl_y:>9,.0f} {wr_y:>5.1f}% ${avg_y:>6,.0f}{flag}")

    # ================================================================
    # SECTION 5: Summary & Recommendations
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 5: RESUME FINAL")
    print("=" * 120)

    print(f"\n  {'Config':<14} {'IS/OOS':>7} {'Perm':>5} {'WF':>8} {'WF PnL':>10} {'WF PF':>6} {'Recommandation'}")
    print(f"  {'-'*80}")
    for name in CONFIGS:
        s_is, s_oos = isoos_results[name]
        isoos_v = "GO" if s_oos["pf"] > 1.0 and s_oos["trades"] >= 10 else "STOP"
        wfs = wf_summary[name]
        wf_v = wfs["verdict"]

        # Recommendation logic
        if isoos_v == "GO" and wf_v == "GO":
            reco = "VALIDE"
        elif isoos_v == "GO" or wf_v == "GO":
            reco = "MARGINAL"
        else:
            reco = "REJETE"

        print(f"  {name:<14} {isoos_v:>7} {'--':>5} {wfs['n_prof']}/{wfs['n_win']:>3} "
              f"${wfs['total_pnl']:>9,.0f} {wfs['avg_pf']:>6.2f}  {reco}")

    elapsed = time_mod.time() - t_start
    print(f"\n\n{'=' * 120}")
    print(f" VALIDATION COMPLETE en {elapsed:.0f}s")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
