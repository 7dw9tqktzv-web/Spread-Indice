"""Etape 1b — Exploration approfondie Kalman NQ/RTY + ES/RTY.

More configs around the promising zones, different windows and profiles.

Usage:
    python scripts/explore_pairs_v2.py
"""

import sys
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
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930  # 15:30 CT
CONF_CFG = ConfidenceConfig()

METRICS_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court": MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "moyen": MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
}

WINDOWS = [
    ("02:00-14:00", 120, 840),
    ("03:00-12:00", 180, 720),
    ("04:00-12:00", 240, 720),
    ("04:00-13:00", 240, 780),
    ("05:00-12:00", 300, 720),
]

KALMAN_CONFIGS = [
    # Refining NQ/RTY promising zone: alpha 1e-7 to 3e-7, ze 1.5-2.25, conf 60-75
    {"label": "K1", "alpha": 1e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 2.75, "conf": 60, "profile": "tres_court"},
    {"label": "K2", "alpha": 1e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 2.75, "conf": 65, "profile": "tres_court"},
    {"label": "K3", "alpha": 1e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 2.75, "conf": 70, "profile": "tres_court"},
    {"label": "K4", "alpha": 1e-7,   "ze": 2.25,  "zx": 0.5,   "zs": 2.75, "conf": 60, "profile": "tres_court"},
    {"label": "K5", "alpha": 1.5e-7, "ze": 2.0,   "zx": 0.5,   "zs": 2.75, "conf": 60, "profile": "tres_court"},
    {"label": "K6", "alpha": 1.5e-7, "ze": 2.0,   "zx": 0.75,  "zs": 2.75, "conf": 65, "profile": "tres_court"},
    {"label": "K7", "alpha": 1.5e-7, "ze": 1.75,  "zx": 0.375, "zs": 2.75, "conf": 70, "profile": "tres_court"},
    {"label": "K8", "alpha": 2e-7,   "ze": 1.75,  "zx": 0.5,   "zs": 2.75, "conf": 65, "profile": "tres_court"},
    {"label": "K9", "alpha": 2e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 2.75, "conf": 60, "profile": "tres_court"},
    {"label": "K10","alpha": 3e-7,   "ze": 1.5,   "zx": 0.25,  "zs": 2.75, "conf": 60, "profile": "tres_court"},
    {"label": "K11","alpha": 3e-7,   "ze": 1.5,   "zx": 0.25,  "zs": 2.75, "conf": 70, "profile": "tres_court"},
    {"label": "K12","alpha": 3e-7,   "ze": 1.5,   "zx": 0.5,   "zs": 2.5,  "conf": 65, "profile": "tres_court"},
    # Profile variations
    {"label": "K13","alpha": 1e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 2.75, "conf": 60, "profile": "court"},
    {"label": "K14","alpha": 1e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 2.75, "conf": 60, "profile": "moyen"},
    {"label": "K15","alpha": 3e-7,   "ze": 1.5,   "zx": 0.25,  "zs": 2.75, "conf": 70, "profile": "court"},
    {"label": "K16","alpha": 3e-7,   "ze": 1.5,   "zx": 0.25,  "zs": 2.75, "conf": 70, "profile": "moyen"},
    # z_stop variations
    {"label": "K17","alpha": 1e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 2.5,  "conf": 60, "profile": "tres_court"},
    {"label": "K18","alpha": 1e-7,   "ze": 2.0,   "zx": 0.5,   "zs": 3.0,  "conf": 60, "profile": "tres_court"},
]

PAIRS = [
    ("NQ_RTY", Instrument.NQ, Instrument.RTY, 20.0, 50.0, 0.25, 0.10),
    ("ES_RTY", Instrument.ES, Instrument.RTY, 50.0, 50.0, 0.25, 0.10),
]


def run_pair(pair_name, leg_a, leg_b, mult_a, mult_b, tick_a, tick_b):
    print(f"\n{'='*140}")
    print(f" EXPLORATION APPROFONDIE: {pair_name}")
    print(f"{'='*140}")

    pair = SpreadPair(leg_a=leg_a, leg_b=leg_b)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print(f"  ERREUR: pas de cache pour {pair_name}")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years = (idx[-1] - idx[0]).days / 365.25

    # Pre-compute Kalman estimates for each alpha (avoid re-estimation)
    kalman_cache = {}

    all_results = []

    for cfg in KALMAN_CONFIGS:
        alpha = cfg["alpha"]
        profile = cfg["profile"]

        # Cache Kalman estimation
        cache_key = alpha
        if cache_key not in kalman_cache:
            est = create_estimator("kalman", alpha_ratio=alpha, warmup=200, gap_P_multiplier=5.0)
            hr = est.estimate(aligned)
            kalman_cache[cache_key] = hr

        hr = kalman_cache[cache_key]
        beta = hr.beta.values
        zscore = np.ascontiguousarray(
            np.nan_to_num(hr.zscore.values, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float64,
        )

        # Metrics + confidence for this profile
        profile_cfg = METRICS_PROFILES[profile]
        metrics = compute_all_metrics(hr.spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, CONF_CFG).values

        # Signals
        raw = generate_signals_numba(zscore, cfg["ze"], cfg["zx"], cfg["zs"])
        sig_conf = _apply_conf_filter_numba(raw, confidence, cfg["conf"])

        for win_name, entry_start, entry_end in WINDOWS:
            sig = apply_window_filter_numba(sig_conf.copy(), minutes, entry_start, entry_end, FLAT_MIN)

            bt = run_backtest_vectorized(
                px_a, px_b, sig, beta,
                mult_a, mult_b, tick_a, tick_b,
                SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
            )

            n = bt["trades"]
            if n == 0:
                continue

            pnl = bt["pnl"]
            pf = bt["profit_factor"]
            wr = bt["win_rate"]
            avg_pnl = bt["avg_pnl_trade"]
            avg_dur = bt["avg_duration_bars"]
            trd_y = n / years if years > 0 else 0

            # Long/short
            trade_sides = bt.get("trade_sides", None)
            trade_pnls = bt["trade_pnls"]
            long_pnl = short_pnl = 0
            if trade_sides is not None:
                long_mask = trade_sides == 1
                short_mask = trade_sides == -1
                long_pnl = float(trade_pnls[long_mask].sum()) if long_mask.any() else 0
                short_pnl = float(trade_pnls[short_mask].sum()) if short_mask.any() else 0

            all_results.append({
                "label": cfg["label"], "window": win_name,
                "alpha": alpha, "profile": profile,
                "ze": cfg["ze"], "zx": cfg["zx"], "zs": cfg["zs"], "conf": cfg["conf"],
                "trades": n, "wr": wr, "pnl": pnl, "pf": pf,
                "avg_pnl": avg_pnl, "avg_dur": avg_dur, "trd_y": trd_y,
                "long_pnl": long_pnl, "short_pnl": short_pnl,
            })

    # Sort by PnL
    profitable = [r for r in all_results if r["pnl"] > 0]
    profitable.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n  {len(profitable)} / {len(all_results)} configs profitables ({len(profitable)/max(len(all_results),1)*100:.1f}%)")

    # Print top 30 by PnL
    print("\n  TOP 30 BY PNL:")
    header = (f"  {'Label':<6} {'Window':<12} {'a':>8} {'Prof':<10} "
              f"{'ze':>5} {'zx':>4} {'zs':>4} {'c':>3} | "
              f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7} {'Dur':>5} | "
              f"{'Long$':>9} {'Short$':>9} {'L%':>5}")
    print(header)
    print("  " + "-" * 130)

    for r in profitable[:30]:
        total_abs = abs(r["long_pnl"]) + abs(r["short_pnl"])
        l_pct = (r["long_pnl"] / total_abs * 100) if total_abs > 0 else 50
        flag = ""
        if abs(l_pct) > 80:
            flag = " **"
        elif abs(l_pct) > 60:
            flag = " *"

        print(
            f"  {r['label']:<6} {r['window']:<12} {r['alpha']:>8.1e} {r['profile']:<10} "
            f"{r['ze']:>5.3f} {r['zx']:>4.2f} {r['zs']:>4.2f} {r['conf']:>3.0f} | "
            f"{r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>6.2f} "
            f"${r['avg_pnl']:>6,.0f} {r['avg_dur']:>5.1f} | "
            f"${r['long_pnl']:>8,.0f} ${r['short_pnl']:>8,.0f} {l_pct:>4.0f}%{flag}"
        )

    # Best by PF (min 30 trades)
    pf_sorted = [r for r in profitable if r["trades"] >= 30]
    pf_sorted.sort(key=lambda x: x["pf"], reverse=True)
    if pf_sorted:
        print("\n  TOP 10 BY PF (min 30 trades):")
        print(header)
        print("  " + "-" * 130)
        for r in pf_sorted[:10]:
            total_abs = abs(r["long_pnl"]) + abs(r["short_pnl"])
            l_pct = (r["long_pnl"] / total_abs * 100) if total_abs > 0 else 50
            print(
                f"  {r['label']:<6} {r['window']:<12} {r['alpha']:>8.1e} {r['profile']:<10} "
                f"{r['ze']:>5.3f} {r['zx']:>4.02f} {r['zs']:>4.2f} {r['conf']:>3.0f} | "
                f"{r['trades']:>5} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>6.2f} "
                f"${r['avg_pnl']:>6,.0f} {r['avg_dur']:>5.1f} | "
                f"${r['long_pnl']:>8,.0f} ${r['short_pnl']:>8,.0f} {l_pct:>4.0f}%"
            )


if __name__ == "__main__":
    for pair_name, leg_a, leg_b, mult_a, mult_b, tick_a, tick_b in PAIRS:
        run_pair(pair_name, leg_a, leg_b, mult_a, mult_b, tick_a, tick_b)
