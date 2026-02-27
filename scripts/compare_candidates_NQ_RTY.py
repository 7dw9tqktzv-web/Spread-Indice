"""Compare les 3 configs selectionnees + 7 candidates complementaires.

Calcule Sharpe, Calmar, MaxDD, yearly + overlap trades entre configs.
Le but est de trouver des patterns structurellement differents des 3 selectionnees.

Usage:
    python scripts/compare_candidates_NQ_RTY.py
"""

import sys
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
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.config.instruments import get_pair_specs

_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
BARS_PER_YEAR = 264 * 252

METRIC_PROFILES = {
    "p12_64":  MetricsConfig(adf_window=12, hurst_window=64,  halflife_window=12, correlation_window=6),
    "p16_80":  MetricsConfig(adf_window=16, hurst_window=80,  halflife_window=16, correlation_window=8),
    "p18_96":  MetricsConfig(adf_window=18, hurst_window=96,  halflife_window=18, correlation_window=9),
    "p20_100": MetricsConfig(adf_window=20, hurst_window=100, halflife_window=20, correlation_window=10),
    "p24_128": MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "p28_144": MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p30_160": MetricsConfig(adf_window=30, hurst_window=160, halflife_window=30, correlation_window=15),
    "p36_192": MetricsConfig(adf_window=36, hurst_window=192, halflife_window=36, correlation_window=18),
    "p42_224": MetricsConfig(adf_window=42, hurst_window=224, halflife_window=42, correlation_window=21),
    "p48_256": MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "p60_320": MetricsConfig(adf_window=60, hurst_window=320, halflife_window=60, correlation_window=30),
    "p18_192": MetricsConfig(adf_window=18, hurst_window=192, halflife_window=18, correlation_window=18),
    "p24_256": MetricsConfig(adf_window=24, hurst_window=256, halflife_window=24, correlation_window=24),
    "p30_256": MetricsConfig(adf_window=30, hurst_window=256, halflife_window=30, correlation_window=24),
    "p48_128": MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
    "p48_96":  MetricsConfig(adf_window=48, hurst_window=96,  halflife_window=48, correlation_window=9),
    "p36_96":  MetricsConfig(adf_window=36, hurst_window=96,  halflife_window=36, correlation_window=9),
}
WINDOWS_MAP = {
    "02:00-14:00": (120, 840), "04:00-14:00": (240, 840), "06:00-14:00": (360, 840),
    "08:00-14:00": (480, 840), "08:00-12:00": (480, 720), "06:00-12:00": (360, 720),
}
CONF_WEIGHTS = ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00)

# The 3 selected configs (recurrent across multiple ranking groups)
SELECTED = [
    {"name": "A_RTY", "ols_w": 9240, "zw": 20, "prof": "p36_96",
     "win": "06:00-14:00", "ze": 3.00, "zx": 0.75, "zs": 5.0, "conf": 75,
     "reason": "Multi-critere: Sharpe#5 Calmar#2 PropFirm#3 Equilibre#3"},
    {"name": "B_RTY", "ols_w": 7920, "zw": 20, "prof": "p36_96",
     "win": "06:00-14:00", "ze": 3.00, "zx": 0.75, "zs": 3.5, "conf": 75,
     "reason": "Multi-critere: Sharpe#2 Equilibre#1, 0 annees neg"},
    {"name": "C_RTY", "ols_w": 7920, "zw": 20, "prof": "p42_224",
     "win": "04:00-14:00", "ze": 3.00, "zx": 0.75, "zs": 3.5, "conf": 75,
     "reason": "Multi-critere: Calmar#1 PropFirm#1, MaxDD -$4,145"},
]

# Candidates with STRUCTURALLY DIFFERENT patterns
CANDIDATES = [
    # --- Pattern 1: OLS court + conf stricte ---
    {"name": "D_court", "ols_w": 3960, "zw": 28, "prof": "p28_144",
     "win": "02:00-14:00", "ze": 3.00, "zx": 1.25, "zs": 5.5, "conf": 80,
     "reason": "OLS court 15j + ZW 28 + z_exit haut 1.25 + conf 80% (Calmar#3, MaxDD -$4,340)"},
    # --- Pattern 2: ADF tres lent + high WR ---
    {"name": "E_hiWR", "ols_w": 9240, "zw": 36, "prof": "p48_128",
     "win": "02:00-14:00", "ze": 3.25, "zx": 1.25, "zs": 4.5, "conf": 80,
     "reason": "ADF lent 48 + Hurst rapide 128 + 73.7% WR + ZW lent 36 (Sharpe#3)"},
    # --- Pattern 3: Fenetre etroite US ---
    {"name": "F_narrow", "ols_w": 6600, "zw": 60, "prof": "p28_144",
     "win": "08:00-12:00", "ze": 3.25, "zx": 0.75, "zs": 5.5, "conf": 80,
     "reason": "Fenetre 08-12h + ZW tres lent 60 + OLS moyen 25j (Sharpe#4, 73% WR)"},
    # --- Pattern 4: Sniper ze haut + OLS court ---
    {"name": "G_sniper", "ols_w": 3960, "zw": 24, "prof": "p16_80",
     "win": "06:00-14:00", "ze": 3.50, "zx": 0.50, "zs": 4.5, "conf": 70,
     "reason": "ze haut 3.50 (selectif) + zx bas 0.50 (courir) + OLS court (Calmar#5, PF 2.17)"},
    # --- Pattern 5: Profil classique + US only ---
    {"name": "H_classic", "ols_w": 9240, "zw": 36, "prof": "p24_128",
     "win": "08:00-14:00", "ze": 3.50, "zx": 0.25, "zs": 6.0, "conf": 80,
     "reason": "Profil court original + fenetre US 08-14h + 0 annees neg (PropFirm#4)"},
    # --- Pattern 6: Volume equilibre PF>1.5 + ze selectif ---
    {"name": "I_volume", "ols_w": 9240, "zw": 36, "prof": "p36_96",
     "win": "04:00-14:00", "ze": 3.50, "zx": 1.25, "zs": 5.5, "conf": 75,
     "reason": "ze=3.50 selectif + zx=1.25 prend profit, PF 2.15, 160t (Calmar#4, Equil#2)"},
    # --- Pattern 7: Ultra safe propfirm ---
    {"name": "J_safe", "ols_w": 3960, "zw": 28, "prof": "p18_96",
     "win": "02:00-14:00", "ze": 3.00, "zx": 1.25, "zs": 5.5, "conf": 80,
     "reason": "MaxDD -$4,500 + 0 neg years + streak=4 + PF 2.03 (PropFirm#2)"},
]

ALL = SELECTED + CANDIDATES


def main():
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    hedge_cache = {}
    results = []

    for i, cfg in enumerate(ALL):
        ols_w = cfg["ols_w"]
        zw = cfg["zw"]
        prof = cfg["prof"]
        win = cfg["win"]
        ze, zx, zs, conf = cfg["ze"], cfg["zx"], cfg["zs"], cfg["conf"]

        print(f"  [{i+1}/{len(ALL)}] {cfg['name']}", flush=True)

        hkey = (ols_w, zw)
        if hkey not in hedge_cache:
            est = create_estimator("ols_rolling", window=ols_w, zscore_window=zw)
            hr = est.estimate(aligned)
            spread = hr.spread
            beta = hr.beta.values
            mu = spread.rolling(zw).mean()
            sigma = spread.rolling(zw).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                zscore_arr = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
            zscore_arr = np.ascontiguousarray(np.nan_to_num(zscore_arr, nan=0.0), dtype=np.float64)
            hedge_cache[hkey] = (beta, zscore_arr, spread)
        else:
            beta, zscore_arr, spread = hedge_cache[hkey]

        profile_cfg = METRIC_PROFILES[prof]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, CONF_WEIGHTS).values

        raw = generate_signals_numba(zscore_arr, ze, zx, zs)
        sig = _apply_conf_filter_numba(raw, confidence, conf) if conf > 0 else raw.copy()
        entry_start, entry_end = WINDOWS_MAP[win]
        sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        equity = bt["equity"]
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd = drawdown.min()
        max_dd_pct = abs((drawdown / running_max).min()) * 100

        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(equity) / equity[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(BARS_PER_YEAR) if np.std(returns) > 0 else 0.0

        total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0 else 0.0
        with np.errstate(invalid="ignore"):
            ann_return = (1 + total_return) ** (BARS_PER_YEAR / max(len(equity), 1)) - 1
        if np.isnan(ann_return) or np.isinf(ann_return):
            ann_return = 0.0
        calmar = (ann_return * 100) / max_dd_pct if max_dd_pct > 0 else 0.0

        trade_pnls = bt["trade_pnls"]
        max_streak = 0
        cur = 0
        for p in trade_pnls:
            if p < 0:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0

        entry_bars = bt["trade_entry_bars"]
        entry_dates = idx[entry_bars]
        trade_df = pd.DataFrame({"year": entry_dates.year, "pnl": trade_pnls})
        yearly = trade_df.groupby("year")["pnl"].sum()
        neg_years = (yearly < 0).sum()

        results.append({
            "name": cfg["name"], "reason": cfg["reason"],
            "ols_w": ols_w, "zw": zw, "prof": prof, "win": win,
            "ze": ze, "zx": zx, "zs": zs, "conf": conf,
            "trades": bt["trades"], "wr": bt["win_rate"], "pnl": bt["pnl"],
            "pf": bt["profit_factor"], "max_dd": max_dd, "sharpe": sharpe, "calmar": calmar,
            "avg_pnl": bt["avg_pnl_trade"], "max_streak": max_streak, "neg_years": neg_years,
            "yearly": yearly, "entry_bars_set": set(entry_bars.tolist()),
        })

    # ================================================================
    # DISPLAY
    # ================================================================
    sel_entries = [r["entry_bars_set"] for r in results[:3]]
    sel_union = sel_entries[0] | sel_entries[1] | sel_entries[2]

    print(f"\n{'='*155}")
    print(f"  CONFIGS SELECTIONNEES (A/B/C) + CANDIDATES COMPLEMENTAIRES (D-J)")
    print(f"{'='*155}")

    print(f"\n  {'Name':<10} {'OLS':>5} {'ZW':>3} {'prof':<10} {'window':<14} "
          f"{'ze':>5} {'zx':>5} {'zs':>4} {'cf':>3} | "
          f"{'Trd':>4} {'WR%':>6} {'PnL':>10} {'PF':>5} {'MaxDD':>9} "
          f"{'Shrp':>5} {'Calm':>5} {'Strk':>4} {'Neg':>3} {'Ovlp%':>6}")
    print("  " + "-" * 145)

    for r in results:
        if r["name"] in ["A_RTY", "B_RTY", "C_RTY"]:
            ovlp_str = "  --- "
        else:
            ovlp = len(r["entry_bars_set"] & sel_union) / max(len(r["entry_bars_set"]), 1) * 100
            ovlp_str = f"{ovlp:5.1f}%"

        dd_flag = " <<<" if abs(r["max_dd"]) >= 8000 else ""
        safe_flag = " SAFE" if abs(r["max_dd"]) < 5000 else ""

        print(f"  {r['name']:<10} {r['ols_w']:>5} {r['zw']:>3} {r['prof']:<10} {r['win']:<14} "
              f"{r['ze']:>5.2f} {r['zx']:>5.2f} {r['zs']:>4.1f} {r['conf']:>3.0f} | "
              f"{r['trades']:>4} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>5.2f} "
              f"${r['max_dd']:>8,.0f} {r['sharpe']:>5.2f} {r['calmar']:>5.2f} "
              f"{r['max_streak']:>4} {r['neg_years']:>3} {ovlp_str}"
              f"{dd_flag}{safe_flag}")

        years_str = "             "
        for y, p in r["yearly"].items():
            flag = "***" if p < 0 else ""
            years_str += f"{y}=${p:+,.0f}{flag}  "
        print(years_str)
        if r["name"] not in ["A_RTY", "B_RTY", "C_RTY"]:
            print(f"             >> {r['reason']}")
        print()

    # ================================================================
    # OVERLAP MATRIX
    # ================================================================
    print(f"\n{'='*155}")
    print(f"  MATRICE D'OVERLAP (% trades communs)")
    print(f"{'='*155}")

    names = [r["name"] for r in results]
    print(f"\n  {'':>10}", end="")
    for n in names:
        print(f" {n:>9}", end="")
    print()
    print("  " + "-" * (10 + 10 * len(names)))

    for ri in results:
        print(f"  {ri['name']:>10}", end="")
        for rj in results:
            if ri["name"] == rj["name"]:
                print(f"     {'--':>4}", end="")
            else:
                common = len(ri["entry_bars_set"] & rj["entry_bars_set"])
                pct = common / max(len(ri["entry_bars_set"]), 1) * 100
                print(f"    {pct:>4.0f}%", end="")
        print(f"  ({ri['trades']}t)")

    # Pairwise A vs B vs C
    print(f"\n  --- OVERLAP DETAILLE A/B/C ---")
    for i in range(3):
        for j in range(i+1, 3):
            ri, rj = results[i], results[j]
            common = len(ri["entry_bars_set"] & rj["entry_bars_set"])
            pct_i = common / max(len(ri["entry_bars_set"]), 1) * 100
            pct_j = common / max(len(rj["entry_bars_set"]), 1) * 100
            print(f"  {ri['name']} vs {rj['name']}: {common} trades communs "
                  f"({pct_i:.0f}% de {ri['name']}, {pct_j:.0f}% de {rj['name']})")

    # Best complementary candidates (low overlap with ABC)
    print(f"\n  --- CANDIDATES PAR COMPLEMENTARITE (overlap faible avec A+B+C) ---")
    cands = [(r, len(r["entry_bars_set"] & sel_union) / max(len(r["entry_bars_set"]), 1) * 100)
             for r in results[3:]]
    cands.sort(key=lambda x: x[1])
    for r, ovlp in cands:
        safe = "SAFE" if abs(r["max_dd"]) < 5000 else "OK" if abs(r["max_dd"]) < 8000 else "DANGER"
        print(f"  {r['name']:<10} Overlap={ovlp:4.1f}% | {r['trades']}t PF={r['pf']:.2f} "
              f"MaxDD=${r['max_dd']:,.0f} ({safe}) Sharpe={r['sharpe']:.2f}")


if __name__ == "__main__":
    main()
