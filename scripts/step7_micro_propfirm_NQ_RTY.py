"""Etape 7 -- Micro contracts + Propfirm sizing NQ/RTY OLS (10 configs).

Test micro x1 et x2 (MNQ/M2K) avec commission $0.62 RT/contrat.
Propfirm compatibility: $5K trailing DD, $300/day target.

Usage:
    python scripts/step7_micro_propfirm_NQ_RTY.py
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
    ConfidenceConfig, compute_confidence,
    _apply_conf_filter_numba, apply_window_filter_numba,
    apply_time_stop,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.config.instruments import get_pair_specs

# ======================================================================
# Constants
# ======================================================================
# E-mini (reference)
_NQ, _RTY = get_pair_specs("NQ", "RTY")
EMINI_MULT_A, EMINI_MULT_B = _NQ.multiplier, _RTY.multiplier
EMINI_TICK_A, EMINI_TICK_B = _NQ.tick_size, _RTY.tick_size
EMINI_COMMISSION = 2.50

# Micro contracts (1/10 of E-mini)
MICRO_MULT_A, MICRO_MULT_B = 2.0, 5.0
MICRO_TICK_A, MICRO_TICK_B = 0.25, 0.10
MICRO_COMMISSION_PER_CONTRACT = 0.62

SLIPPAGE = 1
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
BARS_PER_DAY = 264
BARS_PER_YEAR = BARS_PER_DAY * 252

# Propfirm constraints
PROPFIRM_MAX_DD = 5000.0
PROPFIRM_DAILY_TARGET = 300.0

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


def build_signal(aligned, minutes, cfg, ts_bars):
    """Build signal chain (same for all contract sizes)."""
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

    if ts_bars > 0:
        sig = apply_time_stop(sig.copy(), ts_bars)

    return sig, beta


def run_bt_full(px_a, px_b, sig, beta, mult_a, mult_b, tick_a, tick_b,
                commission, max_mult=1):
    """Run backtest with given contract specs."""
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        mult_a, mult_b, tick_a, tick_b,
        SLIPPAGE, commission, INITIAL_CAPITAL,
        max_multiplier=max_mult,
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
        "max_dd": round(max_dd, 0),
        "sharpe": round(sharpe, 2),
        "avg_pnl": bt["avg_pnl_trade"],
        "equity": equity,
    }


def compute_daily_pnl(equity, dates):
    """Compute daily PnL from equity curve."""
    daily = {}
    for i in range(1, len(dates)):
        d = dates[i].date()
        if d not in daily:
            # First bar of day: use previous close as reference
            daily[d] = {"first_eq": equity[i - 1], "last_eq": equity[i]}
        else:
            daily[d]["last_eq"] = equity[i]

    daily_pnls = []
    for d in sorted(daily.keys()):
        pnl = daily[d]["last_eq"] - daily[d]["first_eq"]
        if abs(pnl) > 0.01:  # Skip flat days
            daily_pnls.append(pnl)
    return daily_pnls


def compute_yearly(equity, dates):
    """Yearly PnL."""
    years_set = sorted(set(d.year for d in dates))
    yearly = {}
    for y in years_set:
        mask = np.array([d.year == y for d in dates])
        if mask.sum() < 10:
            continue
        eq = equity[mask]
        yearly[y] = float(eq[-1] - eq[0])
    return yearly


def main():
    t_start = time_mod.time()

    print("=" * 130)
    print("  ETAPE 7 -- MICRO CONTRACTS + PROPFIRM SIZING NQ/RTY OLS")
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
    dates = idx.to_pydatetime()
    total_years = (idx[-1] - idx[0]).days / 365.25

    # Load configs
    top10_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step3_top10.csv"
    configs_df = pd.read_csv(top10_path)

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
    # SECTION A: E-mini vs Micro x1 vs Micro x2 comparison
    # ==================================================================
    print(f"\n{'='*130}")
    print("  SECTION A: E-MINI vs MICRO x1 vs MICRO x2")
    print(f"{'='*130}")

    # Estimate average commission for micro
    # Micro x1: 1 MNQ + ~N_b M2K. N_b ~ 2-4 contracts avg. Commission = (1+N_b)*$0.62
    # Micro x2: 2 MNQ + ~2*N_b M2K. Commission = (2+2*N_b)*$0.62
    # Conservative estimates: micro x1 ~ $2.48 (4 contracts), micro x2 ~ $4.96 (8 contracts)
    MICRO_X1_COMMISSION = 2.48
    MICRO_X2_COMMISSION = 4.96

    all_results = []

    for cfg in configs:
        label = cfg["label"]
        tier = cfg["tier"]
        ts = cfg["ts_bars"]

        sig, beta = build_signal(aligned, minutes, cfg, ts)

        # E-mini
        bt_emini = run_bt_full(px_a, px_b, sig, beta,
                               EMINI_MULT_A, EMINI_MULT_B, EMINI_TICK_A, EMINI_TICK_B,
                               EMINI_COMMISSION, max_mult=1)

        # Micro x1
        bt_micro1 = run_bt_full(px_a, px_b, sig, beta,
                                MICRO_MULT_A, MICRO_MULT_B, MICRO_TICK_A, MICRO_TICK_B,
                                MICRO_X1_COMMISSION, max_mult=1)

        # Micro x2
        bt_micro2 = run_bt_full(px_a, px_b, sig, beta,
                                MICRO_MULT_A, MICRO_MULT_B, MICRO_TICK_A, MICRO_TICK_B,
                                MICRO_X2_COMMISSION, max_mult=2)

        all_results.append({
            "label": label,
            "tier": tier,
            "emini": bt_emini,
            "micro1": bt_micro1,
            "micro2": bt_micro2,
        })

    # Print comparison table
    print(f"\n  {'Label':<42} {'Tier':<6} | {'--- E-MINI ---':>30} | {'--- MICRO x1 ---':>30} | {'--- MICRO x2 ---':>30}")
    print(f"  {'':>48} | {'Trd PF   PnL     MaxDD':>30} | {'Trd PF   PnL     MaxDD':>30} | {'Trd PF   PnL     MaxDD':>30}")
    print(f"  {'-'*160}")

    for r in all_results:
        e = r["emini"]
        m1 = r["micro1"]
        m2 = r["micro2"]
        print(f"  {r['label']:<42} {r['tier']:<6} | "
              f"{e['trades']:>3} {e['pf']:>4.2f} ${e['pnl']:>7,.0f} ${e['max_dd']:>7,.0f} | "
              f"{m1['trades']:>3} {m1['pf']:>4.2f} ${m1['pnl']:>7,.0f} ${m1['max_dd']:>7,.0f} | "
              f"{m2['trades']:>3} {m2['pf']:>4.2f} ${m2['pnl']:>7,.0f} ${m2['max_dd']:>7,.0f}")

    # ==================================================================
    # SECTION B: Propfirm compatibility (MaxDD < $5K)
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION B: PROPFIRM COMPATIBILITY ($5K trailing DD)")
    print(f"{'='*130}")

    print(f"\n  {'#':>2} {'Label':<42} {'Size':>7} {'Trd':>4} {'PF':>5} {'PnL':>9} "
          f"{'MaxDD':>9} {'DD%5K':>6} {'Status':>8} {'$/trade':>8} {'$/day':>7}")
    print(f"  {'-'*120}")

    propfirm_candidates = []

    for i, r in enumerate(all_results, 1):
        label = r["label"]
        tier = r["tier"]

        for size_name, bt, mult_label in [
            ("E-mini", r["emini"], "emini"),
            ("Mx1", r["micro1"], "micro1"),
            ("Mx2", r["micro2"], "micro2"),
        ]:
            dd = abs(bt["max_dd"])
            dd_pct = dd / PROPFIRM_MAX_DD * 100
            status = "SAFE" if dd < PROPFIRM_MAX_DD * 0.90 else (
                "WARN" if dd < PROPFIRM_MAX_DD else "DANGER")
            avg_daily = bt["pnl"] / (252 * total_years) if total_years > 0 else 0

            if status in ("SAFE", "WARN"):
                propfirm_candidates.append({
                    "rank": i,
                    "label": label,
                    "tier": tier,
                    "size": size_name,
                    "trades": bt["trades"],
                    "pf": bt["pf"],
                    "pnl": bt["pnl"],
                    "max_dd": bt["max_dd"],
                    "dd_pct": dd_pct,
                    "status": status,
                    "avg_pnl_trade": bt["avg_pnl"],
                    "avg_daily_pnl": round(avg_daily, 0),
                })

            # Only print interesting rows
            if status != "DANGER" or size_name == "E-mini":
                print(f"  {i:>2} {label:<42} {size_name:>7} {bt['trades']:>4} "
                      f"{bt['pf']:>5.2f} ${bt['pnl']:>8,.0f} ${bt['max_dd']:>8,.0f} "
                      f"{dd_pct:>5.0f}% {status:>8} ${bt['avg_pnl']:>7,.0f} ${avg_daily:>6,.0f}")

    # ==================================================================
    # SECTION C: Daily PnL distribution for propfirm candidates
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION C: DISTRIBUTION DAILY PnL -- CANDIDATS PROPFIRM")
    print(f"{'='*130}")

    for r in all_results:
        label = r["label"]

        # Test micro x2 (most likely propfirm choice)
        for size_name, bt_data in [("Micro x2", r["micro2"]), ("E-mini", r["emini"])]:
            dd = abs(bt_data["max_dd"])
            if dd > PROPFIRM_MAX_DD:
                continue

            daily_pnls = compute_daily_pnl(bt_data["equity"], dates)
            if not daily_pnls:
                continue

            daily_arr = np.array(daily_pnls)
            pos_days = np.sum(daily_arr > 0)
            neg_days = np.sum(daily_arr < 0)
            total_days = len(daily_arr)
            pct_pos = pos_days / total_days * 100 if total_days > 0 else 0

            worst_day = daily_arr.min()
            best_day = daily_arr.max()
            avg_pos = daily_arr[daily_arr > 0].mean() if pos_days > 0 else 0
            avg_neg = daily_arr[daily_arr < 0].mean() if neg_days > 0 else 0

            # Days above $300 target
            above_target = np.sum(daily_arr >= PROPFIRM_DAILY_TARGET)
            pct_target = above_target / total_days * 100 if total_days > 0 else 0

            print(f"\n  {label} [{size_name}] (MaxDD ${bt_data['max_dd']:,.0f})")
            print(f"    Active days: {total_days} ({pos_days}+ / {neg_days}-) = {pct_pos:.0f}% positive")
            print(f"    Avg +day: ${avg_pos:,.0f}  |  Avg -day: ${avg_neg:,.0f}  |  Ratio: {abs(avg_pos/avg_neg) if avg_neg != 0 else 0:.2f}")
            print(f"    Best day: ${best_day:,.0f}  |  Worst day: ${worst_day:,.0f}")
            print(f"    Days >= $300 target: {above_target}/{total_days} = {pct_target:.0f}%")

            # Percentiles
            pcts = [5, 10, 25, 50, 75, 90, 95]
            vals = np.percentile(daily_arr, pcts)
            print(f"    Percentiles: ", end="")
            for p, v in zip(pcts, vals):
                print(f"P{p}=${v:+,.0f} ", end="")
            print()

    # ==================================================================
    # SECTION D: Yearly decomposition micro x2
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION D: DECOMPOSITION ANNUELLE -- MICRO x2")
    print(f"{'='*130}")

    all_years = sorted(set(d.year for d in dates))
    all_years = [y for y in all_years if y >= 2021]

    header = f"  {'#':>2} {'Label':<42} {'Tier':<6}"
    for y in all_years:
        header += f" {y:>8}"
    header += f" {'NegYr':>5} {'MaxDD':>8} {'Status':>7}"
    print(f"\n{header}")
    print(f"  {'-'*(60 + 9*len(all_years) + 22)}")

    for i, r in enumerate(all_results, 1):
        bt = r["micro2"]
        yearly = compute_yearly(bt["equity"], dates)
        dd = abs(bt["max_dd"])
        status = "SAFE" if dd < PROPFIRM_MAX_DD * 0.90 else (
            "WARN" if dd < PROPFIRM_MAX_DD else "DANGER")

        line = f"  {i:>2} {r['label']:<42} {r['tier']:<6}"
        neg = 0
        for y in all_years:
            pnl_y = yearly.get(y, 0)
            if pnl_y < 0:
                neg += 1
                line += f" ${pnl_y:>7,.0f}*"
            else:
                line += f" ${pnl_y:>7,.0f}"
        line += f" {neg:>5} ${bt['max_dd']:>7,.0f} {status:>7}"
        print(line)

    # ==================================================================
    # SECTION E: FINAL RECOMMENDATION
    # ==================================================================
    print(f"\n\n{'='*130}")
    print("  SECTION E: RECOMMANDATION FINALE PROPFIRM")
    print(f"{'='*130}")

    if propfirm_candidates:
        df_pf = pd.DataFrame(propfirm_candidates)
        df_pf = df_pf.sort_values(["status", "pf"], ascending=[True, False])

        print(f"\n  {len(df_pf)} configs propfirm-compatibles:")
        print(f"\n  {'#':>2} {'Label':<42} {'Size':>6} {'Trd':>4} {'PF':>5} "
              f"{'PnL':>9} {'MaxDD':>8} {'DD%':>5} {'$/day':>7} {'Status':>6}")
        print(f"  {'-'*105}")

        for _, c in df_pf.iterrows():
            print(f"  {c['rank']:>2} {c['label']:<42} {c['size']:>6} {c['trades']:>4} "
                  f"{c['pf']:>5.2f} ${c['pnl']:>8,.0f} ${c['max_dd']:>7,.0f} "
                  f"{c['dd_pct']:>4.0f}% ${c['avg_daily_pnl']:>6,.0f} {c['status']:>6}")
    else:
        print("\n  AUCUNE config propfirm-compatible trouvee.")

    # Save
    out_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step7_micro_propfirm.csv"
    if propfirm_candidates:
        pd.DataFrame(propfirm_candidates).to_csv(out_path, index=False)
    print(f"\n  Sauvegarde: {out_path}")

    elapsed = time_mod.time() - t_start
    print(f"\n  Etape 7 complete en {elapsed:.0f}s")


if __name__ == "__main__":
    main()
