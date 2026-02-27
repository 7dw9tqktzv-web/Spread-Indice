"""Etape 8 — Cross-pair complementarity: NQ/RTY Kalman vs NQ/YM (OLS + Kalman).

Compares trade-level overlap, daily PnL correlation, and diversification benefit.

Usage:
    python scripts/analyze_complementarity.py
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
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930

METRICS_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "court": MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "moyen": MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "03:00-12:00": (180, 720),
    "04:00-12:00": (240, 720),
    "04:00-13:00": (240, 780),
    "05:00-12:00": (300, 720),
}


# ======================================================================
# Configs to compare
# ======================================================================

# NQ/YM Kalman K_Balanced (champion validated, from CLAUDE.md)
NQ_YM_KALMAN = {
    "pair": ("NQ", "YM"),
    "leg_a": Instrument.NQ, "leg_b": Instrument.YM,
    "mult_a": 20.0, "mult_b": 5.0,
    "tick_a": 0.25, "tick_b": 1.0,
    "alpha": 3e-7, "profil": "tres_court",
    "z_entry": 1.375, "z_exit": 0.25, "z_stop": 2.75,
    "conf": 75.0, "window": "03:00-12:00",
    "label": "NQ_YM K_Balanced",
}

# NQ/RTY Best candidates (marginal but interesting for complement)
NQ_RTY_SNIPER = {
    "pair": ("NQ", "RTY"),
    "leg_a": Instrument.NQ, "leg_b": Instrument.RTY,
    "mult_a": 20.0, "mult_b": 50.0,
    "tick_a": 0.25, "tick_b": 0.10,
    "alpha": 2.5e-7, "profil": "moyen",
    "z_entry": 1.875, "z_exit": 0.25, "z_stop": 3.25,
    "conf": 70.0, "window": "04:00-13:00",
    "label": "NQ_RTY K_Sniper",
}

NQ_RTY_PROPFIRM = {
    "pair": ("NQ", "RTY"),
    "leg_a": Instrument.NQ, "leg_b": Instrument.RTY,
    "mult_a": 20.0, "mult_b": 50.0,
    "tick_a": 0.25, "tick_b": 0.10,
    "alpha": 2.5e-7, "profil": "tres_court",
    "z_entry": 1.5625, "z_exit": 1.5, "z_stop": 2.25,
    "conf": 60.0, "window": "05:00-12:00",
    "label": "NQ_RTY K_PropFirm",
}

NQ_RTY_VOLUME = {
    "pair": ("NQ", "RTY"),
    "leg_a": Instrument.NQ, "leg_b": Instrument.RTY,
    "mult_a": 20.0, "mult_b": 50.0,
    "tick_a": 0.25, "tick_b": 0.10,
    "alpha": 2.5e-7, "profil": "court",
    "z_entry": 1.3125, "z_exit": 0.25, "z_stop": 3.25,
    "conf": 50.0, "window": "05:00-12:00",
    "label": "NQ_RTY K_Volume",
}


def run_config(cfg):
    """Run a single config and return trade-level data."""
    pair = SpreadPair(leg_a=cfg["leg_a"], leg_b=cfg["leg_b"])
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print(f"  ERREUR: pas de cache pour {cfg['label']}")
        return None

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    est = create_estimator("kalman", alpha_ratio=cfg["alpha"],
                           warmup=200, gap_P_multiplier=5.0)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    zscore = np.ascontiguousarray(
        np.nan_to_num(hr.zscore.values, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float64,
    )

    profile_cfg = METRICS_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(hr.spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
    confidence = compute_confidence(metrics, CONF_CFG).values

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])
    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        cfg["mult_a"], cfg["mult_b"], cfg["tick_a"], cfg["tick_b"],
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )

    if bt["trades"] == 0:
        return None

    # Build trade DataFrame with dates
    entry_bars = bt["trade_entry_bars"]
    exit_bars = bt["trade_exit_bars"]
    trade_pnls = bt["trade_pnls"]
    trade_sides = bt.get("trade_sides", np.zeros(len(trade_pnls)))

    entry_dates = idx[entry_bars]
    exit_dates = idx[exit_bars]

    trades_df = pd.DataFrame({
        "entry_date": entry_dates,
        "exit_date": exit_dates,
        "entry_day": entry_dates.date,
        "pnl": trade_pnls,
        "side": trade_sides,
    })

    return {
        "label": cfg["label"],
        "bt": bt,
        "trades_df": trades_df,
        "idx": idx,
        "equity": bt["equity"],
    }


def compute_daily_pnl(trades_df):
    """Aggregate trade PnL by entry day."""
    daily = trades_df.groupby("entry_day")["pnl"].sum()
    return daily


def analyze_overlap(res_a, res_b):
    """Analyze temporal overlap between two strategies."""
    label_a = res_a["label"]
    label_b = res_b["label"]

    daily_a = compute_daily_pnl(res_a["trades_df"])
    daily_b = compute_daily_pnl(res_b["trades_df"])

    # Days where each strategy trades
    days_a = set(daily_a.index)
    days_b = set(daily_b.index)
    common_days = days_a & days_b
    only_a = days_a - days_b
    only_b = days_b - days_a

    print(f"\n  --- {label_a} vs {label_b} ---")
    print(f"  Trading days A: {len(days_a)}, B: {len(days_b)}")
    print(f"  Common days:    {len(common_days)} ({len(common_days)/max(len(days_a),1)*100:.0f}% of A, "
          f"{len(common_days)/max(len(days_b),1)*100:.0f}% of B)")
    print(f"  Only A:         {len(only_a)}")
    print(f"  Only B:         {len(only_b)}")

    if common_days:
        # Correlation on common days
        common_idx = sorted(common_days)
        pnl_a_common = daily_a.reindex(common_idx).fillna(0)
        pnl_b_common = daily_b.reindex(common_idx).fillna(0)

        corr = pnl_a_common.corr(pnl_b_common)
        print(f"  PnL correlation (common days): {corr:.3f}")

        # How often both lose
        both_lose = ((pnl_a_common < 0) & (pnl_b_common < 0)).sum()
        both_win = ((pnl_a_common > 0) & (pnl_b_common > 0)).sum()
        print(f"  Both win:  {both_win}/{len(common_idx)} ({both_win/len(common_idx)*100:.0f}%)")
        print(f"  Both lose: {both_lose}/{len(common_idx)} ({both_lose/len(common_idx)*100:.0f}%)")

    # Losing days analysis
    losing_days_a = set(d for d in daily_a.index if daily_a[d] < 0)
    losing_days_b = set(d for d in daily_b.index if daily_b[d] < 0)
    common_losing = losing_days_a & losing_days_b
    print(f"\n  Losing days A: {len(losing_days_a)}, B: {len(losing_days_b)}")
    print(f"  Common losing days: {len(common_losing)}")

    # Combined portfolio analysis
    all_days = sorted(days_a | days_b)
    combined_daily = pd.Series(0.0, index=all_days)
    for d in all_days:
        combined_daily[d] = daily_a.get(d, 0) + daily_b.get(d, 0)

    total_pnl = combined_daily.sum()
    losing_combined = (combined_daily < 0).sum()
    max_daily_loss = combined_daily.min()
    max_daily_win = combined_daily.max()

    print("\n  Combined portfolio:")
    print(f"  Total PnL:       ${total_pnl:,.0f}")
    print(f"  Trading days:    {len(all_days)}")
    print(f"  Losing days:     {losing_combined} ({losing_combined/len(all_days)*100:.0f}%)")
    print(f"  Max daily loss:  ${max_daily_loss:,.0f}")
    print(f"  Max daily win:   ${max_daily_win:,.0f}")

    # Yearly combined
    combined_series = pd.Series(combined_daily.values, index=pd.DatetimeIndex(all_days))
    yearly = combined_series.groupby(combined_series.index.year).agg(["sum", "count"])
    yearly.columns = ["pnl", "days"]

    print(f"\n  {'Year':>6} {'PnL':>10} {'Days':>5}")
    print(f"  {'-'*25}")
    for y, row in yearly.iterrows():
        flag = " ***" if row["pnl"] < 0 else ""
        print(f"  {y:>6} ${row['pnl']:>9,.0f} {row['days']:>5.0f}{flag}")


def main():
    print("=" * 120)
    print(" CROSS-PAIR COMPLEMENTARITY: NQ/RTY vs NQ/YM")
    print("=" * 120)

    # Run all configs
    configs = [NQ_YM_KALMAN, NQ_RTY_SNIPER, NQ_RTY_PROPFIRM, NQ_RTY_VOLUME]
    results = {}

    for cfg in configs:
        print(f"\n  Running {cfg['label']}...")
        res = run_config(cfg)
        if res is not None:
            results[cfg["label"]] = res
            bt = res["bt"]
            trades_df = res["trades_df"]
            n = bt["trades"]
            pnl = bt["pnl"]
            pf = bt["profit_factor"]
            wr = bt["win_rate"]

            # Long/short
            long_pnl = trades_df[trades_df["side"] == 1]["pnl"].sum()
            short_pnl = trades_df[trades_df["side"] == -1]["pnl"].sum()
            total_abs = abs(long_pnl) + abs(short_pnl)
            l_pct = (long_pnl / total_abs * 100) if total_abs > 0 else 50

            print(f"    {n} trades, WR {wr:.1f}%, PnL ${pnl:,.0f}, PF {pf:.2f}")
            print(f"    Long ${long_pnl:,.0f} / Short ${short_pnl:,.0f} ({l_pct:.0f}% long)")

    # ================================================================
    # Pairwise complementarity
    # ================================================================
    print(f"\n\n{'='*120}")
    print(" PAIRWISE COMPLEMENTARITY ANALYSIS")
    print("=" * 120)

    nym = results.get("NQ_YM K_Balanced")
    if nym is None:
        print("  NQ_YM K_Balanced not available. Abort.")
        return

    for nq_rty_label in ["NQ_RTY K_Sniper", "NQ_RTY K_PropFirm", "NQ_RTY K_Volume"]:
        nrt = results.get(nq_rty_label)
        if nrt:
            analyze_overlap(nym, nrt)

    # ================================================================
    # Best combo analysis
    # ================================================================
    print(f"\n\n{'='*120}")
    print(" BEST PORTFOLIO COMBO (NQ_YM + best NQ_RTY)")
    print("=" * 120)

    # Compare all combos
    for nq_rty_label in ["NQ_RTY K_Sniper", "NQ_RTY K_PropFirm", "NQ_RTY K_Volume"]:
        nrt = results.get(nq_rty_label)
        if not nrt:
            continue

        daily_nym = compute_daily_pnl(nym["trades_df"])
        daily_nrt = compute_daily_pnl(nrt["trades_df"])

        all_days = sorted(set(daily_nym.index) | set(daily_nrt.index))
        combined = pd.Series(0.0, index=all_days)
        for d in all_days:
            combined[d] = daily_nym.get(d, 0) + daily_nrt.get(d, 0)

        # Compute drawdown from daily PnL
        cumulative = combined.cumsum()
        running_max = cumulative.cummax()
        dd = (cumulative - running_max).min()

        losing_days = (combined < 0).sum()
        max_consec_loss = 0
        current_streak = 0
        for v in combined.values:
            if v < 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        print(f"\n  NQ_YM + {nq_rty_label}:")
        print(f"    Total PnL:      ${combined.sum():,.0f}")
        print(f"    Trading days:   {len(all_days)}")
        print(f"    Losing days:    {losing_days} ({losing_days/len(all_days)*100:.0f}%)")
        print(f"    Max DD (daily): ${dd:,.0f}")
        print(f"    Max consec loss:{max_consec_loss} days")
        print(f"    Daily Sharpe:   {combined.mean()/combined.std()*np.sqrt(252):.2f}" if combined.std() > 0 else "")


if __name__ == "__main__":
    main()
