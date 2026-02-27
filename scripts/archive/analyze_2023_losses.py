"""Autopsie des trades 2023 — 10 dimensions d'analyse.

Objectif: comprendre pourquoi 2023 est la seule annee negative sur les 5 configs Kalman.
Identifier des patterns filtrables avec les outils existants.

Dimensions:
    1. Liste complete des trades 2023
    2. Heure d'entree (profil horaire)
    3. Jour de la semaine
    4. Mois
    5. Confidence score a l'entree
    6. Z-score a l'entree
    7. Side (Long vs Short)
    8. Duree et type de sortie (stop vs exit vs flat)
    9. Spread behavior / correlation NQ-YM
   10. Cross-check OLS (overlay complementarite)

Usage:
    python scripts/analyze_2023_losses.py
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
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "03:00-12:00": (180, 720),
    "04:00-13:00": (240, 780),
    "05:00-12:00": (300, 720),
}

# Kalman configs (top 5)
KALMAN_CONFIGS = {
    "K_Sniper": {
        "alpha": 3e-7, "profil": "court",
        "z_entry": 1.8125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 65.0, "window": "05:00-12:00",
    },
    "K_BestPnL": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.375, "z_exit": 0.25, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
    },
    "K_Balanced": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "03:00-12:00",
    },
    "K_Quality": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "04:00-13:00",
    },
    "K_ShortWin": {
        "alpha": 3e-7, "profil": "tres_court",
        "z_entry": 1.3125, "z_exit": 0.375, "z_stop": 2.75,
        "conf": 75.0, "window": "05:00-12:00",
    },
}

# OLS Config E
OLS_CONFIG = {
    "window": 3300, "zscore_window": 30,
    "z_entry": 3.15, "z_exit": 1.00, "z_stop": 4.50,
    "conf": 67.0, "profil": "tres_court",
    "window_filter": "02:00-14:00",
}

DAY_NAMES = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
MONTH_NAMES = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
               "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]


# ======================================================================
# Pipeline helpers
# ======================================================================

def run_kalman_pipeline(aligned, px_a, px_b, idx, minutes, cfg):
    """Run Kalman pipeline, return backtest result + intermediate data."""
    est = create_estimator("kalman", alpha_ratio=cfg["alpha"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    zscore = np.ascontiguousarray(hr.zscore.values, dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    metrics_cfg = METRICS_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(
        hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg
    )
    confidence = compute_confidence(metrics, CONF_CFG).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )
    return bt, zscore, confidence, sig, hr


def run_ols_pipeline(aligned, px_a, px_b, idx, minutes):
    """Run OLS Config E pipeline, return backtest result + intermediate data."""
    cfg = OLS_CONFIG
    est = create_estimator("ols_rolling", window=cfg["window"], zscore_window=cfg["zscore_window"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    zscore = np.ascontiguousarray(hr.zscore.values, dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    metrics_cfg = METRICS_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(
        hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg
    )
    confidence = compute_confidence(metrics, CONF_CFG).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
    entry_start, entry_end = WINDOWS_MAP[cfg["window_filter"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )
    return bt, zscore, confidence, sig, hr


def build_trade_df(bt, idx, zscore, confidence, sig):
    """Build a DataFrame with per-trade details."""
    if bt["trades"] == 0:
        return pd.DataFrame()

    entry_bars = bt["trade_entry_bars"]
    exit_bars = bt["trade_exit_bars"]
    pnls = bt["trade_pnls"]
    sides = bt["trade_sides"]

    rows = []
    for i in range(len(entry_bars)):
        eb = entry_bars[i]
        xb = exit_bars[i]
        entry_dt = idx[eb]
        exit_dt = idx[xb]
        duration = xb - eb  # in bars

        # Determine exit type
        z_at_exit = zscore[xb] if xb < len(zscore) else 0.0
        exit_minute = idx[xb].hour * 60 + idx[xb].minute
        if exit_minute >= FLAT_MIN - 5:  # within 5min of flat time
            exit_type = "FLAT"
        elif sides[i] == 1 and z_at_exit <= -2.75 or sides[i] == -1 and z_at_exit >= 2.75:
            exit_type = "STOP"
        else:
            exit_type = "EXIT"

        rows.append({
            "entry_date": entry_dt.strftime("%Y-%m-%d"),
            "entry_time": entry_dt.strftime("%H:%M"),
            "entry_hour": entry_dt.hour,
            "entry_dow": entry_dt.dayofweek,  # 0=Mon
            "entry_month": entry_dt.month,
            "entry_year": entry_dt.year,
            "side": "L" if sides[i] == 1 else "S",
            "pnl": float(pnls[i]),
            "win": pnls[i] > 0,
            "duration_bars": duration,
            "duration_min": duration * 5,
            "z_entry": float(zscore[eb]),
            "z_exit": float(z_at_exit),
            "conf_entry": float(confidence[eb]) if eb < len(confidence) else 0.0,
            "exit_type": exit_type,
            "entry_bar": eb,
            "exit_bar": xb,
        })

    return pd.DataFrame(rows)


# ======================================================================
# Analysis functions
# ======================================================================

def analyze_dimension_1(df_2023, df_rest):
    """1. Liste complete des trades 2023."""
    print(f"\n  --- DIM 1: Liste complete des trades 2023 ({len(df_2023)} trades) ---")
    print(f"  {'Date':>12} {'Time':>6} {'Side':>5} {'PnL':>9} {'Dur(min)':>9}"
          f" {'Z_entry':>8} {'Conf%':>6} {'Exit':>5}")
    print(f"  {'-' * 70}")

    for _, t in df_2023.sort_values("entry_date").iterrows():
        flag = " ***" if t["pnl"] < -2000 else ""
        print(f"  {t['entry_date']:>12} {t['entry_time']:>6} {t['side']:>5}"
              f" ${t['pnl']:>8,.0f} {t['duration_min']:>8}m"
              f" {t['z_entry']:>+8.3f} {t['conf_entry']:>5.0f}% {t['exit_type']:>5}{flag}")

    # Summary
    n_win = df_2023["win"].sum()
    n_loss = len(df_2023) - n_win
    print(f"\n  Gagnants: {n_win} (${df_2023[df_2023['win']]['pnl'].sum():,.0f})"
          f" | Perdants: {n_loss} (${df_2023[~df_2023['win']]['pnl'].sum():,.0f})"
          f" | Net: ${df_2023['pnl'].sum():,.0f}")


def analyze_dimension_2(df_2023, df_rest):
    """2. Profil horaire — heure d'entree."""
    print("\n  --- DIM 2: Heure d'entree (2023 vs annees profitables) ---")
    print(f"  {'Hour':>6}  {'------ 2023 ------':^28}  {'--- Profitable yrs ---':^28}")
    print(f"  {'':>6}  {'Trd':>5} {'PnL':>9} {'WR%':>6}    {'Trd':>5} {'PnL':>9} {'WR%':>6}")
    print(f"  {'-' * 70}")

    for h in range(3, 16):
        m23 = df_2023[df_2023["entry_hour"] == h]
        mrest = df_rest[df_rest["entry_hour"] == h]
        if len(m23) == 0 and len(mrest) == 0:
            continue
        n23, pnl23 = len(m23), m23["pnl"].sum() if len(m23) > 0 else 0
        wr23 = m23["win"].mean() * 100 if len(m23) > 0 else 0
        nr, pnlr = len(mrest), mrest["pnl"].sum() if len(mrest) > 0 else 0
        wrr = mrest["win"].mean() * 100 if len(mrest) > 0 else 0
        flag = " <<<" if n23 > 0 and pnl23 < -1000 else ""
        print(f"  {h:>5}h  {n23:>5} ${pnl23:>8,.0f} {wr23:>5.0f}%"
              f"    {nr:>5} ${pnlr:>8,.0f} {wrr:>5.0f}%{flag}")


def analyze_dimension_3(df_2023, df_rest):
    """3. Jour de la semaine."""
    print("\n  --- DIM 3: Jour de la semaine (2023 vs annees profitables) ---")
    print(f"  {'Day':>6}  {'------ 2023 ------':^28}  {'--- Profitable yrs ---':^28}")
    print(f"  {'':>6}  {'Trd':>5} {'PnL':>9} {'WR%':>6}    {'Trd':>5} {'PnL':>9} {'WR%':>6}")
    print(f"  {'-' * 70}")

    for d in range(5):  # Mon-Fri
        m23 = df_2023[df_2023["entry_dow"] == d]
        mrest = df_rest[df_rest["entry_dow"] == d]
        n23, pnl23 = len(m23), m23["pnl"].sum() if len(m23) > 0 else 0
        wr23 = m23["win"].mean() * 100 if len(m23) > 0 else 0
        nr, pnlr = len(mrest), mrest["pnl"].sum() if len(mrest) > 0 else 0
        wrr = mrest["win"].mean() * 100 if len(mrest) > 0 else 0
        flag = " <<<" if n23 > 0 and pnl23 < -1000 else ""
        print(f"  {DAY_NAMES[d]:>6}  {n23:>5} ${pnl23:>8,.0f} {wr23:>5.0f}%"
              f"    {nr:>5} ${pnlr:>8,.0f} {wrr:>5.0f}%{flag}")


def analyze_dimension_4(df_2023, df_rest):
    """4. Mois."""
    print("\n  --- DIM 4: PnL mensuel 2023 ---")
    print(f"  {'Month':>6} {'Trd':>5} {'PnL':>9} {'WR%':>6} {'Avg PnL':>9}")
    print(f"  {'-' * 40}")

    for m in range(1, 13):
        mm = df_2023[df_2023["entry_month"] == m]
        if len(mm) == 0:
            print(f"  {MONTH_NAMES[m-1]:>6} {0:>5} ${0:>8,.0f} {'':>6} {'':>9}")
            continue
        pnl = mm["pnl"].sum()
        wr = mm["win"].mean() * 100
        avg = mm["pnl"].mean()
        flag = " <<<" if pnl < -1000 else ""
        print(f"  {MONTH_NAMES[m-1]:>6} {len(mm):>5} ${pnl:>8,.0f} {wr:>5.0f}% ${avg:>8,.0f}{flag}")


def analyze_dimension_5(df_2023, df_rest):
    """5. Confidence score a l'entree."""
    print("\n  --- DIM 5: Confidence score a l'entree ---")

    for label, df in [("2023", df_2023), ("Profitable yrs", df_rest)]:
        if len(df) == 0:
            continue
        winners = df[df["win"]]
        losers = df[~df["win"]]
        print(f"\n  {label}:")
        print(f"    Winners  (n={len(winners):>3}): conf mean={winners['conf_entry'].mean():.1f}%,"
              f" median={winners['conf_entry'].median():.1f}%")
        print(f"    Losers   (n={len(losers):>3}): conf mean={losers['conf_entry'].mean():.1f}%,"
              f" median={losers['conf_entry'].median():.1f}%")

    # Test: what if confidence threshold was higher in 2023?
    print("\n  Sensitivity: confidence threshold impact on 2023:")
    print(f"  {'Threshold':>10} {'Trd':>5} {'PnL':>9} {'WR%':>6} {'Filtered':>9}")
    print(f"  {'-' * 45}")
    for thr in [75, 80, 85, 90, 95]:
        above = df_2023[df_2023["conf_entry"] >= thr]
        n_filtered = len(df_2023) - len(above)
        if len(above) > 0:
            pnl = above["pnl"].sum()
            wr = above["win"].mean() * 100
        else:
            pnl, wr = 0, 0
        print(f"  {thr:>9}% {len(above):>5} ${pnl:>8,.0f} {wr:>5.0f}% {n_filtered:>8} cut")


def analyze_dimension_6(df_2023, df_rest):
    """6. Z-score a l'entree."""
    print("\n  --- DIM 6: Z-score a l'entree ---")

    for label, df in [("2023", df_2023), ("Profitable yrs", df_rest)]:
        if len(df) == 0:
            continue
        winners = df[df["win"]]
        losers = df[~df["win"]]
        print(f"\n  {label}:")
        print(f"    Winners  |z|: mean={winners['z_entry'].abs().mean():.3f},"
              f" median={winners['z_entry'].abs().median():.3f}")
        print(f"    Losers   |z|: mean={losers['z_entry'].abs().mean():.3f},"
              f" median={losers['z_entry'].abs().median():.3f}")

    # Distribution
    print("\n  |z| buckets 2023:")
    print(f"  {'|z| range':>12} {'Trd':>5} {'PnL':>9} {'WR%':>6}")
    print(f"  {'-' * 35}")
    z_abs = df_2023["z_entry"].abs()
    for lo, hi, label in [(0, 1.4, "<1.4"), (1.4, 1.6, "1.4-1.6"),
                           (1.6, 2.0, "1.6-2.0"), (2.0, 2.5, "2.0-2.5"), (2.5, 5.0, ">2.5")]:
        mask = (z_abs >= lo) & (z_abs < hi)
        sub = df_2023[mask]
        if len(sub) > 0:
            print(f"  {label:>12} {len(sub):>5} ${sub['pnl'].sum():>8,.0f}"
                  f" {sub['win'].mean() * 100:>5.0f}%")


def analyze_dimension_7(df_2023, df_rest):
    """7. Side (Long vs Short)."""
    print("\n  --- DIM 7: Side (Long vs Short) ---")
    print(f"  {'':>15} {'------ 2023 ------':^28}  {'--- Profitable yrs ---':^28}")
    print(f"  {'Side':>15} {'Trd':>5} {'PnL':>9} {'WR%':>6}    {'Trd':>5} {'PnL':>9} {'WR%':>6}")
    print(f"  {'-' * 75}")

    for side_label, side_val in [("Long (NQ>YM)", "L"), ("Short (NQ<YM)", "S")]:
        m23 = df_2023[df_2023["side"] == side_val]
        mrest = df_rest[df_rest["side"] == side_val]
        n23, pnl23 = len(m23), m23["pnl"].sum() if len(m23) > 0 else 0
        wr23 = m23["win"].mean() * 100 if len(m23) > 0 else 0
        nr, pnlr = len(mrest), mrest["pnl"].sum() if len(mrest) > 0 else 0
        wrr = mrest["win"].mean() * 100 if len(mrest) > 0 else 0
        print(f"  {side_label:>15} {n23:>5} ${pnl23:>8,.0f} {wr23:>5.0f}%"
              f"    {nr:>5} ${pnlr:>8,.0f} {wrr:>5.0f}%")


def analyze_dimension_8(df_2023, df_rest):
    """8. Duree et type de sortie."""
    print("\n  --- DIM 8: Duree et type de sortie ---")

    for label, df in [("2023", df_2023), ("Profitable yrs", df_rest)]:
        if len(df) == 0:
            continue
        winners = df[df["win"]]
        losers = df[~df["win"]]
        print(f"\n  {label}:")
        print(f"    Winners duration: mean={winners['duration_min'].mean():.0f}min,"
              f" median={winners['duration_min'].median():.0f}min")
        if len(losers) > 0:
            print(f"    Losers  duration: mean={losers['duration_min'].mean():.0f}min,"
                  f" median={losers['duration_min'].median():.0f}min")

    # Exit type breakdown
    print("\n  Exit type (2023 vs profitable years):")
    print(f"  {'Type':>8}  {'--- 2023 ---':^20}  {'--- Prof yrs ---':^20}")
    print(f"  {'':>8}  {'Trd':>5} {'PnL':>9}    {'Trd':>5} {'PnL':>9}")
    print(f"  {'-' * 55}")
    for et in ["EXIT", "STOP", "FLAT"]:
        m23 = df_2023[df_2023["exit_type"] == et]
        mrest = df_rest[df_rest["exit_type"] == et]
        print(f"  {et:>8}  {len(m23):>5} ${m23['pnl'].sum():>8,.0f}"
              f"    {len(mrest):>5} ${mrest['pnl'].sum():>8,.0f}")


def analyze_dimension_9(df_2023, aligned, idx):
    """9. Spread behavior / correlation NQ-YM en 2023."""
    print("\n  --- DIM 9: Spread regime 2023 vs other years ---")

    close_a = aligned.df["close_a"]
    close_b = aligned.df["close_b"]

    # Annual rolling correlation (60-day ~ 15840 bars at 5min, use daily returns)
    # Simpler: compute daily returns and annual correlation
    # Resample to daily
    daily_a = close_a.resample("D").last().dropna()
    daily_b = close_b.resample("D").last().dropna()
    common = daily_a.index.intersection(daily_b.index)
    ret_a = daily_a.loc[common].pct_change().dropna()
    ret_b = daily_b.loc[common].pct_change().dropna()

    print("\n  Annual NQ-YM daily return correlation:")
    print(f"  {'Year':>6} {'Corr':>8} {'NQ ret%':>9} {'YM ret%':>9} {'Spread drift':>13}")
    print(f"  {'-' * 50}")

    daily_a_common = daily_a.loc[common]
    daily_b_common = daily_b.loc[common]

    for y in sorted(ret_a.index.year.unique()):
        if y < 2021:
            continue
        mask_ret = ret_a.index.year == y
        if mask_ret.sum() < 20:
            continue
        corr = float(ret_a[mask_ret].corr(ret_b[mask_ret]))
        mask_daily = daily_a_common.index.year == y
        da_y = daily_a_common[mask_daily]
        db_y = daily_b_common[mask_daily]
        nq_ret = float((da_y.iloc[-1] / da_y.iloc[0] - 1) * 100) if len(da_y) > 1 else 0
        ym_ret = float((db_y.iloc[-1] / db_y.iloc[0] - 1) * 100) if len(db_y) > 1 else 0
        flag = " <-- TARGET" if y == 2023 else ""
        print(f"  {y:>6} {corr:>8.3f} {nq_ret:>+8.1f}% {ym_ret:>+8.1f}% {nq_ret - ym_ret:>+12.1f}%{flag}")

    # Hurst by year on 5min spread (using simple variance ratio)
    print("\n  Annual spread characteristics (log spread):")
    log_a = np.log(close_a)
    log_b = np.log(close_b)

    for y in range(2021, 2027):
        mask = idx.year == y
        if mask.sum() < 1000:
            continue
        la = log_a[mask].values
        lb = log_b[mask].values
        # Simple spread with beta ~1.2 (approximate)
        spread_y = la - 1.2 * lb
        spread_std = float(np.std(spread_y))
        # Simple mean reversion test: autocorrelation of changes
        dspread = np.diff(spread_y)
        if len(dspread) > 100:
            acf1 = float(np.corrcoef(dspread[:-1], dspread[1:])[0, 1])
        else:
            acf1 = 0
        flag = " <-- TARGET" if y == 2023 else ""
        print(f"  {y}: spread_std={spread_std:.6f}, dSpread ACF(1)={acf1:+.4f}{flag}")


def analyze_dimension_10(df_2023_kalman, ols_bt, ols_zscore, ols_conf, idx):
    """10. Cross-check OLS — overlay complementarite en 2023."""
    print("\n  --- DIM 10: Cross-check OLS Config E (overlay complementarite) ---")

    if ols_bt["trades"] == 0:
        print("  OLS: 0 trades")
        return

    # Build OLS trade dates for 2023
    ols_entry_bars = ols_bt["trade_entry_bars"]
    ols_pnls = ols_bt["trade_pnls"]
    ols_dates = idx[ols_entry_bars]
    ols_2023_mask = ols_dates.year == 2023

    ols_2023_dates = set(ols_dates[ols_2023_mask].strftime("%Y-%m-%d"))
    ols_2023_pnls = ols_pnls[ols_2023_mask]

    # Kalman 2023 dates
    kalman_2023_dates = set(df_2023_kalman["entry_date"].values)

    # Overlap
    common_dates = kalman_2023_dates & ols_2023_dates
    kalman_only = kalman_2023_dates - ols_2023_dates
    ols_only = ols_2023_dates - kalman_2023_dates

    print("\n  2023 trade days:")
    print(f"    Kalman: {len(kalman_2023_dates)} days")
    print(f"    OLS:    {len(ols_2023_dates)} days")
    print(f"    Common: {len(common_dates)} days ({len(common_dates)/max(len(kalman_2023_dates),1)*100:.0f}%)")
    print(f"    Kalman only: {len(kalman_only)} days")
    print(f"    OLS only:    {len(ols_only)} days")

    # PnL on common days vs exclusive days
    # Kalman PnL on common vs exclusive
    k_common_pnl = df_2023_kalman[df_2023_kalman["entry_date"].isin(common_dates)]["pnl"].sum()
    k_exclusive_pnl = df_2023_kalman[df_2023_kalman["entry_date"].isin(kalman_only)]["pnl"].sum()

    # OLS PnL on common vs exclusive
    ols_common_mask = np.array([d.strftime("%Y-%m-%d") in common_dates for d in ols_dates[ols_2023_mask]])
    ols_common_pnl = float(ols_2023_pnls[ols_common_mask].sum()) if ols_common_mask.any() else 0
    ols_exclusive_mask = np.array([d.strftime("%Y-%m-%d") in ols_only for d in ols_dates[ols_2023_mask]])
    ols_exclusive_pnl = float(ols_2023_pnls[ols_exclusive_mask].sum()) if ols_exclusive_mask.any() else 0

    print("\n  2023 PnL breakdown:")
    print(f"    {'':>20} {'Kalman':>10} {'OLS':>10}")
    print(f"    {'-' * 42}")
    print(f"    {'Common days':>20} ${k_common_pnl:>9,.0f} ${ols_common_pnl:>9,.0f}")
    print(f"    {'Exclusive days':>20} ${k_exclusive_pnl:>9,.0f} ${ols_exclusive_pnl:>9,.0f}")
    print(f"    {'Total 2023':>20} ${df_2023_kalman['pnl'].sum():>9,.0f} ${float(ols_2023_pnls.sum()):>9,.0f}")

    # Key question: when Kalman loses, does OLS win?
    print("\n  Critical test: Kalman losing days in 2023 — what does OLS do?")
    k_losing_days = set(df_2023_kalman[~df_2023_kalman["win"]]["entry_date"].values)

    print(f"  {'Kalman loss date':>18} {'K PnL':>9} {'OLS trade?':>11} {'OLS PnL':>9} {'Overlay':>9}")
    print(f"  {'-' * 60}")

    combined_pnl = 0
    overlay_wins = 0
    overlay_total = 0

    for date_str in sorted(k_losing_days):
        k_pnl = float(df_2023_kalman[df_2023_kalman["entry_date"] == date_str]["pnl"].sum())

        # Find OLS trades on this date
        ols_on_date = np.array([d.strftime("%Y-%m-%d") == date_str for d in ols_dates[ols_2023_mask]])
        if ols_on_date.any():
            o_pnl = float(ols_2023_pnls[ols_on_date].sum())
            ols_flag = "YES"
        else:
            o_pnl = 0
            ols_flag = "no"

        overlay = k_pnl + o_pnl
        combined_pnl += overlay
        overlay_total += 1
        if overlay > 0:
            overlay_wins += 1

        prot = " +PROTECTED" if o_pnl > abs(k_pnl) * 0.5 else ""
        print(f"  {date_str:>18} ${k_pnl:>8,.0f} {ols_flag:>11} ${o_pnl:>8,.0f} ${overlay:>8,.0f}{prot}")

    print(f"\n  Kalman losing days: {overlay_total}")
    print(f"  OLS compensates (overlay > 0): {overlay_wins}/{overlay_total}"
          f" ({overlay_wins/max(overlay_total,1)*100:.0f}%)")
    print(f"  Combined PnL on Kalman losing days: ${combined_pnl:,.0f}")

    # Overall 2023 overlay
    total_overlay = df_2023_kalman["pnl"].sum() + float(ols_2023_pnls.sum())
    print(f"\n  2023 OVERLAY total: Kalman ${df_2023_kalman['pnl'].sum():,.0f}"
          f" + OLS ${float(ols_2023_pnls.sum()):,.0f}"
          f" = ${total_overlay:,.0f}")


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time_mod.time()

    print("Loading NQ_YM 5min data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    print(f"Data: {len(px_a):,} bars, {(idx[-1] - idx[0]).days / 365.25:.1f} years")
    print(f"Period: {idx[0].strftime('%Y-%m-%d')} to {idx[-1].strftime('%Y-%m-%d')}")

    # Run OLS Config E once (for dimension 10)
    print("\nRunning OLS Config E...")
    ols_bt, ols_zscore, ols_conf, ols_sig, ols_hr = run_ols_pipeline(
        aligned, px_a, px_b, idx, minutes
    )
    ols_2023_mask = idx[ols_bt["trade_entry_bars"]].year == 2023
    print(f"OLS Config E: {ols_bt['trades']} trades total,"
          f" {ols_2023_mask.sum()} in 2023,"
          f" PnL 2023=${float(ols_bt['trade_pnls'][ols_2023_mask].sum()):,.0f}")

    # Run each Kalman config
    for cfg_name, cfg in KALMAN_CONFIGS.items():
        print(f"\n\n{'=' * 110}")
        print(f" {cfg_name} — AUTOPSIE 2023")
        print(f" alpha={cfg['alpha']:.0e} profil={cfg['profil']} "
              f"ze={cfg['z_entry']} zx={cfg['z_exit']} zs={cfg['z_stop']} "
              f"c={cfg['conf']}% win={cfg['window']}")
        print(f"{'=' * 110}")

        bt, zscore, confidence, sig, hr = run_kalman_pipeline(
            aligned, px_a, px_b, idx, minutes, cfg
        )
        trades_df = build_trade_df(bt, idx, zscore, confidence, sig)

        if len(trades_df) == 0:
            print("  0 trades - skip")
            continue

        df_2023 = trades_df[trades_df["entry_year"] == 2023]
        # Profitable years: 2022 + 2024 + 2025 (skip 2020/2021 low sample, 2026 partial)
        df_rest = trades_df[trades_df["entry_year"].isin([2022, 2024, 2025])]

        print(f"\n  Total: {len(trades_df)} trades | 2023: {len(df_2023)} trades"
              f" | Ref years (2022+2024+2025): {len(df_rest)} trades")

        # All 10 dimensions
        analyze_dimension_1(df_2023, df_rest)
        analyze_dimension_2(df_2023, df_rest)
        analyze_dimension_3(df_2023, df_rest)
        analyze_dimension_4(df_2023, df_rest)
        analyze_dimension_5(df_2023, df_rest)
        analyze_dimension_6(df_2023, df_rest)
        analyze_dimension_7(df_2023, df_rest)
        analyze_dimension_8(df_2023, df_rest)
        analyze_dimension_9(df_2023, aligned, idx)
        analyze_dimension_10(df_2023, ols_bt, ols_zscore, ols_conf, idx)

    elapsed = time_mod.time() - t_start
    print(f"\n\n{'=' * 110}")
    print(f" COMPLETE en {elapsed:.0f}s")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
