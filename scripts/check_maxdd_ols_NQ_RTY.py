"""Calcule le MaxDD des top 5 OLS NQ/RTY via run_backtest_vectorized.

Usage:
    python scripts/check_maxdd_ols_NQ_RTY.py
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

MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

METRIC_PROFILES = {
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
    "06:00-11:00": (360, 660),
    "08:00-14:00": (480, 840),
}

CONFIGS = {
    "OLS_Balanced": {
        "ols_window": 7920, "zscore_window": 48,
        "profile": "court", "window": "08:00-14:00",
        "z_entry": 3.50, "z_exit": 0.50, "z_stop": 6.00, "min_confidence": 75,
    },
    "OLS_Quality": {
        "ols_window": 6600, "zscore_window": 36,
        "profile": "moyen", "window": "02:00-14:00",
        "z_entry": 3.50, "z_exit": 0.00, "z_stop": 6.00, "min_confidence": 80,
    },
    "OLS_Volume": {
        "ols_window": 792, "zscore_window": 30,
        "profile": "court", "window": "04:00-13:00",
        "z_entry": 1.50, "z_exit": 0.00, "z_stop": 6.00, "min_confidence": 75,
    },
    "OLS_Sniper": {
        "ols_window": 6600, "zscore_window": 42,
        "profile": "moyen", "window": "05:00-12:00",
        "z_entry": 3.75, "z_exit": 0.00, "z_stop": 6.00, "min_confidence": 80,
    },
    "OLS_PropFirm": {
        "ols_window": 10560, "zscore_window": 48,
        "profile": "court", "window": "03:00-12:00",
        "z_entry": 1.50, "z_exit": 0.00, "z_stop": 6.00, "min_confidence": 70,
    },
}


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

    print(f"{'Config':<18} {'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} "
          f"{'MaxDD':>10} {'AvgPnl':>8} {'MaxStrk':>7}")
    print("-" * 80)

    for name, cfg in CONFIGS.items():
        est = create_estimator("ols_rolling",
                               window=cfg["ols_window"],
                               zscore_window=cfg["zscore_window"])
        hr = est.estimate(aligned)
        spread = hr.spread
        beta = hr.beta.values

        mu = spread.rolling(cfg["zscore_window"]).mean()
        sigma = spread.rolling(cfg["zscore_window"]).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

        profile_cfg = METRIC_PROFILES[cfg["profile"]]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, ConfidenceConfig()).values

        raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

        if cfg["min_confidence"] > 0:
            sig = _apply_conf_filter_numba(raw, confidence, cfg["min_confidence"])
        else:
            sig = raw.copy()

        entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
        sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        # MaxDD from equity
        equity = bt["equity"]
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd = drawdown.min()

        # Max losing streak
        trade_pnls = bt["trade_pnls"]
        max_streak = 0
        current = 0
        for p in trade_pnls:
            if p < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        print(f"{name:<18} {bt['trades']:>5} {bt['win_rate']:>5.1f}% "
              f"${bt['pnl']:>9,.0f} {bt['profit_factor']:>6.2f} "
              f"${max_dd:>9,.0f} ${bt['avg_pnl_trade']:>7,.0f} {max_streak:>7}")

        # Yearly breakdown
        entry_bars = bt["trade_entry_bars"]
        entry_dates = idx[entry_bars]
        trade_df = pd.DataFrame({
            "year": entry_dates.year,
            "pnl": trade_pnls,
        })
        yearly = trade_df.groupby("year").agg(
            trades=("pnl", "count"),
            pnl=("pnl", "sum"),
            wr=("pnl", lambda x: (x > 0).sum() / len(x) * 100),
        )
        years_str = "  Years: "
        for y, row in yearly.iterrows():
            flag = " ***" if row["pnl"] < 0 else ""
            years_str += f"{y}=${row['pnl']:+,.0f}({row['trades']:.0f}t){flag}  "
        print(f"  {years_str}")
        print()


if __name__ == "__main__":
    main()
