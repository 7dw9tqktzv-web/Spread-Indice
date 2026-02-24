"""Diagnostic: ADF threshold sweep with gate pass rates and trade quality."""

import sys
from pathlib import Path

import numpy as np
from numba import njit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
from src.stats.stationarity import adf_statistic_simple
from src.stats.hurst import hurst_rolling
from src.stats.correlation import rolling_correlation
from src.signals.generator import generate_signals_numba
from src.signals.filters import apply_window_filter_numba
from src.backtest.engine import run_backtest_vectorized

pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
aligned = load_aligned_pair_cache(pair, "5min")
px_a = aligned.df["close_a"].values
px_b = aligned.df["close_b"].values
idx = aligned.df.index
minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
FLAT_MIN = 930

est = create_estimator("ols_rolling", window=3000, zscore_window=30)
hr = est.estimate(aligned)
beta = hr.beta.values
spread = hr.spread

# Precompute gates
hurst_64 = hurst_rolling(spread, window=64, step=1).values
corr_24 = rolling_correlation(aligned.df["close_a"], aligned.df["close_b"], window=24).values
gate_hurst = (hurst_64 < 0.50) & ~np.isnan(hurst_64)
gate_corr = (corr_24 > 0.70) & ~np.isnan(corr_24)
hc_mask = gate_hurst & gate_corr

# Zscore
mu = spread.rolling(30).mean()
sigma = spread.rolling(30).std()
with np.errstate(divide="ignore", invalid="ignore"):
    zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
zscore = np.ascontiguousarray(zscore, dtype=np.float64)

raw_signals = generate_signals_numba(zscore, 2.75, 0.75, 4.50)


@njit
def apply_gate(sig, mask):
    out = sig.copy()
    prev = np.int8(0)
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0 and not mask[t]:
            out[t] = np.int8(0)
        prev = out[t]
    return out


def run_config(gate_mask):
    sig_gated = apply_gate(raw_signals, gate_mask)
    sig_final = apply_window_filter_numba(sig_gated, minutes, 360, 840, FLAT_MIN)
    bt = run_backtest_vectorized(
        px_a, px_b, sig_final, beta,
        MULT_A, MULT_B, TICK_A, TICK_B, 1, 2.50, 100_000.0,
    )
    return bt


THRESHOLDS = [-3.50, -2.86, -2.57, -2.30, -2.00, -1.65, None]

print("Gate pass rates + trade quality -- OLS=3000, ZW=30, ze=2.75, zx=0.75, zs=4.50")
print("Window 06:00-14:00, time_stop=0")

for adf_w in [24, 48, 64]:
    adf_vals = adf_statistic_simple(spread, window=adf_w, step=1).values
    valid_adf = ~np.isnan(adf_vals)

    print(f"\n=== ADF window = {adf_w} ===")
    print(f"{'ADF_thresh':>11}  {'ADF_pass':>8}  {'H+C_pass':>8}  {'ALL_pass':>8}  "
          f"{'Trades':>6}  {'PF':>6}  {'PnL':>10}  {'WR':>5}  {'MaxDD':>10}")
    print("-" * 95)

    for thresh in THRESHOLDS:
        if thresh is None:
            label = "NO ADF"
            gate_all = hc_mask
            pct_adf = 100.0
        else:
            label = f"{thresh:.2f}"
            gate_adf = (adf_vals < thresh) & valid_adf
            pct_adf = gate_adf.sum() / valid_adf.sum() * 100
            gate_all = gate_adf & hc_mask

        valid_all = valid_adf & ~np.isnan(hurst_64) & ~np.isnan(corr_24)
        pct_hc = hc_mask[valid_all].sum() / valid_all.sum() * 100
        pct_all = gate_all[valid_all].sum() / valid_all.sum() * 100

        bt = run_config(gate_all)
        num = bt["trades"]
        pf = bt["profit_factor"] if num > 0 else 0
        pnl = bt["pnl"] if num > 0 else 0
        wr = bt["win_rate"] if num > 0 else 0
        eq = bt["equity"]
        dd = float((eq - np.maximum.accumulate(eq)).min()) if num > 0 else 0

        print(f"{label:>11}  {pct_adf:>7.1f}%  {pct_hc:>7.1f}%  {pct_all:>7.1f}%  "
              f"{num:>6}  {pf:>6.3f}  {pnl:>10,.2f}  {wr:>4.1f}%  {dd:>10,.2f}")
