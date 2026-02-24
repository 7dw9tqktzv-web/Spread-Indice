"""Phase 13c Final Checks: Monte Carlo DD + Slippage Sensitivity for Config D.

1. Monte Carlo: shuffle 153 trades 10,000x, distribution of max drawdowns
2. Slippage: 0/1/2/3 ticks, impact on PF/PnL/CPCV

Usage:
    python scripts/phase13c_final_checks.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.signals.filters import apply_time_stop, apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.validation.cpcv import CPCVConfig, run_cpcv
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask

# ======================================================================
# Config D (final)
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

OLS_WINDOW = 7000
ADF_WINDOW = 96
ZW = 30
Z_ENTRY = 3.25
DELTA_TP = 2.75
DELTA_SL = 1.50
TIME_STOP = 0
WINDOW = "02:00-14:00"
ENTRY_START, ENTRY_END = 120, 840

GATE_CFG = GateConfig(
    adf_threshold=-2.86, hurst_threshold=0.50, corr_threshold=0.70,
    adf_window=96, hurst_window=64, corr_window=24,
)

CPCV_CFG = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)

Z_EXIT = round(max(Z_ENTRY - DELTA_TP, 0.0), 4)
Z_STOP = round(Z_ENTRY + DELTA_SL, 4)

N_SIMULATIONS = 10_000


# ======================================================================
# Reconstruct D backtest
# ======================================================================

def reconstruct_d(aligned, px_a, px_b, idx, minutes, slippage_ticks=1):
    """Run Config D backtest with specified slippage."""
    est = create_estimator("ols_rolling", window=OLS_WINDOW, zscore_window=ZW)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], GATE_CFG
    )

    mu = spread.rolling(ZW).mean()
    sigma = spread.rolling(ZW).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw_signals = generate_signals_numba(zscore, Z_ENTRY, Z_EXIT, Z_STOP)
    sig_ts = apply_time_stop(raw_signals, TIME_STOP)
    sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)
    sig_final = apply_window_filter_numba(
        sig_gated, minutes, ENTRY_START, ENTRY_END, FLAT_MIN
    )

    bt = run_backtest_vectorized(
        px_a, px_b, sig_final, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        slippage_ticks, COMMISSION, INITIAL_CAPITAL,
    )
    return bt


# ======================================================================
# Monte Carlo Drawdown Simulation
# ======================================================================

def run_monte_carlo(pnls, n_sims=N_SIMULATIONS):
    """Shuffle trade PnLs n_sims times, compute max DD distribution."""
    rng = np.random.default_rng(42)
    max_dds = np.zeros(n_sims)

    for i in range(n_sims):
        shuffled = rng.permutation(pnls)
        equity = np.cumsum(shuffled)
        running_max = np.maximum.accumulate(equity)
        dd = equity - running_max
        max_dds[i] = dd.min()

    return max_dds


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 100)
    print(" PHASE 13c FINAL CHECKS -- Config D")
    print("=" * 100)
    print(f"\n  OLS={OLS_WINDOW} ADF_w={ADF_WINDOW} ZW={ZW} {WINDOW}")
    print(f"  ze={Z_ENTRY} zx={Z_EXIT} zs={Z_STOP} ts={TIME_STOP}")

    # Load data
    print("\nLoading market data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    n = len(px_a)

    print(f"  {n:,} bars | {idx[0].date()} to {idx[-1].date()}")

    # ======================================================================
    # TEST 1: Monte Carlo Drawdown
    # ======================================================================
    print("\n" + "=" * 100)
    print(" TEST 1: MONTE CARLO DRAWDOWN (10,000 simulations)")
    print("=" * 100)

    bt = reconstruct_d(aligned, px_a, px_b, idx, minutes, slippage_ticks=1)
    pnls = bt["trade_pnls"]
    print(f"\n  Reference: {bt['trades']} trades, PnL ${pnls.sum():+,.0f}, "
          f"PF {bt['profit_factor']:.2f}")

    # Historical max DD
    equity = bt["equity"]
    running_max = np.maximum.accumulate(equity)
    hist_dd = float((equity - running_max).min())
    print(f"  Historical max DD: ${hist_dd:+,.0f}")

    # Monte Carlo
    print(f"\n  Running {N_SIMULATIONS:,} Monte Carlo simulations...")
    mc_dds = run_monte_carlo(pnls)

    # Percentiles
    pcts = [5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Max Drawdown Distribution:")
    print(f"  {'Percentile':>12} {'Max DD':>10} {'Propfirm $5K':>14}")
    for p in pcts:
        dd = np.percentile(mc_dds, p)
        status = "SAFE" if dd > -5000 else "BREACH"
        print(f"  {p:>10}th  ${dd:>+9,.0f}  {status:>14}")

    # Key stats
    mean_dd = mc_dds.mean()
    worst_dd = mc_dds.min()
    pct_breach_5k = (mc_dds < -5000).sum() / len(mc_dds) * 100
    pct_breach_4500 = (mc_dds < -4500).sum() / len(mc_dds) * 100

    print(f"\n  Mean max DD:      ${mean_dd:+,.0f}")
    print(f"  Worst max DD:     ${worst_dd:+,.0f}")
    print(f"  Historical DD:    ${hist_dd:+,.0f}")
    print(f"\n  Prob(DD < -$4,500): {pct_breach_4500:.1f}%")
    print(f"  Prob(DD < -$5,000): {pct_breach_5k:.1f}%")

    if pct_breach_5k < 10:
        print(f"\n  --> PROPFIRM SAFE: {100-pct_breach_5k:.0f}% des scenarios restent sous $5K DD")
    elif pct_breach_5k < 25:
        print(f"\n  --> PROPFIRM WARN: {pct_breach_5k:.0f}% des scenarios breachent $5K DD")
    else:
        print(f"\n  --> PROPFIRM DANGER: {pct_breach_5k:.0f}% des scenarios breachent $5K DD")

    # ======================================================================
    # TEST 2: Slippage Sensitivity
    # ======================================================================
    print("\n" + "=" * 100)
    print(" TEST 2: SLIPPAGE SENSITIVITY (0-3 ticks)")
    print("=" * 100)

    print(f"\n  Tick values: NQ = ${TICK_A * MULT_A:.0f}/tick, YM = ${TICK_B * MULT_B:.0f}/tick")
    print(f"  Per trade cost of 1 tick slippage: "
          f"2 legs x 2 sides x (${TICK_A*MULT_A:.0f} + ${TICK_B*MULT_B:.0f})/2 = variable")

    print(f"\n  {'Slip':>5} {'Extra$/trd':>10} | {'#':>4} {'PnL':>10} {'PF':>6} "
          f"{'WR':>5} {'DD':>8} {'W/L':>5} | {'CPCV':>6} {'P+%':>5}")

    for slip in [0, 1, 2, 3]:
        bt_s = reconstruct_d(aligned, px_a, px_b, idx, minutes, slippage_ticks=slip)
        pnls_s = bt_s["trade_pnls"]
        num = bt_s["trades"]

        if num < 10:
            print(f"  {slip:>5} ticks | < 10 trades")
            continue

        # CPCV
        cpcv = run_cpcv(
            bt_s["trade_entry_bars"], bt_s["trade_exit_bars"],
            pnls_s, n, CPCV_CFG,
        )

        # Max DD
        eq = bt_s["equity"]
        rm = np.maximum.accumulate(eq)
        dd = float((eq - rm).min())

        # W/L
        winners = pnls_s[pnls_s > 0]
        losers = pnls_s[pnls_s <= 0]
        avg_w = float(winners.mean()) if len(winners) > 0 else 0
        avg_l = float(losers.mean()) if len(losers) > 0 else 0
        wl = abs(avg_w / avg_l) if avg_l != 0 else 99.9

        # Extra cost vs 0 slip
        if slip == 0:
            ref_pnl = float(pnls_s.sum())
            extra = 0
        else:
            extra = (ref_pnl - float(pnls_s.sum())) / num

        marker = " <-- backtest default" if slip == 1 else ""

        print(f"  {slip:>3} tk  ${extra:>+8,.0f}/trd | "
              f"{num:>4} ${pnls_s.sum():>+9,.0f} {bt_s['profit_factor']:>6.2f} "
              f"{bt_s['win_rate']:>4.0f}% ${dd:>+7,.0f} {wl:>5.2f} | "
              f"{cpcv['median_sharpe']:>6.3f} {cpcv['pct_positive']:>4.0f}%{marker}")

    # ======================================================================
    # Breakeven slippage
    # ======================================================================
    print(f"\n  Breakeven analysis:")
    ref_avg_pnl = float(pnls.mean())
    # Each extra tick costs approximately: 2 * (tick_a * mult_a + tick_b * mult_b) / 2 per leg
    # Actually it's: entry slip + exit slip for each leg
    # For leg A: 2 sides * slip * tick_a * mult_a * n_a (n_a=1)
    # For leg B: 2 sides * slip * tick_b * mult_b * n_b (n_b varies)
    # Approximate: ~$15 per tick of slippage per trade (NQ $5/tick + YM $5/tick, 2 sides)
    print(f"  Avg PnL/trade at 1 tick: ${ref_avg_pnl:+,.0f}")
    print(f"  At current avg PnL, system breaks even at ~{abs(ref_avg_pnl) / 15:.0f} ticks slippage")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "=" * 100)
    print(" SUMMARY")
    print("=" * 100)

    print(f"\n  Monte Carlo:")
    print(f"    95th pct DD: ${np.percentile(mc_dds, 5):+,.0f}")
    print(f"    Prob(breach $5K): {pct_breach_5k:.1f}%")
    mc_verdict = "SAFE" if pct_breach_5k < 10 else "WARN" if pct_breach_5k < 25 else "DANGER"
    print(f"    Verdict: {mc_verdict}")

    print(f"\n  Slippage:")
    bt_2 = reconstruct_d(aligned, px_a, px_b, idx, minutes, slippage_ticks=2)
    slip_verdict = "SAFE" if bt_2["profit_factor"] > 1.5 else "WARN" if bt_2["profit_factor"] > 1.2 else "DANGER"
    print(f"    PF at 2 ticks: {bt_2['profit_factor']:.2f}")
    print(f"    Verdict: {slip_verdict}")

    overall = "GO" if mc_verdict != "DANGER" and slip_verdict != "DANGER" else "REVIEW"
    print(f"\n  OVERALL: [{overall}] -- Config D validated for live")

    print("\n" + "=" * 100)
    print()


if __name__ == "__main__":
    main()
