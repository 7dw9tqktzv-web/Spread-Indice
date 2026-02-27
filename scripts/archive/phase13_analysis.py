"""Phase 13 Analysis: DSR + Neighborhood + Gate Sensitivity + Walk-Forward + Propfirm.

Reads phase13_grid_cpcv.csv and runs 5 post-grid analyses on the top N configs.

Usage:
    python scripts/phase13_analysis.py --top 10
    python scripts/phase13_analysis.py --top 5 --skip-wf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.validation.deflated_sharpe import compute_dsr_for_config
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask
from src.validation.neighborhood import (
    compute_neighborhood_robustness,
    get_neighbor_configs,
)
from src.validation.propfirm import PropfirmConfig, compute_propfirm_metrics

# ======================================================================
# Constants (same as grid script)
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0  # NQ, YM
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930  # 15:30 CT

GATE_CFG = GateConfig()  # default windows: adf=24, hurst=64, corr=24

CPCV_CFG = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)

# Grid axes for neighborhood lookup
GRID_AXES = {
    "ols": [2000, 3000, 4000, 5000],
    "zw": [15, 20, 30, 40, 50],
    "z_entry": [2.25, 2.50, 2.75, 3.00, 3.25],
    "z_exit": [0.25, 0.50, 0.75, 1.00, 1.25],
    "z_stop": [3.50, 4.00, 4.50, 5.00],
    "time_stop": [0, 6, 12, 20],
}

WINDOW_MAP = {
    "02:00-14:00": (120, 840),
    "06:00-14:00": (360, 840),
}

# Gate sensitivity ranges (one at a time, defaults adf=24, hurst=64, corr=24)
ADF_WINDOWS = [12, 24, 36, 48, 64]
HURST_WINDOWS = [32, 48, 64, 96, 128]
CORR_WINDOWS = [12, 24, 36, 48]

# Walk-forward config: IS=2y, OOS=6m, step=6m
WF_IS_BARS = 2 * 252 * 78  # ~2 years of 5-min bars (252 days, 78 bars/day)
WF_OOS_BARS = 126 * 78     # ~6 months
WF_STEP_BARS = 126 * 78    # ~6 months step


# ======================================================================
# Helper: reconstruct a single config's backtest
# ======================================================================

def reconstruct_backtest(row, aligned, px_a, px_b, idx, minutes):
    """Run a single config and return full backtest result."""
    ols_window = int(row["ols"])
    zw = int(row["zw"])
    z_entry = float(row["z_entry"])
    z_exit = float(row["z_exit"])
    z_stop = float(row["z_stop"])
    ts = int(row["time_stop"])
    win_name = row["window"]
    entry_start, entry_end = WINDOW_MAP[win_name]

    est = create_estimator("ols_rolling", window=ols_window, zscore_window=zw)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], GATE_CFG
    )

    mu = spread.rolling(zw).mean()
    sigma = spread.rolling(zw).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)
    sig_ts = apply_time_stop(raw_signals, ts)
    sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)
    sig_final = apply_window_filter_numba(
        sig_gated, minutes, entry_start, entry_end, FLAT_MIN
    )

    bt = run_backtest_vectorized(
        px_a, px_b, sig_final, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )
    return bt, spread, beta, zscore


# ======================================================================
# Section 1: Deflated Sharpe Ratio
# ======================================================================

def run_dsr_analysis(df_grid, top_rows, aligned, px_a, px_b, idx, minutes):
    """Compute DSR for top configs using all grid Sharpes as benchmark."""
    print("\n" + "=" * 100)
    print(" SECTION 1: DEFLATED SHARPE RATIO")
    print("=" * 100)

    all_sharpes = df_grid["cpcv_median_sharpe"].values
    n_trials = len(df_grid)

    results = []
    for i, (_, row) in enumerate(top_rows.iterrows()):
        bt, *_ = reconstruct_backtest(row, aligned, px_a, px_b, idx, minutes)
        pnls = bt["trade_pnls"]
        observed_sr = float(row["cpcv_median_sharpe"])

        dsr_result = compute_dsr_for_config(observed_sr, pnls, n_trials, all_sharpes)
        dsr_result["rank"] = i + 1
        dsr_result["config"] = (
            f"OLS={int(row['ols'])} ZW={int(row['zw'])} "
            f"ze={row['z_entry']} zx={row['z_exit']} zs={row['z_stop']} "
            f"ts={int(row['time_stop'])} {row['window']}"
        )
        results.append(dsr_result)

        status = "PASS" if dsr_result["dsr"] > 0.95 else "FAIL"
        print(f"\n  #{i+1} {dsr_result['config']}")
        print(f"      Observed SR: {dsr_result['observed_sharpe']:.4f}  "
              f"Benchmark E[max]: {dsr_result['sr_benchmark']:.4f}  "
              f"DSR: {dsr_result['dsr']:.4f}  [{status}]")
        print(f"      n_trades={dsr_result['n_trades']}  "
              f"skew={dsr_result['skewness']:.2f}  kurt={dsr_result['kurtosis']:.2f}")

    n_pass = sum(1 for r in results if r["dsr"] > 0.95)
    print(f"\n  DSR > 0.95: {n_pass} / {len(results)}")
    return results


# ======================================================================
# Section 2: Neighborhood Robustness
# ======================================================================

def run_neighborhood_analysis(df_grid, top_rows):
    """Check that neighbors of top configs are also profitable."""
    print("\n" + "=" * 100)
    print(" SECTION 2: NEIGHBORHOOD ROBUSTNESS")
    print("=" * 100)

    results = []
    for i, (_, row) in enumerate(top_rows.iterrows()):
        center = {
            "ols": int(row["ols"]),
            "zw": int(row["zw"]),
            "z_entry": float(row["z_entry"]),
            "z_exit": float(row["z_exit"]),
            "z_stop": float(row["z_stop"]),
            "time_stop": int(row["time_stop"]),
        }

        neighbors = get_neighbor_configs(center, GRID_AXES)
        neighbor_sharpes = []
        neighbor_pfs = []

        for nb in neighbors:
            # Lookup neighbor in grid CSV (same window)
            mask = (
                (df_grid["ols"] == nb["ols"])
                & (df_grid["zw"] == nb["zw"])
                & (df_grid["z_entry"] == nb["z_entry"])
                & (df_grid["z_exit"] == nb["z_exit"])
                & (df_grid["z_stop"] == nb["z_stop"])
                & (df_grid["time_stop"] == nb["time_stop"])
                & (df_grid["window"] == row["window"])
            )
            matches = df_grid[mask]
            if len(matches) > 0:
                neighbor_sharpes.append(float(matches.iloc[0]["cpcv_median_sharpe"]))
                neighbor_pfs.append(float(matches.iloc[0]["pf"]))

        nr = compute_neighborhood_robustness(
            float(row["cpcv_median_sharpe"]),
            neighbor_sharpes,
            neighbor_pfs,
        )
        results.append(nr)

        status = "ROBUST" if nr.is_robust else "NOT ROBUST"
        print(f"\n  #{i+1} OLS={int(row['ols'])} ZW={int(row['zw'])} "
              f"ze={row['z_entry']} zx={row['z_exit']} zs={row['z_stop']} "
              f"ts={int(row['time_stop'])} {row['window']}")
        print(f"      Center SR: {nr.center_sharpe:.4f}  "
              f"Neighbors: {nr.n_neighbors} found / {len(neighbors)} expected")
        print(f"      Profitable: {nr.pct_profitable:.0f}%  "
              f"Mean neighbor SR: {nr.mean_neighbor_sharpe:.4f}  "
              f"Min: {nr.min_neighbor_sharpe:.4f}  "
              f"Degradation: {nr.sharpe_degradation_pct:.1f}%  [{status}]")

    n_robust = sum(1 for r in results if r.is_robust)
    print(f"\n  Robust: {n_robust} / {len(results)}")
    return results


# ======================================================================
# Section 3: Gate Sensitivity (rolling window sweep)
# ======================================================================

def run_gate_sensitivity(top_row, aligned, px_a, px_b, idx, minutes):
    """Sweep gate rolling windows one at a time for the #1 config."""
    print("\n" + "=" * 100)
    print(" SECTION 3: GATE ROLLING WINDOW SENSITIVITY (Config #1)")
    print("=" * 100)

    ols_window = int(top_row["ols"])
    zw = int(top_row["zw"])
    z_entry = float(top_row["z_entry"])
    z_exit = float(top_row["z_exit"])
    z_stop = float(top_row["z_stop"])
    ts = int(top_row["time_stop"])
    win_name = top_row["window"]
    entry_start, entry_end = WINDOW_MAP[win_name]
    n = len(px_a)

    est = create_estimator("ols_rolling", window=ols_window, zscore_window=zw)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    mu = spread.rolling(zw).mean()
    sigma = spread.rolling(zw).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)
    sig_ts = apply_time_stop(raw_signals, ts)

    def run_with_gate_cfg(cfg):
        gate_mask = compute_gate_mask(
            spread, aligned.df["close_a"], aligned.df["close_b"], cfg
        )
        sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)
        sig_final = apply_window_filter_numba(
            sig_gated, minutes, entry_start, entry_end, FLAT_MIN
        )
        bt = run_backtest_vectorized(
            px_a, px_b, sig_final, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )
        num = bt["trades"]
        if num < 10:
            return {"trades": num, "pf": 0, "pnl": 0, "cpcv_med": 0}
        cpcv = run_cpcv(
            bt["trade_entry_bars"], bt["trade_exit_bars"],
            bt["trade_pnls"], n, CPCV_CFG,
        )
        return {
            "trades": num,
            "pf": round(bt["profit_factor"], 3),
            "pnl": round(bt["pnl"], 2),
            "cpcv_med": round(cpcv["median_sharpe"], 4),
        }

    # Sweep ADF window
    print(f"\n  ADF window sweep (Hurst={GATE_CFG.hurst_window}, Corr={GATE_CFG.corr_window}):")
    print(f"  {'ADF_w':>6}  {'Trades':>6}  {'PF':>7}  {'PnL':>10}  {'CPCV_med':>9}")
    for aw in ADF_WINDOWS:
        cfg = GateConfig(adf_window=aw, hurst_window=GATE_CFG.hurst_window,
                         corr_window=GATE_CFG.corr_window)
        r = run_with_gate_cfg(cfg)
        marker = " <-- default" if aw == GATE_CFG.adf_window else ""
        print(f"  {aw:>6}  {r['trades']:>6}  {r['pf']:>7.3f}  {r['pnl']:>10,.2f}  "
              f"{r['cpcv_med']:>9.4f}{marker}")

    # Sweep Hurst window
    print(f"\n  Hurst window sweep (ADF={GATE_CFG.adf_window}, Corr={GATE_CFG.corr_window}):")
    print(f"  {'Hurst_w':>7}  {'Trades':>6}  {'PF':>7}  {'PnL':>10}  {'CPCV_med':>9}")
    for hw in HURST_WINDOWS:
        cfg = GateConfig(adf_window=GATE_CFG.adf_window, hurst_window=hw,
                         corr_window=GATE_CFG.corr_window)
        r = run_with_gate_cfg(cfg)
        marker = " <-- default" if hw == GATE_CFG.hurst_window else ""
        print(f"  {hw:>7}  {r['trades']:>6}  {r['pf']:>7.3f}  {r['pnl']:>10,.2f}  "
              f"{r['cpcv_med']:>9.4f}{marker}")

    # Sweep Corr window
    print(f"\n  Corr window sweep (ADF={GATE_CFG.adf_window}, Hurst={GATE_CFG.hurst_window}):")
    print(f"  {'Corr_w':>6}  {'Trades':>6}  {'PF':>7}  {'PnL':>10}  {'CPCV_med':>9}")
    for cw in CORR_WINDOWS:
        cfg = GateConfig(adf_window=GATE_CFG.adf_window, hurst_window=GATE_CFG.hurst_window,
                         corr_window=cw)
        r = run_with_gate_cfg(cfg)
        marker = " <-- default" if cw == GATE_CFG.corr_window else ""
        print(f"  {cw:>6}  {r['trades']:>6}  {r['pf']:>7.3f}  {r['pnl']:>10,.2f}  "
              f"{r['cpcv_med']:>9.4f}{marker}")


# ======================================================================
# Section 4: Walk-Forward (IS=2y, OOS=6m, step=6m)
# ======================================================================

def run_walkforward(top_row, aligned, px_a, px_b, idx, minutes):
    """Sequential walk-forward with fixed params, rebuilt hedge per fold."""
    print("\n" + "=" * 100)
    print(" SECTION 4: WALK-FORWARD VALIDATION (Config #1)")
    print("=" * 100)

    ols_window = int(top_row["ols"])
    zw = int(top_row["zw"])
    z_entry = float(top_row["z_entry"])
    z_exit = float(top_row["z_exit"])
    z_stop = float(top_row["z_stop"])
    ts = int(top_row["time_stop"])
    win_name = top_row["window"]
    entry_start, entry_end = WINDOW_MAP[win_name]

    n = len(px_a)

    # Build folds
    folds = []
    start = 0
    while start + WF_IS_BARS + WF_OOS_BARS <= n:
        is_start = start
        is_end = start + WF_IS_BARS
        oos_start = is_end
        oos_end = min(is_end + WF_OOS_BARS, n)
        folds.append((is_start, is_end, oos_start, oos_end))
        start += WF_STEP_BARS

    if not folds:
        print("  Not enough data for walk-forward.")
        return []

    print(f"\n  {len(folds)} folds: IS={WF_IS_BARS} bars (~2y), "
          f"OOS={WF_OOS_BARS} bars (~6m), step={WF_STEP_BARS} bars")

    fold_results = []
    for fi, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        # OOS only: run backtest on full data, then extract OOS trades
        est = create_estimator("ols_rolling", window=ols_window, zscore_window=zw)
        hr = est.estimate(aligned)
        beta = hr.beta.values
        spread = hr.spread

        gate_mask = compute_gate_mask(
            spread, aligned.df["close_a"], aligned.df["close_b"], GATE_CFG
        )

        mu = spread.rolling(zw).mean()
        sigma = spread.rolling(zw).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(zscore, dtype=np.float64)

        raw_signals = generate_signals_numba(zscore, z_entry, z_exit, z_stop)
        sig_ts = apply_time_stop(raw_signals, ts)
        sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)
        sig_final = apply_window_filter_numba(
            sig_gated, minutes, entry_start, entry_end, FLAT_MIN
        )

        bt = run_backtest_vectorized(
            px_a, px_b, sig_final, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        # Filter trades to OOS window
        entries = bt["trade_entry_bars"]
        exits = bt["trade_exit_bars"]
        pnls = bt["trade_pnls"]

        oos_mask = (entries >= oos_s) & (exits <= oos_e)
        oos_pnls = pnls[oos_mask]
        oos_trades = len(oos_pnls)
        oos_pnl = float(oos_pnls.sum()) if oos_trades > 0 else 0.0

        gross_profit = float(oos_pnls[oos_pnls > 0].sum()) if oos_trades > 0 else 0.0
        gross_loss = abs(float(oos_pnls[oos_pnls < 0].sum())) if oos_trades > 0 else 0.0
        oos_pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

        oos_wr = 0.0
        if oos_trades > 0:
            oos_wr = float((oos_pnls > 0).sum() / oos_trades * 100)

        is_go = oos_pf > 1.0 and oos_trades >= 5
        status = "GO" if is_go else "NO-GO"

        is_dates = f"{idx[is_s].date()} to {idx[is_e-1].date()}"
        oos_dates = f"{idx[oos_s].date()} to {idx[min(oos_e-1, n-1)].date()}"

        fold_results.append({
            "fold": fi + 1,
            "is_dates": is_dates,
            "oos_dates": oos_dates,
            "oos_trades": oos_trades,
            "oos_pnl": round(oos_pnl, 2),
            "oos_pf": round(oos_pf, 3),
            "oos_wr": round(oos_wr, 1),
            "is_go": is_go,
        })

        print(f"\n  Fold {fi+1}: IS {is_dates}  |  OOS {oos_dates}")
        print(f"    OOS: {oos_trades} trades, PnL ${oos_pnl:,.2f}, "
              f"PF {oos_pf:.3f}, WR {oos_wr:.1f}%  [{status}]")

    n_go = sum(1 for f in fold_results if f["is_go"])
    print(f"\n  Walk-Forward: {n_go}/{len(fold_results)} folds GO")
    return fold_results


# ======================================================================
# Section 5: Propfirm $150k Compliance
# ======================================================================

def run_propfirm_analysis(top_rows, aligned, px_a, px_b, idx, minutes):
    """Check propfirm compliance for top configs."""
    print("\n" + "=" * 100)
    print(" SECTION 5: PROPFIRM $150K COMPLIANCE")
    print("=" * 100)

    bar_dates = idx.date

    config = PropfirmConfig()
    print(f"\n  Account: ${config.account_size:,.0f}  "
          f"Max daily loss: ${config.max_daily_loss:,.0f}  "
          f"Trailing DD: ${config.trailing_max_dd:,.0f}  "
          f"Target: ${config.daily_target:,.0f}/day")

    results = []
    for i, (_, row) in enumerate(top_rows.iterrows()):
        bt, *_ = reconstruct_backtest(row, aligned, px_a, px_b, idx, minutes)

        pr = compute_propfirm_metrics(
            bt["trade_entry_bars"],
            bt["trade_exit_bars"],
            bt["trade_pnls"],
            bar_dates,
            bt["equity"],
            config,
        )
        results.append(pr)

        status = "COMPLIANT" if pr.is_compliant else "VIOLATION"
        print(f"\n  #{i+1} OLS={int(row['ols'])} ZW={int(row['zw'])} "
              f"ze={row['z_entry']} zx={row['z_exit']} zs={row['z_stop']} "
              f"ts={int(row['time_stop'])} {row['window']}")
        print(f"      Trading days: {pr.n_trading_days}  "
              f"Avg daily PnL: ${pr.avg_daily_pnl:,.2f}  "
              f"Profitable days: {pr.pct_days_profitable:.1f}%")
        print(f"      Max daily loss: ${pr.max_daily_loss_observed:,.2f}  "
              f"Exceed daily limit: {pr.n_days_exceed_daily_limit} days")
        print(f"      Max trailing DD: ${pr.max_trailing_dd:,.2f}  "
              f"Exceed trailing DD: {pr.n_days_exceed_trailing_dd} bars  [{status}]")

    n_compliant = sum(1 for r in results if r.is_compliant)
    print(f"\n  Propfirm compliant: {n_compliant} / {len(results)}")
    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 13 Post-Grid Analysis")
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to analyze")
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward (slower section)")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip gate sensitivity")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "output" / "NQ_YM" / "phase13_grid_cpcv.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run phase13_grid_cpcv.py first.")
        return

    df_grid = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df_grid)} configs from {csv_path}")
    print(f"Top CPCV median Sharpe: {df_grid['cpcv_median_sharpe'].max():.4f}")

    top_rows = df_grid.head(args.top)

    # Load data once
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    # Section 1: DSR
    dsr_results = run_dsr_analysis(df_grid, top_rows, aligned, px_a, px_b, idx, minutes)

    # Section 2: Neighborhood
    nb_results = run_neighborhood_analysis(df_grid, top_rows)

    # Section 3: Gate Sensitivity
    if not args.skip_sensitivity:
        run_gate_sensitivity(top_rows.iloc[0], aligned, px_a, px_b, idx, minutes)
    else:
        print("\n  [Skipping gate sensitivity]")

    # Section 4: Walk-Forward
    if not args.skip_wf:
        wf_results = run_walkforward(top_rows.iloc[0], aligned, px_a, px_b, idx, minutes)
    else:
        print("\n  [Skipping walk-forward]")

    # Section 5: Propfirm
    pf_results = run_propfirm_analysis(top_rows, aligned, px_a, px_b, idx, minutes)

    # ======================================================================
    # Final Summary
    # ======================================================================
    print("\n" + "=" * 100)
    print(" FINAL SUMMARY")
    print("=" * 100)

    for i, (_, row) in enumerate(top_rows.iterrows()):
        dsr_pass = dsr_results[i]["dsr"] > 0.95 if i < len(dsr_results) else False
        nb_pass = nb_results[i].is_robust if i < len(nb_results) else False
        pf_pass = pf_results[i].is_compliant if i < len(pf_results) else False

        flags = []
        if dsr_pass:
            flags.append("DSR")
        if nb_pass:
            flags.append("ROBUST")
        if pf_pass:
            flags.append("PROPFIRM")

        verdict = " + ".join(flags) if flags else "---"
        all_pass = dsr_pass and nb_pass and pf_pass

        print(f"\n  #{i+1} OLS={int(row['ols'])} ZW={int(row['zw'])} "
              f"ze={row['z_entry']} zx={row['z_exit']} zs={row['z_stop']} "
              f"ts={int(row['time_stop'])} {row['window']}")
        print(f"      CPCV med SR: {row['cpcv_median_sharpe']:.4f}  "
              f"Trades: {int(row['trades'])}  PnL: ${row['pnl']:,.2f}  PF: {row['pf']:.3f}")
        print(f"      DSR: {dsr_results[i]['dsr']:.4f}  "
              f"Neighborhood: {'ROBUST' if nb_pass else 'FRAGILE'}  "
              f"Propfirm: {'OK' if pf_pass else 'VIOLATION'}")
        print(f"      --> {verdict}" + ("  *** CANDIDATE ***" if all_pass else ""))

    if not args.skip_wf:
        print(f"\n  Walk-Forward (Config #1): "
              f"{sum(1 for f in wf_results if f['is_go'])}/{len(wf_results)} GO")

    print()


if __name__ == "__main__":
    main()
