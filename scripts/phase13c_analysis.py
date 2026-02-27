"""Phase 13c Analysis: DSR + Neighborhood + Hurst/Corr Sensitivity + WF + Propfirm.

Analyzes 4 specific candidates from phase13c_grid_massif.csv.

Usage:
    python scripts/phase13c_analysis.py
    python scripts/phase13c_analysis.py --skip-wf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized

# ======================================================================
# Constants
# ======================================================================
from src.config.instruments import DEFAULT_SLIPPAGE_TICKS, get_pair_specs
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

_NQ, _YM = get_pair_specs("NQ", "YM")
MULT_A, MULT_B = _NQ.multiplier, _YM.multiplier
TICK_A, TICK_B = _NQ.tick_size, _YM.tick_size
SLIPPAGE = DEFAULT_SLIPPAGE_TICKS
COMMISSION = _NQ.commission
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

CPCV_CFG = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)

WINDOW_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
}

GATE_ADF_THRESH = -2.86
GATE_HURST_THRESH = 0.50
GATE_CORR_THRESH = 0.70
GATE_HURST_WINDOW = 64
GATE_CORR_WINDOW = 24

# Grid axes for neighborhood (Phase 13c grid)
GRID_AXES = {
    "ols": [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000],
    "adf_w": [12, 18, 24, 30, 36, 48, 64, 96, 128],
    "zw": [10, 15, 20, 25, 28, 30, 35, 40, 45, 50, 60],
    "z_entry": [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75],
    "delta_tp": [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],
    "delta_sl": [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50],
    "time_stop": [0, 3, 6, 10, 12, 16, 20, 30, 40, 50],
}

HURST_WINDOWS = [32, 48, 64, 96, 128]
CORR_WINDOWS = [12, 24, 36, 48]

WF_IS_BARS = 2 * 252 * 78
WF_OOS_BARS = 126 * 78
WF_STEP_BARS = 126 * 78

# 4 candidates (grid params)
CANDIDATES = {
    "A": {"ols": 6000, "adf_w": 30, "zw": 28, "window": "04:00-14:00",
           "z_entry": 3.25, "delta_tp": 2.75, "delta_sl": 1.25, "time_stop": 12},
    "B": {"ols": 7000, "adf_w": 96, "zw": 25, "window": "06:00-14:00",
           "z_entry": 3.00, "delta_tp": 2.75, "delta_sl": 1.50, "time_stop": 30},
    "C": {"ols": 7000, "adf_w": 30, "zw": 15, "window": "04:00-14:00",
           "z_entry": 2.50, "delta_tp": 2.50, "delta_sl": 1.00, "time_stop": 0},
    "D": {"ols": 7000, "adf_w": 96, "zw": 30, "window": "02:00-14:00",
           "z_entry": 3.25, "delta_tp": 2.75, "delta_sl": 1.50, "time_stop": 0},
    "T1": {"ols": 7000, "adf_w": 96, "zw": 15, "window": "04:00-14:00",
            "z_entry": 2.25, "delta_tp": 1.75, "delta_sl": 1.25, "time_stop": 0},
}


def derive_params(cfg):
    """Derive z_exit, z_stop from delta params."""
    z_exit = round(max(cfg["z_entry"] - cfg["delta_tp"], 0.0), 4)
    z_stop = round(cfg["z_entry"] + cfg["delta_sl"], 4)
    return z_exit, z_stop


def cfg_label(name, cfg):
    z_exit, z_stop = derive_params(cfg)
    return (f"{name}: OLS={cfg['ols']} ADF_w={cfg['adf_w']} ZW={cfg['zw']} "
            f"{cfg['window']} ze={cfg['z_entry']} zx={z_exit} zs={z_stop} ts={cfg['time_stop']}")


# ======================================================================
# Reconstruct backtest
# ======================================================================

def reconstruct_backtest(cfg, aligned, px_a, px_b, idx, minutes):
    """Run a single config and return full backtest result."""
    z_exit, z_stop = derive_params(cfg)
    entry_start, entry_end = WINDOW_MAP[cfg["window"]]

    est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    gate_cfg = GateConfig(
        adf_threshold=GATE_ADF_THRESH,
        hurst_threshold=GATE_HURST_THRESH,
        corr_threshold=GATE_CORR_THRESH,
        adf_window=cfg["adf_w"],
        hurst_window=GATE_HURST_WINDOW,
        corr_window=GATE_CORR_WINDOW,
    )
    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg
    )

    mu = spread.rolling(cfg["zw"]).mean()
    sigma = spread.rolling(cfg["zw"]).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw_signals = generate_signals_numba(zscore, cfg["z_entry"], z_exit, z_stop)
    sig_ts = apply_time_stop(raw_signals, cfg["time_stop"])
    sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)
    sig_final = apply_window_filter_numba(
        sig_gated, minutes, entry_start, entry_end, FLAT_MIN
    )

    bt = run_backtest_vectorized(
        px_a, px_b, sig_final, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )
    return bt, spread, beta, zscore, gate_mask


# ======================================================================
# Section 1: DSR
# ======================================================================

def run_dsr_analysis(candidates, all_sharpes, aligned, px_a, px_b, idx, minutes):
    print("\n" + "=" * 100)
    print(" SECTION 1: DEFLATED SHARPE RATIO")
    print("=" * 100)

    n_trials = 24_710_400  # total grid combos

    results = {}
    for name, cfg in candidates.items():
        bt, *_ = reconstruct_backtest(cfg, aligned, px_a, px_b, idx, minutes)
        pnls = bt["trade_pnls"]

        # Use CPCV median Sharpe as observed SR
        bt["equity"]
        sharpe_full = 0.0
        if pnls.std() > 1e-12:
            sharpe_full = float(pnls.mean() / pnls.std())

        # For DSR, use the full-sample Sharpe from trade PnLs
        dsr_result = compute_dsr_for_config(sharpe_full, pnls, n_trials, all_sharpes)
        results[name] = dsr_result

        status = "PASS" if dsr_result["dsr"] > 0.95 else "FAIL"
        print(f"\n  {cfg_label(name, cfg)}")
        print(f"      Observed SR: {dsr_result['observed_sharpe']:.4f}  "
              f"Benchmark E[max]: {dsr_result['sr_benchmark']:.4f}  "
              f"DSR: {dsr_result['dsr']:.4f}  [{status}]")
        print(f"      n_trades={dsr_result['n_trades']}  "
              f"skew={dsr_result['skewness']:.2f}  kurt={dsr_result['kurtosis']:.2f}")

    return results


# ======================================================================
# Section 2: Neighborhood
# ======================================================================

def run_neighborhood_analysis(candidates, df_grid):
    print("\n" + "=" * 100)
    print(" SECTION 2: NEIGHBORHOOD ROBUSTNESS")
    print("=" * 100)

    results = {}
    for name, cfg in candidates.items():
        center = {k: cfg[k] for k in GRID_AXES.keys()}
        neighbors = get_neighbor_configs(center, GRID_AXES)

        neighbor_sharpes = []
        neighbor_pfs = []
        neighbor_details = []

        for nb in neighbors:
            # Lookup in CSV by grid params + same window
            mask = (
                (df_grid["ols"] == nb["ols"])
                & (df_grid["adf_w"] == nb["adf_w"])
                & (df_grid["zw"] == nb["zw"])
                & (df_grid["window"] == cfg["window"])
                & (np.isclose(df_grid["z_entry"], nb["z_entry"]))
                & (np.isclose(df_grid["delta_tp"], nb["delta_tp"]))
                & (np.isclose(df_grid["delta_sl"], nb["delta_sl"]))
                & (df_grid["time_stop"] == nb["time_stop"])
            )
            matches = df_grid[mask]
            if len(matches) > 0:
                row = matches.iloc[0]
                neighbor_sharpes.append(float(row["cpcv_median_sharpe"]))
                neighbor_pfs.append(float(row["pf"]))
                # Track which param changed
                changed = [k for k in center if nb[k] != center[k]]
                neighbor_details.append({
                    "changed": changed[0] if changed else "?",
                    "value": nb[changed[0]] if changed else "?",
                    "sharpe": float(row["cpcv_median_sharpe"]),
                    "pf": float(row["pf"]),
                    "trades": int(row["trades"]),
                })

        # Get center's CPCV median Sharpe from CSV
        center_mask = (
            (df_grid["ols"] == cfg["ols"])
            & (df_grid["adf_w"] == cfg["adf_w"])
            & (df_grid["zw"] == cfg["zw"])
            & (df_grid["window"] == cfg["window"])
            & (np.isclose(df_grid["z_entry"], cfg["z_entry"]))
            & (np.isclose(df_grid["delta_tp"], cfg["delta_tp"]))
            & (np.isclose(df_grid["delta_sl"], cfg["delta_sl"]))
            & (df_grid["time_stop"] == cfg["time_stop"])
        )
        center_rows = df_grid[center_mask]
        center_sharpe = float(center_rows.iloc[0]["cpcv_median_sharpe"]) if len(center_rows) > 0 else 0.0

        nr = compute_neighborhood_robustness(center_sharpe, neighbor_sharpes, neighbor_pfs)
        results[name] = nr

        status = "ROBUST" if nr.is_robust else "NOT ROBUST"
        print(f"\n  {cfg_label(name, cfg)}")
        print(f"      Center CPCV SR: {nr.center_sharpe:.4f}  "
              f"Neighbors: {nr.n_neighbors} found / {len(neighbors)} expected")
        print(f"      Profitable: {nr.pct_profitable:.0f}%  "
              f"Mean neighbor SR: {nr.mean_neighbor_sharpe:.4f}  "
              f"Min: {nr.min_neighbor_sharpe:.4f}  "
              f"Degradation: {nr.sharpe_degradation_pct:.1f}%  [{status}]")

        # Detail by dimension
        if neighbor_details:
            print(f"      {'Param':>10} {'Value':>8} {'CPCV_SR':>9} {'PF':>7} {'Trades':>7}")
            for d in sorted(neighbor_details, key=lambda x: x["changed"]):
                print(f"      {d['changed']:>10} {str(d['value']):>8} "
                      f"{d['sharpe']:>9.4f} {d['pf']:>7.3f} {d['trades']:>7}")

    return results


# ======================================================================
# Section 3: Hurst/Corr Gate Sensitivity
# ======================================================================

def run_gate_sensitivity(candidates, aligned, px_a, px_b, idx, minutes):
    print("\n" + "=" * 100)
    print(" SECTION 3: HURST/CORR GATE SENSITIVITY (all candidates)")
    print("=" * 100)
    print("  ADF window is per-config (in grid). Only sweeping Hurst + Corr.")

    n = len(px_a)

    for name, cfg in candidates.items():
        z_exit, z_stop = derive_params(cfg)
        entry_start, entry_end = WINDOW_MAP[cfg["window"]]

        est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
        hr = est.estimate(aligned)
        beta = hr.beta.values
        spread = hr.spread

        mu = spread.rolling(cfg["zw"]).mean()
        sigma = spread.rolling(cfg["zw"]).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(zscore, dtype=np.float64)

        raw_signals = generate_signals_numba(zscore, cfg["z_entry"], z_exit, z_stop)
        sig_ts = apply_time_stop(raw_signals, cfg["time_stop"])

        def run_with_gate(
            adf_w, hurst_w, corr_w,
            spread=spread, sig_ts=sig_ts,
            entry_start=entry_start, entry_end=entry_end, beta=beta,
        ):
            gate_cfg = GateConfig(
                adf_threshold=GATE_ADF_THRESH,
                hurst_threshold=GATE_HURST_THRESH,
                corr_threshold=GATE_CORR_THRESH,
                adf_window=adf_w,
                hurst_window=hurst_w,
                corr_window=corr_w,
            )
            gate_mask = compute_gate_mask(
                spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg
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
            if num < 5:
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

        print(f"\n  --- {cfg_label(name, cfg)} ---")

        # Hurst sweep
        print(f"\n  Hurst window sweep (ADF={cfg['adf_w']}, Corr={GATE_CORR_WINDOW}):")
        print(f"  {'Hurst_w':>7}  {'Trades':>6}  {'PF':>7}  {'PnL':>10}  {'CPCV_med':>9}")
        for hw in HURST_WINDOWS:
            r = run_with_gate(cfg["adf_w"], hw, GATE_CORR_WINDOW)
            marker = " <-- default" if hw == GATE_HURST_WINDOW else ""
            print(f"  {hw:>7}  {r['trades']:>6}  {r['pf']:>7.3f}  {r['pnl']:>10,.2f}  "
                  f"{r['cpcv_med']:>9.4f}{marker}")

        # Corr sweep
        print(f"\n  Corr window sweep (ADF={cfg['adf_w']}, Hurst={GATE_HURST_WINDOW}):")
        print(f"  {'Corr_w':>6}  {'Trades':>6}  {'PF':>7}  {'PnL':>10}  {'CPCV_med':>9}")
        for cw in CORR_WINDOWS:
            r = run_with_gate(cfg["adf_w"], GATE_HURST_WINDOW, cw)
            marker = " <-- default" if cw == GATE_CORR_WINDOW else ""
            print(f"  {cw:>6}  {r['trades']:>6}  {r['pf']:>7.3f}  {r['pnl']:>10,.2f}  "
                  f"{r['cpcv_med']:>9.4f}{marker}")


# ======================================================================
# Section 4: Walk-Forward
# ======================================================================

def run_walkforward(candidates, aligned, px_a, px_b, idx, minutes):
    print("\n" + "=" * 100)
    print(" SECTION 4: WALK-FORWARD VALIDATION (all candidates)")
    print("=" * 100)

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

    print(f"\n  {len(folds)} folds: IS={WF_IS_BARS} bars (~2y), "
          f"OOS={WF_OOS_BARS} bars (~6m), step={WF_STEP_BARS} bars")

    all_results = {}
    for name, cfg in candidates.items():
        z_exit, z_stop = derive_params(cfg)
        entry_start, entry_end = WINDOW_MAP[cfg["window"]]

        print(f"\n  --- {cfg_label(name, cfg)} ---")

        est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
        hr = est.estimate(aligned)
        beta = hr.beta.values
        spread = hr.spread

        gate_cfg = GateConfig(
            adf_threshold=GATE_ADF_THRESH,
            hurst_threshold=GATE_HURST_THRESH,
            corr_threshold=GATE_CORR_THRESH,
            adf_window=cfg["adf_w"],
            hurst_window=GATE_HURST_WINDOW,
            corr_window=GATE_CORR_WINDOW,
        )
        gate_mask = compute_gate_mask(
            spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg
        )

        mu = spread.rolling(cfg["zw"]).mean()
        sigma = spread.rolling(cfg["zw"]).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(zscore, dtype=np.float64)

        raw_signals = generate_signals_numba(zscore, cfg["z_entry"], z_exit, z_stop)
        sig_ts = apply_time_stop(raw_signals, cfg["time_stop"])
        sig_gated = apply_gate_filter_numba(sig_ts, gate_mask)
        sig_final = apply_window_filter_numba(
            sig_gated, minutes, entry_start, entry_end, FLAT_MIN
        )

        bt = run_backtest_vectorized(
            px_a, px_b, sig_final, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        entries = bt["trade_entry_bars"]
        exits = bt["trade_exit_bars"]
        pnls = bt["trade_pnls"]

        fold_results = []
        for fi, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
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

            is_go = oos_pf > 1.0 and oos_trades >= 3
            status = "GO" if is_go else "NO-GO"

            is_dates = f"{idx[is_s].date()} to {idx[is_e-1].date()}"
            oos_dates = f"{idx[oos_s].date()} to {idx[min(oos_e-1, n-1)].date()}"

            fold_results.append({
                "fold": fi + 1, "oos_trades": oos_trades,
                "oos_pnl": round(oos_pnl, 2), "oos_pf": round(oos_pf, 3),
                "oos_wr": round(oos_wr, 1), "is_go": is_go,
            })

            print(f"    Fold {fi+1}: IS {is_dates} | OOS {oos_dates}")
            print(f"      {oos_trades} trades, PnL ${oos_pnl:,.2f}, "
                  f"PF {oos_pf:.3f}, WR {oos_wr:.1f}%  [{status}]")

        n_go = sum(1 for f in fold_results if f["is_go"])
        print(f"    --> {n_go}/{len(fold_results)} folds GO")
        all_results[name] = fold_results

    return all_results


# ======================================================================
# Section 5: Propfirm
# ======================================================================

def run_propfirm_analysis(candidates, aligned, px_a, px_b, idx, minutes):
    print("\n" + "=" * 100)
    print(" SECTION 5: PROPFIRM $150K COMPLIANCE")
    print("=" * 100)

    bar_dates = idx.date
    config = PropfirmConfig()
    print(f"\n  Account: ${config.account_size:,.0f}  "
          f"Max daily loss: ${config.max_daily_loss:,.0f}  "
          f"Trailing DD: ${config.trailing_max_dd:,.0f}")

    results = {}
    for name, cfg in candidates.items():
        bt, *_ = reconstruct_backtest(cfg, aligned, px_a, px_b, idx, minutes)

        pr = compute_propfirm_metrics(
            bt["trade_entry_bars"], bt["trade_exit_bars"],
            bt["trade_pnls"], bar_dates, bt["equity"], config,
        )
        results[name] = pr

        status = "COMPLIANT" if pr.is_compliant else "VIOLATION"
        print(f"\n  {cfg_label(name, cfg)}")
        print(f"      Trading days: {pr.n_trading_days}  "
              f"Avg daily PnL: ${pr.avg_daily_pnl:,.2f}  "
              f"Profitable days: {pr.pct_days_profitable:.1f}%")
        print(f"      Max daily loss: ${pr.max_daily_loss_observed:,.2f}  "
              f"Exceed daily limit: {pr.n_days_exceed_daily_limit} days")
        print(f"      Max trailing DD: ${pr.max_trailing_dd:,.2f}  "
              f"Exceed trailing DD: {pr.n_days_exceed_trailing_dd} bars  [{status}]")

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 13c Analysis")
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip gate sensitivity")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "output" / "NQ_YM" / "phase13c_grid_massif.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        return

    print("Loading CSV (15.7M rows)...")
    df_grid = pd.read_csv(csv_path)
    print(f"Loaded {len(df_grid):,} configs")

    # Show candidates
    print("\n" + "=" * 100)
    print(" PHASE 13c ANALYSIS -- 4 CANDIDATES")
    print("=" * 100)
    for name, cfg in CANDIDATES.items():
        z_exit, z_stop = derive_params(cfg)
        # Lookup in CSV
        mask = (
            (df_grid["ols"] == cfg["ols"])
            & (df_grid["adf_w"] == cfg["adf_w"])
            & (df_grid["zw"] == cfg["zw"])
            & (df_grid["window"] == cfg["window"])
            & (np.isclose(df_grid["z_entry"], cfg["z_entry"]))
            & (np.isclose(df_grid["delta_tp"], cfg["delta_tp"]))
            & (np.isclose(df_grid["delta_sl"], cfg["delta_sl"]))
            & (df_grid["time_stop"] == cfg["time_stop"])
        )
        matches = df_grid[mask]
        if len(matches) > 0:
            r = matches.iloc[0]
            print(f"\n  {name}: OLS={cfg['ols']} ADF_w={cfg['adf_w']} ZW={cfg['zw']} "
                  f"{cfg['window']} ze={cfg['z_entry']} zx={z_exit} zs={z_stop} ts={cfg['time_stop']}")
            print(f"      Trades={int(r['trades'])} PnL=${r['pnl']:,.0f} PF={r['pf']:.3f} "
                  f"DD=${r['max_dd']:,.0f} CPCV_med={r['cpcv_median_sharpe']:.4f} "
                  f"Paths+={r['cpcv_pct_positive']:.1f}%")
        else:
            print(f"\n  {name}: NOT FOUND IN CSV!")

    # Load market data
    print("\nLoading market data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    all_sharpes = df_grid["cpcv_median_sharpe"].values

    # Section 1: DSR
    dsr_results = run_dsr_analysis(CANDIDATES, all_sharpes, aligned, px_a, px_b, idx, minutes)

    # Section 2: Neighborhood
    print("\nRunning neighborhood analysis (CSV lookups)...")
    nb_results = run_neighborhood_analysis(CANDIDATES, df_grid)

    # Free CSV memory before compute-heavy sections
    del df_grid

    # Section 3: Gate Sensitivity
    if not args.skip_sensitivity:
        run_gate_sensitivity(CANDIDATES, aligned, px_a, px_b, idx, minutes)
    else:
        print("\n  [Skipping gate sensitivity]")

    # Section 4: Walk-Forward
    wf_results = {}
    if not args.skip_wf:
        wf_results = run_walkforward(CANDIDATES, aligned, px_a, px_b, idx, minutes)
    else:
        print("\n  [Skipping walk-forward]")

    # Section 5: Propfirm
    pf_results = run_propfirm_analysis(CANDIDATES, aligned, px_a, px_b, idx, minutes)

    # ======================================================================
    # Final Summary Table
    # ======================================================================
    print("\n" + "=" * 100)
    print(" FINAL SUMMARY")
    print("=" * 100)

    header = f"{'Candidate':>10} | {'DSR':>6} | {'Neighborhood':>14} | {'Hurst/Corr':>12} | {'WF':>10} | {'Propfirm':>10}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for name in CANDIDATES:
        dsr_val = dsr_results[name]["dsr"]
        dsr_str = f"{dsr_val:.2f}" + (" PASS" if dsr_val > 0.95 else " FAIL")

        nb = nb_results[name]
        nb_str = "ROBUST" if nb.is_robust else "FRAGILE"

        # Sensitivity: summarize as stable/unstable based on the output
        sens_str = "see above"

        if name in wf_results:
            n_go = sum(1 for f in wf_results[name] if f["is_go"])
            n_total = len(wf_results[name])
            wf_str = f"{n_go}/{n_total} GO"
        else:
            wf_str = "skipped"

        pf = pf_results[name]
        pf_str = "COMPLIANT" if pf.is_compliant else "VIOLATION"

        print(f"  {name:>10} | {dsr_str:>6} | {nb_str:>14} | {sens_str:>12} | {wf_str:>10} | {pf_str:>10}")

    print()
    for name, cfg in CANDIDATES.items():
        z_exit, z_stop = derive_params(cfg)
        nb = nb_results[name]
        pf = pf_results[name]
        dsr_val = dsr_results[name]["dsr"]

        passes = []
        if dsr_val > 0.95:
            passes.append("DSR")
        if nb.is_robust:
            passes.append("ROBUST")
        if pf.is_compliant:
            passes.append("PROPFIRM")
        if name in wf_results:
            n_go = sum(1 for f in wf_results[name] if f["is_go"])
            n_total = len(wf_results[name])
            if n_go >= n_total * 0.5:
                passes.append("WF")

        verdict = " + ".join(passes) if passes else "---"
        print(f"  {name}: {verdict}")

    print()


if __name__ == "__main__":
    main()
