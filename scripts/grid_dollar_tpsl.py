"""Dollar TP/SL Grid -- 1-second precision + CPCV validation.

Config D base: OLS=7000, ADF_w=96, ZW=30, ze=3.25/zx=0.50/zs=4.75, 02:00-14:00.
Adds dollar-based TP and SL exits, checked at 1-second resolution.

Grid:
  TP dollar: [0, 200, 300, 400, 500, 600, 700]   (0 = no dollar TP)
  SL dollar: [0, 500, 750, 1000, 1250, 1500, 1750, 2000]  (0 = no dollar SL)
  Total: 56 configs  (0/0 = baseline = original Config D)

Architecture:
  1. Entries via 5min z-score signals (unchanged)
  2. For each trade, walk 1-second PnL curve (raw unrealized)
  3. If raw_unrealized >= TP or <= -SL BEFORE original z-exit -> dollar exit
  4. Otherwise -> original z-exit (z_exit / z_stop / flat)
  5. Dollar exit PnL = raw_unrealized at trigger - exit_slippage - commissions
  6. CPCV(10,2) on modified PnLs
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized

# --- Constants ---
from src.config.instruments import DEFAULT_SLIPPAGE_TICKS, get_pair_specs
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.signals.filters import apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.validation.cpcv import CPCVConfig, run_cpcv
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask

_NQ, _YM = get_pair_specs("NQ", "YM")
MULT_A, MULT_B = _NQ.multiplier, _YM.multiplier
TICK_A, TICK_B = _NQ.tick_size, _YM.tick_size
SLIPPAGE = DEFAULT_SLIPPAGE_TICKS
COMMISSION = _NQ.commission

TP_VALUES = [0, 200, 300, 400, 500, 600, 700]
SL_VALUES = [0, 500, 750, 1000, 1250, 1500, 1750, 2000]


# =========================================================================
# Step 1: Reconstruct Config D
# =========================================================================

def reconstruct_config_d():
    """Reconstruct Config D backtest on 5min bars.

    Returns trades list, zscore array, minutes array, entry/exit bars, n_bars.
    """
    print("  [1/4] Reconstructing Config D on 5min...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n_bars = len(px_a)
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    est = create_estimator("ols_rolling", window=7000, zscore_window=30)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    gate_cfg = GateConfig(
        adf_threshold=-2.86, hurst_threshold=0.50, corr_threshold=0.70,
        adf_window=96, hurst_window=64, corr_window=24,
    )
    gate_mask = compute_gate_mask(
        spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg
    )

    mu = spread.rolling(30).mean()
    sigma = spread.rolling(30).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw = generate_signals_numba(zscore, 3.25, 0.50, 4.75)
    sig_gated = apply_gate_filter_numba(raw, gate_mask)
    sig_final = apply_window_filter_numba(sig_gated, minutes, 120, 840, 930)

    bt = run_backtest_vectorized(
        px_a, px_b, sig_final, beta, MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, 100_000.0,
    )

    entries = bt["trade_entry_bars"]
    exits = bt["trade_exit_bars"]
    sides = bt["trade_sides"]
    pnls = bt["trade_pnls"]
    num = bt["trades"]

    print(f"    {num} trades, PF {bt['profit_factor']:.2f}, PnL ${pnls.sum():+,.0f}")

    # Build trade list
    trades = []
    for i in range(num):
        eb, xb = entries[i], exits[i]
        side = sides[i]
        b = beta[eb]
        # Match vectorized backtest sizing: n_b = round((not_a / not_b) * |beta|)
        not_a = px_a[eb] * MULT_A
        not_b = px_b[eb] * MULT_B
        n_b = max(1, round(not_a / not_b * abs(b)))

        # Entry prices with slippage (same as vectorized backtest)
        if side == 1:  # long spread: buy NQ, sell YM
            adj_nq = px_a[eb] + SLIPPAGE * TICK_A
            adj_ym = px_b[eb] - SLIPPAGE * TICK_B
        else:  # short spread: sell NQ, buy YM
            adj_nq = px_a[eb] - SLIPPAGE * TICK_A
            adj_ym = px_b[eb] + SLIPPAGE * TICK_B

        trades.append({
            "idx": i,
            "side": side,
            "entry_bar": eb,
            "exit_bar": xb,
            "entry_time": idx[eb],
            "exit_time": idx[xb],
            "entry_nq": adj_nq,
            "entry_ym": adj_ym,
            "n_a": 1,
            "n_b": n_b,
            "pnl_5min": pnls[i],
        })

    return trades, zscore, minutes, entries, exits, n_bars


# =========================================================================
# Step 2: Load 1-second data
# =========================================================================

def load_1s_for_trades(trades):
    """Load 1s data only for trade intervals. Returns NQ and YM DataFrames."""
    print("  [2/4] Loading 1-second data for trade intervals...")

    # Build date set (Sierra CSV uses non-zero-padded dates: 2020/12/2)
    trade_dates = set()
    for t in trades:
        for d in [t["entry_time"].date(), t["exit_time"].date()]:
            trade_dates.add(f"{d.year}/{d.month}/{d.day}")

    print(f"    {len(trade_dates)} unique trade dates")

    # Time intervals: entry/exit happen at bar CLOSE = bar_timestamp + 5min
    # Add buffer for loading
    intervals = []
    for t in trades:
        start = t["entry_time"] + pd.Timedelta(minutes=4)  # near actual entry
        end = t["exit_time"] + pd.Timedelta(minutes=6)     # past actual exit
        intervals.append((start, end))

    def load_1s_file(fname, label):
        print(f"    Scanning {label}...", end="", flush=True)
        t0 = time_mod.time()
        parts = []
        rows_kept = 0
        rows_total = 0
        for chunk in pd.read_csv(fname, chunksize=2_000_000, skipinitialspace=True):
            chunk.columns = chunk.columns.str.strip()
            rows_total += len(chunk)
            dates = chunk["Date"].str.strip()
            date_mask = dates.isin(trade_dates)
            if not date_mask.any():
                continue
            sub = chunk[date_mask].copy()
            sub["dt"] = pd.to_datetime(
                sub["Date"].str.strip() + " " + sub["Time"].str.strip(),
                format="%Y/%m/%d %H:%M:%S",
            )
            keep = np.zeros(len(sub), dtype=bool)
            for start, end in intervals:
                keep |= (sub["dt"].values >= np.datetime64(start)) & (
                    sub["dt"].values <= np.datetime64(end)
                )
            filtered = sub[keep]
            if len(filtered) > 0:
                parts.append(filtered[["dt", "Last"]].copy())
                rows_kept += len(filtered)

        elapsed = time_mod.time() - t0
        print(f" {rows_total/1e6:.1f}M rows, {rows_kept:,} kept ({elapsed:.0f}s)")

        if not parts:
            return pd.DataFrame(columns=["dt", "Last"])
        df = pd.concat(parts).sort_values("dt").reset_index(drop=True)
        return df

    nq_1s = load_1s_file(
        str(PROJECT_ROOT / "raw" / "NQH26_FUT_CME_1s.scid_BarData.txt"), "NQ 1s"
    )
    ym_1s = load_1s_file(
        str(PROJECT_ROOT / "raw" / "YMH26_FUT_CME_1s.scid_BarData.txt"), "YM 1s"
    )

    return nq_1s, ym_1s


# =========================================================================
# Step 3: Pre-compute 1s PnL curves
# =========================================================================

def precompute_1s_curves(trades, nq_1s, ym_1s):
    """Pre-compute raw unrealized PnL curves at 1s for each trade.

    Returns list of (unrealized_curve, exit_cost) per trade.
    - unrealized_curve: raw unrealized PnL (entry slippage applied, no exit costs)
    - exit_cost: exit slippage + total commissions (constant per trade)
    """
    print("  [3/4] Pre-computing 1s PnL curves...")

    nq_indexed = nq_1s.set_index("dt").sort_index()
    ym_indexed = ym_1s.set_index("dt").sort_index()

    curves = []
    no_data = 0

    for t in trades:
        # Actual entry/exit at bar CLOSE = bar_timestamp + 5min
        start = t["entry_time"] + pd.Timedelta(minutes=5)
        end = t["exit_time"] + pd.Timedelta(minutes=5)
        side = t["side"]
        entry_nq = t["entry_nq"]
        entry_ym = t["entry_ym"]
        n_a = t["n_a"]
        n_b = t["n_b"]

        nq_slice = nq_indexed.loc[start:end]
        ym_slice = ym_indexed.loc[start:end]

        if len(nq_slice) < 2 or len(ym_slice) < 2:
            no_data += 1
            exit_cost = (
                SLIPPAGE * (TICK_A * MULT_A * n_a + TICK_B * MULT_B * n_b)
                + COMMISSION * (n_a + n_b) * 2
            )
            curves.append((np.array([]), exit_cost))
            continue

        # Align on common timestamps (forward fill gaps)
        combined = pd.DataFrame(index=nq_slice.index.union(ym_slice.index))
        combined["nq"] = nq_slice["Last"]
        combined["ym"] = ym_slice["Last"]
        combined = combined.ffill().dropna()

        if len(combined) < 2:
            no_data += 1
            exit_cost = (
                SLIPPAGE * (TICK_A * MULT_A * n_a + TICK_B * MULT_B * n_b)
                + COMMISSION * (n_a + n_b) * 2
            )
            curves.append((np.array([]), exit_cost))
            continue

        nq_prices = combined["nq"].values
        ym_prices = combined["ym"].values

        # Raw unrealized PnL (entry slippage in entry prices, no exit costs)
        if side == 1:  # long spread: bought NQ, sold YM
            unrealized = (
                (nq_prices - entry_nq) * MULT_A * n_a
                + (entry_ym - ym_prices) * MULT_B * n_b
            )
        else:  # short spread: sold NQ, bought YM
            unrealized = (
                (entry_nq - nq_prices) * MULT_A * n_a
                + (ym_prices - entry_ym) * MULT_B * n_b
            )

        # Exit cost = exit slippage + all commissions (entry + exit)
        exit_slip = SLIPPAGE * (TICK_A * MULT_A * n_a + TICK_B * MULT_B * n_b)
        total_comm = COMMISSION * (n_a + n_b) * 2
        exit_cost = exit_slip + total_comm

        curves.append((unrealized, exit_cost))

    if no_data > 0:
        print(f"    WARNING: {no_data} trades with insufficient 1s data")
    print(f"    {len(trades) - no_data}/{len(trades)} trades with 1s curves")

    # Sanity check: unrealized at end of curve - exit_cost should approximate pnl_5min
    errors = []
    for i, t in enumerate(trades):
        curve, exit_cost = curves[i]
        if len(curve) == 0:
            continue
        implied_pnl = curve[-1] - exit_cost
        actual_pnl = t["pnl_5min"]
        err = abs(implied_pnl - actual_pnl)
        errors.append(err)
    if errors:
        med_err = np.median(errors)
        max_err = np.max(errors)
        print(f"    Sanity check: median|implied-actual| = ${med_err:.0f}, "
              f"max = ${max_err:.0f}")

    return curves


# =========================================================================
# Step 3b: Classify original exit types
# =========================================================================

def classify_original_exits(trades, zscore, minutes):
    """Classify each trade's original exit type: Z_EXIT, Z_STOP, or FLAT."""
    exit_types = []
    for t in trades:
        xb = t["exit_bar"]
        min_at_exit = minutes[xb]
        z_at_exit = abs(zscore[xb]) if np.isfinite(zscore[xb]) else 0.0

        if min_at_exit >= 930:
            exit_types.append("FLAT")
        elif z_at_exit >= 4.25:  # near z_stop=4.75 (tolerance for bar discretization)
            exit_types.append("Z_STOP")
        else:
            exit_types.append("Z_EXIT")

    counts = {}
    for et in exit_types:
        counts[et] = counts.get(et, 0) + 1
    print(f"    Original exits: {counts}")

    return exit_types


# =========================================================================
# Step 4: Apply dollar TP/SL and compute metrics
# =========================================================================

def apply_dollar_tpsl(trades, curves, dollar_tp, dollar_sl, original_exits):
    """Apply dollar TP/SL to pre-computed 1s curves.

    Returns (modified_pnls, exit_types).
    """
    n = len(trades)
    pnls = np.zeros(n)
    exit_types = []

    for i in range(n):
        curve, exit_cost = curves[i]

        if len(curve) == 0:
            # No 1s data -> use original
            pnls[i] = trades[i]["pnl_5min"]
            exit_types.append(original_exits[i])
            continue

        # Find first TP and SL trigger indices
        if dollar_tp > 0:
            tp_indices = np.where(curve >= dollar_tp)[0]
            first_tp = tp_indices[0] if len(tp_indices) > 0 else len(curve) + 1
        else:
            first_tp = len(curve) + 1  # never triggers

        if dollar_sl > 0:
            sl_indices = np.where(curve <= -dollar_sl)[0]
            first_sl = sl_indices[0] if len(sl_indices) > 0 else len(curve) + 1
        else:
            first_sl = len(curve) + 1  # never triggers

        if first_tp <= first_sl and first_tp < len(curve) + 1:
            # Dollar TP triggered first
            pnls[i] = curve[first_tp] - exit_cost
            exit_types.append("DOLLAR_TP")
        elif first_sl < first_tp and first_sl < len(curve) + 1:
            # Dollar SL triggered first
            pnls[i] = curve[first_sl] - exit_cost
            exit_types.append("DOLLAR_SL")
        else:
            # Neither triggered -> original exit
            pnls[i] = trades[i]["pnl_5min"]
            exit_types.append(original_exits[i])

    return pnls, exit_types


def compute_max_dd(pnls):
    """Max drawdown from sequential trade PnLs."""
    if len(pnls) == 0:
        return 0.0
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    return float(dd.min())


def compute_max_consec_losses(pnls):
    """Max consecutive losing trades."""
    max_c = 0
    current = 0
    for p in pnls:
        if p <= 0:
            current += 1
            max_c = max(max_c, current)
        else:
            current = 0
    return max_c


def compute_metrics(pnls, exit_types, entry_bars, exit_bars, n_bars):
    """Compute full metrics + CPCV for a set of modified trade PnLs."""
    n = len(pnls)
    if n == 0:
        return {}

    total_pnl = float(pnls.sum())
    wins = pnls > 0
    losses = pnls <= 0
    n_wins = int(wins.sum())
    wr = n_wins / n * 100 if n > 0 else 0.0

    gross_gains = float(pnls[wins].sum()) if wins.any() else 0.0
    gross_losses = float(abs(pnls[losses].sum())) if losses.any() else 0.0
    pf = gross_gains / gross_losses if gross_losses > 0 else (
        float("inf") if gross_gains > 0 else 0.0
    )

    avg_win = float(pnls[wins].mean()) if wins.any() else 0.0
    avg_loss = float(pnls[losses].mean()) if losses.any() else 0.0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    worst = float(pnls.min())
    max_dd = compute_max_dd(pnls)
    max_consec = compute_max_consec_losses(pnls)

    # CPCV
    cpcv_cfg = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)
    cpcv = run_cpcv(entry_bars, exit_bars, pnls, n_bars, cpcv_cfg)

    # Exit type breakdown
    type_counts = {}
    type_pnls = {}
    for et, p in zip(exit_types, pnls):
        type_counts[et] = type_counts.get(et, 0) + 1
        if et not in type_pnls:
            type_pnls[et] = []
        type_pnls[et].append(p)

    exit_breakdown = {}
    for et in ["Z_EXIT", "Z_STOP", "FLAT", "DOLLAR_TP", "DOLLAR_SL"]:
        cnt = type_counts.get(et, 0)
        pct = cnt / n * 100 if n > 0 else 0.0
        avg_p = float(np.mean(type_pnls[et])) if et in type_pnls else 0.0
        exit_breakdown[et] = {"count": cnt, "pct": pct, "avg_pnl": avg_p}

    return {
        "trades": n,
        "pnl": total_pnl,
        "pf": pf,
        "wr": wr,
        "max_dd": max_dd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "wl_ratio": wl_ratio,
        "worst_trade": worst,
        "max_consec": max_consec,
        "cpcv_median": cpcv["median_sharpe"],
        "cpcv_paths_pct": cpcv["pct_positive"],
        "exit_breakdown": exit_breakdown,
    }


# =========================================================================
# Step 5: Print results
# =========================================================================

def print_results(results):
    """Print formatted results table + exit type details."""
    print(f"\n{'='*120}")
    print("  DOLLAR TP/SL GRID -- Config D + 1s precision + CPCV(10,2)")
    print(f"{'='*120}\n")

    # Header
    hdr = (
        f"{'TP':>5} {'SL':>5} | {'Trd':>4} {'PnL':>9} {'PF':>5} {'WR':>5} "
        f"{'MaxDD':>8} {'AvgW':>7} {'AvgL':>7} {'W/L':>5} {'Worst':>7} "
        f"{'MCL':>3} | {'CPCV':>6} {'P+%':>5} | "
        f"{'zExit':>5} {'zStop':>5} {'Flat':>4} {'$TP':>4} {'$SL':>4}"
    )
    print(hdr)
    print("-" * len(hdr))

    # Sort by PF descending
    sorted_results = sorted(results, key=lambda x: x.get("pf", 0), reverse=True)

    for r in sorted_results:
        tp = r["tp"]
        sl = r["sl"]
        eb = r.get("exit_breakdown", {})

        # Exit percentages
        ze_pct = eb.get("Z_EXIT", {}).get("pct", 0)
        zs_pct = eb.get("Z_STOP", {}).get("pct", 0)
        fl_pct = eb.get("FLAT", {}).get("pct", 0)
        dtp_pct = eb.get("DOLLAR_TP", {}).get("pct", 0)
        dsl_pct = eb.get("DOLLAR_SL", {}).get("pct", 0)

        # Baseline marker
        marker = " <-- BASELINE" if tp == 0 and sl == 0 else ""

        pf_str = f"{r['pf']:5.2f}" if r["pf"] < 100 else " inf"

        print(
            f"{tp:>5} {sl:>5} | {r['trades']:>4} ${r['pnl']:>+8,.0f} {pf_str} "
            f"{r['wr']:>4.1f}% ${r['max_dd']:>+7,.0f} ${r['avg_win']:>+6,.0f} "
            f"${r['avg_loss']:>+6,.0f} {r['wl_ratio']:>5.2f} ${r['worst_trade']:>+6,.0f} "
            f"{r['max_consec']:>3} | {r['cpcv_median']:>6.3f} {r['cpcv_paths_pct']:>4.1f}% | "
            f"{ze_pct:>4.0f}% {zs_pct:>4.0f}% {fl_pct:>3.0f}% {dtp_pct:>3.0f}% "
            f"{dsl_pct:>3.0f}%{marker}"
        )

    # --- Detailed exit PnL breakdown for top configs ---
    print(f"\n{'='*90}")
    print("  EXIT TYPE PnL BREAKDOWN -- Top 10 configs by PF")
    print(f"{'='*90}\n")

    for r in sorted_results[:10]:
        tp, sl = r["tp"], r["sl"]
        eb = r.get("exit_breakdown", {})
        label = f"TP=${tp} SL=${sl}" if tp > 0 or sl > 0 else "BASELINE"
        print(f"  {label} (PF {r['pf']:.2f}, {r['trades']} trades, ${r['pnl']:+,.0f}):")
        for et in ["Z_EXIT", "Z_STOP", "FLAT", "DOLLAR_TP", "DOLLAR_SL"]:
            info = eb.get(et, {})
            cnt = info.get("count", 0)
            if cnt == 0:
                continue
            pct = info.get("pct", 0)
            avg_p = info.get("avg_pnl", 0)
            print(f"    {et:>10}: {cnt:>3} ({pct:>4.1f}%), avg PnL ${avg_p:>+,.0f}")
        print()

    # --- Matrix views ---
    print(f"\n{'='*70}")
    print("  PF MATRIX  (rows=SL, cols=TP)")
    print(f"{'='*70}\n")

    # Build lookup
    lookup = {}
    for r in results:
        lookup[(r["tp"], r["sl"])] = r

    # Header row
    sl_tp = "SL/TP"
    tp_header = f"{sl_tp:>8}"
    for tp in TP_VALUES:
        tp_header += f"  {tp if tp > 0 else 'off':>6}"
    print(tp_header)
    print("-" * len(tp_header))

    for sl in SL_VALUES:
        sl_label = "off" if sl == 0 else str(sl)
        row = f"{sl_label:>8}"
        for tp in TP_VALUES:
            r = lookup.get((tp, sl))
            if r:
                pf = r["pf"]
                row += f"  {pf:>6.2f}" if pf < 100 else "    inf"
            else:
                row += "     --"
        print(row)

    # PnL matrix
    print(f"\n{'='*70}")
    print("  PnL MATRIX ($)  (rows=SL, cols=TP)")
    print(f"{'='*70}\n")

    tp_header = f"{sl_tp:>8}"
    for tp in TP_VALUES:
        tp_header += f"  {tp if tp > 0 else 'off':>7}"
    print(tp_header)
    print("-" * len(tp_header))

    for sl in SL_VALUES:
        sl_label = "off" if sl == 0 else str(sl)
        row = f"{sl_label:>8}"
        for tp in TP_VALUES:
            r = lookup.get((tp, sl))
            if r:
                row += f"  {r['pnl']:>+7,.0f}"
            else:
                row += "       --"
        print(row)

    # CPCV paths+ matrix
    print(f"\n{'='*70}")
    print("  CPCV PATHS+ (%)  (rows=SL, cols=TP)")
    print(f"{'='*70}\n")

    tp_header = f"{sl_tp:>8}"
    for tp in TP_VALUES:
        tp_header += f"  {tp if tp > 0 else 'off':>6}"
    print(tp_header)
    print("-" * len(tp_header))

    for sl in SL_VALUES:
        sl_label = "off" if sl == 0 else str(sl)
        row = f"{sl_label:>8}"
        for tp in TP_VALUES:
            r = lookup.get((tp, sl))
            if r:
                row += f"  {r['cpcv_paths_pct']:>5.1f}%"
            else:
                row += "     --%"
        print(row)


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time_mod.time()
    print("Dollar TP/SL Grid -- Config D + 1s precision\n")
    print(f"  Grid: TP = {TP_VALUES}")
    print(f"         SL = {SL_VALUES}")
    print(f"  Total configs: {len(TP_VALUES) * len(SL_VALUES)}\n")

    # 1. Reconstruct Config D
    trades, zscore, minutes, entry_bars, exit_bars, n_bars = reconstruct_config_d()

    # 2. Load 1s data
    nq_1s, ym_1s = load_1s_for_trades(trades)

    # 3. Pre-compute 1s PnL curves (once)
    curves = precompute_1s_curves(trades, nq_1s, ym_1s)

    # 3b. Classify original exit types
    original_exits = classify_original_exits(trades, zscore, minutes)

    # 4. Run grid
    print(f"\n  [4/4] Running {len(TP_VALUES)*len(SL_VALUES)} configs...")
    results = []
    for tp in TP_VALUES:
        for sl in SL_VALUES:
            pnls, exit_types = apply_dollar_tpsl(
                trades, curves, tp, sl, original_exits
            )
            metrics = compute_metrics(
                pnls, exit_types, entry_bars, exit_bars, n_bars
            )
            metrics["tp"] = tp
            metrics["sl"] = sl
            results.append(metrics)

    # 5. Print results
    print_results(results)

    # 6. Save to CSV
    outdir = PROJECT_ROOT / "output" / "NQ_YM"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        eb = r.get("exit_breakdown", {})
        row = {
            "tp": r["tp"], "sl": r["sl"],
            "trades": r["trades"], "pnl": r["pnl"], "pf": r["pf"],
            "wr": r["wr"], "max_dd": r["max_dd"],
            "avg_win": r["avg_win"], "avg_loss": r["avg_loss"],
            "wl_ratio": r["wl_ratio"], "worst_trade": r["worst_trade"],
            "max_consec": r["max_consec"],
            "cpcv_median": r["cpcv_median"], "cpcv_paths_pct": r["cpcv_paths_pct"],
        }
        for et in ["Z_EXIT", "Z_STOP", "FLAT", "DOLLAR_TP", "DOLLAR_SL"]:
            info = eb.get(et, {})
            row[f"{et}_pct"] = info.get("pct", 0)
            row[f"{et}_avg_pnl"] = info.get("avg_pnl", 0)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    csv_path = outdir / "grid_dollar_tpsl.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")

    elapsed = time_mod.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
