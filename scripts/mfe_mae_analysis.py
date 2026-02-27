"""MFE/MAE Analysis on Config D trades using 1-second data.

For each of the 153 trades, computes the dollar PnL at 1-second resolution
to extract Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE).
"""

import sys
import time as time_mod
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.config.instruments import DEFAULT_SLIPPAGE_TICKS, get_pair_specs
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.signals.filters import apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask

_NQ, _YM = get_pair_specs("NQ", "YM")
MULT_A, MULT_B = _NQ.multiplier, _YM.multiplier
TICK_A, TICK_B = _NQ.tick_size, _YM.tick_size
SLIPPAGE = DEFAULT_SLIPPAGE_TICKS
COMMISSION = _NQ.commission


def reconstruct_config_d():
    """Reconstruct Config D backtest, return trades + 5min data."""
    print("  Reconstructing Config D on 5min...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
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

    print(f"  {num} trades, PF {bt['profit_factor']:.2f}, PnL ${pnls.sum():+,.0f}")

    # Build trade list with timestamps and prices
    trades = []
    for i in range(num):
        eb, xb = entries[i], exits[i]
        side = sides[i]
        entry_time = idx[eb]
        exit_time = idx[xb]
        entry_nq = px_a[eb]
        entry_ym = px_b[eb]
        b = beta[eb]
        # Match vectorized backtest sizing: n_b = round((not_a / not_b) * |beta|)
        not_a = entry_nq * MULT_A
        not_b = entry_ym * MULT_B
        n_b = max(1, round(not_a / not_b * abs(b)))

        # Slippage on entry
        if side == 1:  # long spread: buy NQ, sell YM
            adj_nq = entry_nq + SLIPPAGE * TICK_A
            adj_ym = entry_ym - SLIPPAGE * TICK_B
        else:  # short spread: sell NQ, buy YM
            adj_nq = entry_nq - SLIPPAGE * TICK_A
            adj_ym = entry_ym + SLIPPAGE * TICK_B

        trades.append({
            "idx": i,
            "side": side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_nq": adj_nq,
            "entry_ym": adj_ym,
            "n_a": 1,
            "n_b": n_b,
            "beta": b,
            "pnl_5min": pnls[i],
        })

    return trades


def load_1s_for_trades(trades):
    """Load 1s data only for trade intervals. Returns aligned NQ+YM at 1s."""
    print("  Loading 1s data for trade intervals...")

    # Build date set for filtering (Sierra CSV uses non-zero-padded dates: 2020/12/2)
    trade_dates = set()
    for t in trades:
        d = t["entry_time"].date()
        trade_dates.add(f"{d.year}/{d.month}/{d.day}")
        # Exit might be next day
        d2 = t["exit_time"].date()
        trade_dates.add(f"{d2.year}/{d2.month}/{d2.day}")

    print(f"  {len(trade_dates)} unique trade dates")

    # Build time intervals (with 1min buffer)
    intervals = []
    for t in trades:
        start = t["entry_time"] - pd.Timedelta(minutes=1)
        end = t["exit_time"] + pd.Timedelta(minutes=1)
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
            # Quick date filter first
            dates = chunk["Date"].str.strip()
            date_mask = dates.isin(trade_dates)
            if not date_mask.any():
                continue
            sub = chunk[date_mask].copy()
            # Parse datetime
            sub["dt"] = pd.to_datetime(
                sub["Date"].str.strip() + " " + sub["Time"].str.strip(),
                format="%Y/%m/%d %H:%M:%S",
            )
            # Fine filter: keep only rows within trade intervals
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
        print(f" {rows_total/1e6:.1f}M rows scanned, {rows_kept:,} kept ({elapsed:.0f}s)")

        if not parts:
            return pd.DataFrame(columns=["dt", "Last"])
        df = pd.concat(parts).sort_values("dt").reset_index(drop=True)
        return df

    nq_1s = load_1s_file(
        str(PROJECT_ROOT / "raw" / "NQH26_FUT_CME_1s.scid_BarData.txt"), "NQ"
    )
    ym_1s = load_1s_file(
        str(PROJECT_ROOT / "raw" / "YMH26_FUT_CME_1s.scid_BarData.txt"), "YM"
    )

    return nq_1s, ym_1s


def compute_mfe_mae(trades, nq_1s, ym_1s):
    """Compute MFE/MAE for each trade at 1s resolution."""
    print("  Computing MFE/MAE per trade...")

    # Index by datetime for fast lookup
    nq_1s = nq_1s.set_index("dt").sort_index()
    ym_1s = ym_1s.set_index("dt").sort_index()

    results = []
    no_data_count = 0

    for t in trades:
        start = t["entry_time"]
        end = t["exit_time"]
        side = t["side"]
        entry_nq = t["entry_nq"]
        entry_ym = t["entry_ym"]
        n_a = t["n_a"]
        n_b = t["n_b"]

        # Get 1s slices
        nq_slice = nq_1s.loc[start:end]
        ym_slice = ym_1s.loc[start:end]

        if len(nq_slice) < 2 or len(ym_slice) < 2:
            no_data_count += 1
            results.append({
                **t, "mfe": np.nan, "mae": np.nan, "mfe_time": np.nan,
                "mae_time": np.nan, "n_points": 0,
            })
            continue

        # Align on common timestamps (forward fill for gaps)
        combined = pd.DataFrame(index=nq_slice.index.union(ym_slice.index))
        combined["nq"] = nq_slice["Last"]
        combined["ym"] = ym_slice["Last"]
        combined = combined.ffill().dropna()

        if len(combined) < 2:
            no_data_count += 1
            results.append({
                **t, "mfe": np.nan, "mae": np.nan, "mfe_time": np.nan,
                "mae_time": np.nan, "n_points": 0,
            })
            continue

        # Compute PnL at each 1s point (no slippage on exit for MFE/MAE)
        nq_prices = combined["nq"].values
        ym_prices = combined["ym"].values

        if side == 1:  # long spread: bought NQ, sold YM
            pnl_curve = (
                (nq_prices - entry_nq) * MULT_A * n_a
                + (entry_ym - ym_prices) * MULT_B * n_b
            )
        else:  # short spread: sold NQ, bought YM
            pnl_curve = (
                (entry_nq - nq_prices) * MULT_A * n_a
                + (ym_prices - entry_ym) * MULT_B * n_b
            )

        # Subtract entry commission (already paid)
        entry_cost = COMMISSION * (n_a + n_b)
        pnl_curve = pnl_curve - entry_cost

        mfe = pnl_curve.max()
        mae = pnl_curve.min()
        mfe_idx = np.argmax(pnl_curve)
        mae_idx = np.argmin(pnl_curve)
        mfe_time_sec = (combined.index[mfe_idx] - start).total_seconds()
        mae_time_sec = (combined.index[mae_idx] - start).total_seconds()

        results.append({
            **t,
            "mfe": mfe,
            "mae": mae,
            "mfe_time_sec": mfe_time_sec,
            "mae_time_sec": mae_time_sec,
            "n_points": len(combined),
        })

    if no_data_count > 0:
        print(f"  WARNING: {no_data_count} trades with insufficient 1s data")
    print(f"  {len(results) - no_data_count}/{len(results)} trades analyzed at 1s")

    return results


def print_analysis(results):
    """Print MFE/MAE summary statistics."""
    df = pd.DataFrame(results)
    valid = df[df["mfe"].notna()].copy()
    n = len(valid)

    print(f"\n{'='*70}")
    print(f"  MFE/MAE ANALYSIS -- Config D ({n} trades at 1s resolution)")
    print(f"{'='*70}\n")

    mfe = valid["mfe"].values
    mae = valid["mae"].values
    pnl = valid["pnl_5min"].values
    sides = valid["side"].values

    # Basic stats
    print(f"{'':>25} {'MFE':>10} {'MAE':>10} {'PnL Final':>10}")
    print(f"  {'Median':>23} ${np.median(mfe):>+9,.0f} ${np.median(mae):>+9,.0f} ${np.median(pnl):>+9,.0f}")
    print(f"  {'Mean':>23} ${np.mean(mfe):>+9,.0f} ${np.mean(mae):>+9,.0f} ${np.mean(pnl):>+9,.0f}")
    print(f"  {'P10':>23} ${np.percentile(mfe, 10):>+9,.0f} ${np.percentile(mae, 10):>+9,.0f}")
    print(f"  {'P25':>23} ${np.percentile(mfe, 25):>+9,.0f} ${np.percentile(mae, 25):>+9,.0f}")
    print(f"  {'P75':>23} ${np.percentile(mfe, 75):>+9,.0f} ${np.percentile(mae, 75):>+9,.0f}")
    print(f"  {'P90':>23} ${np.percentile(mfe, 90):>+9,.0f} ${np.percentile(mae, 90):>+9,.0f}")
    print(f"  {'Max/Min':>23} ${mfe.max():>+9,.0f} ${mae.min():>+9,.0f}")

    # How much profit left on table
    left_on_table = mfe - pnl
    print("\n  Profit left on table (MFE - PnL final):")
    print(f"    Median: ${np.median(left_on_table):,.0f}")
    print(f"    Mean:   ${np.mean(left_on_table):,.0f}")
    print(f"    >$200:  {(left_on_table > 200).sum()}/{n} trades ({(left_on_table > 200).sum()/n*100:.0f}%)")
    print(f"    >$500:  {(left_on_table > 500).sum()}/{n} trades ({(left_on_table > 500).sum()/n*100:.0f}%)")

    # Trades that were winning at MFE but lost
    winning_mfe_losing_trade = ((mfe > 0) & (pnl <= 0)).sum()
    print(f"\n  Trades winning at peak but ended losing: {winning_mfe_losing_trade}/{n} ({winning_mfe_losing_trade/n*100:.0f}%)")

    # MAE before recovery
    winners = valid[valid["pnl_5min"] > 0]
    losers = valid[valid["pnl_5min"] <= 0]
    print("\n  MAE by outcome:")
    print(f"    Winners ({len(winners)}): median MAE ${winners['mae'].median():+,.0f}, mean ${winners['mae'].mean():+,.0f}")
    print(f"    Losers  ({len(losers)}):  median MAE ${losers['mae'].median():+,.0f}, mean ${losers['mae'].mean():+,.0f}")

    # MFE timing (when does the peak happen?)
    mfe_times = valid["mfe_time_sec"].values / 60  # convert to minutes
    print("\n  MFE timing (minutes after entry):")
    print(f"    Median: {np.median(mfe_times):.1f} min")
    print(f"    Mean:   {np.mean(mfe_times):.1f} min")
    print(f"    P25:    {np.percentile(mfe_times, 25):.1f} min")
    print(f"    P75:    {np.percentile(mfe_times, 75):.1f} min")

    # TP/SL dollar suggestions
    print("\n  --- TP/SL Dollar Suggestions ---")
    for tp in [150, 200, 250, 300, 400, 500]:
        hit = (mfe >= tp).sum()
        pct = hit / n * 100
        # For trades that hit TP, what's their avg final PnL?
        hit_mask = mfe >= tp
        avg_final = pnl[hit_mask].mean() if hit_mask.sum() > 0 else 0
        print(f"    TP ${tp:>4}: {hit:>3}/{n} trades hit ({pct:.0f}%), avg final PnL ${avg_final:+,.0f}")

    print()
    for sl in [-200, -300, -400, -500, -750, -1000, -1500, -2000]:
        hit = (mae <= sl).sum()
        pct = hit / n * 100
        print(f"    SL ${sl:>6}: {hit:>3}/{n} trades hit ({pct:.0f}%)")

    # Direction breakdown
    print("\n  --- By Direction ---")
    for label, s in [("LONG", 1), ("SHORT", -1)]:
        mask = sides == s
        cnt = mask.sum()
        if cnt == 0:
            continue
        m_mfe = np.median(mfe[mask])
        m_mae = np.median(mae[mask])
        m_pnl = np.median(pnl[mask])
        print(f"    {label:>5} ({cnt:>3}): MFE ${m_mfe:+,.0f}, MAE ${m_mae:+,.0f}, PnL ${m_pnl:+,.0f}")

    return valid


def make_charts(valid):
    """Generate MFE/MAE charts."""
    print("  Generating charts...")
    outdir = PROJECT_ROOT / "output" / "NQ_YM"
    outdir.mkdir(parents=True, exist_ok=True)

    mfe = valid["mfe"].values
    mae = valid["mae"].values
    pnl = valid["pnl_5min"].values
    sides = valid["side"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Config D -- MFE/MAE Analysis (1s resolution)", fontsize=14, fontweight="bold")

    # 1. Scatter MFE vs MAE
    ax = axes[0, 0]
    colors = ["green" if p > 0 else "red" for p in pnl]
    ax.scatter(mae, mfe, c=colors, alpha=0.6, s=30, edgecolors="black", linewidth=0.3)
    ax.set_xlabel("MAE ($)")
    ax.set_ylabel("MFE ($)")
    ax.set_title("MFE vs MAE (green=winner, red=loser)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # 2. MFE vs Final PnL
    ax = axes[0, 1]
    ax.scatter(mfe, pnl, c=colors, alpha=0.6, s=30, edgecolors="black", linewidth=0.3)
    ax.plot([0, mfe.max()], [0, mfe.max()], "k--", alpha=0.3, label="PnL = MFE (no giveback)")
    ax.set_xlabel("MFE ($)")
    ax.set_ylabel("Final PnL ($)")
    ax.set_title("MFE vs Final PnL (distance from diagonal = giveback)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Distribution of MFE
    ax = axes[1, 0]
    ax.hist(mfe, bins=40, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(np.median(mfe), color="red", linestyle="--", label=f"Median ${np.median(mfe):,.0f}")
    ax.set_xlabel("MFE ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution MFE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Distribution of MAE
    ax = axes[1, 1]
    ax.hist(mae, bins=40, color="salmon", edgecolor="black", alpha=0.7)
    ax.axvline(np.median(mae), color="red", linestyle="--", label=f"Median ${np.median(mae):,.0f}")
    ax.set_xlabel("MAE ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution MAE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = outdir / "config_D_mfe_mae.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def main():
    t0 = time_mod.time()
    print("MFE/MAE Analysis -- Config D (1s resolution)\n")

    trades = reconstruct_config_d()
    nq_1s, ym_1s = load_1s_for_trades(trades)
    results = compute_mfe_mae(trades, nq_1s, ym_1s)
    valid = print_analysis(results)
    make_charts(valid)

    elapsed = time_mod.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
