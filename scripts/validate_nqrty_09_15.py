"""Validation complete NQ_RTY 02:00-15:00 -- 2 configs candidates.

Analyses :
  1. CPCV(10,2) -- 45 chemins
  2. Walk-Forward rolling (2y IS / 6m OOS / 6m step)
  3. Equity curve complete
  4. Analyse trades perdants (distribution, par heure, par annee)
  5. PnL par heure de la journee
  6. Propfirm compliance

Usage:
    python scripts/validate_nqrty_09_15.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import apply_window_filter_numba
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.validation.cpcv import CPCVConfig, run_cpcv

OUTPUT_DIR = PROJECT_ROOT / "output" / "NQ_RTY"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# Constants
# ======================================================================

MULT_A, MULT_B = 20.0, 50.0
TICK_A, TICK_B = 0.25, 0.10
SLIPPAGE = 1
COMMISSION = 2.20

ENTRY_START_MIN = 2 * 60    # 02:00 CT
ENTRY_END_MIN = 15 * 60     # 15:00 CT
FLAT_MIN = 15 * 60 + 30     # 15:30 CT

Z_STOP_DISABLED = 20.0

# Walk-Forward params (same as Phase 13c)
WF_IS_BARS = 2 * 252 * 78     # ~2 years IS
WF_OOS_BARS = 126 * 78        # ~6 months OOS
WF_STEP_BARS = 126 * 78       # ~6 months step


# ======================================================================
# Config definitions
# ======================================================================

@dataclass
class ConfigDef:
    name: str
    ols_window: int
    zscore_window: int
    z_entry: float
    z_exit: float
    z_stop: float
    time_stop: int
    adf_threshold: float
    corr_threshold: float
    adf_window: int
    corr_window: int


CONFIGS = [
    ConfigDef(
        name="Config_A_volume",
        ols_window=6600, zscore_window=15,
        z_entry=2.75, z_exit=0.50, z_stop=Z_STOP_DISABLED,
        time_stop=0,
        adf_threshold=-2.86, corr_threshold=0.70,
        adf_window=48, corr_window=24,
    ),
    ConfigDef(
        name="Config_B_robust",
        ols_window=5280, zscore_window=15,
        z_entry=3.00, z_exit=0.00, z_stop=Z_STOP_DISABLED,
        time_stop=0,
        adf_threshold=-2.86, corr_threshold=0.70,
        adf_window=48, corr_window=24,
    ),
]


# ======================================================================
# Gate filter (ADF + Corr AND)
# ======================================================================

def apply_double_gate(sig: np.ndarray,
                      adf_values: np.ndarray, adf_threshold: float,
                      corr_values: np.ndarray, corr_threshold: float) -> np.ndarray:
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0:
            adf_val = adf_values[t]
            corr_val = corr_values[t]
            adf_ok = not np.isnan(adf_val) and adf_val < adf_threshold
            corr_ok = not np.isnan(corr_val) and corr_val > corr_threshold
            if not (adf_ok and corr_ok):
                out[t] = 0
        prev = out[t]
    return out


# ======================================================================
# Run one config
# ======================================================================

def run_config(cfg: ConfigDef, aligned, idx):
    """Run backtest + all validations for one config."""
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values

    # OLS
    est = create_estimator("ols_rolling", window=cfg.ols_window,
                           zscore_window=cfg.zscore_window)
    hr = est.estimate(aligned)
    spread = hr.spread
    beta = hr.beta.values

    mu = spread.rolling(cfg.zscore_window).mean()
    sigma = spread.rolling(cfg.zscore_window).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace(
            [np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

    # Metrics for gates
    mc = MetricsConfig(adf_window=cfg.adf_window, hurst_window=128,
                       halflife_window=cfg.adf_window,
                       correlation_window=cfg.corr_window)
    metrics = compute_all_metrics(spread, aligned.df["close_a"],
                                  aligned.df["close_b"], mc)
    adf_values = metrics["adf_stat"].values
    corr_values = metrics["correlation"].values

    # Signals
    raw_sig = generate_signals_numba(zscore, cfg.z_entry, cfg.z_exit, cfg.z_stop)

    # Window filter
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    sig = apply_window_filter_numba(raw_sig.copy(), minutes,
                                    ENTRY_START_MIN, ENTRY_END_MIN, FLAT_MIN)

    # Double gate
    sig = apply_double_gate(sig, adf_values, cfg.adf_threshold,
                            corr_values, cfg.corr_threshold)

    # Backtest (vectorized for full trade data)
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        mult_a=MULT_A, mult_b=MULT_B,
        tick_a=TICK_A, tick_b=TICK_B,
        slippage_ticks=SLIPPAGE, commission=COMMISSION)

    return bt, idx


# ======================================================================
# Analysis functions
# ======================================================================

def print_summary(cfg: ConfigDef, bt: dict):
    print(f"\n{'='*80}")
    print(f"  {cfg.name}")
    print(f"{'='*80}")
    print(f"  OLS={cfg.ols_window} ZW={cfg.zscore_window} "
          f"ze={cfg.z_entry} zx={cfg.z_exit} zs=OFF ts={cfg.time_stop}")
    print(f"  Gates: ADF < {cfg.adf_threshold} (w={cfg.adf_window}), "
          f"Corr > {cfg.corr_threshold} (w={cfg.corr_window})")
    print("  Window: 02:00-15:00 CT, Flat 15:30")
    print(f"  Commission: ${COMMISSION}/side (Phidias)")
    print("  ---")
    print(f"  Trades: {bt['trades']}")
    print(f"  Win Rate: {bt['win_rate']:.1f}%")
    print(f"  PnL: ${bt['pnl']:,.0f}")
    print(f"  Profit Factor: {bt['profit_factor']:.2f}")
    print(f"  Avg PnL/trade: ${bt['avg_pnl_trade']:.0f}")
    print(f"  Avg Duration: {bt['avg_duration_bars']:.1f} bars "
          f"({bt['avg_duration_bars']*5:.0f} min)")

    # Max drawdown from equity
    eq = bt["equity"]
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = dd.min()
    print(f"  Max Drawdown: ${max_dd:,.0f}")


def run_cpcv_analysis(cfg: ConfigDef, bt: dict, n_bars: int):
    print("\n  --- CPCV(10,2) ---")
    entries = bt["trade_entry_bars"]
    exits = bt["trade_exit_bars"]
    pnls = bt["trade_pnls"]

    cpcv_cfg = CPCVConfig(n_folds=10, n_test_folds=2,
                          purge_bars=100, min_trades_per_path=5)
    result = run_cpcv(entries, exits, pnls, n_bars, cpcv_cfg)

    print(f"  Paths: {result['n_valid_paths']}/{result['n_paths']}")
    print(f"  Median Sharpe: {result['median_sharpe']:.4f}")
    print(f"  Mean Sharpe: {result['mean_sharpe']:.4f}")
    print(f"  % Positive: {result['pct_positive']:.1f}%")
    print(f"  Min/Max Sharpe: {result['min_sharpe']:.4f} / {result['max_sharpe']:.4f}")

    return result


def run_walkforward(cfg: ConfigDef, bt: dict, n_bars: int):
    print("\n  --- Walk-Forward (2y IS / 6m OOS / 6m step) ---")
    entries = bt["trade_entry_bars"]
    exits = bt["trade_exit_bars"]
    pnls = bt["trade_pnls"]

    folds = []
    start = 0
    while start + WF_IS_BARS + WF_OOS_BARS <= n_bars:
        is_s, is_e = start, start + WF_IS_BARS
        oos_s, oos_e = is_e, min(is_e + WF_OOS_BARS, n_bars)
        folds.append((is_s, is_e, oos_s, oos_e))
        start += WF_STEP_BARS

    print(f"  Folds: {len(folds)}")
    n_go = 0
    for i, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        # IS trades
        is_mask = (entries >= is_s) & (exits <= is_e)
        is_pnls = pnls[is_mask]
        is_gross_win = is_pnls[is_pnls > 0].sum()
        is_gross_loss = abs(is_pnls[is_pnls < 0].sum())
        is_pf = is_gross_win / is_gross_loss if is_gross_loss > 0 else 99.0

        # OOS trades
        oos_mask = (entries >= oos_s) & (exits <= oos_e)
        oos_pnls = pnls[oos_mask]
        oos_gross_win = oos_pnls[oos_pnls > 0].sum()
        oos_gross_loss = abs(oos_pnls[oos_pnls < 0].sum())
        oos_pf = oos_gross_win / oos_gross_loss if oos_gross_loss > 0 else 99.0
        oos_pnl = float(oos_pnls.sum())

        is_go = oos_pf > 1.0 and len(oos_pnls) >= 3
        if is_go:
            n_go += 1
        verdict = "GO" if is_go else "NO-GO"
        print(f"    Fold {i+1}: IS {len(is_pnls)}t PF {is_pf:.2f} | "
              f"OOS {len(oos_pnls)}t PF {oos_pf:.2f} ${oos_pnl:,.0f} [{verdict}]")

    print(f"  Verdict: {n_go}/{len(folds)} GO")
    return n_go, len(folds)


def analyze_trades(cfg: ConfigDef, bt: dict, idx: pd.DatetimeIndex):
    """Full trade-level analysis."""
    entries = bt["trade_entry_bars"]
    exits = bt["trade_exit_bars"]
    pnls = bt["trade_pnls"]
    sides = bt["trade_sides"]

    entry_times = idx[entries]
    exit_times = idx[exits]
    durations = (exits - entries)

    # Build trade DataFrame
    trades_df = pd.DataFrame({
        "entry_time": entry_times,
        "exit_time": exit_times,
        "entry_hour": entry_times.hour,
        "side": sides,
        "pnl": pnls,
        "duration_bars": durations,
        "duration_min": durations * 5,
        "year": entry_times.year,
    })
    trades_df["is_winner"] = trades_df["pnl"] > 0
    trades_df["is_loser"] = trades_df["pnl"] < 0

    # --- Losing trades analysis ---
    losers = trades_df[trades_df["is_loser"]].copy()
    winners = trades_df[trades_df["is_winner"]].copy()

    print("\n  --- TRADE ANALYSIS ---")
    print(f"  Winners: {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%), "
          f"avg ${winners['pnl'].mean():,.0f}, median ${winners['pnl'].median():,.0f}")
    print(f"  Losers:  {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%), "
          f"avg ${losers['pnl'].mean():,.0f}, median ${losers['pnl'].median():,.0f}")
    print(f"  Flat:    {(trades_df['pnl'] == 0).sum()}")

    print(f"\n  Winners avg duration: {winners['duration_min'].mean():.0f} min")
    print(f"  Losers avg duration:  {losers['duration_min'].mean():.0f} min")

    # --- Losers distribution ---
    print("\n  --- LOSING TRADES DISTRIBUTION ---")
    bins = [(-np.inf, -1000), (-1000, -500), (-500, -250), (-250, -100), (-100, 0)]
    for lo, hi in bins:
        count = ((losers['pnl'] > lo) & (losers['pnl'] <= hi)).sum()
        pct = count / len(losers) * 100 if len(losers) > 0 else 0
        pnl_sum = losers[(losers['pnl'] > lo) & (losers['pnl'] <= hi)]['pnl'].sum()
        if lo == -np.inf:
            label = f"< ${hi:,.0f}"
        else:
            label = f"${lo:,.0f} to ${hi:,.0f}"
        print(f"    {label:>20} : {count:>3} trades ({pct:>5.1f}%), total ${pnl_sum:,.0f}")

    # Worst 5 losers
    print("\n  --- WORST 5 LOSERS ---")
    worst = losers.nsmallest(5, 'pnl')
    for _, t in worst.iterrows():
        side_str = "LONG" if t['side'] == 1 else "SHORT"
        print(f"    {t['entry_time'].strftime('%Y-%m-%d %H:%M')} -> "
              f"{t['exit_time'].strftime('%H:%M')} | {side_str} | "
              f"${t['pnl']:,.0f} | {t['duration_min']:.0f}min")

    # --- PnL by hour ---
    print("\n  --- PNL PAR HEURE (entry hour CT) ---")
    hourly = trades_df.groupby("entry_hour").agg(
        trades=("pnl", "count"),
        pnl_sum=("pnl", "sum"),
        pnl_avg=("pnl", "mean"),
        wr=("is_winner", "mean"),
        losers_count=("is_loser", "sum"),
        losers_pnl=("pnl", lambda x: x[x < 0].sum()),
    ).reindex(range(2, 15))

    print(f"  {'Hour':>6} {'Trades':>7} {'PnL':>10} {'Avg$/t':>8} {'WR%':>6} "
          f"{'Losers':>7} {'Loss$':>10}")
    print(f"  {'-'*60}")
    for h, row in hourly.iterrows():
        if pd.isna(row['trades']) or row['trades'] == 0:
            print(f"  {h:>5}h {'0':>7}")
            continue
        print(f"  {h:>5}h {int(row['trades']):>7} ${row['pnl_sum']:>9,.0f} "
              f"${row['pnl_avg']:>7,.0f} {row['wr']*100:>5.1f}% "
              f"{int(row['losers_count']):>7} ${row['losers_pnl']:>9,.0f}")

    # Best/worst hours
    if not hourly.dropna().empty:
        best_h = hourly['pnl_sum'].idxmax()
        worst_h = hourly['pnl_sum'].idxmin()
        print(f"\n  Best hour:  {best_h}h CT (${hourly.loc[best_h, 'pnl_sum']:,.0f})")
        print(f"  Worst hour: {worst_h}h CT (${hourly.loc[worst_h, 'pnl_sum']:,.0f})")

    # --- PnL by year ---
    print("\n  --- PNL PAR ANNEE ---")
    yearly = trades_df.groupby("year").agg(
        trades=("pnl", "count"),
        pnl_sum=("pnl", "sum"),
        wr=("is_winner", "mean"),
    )
    for y, row in yearly.iterrows():
        print(f"    {y}: {int(row['trades'])}t, ${row['pnl_sum']:>8,.0f}, "
              f"WR {row['wr']*100:.1f}%")

    # --- Long vs Short ---
    print("\n  --- LONG vs SHORT ---")
    for side, label in [(1, "LONG"), (-1, "SHORT")]:
        sub = trades_df[trades_df['side'] == side]
        if len(sub) == 0:
            continue
        print(f"    {label}: {len(sub)}t ({len(sub)/len(trades_df)*100:.1f}%), "
              f"PnL ${sub['pnl'].sum():,.0f}, WR {sub['is_winner'].mean()*100:.1f}%, "
              f"avg ${sub['pnl'].mean():,.0f}/t")

    return trades_df


def plot_equity_curve(cfg: ConfigDef, bt: dict, idx: pd.DatetimeIndex):
    """Plot and save equity curve."""
    eq = bt["equity"]
    peak = np.maximum.accumulate(eq)
    dd = eq - peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.1})

    # Equity
    ax1.plot(idx, eq, color='#2196F3', linewidth=0.8, label='Equity')
    ax1.axhline(y=100_000, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Equity ($)')
    ax1.set_title(f'{cfg.name} -- Equity Curve\n'
                  f'OLS={cfg.ols_window} ZW={cfg.zscore_window} '
                  f'ze={cfg.z_entry} zx={cfg.z_exit} | '
                  f'ADF<{cfg.adf_threshold} Corr>{cfg.corr_threshold} | '
                  f'{bt["trades"]}t PF={bt["profit_factor"]:.2f}')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Mark trades
    entries = bt["trade_entry_bars"]
    pnls = bt["trade_pnls"]
    for i in range(len(entries)):
        bar = entries[i]
        if bar < len(idx):
            color = '#4CAF50' if pnls[i] > 0 else '#F44336'
            alpha = 0.3 if abs(pnls[i]) < 500 else 0.6
            ax1.axvline(x=idx[bar], color=color, alpha=alpha, linewidth=0.3)

    # Drawdown
    ax2.fill_between(idx, dd, 0, color='#F44336', alpha=0.4)
    ax2.set_ylabel('Drawdown ($)')
    ax2.set_xlabel('Date (CT)')
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.tight_layout()
    path = OUTPUT_DIR / f"equity_{cfg.name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Equity curve saved: {path}")


def plot_hourly_pnl(cfg: ConfigDef, trades_df: pd.DataFrame):
    """Plot PnL by entry hour."""
    hourly = trades_df.groupby("entry_hour").agg(
        pnl_sum=("pnl", "sum"),
        trades=("pnl", "count"),
        wr=("is_winner", "mean"),
    ).reindex(range(2, 15), fill_value=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in hourly['pnl_sum']]
    ax1.bar(hourly.index, hourly['pnl_sum'], color=colors, width=0.7)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('Total PnL ($)')
    ax1.set_title(f'{cfg.name} -- PnL by Entry Hour (CT)')
    ax1.grid(True, alpha=0.3, axis='y')

    for _i, (h, row) in enumerate(hourly.iterrows()):
        if row['pnl_sum'] != 0:
            ax1.annotate(f"${row['pnl_sum']:,.0f}",
                        xy=(h, row['pnl_sum']),
                        ha='center', va='bottom' if row['pnl_sum'] > 0 else 'top',
                        fontsize=7)

    ax2.bar(hourly.index, hourly['trades'], color='#607D8B', width=0.7)
    ax2.set_ylabel('# Trades')
    ax2.set_xlabel('Entry Hour (CT)')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.xticks(range(2, 15), [f"{h}:00" for h in range(2, 15)])
    plt.tight_layout()
    path = OUTPUT_DIR / f"hourly_pnl_{cfg.name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Hourly PnL plot saved: {path}")


def plot_loser_analysis(cfg: ConfigDef, trades_df: pd.DataFrame):
    """Plot losing trades analysis."""
    losers = trades_df[trades_df['pnl'] < 0].copy()
    if losers.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loser PnL distribution
    ax = axes[0, 0]
    ax.hist(losers['pnl'], bins=30, color='#F44336', alpha=0.7, edgecolor='white')
    ax.axvline(x=losers['pnl'].median(), color='black', linestyle='--',
               label=f"Median: ${losers['pnl'].median():,.0f}")
    ax.set_xlabel('PnL ($)')
    ax.set_ylabel('Count')
    ax.set_title('Losing Trades Distribution')
    ax.legend()

    # 2. Losers by hour
    ax = axes[0, 1]
    hourly_losers = losers.groupby('entry_hour').agg(
        count=('pnl', 'count'),
        total_loss=('pnl', 'sum'),
    ).reindex(range(2, 15), fill_value=0)
    ax.bar(hourly_losers.index, hourly_losers['total_loss'], color='#F44336', width=0.7)
    ax.set_xlabel('Entry Hour (CT)')
    ax.set_ylabel('Total Loss ($)')
    ax.set_title('Losing Trades by Hour')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # 3. Loser duration distribution
    ax = axes[1, 0]
    ax.hist(losers['duration_min'], bins=25, color='#FF9800', alpha=0.7, edgecolor='white')
    ax.axvline(x=losers['duration_min'].median(), color='black', linestyle='--',
               label=f"Median: {losers['duration_min'].median():.0f}min")
    ax.set_xlabel('Duration (min)')
    ax.set_ylabel('Count')
    ax.set_title('Losing Trades Duration')
    ax.legend()

    # 4. Cumulative PnL losers over time
    ax = axes[1, 1]
    losers_sorted = losers.sort_values('entry_time')
    cum_loss = losers_sorted['pnl'].cumsum()
    ax.plot(losers_sorted['entry_time'].values, cum_loss.values,
            color='#F44336', linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Loss ($)')
    ax.set_title('Cumulative Losing Trades PnL')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{cfg.name} -- Losing Trades Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = OUTPUT_DIR / f"losers_{cfg.name}.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Loser analysis plot saved: {path}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 80)
    print("  VALIDATION NQ_RTY 02:00-15:00 CT")
    print("  2 configs: Config_A_volume + Config_B_robust")
    print("=" * 80)

    # Load data
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERROR: No cached data for NQ_RTY 5min")
        return
    idx = aligned.df.index
    n_bars = len(idx)
    print(f"\nData: {n_bars:,} bars, {idx[0]} -> {idx[-1]}")

    for cfg in CONFIGS:
        print(f"\n{'#'*80}")
        print(f"  Processing: {cfg.name}")
        print(f"{'#'*80}")

        # Run backtest
        bt, _ = run_config(cfg, aligned, idx)

        if bt["trades"] < 10:
            print(f"  WARNING: Only {bt['trades']} trades, skipping validation.")
            continue

        # 1. Summary
        print_summary(cfg, bt)

        # 2. CPCV
        cpcv_result = run_cpcv_analysis(cfg, bt, n_bars)

        # 3. Walk-Forward
        wf_go, wf_total = run_walkforward(cfg, bt, n_bars)

        # 4. Trade analysis (losers, hourly, yearly)
        trades_df = analyze_trades(cfg, bt, idx)

        # 5. Plots
        plot_equity_curve(cfg, bt, idx)
        plot_hourly_pnl(cfg, trades_df)
        plot_loser_analysis(cfg, trades_df)

        # 6. Propfirm summary
        eq = bt["equity"]
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = dd.min()

        print("\n  --- PROPFIRM COMPLIANCE ---")
        print(f"  Max Drawdown: ${max_dd:,.0f}")
        dd_ok = abs(max_dd) < 5000
        print(f"  DD < $5,000 : {'PASS' if dd_ok else 'FAIL'}")

        # Daily PnL
        entries = bt["trade_entry_bars"]
        entry_dates = pd.Series(idx[entries].date)
        daily_pnl = pd.Series(bt["trade_pnls"]).groupby(entry_dates.values).sum()
        worst_day = daily_pnl.min()
        print(f"  Worst day: ${worst_day:,.0f}")
        print(f"  Daily limit $-4,500: {'PASS' if worst_day > -4500 else 'FAIL'}")
        print(f"  % days profitable: {(daily_pnl > 0).mean()*100:.1f}%")
        print(f"  Avg daily PnL: ${daily_pnl.mean():,.0f}")

        # Final verdict
        print(f"\n  {'='*60}")
        cpcv_ok = cpcv_result['pct_positive'] >= 60
        wf_ok = wf_go >= wf_total * 0.5
        print(f"  CPCV {cpcv_result['pct_positive']:.0f}% paths+ "
              f"({'PASS' if cpcv_ok else 'FAIL'})")
        print(f"  WF {wf_go}/{wf_total} GO "
              f"({'PASS' if wf_ok else 'FAIL'})")
        print(f"  DD {'PASS' if dd_ok else 'FAIL'}")
        all_pass = cpcv_ok and wf_ok and dd_ok
        print(f"  VERDICT: {'GO' if all_pass else 'REVIEW'}")
        print(f"  {'='*60}")

    print(f"\n\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
