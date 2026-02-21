"""Deep analysis of refined grid search results.

Re-runs top configs with full backtest engine to compute:
- Sharpe, Calmar, Max DD, equity curve
- Winning vs losing trade breakdown
- Hourly entry analysis (best/worst hours)
- Duration analysis (winners vs losers)
- Multi-criteria ranking (PnL, Sharpe, balanced, Calmar)
- Improvement axes

Usage:
    python scripts/analyze_grid_results.py
    python scripts/analyze_grid_results.py --csv output/refined_grid_results.csv --top 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.generator import generate_signals_numba
from src.signals.filters import (
    ConfidenceConfig, compute_confidence,
    _apply_conf_filter_numba, apply_time_stop, apply_window_filter_numba,
)
from src.backtest.engine import run_backtest_vectorized
from src.backtest.performance import PerformanceMetrics

# ======================================================================
# Constants
# ======================================================================

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
METRICS_CFG = MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6)
CONF_CFG = ConfidenceConfig()
FLAT_MIN = 930

WINDOWS_MAP = {
    "02:00-12:00": (120, 720), "02:00-13:00": (120, 780),
    "02:00-14:00": (120, 840), "02:00-15:00": (120, 900),
    "03:00-13:00": (180, 780), "03:00-14:00": (180, 840),
    "04:00-13:00": (240, 780), "04:00-14:00": (240, 840),
    "04:00-15:00": (240, 900),
}


def load_and_prepare():
    """Load data and precompute shared arrays."""
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    return aligned, px_a, px_b, idx, minutes


def run_full_backtest_for_config(row, aligned, px_a, px_b, idx, minutes):
    """Run full backtest for a single config and return detailed results."""
    ols_w = int(row["ols"])
    zw = int(row["zw"])
    z_entry = row["z_entry"]
    z_exit = row["z_exit"]
    z_stop = row["z_stop"]
    min_conf = row["conf"]
    ts = int(row["time_stop"])
    win_name = row["window"]
    entry_start, entry_end = WINDOWS_MAP[win_name]

    # Hedge + spread
    est = create_estimator("ols_rolling", window=ols_w, zscore_window=zw)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    # Z-score
    mu = spread.rolling(zw).mean()
    sigma = spread.rolling(zw).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    # Signals + filters
    raw = generate_signals_numba(zscore, z_entry, z_exit, z_stop)
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], METRICS_CFG)
    confidence = compute_confidence(metrics, CONF_CFG).values
    sig = apply_time_stop(raw, ts)
    sig = _apply_conf_filter_numba(sig, confidence, min_conf)
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    # Full backtest
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        MULT_A, MULT_B, TICK_A, TICK_B,
        SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
    )

    return bt, sig


def compute_detailed_metrics(bt, idx):
    """Compute Sharpe, Calmar, DD, hourly analysis from full backtest result."""
    result = {}
    n_trades = bt["trades"]
    if n_trades == 0:
        return None

    trade_pnls = bt["trade_pnls"]
    trade_sides = bt["trade_sides"]
    te = bt["trade_entry_bars"]
    tx = bt["trade_exit_bars"]
    equity = bt["equity"]

    # Basic
    result["trades"] = n_trades
    result["pnl"] = bt["pnl"]
    result["pf"] = bt["profit_factor"]
    result["win_rate"] = bt["win_rate"]
    result["avg_pnl"] = bt["avg_pnl_trade"]
    result["avg_dur"] = bt["avg_duration_bars"]

    # Winners vs Losers
    wins_mask = trade_pnls > 0
    losses_mask = trade_pnls <= 0
    n_wins = int(wins_mask.sum())
    n_losses = int(losses_mask.sum())
    durations = tx - te

    result["n_wins"] = n_wins
    result["n_losses"] = n_losses
    result["avg_win"] = float(trade_pnls[wins_mask].mean()) if n_wins > 0 else 0
    result["avg_loss"] = float(trade_pnls[losses_mask].mean()) if n_losses > 0 else 0
    result["max_win"] = float(trade_pnls.max())
    result["max_loss"] = float(trade_pnls.min())
    result["avg_dur_win"] = float(durations[wins_mask].mean()) if n_wins > 0 else 0
    result["avg_dur_loss"] = float(durations[losses_mask].mean()) if n_losses > 0 else 0

    # Long vs Short
    longs = trade_sides == 1
    shorts = trade_sides == -1
    result["n_long"] = int(longs.sum())
    result["n_short"] = int(shorts.sum())
    result["pnl_long"] = float(trade_pnls[longs].sum()) if longs.any() else 0
    result["pnl_short"] = float(trade_pnls[shorts].sum()) if shorts.any() else 0
    result["wr_long"] = float((trade_pnls[longs] > 0).sum() / longs.sum() * 100) if longs.any() else 0
    result["wr_short"] = float((trade_pnls[shorts] > 0).sum() / shorts.sum() * 100) if shorts.any() else 0

    # Sharpe (annualized from trade PnLs)
    years = (idx[-1] - idx[0]).days / 365.25
    trades_per_year = n_trades / years if years > 0 else 0
    if n_trades > 1:
        result["sharpe"] = float(trade_pnls.mean() / trade_pnls.std() * np.sqrt(trades_per_year))
    else:
        result["sharpe"] = 0.0

    # Max Drawdown from equity curve
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    result["max_dd"] = float(dd.min())
    result["max_dd_pct"] = float(dd.min() / INITIAL_CAPITAL * 100)

    # Calmar = annualized return / |max DD|
    annual_return = bt["pnl"] / years if years > 0 else 0
    result["calmar"] = float(annual_return / abs(result["max_dd"])) if result["max_dd"] != 0 else 0

    # Trades per year
    result["trd_an"] = round(trades_per_year, 1)

    # Hourly entry analysis
    entry_hours = idx[te].hour
    hourly_stats = {}
    for h in sorted(set(entry_hours)):
        h_mask = entry_hours == h
        h_pnls = trade_pnls[h_mask]
        hourly_stats[h] = {
            "trades": int(h_mask.sum()),
            "pnl": float(h_pnls.sum()),
            "wr": float((h_pnls > 0).sum() / h_mask.sum() * 100) if h_mask.sum() > 0 else 0,
            "avg_pnl": float(h_pnls.mean()),
        }
    result["hourly"] = hourly_stats

    # Quarterly PnL
    entry_dates = idx[te]
    quarters = entry_dates.to_period("Q")
    quarterly_stats = {}
    for q in sorted(set(quarters)):
        q_mask = quarters == q
        q_pnls = trade_pnls[q_mask]
        quarterly_stats[str(q)] = {
            "trades": int(q_mask.sum()),
            "pnl": float(q_pnls.sum()),
            "wr": float((q_pnls > 0).sum() / q_mask.sum() * 100) if q_mask.sum() > 0 else 0,
        }
    result["quarterly"] = quarterly_stats

    return result


def print_config_header(row, rank, criterion):
    """Print config header."""
    ts_label = f"{row['time_stop']:.0f}b" if row['time_stop'] > 0 else "none"
    print(f"\n  #{rank} [{criterion}] OLS={row['ols']:.0f} ZW={row['zw']:.0f} "
          f"z=({row['z_entry']:.2f}/{row['z_exit']:.2f}/{row['z_stop']:.2f}) "
          f"conf={row['conf']:.0f} ts={ts_label} win={row['window']}")


def print_detailed_result(r):
    """Print detailed analysis for one config."""
    print(f"    PnL: ${r['pnl']:,.0f} | PF: {r['pf']:.2f} | Win: {r['win_rate']:.1f}% "
          f"| Sharpe: {r['sharpe']:.2f} | Calmar: {r['calmar']:.2f}")
    print(f"    Trades: {r['trades']} ({r['trd_an']:.0f}/an) | Max DD: ${r['max_dd']:,.0f} ({r['max_dd_pct']:.1f}%)")
    print(f"    Avg PnL: ${r['avg_pnl']:,.0f} | Avg Win: ${r['avg_win']:,.0f} | Avg Loss: ${r['avg_loss']:,.0f}")
    print(f"    Max Win: ${r['max_win']:,.0f} | Max Loss: ${r['max_loss']:,.0f}")
    print(f"    Dur moy: {r['avg_dur']:.1f}b | Win: {r['avg_dur_win']:.1f}b | Loss: {r['avg_dur_loss']:.1f}b")
    print(f"    Long: {r['n_long']} trades, ${r['pnl_long']:,.0f}, WR={r['wr_long']:.0f}% "
          f"| Short: {r['n_short']} trades, ${r['pnl_short']:,.0f}, WR={r['wr_short']:.0f}%")


def select_diverse_top(df, criterion_col, n=5, ascending=False, min_trades=15):
    """Select top N configs by criterion, ensuring diversity (different param patterns)."""
    df_filtered = df[df["trades"] >= min_trades].copy()
    df_sorted = df_filtered.sort_values(criterion_col, ascending=ascending)

    selected = []
    seen_patterns = set()

    for _, row in df_sorted.iterrows():
        # Pattern = (ols, zw, z_entry, window) — allow variation in other params
        pattern = (row["ols"], row["zw"], row["z_entry"], row["window"])
        if pattern not in seen_patterns:
            selected.append(row)
            seen_patterns.add(pattern)
        if len(selected) >= n:
            break

    return pd.DataFrame(selected)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "output" / "grids" / "refined_grid_results.csv"))
    parser.add_argument("--top", type=int, default=30, help="Number of top configs to deep-analyze")
    args = parser.parse_args()

    # Load grid results
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} profitable configs from {args.csv}")
    print(f"Total trades range: {df['trades'].min()}-{df['trades'].max()}")
    print(f"PnL range: ${df['pnl'].min():,.0f} - ${df['pnl'].max():,.0f}")

    # Load market data
    print("\nLoading NQ_YM data...")
    aligned, px_a, px_b, idx, minutes = load_and_prepare()
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years:.1f} years")

    # ================================================================
    # SECTION 1: Multi-criteria rankings
    # ================================================================
    print("\n" + "=" * 120)
    print(" SECTION 1: TOP 5 PAR CRITERE (patterns differents)")
    print("=" * 120)

    # We need a proxy for Sharpe/Calmar from grid results
    # Trade Sharpe proxy: avg_pnl / estimated_std, using PF relationship
    # Better: just select top by different grid metrics, then deep-analyze
    criteria = {
        "PnL": ("pnl", False),
        "Profit Factor": ("pf", False),
        "Win Rate": ("win_rate", False),
        "Trades (volume)": ("trades", False),
    }

    # Also compute a balanced score
    df["balanced"] = (
        (df["pf"].clip(upper=5) / 5) * 0.30 +
        (df["pnl"] / df["pnl"].max()) * 0.30 +
        (df["win_rate"] / 100) * 0.20 +
        (df["trades"] / df["trades"].max()).clip(upper=1) * 0.20
    )
    criteria["Balanced (PF+PnL+WR+Vol)"] = ("balanced", False)

    # Collect all unique configs to deep-analyze
    configs_to_analyze = {}

    for crit_name, (col, asc) in criteria.items():
        top = select_diverse_top(df, col, n=5, ascending=asc)
        print(f"\n  --- TOP 5 by {crit_name} ---")
        for i, (_, row) in enumerate(top.iterrows()):
            ts_label = f"{row['time_stop']:.0f}b" if row['time_stop'] > 0 else "none"
            print(f"    #{i+1} OLS={row['ols']:.0f} ZW={row['zw']:.0f} "
                  f"z=({row['z_entry']:.2f}/{row['z_exit']:.2f}/{row['z_stop']:.2f}) "
                  f"conf={row['conf']:.0f} ts={ts_label} win={row['window']} | "
                  f"Trd={row['trades']:.0f} WR={row['win_rate']:.1f}% PnL=${row['pnl']:,.0f} PF={row['pf']:.2f}")
            key = (row["ols"], row["zw"], row["z_entry"], row["z_exit"],
                   row["z_stop"], row["conf"], row["time_stop"], row["window"])
            configs_to_analyze[key] = row

    # ================================================================
    # SECTION 2: Deep analysis of top configs
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(f" SECTION 2: ANALYSE APPROFONDIE ({len(configs_to_analyze)} configs uniques)")
    print(f"{'=' * 120}")

    deep_results = []

    for i, (key, row) in enumerate(configs_to_analyze.items()):
        print(f"\n  Analyzing {i+1}/{len(configs_to_analyze)}: OLS={key[0]} ZW={key[1]} "
              f"z=({key[2]:.2f}/{key[3]:.2f}/{key[4]:.2f}) conf={key[5]:.0f}...", end="", flush=True)

        bt, sig = run_full_backtest_for_config(row, aligned, px_a, px_b, idx, minutes)
        r = compute_detailed_metrics(bt, idx)

        if r is not None:
            r["config"] = row.to_dict()
            deep_results.append(r)
            print(f" PnL=${r['pnl']:,.0f} Sharpe={r['sharpe']:.2f} Calmar={r['calmar']:.2f}")
        else:
            print(" NO TRADES")

    if not deep_results:
        print("\nAucun resultat a analyser.")
        return

    # ================================================================
    # SECTION 3: Rankings by Sharpe, Calmar, Balanced
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 3: CLASSEMENTS DETAILLES")
    print("=" * 120)

    dr = pd.DataFrame([{
        **{k: v for k, v in r["config"].items()},
        **{k: v for k, v in r.items() if k not in ("config", "hourly", "quarterly")}
    } for r in deep_results])

    for rank_name, rank_col in [("PnL", "pnl"), ("Sharpe", "sharpe"),
                                  ("Calmar", "calmar"), ("Balanced", "balanced_deep")]:
        if rank_col == "balanced_deep":
            dr["balanced_deep"] = (
                (dr["sharpe"].clip(lower=0) / dr["sharpe"].max()) * 0.30 +
                (dr["pnl"] / dr["pnl"].max()) * 0.25 +
                (dr["calmar"].clip(lower=0) / max(dr["calmar"].max(), 0.01)) * 0.25 +
                (dr["win_rate"] / 100) * 0.20
            )

        top5 = dr.sort_values(rank_col, ascending=False).head(5)
        print(f"\n  --- TOP 5 by {rank_name} ---")
        print(f"  {'#':>2} {'OLS':>5} {'ZW':>3} {'z_e':>5} {'z_x':>4} {'z_s':>5} {'Cf':>3} {'TS':>4} "
              f"{'Window':<12} {'Trd':>4} {'WR%':>5} {'PnL':>10} {'PF':>5} {'Sharpe':>6} "
              f"{'Calmar':>6} {'MaxDD':>8} {'Avg$':>7} {'Trd/An':>6}")
        print("  " + "-" * 115)
        for rank, (_, r) in enumerate(top5.iterrows()):
            ts_l = f"{r['time_stop']:.0f}b" if r['time_stop'] > 0 else "  -"
            print(f"  {rank+1:>2} {r['ols']:>5.0f} {r['zw']:>3.0f} {r['z_entry']:>5.2f} {r['z_exit']:>4.2f} "
                  f"{r['z_stop']:>5.2f} {r['conf']:>3.0f} {ts_l:>4} {r['window']:<12} "
                  f"{r['trades']:>4.0f} {r['win_rate']:>5.1f} ${r['pnl']:>9,.0f} {r['pf']:>5.2f} "
                  f"{r['sharpe']:>6.2f} {r['calmar']:>6.2f} ${r['max_dd']:>7,.0f} "
                  f"${r['avg_pnl']:>6,.0f} {r['trd_an']:>6.1f}")

    # ================================================================
    # SECTION 4: Best config deep dive
    # ================================================================
    # Pick the balanced #1
    dr["balanced_deep"] = (
        (dr["sharpe"].clip(lower=0) / max(dr["sharpe"].max(), 0.01)) * 0.30 +
        (dr["pnl"] / max(dr["pnl"].max(), 1)) * 0.25 +
        (dr["calmar"].clip(lower=0) / max(dr["calmar"].max(), 0.01)) * 0.25 +
        (dr["win_rate"] / 100) * 0.20
    )
    best_idx = dr["balanced_deep"].idxmax()
    best_r = None
    for r in deep_results:
        cfg = r["config"]
        if (cfg["ols"] == dr.loc[best_idx, "ols"] and
            cfg["zw"] == dr.loc[best_idx, "zw"] and
            cfg["z_entry"] == dr.loc[best_idx, "z_entry"]):
            best_r = r
            break

    if best_r is None:
        best_r = deep_results[0]

    print(f"\n\n{'=' * 120}")
    print(" SECTION 4: DEEP DIVE — MEILLEURE CONFIG BALANCED")
    print("=" * 120)

    cfg = best_r["config"]
    ts_label = f"{cfg['time_stop']:.0f}b" if cfg['time_stop'] > 0 else "none"
    print(f"\n  Config: OLS={cfg['ols']:.0f} ZW={cfg['zw']:.0f} "
          f"z=({cfg['z_entry']:.2f}/{cfg['z_exit']:.2f}/{cfg['z_stop']:.2f}) "
          f"conf={cfg['conf']:.0f} ts={ts_label} window={cfg['window']}")

    print(f"\n  --- Performance globale ---")
    print_detailed_result(best_r)

    # Hourly analysis
    print(f"\n  --- Analyse horaire (heure d'entree CT) ---")
    print(f"  {'Heure':>6} {'Trades':>6} {'PnL':>10} {'WR%':>6} {'Avg PnL':>8}")
    print(f"  {'-'*40}")
    hourly = best_r["hourly"]
    for h in sorted(hourly.keys()):
        hs = hourly[h]
        flag = " ***" if hs["pnl"] < 0 and hs["trades"] >= 3 else (" ++++" if hs["avg_pnl"] > 200 else "")
        print(f"  {h:>5}h {hs['trades']:>6} ${hs['pnl']:>9,.0f} {hs['wr']:>5.1f}% ${hs['avg_pnl']:>7,.0f}{flag}")

    # Quarterly analysis
    print(f"\n  --- Analyse trimestrielle ---")
    print(f"  {'Trimestre':>10} {'Trades':>6} {'PnL':>10} {'WR%':>6}")
    print(f"  {'-'*36}")
    quarterly = best_r["quarterly"]
    losing_quarters = 0
    for q in sorted(quarterly.keys()):
        qs = quarterly[q]
        flag = " ***" if qs["pnl"] < 0 else ""
        print(f"  {q:>10} {qs['trades']:>6} ${qs['pnl']:>9,.0f} {qs['wr']:>5.1f}%{flag}")
        if qs["pnl"] < 0:
            losing_quarters += 1

    total_q = len(quarterly)
    print(f"\n  Trimestres perdants: {losing_quarters}/{total_q} ({losing_quarters/total_q*100:.0f}%)")

    # ================================================================
    # SECTION 5: Losing trades analysis (across all deep-analyzed configs)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 5: ANALYSE DES TRADES PERDANTS (top configs)")
    print("=" * 120)

    all_avg_losses = [r["avg_loss"] for r in deep_results if r["avg_loss"] < 0]
    all_max_losses = [r["max_loss"] for r in deep_results]
    all_dur_losses = [r["avg_dur_loss"] for r in deep_results if r["avg_dur_loss"] > 0]
    all_dur_wins = [r["avg_dur_win"] for r in deep_results if r["avg_dur_win"] > 0]

    print(f"\n  Across {len(deep_results)} configs:")
    print(f"    Avg loss moyen:    ${np.mean(all_avg_losses):,.0f}")
    print(f"    Max loss moyen:    ${np.mean(all_max_losses):,.0f}")
    print(f"    Duree moy loss:    {np.mean(all_dur_losses):.1f} bars ({np.mean(all_dur_losses)*5:.0f} min)")
    print(f"    Duree moy win:     {np.mean(all_dur_wins):.1f} bars ({np.mean(all_dur_wins)*5:.0f} min)")
    print(f"    Ratio dur win/loss: {np.mean(all_dur_wins)/np.mean(all_dur_losses):.2f}x")

    # ================================================================
    # SECTION 6: Parameter sensitivity (from grid CSV)
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 6: SENSIBILITE PAR PARAMETRE (toutes configs PF > 1.3)")
    print("=" * 120)

    df_good = df[df["pf"] > 1.3].copy()
    if len(df_good) > 0:
        for dim, col in [("OLS", "ols"), ("ZW", "zw"), ("Window", "window"),
                          ("z_entry", "z_entry"), ("z_exit", "z_exit"),
                          ("z_stop", "z_stop"), ("Confidence", "conf"),
                          ("Time Stop", "time_stop")]:
            grp = df_good.groupby(col).agg(
                count=("pf", "size"),
                avg_pf=("pf", "mean"),
                med_pf=("pf", "median"),
                avg_wr=("win_rate", "mean"),
                avg_trades=("trades", "mean"),
                avg_pnl=("pnl", "mean"),
                max_pnl=("pnl", "max"),
            ).round(2)
            print(f"\n  {dim}:")
            print(f"    {'Value':<12} {'Count':>6} {'Avg PF':>7} {'Med PF':>7} {'Avg WR%':>7} "
                  f"{'Avg Trd':>7} {'Avg PnL':>10} {'Max PnL':>10}")
            print(f"    {'-'*72}")
            for idx_val, row in grp.iterrows():
                print(f"    {str(idx_val):<12} {row['count']:>6.0f} {row['avg_pf']:>7.2f} {row['med_pf']:>7.2f} "
                      f"{row['avg_wr']:>7.1f} {row['avg_trades']:>7.0f} ${row['avg_pnl']:>9,.0f} "
                      f"${row['max_pnl']:>9,.0f}")

    # ================================================================
    # SECTION 7: Axes d'amelioration
    # ================================================================
    print(f"\n\n{'=' * 120}")
    print(" SECTION 7: AXES D'AMELIORATION")
    print("=" * 120)

    # Auto-detect improvement axes from data
    print("\n  1. HEURES CRITIQUES:")
    if best_r and "hourly" in best_r:
        bad_hours = [h for h, s in best_r["hourly"].items()
                     if s["pnl"] < 0 and s["trades"] >= 3]
        good_hours = sorted(best_r["hourly"].items(), key=lambda x: x[1]["avg_pnl"], reverse=True)
        if bad_hours:
            print(f"     Heures perdantes (>= 3 trades): {bad_hours}")
            print(f"     -> Bloquer les entrees a ces heures pourrait ameliorer le PF")
        if good_hours:
            best_h = good_hours[0]
            print(f"     Meilleure heure: {best_h[0]}h ({best_h[1]['trades']} trades, "
                  f"avg ${best_h[1]['avg_pnl']:,.0f})")

    print("\n  2. DUREE DES TRADES:")
    if all_dur_wins and all_dur_losses:
        avg_win_dur = np.mean(all_dur_wins)
        avg_loss_dur = np.mean(all_dur_losses)
        if avg_loss_dur > avg_win_dur:
            print(f"     Perdants durent plus ({avg_loss_dur:.1f}b) que gagnants ({avg_win_dur:.1f}b)")
            print(f"     -> Time stop optimal autour de {int(avg_win_dur * 1.5)}-{int(avg_win_dur * 2)} bars")
        else:
            print(f"     Gagnants ({avg_win_dur:.1f}b) durent plus que perdants ({avg_loss_dur:.1f}b) — bon signe")

    print("\n  3. EQUILIBRE LONG/SHORT:")
    if best_r:
        ratio = best_r["pnl_long"] / max(abs(best_r["pnl_short"]), 1)
        if ratio > 3:
            print(f"     DESEQUILIBRE: Long={best_r['pnl_long']:,.0f} vs Short={best_r['pnl_short']:,.0f}")
            print(f"     -> Le profit vient surtout du cote long — risque de biais directionnel")
        elif ratio < 0.33:
            print(f"     DESEQUILIBRE: Short domine (Long={best_r['pnl_long']:,.0f}, Short={best_r['pnl_short']:,.0f})")
        else:
            print(f"     Equilibre OK: Long=${best_r['pnl_long']:,.0f} ({best_r['n_long']} trades) "
                  f"vs Short=${best_r['pnl_short']:,.0f} ({best_r['n_short']} trades)")

    print("\n  4. STABILITE TEMPORELLE:")
    if best_r and "quarterly" in best_r:
        q_pnls = [v["pnl"] for v in best_r["quarterly"].values()]
        if q_pnls:
            q_positive = sum(1 for p in q_pnls if p > 0)
            q_total = len(q_pnls)
            print(f"     {q_positive}/{q_total} trimestres positifs ({q_positive/q_total*100:.0f}%)")
            if q_positive / q_total < 0.6:
                print(f"     -> ATTENTION: moins de 60% de trimestres positifs — fragile")
            else:
                print(f"     -> Bonne stabilite temporelle")

    print("\n  5. SUGGESTIONS GRID SUIVANTE:")
    if len(df_good) > 0:
        # Find optimal ranges per parameter
        for col_name, col in [("z_entry", "z_entry"), ("z_exit", "z_exit"),
                               ("z_stop", "z_stop"), ("conf", "conf"), ("time_stop", "time_stop")]:
            grp = df_good.groupby(col)["pf"].mean()
            best_val = grp.idxmax()
            print(f"     {col_name}: optimal={best_val}, affiner autour de [{best_val*0.95:.2f}, {best_val*1.05:.2f}]")

    print(f"\n{'=' * 120}")
    print(" FIN DE L'ANALYSE")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
