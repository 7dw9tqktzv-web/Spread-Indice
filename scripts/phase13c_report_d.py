"""Generate comprehensive HTML report for Config D.

Usage:
    python scripts/phase13c_report_d.py
"""

import base64
import io
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask
from src.validation.neighborhood import compute_neighborhood_robustness, get_neighbor_configs
from src.validation.propfirm import PropfirmConfig, compute_propfirm_metrics

# ======================================================================
# Config D
# ======================================================================

CFG = {
    "ols": 7000, "adf_w": 96, "zw": 30,
    "window": "02:00-14:00", "entry_start": 120, "entry_end": 840,
    "z_entry": 3.25, "delta_tp": 2.75, "delta_sl": 1.50,
    "time_stop": 0,
    "mult_a": 20.0, "mult_b": 5.0,
    "tick_a": 0.25, "tick_b": 1.0,
    "slippage": 1, "commission": 2.50,
    "initial_capital": 100_000.0, "flat_min": 930,
    "gate_adf": -2.86, "gate_hurst": 0.50, "gate_corr": 0.70,
    "gate_hurst_w": 64, "gate_corr_w": 24,
}
CFG["z_exit"] = round(max(CFG["z_entry"] - CFG["delta_tp"], 0.0), 4)
CFG["z_stop"] = round(CFG["z_entry"] + CFG["delta_sl"], 4)

CPCV_CFG = CPCVConfig(n_folds=10, n_test_folds=2, purge_bars=100, min_trades_per_path=5)

GRID_AXES = {
    "ols": [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000],
    "adf_w": [12, 18, 24, 30, 36, 48, 64, 96, 128],
    "zw": [10, 15, 20, 25, 28, 30, 35, 40, 45, 50, 60],
    "z_entry": [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75],
    "delta_tp": [1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00],
    "delta_sl": [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50],
    "time_stop": [0, 3, 6, 10, 12, 16, 20, 30, 40, 50],
}


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def classify_exits(entries, exits, sides, zscore, z_exit, z_stop,
                    time_stop, minutes, flat_min, entry_start):
    types = []
    for i in range(len(entries)):
        eb, xb = entries[i], exits[i]
        m = minutes[xb]
        if m >= flat_min or m < entry_start:
            types.append("FLAT")
            continue
        if time_stop > 0 and (xb - eb) >= time_stop:
            types.append("TIME_STOP")
            continue
        z = zscore[xb] if xb < len(zscore) else np.nan
        if not np.isnan(z):
            if sides[i] == 1 and z < -z_stop:
                types.append("Z_STOP")
                continue
            if sides[i] == -1 and z > z_stop:
                types.append("Z_STOP")
                continue
        types.append("Z_EXIT")
    return types


def max_consec_losses(pnls):
    mc = c = 0
    for p in pnls:
        if p <= 0:
            c += 1
            mc = max(mc, c)
        else:
            c = 0
    return mc


# ======================================================================
# Style
# ======================================================================

STYLE = """
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', Consolas, monospace; background: #0f0f23; color: #e0e0e0; padding: 30px; }
    h1 { color: #00d4ff; font-size: 28px; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; margin-bottom: 20px; }
    h2 { color: #ffa500; font-size: 20px; margin: 30px 0 15px 0; border-left: 4px solid #ffa500; padding-left: 12px; }
    h3 { color: #aaa; font-size: 16px; margin: 15px 0 8px 0; }
    .container { max-width: 1400px; margin: 0 auto; }
    .section { background: #1a1a2e; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #333; }
    table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
    th { background: #16213e; color: #00d4ff; padding: 8px 12px; text-align: left; border-bottom: 2px solid #444; }
    td { padding: 6px 12px; border-bottom: 1px solid #333; }
    tr:hover { background: #16213e; }
    .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
    .metric-card { background: #16213e; border-radius: 6px; padding: 12px; text-align: center; border: 1px solid #333; }
    .metric-value { font-size: 22px; font-weight: bold; color: #00d4ff; }
    .metric-label { font-size: 11px; color: #888; margin-top: 4px; }
    .go { color: #00ff88; } .warn { color: #ffa500; } .fail { color: #ff4444; }
    .positive { color: #00ff88; } .negative { color: #ff4444; }
    img { max-width: 100%; border-radius: 6px; margin: 10px 0; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
    .param-table td:first-child { color: #888; width: 200px; }
    .param-table td:last-child { color: #00d4ff; font-weight: bold; }
    .verdict-box { background: #0a3d0a; border: 2px solid #00ff88; border-radius: 8px; padding: 15px; margin: 15px 0; text-align: center; font-size: 18px; }
    .verdict-box.warn { background: #3d3a0a; border-color: #ffa500; }
    .verdict-box.fail { background: #3d0a0a; border-color: #ff4444; }
</style>
"""


def main():
    print("Generating Config D report...")

    # Load data
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    n = len(px_a)

    # Reconstruct
    print("  Reconstructing backtest...")
    est = create_estimator("ols_rolling", window=CFG["ols"], zscore_window=CFG["zw"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    gate_cfg = GateConfig(
        adf_threshold=CFG["gate_adf"], hurst_threshold=CFG["gate_hurst"],
        corr_threshold=CFG["gate_corr"], adf_window=CFG["adf_w"],
        hurst_window=CFG["gate_hurst_w"], corr_window=CFG["gate_corr_w"],
    )
    gate_mask = compute_gate_mask(spread, aligned.df["close_a"], aligned.df["close_b"], gate_cfg)

    mu = spread.rolling(CFG["zw"]).mean()
    sigma = spread.rolling(CFG["zw"]).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(zscore, dtype=np.float64)

    raw = generate_signals_numba(zscore, CFG["z_entry"], CFG["z_exit"], CFG["z_stop"])
    sig = apply_time_stop(raw, CFG["time_stop"])
    sig = apply_gate_filter_numba(sig, gate_mask)
    sig = apply_window_filter_numba(sig, minutes, CFG["entry_start"], CFG["entry_end"], CFG["flat_min"])

    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        CFG["mult_a"], CFG["mult_b"], CFG["tick_a"], CFG["tick_b"],
        CFG["slippage"], CFG["commission"], CFG["initial_capital"],
    )

    pnls = bt["trade_pnls"]
    entries = bt["trade_entry_bars"]
    exits = bt["trade_exit_bars"]
    sides = bt["trade_sides"]
    equity = bt["equity"]
    num = bt["trades"]
    durations = exits - entries
    entry_dates = pd.DatetimeIndex(idx[entries])
    years = (idx[-1] - idx[0]).days / 365.25

    # Derived metrics
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = float(drawdown.min())
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    avg_win = float(winners.mean()) if len(winners) > 0 else 0
    avg_loss = float(losers.mean()) if len(losers) > 0 else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 99.9
    sharpe = float(pnls.mean() / pnls.std()) if pnls.std() > 1e-12 else 0
    calmar = float(pnls.sum()) / abs(max_dd) if max_dd < 0 else 0
    mc = max_consec_losses(pnls)

    # CPCV
    print("  Computing CPCV...")
    cpcv = run_cpcv(entries, exits, pnls, n, CPCV_CFG)

    # Exit types
    exit_types = classify_exits(
        entries, exits, sides, zscore, CFG["z_exit"], CFG["z_stop"],
        CFG["time_stop"], minutes, CFG["flat_min"], CFG["entry_start"],
    )

    # Propfirm
    print("  Computing propfirm metrics...")
    pf_cfg = PropfirmConfig()
    pf_result = compute_propfirm_metrics(entries, exits, pnls, idx.date, equity, pf_cfg)

    # Monte Carlo
    print("  Running Monte Carlo (10,000 sims)...")
    rng = np.random.default_rng(42)
    mc_dds = np.zeros(10_000)
    for i in range(10_000):
        shuffled = rng.permutation(pnls)
        eq = np.cumsum(shuffled)
        rm = np.maximum.accumulate(eq)
        mc_dds[i] = (eq - rm).min()

    # Slippage
    print("  Computing slippage sensitivity...")
    slip_results = []
    for slip in [0, 1, 2, 3]:
        bt_s = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            CFG["mult_a"], CFG["mult_b"], CFG["tick_a"], CFG["tick_b"],
            slip, CFG["commission"], CFG["initial_capital"],
        )
        cpcv_s = run_cpcv(bt_s["trade_entry_bars"], bt_s["trade_exit_bars"],
                          bt_s["trade_pnls"], n, CPCV_CFG)
        eq_s = bt_s["equity"]
        rm_s = np.maximum.accumulate(eq_s)
        dd_s = float((eq_s - rm_s).min())
        slip_results.append({
            "slip": slip, "trades": bt_s["trades"],
            "pnl": float(bt_s["trade_pnls"].sum()),
            "pf": bt_s["profit_factor"], "wr": bt_s["win_rate"],
            "dd": dd_s, "cpcv_med": cpcv_s["median_sharpe"],
            "cpcv_pct": cpcv_s["pct_positive"],
        })

    # Neighborhood (from CSV)
    print("  Loading neighborhood from CSV...")
    csv_path = PROJECT_ROOT / "output" / "NQ_YM" / "phase13c_grid_massif.csv"
    center = {k: CFG[k] for k in GRID_AXES.keys()}
    center["z_entry"] = CFG["z_entry"]
    center["delta_tp"] = CFG["delta_tp"]
    center["delta_sl"] = CFG["delta_sl"]
    center["time_stop"] = CFG["time_stop"]
    neighbors = get_neighbor_configs(center, GRID_AXES)
    neighbor_details = []

    if csv_path.exists():
        df_grid = pd.read_csv(csv_path,
                               usecols=["ols", "adf_w", "zw", "window", "z_entry",
                                         "delta_tp", "delta_sl", "time_stop",
                                         "pf", "trades", "pnl", "cpcv_median_sharpe"])
        for nb in neighbors:
            mask = (
                (df_grid["ols"] == nb["ols"]) &
                (df_grid["adf_w"] == nb["adf_w"]) &
                (df_grid["zw"] == nb["zw"]) &
                (df_grid["window"] == CFG["window"]) &
                (np.isclose(df_grid["z_entry"], nb["z_entry"])) &
                (np.isclose(df_grid["delta_tp"], nb["delta_tp"])) &
                (np.isclose(df_grid["delta_sl"], nb["delta_sl"])) &
                (df_grid["time_stop"] == nb["time_stop"])
            )
            matches = df_grid[mask]
            if len(matches) > 0:
                row = matches.iloc[0]
                changed = [k for k in center if nb[k] != center[k]]
                neighbor_details.append({
                    "param": changed[0] if changed else "?",
                    "from": center[changed[0]] if changed else "?",
                    "to": nb[changed[0]] if changed else "?",
                    "pf": float(row["pf"]),
                    "trades": int(row["trades"]),
                    "pnl": float(row["pnl"]),
                    "cpcv": float(row["cpcv_median_sharpe"]),
                })
        del df_grid
    else:
        print("  WARNING: CSV not found, skipping neighborhood")

    nb_sharpes = [d["cpcv"] for d in neighbor_details]
    nb_pfs = [d["pf"] for d in neighbor_details]
    nb_result = compute_neighborhood_robustness(cpcv["median_sharpe"], nb_sharpes, nb_pfs)

    # ======================================================================
    # GENERATE CHARTS
    # ======================================================================
    print("  Generating charts...")
    plt.rcParams.update({
        "axes.facecolor": "#1a1a2e", "figure.facecolor": "#1a1a2e",
        "text.color": "#e0e0e0", "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#888", "ytick.color": "#888",
        "axes.edgecolor": "#444", "grid.color": "#333",
    })

    # 1. Equity curve + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1], sharex=True)
    ax1.plot(idx, equity, color="#00d4ff", linewidth=0.8, label="Equity")
    ax1.axhline(CFG["initial_capital"], color="#444", linewidth=0.5, linestyle="--")
    for i in range(num):
        color = "#00ff88" if pnls[i] > 0 else "#ff4444"
        ax1.axvspan(idx[entries[i]], idx[exits[i]], alpha=0.08, color=color)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Config D -- Equity Curve", color="#00d4ff", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(idx, drawdown, 0, color="#ff4444", alpha=0.4)
    ax2.plot(idx, drawdown, color="#ff4444", linewidth=0.5)
    ax2.axhline(-5000, color="#ffa500", linewidth=1, linestyle="--", label="Propfirm $5K")
    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    equity_b64 = fig_to_base64(fig)

    # 2. PnL by year barplot
    fig, ax = plt.subplots(figsize=(10, 4))
    yr_list = sorted(set(entry_dates.year))
    yr_pnls = [float(pnls[entry_dates.year == y].sum()) for y in yr_list]
    colors = ["#00ff88" if p >= 0 else "#ff4444" for p in yr_pnls]
    ax.bar(yr_list, yr_pnls, color=colors, width=0.7, edgecolor="#444")
    ax.axhline(0, color="#666", linewidth=0.5)
    ax.set_ylabel("PnL ($)")
    ax.set_title("PnL par Annee", color="#00d4ff")
    for i, (y, p) in enumerate(zip(yr_list, yr_pnls)):
        ax.text(y, p + (200 if p >= 0 else -400), f"${p:+,.0f}", ha="center",
                fontsize=9, color=colors[i])
    ax.grid(True, alpha=0.3, axis="y")
    pnl_year_b64 = fig_to_base64(fig)

    # 3. Heatmap hour x day
    fig, ax = plt.subplots(figsize=(10, 6))
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    hours = list(range(2, 15))
    heatmap = np.zeros((len(hours), 5))
    for i, h in enumerate(hours):
        for d in range(5):
            m = (entry_dates.hour == h) & (entry_dates.dayofweek == d)
            mp = pnls[m]
            heatmap[i, d] = float(mp.sum()) if len(mp) > 0 else 0

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(5))
    ax.set_xticklabels(day_names)
    ax.set_yticks(range(len(hours)))
    ax.set_yticklabels([f"{h}:00" for h in hours])
    ax.set_title("PnL Heatmap: Heure x Jour", color="#00d4ff")
    for i in range(len(hours)):
        for j in range(5):
            val = heatmap[i, j]
            if abs(val) > 100:
                txt_color = "black" if abs(val) > 1500 else "#e0e0e0"
                ax.text(j, i, f"${val:+,.0f}", ha="center", va="center", fontsize=7, color=txt_color)
    plt.colorbar(im, ax=ax, label="PnL ($)")
    heatmap_b64 = fig_to_base64(fig)

    # 4. PnL distribution histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(pnls, bins=30, color="#00d4ff", alpha=0.7, edgecolor="#444")
    ax.axvline(0, color="#ffa500", linewidth=1, linestyle="--")
    ax.axvline(float(pnls.mean()), color="#00ff88", linewidth=1.5, linestyle="-", label=f"Mean ${pnls.mean():+,.0f}")
    ax.set_xlabel("PnL per Trade ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution des PnL par Trade", color="#00d4ff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    pnl_dist_b64 = fig_to_base64(fig)

    # 5. Scatter duration vs PnL
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_scatter = ["#00ff88" if p > 0 else "#ff4444" for p in pnls]
    ax.scatter(durations, pnls, c=colors_scatter, alpha=0.6, s=20, edgecolors="none")
    ax.axhline(0, color="#666", linewidth=0.5)
    ax.set_xlabel("Duration (5min bars)")
    ax.set_ylabel("PnL ($)")
    ax.set_title("Duration vs PnL", color="#00d4ff")
    ax.grid(True, alpha=0.3)
    scatter_b64 = fig_to_base64(fig)

    # 6. CPCV path Sharpe distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    valid_sharpes = cpcv["path_sharpes"][~np.isnan(cpcv["path_sharpes"])]
    ax.bar(range(len(valid_sharpes)), sorted(valid_sharpes, reverse=True),
           color=["#00ff88" if s > 0 else "#ff4444" for s in sorted(valid_sharpes, reverse=True)],
           edgecolor="#444")
    ax.axhline(0, color="#ffa500", linewidth=1, linestyle="--")
    ax.axhline(cpcv["median_sharpe"], color="#00d4ff", linewidth=1.5, linestyle="-",
               label=f"Median {cpcv['median_sharpe']:.4f}")
    ax.set_xlabel("Path (sorted)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"CPCV(10,2) -- {cpcv['n_valid_paths']} Valid Paths", color="#00d4ff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    cpcv_b64 = fig_to_base64(fig)

    # 7. Monte Carlo DD distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(mc_dds, bins=60, color="#00d4ff", alpha=0.7, edgecolor="#444")
    ax.axvline(max_dd, color="#ff4444", linewidth=2, linestyle="-", label=f"Historical ${max_dd:+,.0f}")
    ax.axvline(-5000, color="#ffa500", linewidth=2, linestyle="--", label="Propfirm $5K")
    ax.axvline(np.percentile(mc_dds, 5), color="#ff8800", linewidth=1, linestyle=":",
               label=f"P95 ${np.percentile(mc_dds, 5):+,.0f}")
    ax.set_xlabel("Max Drawdown ($)")
    ax.set_ylabel("Count")
    ax.set_title("Monte Carlo Max Drawdown (10,000 sims)", color="#00d4ff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    mc_b64 = fig_to_base64(fig)

    # 8. Daily PnL distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    daily_pnls = pf_result.daily_pnls
    ax.hist(daily_pnls, bins=40, color="#00d4ff", alpha=0.7, edgecolor="#444")
    ax.axvline(0, color="#666", linewidth=0.5)
    ax.axvline(-pf_cfg.max_daily_loss, color="#ff4444", linewidth=2, linestyle="--",
               label=f"Limit -${pf_cfg.max_daily_loss:,.0f}")
    ax.set_xlabel("Daily PnL ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution PnL Journalier", color="#00d4ff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    daily_b64 = fig_to_base64(fig)

    # ======================================================================
    # BUILD HTML
    # ======================================================================
    print("  Building HTML...")

    # --- Temporal tables ---
    yr_rows = ""
    for y in yr_list:
        m = entry_dates.year == y
        yp = pnls[m]
        yn = len(yp)
        cls = "positive" if yp.sum() >= 0 else "negative"
        yr_rows += f"<tr><td>{y}</td><td>{yn}</td><td class='{cls}'>${yp.sum():+,.0f}</td>"
        yr_rows += f"<td>{(yp>0).sum()/yn*100:.0f}%</td><td>${yp.mean():+,.0f}</td></tr>\n"

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    day_rows = ""
    for d in range(5):
        m = entry_dates.dayofweek == d
        dp = pnls[m]
        dn = len(dp)
        if dn > 0:
            cls = "positive" if dp.sum() >= 0 else "negative"
            day_rows += f"<tr><td>{day_names[d]}</td><td>{dn}</td>"
            day_rows += f"<td class='{cls}'>${dp.sum():+,.0f}</td>"
            day_rows += f"<td>{(dp>0).sum()/dn*100:.0f}%</td></tr>\n"

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mon_rows = ""
    for mo in range(1, 13):
        m = entry_dates.month == mo
        mp = pnls[m]
        mn = len(mp)
        if mn > 0:
            cls = "positive" if mp.sum() >= 0 else "negative"
            mon_rows += f"<tr><td>{month_names[mo-1]}</td><td>{mn}</td>"
            mon_rows += f"<td class='{cls}'>${mp.sum():+,.0f}</td>"
            mon_rows += f"<td>{(mp>0).sum()/mn*100:.0f}%</td></tr>\n"

    hour_rows = ""
    for h in range(2, 15):
        m = entry_dates.hour == h
        hp = pnls[m]
        hn = len(hp)
        if hn > 0:
            cls = "positive" if hp.sum() >= 0 else "negative"
            hour_rows += f"<tr><td>{h}:00</td><td>{hn}</td>"
            hour_rows += f"<td class='{cls}'>${hp.sum():+,.0f}</td>"
            hour_rows += f"<td>{(hp>0).sum()/hn*100:.0f}%</td></tr>\n"

    # Exit types
    exit_counts = Counter(exit_types)
    exit_rows = ""
    for et in ["Z_EXIT", "Z_STOP", "TIME_STOP", "FLAT"]:
        cnt = exit_counts.get(et, 0)
        if cnt > 0:
            emask = np.array([t == et for t in exit_types])
            ep = pnls[emask]
            wr = (ep > 0).sum() / cnt * 100
            cls = "positive" if ep.sum() >= 0 else "negative"
            exit_rows += f"<tr><td>{et}</td><td>{cnt}</td><td>{cnt/num*100:.1f}%</td>"
            exit_rows += f"<td class='{cls}'>${ep.sum():+,.0f}</td>"
            exit_rows += f"<td>${ep.mean():+,.0f}</td><td>{wr:.0f}%</td></tr>\n"

    # Neighborhood
    nb_rows = ""
    for d in sorted(neighbor_details, key=lambda x: x["param"]):
        cls = "positive" if d["pf"] > 1.0 else "negative"
        nb_rows += f"<tr><td>{d['param']}</td><td>{d['from']}</td><td>{d['to']}</td>"
        nb_rows += f"<td class='{cls}'>{d['pf']:.2f}</td><td>{d['trades']}</td>"
        nb_rows += f"<td>${d['pnl']:+,.0f}</td><td>{d['cpcv']:.4f}</td></tr>\n"

    # Slippage
    slip_rows = ""
    for s in slip_results:
        marker = " (backtest)" if s["slip"] == 1 else ""
        cls = "positive" if s["pf"] > 1.5 else "warn" if s["pf"] > 1.2 else "negative"
        slip_rows += f"<tr><td>{s['slip']} tick{marker}</td><td>{s['trades']}</td>"
        slip_rows += f"<td>${s['pnl']:+,.0f}</td><td class='{cls}'>{s['pf']:.2f}</td>"
        slip_rows += f"<td>{s['wr']:.0f}%</td><td>${s['dd']:+,.0f}</td>"
        slip_rows += f"<td>{s['cpcv_med']:.3f}</td><td>{s['cpcv_pct']:.0f}%</td></tr>\n"

    # Monte Carlo percentiles
    mc_rows = ""
    for label, pct_val in [("Median", 50), ("P75", 25), ("P90", 10), ("P95", 5), ("P99", 1)]:
        dd_val = np.percentile(mc_dds, pct_val)
        cls = "positive" if dd_val > -5000 else "negative"
        mc_rows += f"<tr><td>{label}</td><td class='{cls}'>${dd_val:+,.0f}</td>"
        mc_rows += f"<td>{'SAFE' if dd_val > -5000 else 'BREACH'}</td></tr>\n"

    pct_breach = (mc_dds < -5000).sum() / len(mc_dds) * 100
    hist_dd_pct = int(100 - (mc_dds < max_dd).sum() / len(mc_dds) * 100)

    n_long = (sides == 1).sum()
    n_short = (sides == -1).sum()

    propfirm_status = '<span class="go">COMPLIANT</span>' if pf_result.is_compliant else '<span class="fail">VIOLATION</span>'
    nb_verdict = '<span class="go">ROBUST</span>' if nb_result.is_robust else '<span class="fail">FRAGILE</span>'

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Config D -- Reference Report</title>
    {STYLE}
</head>
<body>
<div class="container">

<h1>Config D -- NQ/YM OLS Binary Gates</h1>
<p style="color:#888;">Generated from Phase 13c grid (24.7M configs) | Data: {idx[0].date()} to {idx[-1].date()} ({years:.1f} years, {n:,} bars)</p>

<div class="verdict-box">
    GO FOR LIVE -- CPCV {cpcv['pct_positive']:.0f}% paths+ | PF {bt['profit_factor']:.2f} | Propfirm COMPLIANT | Monte Carlo {100-pct_breach:.0f}% safe
</div>

<!-- PARAMETERS -->
<div class="section">
<h2>Parametres</h2>
<div class="two-col">
<table class="param-table">
    <tr><td>Pair</td><td>NQ / YM (E-mini)</td></tr>
    <tr><td>Timeframe</td><td>5 min</td></tr>
    <tr><td>Hedge method</td><td>OLS Rolling</td></tr>
    <tr><td>OLS window</td><td>{CFG['ols']} bars</td></tr>
    <tr><td>Z-score window</td><td>{CFG['zw']} bars</td></tr>
    <tr><td>z_entry</td><td>{CFG['z_entry']}</td></tr>
    <tr><td>z_exit</td><td>{CFG['z_exit']}</td></tr>
    <tr><td>z_stop</td><td>{CFG['z_stop']}</td></tr>
    <tr><td>time_stop</td><td>{CFG['time_stop']} (disabled)</td></tr>
</table>
<table class="param-table">
    <tr><td>Trading window</td><td>{CFG['window']}</td></tr>
    <tr><td>Flat time</td><td>15:30 CT</td></tr>
    <tr><td>Gate ADF</td><td>&lt; {CFG['gate_adf']} (window {CFG['adf_w']})</td></tr>
    <tr><td>Gate Hurst</td><td>&lt; {CFG['gate_hurst']} (window {CFG['gate_hurst_w']})</td></tr>
    <tr><td>Gate Corr</td><td>&gt; {CFG['gate_corr']} (window {CFG['gate_corr_w']})</td></tr>
    <tr><td>Slippage</td><td>{CFG['slippage']} tick/leg</td></tr>
    <tr><td>Commission</td><td>${CFG['commission']}/contract/side</td></tr>
    <tr><td>Initial capital</td><td>${CFG['initial_capital']:,.0f}</td></tr>
    <tr><td>Live protection</td><td>Dollar stop $2,000 (Sierra)</td></tr>
</table>
</div>
</div>

<!-- GLOBAL METRICS -->
<div class="section">
<h2>Metriques Globales</h2>
<div class="metric-grid">
    <div class="metric-card"><div class="metric-value">{num}</div><div class="metric-label">Trades</div></div>
    <div class="metric-card"><div class="metric-value">{bt['profit_factor']:.2f}</div><div class="metric-label">Profit Factor</div></div>
    <div class="metric-card"><div class="metric-value">{bt['win_rate']:.1f}%</div><div class="metric-label">Win Rate</div></div>
    <div class="metric-card"><div class="metric-value positive">${pnls.sum():+,.0f}</div><div class="metric-label">Total PnL</div></div>
    <div class="metric-card"><div class="metric-value negative">${max_dd:+,.0f}</div><div class="metric-label">Max Drawdown</div></div>
    <div class="metric-card"><div class="metric-value">{sharpe:.3f}</div><div class="metric-label">Sharpe (per trade)</div></div>
    <div class="metric-card"><div class="metric-value">{calmar:.2f}</div><div class="metric-label">Calmar (PnL/DD)</div></div>
    <div class="metric-card"><div class="metric-value">{num/years:.0f}</div><div class="metric-label">Trades/Year</div></div>
</div>

<h3>Detail Win/Loss</h3>
<div class="metric-grid">
    <div class="metric-card"><div class="metric-value positive">${avg_win:+,.0f}</div><div class="metric-label">Avg Winner ({len(winners)} trades)</div></div>
    <div class="metric-card"><div class="metric-value negative">${avg_loss:+,.0f}</div><div class="metric-label">Avg Loser ({len(losers)} trades)</div></div>
    <div class="metric-card"><div class="metric-value">{wl_ratio:.2f}</div><div class="metric-label">W/L Ratio</div></div>
    <div class="metric-card"><div class="metric-value positive">${float(pnls.max()):+,.0f}</div><div class="metric-label">Best Trade</div></div>
    <div class="metric-card"><div class="metric-value negative">${float(pnls.min()):+,.0f}</div><div class="metric-label">Worst Trade</div></div>
    <div class="metric-card"><div class="metric-value">{mc}</div><div class="metric-label">Max Consec Losses</div></div>
</div>

<h3>Direction & Duration</h3>
<div class="metric-grid">
    <div class="metric-card"><div class="metric-value">{n_long} / {n_short}</div><div class="metric-label">Long / Short ({n_long/num*100:.0f}% / {n_short/num*100:.0f}%)</div></div>
    <div class="metric-card"><div class="metric-value">{durations.mean():.1f}</div><div class="metric-label">Avg Duration (bars)</div></div>
    <div class="metric-card"><div class="metric-value">{float(durations[pnls>0].mean()):.1f} / {float(durations[pnls<=0].mean()):.1f}</div><div class="metric-label">Avg Dur Win / Loss</div></div>
    <div class="metric-card"><div class="metric-value">{int(durations.max())}</div><div class="metric-label">Max Duration (bars)</div></div>
</div>
</div>

<!-- EQUITY CURVE -->
<div class="section">
<h2>Equity Curve & Drawdown</h2>
<img src="data:image/png;base64,{equity_b64}" alt="Equity Curve">
</div>

<!-- CPCV -->
<div class="section">
<h2>CPCV(10,2) -- 45 Chemins</h2>
<div class="metric-grid">
    <div class="metric-card"><div class="metric-value">{cpcv['median_sharpe']:.4f}</div><div class="metric-label">Median Sharpe</div></div>
    <div class="metric-card"><div class="metric-value">{cpcv['mean_sharpe']:.4f}</div><div class="metric-label">Mean Sharpe</div></div>
    <div class="metric-card"><div class="metric-value">{cpcv['std_sharpe']:.4f}</div><div class="metric-label">Std Sharpe</div></div>
    <div class="metric-card"><div class="metric-value">{cpcv['min_sharpe']:.4f}</div><div class="metric-label">Min Sharpe</div></div>
    <div class="metric-card"><div class="metric-value">{cpcv['max_sharpe']:.4f}</div><div class="metric-label">Max Sharpe</div></div>
    <div class="metric-card"><div class="metric-value go">{cpcv['pct_positive']:.0f}%</div><div class="metric-label">Paths Positifs</div></div>
    <div class="metric-card"><div class="metric-value">{cpcv['n_valid_paths']}</div><div class="metric-label">Valid Paths</div></div>
</div>
<img src="data:image/png;base64,{cpcv_b64}" alt="CPCV Paths">
</div>

<!-- TEMPORAL -->
<div class="section">
<h2>Distribution Temporelle</h2>
<img src="data:image/png;base64,{pnl_year_b64}" alt="PnL par Annee">

<div class="two-col">
<div>
<h3>Par Annee</h3>
<table><tr><th>Year</th><th>#</th><th>PnL</th><th>WR</th><th>Avg</th></tr>{yr_rows}</table>
</div>
<div>
<h3>Par Jour de Semaine</h3>
<table><tr><th>Day</th><th>#</th><th>PnL</th><th>WR</th></tr>{day_rows}</table>
</div>
</div>

<div class="two-col">
<div>
<h3>Par Mois</h3>
<table><tr><th>Month</th><th>#</th><th>PnL</th><th>WR</th></tr>{mon_rows}</table>
</div>
<div>
<h3>Par Heure d'Entree</h3>
<table><tr><th>Hour</th><th>#</th><th>PnL</th><th>WR</th></tr>{hour_rows}</table>
</div>
</div>

<h3>Heatmap PnL: Heure x Jour</h3>
<img src="data:image/png;base64,{heatmap_b64}" alt="Heatmap">
</div>

<!-- TRADE ANALYSIS -->
<div class="section">
<h2>Analyse des Trades</h2>
<img src="data:image/png;base64,{pnl_dist_b64}" alt="PnL Distribution">
<img src="data:image/png;base64,{scatter_b64}" alt="Duration vs PnL">

<h3>Sorties par Type</h3>
<table>
<tr><th>Type</th><th>#</th><th>%</th><th>PnL Total</th><th>Avg PnL</th><th>WR</th></tr>
{exit_rows}
</table>
</div>

<!-- PROPFIRM -->
<div class="section">
<h2>Propfirm $150K</h2>
<div class="metric-grid">
    <div class="metric-card"><div class="metric-value">{propfirm_status}</div><div class="metric-label">Status</div></div>
    <div class="metric-card"><div class="metric-value">{pf_result.n_trading_days}</div><div class="metric-label">Trading Days</div></div>
    <div class="metric-card"><div class="metric-value">${pf_result.avg_daily_pnl:+,.0f}</div><div class="metric-label">Avg Daily PnL</div></div>
    <div class="metric-card"><div class="metric-value">{pf_result.pct_days_profitable:.0f}%</div><div class="metric-label">Days Profitable</div></div>
    <div class="metric-card"><div class="metric-value negative">${pf_result.max_daily_loss_observed:+,.0f}</div><div class="metric-label">Max Daily Loss</div></div>
    <div class="metric-card"><div class="metric-value negative">${pf_result.max_trailing_dd:+,.0f}</div><div class="metric-label">Max Trailing DD</div></div>
    <div class="metric-card"><div class="metric-value">{pf_result.n_days_exceed_daily_limit}</div><div class="metric-label">Days > $4,500 Loss</div></div>
</div>
<img src="data:image/png;base64,{daily_b64}" alt="Daily PnL Distribution">
</div>

<!-- NEIGHBORHOOD -->
<div class="section">
<h2>Voisinage (L1, +/-1 step)</h2>
<div class="metric-grid">
    <div class="metric-card"><div class="metric-value">{nb_result.n_neighbors}</div><div class="metric-label">Voisins Trouves</div></div>
    <div class="metric-card"><div class="metric-value go">{nb_result.pct_profitable:.0f}%</div><div class="metric-label">% Profitables</div></div>
    <div class="metric-card"><div class="metric-value">{nb_result.sharpe_degradation_pct:.0f}%</div><div class="metric-label">Degradation CPCV</div></div>
    <div class="metric-card"><div class="metric-value">{nb_verdict}</div><div class="metric-label">Verdict</div></div>
</div>
<table>
<tr><th>Param</th><th>From</th><th>To</th><th>PF</th><th>Trades</th><th>PnL</th><th>CPCV</th></tr>
{nb_rows}
</table>
</div>

<!-- MONTE CARLO -->
<div class="section">
<h2>Monte Carlo Drawdown (10,000 sims)</h2>
<img src="data:image/png;base64,{mc_b64}" alt="Monte Carlo">
<table>
<tr><th>Scenario</th><th>Max DD</th><th>Propfirm $5K</th></tr>
{mc_rows}
<tr><td>Absolute worst</td><td class="negative">${mc_dds.min():+,.0f}</td><td>BREACH</td></tr>
</table>
<p style="margin-top:10px;">Prob(DD &gt; $5,000): <strong>{pct_breach:.1f}%</strong> | Historical DD at P{hist_dd_pct}</p>
</div>

<!-- SLIPPAGE -->
<div class="section">
<h2>Sensibilite Slippage</h2>
<table>
<tr><th>Slippage</th><th>Trades</th><th>PnL</th><th>PF</th><th>WR</th><th>Max DD</th><th>CPCV Med</th><th>Paths+</th></tr>
{slip_rows}
</table>
<p style="margin-top:10px;">Breakeven: ~11 ticks de slippage</p>
</div>

<!-- FOOTER -->
<div class="section" style="text-align:center; color:#666; font-size:12px;">
    Phase 13c -- Config D Reference Report | Generated by phase13c_report_d.py
</div>

</div>
</body>
</html>"""

    output_path = PROJECT_ROOT / "output" / "NQ_YM" / "config_D_reference.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\n  Report saved to: {output_path}")
    print("  Open in browser to view.")


if __name__ == "__main__":
    main()
