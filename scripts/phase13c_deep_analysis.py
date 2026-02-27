"""Phase 13c Deep Analysis: Overlay + Autopsy + WF Folds.

Three sections + decision matrix:
1. Overlay temporal A vs D vs T1 (overlap, trades uniques, periodes)
2. Autopsie trades (exit types, direction, concentration DD, calendarite)
3. WF folds mapped to calendar dates + recency score
4. Decision matrix (thresholds fixed ex ante)

Usage:
    python scripts/phase13c_deep_analysis.py
"""

import sys
from collections import Counter
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
from src.validation.gates import GateConfig, apply_gate_filter_numba, compute_gate_mask

_NQ, _YM = get_pair_specs("NQ", "YM")
MULT_A, MULT_B = _NQ.multiplier, _YM.multiplier
TICK_A, TICK_B = _NQ.tick_size, _YM.tick_size
SLIPPAGE = DEFAULT_SLIPPAGE_TICKS
COMMISSION = _NQ.commission
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930

GATE_ADF_THRESH = -2.86
GATE_HURST_THRESH = 0.50
GATE_CORR_THRESH = 0.70
GATE_HURST_WINDOW = 64
GATE_CORR_WINDOW = 24

WINDOW_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
}

WF_IS_BARS = 2 * 252 * 78    # ~2 years
WF_OOS_BARS = 126 * 78       # ~6 months
WF_STEP_BARS = 126 * 78      # ~6 months

# Only viable candidates (B: propfirm violation, C: dead after mid-2022)
CANDIDATES = {
    "A": {"ols": 6000, "adf_w": 30, "zw": 28, "window": "04:00-14:00",
           "z_entry": 3.25, "delta_tp": 2.75, "delta_sl": 1.25, "time_stop": 12},
    "D": {"ols": 7000, "adf_w": 96, "zw": 30, "window": "02:00-14:00",
           "z_entry": 3.25, "delta_tp": 2.75, "delta_sl": 1.50, "time_stop": 0},
    "T1": {"ols": 7000, "adf_w": 96, "zw": 15, "window": "04:00-14:00",
            "z_entry": 2.25, "delta_tp": 1.75, "delta_sl": 1.25, "time_stop": 0},
}

# Decision thresholds (fixed BEFORE seeing results)
THRESHOLDS = {
    "overlap_doublon": 60,     # > 60% overlap = doublon
    "overlap_partial": 30,     # 30-60% = partial
    "worst_trade_kill": -3500,
    "worst_trade_warn": -2000,
    "max_consec_kill": 7,
    "max_consec_warn": 5,
    "win_loss_kill": 1.0,
    "win_loss_warn": 1.5,
    "recency_go": 50,          # >= 50% of last 6 folds
    "recency_warn": 33,        # 33-49%
}


# ======================================================================
# Helpers
# ======================================================================

def derive_params(cfg):
    z_exit = round(max(cfg["z_entry"] - cfg["delta_tp"], 0.0), 4)
    z_stop = round(cfg["z_entry"] + cfg["delta_sl"], 4)
    return z_exit, z_stop


def cfg_label(name, cfg):
    z_exit, z_stop = derive_params(cfg)
    return (f"{name}: OLS={cfg['ols']} ADF_w={cfg['adf_w']} ZW={cfg['zw']} "
            f"{cfg['window']} ze={cfg['z_entry']} zx={z_exit} zs={z_stop} ts={cfg['time_stop']}")


def reconstruct_full(cfg, aligned, px_a, px_b, idx, minutes):
    """Run backtest and return all intermediate data for analysis."""
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

    return {
        "bt": bt, "zscore": zscore, "beta": beta,
        "z_exit": z_exit, "z_stop": z_stop,
        "entry_start": entry_start,
    }


def classify_exits(entries, exits, sides, zscore, z_exit, z_stop,
                    time_stop, minutes, flat_min, entry_start_min):
    """Classify each trade exit: Z_EXIT, Z_STOP, TIME_STOP, FLAT."""
    types = []
    for i in range(len(entries)):
        eb, xb = entries[i], exits[i]
        m = minutes[xb]

        # Priority 1: FLAT (EOD or before entry window)
        if m >= flat_min or m < entry_start_min:
            types.append("FLAT")
            continue

        # Priority 2: TIME_STOP (duration matches time_stop exactly)
        dur = xb - eb
        if time_stop > 0 and dur >= time_stop:
            types.append("TIME_STOP")
            continue

        # Priority 3: Z_STOP (state machine hit stop)
        z = zscore[xb] if xb < len(zscore) else np.nan
        if not np.isnan(z):
            if sides[i] == 1 and z < -z_stop:
                types.append("Z_STOP")
                continue
            if sides[i] == -1 and z > z_stop:
                types.append("Z_STOP")
                continue

        # Default: Z_EXIT (natural mean reversion)
        types.append("Z_EXIT")

    return types


def max_consecutive_losses(pnls):
    """Count maximum consecutive losing trades."""
    max_c = 0
    cur = 0
    for p in pnls:
        if p <= 0:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


# ======================================================================
# SECTION 1: Overlay Temporal
# ======================================================================

def run_overlay(all_data, idx):
    print("\n" + "=" * 100)
    print(" SECTION 1: OVERLAY TEMPOREL")
    print("=" * 100)

    names = list(all_data.keys())

    # Pairwise overlap analysis
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            bt1 = all_data[n1]["bt"]
            bt2 = all_data[n2]["bt"]
            e1, x1 = bt1["trade_entry_bars"], bt1["trade_exit_bars"]
            e2, x2 = bt2["trade_entry_bars"], bt2["trade_exit_bars"]
            s1, s2 = bt1["trade_sides"], bt2["trade_sides"]
            p1, p2 = bt1["trade_pnls"], bt2["trade_pnls"]

            # Find overlapping trades (intervals intersect)
            ol_1 = set()
            ol_2 = set()
            same_dir = 0
            for i1 in range(len(e1)):
                for i2 in range(len(e2)):
                    if e1[i1] < x2[i2] and e2[i2] < x1[i1]:
                        if i1 not in ol_1:
                            ol_1.add(i1)
                            if s1[i1] == s2[i2]:
                                same_dir += 1
                        ol_2.add(i2)

            n_ol1 = len(ol_1)
            n_ol2 = len(ol_2)
            pct1 = n_ol1 / len(e1) * 100 if len(e1) else 0
            pct2 = n_ol2 / len(e2) * 100 if len(e2) else 0
            pct_avg = (pct1 + pct2) / 2

            status = ("DOUBLON" if pct_avg > THRESHOLDS["overlap_doublon"]
                      else "PARTIAL" if pct_avg > THRESHOLDS["overlap_partial"]
                      else "COMPLEMENT")

            # PnL of unique trades
            unique_pnl_1 = sum(p1[j] for j in range(len(e1)) if j not in ol_1)
            unique_pnl_2 = sum(p2[j] for j in range(len(e2)) if j not in ol_2)

            print(f"\n  {n1} vs {n2}:  [{status}]  (avg overlap {pct_avg:.0f}%)")
            print(f"    {n1}: {len(e1):>4} trades | {n_ol1:>3} overlap ({pct1:.0f}%) | "
                  f"{len(e1)-n_ol1:>3} unique (PnL ${unique_pnl_1:>+,.0f})")
            print(f"    {n2}: {len(e2):>4} trades | {n_ol2:>3} overlap ({pct2:.0f}%) | "
                  f"{len(e2)-n_ol2:>3} unique (PnL ${unique_pnl_2:>+,.0f})")
            print(f"    Same direction in overlap: {same_dir}/{n_ol1}")

    # Activity by year
    print("\n  --- Trades by Year ---")
    print(f"  {'Year':>6}", end="")
    for name in names:
        print(f"  {name+' #':>7} {name+' $':>10}", end="")
    print()

    years = sorted(set(idx.year))
    for year in years:
        print(f"  {year:>6}", end="")
        for name in names:
            bt = all_data[name]["bt"]
            e = bt["trade_entry_bars"]
            p = bt["trade_pnls"]
            mask = pd.DatetimeIndex(idx[e]).year == year
            print(f"  {mask.sum():>7} ${p[mask].sum():>+9,.0f}", end="")
        print()

    # Activity by hour
    print("\n  --- Trades by Entry Hour ---")
    print(f"  {'Hour':>6}", end="")
    for name in names:
        print(f"  {name:>6}", end="")
    print()
    for hour in range(2, 16):
        has_any = False
        vals = []
        for name in names:
            bt = all_data[name]["bt"]
            e = bt["trade_entry_bars"]
            n_h = (idx[e].hour == hour).sum()
            vals.append(n_h)
            if n_h > 0:
                has_any = True
        if has_any:
            print(f"  {hour:>4}:00", end="")
            for v in vals:
                print(f"  {v:>6}", end="")
            print()


# ======================================================================
# SECTION 2: Autopsie des Trades
# ======================================================================

def run_autopsy(all_data, idx, minutes):
    print("\n" + "=" * 100)
    print(" SECTION 2: AUTOPSIE DES TRADES")
    print("=" * 100)

    for name, data in all_data.items():
        cfg = CANDIDATES[name]
        bt = data["bt"]
        entries = bt["trade_entry_bars"]
        exits = bt["trade_exit_bars"]
        sides = bt["trade_sides"]
        pnls = bt["trade_pnls"]
        n_trades = bt["trades"]
        z_exit_val = data["z_exit"]
        z_stop_val = data["z_stop"]

        print(f"\n  {'=' * 85}")
        print(f"  {cfg_label(name, cfg)}")
        print(f"  {n_trades} trades | PnL ${pnls.sum():,.0f} | PF {bt['profit_factor']:.2f} | "
              f"WR {bt['win_rate']:.1f}%")
        print(f"  {'=' * 85}")

        # --- Direction ---
        (sides == 1).sum()
        (sides == -1).sum()
        for label, mask_val in [("LONG", 1), ("SHORT", -1)]:
            m = sides == mask_val
            cnt = m.sum()
            if cnt == 0:
                continue
            mpnl = pnls[m]
            wr = (mpnl > 0).sum() / cnt * 100
            print(f"\n    {label:>5}: {cnt:>4} ({cnt/n_trades*100:.0f}%)  "
                  f"PnL ${mpnl.sum():>+,.0f}  WR {wr:.0f}%  "
                  f"Avg ${mpnl.mean():>+,.0f}")

        # --- Exit Types ---
        exit_types = classify_exits(
            entries, exits, sides, data["zscore"],
            z_exit_val, z_stop_val, cfg["time_stop"],
            minutes, FLAT_MIN, data["entry_start"],
        )
        exit_counts = Counter(exit_types)

        print("\n    Exit Types:")
        for etype in ["Z_EXIT", "Z_STOP", "TIME_STOP", "FLAT"]:
            cnt = exit_counts.get(etype, 0)
            if cnt == 0:
                continue
            emask = np.array([et == etype for et in exit_types])
            epnl = pnls[emask]
            wr = (epnl > 0).sum() / cnt * 100
            print(f"      {etype:>10}: {cnt:>4} ({cnt/n_trades*100:>5.1f}%)  "
                  f"PnL ${epnl.sum():>+9,.0f}  WR {wr:.0f}%  "
                  f"Avg ${epnl.mean():>+,.0f}")

        # --- Win/Loss ---
        winners = pnls[pnls > 0]
        losers = pnls[pnls <= 0]
        avg_win = float(winners.mean()) if len(winners) > 0 else 0
        avg_loss = float(losers.mean()) if len(losers) > 0 else 0
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        print("\n    Win/Loss:")
        print(f"      Avg winner:    ${avg_win:>+9,.0f}  ({len(winners)} trades)")
        print(f"      Avg loser:     ${avg_loss:>+9,.0f}  ({len(losers)} trades)")
        print(f"      W/L ratio:     {wl_ratio:.2f}")

        # --- Duration ---
        durations = exits - entries
        dur_win = durations[pnls > 0]
        dur_loss = durations[pnls <= 0]

        print("\n    Duration (5min bars):")
        print(f"      All:     avg={durations.mean():.1f}  "
              f"median={np.median(durations):.0f}  max={durations.max()}")
        if len(dur_win) > 0:
            print(f"      Winners: avg={dur_win.mean():.1f}  "
                  f"median={np.median(dur_win):.0f}")
        if len(dur_loss) > 0:
            print(f"      Losers:  avg={dur_loss.mean():.1f}  "
                  f"median={np.median(dur_loss):.0f}")

        # --- Worst trades & DD concentration ---
        sorted_pnl = np.sort(pnls)
        total_losses = float(sorted_pnl[sorted_pnl < 0].sum())
        max_consec = max_consecutive_losses(pnls)

        print("\n    Drawdown Concentration:")
        print(f"      Worst trade:       ${sorted_pnl[0]:>+9,.0f}")
        if len(sorted_pnl) >= 3:
            w3 = float(sorted_pnl[:3].sum())
            pct3 = w3 / total_losses * 100 if total_losses < 0 else 0
            print(f"      Worst 3 trades:    ${w3:>+9,.0f}  "
                  f"({pct3:.0f}% of total losses)")
        if len(sorted_pnl) >= 5:
            w5 = float(sorted_pnl[:5].sum())
            pct5 = w5 / total_losses * 100 if total_losses < 0 else 0
            print(f"      Worst 5 trades:    ${w5:>+9,.0f}  "
                  f"({pct5:.0f}% of total losses)")
        print(f"      Total losses:      ${total_losses:>+9,.0f}")
        print(f"      Max consec losses: {max_consec}")

        # --- Distribution by Year ---
        entry_dates = pd.DatetimeIndex(idx[entries])

        print("\n    By Year:")
        print(f"    {'Year':>6} {'#':>5} {'PnL':>10} {'WR':>5} {'Avg':>8}")
        for year in sorted(set(entry_dates.year)):
            m = entry_dates.year == year
            yp = pnls[m]
            yn = len(yp)
            print(f"    {year:>6} {yn:>5} ${yp.sum():>+9,.0f} "
                  f"{(yp>0).sum()/yn*100:>4.0f}% ${yp.mean():>+7,.0f}")

        # --- Distribution by Day of Week ---
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        print("\n    By Day of Week:")
        print(f"    {'Day':>5} {'#':>5} {'PnL':>10} {'WR':>5}")
        for d in range(5):
            m = entry_dates.dayofweek == d
            dp = pnls[m]
            dn = len(dp)
            if dn > 0:
                print(f"    {day_names[d]:>5} {dn:>5} ${dp.sum():>+9,.0f} "
                      f"{(dp>0).sum()/dn*100:>4.0f}%")

        # --- Distribution by Month ---
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        print("\n    By Month:")
        print(f"    {'Mon':>5} {'#':>5} {'PnL':>10} {'WR':>5}")
        for mo in range(1, 13):
            m = entry_dates.month == mo
            mp = pnls[m]
            mn = len(mp)
            if mn > 0:
                print(f"    {month_names[mo-1]:>5} {mn:>5} ${mp.sum():>+9,.0f} "
                      f"{(mp>0).sum()/mn*100:>4.0f}%")

        # --- Distribution by Entry Hour ---
        print("\n    By Entry Hour:")
        print(f"    {'Hour':>6} {'#':>5} {'PnL':>10} {'WR':>5}")
        for h in range(2, 16):
            m = entry_dates.hour == h
            hp = pnls[m]
            hn = len(hp)
            if hn > 0:
                print(f"    {h:>4}:00 {hn:>5} ${hp.sum():>+9,.0f} "
                      f"{(hp>0).sum()/hn*100:>4.0f}%")


# ======================================================================
# SECTION 3: Walk-Forward Folds (mapped to dates)
# ======================================================================

def run_wf_folds(all_data, idx):
    print("\n" + "=" * 100)
    print(" SECTION 3: WALK-FORWARD FOLDS (calendar dates)")
    print("=" * 100)

    n = len(idx)

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

    print(f"\n  {len(folds)} folds | IS={WF_IS_BARS} bars (~2y) | "
          f"OOS={WF_OOS_BARS} bars (~6m) | step={WF_STEP_BARS}")

    wf_results = {}

    for name, data in all_data.items():
        cfg = CANDIDATES[name]
        bt = data["bt"]
        entries = bt["trade_entry_bars"]
        exits = bt["trade_exit_bars"]
        pnls = bt["trade_pnls"]

        print(f"\n  --- {cfg_label(name, cfg)} ---")
        print(f"  {'#':>3} {'OOS Period':>28} {'Trades':>7} "
              f"{'PnL':>10} {'PF':>7} {'WR':>6} {'Status':>7}")

        fold_list = []
        for fi, (_is_s, _is_e, oos_s, oos_e) in enumerate(folds):
            oos_mask = (entries >= oos_s) & (exits <= oos_e)
            oos_pnls = pnls[oos_mask]
            oos_n = len(oos_pnls)
            oos_pnl = float(oos_pnls.sum()) if oos_n > 0 else 0.0

            gp = float(oos_pnls[oos_pnls > 0].sum()) if oos_n > 0 else 0.0
            gl = abs(float(oos_pnls[oos_pnls < 0].sum())) if oos_n > 0 else 0.0
            oos_pf = gp / gl if gl > 0 else 0.0
            oos_wr = float((oos_pnls > 0).sum() / oos_n * 100) if oos_n > 0 else 0.0

            is_go = oos_pf > 1.0 and oos_n >= 3
            status = " GO" if is_go else " NO"

            d_start = idx[oos_s].strftime("%Y-%m-%d")
            d_end = idx[min(oos_e - 1, n - 1)].strftime("%Y-%m-%d")

            fold_list.append({
                "fold": fi + 1, "oos_n": oos_n, "oos_pnl": oos_pnl,
                "oos_pf": oos_pf, "is_go": is_go, "d_start": d_start,
            })

            print(f"  {fi+1:>3} {d_start} to {d_end} {oos_n:>7} "
                  f"${oos_pnl:>+9,.0f} {oos_pf:>7.2f} {oos_wr:>5.0f}%{status}")

        n_go = sum(1 for f in fold_list if f["is_go"])
        n_total = len(fold_list)

        # Recency: last 6 folds (~3 years most recent)
        last_6 = fold_list[-6:]
        n_go_6 = sum(1 for f in last_6 if f["is_go"])
        recency_pct = n_go_6 / len(last_6) * 100 if last_6 else 0
        rec_status = ("GO" if recency_pct >= THRESHOLDS["recency_go"]
                      else "WARN" if recency_pct >= THRESHOLDS["recency_warn"]
                      else "KILL")

        print(f"\n  Overall: {n_go}/{n_total} GO ({n_go/n_total*100:.0f}%)")
        print(f"  Recency (last 6): {n_go_6}/{len(last_6)} GO "
              f"({recency_pct:.0f}%)  [{rec_status}]")
        recent_str = "  "
        for f in last_6:
            m = "+" if f["is_go"] else "-"
            recent_str += f"{f['d_start'][:7]}({m}) "
        print(recent_str)

        wf_results[name] = {
            "folds": fold_list, "n_go": n_go, "n_total": n_total,
            "recency_pct": recency_pct, "rec_status": rec_status,
        }

    return wf_results


# ======================================================================
# DECISION MATRIX
# ======================================================================

def run_decision_matrix(all_data, wf_results):
    print("\n" + "=" * 100)
    print(" DECISION MATRIX (thresholds fixed ex ante)")
    print("=" * 100)

    print("\n  Pre-set thresholds:")
    print(f"    Overlap:     > {THRESHOLDS['overlap_doublon']}% = DOUBLON, "
          f"> {THRESHOLDS['overlap_partial']}% = PARTIAL")
    print(f"    Worst trade: < ${THRESHOLDS['worst_trade_kill']:,} = KILL, "
          f"< ${THRESHOLDS['worst_trade_warn']:,} = WARN")
    print(f"    Max consec:  >= {THRESHOLDS['max_consec_kill']} = KILL, "
          f">= {THRESHOLDS['max_consec_warn']} = WARN")
    print(f"    W/L ratio:   < {THRESHOLDS['win_loss_kill']:.1f} = KILL, "
          f"< {THRESHOLDS['win_loss_warn']:.1f} = WARN")
    print(f"    Recency:     >= {THRESHOLDS['recency_go']}% = GO, "
          f">= {THRESHOLDS['recency_warn']}% = WARN")

    print(f"\n  {'Config':>6} | {'Worst Trd':>11} | {'MaxConsec':>9} | "
          f"{'W/L Ratio':>10} | {'Recency':>10} | {'Verdict':>8}")
    print(f"  {'-' * 72}")

    for name, data in all_data.items():
        bt = data["bt"]
        pnls = bt["trade_pnls"]

        worst = float(pnls.min())
        winners = pnls[pnls > 0]
        losers = pnls[pnls <= 0]
        avg_win = float(winners.mean()) if len(winners) > 0 else 0
        avg_loss = float(losers.mean()) if len(losers) > 0 else 0
        wl = abs(avg_win / avg_loss) if avg_loss != 0 else 99.9
        mc = max_consecutive_losses(pnls)

        rec = wf_results[name]["recency_pct"] if name in wf_results else 0

        # Classify each metric
        def grade(val, kill_thresh, warn_thresh, lower_is_worse=True):
            if lower_is_worse:
                if val < kill_thresh:
                    return "KILL"
                if val < warn_thresh:
                    return "WARN"
                return "GO"
            else:  # higher is worse
                if val >= kill_thresh:
                    return "KILL"
                if val >= warn_thresh:
                    return "WARN"
                return "GO"

        g_worst = grade(worst, THRESHOLDS["worst_trade_kill"],
                        THRESHOLDS["worst_trade_warn"])
        g_consec = grade(mc, THRESHOLDS["max_consec_kill"],
                         THRESHOLDS["max_consec_warn"], lower_is_worse=False)
        g_wl = grade(wl, THRESHOLDS["win_loss_kill"],
                     THRESHOLDS["win_loss_warn"])
        g_rec = grade(rec, THRESHOLDS["recency_warn"],
                      THRESHOLDS["recency_go"])

        grades = [g_worst, g_consec, g_wl, g_rec]
        if "KILL" in grades:
            verdict = "KILL"
        elif "WARN" in grades:
            verdict = "WARN"
        else:
            verdict = "GO"

        print(f"  {name:>6} | ${worst:>+8,.0f} [{g_worst:>4}] | "
              f"{mc:>5} [{g_consec:>4}] | "
              f"{wl:>6.2f} [{g_wl:>4}] | "
              f"{rec:>5.0f}% [{g_rec:>4}] | [{verdict:>4}]")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 100)
    print(" PHASE 13c DEEP ANALYSIS -- A, D, T1")
    print("=" * 100)

    # Load data
    print("\nLoading market data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    print(f"  {len(px_a):,} bars | {idx[0].date()} to {idx[-1].date()}")

    # Reconstruct all configs
    print("\nReconstructing backtests...")
    all_data = {}
    for name, cfg in CANDIDATES.items():
        data = reconstruct_full(cfg, aligned, px_a, px_b, idx, minutes)
        bt = data["bt"]
        print(f"  {cfg_label(name, cfg)}")
        print(f"    -> {bt['trades']} trades, ${bt['pnl']:,.0f}, PF {bt['profit_factor']:.2f}")
        all_data[name] = data

    # Section 1
    run_overlay(all_data, idx)

    # Section 2
    run_autopsy(all_data, idx, minutes)

    # Section 3
    wf_results = run_wf_folds(all_data, idx)

    # Decision Matrix
    run_decision_matrix(all_data, wf_results)

    print("\n" + "=" * 100)
    print(" ANALYSIS COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
