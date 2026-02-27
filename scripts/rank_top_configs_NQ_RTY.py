"""Ranking multi-critere des top configs OLS NQ/RTY.

Selectionne les candidats prometteurs (pre-filtre PF/trades),
calcule Sharpe, Calmar, MaxDD via run_backtest_vectorized,
classe en 5 groupes: Sharpe, Calmar, PnL, PropFirm, Equilibre.
Chaque groupe = top 5 avec patterns diversifies.

Usage:
    python scripts/rank_top_configs_NQ_RTY.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.config.instruments import get_pair_specs
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import (
    ConfidenceConfig,
    _apply_conf_filter_numba,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
BARS_PER_DAY = 264
BARS_PER_YEAR = BARS_PER_DAY * 252

METRIC_PROFILES = {
    "p12_64":    MetricsConfig(adf_window=12, hurst_window=64,  halflife_window=12, correlation_window=6),
    "p16_80":    MetricsConfig(adf_window=16, hurst_window=80,  halflife_window=16, correlation_window=8),
    "p18_96":    MetricsConfig(adf_window=18, hurst_window=96,  halflife_window=18, correlation_window=9),
    "p20_100":   MetricsConfig(adf_window=20, hurst_window=100, halflife_window=20, correlation_window=10),
    "p24_128":   MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "p28_144":   MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p30_160":   MetricsConfig(adf_window=30, hurst_window=160, halflife_window=30, correlation_window=15),
    "p36_192":   MetricsConfig(adf_window=36, hurst_window=192, halflife_window=36, correlation_window=18),
    "p42_224":   MetricsConfig(adf_window=42, hurst_window=224, halflife_window=42, correlation_window=21),
    "p48_256":   MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "p60_320":   MetricsConfig(adf_window=60, hurst_window=320, halflife_window=60, correlation_window=30),
    "p18_192":   MetricsConfig(adf_window=18, hurst_window=192, halflife_window=18, correlation_window=18),
    "p24_256":   MetricsConfig(adf_window=24, hurst_window=256, halflife_window=24, correlation_window=24),
    "p30_256":   MetricsConfig(adf_window=30, hurst_window=256, halflife_window=30, correlation_window=24),
    "p48_128":   MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
    "p48_96":    MetricsConfig(adf_window=48, hurst_window=96,  halflife_window=48, correlation_window=9),
    "p36_96":    MetricsConfig(adf_window=36, hurst_window=96,  halflife_window=36, correlation_window=9),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
    "08:00-14:00": (480, 840),
    "08:00-12:00": (480, 720),
    "06:00-12:00": (360, 720),
}

CONF_WEIGHTS = ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00)


def config_pattern(r):
    """Generate a pattern signature for diversity: OLS_bucket + ZW_bucket + profil_family."""
    ols = int(r["ols_window"])
    zw = int(r["zscore_window"])
    prof = r["profil"]
    # OLS bucket
    if ols <= 5000:
        ols_b = "short"
    elif ols <= 8000:
        ols_b = "mid"
    else:
        ols_b = "long"
    # ZW bucket
    if zw <= 24:
        zw_b = "fast"
    elif zw <= 36:
        zw_b = "med"
    else:
        zw_b = "slow"
    # Profile family
    if "96" in prof and prof.startswith("p3") or prof.startswith("p4"):
        pf = "adf_slow"
    elif "192" in prof or "256" in prof or "320" in prof:
        pf = "hurst_slow"
    else:
        pf = "aligned"
    return f"{ols_b}_{zw_b}_{pf}"


def select_diverse_top(candidates_df, sort_col, n=5, ascending=False):
    """Select top N by sort_col with diverse patterns."""
    sorted_df = candidates_df.sort_values(sort_col, ascending=ascending)
    selected = []
    seen_patterns = set()
    for _, r in sorted_df.iterrows():
        pat = config_pattern(r)
        if pat not in seen_patterns:
            seen_patterns.add(pat)
            selected.append(r)
            if len(selected) >= n:
                break
    # If not enough diverse, fill with remaining top
    if len(selected) < n:
        for _, r in sorted_df.iterrows():
            key = (int(r["ols_window"]), int(r["zscore_window"]), r["profil"])
            already = any(
                (int(s["ols_window"]), int(s["zscore_window"]), s["profil"]) == key
                for s in selected
            )
            if not already:
                selected.append(r)
                if len(selected) >= n:
                    break
    return selected


def main():
    csv_path = PROJECT_ROOT / "output" / "NQ_RTY" / "grid_refined_ols_filtered.csv"
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["profit_factor"])
    df = df[df["profit_factor"] < 100]
    print(f"Configs filtrees chargees: {len(df):,}")

    # Pre-select broader set of candidates for full backtest
    # We need diversity: take top by PnL, PF, trades from different clusters
    pre_candidates = []

    # Cluster = OLS + ZW + profil
    df["cluster"] = (df["ols_window"].astype(str) + "_" +
                     df["zscore_window"].astype(str) + "_" +
                     df["profil"])

    # Top 20 by PnL (deduplicated by cluster)
    for _, r in df.sort_values("pnl", ascending=False).drop_duplicates("cluster").head(20).iterrows():
        pre_candidates.append(r)

    # Top 20 by PF (deduplicated by cluster)
    for _, r in df.sort_values("profit_factor", ascending=False).drop_duplicates("cluster").head(20).iterrows():
        pre_candidates.append(r)

    # Top 20 by trades with PF > 1.4
    high_pf = df[df["profit_factor"] > 1.4]
    for _, r in high_pf.sort_values("trades", ascending=False).drop_duplicates("cluster").head(20).iterrows():
        pre_candidates.append(r)

    # Top 20 by PnL/trades ratio (efficiency)
    df_eff = df.copy()
    df_eff["efficiency"] = df_eff["pnl"] / df_eff["trades"]
    for _, r in df_eff.sort_values("efficiency", ascending=False).drop_duplicates("cluster").head(20).iterrows():
        pre_candidates.append(r)

    # Deduplicate
    seen = set()
    unique = []
    for r in pre_candidates:
        key = (int(r["ols_window"]), int(r["zscore_window"]), r["profil"], r["window"],
               r["z_entry"], r["z_exit"], r["z_stop"], r["min_confidence"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"Candidats uniques a backtester: {len(unique)}")

    # Load data
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    hedge_cache = {}
    results = []

    for i, r in enumerate(unique):
        ols_w = int(r["ols_window"])
        zw = int(r["zscore_window"])
        profile = r["profil"]
        window = r["window"]
        ze = r["z_entry"]
        zx = r["z_exit"]
        zs = r["z_stop"]
        conf = r["min_confidence"]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(unique)}] ...", flush=True)

        # Get hedge ratio (cached)
        hkey = (ols_w, zw)
        if hkey not in hedge_cache:
            est = create_estimator("ols_rolling", window=ols_w, zscore_window=zw)
            hr = est.estimate(aligned)
            spread = hr.spread
            beta = hr.beta.values
            mu = spread.rolling(zw).mean()
            sigma = spread.rolling(zw).std()
            with np.errstate(divide="ignore", invalid="ignore"):
                zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
            zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)
            hedge_cache[hkey] = (beta, zscore, spread)
        else:
            beta, zscore, spread = hedge_cache[hkey]

        # Metrics + confidence
        profile_cfg = METRIC_PROFILES[profile]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
        confidence = compute_confidence(metrics, CONF_WEIGHTS).values

        raw = generate_signals_numba(zscore, ze, zx, zs)
        if conf > 0:
            sig = _apply_conf_filter_numba(raw, confidence, conf)
        else:
            sig = raw.copy()

        entry_start, entry_end = WINDOWS_MAP[window]
        sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        # Equity metrics
        equity = bt["equity"]
        running_max = np.maximum.accumulate(equity)
        drawdown = equity - running_max
        max_dd = drawdown.min()
        max_dd_pct = abs((drawdown / running_max).min()) * 100

        # Sharpe
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(equity) / equity[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(BARS_PER_YEAR)
        else:
            sharpe = 0.0

        # Calmar
        total_return = (equity[-1] - equity[0]) / equity[0] if equity[0] != 0 else 0.0
        n_bars = len(equity)
        with np.errstate(invalid="ignore"):
            ann_return = (1 + total_return) ** (BARS_PER_YEAR / max(n_bars, 1)) - 1
        if np.isnan(ann_return) or np.isinf(ann_return):
            ann_return = 0.0
        calmar = (ann_return * 100) / max_dd_pct if max_dd_pct > 0 else 0.0

        # Max losing streak
        trade_pnls = bt["trade_pnls"]
        max_streak = 0
        current = 0
        for p in trade_pnls:
            if p < 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0

        # Yearly
        entry_bars = bt["trade_entry_bars"]
        entry_dates = idx[entry_bars]
        trade_df = pd.DataFrame({"year": entry_dates.year, "pnl": trade_pnls})
        yearly = trade_df.groupby("year")["pnl"].sum()
        neg_years = (yearly < 0).sum()

        # Consecutive drawdown (max $ lost in a row of losing trades)
        cons_dd = 0
        current_dd = 0
        for p in trade_pnls:
            if p < 0:
                current_dd += p
                cons_dd = min(cons_dd, current_dd)
            else:
                current_dd = 0

        results.append({
            "ols_w": ols_w, "zw": zw, "profile": profile, "window": window,
            "ze": ze, "zx": zx, "zs": zs, "conf": conf,
            "trades": bt["trades"], "wr": bt["win_rate"], "pnl": bt["pnl"],
            "pf": bt["profit_factor"], "max_dd": max_dd, "max_dd_pct": max_dd_pct,
            "sharpe": sharpe, "calmar": calmar,
            "avg_pnl": bt["avg_pnl_trade"],
            "max_streak": max_streak, "neg_years": neg_years,
            "cons_dd": cons_dd, "yearly": yearly,
            "pattern": config_pattern(r),
        })

    print(f"\n  Backtests termines: {len(results)}")

    # Convert to DataFrame for ranking
    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != "yearly"} for r in results])

    # ================================================================
    # DISPLAY 5 GROUPS
    # ================================================================
    groups = {
        "SHARPE": {"sort": "sharpe", "ascending": False, "desc": "Meilleur ratio rendement/risque annualise"},
        "CALMAR": {"sort": "calmar", "ascending": False, "desc": "Meilleur rendement annualise / drawdown"},
        "PNL": {"sort": "pnl", "ascending": False, "desc": "PnL absolu maximum"},
        "PROPFIRM": {"sort": "propfirm_score", "ascending": False, "desc": "MaxDD < $5k + regularite + faible streak"},
        "EQUILIBRE": {"sort": "balanced_score", "ascending": False, "desc": "Score composite (Sharpe + PF + PnL + securite)"},
    }

    # Compute PropFirm score: prioritize MaxDD < $5000, low streak, no neg years
    res_df["propfirm_score"] = (
        np.where(res_df["max_dd"].abs() < 5000, 50, 0) +       # MaxDD safe
        np.where(res_df["max_dd"].abs() < 8000, 20, 0) +       # MaxDD acceptable
        np.where(res_df["neg_years"] == 0, 20, 0) +             # No neg years
        np.where(res_df["max_streak"] <= 4, 10, 0) +            # Low streak
        res_df["pf"] * 5 +                                       # PF bonus
        res_df["pnl"] / 5000                                     # PnL bonus (scaled)
    )

    # Compute Balanced score: Sharpe + PF + PnL normalized + safety
    sharpe_norm = (res_df["sharpe"] - res_df["sharpe"].min()) / max(res_df["sharpe"].max() - res_df["sharpe"].min(), 1e-9)
    pf_norm = (res_df["pf"] - res_df["pf"].min()) / max(res_df["pf"].max() - res_df["pf"].min(), 1e-9)
    pnl_norm = (res_df["pnl"] - res_df["pnl"].min()) / max(res_df["pnl"].max() - res_df["pnl"].min(), 1e-9)
    dd_penalty = np.where(res_df["max_dd"].abs() > 8000, -0.3, 0) + np.where(res_df["max_dd"].abs() > 15000, -0.3, 0)
    neg_penalty = res_df["neg_years"] * -0.1
    res_df["balanced_score"] = sharpe_norm * 0.30 + pf_norm * 0.25 + pnl_norm * 0.25 + dd_penalty + neg_penalty + 0.20

    # Map results back
    for i, row in res_df.iterrows():
        results[i]["propfirm_score"] = row["propfirm_score"]
        results[i]["balanced_score"] = row["balanced_score"]

    for group_name, group_cfg in groups.items():
        print(f"\n{'='*140}")
        print(f"  TOP 5 {group_name} — {group_cfg['desc']}")
        print(f"{'='*140}")

        sort_col = group_cfg["sort"]
        asc = group_cfg["ascending"]

        # Select diverse top 5
        sorted_results = sorted(results, key=lambda x: x[sort_col], reverse=(not asc))
        selected = []
        seen_patterns = set()
        for r in sorted_results:
            pat = r["pattern"]
            if pat not in seen_patterns:
                seen_patterns.add(pat)
                selected.append(r)
                if len(selected) >= 5:
                    break
        # Fill if not enough
        if len(selected) < 5:
            for r in sorted_results:
                key = (r["ols_w"], r["zw"], r["profile"])
                already = any((s["ols_w"], s["zw"], s["profile"]) == key for s in selected)
                if not already:
                    selected.append(r)
                    if len(selected) >= 5:
                        break

        print(f"\n  {'#':>2} {'OLS':>6} {'ZW':>3} {'prof':<10} {'window':<14} "
              f"{'ze':>5} {'zx':>5} {'zs':>4} {'cf':>3} | "
              f"{'Trd':>4} {'WR%':>6} {'PnL':>10} {'PF':>5} {'MaxDD':>9} "
              f"{'Sharpe':>7} {'Calmar':>7} {'Strk':>4} {'Neg':>3} {'Pattern':<20}")
        print("  " + "-" * 135)

        for rank, r in enumerate(selected, 1):
            dd_flag = " <<<" if abs(r["max_dd"]) >= 8000 else ""
            safe_flag = " SAFE" if abs(r["max_dd"]) < 5000 else ""
            print(f"  {rank:>2} {r['ols_w']:>6} {r['zw']:>3} {r['profile']:<10} {r['window']:<14} "
                  f"{r['ze']:>5.2f} {r['zx']:>5.2f} {r['zs']:>4.1f} {r['conf']:>3.0f} | "
                  f"{r['trades']:>4} {r['wr']:>5.1f}% ${r['pnl']:>9,.0f} {r['pf']:>5.2f} "
                  f"${r['max_dd']:>8,.0f} {r['sharpe']:>7.2f} {r['calmar']:>7.2f} "
                  f"{r['max_streak']:>4} {r['neg_years']:>3} {r['pattern']:<20}"
                  f"{dd_flag}{safe_flag}")
            # Yearly
            years_str = "     "
            for y, p in r["yearly"].items():
                flag = "***" if p < 0 else ""
                years_str += f"{y}=${p:+,.0f}{flag}  "
            print(years_str)
            print()


if __name__ == "__main__":
    main()
