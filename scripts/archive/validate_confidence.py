"""Validation quant du confidence score NQ_YM.

Sections:
  --temporal     : 1A distribution temporelle des trades
  --decompose    : 1B decomposition composantes du score
  --isoos        : 3A split IS/OOS temporel
  --permutation  : 3B test de permutation (1000 randomisations)
  --sensitivity  : 2A robustesse parametrique confidence
  --all          : tout lancer

Config robuste: OLS=2640, ZW=36, entry=3.0, exit=1.5, stop=4.0, profil tres_court, conf>=70%
"""

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache import load_aligned_pair_cache
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.generator import SignalConfig, SignalGenerator
from src.signals.filters import (
    ConfidenceConfig, compute_confidence, _score_linear, _score_halflife,
)
from src.backtest.engine import run_backtest_vectorized

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

OLS_WINDOW = 2640
ZSCORE_WINDOW = 36
Z_ENTRY = 3.0
Z_EXIT = 1.5
Z_STOP = 4.0
MIN_CONFIDENCE = 70.0
METRICS_CFG = MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6)
CONF_CFG = ConfidenceConfig()

MULT_A, MULT_B = 20.0, 5.0
TICK_A, TICK_B = 0.25, 1.0
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0


# ══════════════════════════════════════════════════════════════════════
# Pipeline helper
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(aligned, metrics_cfg=None, conf_cfg=None, min_conf=None):
    """Run full pipeline: hedge -> metrics -> signals -> confidence filter -> backtest.
    Returns (bt_dict, raw_signals, confidence, metrics, idx)."""
    if metrics_cfg is None:
        metrics_cfg = METRICS_CFG
    if conf_cfg is None:
        conf_cfg = CONF_CFG
    if min_conf is None:
        min_conf = MIN_CONFIDENCE

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    # Hedge
    est = create_estimator("ols_rolling", window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    hr = est.estimate(aligned)
    beta = hr.beta.values

    # Metrics
    metrics = compute_all_metrics(hr.spread, aligned.df["close_a"], aligned.df["close_b"], metrics_cfg)

    # Confidence
    confidence = compute_confidence(metrics, conf_cfg).values

    # Signals
    gen = SignalGenerator(config=SignalConfig(z_entry=Z_ENTRY, z_exit=Z_EXIT, z_stop=Z_STOP))
    raw_signals = gen.generate(hr.zscore).values

    # Apply confidence filter (entry-only)
    sig = raw_signals.copy()
    prev = 0
    for t in range(n):
        curr = sig[t]
        if (prev == 0) and (curr != 0) and (confidence[t] < min_conf):
            sig[t] = 0
        prev = sig[t]

    # Trading window [04:00-14:00) CT
    minutes = idx.hour * 60 + idx.minute
    tw_mask = (minutes >= 240) & (minutes < 840)
    sig[~tw_mask] = 0

    # Backtest
    bt = run_backtest_vectorized(
        px_a, px_b, sig, beta,
        mult_a=MULT_A, mult_b=MULT_B, tick_a=TICK_A, tick_b=TICK_B,
        slippage_ticks=SLIPPAGE, commission=COMMISSION, initial_capital=INITIAL_CAPITAL,
    )

    return bt, raw_signals, confidence, metrics, idx


# ══════════════════════════════════════════════════════════════════════
# 1A — Distribution temporelle
# ══════════════════════════════════════════════════════════════════════

def section_temporal(aligned):
    print("\n" + "=" * 100)
    print(" 1A - DISTRIBUTION TEMPORELLE DES TRADES")
    print("=" * 100)

    bt, raw_signals, confidence, metrics, idx = run_pipeline(aligned)
    n_trades = bt["trades"]
    if n_trades == 0:
        print(" AUCUN TRADE. STOP.")
        return

    te = bt["trade_entry_bars"]
    tx = bt["trade_exit_bars"]
    pnls = bt["trade_pnls"]
    sides = bt["trade_sides"]

    entry_times = idx[te]
    exit_times = idx[tx]

    print(f"\n Total: {n_trades} trades, PnL ${bt['pnl']:,.0f}, PF {bt['profit_factor']:.2f}")
    print(f" Periode: {entry_times[0].strftime('%Y-%m-%d')} -> {entry_times[-1].strftime('%Y-%m-%d')}")

    # --- Par annee ---
    print("\n --- PnL PAR ANNEE ---")
    print(f" {'Annee':<6} {'Trades':>6} {'Win%':>6} {'PnL':>10} {'PF':>6} {'Avg':>8}")
    print(" " + "-" * 50)

    years = entry_times.year
    for year in sorted(years.unique()):
        mask = years == year
        yr_pnls = pnls[mask]
        yr_n = len(yr_pnls)
        yr_wins = (yr_pnls > 0).sum()
        yr_wr = yr_wins / yr_n * 100 if yr_n > 0 else 0
        yr_total = yr_pnls.sum()
        yr_gains = yr_pnls[yr_pnls > 0].sum()
        yr_losses = abs(yr_pnls[yr_pnls <= 0].sum())
        yr_pf = yr_gains / yr_losses if yr_losses > 0 else float("inf")
        yr_avg = yr_total / yr_n if yr_n > 0 else 0
        print(f" {year:<6} {yr_n:>6} {yr_wr:>5.1f}% ${yr_total:>9,.0f} {yr_pf:>5.2f} ${yr_avg:>7,.0f}")

    # --- Par trimestre ---
    print("\n --- TRADES PAR TRIMESTRE ---")
    quarters = entry_times.to_period("Q")
    qdf = pd.DataFrame({"quarter": quarters, "pnl": pnls})
    qg = qdf.groupby("quarter")["pnl"]
    print(f" {'Trimestre':<10} {'Trades':>6} {'PnL':>10}")
    print(" " + "-" * 30)
    for q, g in qg:
        print(f" {str(q):<10} {len(g):>6} ${g.sum():>9,.0f}")

    # --- Distribution horaire ---
    print("\n --- DISTRIBUTION HORAIRE DES ENTREES ---")
    hours = entry_times.hour
    print(f" {'Heure CT':<10} {'Trades':>6} {'PnL':>10} {'Win%':>6}")
    print(" " + "-" * 36)
    for h in range(4, 14):
        mask = hours == h
        h_pnls = pnls[mask]
        h_n = len(h_pnls)
        if h_n == 0:
            continue
        h_wr = (h_pnls > 0).sum() / h_n * 100
        print(f" {h:>2}:00 CT   {h_n:>6} ${h_pnls.sum():>9,.0f} {h_wr:>5.1f}%")

    # --- Long vs Short ---
    print("\n --- LONG vs SHORT ---")
    print(f" {'Side':<8} {'Trades':>6} {'Win%':>6} {'PnL':>10} {'PF':>6} {'Avg':>8}")
    print(" " + "-" * 50)
    for side_val, side_name in [(1, "LONG"), (-1, "SHORT")]:
        mask = sides == side_val
        s_pnls = pnls[mask]
        s_n = len(s_pnls)
        if s_n == 0:
            print(f" {side_name:<8} {0:>6} {'---':>6} {'---':>10} {'---':>6} {'---':>8}")
            continue
        s_wr = (s_pnls > 0).sum() / s_n * 100
        s_gains = s_pnls[s_pnls > 0].sum()
        s_losses = abs(s_pnls[s_pnls <= 0].sum())
        s_pf = s_gains / s_losses if s_losses > 0 else float("inf")
        s_avg = s_pnls.sum() / s_n
        print(f" {side_name:<8} {s_n:>6} {s_wr:>5.1f}% ${s_pnls.sum():>9,.0f} {s_pf:>5.2f} ${s_avg:>7,.0f}")

    # --- Criteres GO/STOP ---
    print("\n --- VERDICT 1A ---")
    years_with_trades = years.nunique()
    total_pnl = pnls.sum()
    pnl_by_year = qdf.groupby(entry_times.year)["pnl"].sum()
    max_year_pnl = pnl_by_year.max()
    concentration = max_year_pnl / total_pnl * 100 if total_pnl > 0 else 100

    long_pnl = pnls[sides == 1].sum()
    short_pnl = pnls[sides == -1].sum()
    both_positive = long_pnl > 0 and short_pnl > 0

    print(f" Annees avec trades: {years_with_trades} (GO >= 4)")
    print(f" Concentration max 1 annee: {concentration:.1f}% du PnL (STOP si > 50%)")
    print(f" Long PnL: ${long_pnl:,.0f} | Short PnL: ${short_pnl:,.0f} | Both positive: {both_positive}")

    if years_with_trades >= 4 and concentration < 50 and both_positive:
        print(" >>> VERDICT: GO")
    elif years_with_trades >= 4 and concentration < 50:
        print(" >>> VERDICT: GO (avec reserve - asymetrie long/short)")
    else:
        print(" >>> VERDICT: STOP - edge concentre ou asymetrique")


# ══════════════════════════════════════════════════════════════════════
# 1B — Decomposition composantes
# ══════════════════════════════════════════════════════════════════════

def section_decompose(aligned):
    print("\n" + "=" * 100)
    print(" 1B - DECOMPOSITION DU SCORE PAR COMPOSANTE")
    print("=" * 100)

    bt, raw_signals, confidence, metrics, idx = run_pipeline(aligned)
    n_trades = bt["trades"]
    if n_trades == 0:
        print(" AUCUN TRADE.")
        return

    te = bt["trade_entry_bars"]
    pnls = bt["trade_pnls"]
    conf_cfg = CONF_CFG

    # Extraire les sous-scores a l'entree de chaque trade
    adf_vals = metrics["adf_stat"].values
    hurst_vals = metrics["hurst"].values
    corr_vals = metrics["correlation"].values
    hl_vals = metrics["half_life"].values

    scores = []
    for i in range(n_trades):
        t = te[i]
        s_adf = _score_linear(adf_vals[t], conf_cfg.adf_worst, conf_cfg.adf_best)
        s_hurst = _score_linear(hurst_vals[t], conf_cfg.hurst_worst, conf_cfg.hurst_best)
        s_corr = _score_linear(corr_vals[t], conf_cfg.corr_worst, conf_cfg.corr_best)
        s_hl = _score_halflife(hl_vals[t], conf_cfg)
        scores.append({
            "s_adf": s_adf, "s_hurst": s_hurst, "s_corr": s_corr, "s_hl": s_hl,
            "confidence": confidence[t], "pnl": float(pnls[i]),
            "adf_raw": float(adf_vals[t]), "hurst_raw": float(hurst_vals[t]),
            "corr_raw": float(corr_vals[t]), "hl_raw": float(hl_vals[t]),
        })

    df = pd.DataFrame(scores)

    print(f"\n {n_trades} trades analyses")
    print("\n --- STATISTIQUES DES SOUS-SCORES A L'ENTREE ---")
    for col in ["s_adf", "s_hurst", "s_corr", "s_hl", "confidence"]:
        vals = df[col]
        print(f" {col:<12} mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

    # Correlation de rang (Spearman) entre chaque sous-score et le PnL
    print("\n --- CORRELATION DE RANG (Spearman) vs PnL ---")
    for col in ["s_adf", "s_hurst", "s_corr", "s_hl", "confidence"]:
        rho, pval = sp_stats.spearmanr(df[col], df["pnl"])
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f" {col:<12} rho={rho:+.3f}  p={pval:.3f} {sig}")

    # Winners vs Losers
    print("\n --- SCORES MOYENS: WINNERS vs LOSERS ---")
    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] <= 0]
    print(f" {'Composante':<12} {'Winners':>8} {'Losers':>8} {'Delta':>8}")
    print(" " + "-" * 40)
    for col in ["s_adf", "s_hurst", "s_corr", "s_hl", "confidence"]:
        w_mean = winners[col].mean() if len(winners) > 0 else 0
        l_mean = losers[col].mean() if len(losers) > 0 else 0
        print(f" {col:<12} {w_mean:>7.3f}  {l_mean:>7.3f}  {w_mean - l_mean:>+7.3f}")


# ══════════════════════════════════════════════════════════════════════
# 3A — Split IS/OOS
# ══════════════════════════════════════════════════════════════════════

def section_isoos(aligned):
    print("\n" + "=" * 100)
    print(" 3A - SPLIT IN-SAMPLE / OUT-OF-SAMPLE (60/40)")
    print("=" * 100)

    df = aligned.df
    n = len(df)
    split_idx = int(n * 0.6)
    split_date = df.index[split_idx]

    print(f" Total barres: {n:,}")
    print(f" Split a: {split_date} (barre {split_idx})")
    print(f" IS: {df.index[0]} -> {df.index[split_idx-1]} ({split_idx:,} barres)")
    print(f" OOS: {df.index[split_idx]} -> {df.index[-1]} ({n - split_idx:,} barres)")

    # Create sub-aligned pairs (simple DataFrame slice, keep same structure)
    from src.data.alignment import AlignedPair
    is_df = df.iloc[:split_idx].copy()
    oos_df = df.iloc[split_idx:].copy()

    is_aligned = AlignedPair(df=is_df, pair=aligned.pair, timeframe=aligned.timeframe)
    oos_aligned = AlignedPair(df=oos_df, pair=aligned.pair, timeframe=aligned.timeframe)

    def run_and_report(label, al):
        bt, _, _, _, idx = run_pipeline(al)
        n_tr = bt["trades"]
        if n_tr == 0:
            print(f"\n {label}: 0 trades")
            return bt

        eq = bt["equity"]
        returns = np.diff(eq) / eq[:-1]
        returns = np.nan_to_num(returns, nan=0.0)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(264 * 252) if np.std(returns) > 0 else 0

        running_max = np.maximum.accumulate(eq)
        dd = np.nan_to_num((running_max - eq) / running_max * 100)
        max_dd = dd.max()

        print(f"\n {label}:")
        print(f"   Trades: {n_tr}")
        print(f"   Win rate: {bt['win_rate']:.1f}%")
        print(f"   PnL: ${bt['pnl']:,.0f}")
        print(f"   PF: {bt['profit_factor']:.2f}")
        print(f"   Avg PnL/trade: ${bt['avg_pnl_trade']:,.0f}")
        print(f"   Sharpe: {sharpe:.2f}")
        print(f"   Max DD: {max_dd:.1f}%")
        print(f"   Avg duration: {bt['avg_duration_bars']:.1f} bars ({bt['avg_duration_bars'] * 5 / 60:.1f}h)")
        return bt

    bt_is = run_and_report("IN-SAMPLE (60%)", is_aligned)
    bt_oos = run_and_report("OUT-OF-SAMPLE (40%)", oos_aligned)

    # Verdict
    print("\n --- VERDICT 3A ---")
    if bt_oos["trades"] == 0:
        print(" >>> STOP: 0 trades OOS")
        return

    pf_oos = bt_oos["profit_factor"]
    eq_oos = bt_oos["equity"]
    ret_oos = np.diff(eq_oos) / eq_oos[:-1]
    ret_oos = np.nan_to_num(ret_oos, nan=0.0)
    sharpe_oos = (np.mean(ret_oos) / np.std(ret_oos)) * np.sqrt(264 * 252) if np.std(ret_oos) > 0 else 0
    trades_oos = bt_oos["trades"]

    print(f" PF OOS: {pf_oos:.2f} (GO > 1.3)")
    print(f" Sharpe OOS: {sharpe_oos:.2f} (GO > 0.5)")
    print(f" Trades OOS: {trades_oos} (GO > 25)")

    if pf_oos > 1.3 and sharpe_oos > 0.5 and trades_oos > 25:
        print(" >>> VERDICT: GO")
    elif pf_oos > 1.0 and sharpe_oos > 0:
        print(" >>> VERDICT: MARGINAL (positif mais sous les seuils)")
    else:
        print(" >>> VERDICT: STOP")


# ══════════════════════════════════════════════════════════════════════
# 3B — Test de permutation
# ══════════════════════════════════════════════════════════════════════

def section_permutation(aligned, n_perms=1000):
    print("\n" + "=" * 100)
    print(f" 3B - TEST DE PERMUTATION ({n_perms} randomisations)")
    print("=" * 100)

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    # Compute everything once
    est = create_estimator("ols_rolling", window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    hr = est.estimate(aligned)
    beta = hr.beta.values

    metrics = compute_all_metrics(hr.spread, aligned.df["close_a"], aligned.df["close_b"], METRICS_CFG)
    confidence = compute_confidence(metrics, CONF_CFG).values

    gen = SignalGenerator(config=SignalConfig(z_entry=Z_ENTRY, z_exit=Z_EXIT, z_stop=Z_STOP))
    raw_signals = gen.generate(hr.zscore).values

    minutes = idx.hour * 60 + idx.minute
    tw_mask = (minutes >= 240) & (minutes < 840)

    def backtest_with_confidence(conf_vec, min_conf):
        sig = raw_signals.copy()
        prev = 0
        for t in range(n):
            curr = sig[t]
            if (prev == 0) and (curr != 0) and (conf_vec[t] < min_conf):
                sig[t] = 0
            prev = sig[t]
        sig[~tw_mask] = 0

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            mult_a=MULT_A, mult_b=MULT_B, tick_a=TICK_A, tick_b=TICK_B,
            slippage_ticks=SLIPPAGE, commission=COMMISSION, initial_capital=INITIAL_CAPITAL,
        )
        return bt["profit_factor"], bt["pnl"], bt["trades"]

    # Observed
    pf_obs, pnl_obs, trades_obs = backtest_with_confidence(confidence, MIN_CONFIDENCE)
    print(f"\n Observed: {trades_obs} trades, PnL ${pnl_obs:,.0f}, PF {pf_obs:.2f}")

    # Permutations
    rng = np.random.default_rng(42)
    pf_perms = np.zeros(n_perms)
    pnl_perms = np.zeros(n_perms)
    trades_perms = np.zeros(n_perms, dtype=int)

    print(f" Running {n_perms} permutations...")
    for i in range(n_perms):
        perm_conf = rng.permutation(confidence)
        pf_perms[i], pnl_perms[i], trades_perms[i] = backtest_with_confidence(perm_conf, MIN_CONFIDENCE)

        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{n_perms} done...")

    # Results
    p_value_pf = (pf_perms >= pf_obs).mean()
    p_value_pnl = (pnl_perms >= pnl_obs).mean()

    print(f"\n --- RESULTATS ---")
    print(f" PF observe: {pf_obs:.2f}")
    print(f" PF permutations: mean={pf_perms.mean():.2f}, std={pf_perms.std():.2f}, "
          f"median={np.median(pf_perms):.2f}, max={pf_perms.max():.2f}")
    print(f" p-value (PF): {p_value_pf:.4f} ({(pf_perms >= pf_obs).sum()}/{n_perms} permutations >= observe)")
    print(f"")
    print(f" PnL observe: ${pnl_obs:,.0f}")
    print(f" PnL permutations: mean=${pnl_perms.mean():,.0f}, max=${pnl_perms.max():,.0f}")
    print(f" p-value (PnL): {p_value_pnl:.4f}")
    print(f"")
    print(f" Trades observe: {trades_obs}")
    print(f" Trades permutations: mean={trades_perms.mean():.0f}, std={trades_perms.std():.0f}")

    print("\n --- VERDICT 3B ---")
    print(f" p-value PF: {p_value_pf:.4f} (GO < 0.05, STOP > 0.10)")
    if p_value_pf < 0.05:
        print(" >>> VERDICT: GO - le confidence score capture un signal reel")
    elif p_value_pf < 0.10:
        print(" >>> VERDICT: MARGINAL")
    else:
        print(" >>> VERDICT: STOP - le resultat est reproductible par hasard")


# ══════════════════════════════════════════════════════════════════════
# 2A — Robustesse parametrique confidence
# ══════════════════════════════════════════════════════════════════════

def section_sensitivity(aligned):
    print("\n" + "=" * 100)
    print(" 2A - ROBUSTESSE PARAMETRIQUE DU CONFIDENCE SCORE")
    print("=" * 100)

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    # Pre-compute hedge + signals (shared)
    est = create_estimator("ols_rolling", window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    hr = est.estimate(aligned)
    beta = hr.beta.values

    metrics = compute_all_metrics(hr.spread, aligned.df["close_a"], aligned.df["close_b"], METRICS_CFG)

    gen = SignalGenerator(config=SignalConfig(z_entry=Z_ENTRY, z_exit=Z_EXIT, z_stop=Z_STOP))
    raw_signals = gen.generate(hr.zscore).values

    minutes = idx.hour * 60 + idx.minute
    tw_mask = (minutes >= 240) & (minutes < 840)

    def run_with_conf_config(label, conf_cfg, min_conf):
        confidence = compute_confidence(metrics, conf_cfg).values
        sig = raw_signals.copy()
        prev = 0
        for t in range(n):
            curr = sig[t]
            if (prev == 0) and (curr != 0) and (confidence[t] < min_conf):
                sig[t] = 0
            prev = sig[t]
        sig[~tw_mask] = 0

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            mult_a=MULT_A, mult_b=MULT_B, tick_a=TICK_A, tick_b=TICK_B,
            slippage_ticks=SLIPPAGE, commission=COMMISSION, initial_capital=INITIAL_CAPITAL,
        )
        print(f" {label:<35} {bt['trades']:>5} {bt['win_rate']:>5.1f}% "
              f"${bt['pnl']:>10,.0f} {bt['profit_factor']:>5.2f} ${bt['avg_pnl_trade']:>7,.0f}")
        return bt["profit_factor"]

    header = f" {'Variation':<35} {'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'Avg':>7}"
    divider = " " + "-" * 75

    # --- min_confidence ---
    print("\n --- VARIATION min_confidence ---")
    print(header)
    print(divider)
    pfs = []
    for mc in [55, 60, 65, 70, 75, 80]:
        pf = run_with_conf_config(f"min_confidence={mc}%", CONF_CFG, mc)
        pfs.append(pf)

    # --- adf_gate ---
    print("\n --- VARIATION adf_gate ---")
    print(header)
    print(divider)
    for gate in [-0.50, -0.75, -1.00, -1.25, -1.50]:
        cfg = replace(CONF_CFG, adf_gate=gate, adf_worst=gate)
        pf = run_with_conf_config(f"adf_gate={gate}", cfg, MIN_CONFIDENCE)
        pfs.append(pf)

    # --- adf_best ---
    print("\n --- VARIATION adf_best ---")
    print(header)
    print(divider)
    for ab in [-3.00, -3.50, -4.00, -4.50]:
        cfg = replace(CONF_CFG, adf_best=ab)
        pf = run_with_conf_config(f"adf_best={ab}", cfg, MIN_CONFIDENCE)
        pfs.append(pf)

    # --- hurst bounds ---
    print("\n --- VARIATION hurst_worst / hurst_best ---")
    print(header)
    print(divider)
    for hw, hb in [(0.50, 0.36), (0.52, 0.38), (0.54, 0.40), (0.56, 0.42)]:
        cfg = replace(CONF_CFG, hurst_worst=hw, hurst_best=hb)
        pf = run_with_conf_config(f"hurst=[{hw},{hb}]", cfg, MIN_CONFIDENCE)
        pfs.append(pf)

    # --- corr bounds ---
    print("\n --- VARIATION corr_worst / corr_best ---")
    print(header)
    print(divider)
    for cw, cb in [(0.60, 0.88), (0.65, 0.92), (0.70, 0.94), (0.60, 0.95)]:
        cfg = replace(CONF_CFG, corr_worst=cw, corr_best=cb)
        pf = run_with_conf_config(f"corr=[{cw},{cb}]", cfg, MIN_CONFIDENCE)
        pfs.append(pf)

    # --- weights ---
    print("\n --- VARIATION poids ADF ---")
    print(header)
    print(divider)
    for w_adf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        remaining = 1.0 - w_adf
        # Scale other weights proportionally
        base_other = 0.25 + 0.20 + 0.15  # 0.60
        w_h = 0.25 / base_other * remaining
        w_c = 0.20 / base_other * remaining
        w_hl = 0.15 / base_other * remaining
        cfg = replace(CONF_CFG, w_adf=w_adf, w_hurst=w_h, w_corr=w_c, w_hl=w_hl)
        pf = run_with_conf_config(f"w_adf={w_adf:.2f}", cfg, MIN_CONFIDENCE)
        pfs.append(pf)

    # Verdict
    print("\n --- VERDICT 2A ---")
    pfs_arr = np.array(pfs)
    above_15 = (pfs_arr > 1.5).sum()
    total = len(pfs_arr)
    print(f" {above_15}/{total} configs avec PF > 1.5 ({above_15/total*100:.0f}%)")
    print(f" PF range: [{pfs_arr.min():.2f}, {pfs_arr.max():.2f}]")
    if above_15 / total >= 0.70:
        print(" >>> VERDICT: GO - robuste aux perturbations")
    elif above_15 / total >= 0.50:
        print(" >>> VERDICT: MARGINAL")
    else:
        print(" >>> VERDICT: STOP - trop sensible aux parametres")


# ══════════════════════════════════════════════════════════════════════
# 3C — Walk-Forward Analysis
# ══════════════════════════════════════════════════════════════════════

def section_walkforward(aligned):
    print("\n" + "=" * 100)
    print(" 3C - WALK-FORWARD ANALYSIS (IS=2ans, OOS=6mois, pas=6mois)")
    print("=" * 100)

    from src.data.alignment import AlignedPair

    df = aligned.df
    idx = df.index
    total_start = idx[0]
    total_end = idx[-1]

    # Build windows: IS=2 years, OOS=6 months, step=6 months
    is_years = 2
    oos_months = 6
    step_months = 6

    windows = []
    cursor = total_start + pd.DateOffset(years=is_years)
    while cursor + pd.DateOffset(months=oos_months) <= total_end + pd.Timedelta(days=1):
        is_start = cursor - pd.DateOffset(years=is_years)
        is_end = cursor
        oos_start = cursor
        oos_end = cursor + pd.DateOffset(months=oos_months)
        windows.append((is_start, is_end, oos_start, oos_end))
        cursor += pd.DateOffset(months=step_months)

    print(f" {len(windows)} fenetres OOS identifiees")
    print(f" Donnees: {total_start.strftime('%Y-%m-%d')} -> {total_end.strftime('%Y-%m-%d')}")

    header = (f" {'#':<3} {'OOS Period':<25} {'Trd':>4} {'Win%':>5} "
              f"{'PnL':>10} {'PF':>5} {'Avg':>7} {'Shrp':>5} {'DD%':>5}")
    print(f"\n{header}")
    print(" " + "-" * 80)

    oos_results = []
    all_oos_pnls = []

    for i, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        # Slice OOS data
        oos_mask = (idx >= oos_start) & (idx < oos_end)
        oos_df = df.loc[oos_mask].copy()
        if len(oos_df) < 1000:
            print(f" {i+1:<3} {oos_start.strftime('%Y-%m')} -> {oos_end.strftime('%Y-%m')}   SKIP (too few bars: {len(oos_df)})")
            continue

        oos_aligned = AlignedPair(df=oos_df, pair=aligned.pair, timeframe=aligned.timeframe)
        bt, _, _, _, _ = run_pipeline(oos_aligned)

        n_tr = bt["trades"]
        if n_tr == 0:
            print(f" {i+1:<3} {oos_start.strftime('%Y-%m')} -> {oos_end.strftime('%Y-%m')}   0 trades")
            oos_results.append({"trades": 0, "pnl": 0, "pf": 0, "sharpe": 0})
            continue

        eq = bt["equity"]
        ret = np.diff(eq) / eq[:-1]
        ret = np.nan_to_num(ret, nan=0.0)
        sharpe = (np.mean(ret) / np.std(ret)) * np.sqrt(264 * 252) if np.std(ret) > 0 else 0

        running_max = np.maximum.accumulate(eq)
        dd = np.nan_to_num((running_max - eq) / running_max * 100)
        max_dd = dd.max()

        print(f" {i+1:<3} {oos_start.strftime('%Y-%m')} -> {oos_end.strftime('%Y-%m')}  "
              f"{n_tr:>4} {bt['win_rate']:>5.1f}% "
              f"${bt['pnl']:>9,.0f} {bt['profit_factor']:>5.2f} "
              f"${bt['avg_pnl_trade']:>6,.0f} {sharpe:>5.2f} {max_dd:>5.1f}%")

        oos_results.append({
            "trades": n_tr, "pnl": bt["pnl"], "pf": bt["profit_factor"], "sharpe": sharpe,
        })
        all_oos_pnls.extend(bt["trade_pnls"].tolist())

    # Aggregate
    print("\n --- AGGREGATION OOS ---")
    if not oos_results:
        print(" Aucune fenetre valide.")
        return

    n_windows = len(oos_results)
    n_profitable = sum(1 for r in oos_results if r["pnl"] > 0)
    total_oos_pnl = sum(r["pnl"] for r in oos_results)
    total_oos_trades = sum(r["trades"] for r in oos_results)
    pfs = [r["pf"] for r in oos_results if r["trades"] > 0]
    mean_pf = np.mean(pfs) if pfs else 0

    print(f" Fenetres profitables: {n_profitable}/{n_windows}")
    print(f" PnL total OOS: ${total_oos_pnl:,.0f}")
    print(f" Trades total OOS: {total_oos_trades}")
    print(f" PF moyen OOS: {mean_pf:.2f}")

    print("\n --- VERDICT 3C ---")
    print(f" Fenetres profitables: {n_profitable}/{n_windows} (GO >= {max(4, int(n_windows * 0.66))})")
    print(f" PF moyen OOS: {mean_pf:.2f} (GO > 1.2)")
    threshold = max(4, int(n_windows * 0.66))
    if n_profitable >= threshold and mean_pf > 1.2:
        print(" >>> VERDICT: GO")
    elif n_profitable >= n_windows // 2:
        print(" >>> VERDICT: MARGINAL")
    else:
        print(" >>> VERDICT: STOP")

    return all_oos_pnls


# ══════════════════════════════════════════════════════════════════════
# 4A — Monte Carlo propfirm simulation
# ══════════════════════════════════════════════════════════════════════

def section_propfirm(aligned, oos_pnls=None):
    print("\n" + "=" * 100)
    print(" 4A - SIMULATION MONTE CARLO PROPFIRM")
    print("=" * 100)

    # If no OOS pnls provided, compute from OOS split
    if oos_pnls is None or len(oos_pnls) == 0:
        from src.data.alignment import AlignedPair
        df = aligned.df
        n = len(df)
        split_idx = int(n * 0.6)
        oos_df = df.iloc[split_idx:].copy()
        oos_aligned = AlignedPair(df=oos_df, pair=aligned.pair, timeframe=aligned.timeframe)
        bt, _, _, _, _ = run_pipeline(oos_aligned)
        if bt["trades"] == 0:
            print(" 0 trades OOS. Impossible de simuler.")
            return
        oos_pnls = bt["trade_pnls"].tolist()

    pnls = np.array(oos_pnls)
    n_trades = len(pnls)

    print(f"\n Distribution des trades OOS ({n_trades} trades):")
    print(f"   Mean: ${np.mean(pnls):,.0f}")
    print(f"   Median: ${np.median(pnls):,.0f}")
    print(f"   Std: ${np.std(pnls):,.0f}")
    print(f"   Min: ${np.min(pnls):,.0f} | Max: ${np.max(pnls):,.0f}")
    print(f"   Win rate: {(pnls > 0).mean() * 100:.1f}%")

    # Propfirm parameters
    target_pnl = 75_000.0    # target annuel ($300/jour * 250 jours)
    max_dd = 5_000.0          # max drawdown
    trades_per_year = 252     # ~1 trade/jour (94 trades / 5 ans ~ 19/an, mais on simule plus)
    n_sims = 10_000

    # Estimate realistic trades per year from data
    # 94 trades over ~5.1 years = ~18.4 trades/year
    years_data = (aligned.df.index[-1] - aligned.df.index[0]).days / 365.25
    trades_per_year_actual = n_trades / (years_data * 0.4)  # OOS is 40% of data
    trades_per_year_actual = max(trades_per_year_actual, 1)

    print(f"\n Parametres propfirm:")
    print(f"   Target annuel: ${target_pnl:,.0f} ($300/jour)")
    print(f"   Max drawdown: ${max_dd:,.0f}")
    print(f"   Trades/an estimes (OOS): {trades_per_year_actual:.0f}")
    print(f"   Simulations: {n_sims:,}")

    rng = np.random.default_rng(42)
    n_trades_sim = int(trades_per_year_actual)

    hits_target = 0
    hits_dd = 0
    neither = 0
    final_pnls = np.zeros(n_sims)

    for sim in range(n_sims):
        # Draw n_trades_sim trades with replacement
        trade_seq = rng.choice(pnls, size=n_trades_sim, replace=True)
        cum_pnl = np.cumsum(trade_seq)
        final_pnls[sim] = cum_pnl[-1]

        # Track peak and drawdown
        peak = 0.0
        dd_breached = False
        target_reached = False

        for pnl_cum in cum_pnl:
            if pnl_cum > peak:
                peak = pnl_cum
            dd = peak - pnl_cum
            if dd >= max_dd:
                dd_breached = True
                break
            if pnl_cum >= target_pnl:
                target_reached = True
                break

        if target_reached:
            hits_target += 1
        elif dd_breached:
            hits_dd += 1
        else:
            neither += 1

    prob_target = hits_target / n_sims * 100
    prob_dd = hits_dd / n_sims * 100
    prob_neither = neither / n_sims * 100

    print(f"\n --- RESULTATS MONTE CARLO ({n_sims:,} sims, {n_trades_sim} trades/an) ---")
    print(f"   Atteint target ${target_pnl:,.0f}: {prob_target:.1f}% ({hits_target}/{n_sims})")
    print(f"   Touche DD ${max_dd:,.0f}: {prob_dd:.1f}% ({hits_dd}/{n_sims})")
    print(f"   Ni l'un ni l'autre: {prob_neither:.1f}% ({neither}/{n_sims})")
    print(f"   PnL final moyen: ${final_pnls.mean():,.0f}")
    print(f"   PnL final median: ${np.median(final_pnls):,.0f}")
    print(f"   PnL final P10/P90: ${np.percentile(final_pnls, 10):,.0f} / ${np.percentile(final_pnls, 90):,.0f}")

    # Also simulate with more realistic lower target
    daily_target = 300
    monthly_target = daily_target * 21  # ~$6,300/mois
    print(f"\n --- SCENARIO MENSUEL (target ${monthly_target:,.0f}/mois, {n_trades_sim // 12} trades/mois) ---")
    n_trades_month = max(n_trades_sim // 12, 1)
    month_hits = 0
    month_dd = 0
    for sim in range(n_sims):
        trade_seq = rng.choice(pnls, size=n_trades_month, replace=True)
        cum_pnl = np.cumsum(trade_seq)
        peak = 0.0
        dd_ok = True
        for p in cum_pnl:
            if p > peak:
                peak = p
            if peak - p >= max_dd:
                dd_ok = False
                break
        if dd_ok and cum_pnl[-1] >= monthly_target:
            month_hits += 1
        elif not dd_ok:
            month_dd += 1

    print(f"   Mois profitables (>=${monthly_target:,.0f}): {month_hits/n_sims*100:.1f}%")
    print(f"   Mois avec DD breach: {month_dd/n_sims*100:.1f}%")

    print("\n --- VERDICT 4A ---")
    print(f" P(target) = {prob_target:.1f}% (GO > 60%)")
    if prob_target > 60:
        print(" >>> VERDICT: GO - viable pour propfirm")
    elif prob_target > 30:
        print(" >>> VERDICT: MARGINAL - possible mais risque")
    else:
        print(" >>> VERDICT: STOP - insuffisant pour propfirm")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Validation quant du confidence score NQ_YM")
    parser.add_argument("--temporal", action="store_true", help="1A: distribution temporelle")
    parser.add_argument("--decompose", action="store_true", help="1B: decomposition composantes")
    parser.add_argument("--isoos", action="store_true", help="3A: split IS/OOS")
    parser.add_argument("--permutation", action="store_true", help="3B: test de permutation")
    parser.add_argument("--sensitivity", action="store_true", help="2A: robustesse parametrique")
    parser.add_argument("--walkforward", action="store_true", help="3C: walk-forward analysis")
    parser.add_argument("--propfirm", action="store_true", help="4A: Monte Carlo propfirm")
    parser.add_argument("--all", action="store_true", help="Lancer tout")
    parser.add_argument("--perms", type=int, default=1000, help="Nombre de permutations (defaut: 1000)")
    args = parser.parse_args()

    # Default: if no flag, run all
    run_all = args.all or not any([
        args.temporal, args.decompose, args.isoos, args.permutation,
        args.sensitivity, args.walkforward, args.propfirm,
    ])

    print("=" * 100)
    print(" VALIDATION QUANT - CONFIDENCE SCORE NQ_YM")
    print(f" Config: OLS={OLS_WINDOW} ZW={ZSCORE_WINDOW} entry={Z_ENTRY} exit={Z_EXIT} stop={Z_STOP} conf>={MIN_CONFIDENCE}%")
    print("=" * 100)

    # Load data once
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    print(f" Data: {len(aligned.df):,} barres, {aligned.df.index[0]} -> {aligned.df.index[-1]}")

    oos_pnls = None

    if args.temporal or run_all:
        section_temporal(aligned)

    if args.decompose or run_all:
        section_decompose(aligned)

    if args.isoos or run_all:
        section_isoos(aligned)

    if args.permutation or run_all:
        section_permutation(aligned, n_perms=args.perms)

    if args.sensitivity or run_all:
        section_sensitivity(aligned)

    if args.walkforward or run_all:
        oos_pnls = section_walkforward(aligned)

    if args.propfirm or run_all:
        section_propfirm(aligned, oos_pnls=oos_pnls)

    print("\n" + "=" * 100)
    print(" FIN DE LA VALIDATION")
    print("=" * 100)


if __name__ == "__main__":
    main()
