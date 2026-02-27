"""Etape 3b — MaxDD + Diversite parametrique + Selection NQ/RTY OLS.

Full backtest sur 40 candidats de l'etape 2.
Selection par diversite parametrique (OLS, ZW, profile, window) + tiers de risque.
Overlap trading-day pour information seulement (structurellement eleve NQ/RTY).

Usage:
    python scripts/step3_maxdd_overlap_NQ_RTY.py
"""

import sys
import time as time_mod
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

# ======================================================================
# Constants NQ/RTY
# ======================================================================
_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.50
INITIAL_CAPITAL = 100_000.0
FLAT_MIN = 930
BARS_PER_DAY = 264
BARS_PER_YEAR = BARS_PER_DAY * 252

CONF_WEIGHTS = ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00)

METRIC_PROFILES = {
    "p12_64":    MetricsConfig(adf_window=12, hurst_window=64,  halflife_window=12, correlation_window=6),
    "p16_80":    MetricsConfig(adf_window=16, hurst_window=80,  halflife_window=16, correlation_window=8),
    "p18_96":    MetricsConfig(adf_window=18, hurst_window=96,  halflife_window=18, correlation_window=9),
    "p20_100":   MetricsConfig(adf_window=20, hurst_window=100, halflife_window=20, correlation_window=10),
    "p24_128":   MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "p28_144":   MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p30_160":   MetricsConfig(adf_window=30, hurst_window=160, halflife_window=30, correlation_window=15),
    "p36_192":   MetricsConfig(adf_window=36, hurst_window=192, halflife_window=36, correlation_window=18),
    "p36_96":    MetricsConfig(adf_window=36, hurst_window=96,  halflife_window=36, correlation_window=9),
    "p42_224":   MetricsConfig(adf_window=42, hurst_window=224, halflife_window=42, correlation_window=21),
    "p48_128":   MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
    "p48_96":    MetricsConfig(adf_window=48, hurst_window=96,  halflife_window=48, correlation_window=9),
    "p48_256":   MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "p60_320":   MetricsConfig(adf_window=60, hurst_window=320, halflife_window=60, correlation_window=30),
    "p24_256":   MetricsConfig(adf_window=24, hurst_window=256, halflife_window=24, correlation_window=24),
    "p30_256":   MetricsConfig(adf_window=30, hurst_window=256, halflife_window=30, correlation_window=24),
    "p18_192":   MetricsConfig(adf_window=18, hurst_window=192, halflife_window=18, correlation_window=18),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
    "08:00-14:00": (480, 840),
    "08:00-12:00": (480, 720),
    "06:00-12:00": (360, 720),
}


def build_signal(aligned, minutes, cfg):
    """Build signal + beta + trade day arrays."""
    est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    zw = cfg["zw"]
    mu = spread.rolling(zw).mean()
    sigma = spread.rolling(zw).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
    zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

    raw = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

    profile_cfg = METRIC_PROFILES[cfg["profile"]]
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
    confidence = compute_confidence(metrics, replace_conf(cfg["conf"])).values

    sig = _apply_conf_filter_numba(raw, confidence, cfg["conf"])

    entry_start, entry_end = WINDOWS_MAP[cfg["window"]]
    sig = apply_window_filter_numba(sig, minutes, entry_start, entry_end, FLAT_MIN)

    return sig, beta


def replace_conf(min_conf):
    """Build ConfidenceConfig with custom min_confidence."""
    return ConfidenceConfig(
        w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00,
        min_confidence=min_conf,
    )


def get_trading_days(sig, dates):
    """Extract set of trading days where there's an active position."""
    in_position = sig != 0
    active_dates = dates[in_position]
    return set(active_dates.date)


def compute_yearly(equity, dates, pnl_total):
    """Compute yearly PnL from equity curve."""
    years = sorted(set(d.year for d in dates))
    yearly = {}
    for y in years:
        mask = np.array([d.year == y for d in dates])
        if mask.sum() < 10:
            continue
        eq_year = equity[mask]
        yearly_pnl = eq_year[-1] - eq_year[0]
        yearly[y] = yearly_pnl
    return yearly


def main():
    t_start = time_mod.time()

    print("=" * 120)
    print("  ETAPE 3 — MaxDD + OVERLAP + SELECTION NQ/RTY OLS")
    print("=" * 120)

    # Load data
    print("\nLoading NQ/RTY data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    dates = idx.to_pydatetime()
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years:.1f} years")

    # Load 40 candidates from step 2
    candidates_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step2_candidates_40.csv"
    cands = pd.read_csv(candidates_path)
    print(f"Candidats: {len(cands)}")

    # ================================================================
    # PHASE 1: Full backtest on all 40 candidates
    # ================================================================
    print(f"\n{'='*120}")
    print("  PHASE 1: FULL BACKTEST — 40 candidats")
    print(f"{'='*120}")

    results = []
    signals_map = {}  # store signals for overlap

    for i, row in cands.iterrows():
        cfg = {
            "ols": int(row["ols_window"]),
            "zw": int(row["zscore_window"]),
            "profile": row["profil"],
            "window": row["window"],
            "z_entry": row["z_entry"],
            "z_exit": row["z_exit"],
            "z_stop": row["z_stop"],
            "conf": row["min_confidence"],
        }

        label = f"#{i+1}_{cfg['profile']}_{cfg['ols']}_{cfg['zw']}_{cfg['window']}"

        sig, beta = build_signal(aligned, minutes, cfg)

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            MULT_A, MULT_B, TICK_A, TICK_B,
            SLIPPAGE, COMMISSION, INITIAL_CAPITAL,
        )

        # MaxDD
        equity = bt["equity"]
        running_max = np.maximum.accumulate(equity)
        max_dd = (equity - running_max).min()

        # Sharpe
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(equity) / equity[:-1]
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(BARS_PER_YEAR) if np.std(returns) > 0 else 0.0

        # Calmar
        calmar = bt["pnl"] / abs(max_dd) if max_dd != 0 else 0.0

        # Yearly
        yearly = compute_yearly(equity, dates, bt["pnl"])

        # Consecutive losses
        trades = bt.get("trade_pnls", [])
        max_consec_loss = 0
        curr_streak = 0
        for t_pnl in trades:
            if t_pnl < 0:
                curr_streak += 1
                max_consec_loss = max(max_consec_loss, curr_streak)
            else:
                curr_streak = 0

        # Trading days for overlap
        trading_days = get_trading_days(sig, idx)
        signals_map[label] = {
            "sig": sig,
            "days": trading_days,
            "cfg": cfg,
        }

        # Long/short balance
        long_trades = sum(1 for s in np.diff(sig) if s > 0)
        short_trades = sum(1 for s in np.diff(sig) if s < 0)
        total_entries = long_trades + short_trades
        long_pct = long_trades / total_entries * 100 if total_entries > 0 else 50

        neg_years = [y for y, pnl in yearly.items() if pnl < 0]

        results.append({
            "label": label,
            "ols": cfg["ols"],
            "zw": cfg["zw"],
            "profile": cfg["profile"],
            "window": cfg["window"],
            "z_entry": cfg["z_entry"],
            "z_exit": cfg["z_exit"],
            "z_stop": cfg["z_stop"],
            "conf": cfg["conf"],
            "trades": bt["trades"],
            "wr": bt["win_rate"],
            "pnl": bt["pnl"],
            "pf": bt["profit_factor"],
            "sharpe": round(float(sharpe), 2),
            "max_dd": round(float(max_dd), 0),
            "calmar": round(float(calmar), 2),
            "max_consec_loss": max_consec_loss,
            "long_pct": round(long_pct, 1),
            "neg_years": len(neg_years),
            "neg_years_list": neg_years,
            "yearly": yearly,
            "avg_pnl": bt["avg_pnl_trade"],
        })

    # Sort by composite score
    df_r = pd.DataFrame(results)

    print(f"\n  {'#':>3} {'Label':<45} {'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} "
          f"{'Sharpe':>6} {'MaxDD':>9} {'Calm':>5} {'Strk':>4} {'L%':>5} {'-Yr':>3}")
    print(f"  {'-'*120}")

    for _, r in df_r.sort_values("pf", ascending=False).iterrows():
        dd_flag = " DANGER" if r["max_dd"] < -8000 else (" WARN" if r["max_dd"] < -5000 else "")
        print(f"  {r['label']:<48} {r['trades']:>4} {r['wr']:>5.1f} ${r['pnl']:>8,.0f} "
              f"{r['pf']:>5.2f} {r['sharpe']:>6.2f} ${r['max_dd']:>8,.0f} {r['calmar']:>5.2f} "
              f"{r['max_consec_loss']:>4} {r['long_pct']:>5.1f} {r['neg_years']:>3}{dd_flag}")

    # ================================================================
    # PHASE 2: Overlap analysis
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  PHASE 2: OVERLAP ANALYSIS")
    print(f"{'='*120}")

    labels = list(signals_map.keys())
    n = len(labels)
    overlap_matrix = np.zeros((n, n))

    for i in range(n):
        days_i = signals_map[labels[i]]["days"]
        for j in range(n):
            days_j = signals_map[labels[j]]["days"]
            if len(days_i) == 0 or len(days_j) == 0:
                overlap_matrix[i, j] = 0
            else:
                common = days_i & days_j
                overlap_matrix[i, j] = len(common) / min(len(days_i), len(days_j)) * 100

    # Find high-overlap pairs
    print("\n  Paires avec overlap > 50%:")
    print(f"  {'Config A':<48} {'Config B':<48} {'Overlap%':>8}")
    print(f"  {'-'*110}")

    high_overlap_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if overlap_matrix[i, j] > 50:
                high_overlap_pairs.append((labels[i], labels[j], overlap_matrix[i, j]))

    high_overlap_pairs.sort(key=lambda x: -x[2])
    for a, b, ov in high_overlap_pairs[:30]:
        print(f"  {a:<48} {b:<48} {ov:>7.1f}%")
    if len(high_overlap_pairs) > 30:
        print(f"  ... et {len(high_overlap_pairs) - 30} paires de plus")

    print(f"\n  Total: {len(high_overlap_pairs)} paires avec overlap > 50% sur {n*(n-1)//2} paires possibles")

    # ================================================================
    # PHASE 3: Risk tier classification
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  PHASE 3: CLASSIFICATION PAR TIER DE RISQUE")
    print(f"{'='*120}")

    df_r["tier"] = pd.cut(
        -df_r["max_dd"],
        bins=[0, 5000, 7000, float("inf")],
        labels=["SAFE", "WARN", "DANGER"],
    )

    for tier in ["SAFE", "WARN", "DANGER"]:
        tier_df = df_r[df_r["tier"] == tier].sort_values("pf", ascending=False)
        label_map = {"SAFE": "MaxDD < $5K (propfirm OK)", "WARN": "MaxDD $5K-$7K (micro x2)", "DANGER": "MaxDD > $7K (micro only)"}
        print(f"\n  --- {tier}: {label_map[tier]} ({len(tier_df)} configs) ---")
        if len(tier_df) == 0:
            print("  (aucune)")
            continue
        print(f"  {'Label':<48} {'Trd':>4} {'WR%':>5} {'PnL':>9} {'PF':>5} "
              f"{'Shrp':>6} {'MaxDD':>9} {'Calm':>5} {'-Yr':>3}")
        for _, r in tier_df.iterrows():
            print(f"  {r['label']:<48} {r['trades']:>4} {r['wr']:>5.1f} ${r['pnl']:>8,.0f} "
                  f"{r['pf']:>5.2f} {r['sharpe']:>6.2f} ${r['max_dd']:>8,.0f} {r['calmar']:>5.2f} "
                  f"{r['neg_years']:>3}")

    # ================================================================
    # PHASE 4: Scoring + greedy parametric diversity selection
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  PHASE 4: SELECTION PAR DIVERSITE PARAMETRIQUE (target 8)")
    print(f"{'='*120}")

    # Composite score: MaxDD weighs 25%, higher than before
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    score_features = df_r[["pf", "pnl", "wr", "sharpe", "calmar"]].copy()
    score_features["dd_norm"] = -df_r["max_dd"]
    score_features["neg_yr_penalty"] = -df_r["neg_years"]
    score_features["balance"] = -abs(df_r["long_pct"] - 50)

    features_norm = pd.DataFrame(
        scaler.fit_transform(score_features),
        columns=score_features.columns,
        index=score_features.index,
    )

    df_r["score"] = (
        features_norm["pf"] * 0.20 +
        features_norm["pnl"] * 0.10 +
        features_norm["wr"] * 0.10 +
        features_norm["sharpe"] * 0.10 +
        features_norm["calmar"] * 0.10 +
        features_norm["dd_norm"] * 0.25 +
        features_norm["neg_yr_penalty"] * 0.10 +
        features_norm["balance"] * 0.05
    )

    # Parameter distance: different params = more diverse
    PARAM_COLS = ["ols", "zw", "profile", "window", "z_entry", "z_exit", "z_stop", "conf"]

    def param_distance(row_a, row_b):
        """Count number of different parameters (0-8)."""
        dist = 0
        for col in PARAM_COLS:
            if row_a[col] != row_b[col]:
                dist += 1
        return dist

    # Greedy selection: maximize score + parameter diversity
    # At each step, pick the config that maximizes: score + diversity_bonus
    # diversity_bonus = min_param_distance_to_selected / 8
    selected_indices = []
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # First pick: best overall score
    remaining = df_r.sort_values("score", ascending=False).index.tolist()
    selected_indices.append(remaining[0])
    remaining.remove(remaining[0])

    TARGET = 8

    while len(selected_indices) < TARGET and len(remaining) > 0:
        best_combined = -1
        best_idx = None
        for cand_idx in remaining:
            cand_row = df_r.loc[cand_idx]
            base_score = cand_row["score"]

            # Min param distance to any already selected
            min_dist = 8
            for sel_idx in selected_indices:
                sel_row = df_r.loc[sel_idx]
                d = param_distance(cand_row, sel_row)
                min_dist = min(min_dist, d)

            # Diversity bonus: 0 to 1 (normalized by max possible distance 8)
            diversity_bonus = min_dist / 8.0

            # Combined: 60% score + 40% diversity
            combined = 0.60 * base_score + 0.40 * diversity_bonus

            if combined > best_combined:
                best_combined = combined
                best_idx = cand_idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    selected = df_r.loc[selected_indices].copy()

    # Assign rank by tier then score
    tier_order = {"SAFE": 0, "WARN": 1, "DANGER": 2}
    selected["tier_rank"] = selected["tier"].map(tier_order)
    selected = selected.sort_values(["tier_rank", "score"], ascending=[True, False])

    # Print final selection
    print(f"\n  {len(selected)} configs selectionnees par diversite parametrique")

    print(f"\n  {'#':>3} {'Tier':<7} {'OLS':>5} {'ZW':>3} {'Profil':<10} {'Window':<12} "
          f"{'ze':>5} {'zx':>5} {'zs':>4} {'c':>3} {'Trd':>4} {'WR%':>5} "
          f"{'PnL':>9} {'PF':>5} {'Shrp':>5} {'MaxDD':>9} {'Calm':>5} {'L%':>5} {'Scr':>5}")
    print(f"  {'-'*130}")

    for rank, (_, r) in enumerate(selected.iterrows(), 1):
        print(f"  {rank:>3} {r['tier']:<7} {r['ols']:>5} {r['zw']:>3} {r['profile']:<10} {r['window']:<12} "
              f"{r['z_entry']:>5.2f} {r['z_exit']:>5.2f} {r['z_stop']:>4.1f} {int(r['conf']):>3} "
              f"{r['trades']:>4} {r['wr']:>5.1f} ${r['pnl']:>8,.0f} {r['pf']:>5.2f} "
              f"{r['sharpe']:>5.2f} ${r['max_dd']:>8,.0f} {r['calmar']:>5.2f} "
              f"{r['long_pct']:>5.1f} {r['score']:>5.3f}")

    # Parameter diversity matrix
    print("\n  --- Matrice de distance parametrique (sur 8) ---")
    print(f"  {'':>5}", end="")
    for j in range(len(selected)):
        print(f" {f'#{j+1}':>4}", end="")
    print()
    for i, (idx_i, row_i) in enumerate(selected.iterrows()):
        print(f"  #{i+1:>3}", end="")
        for j, (idx_j, row_j) in enumerate(selected.iterrows()):
            if i == j:
                print(f"  {'--':>3}", end="")
            else:
                d = param_distance(row_i, row_j)
                print(f" {d:>4}", end="")
        print()

    # Overlap between selected (informational)
    print("\n  --- Overlap trading-day entre selectionnees (informatif) ---")
    print(f"  {'':>5}", end="")
    for j in range(len(selected)):
        print(f" {f'#{j+1}':>5}", end="")
    print()
    for i, (idx_i, row_i) in enumerate(selected.iterrows()):
        print(f"  #{i+1:>3}", end="")
        mi = label_to_idx[row_i["label"]]
        for j, (idx_j, row_j) in enumerate(selected.iterrows()):
            mj = label_to_idx[row_j["label"]]
            if i == j:
                print(f"  {'---':>4}", end="")
            else:
                ov = overlap_matrix[mi, mj]
                print(f" {ov:>5.0f}", end="")
        print(" %")

    # Yearly decomposition
    print("\n  --- Decomposition annuelle ---")
    all_years = sorted(set(y for r in results for y in r["yearly"].keys()))
    header = f"  {'#':>3} {'Tier':<6} {'Label':<35}"
    for y in all_years:
        header += f" {y:>8}"
    header += f" {'NegYrs':>6}"
    print(header)
    print(f"  {'-'*(50 + 9*len(all_years) + 6)}")

    for rank, (_, r) in enumerate(selected.iterrows(), 1):
        full_r = [res for res in results if res["label"] == r["label"]][0]
        line = f"  {rank:>3} {r['tier']:<6} {r['label'][:35]:<35}"
        neg = 0
        for y in all_years:
            pnl_y = full_r["yearly"].get(y, 0)
            if pnl_y < 0:
                neg += 1
                line += f" ${pnl_y:>7,.0f}*"
            else:
                line += f" ${pnl_y:>7,.0f}"
        line += f" {neg:>5}"
        print(line)

    # Diversity summary
    print("\n  --- Diversite parametrique ---")
    for col in ["ols", "zw", "profile", "window", "z_entry", "z_exit", "z_stop", "conf"]:
        vals = selected[col].unique()
        print(f"  {col}: {len(vals)} unique -> {sorted(vals) if col not in ['profile', 'window'] else list(vals)}")

    # Tier summary
    print("\n  --- Repartition par tier ---")
    for tier in ["SAFE", "WARN", "DANGER"]:
        tier_sel = selected[selected["tier"] == tier]
        if len(tier_sel) > 0:
            avg_dd = tier_sel["max_dd"].mean()
            avg_pf = tier_sel["pf"].mean()
            print(f"  {tier:>7}: {len(tier_sel)} configs, avg MaxDD ${avg_dd:,.0f}, avg PF {avg_pf:.2f}")

    # Save
    out_path = PROJECT_ROOT / "output" / "NQ_RTY" / "step3_top8.csv"
    cols_save = [c for c in selected.columns if c not in ["neg_years_list", "yearly", "tier_rank"]]
    selected[cols_save].to_csv(out_path, index=False)
    print(f"\n  Sauvegarde: {out_path}")

    elapsed = time_mod.time() - t_start
    print(f"\n  Etape 3b complete en {elapsed:.0f}s")


if __name__ == "__main__":
    main()
