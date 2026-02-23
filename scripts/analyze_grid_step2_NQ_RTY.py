"""Etape 2 — Re-analyse grid NQ/RTY OLS (14.4M combos).

Sweet spots, heatmaps, filtres diversifies, selection candidats.
Poids confirmes: ADF 50%, Hurst 30%, Corr 20%, HL 0%.

Usage:
    python scripts/analyze_grid_step2_NQ_RTY.py
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "output" / "NQ_RTY"

# ======================================================================
# Load data
# ======================================================================

def load_grid():
    """Load filtered grid (PF>1.3, trades>150)."""
    path = OUTPUT_DIR / "grid_refined_ols_filtered.csv"
    df = pd.read_csv(path)
    # Dedup z_stop variants (many identical results)
    df["key"] = (
        df["ols_window"].astype(str) + "_" + df["zscore_window"].astype(str) + "_" +
        df["profil"] + "_" + df["window"] + "_" + df["z_entry"].astype(str) + "_" +
        df["z_exit"].astype(str) + "_" + df["min_confidence"].astype(str)
    )
    return df


# ======================================================================
# Section A: Sweet spots par dimension
# ======================================================================

def section_a(df):
    print(f"\n{'='*110}")
    print("  SECTION A: SWEET SPOTS PAR DIMENSION")
    print(f"{'='*110}")

    dims = {
        "ols_window": "OLS",
        "zscore_window": "ZW",
        "profil": "Profil",
        "window": "Window",
        "z_entry": "z_entry",
        "z_exit": "z_exit",
        "z_stop": "z_stop",
        "min_confidence": "Confidence",
    }

    sweet_spots = {}

    for col, label in dims.items():
        agg = df.groupby(col).agg(
            count=("pnl", "count"),
            avg_pf=("profit_factor", "mean"),
            med_pf=("profit_factor", "median"),
            avg_pnl=("pnl", "mean"),
            avg_trades=("trades", "mean"),
            avg_wr=("win_rate", "mean"),
            pct_pf2=("profit_factor", lambda x: (x >= 2.0).mean() * 100),
            std_pf=("profit_factor", "std"),
        ).sort_values("avg_pf", ascending=False)

        # Best value
        best_val = agg.index[0]
        best_pf = agg.iloc[0]["avg_pf"]
        sweet_spots[label] = (best_val, best_pf)

        print(f"\n  --- {label} ---")
        print(f"  {'Value':<14} {'Count':>7} {'PF moy':>7} {'PF med':>7} {'PnL moy':>10} "
              f"{'Trades':>7} {'WR%':>6} {'%PF>=2':>7} {'StdPF':>6}")
        print(f"  {'-'*85}")

        for val, r in agg.iterrows():
            marker = " <<<" if val == best_val else ""
            val_str = f"{val}" if isinstance(val, str) else f"{val}"
            print(f"  {val_str:<14} {r['count']:>7.0f} {r.avg_pf:>7.2f} {r.med_pf:>7.2f} "
                  f"${r.avg_pnl:>9,.0f} {r.avg_trades:>7.0f} {r.avg_wr:>5.1f}% "
                  f"{r.pct_pf2:>6.1f}% {r.std_pf:>6.2f}{marker}")

    # Summary
    print(f"\n\n  {'='*80}")
    print(f"  RESUME SECTION A — Sweet spots identifies")
    print(f"  {'='*80}")
    print(f"\n  {'Dimension':<14} {'Optimal':>14} {'PF moy':>7} {'Notes'}")
    print(f"  {'-'*70}")
    for label, (val, pf) in sweet_spots.items():
        # Add context notes
        notes = ""
        if label == "OLS":
            notes = "(~35 jours)" if val == 9240 else f"(~{val//264} jours)"
        elif label == "ZW":
            notes = f"({val*5}min)"
        elif label == "z_stop":
            notes = "(peu d'impact, zone plate)"
        print(f"  {label:<14} {str(val):>14} {pf:>7.2f} {notes}")

    return sweet_spots


# ======================================================================
# Section B: Cross-dimension heatmaps
# ======================================================================

def section_b(df):
    print(f"\n\n{'='*110}")
    print("  SECTION B: CROSS-DIMENSION HEATMAPS")
    print(f"{'='*110}")

    heatmaps = [
        ("ols_window", "zscore_window", "OLS x ZW"),
        ("z_entry", "z_exit", "z_entry x z_exit"),
        ("profil", "window", "Profil x Window"),
        ("min_confidence", "profil", "Confidence x Profil"),
    ]

    best_combos = {}

    for dim1, dim2, title in heatmaps:
        print(f"\n  --- {title} (PF moyen) ---")

        pivot = df.groupby([dim1, dim2])["profit_factor"].mean().unstack(fill_value=0)

        # Print heatmap
        col_width = max(8, max(len(str(c)) for c in pivot.columns) + 1)
        header = f"  {'':>14}"
        for c in pivot.columns:
            header += f"{str(c):>{col_width}}"
        print(header)
        print(f"  {'-'*(14 + col_width * len(pivot.columns))}")

        for idx_val, row in pivot.iterrows():
            line = f"  {str(idx_val):>14}"
            for c in pivot.columns:
                val = row[c]
                if val == 0:
                    line += f"{'---':>{col_width}}"
                elif val >= 1.8:
                    line += f"{val:>{col_width}.2f}*"
                else:
                    line += f"{val:>{col_width}.2f}"
            print(line)

        # Best combo
        flat = df.groupby([dim1, dim2]).agg(
            avg_pf=("profit_factor", "mean"),
            count=("pnl", "count"),
            avg_pnl=("pnl", "mean"),
        )
        if len(flat) > 0:
            best_idx = flat["avg_pf"].idxmax()
            best_r = flat.loc[best_idx]
            best_combos[title] = (best_idx, best_r["avg_pf"])
            print(f"\n  Best: {dim1}={best_idx[0]}, {dim2}={best_idx[1]} -> "
                  f"PF={best_r.avg_pf:.2f}, {int(best_r['count'])} configs, PnL=${best_r.avg_pnl:,.0f}")

    # Summary
    print(f"\n\n  {'='*80}")
    print(f"  RESUME SECTION B — Meilleures combinaisons")
    print(f"  {'='*80}")
    for title, (combo, pf) in best_combos.items():
        print(f"  {title:<25} -> {combo} (PF={pf:.2f})")

    return best_combos


# ======================================================================
# Section C: Filtres de selection
# ======================================================================

def section_c(df):
    print(f"\n\n{'='*110}")
    print("  SECTION C: FILTRES DE SELECTION")
    print(f"{'='*110}")

    # Dedup z_stop for counting (avoid inflating counts)
    df_dedup = df.sort_values("profit_factor", ascending=False).drop_duplicates("key", keep="first")

    filters = {
        "STRICT":    {"pf_min": 1.8, "trades_min": 150, "wr_min": 64, "avg_pnl_min": 0},
        "VOLUME":    {"pf_min": 1.4, "trades_min": 300, "wr_min": 0, "avg_pnl_min": 0},
        "SNIPER":    {"pf_min": 2.0, "trades_min": 100, "wr_min": 0, "avg_pnl_min": 0},
        "EQUILIBRE": {"pf_min": 1.5, "trades_min": 180, "wr_min": 0, "avg_pnl_min": 100},
        "PROPFIRM":  {"pf_min": 1.6, "trades_min": 150, "wr_min": 63, "avg_pnl_min": 80},
    }

    all_selected = {}

    for fname, f in filters.items():
        mask = (
            (df_dedup["profit_factor"] >= f["pf_min"]) &
            (df_dedup["trades"] >= f["trades_min"]) &
            (df_dedup["win_rate"] >= f["wr_min"]) &
            (df_dedup["avg_pnl_trade"] >= f["avg_pnl_min"])
        )
        sub = df_dedup[mask].sort_values("profit_factor", ascending=False)
        all_selected[fname] = sub

        print(f"\n  --- {fname} (PF>={f['pf_min']}, trades>={f['trades_min']}, "
              f"WR>={f['wr_min']}%, avg>=${f['avg_pnl_min']}) ---")
        print(f"  {len(sub)} configs retenues")

        if len(sub) > 0:
            print(f"\n  {'#':>3} {'OLS':>5} {'ZW':>3} {'Profil':<10} {'Window':<12} "
                  f"{'ze':>5} {'zx':>5} {'zs':>4} {'c':>3} {'Trd':>5} {'WR%':>5} "
                  f"{'PnL':>9} {'PF':>5} {'$/t':>5}")
            print(f"  {'-'*100}")

            for i, (_, r) in enumerate(sub.head(15).iterrows()):
                print(f"  {i+1:>3} {int(r.ols_window):>5} {int(r.zscore_window):>3} "
                      f"{r.profil:<10} {r.window:<12} {r.z_entry:>5.2f} {r.z_exit:>5.2f} "
                      f"{r.z_stop:>4.1f} {int(r.min_confidence):>3} {int(r.trades):>5} "
                      f"{r.win_rate:>5.1f} ${int(r.pnl):>8} {r.profit_factor:>5.2f} "
                      f"${r.avg_pnl_trade:>4.0f}")
            if len(sub) > 15:
                print(f"  ... et {len(sub)-15} de plus")

    # Cross-filter: configs qui passent 3+ filtres
    print(f"\n\n  --- CONFIGS MULTI-FILTRES (passent >= 3 filtres) ---")
    config_scores = {}
    for fname, sub in all_selected.items():
        for key in sub["key"].values:
            config_scores[key] = config_scores.get(key, 0) + 1

    multi = {k: v for k, v in config_scores.items() if v >= 3}
    multi_keys = sorted(multi.keys(), key=lambda x: -multi[x])

    if multi_keys:
        print(f"  {len(multi_keys)} configs passent >= 3 filtres")
        print(f"\n  {'Filtres':>7} {'OLS':>5} {'ZW':>3} {'Profil':<10} {'Window':<12} "
              f"{'ze':>5} {'zx':>5} {'c':>3} {'Trd':>5} {'WR%':>5} {'PnL':>9} {'PF':>5}")
        print(f"  {'-'*95}")

        for key in multi_keys[:20]:
            row = df_dedup[df_dedup["key"] == key].iloc[0]
            n_filters = multi[key]
            print(f"  {n_filters:>4}/5   {int(row.ols_window):>5} {int(row.zscore_window):>3} "
                  f"{row.profil:<10} {row.window:<12} {row.z_entry:>5.2f} {row.z_exit:>5.2f} "
                  f"{int(row.min_confidence):>3} {int(row.trades):>5} {row.win_rate:>5.1f} "
                  f"${int(row.pnl):>8} {row.profit_factor:>5.2f}")
    else:
        print(f"  Aucune config ne passe 3+ filtres")

    # Summary
    print(f"\n\n  {'='*80}")
    print(f"  RESUME SECTION C — Volume par filtre")
    print(f"  {'='*80}")
    print(f"\n  {'Filtre':<12} {'Configs':>8} {'PF moy':>7} {'PnL moy':>10} {'Trades moy':>10}")
    print(f"  {'-'*55}")
    for fname, sub in all_selected.items():
        if len(sub) > 0:
            print(f"  {fname:<12} {len(sub):>8} {sub.profit_factor.mean():>7.2f} "
                  f"${sub.pnl.mean():>9,.0f} {sub.trades.mean():>10.0f}")
        else:
            print(f"  {fname:<12} {0:>8} {'---':>7} {'---':>10} {'---':>10}")
    print(f"  Multi (>=3)  {len(multi_keys):>8}")

    return all_selected, multi_keys


# ======================================================================
# Section D: Selection top candidats diversifies
# ======================================================================

def section_d(df, all_selected, multi_keys):
    print(f"\n\n{'='*110}")
    print("  SECTION D: SELECTION TOP CANDIDATS DIVERSIFIES")
    print(f"{'='*110}")

    df_dedup = df.sort_values("profit_factor", ascending=False).drop_duplicates("key", keep="first")

    # Pool: union of all filter results + multi-filter configs
    pool_keys = set()
    for fname, sub in all_selected.items():
        pool_keys.update(sub["key"].head(50).values)
    pool_keys.update(multi_keys)

    pool = df_dedup[df_dedup["key"].isin(pool_keys)].copy()
    print(f"\n  Pool initial: {len(pool)} configs candidates")

    # Score composite
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = pool[["profit_factor", "pnl", "win_rate", "trades"]].copy()
    features_norm = pd.DataFrame(
        scaler.fit_transform(features), columns=features.columns, index=features.index
    )
    pool["score"] = (
        features_norm["profit_factor"] * 0.30 +
        features_norm["pnl"] * 0.30 +
        features_norm["win_rate"] * 0.20 +
        features_norm["trades"] * 0.20
    )

    # Greedy diverse selection (max 40)
    div_cols = ["ols_window", "zscore_window", "profil", "window", "z_entry", "z_exit", "min_confidence"]
    selected_indices = []
    remaining = pool.copy()

    for i in range(40):
        if len(remaining) == 0:
            break
        if i == 0:
            best_idx = remaining["score"].idxmax()
        else:
            best_score = -1
            best_idx = None
            for idx in remaining.index:
                row = remaining.loc[idx]
                min_overlap = 7
                for sel_idx in selected_indices:
                    sel_row = pool.loc[sel_idx]
                    overlap = sum(1 for c in div_cols if row[c] == sel_row[c])
                    min_overlap = min(min_overlap, overlap)
                diversity = (7 - min_overlap) / 7
                combined = row["score"] * 0.5 + diversity * 0.5
                if combined > best_score:
                    best_score = combined
                    best_idx = idx
        selected_indices.append(best_idx)
        remaining = remaining.drop(best_idx)

    selected = pool.loc[selected_indices]

    print(f"  Selection finale: {len(selected)} configs diversifiees")

    # Diversity check
    print(f"\n  Diversite:")
    for col in div_cols:
        vals = selected[col].unique()
        print(f"    {col}: {len(vals)} valeurs uniques")

    # Print all selected
    print(f"\n  {'#':>3} {'OLS':>5} {'ZW':>3} {'Profil':<10} {'Window':<12} "
          f"{'ze':>5} {'zx':>5} {'zs':>4} {'c':>3} {'Trd':>5} {'WR%':>5} "
          f"{'PnL':>9} {'PF':>5} {'$/t':>5} {'Score':>5}")
    print(f"  {'-'*110}")

    for i, (_, r) in enumerate(selected.iterrows()):
        print(f"  {i+1:>3} {int(r.ols_window):>5} {int(r.zscore_window):>3} "
              f"{r.profil:<10} {r.window:<12} {r.z_entry:>5.2f} {r.z_exit:>5.2f} "
              f"{r.z_stop:>4.1f} {int(r.min_confidence):>3} {int(r.trades):>5} "
              f"{r.win_rate:>5.1f} ${int(r.pnl):>8} {r.profit_factor:>5.2f} "
              f"${r.avg_pnl_trade:>4.0f} {r.score:>5.3f}")

    # Pattern analysis
    print(f"\n  --- Patterns dominants ---")
    for col in ["ols_window", "zscore_window", "profil", "window"]:
        vc = selected[col].value_counts()
        top = vc.head(3)
        parts = [f"{v}({c})" for v, c in top.items()]
        print(f"  {col}: {', '.join(parts)}")

    # Summary
    print(f"\n\n  {'='*80}")
    print(f"  RESUME SECTION D — {len(selected)} candidats selectionnes")
    print(f"  {'='*80}")
    print(f"\n  PF: {selected.profit_factor.min():.2f} - {selected.profit_factor.max():.2f} "
          f"(moy {selected.profit_factor.mean():.2f})")
    print(f"  PnL: ${selected.pnl.min():,.0f} - ${selected.pnl.max():,.0f} "
          f"(moy ${selected.pnl.mean():,.0f})")
    print(f"  Trades: {int(selected.trades.min())} - {int(selected.trades.max())} "
          f"(moy {selected.trades.mean():.0f})")
    print(f"  WR: {selected.win_rate.min():.1f}% - {selected.win_rate.max():.1f}% "
          f"(moy {selected.win_rate.mean():.1f}%)")

    # Save to CSV
    out_path = OUTPUT_DIR / "step2_candidates_40.csv"
    selected.to_csv(out_path, index=False)
    print(f"\n  Sauvegardes: {out_path}")

    return selected


# ======================================================================
# Main
# ======================================================================

def main():
    t_start = time_mod.time()

    print("=" * 110)
    print("  ETAPE 2 — RE-ANALYSE GRID NQ/RTY OLS")
    print("  Poids confirmes: ADF 50%, Hurst 30%, Corr 20%, HL 0%")
    print("=" * 110)

    print("\nLoading filtered grid...")
    df = load_grid()
    print(f"Grid filtre: {len(df):,} lignes")

    df_dedup = df.sort_values("profit_factor", ascending=False).drop_duplicates("key", keep="first")
    print(f"Apres dedup z_stop: {len(df_dedup):,} configs uniques")
    print(f"PF range: {df.profit_factor.min():.2f} - {df.profit_factor.max():.2f}")
    print(f"Trades range: {int(df.trades.min())} - {int(df.trades.max())}")

    # Section A
    sweet_spots = section_a(df)

    # Section B
    best_combos = section_b(df)

    # Section C
    all_selected, multi_keys = section_c(df)

    # Section D
    candidates = section_d(df, all_selected, multi_keys)

    elapsed = time_mod.time() - t_start
    print(f"\n\nAnalyse complete en {elapsed:.0f}s")


if __name__ == "__main__":
    main()
