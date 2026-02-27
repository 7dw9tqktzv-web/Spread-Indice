"""Etape 3 — Analyse dimensionnelle du grid search Kalman NQ/RTY.

Reads the grid CSV and computes stats by each parameter dimension.
Selects top 5 configs by profile for validation.

Usage:
    python scripts/analyze_grid_NQ_RTY.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    csv_path = PROJECT_ROOT / "output" / "NQ_RTY" / "grid_kalman.csv"
    if not csv_path.exists():
        print(f"ERREUR: {csv_path} introuvable")
        return

    df = pd.read_csv(csv_path)
    print(f"Grid: {len(df)} rows loaded")

    # Rename columns for consistency
    col_map = {"alpha_ratio": "alpha", "profil": "profile"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Filter out infinite PF
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["profit_factor"])

    # Separate profitable
    prof = df[df["pnl"] > 0].copy()
    # Also filter out PF > 100 (artifact from near-zero loss)
    prof = prof[prof["profit_factor"] < 100].copy()
    print(f"Profitable (PF < 100): {len(prof)} / {len(df)} ({len(prof)/len(df)*100:.1f}%)")

    # ================================================================
    # DIMENSION ANALYSIS — average PF, PnL, trades for profitable only
    # ================================================================
    dims = ["alpha", "profile", "window", "z_entry", "z_exit", "z_stop", "min_confidence"]

    for dim in dims:
        if dim not in prof.columns:
            print(f"\n  SKIP: column '{dim}' not found")
            continue

        print(f"\n{'='*80}")
        print(f"  DIMENSION: {dim}")
        print(f"{'='*80}")

        grp = prof.groupby(dim).agg(
            count=("pnl", "size"),
            avg_pf=("profit_factor", "mean"),
            med_pf=("profit_factor", "median"),
            avg_pnl=("pnl", "mean"),
            max_pnl=("pnl", "max"),
            avg_trades=("trades", "mean"),
            avg_wr=("win_rate", "mean"),
        ).sort_values("avg_pf", ascending=False)

        print(f"  {'Value':<18} {'Count':>6} {'AvgPF':>7} {'MedPF':>7} {'AvgPnL':>10} {'MaxPnL':>10} {'AvgTrd':>7} {'AvgWR':>7}")
        print("  " + "-" * 78)
        for idx_val, row in grp.iterrows():
            label = str(idx_val)
            if dim == "alpha" and isinstance(idx_val, float):
                label = f"{idx_val:.1e}"
            print(f"  {label:<18} {row['count']:>6.0f} {row['avg_pf']:>7.3f} {row['med_pf']:>7.3f} "
                  f"${row['avg_pnl']:>9,.0f} ${row['max_pnl']:>9,.0f} {row['avg_trades']:>7.0f} {row['avg_wr']:>6.1f}%")

    # ================================================================
    # TOP CONFIGS — by different profiles
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  TOP CONFIGS BY PROFILE")
    print(f"{'='*120}")

    # Filter: min 50 trades for robustness
    robust = prof[prof["trades"] >= 50].copy()
    print(f"\n  Robust configs (trades >= 50): {len(robust)}")

    # 1. Volume champion: most trades with PF > 1.3
    vol = robust[robust["profit_factor"] > 1.3].sort_values("trades", ascending=False)
    print("\n  --- VOLUME (max trades, PF > 1.3) ---")
    _print_top(vol, 10)

    # 2. Quality champion: highest PF with trades >= 80
    qual = robust[robust["trades"] >= 80].sort_values("profit_factor", ascending=False)
    print("\n  --- QUALITY (max PF, trades >= 80) ---")
    _print_top(qual, 10)

    # 3. Balanced: best PnL * PF product with trades >= 100
    bal = robust[robust["trades"] >= 100].copy()
    bal["score"] = bal["pnl"] * bal["profit_factor"]
    bal = bal.sort_values("score", ascending=False)
    print("\n  --- BALANCED (PnL*PF, trades >= 100) ---")
    _print_top(bal, 10)

    # 4. Sniper: best avg_pnl_trade with trades >= 50
    snp = robust.sort_values("avg_pnl_trade", ascending=False)
    print("\n  --- SNIPER (max avg PnL/trade, trades >= 50) ---")
    _print_top(snp, 10)

    # 5. PropFirm: PF > 1.5, trades >= 60, low risk proxy (high WR)
    pf = robust[(robust["profit_factor"] > 1.5) & (robust["trades"] >= 60)].copy()
    pf = pf.sort_values("win_rate", ascending=False)
    print("\n  --- PROPFIRM (PF > 1.5, trades >= 60, max WR) ---")
    _print_top(pf, 10)

    # ================================================================
    # SWEET SPOT ANALYSIS — 2D heatmaps (alpha x z_entry)
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  SWEET SPOT: alpha x z_entry (avg PF, min 20 configs)")
    print(f"{'='*120}")

    pivot = prof.groupby(["alpha", "z_entry"]).agg(
        avg_pf=("profit_factor", "mean"),
        count=("pnl", "size"),
    )
    pivot = pivot[pivot["count"] >= 20]

    if not pivot.empty:
        piv_pf = pivot["avg_pf"].unstack(level="z_entry")
        alphas = sorted(piv_pf.index.unique())
        entries = sorted(piv_pf.columns)

        # Print header
        header = f"  {'alpha':<10}"
        for ze in entries:
            header += f" {ze:>6.3f}"
        print(header)
        print("  " + "-" * (10 + 7 * len(entries)))

        for a in alphas:
            row_str = f"  {a:<10.1e}"
            for ze in entries:
                val = piv_pf.loc[a, ze] if ze in piv_pf.columns and a in piv_pf.index else np.nan
                if pd.isna(val):
                    row_str += f" {'--':>6}"
                else:
                    row_str += f" {val:>6.3f}"
            print(row_str)

    # ================================================================
    # FINAL TOP 5 SELECTION (for validate script)
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  FINAL TOP 5 SELECTION FOR VALIDATION")
    print(f"{'='*120}")

    selected = []

    # Pick best from each profile, ensuring diversity
    profiles_order = [
        ("Balanced", bal),
        ("Quality", qual),
        ("Volume", vol),
        ("Sniper", snp),
        ("PropFirm", pf),
    ]

    used_keys = set()
    for profile_name, df_pool in profiles_order:
        if df_pool is None or df_pool.empty:
            print(f"\n  {profile_name}: NO CANDIDATE")
            continue

        for _, row in df_pool.iterrows():
            # Unique key to avoid duplicates
            key = (row.get("alpha", 0), row["z_entry"], row["z_exit"], row["z_stop"],
                   row["min_confidence"], row.get("profile", ""), row.get("window", ""))
            if key not in used_keys:
                used_keys.add(key)
                selected.append((profile_name, row))
                break

    print(f"\n  Selected {len(selected)} configs:\n")
    for i, (pname, row) in enumerate(selected):
        print(f"  [{i+1}] {pname}")
        alpha_val = row.get('alpha', 0)
        print(f"      alpha={alpha_val:.1e}, profile={row.get('profile','?')}, "
              f"window={row.get('window','?')}")
        print(f"      ze={row['z_entry']:.4f}, zx={row['z_exit']:.4f}, "
              f"zs={row['z_stop']:.4f}, conf={row['min_confidence']:.0f}")
        print(f"      trades={row['trades']:.0f}, WR={row['win_rate']:.1f}%, "
              f"PnL=${row['pnl']:,.0f}, PF={row['profit_factor']:.3f}")
        if "avg_pnl_trade" in row.index:
            print(f"      avg_pnl=${row['avg_pnl_trade']:,.0f}")
        print()

    # Print as Python dict for validate script
    print("\n  --- PYTHON CONFIG DICT (copy to validate_top_NQ_RTY.py) ---\n")
    print("CONFIGS = {")
    for i, (pname, row) in enumerate(selected):
        window_str = row.get("window", "05:00-12:00")
        alpha_val = row.get("alpha", 0)
        print(f'    "K_{pname}": {{')
        print(f'        "alpha": {alpha_val:.1e},')
        print(f'        "profile": "{row.get("profile", "tres_court")}",')
        print(f'        "window": "{window_str}",')
        print(f'        "z_entry": {row["z_entry"]:.4f},')
        print(f'        "z_exit": {row["z_exit"]:.4f},')
        print(f'        "z_stop": {row["z_stop"]:.4f},')
        print(f'        "min_confidence": {row["min_confidence"]:.0f},')
        print('    },')
    print("}")


def _print_top(df_top, n=10):
    if df_top is None or df_top.empty:
        print("  (aucun)")
        return

    cols_to_show = ["alpha", "profile", "window", "z_entry", "z_exit", "z_stop",
                    "min_confidence", "trades", "win_rate", "pnl", "profit_factor", "avg_pnl_trade"]
    available = [c for c in cols_to_show if c in df_top.columns]

    print(f"  {'#':>3} {'alpha':>8} {'profile':<10} {'window':<14} "
          f"{'ze':>6} {'zx':>5} {'zs':>5} {'conf':>4} | "
          f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7}")
    print("  " + "-" * 105)

    for rank, (_, row) in enumerate(df_top.head(n).iterrows(), 1):
        alpha_str = f"{row['alpha']:.1e}" if "alpha" in row.index else "?"
        prof_str = str(row.get("profile", "?"))
        win_str = str(row.get("window", "?"))
        print(f"  {rank:>3} {alpha_str:>8} {prof_str:<10} {win_str:<14} "
              f"{row.get('z_entry', 0):>6.3f} {row.get('z_exit', 0):>5.3f} "
              f"{row.get('z_stop', 0):>5.3f} {row.get('min_confidence', 0):>4.0f} | "
              f"{row.get('trades', 0):>5.0f} {row.get('win_rate', 0):>5.1f}% "
              f"${row.get('pnl', 0):>9,.0f} {row.get('profit_factor', 0):>6.3f} "
              f"${row.get('avg_pnl_trade', 0):>6,.0f}")


if __name__ == "__main__":
    main()
