"""Analyse dimensionnelle du grid search OLS NQ/RTY (15.9M combos).

Reads the grid CSV and computes stats by each parameter dimension.
Selects top 5 configs by profile for validation.

Usage:
    python scripts/analyze_grid_ols_NQ_RTY.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    csv_path = PROJECT_ROOT / "output" / "NQ_RTY" / "grid_ols_filtered.csv"
    if not csv_path.exists():
        print(f"ERREUR: {csv_path} introuvable")
        return

    df = pd.read_csv(csv_path)
    print(f"Grid profitable: {len(df)} rows loaded")

    # Rename columns for consistency
    col_map = {"profil": "profile"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Filter out infinite PF
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["profit_factor"])
    # Filter PF > 100 (artifact from near-zero loss)
    df = df[df["profit_factor"] < 100].copy()
    print(f"After PF < 100 filter: {len(df)} rows")

    # ================================================================
    # DIMENSION ANALYSIS
    # ================================================================
    dims = ["ols_window", "zscore_window", "profile", "window",
            "z_entry", "z_exit", "z_stop", "min_confidence"]

    for dim in dims:
        if dim not in df.columns:
            print(f"\n  SKIP: column '{dim}' not found")
            continue

        print(f"\n{'='*80}")
        print(f"  DIMENSION: {dim}")
        print(f"{'='*80}")

        grp = df.groupby(dim).agg(
            count=("pnl", "size"),
            avg_pf=("profit_factor", "mean"),
            med_pf=("profit_factor", "median"),
            avg_pnl=("pnl", "mean"),
            max_pnl=("pnl", "max"),
            avg_trades=("trades", "mean"),
            avg_wr=("win_rate", "mean"),
        ).sort_values("avg_pf", ascending=False)

        print(f"  {'Value':<18} {'Count':>8} {'AvgPF':>7} {'MedPF':>7} "
              f"{'AvgPnL':>10} {'MaxPnL':>10} {'AvgTrd':>7} {'AvgWR':>7}")
        print("  " + "-" * 82)
        for idx_val, row in grp.iterrows():
            label = str(idx_val)
            print(f"  {label:<18} {row['count']:>8.0f} {row['avg_pf']:>7.3f} "
                  f"{row['med_pf']:>7.3f} ${row['avg_pnl']:>9,.0f} "
                  f"${row['max_pnl']:>9,.0f} {row['avg_trades']:>7.0f} "
                  f"{row['avg_wr']:>6.1f}%")

    # ================================================================
    # TOP CONFIGS — by different profiles
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  TOP CONFIGS BY PROFILE")
    print(f"{'='*120}")

    # Filter: min 30 trades for robustness
    robust = df[df["trades"] >= 30].copy()
    print(f"\n  Robust configs (trades >= 30): {len(robust)}")

    # 1. Volume champion: most trades with PF > 1.3
    vol = robust[robust["profit_factor"] > 1.3].sort_values("trades", ascending=False)
    print("\n  --- VOLUME (max trades, PF > 1.3) ---")
    _print_top(vol, 10)

    # 2. Quality champion: highest PF with trades >= 50
    qual = robust[robust["trades"] >= 50].sort_values("profit_factor", ascending=False)
    print("\n  --- QUALITY (max PF, trades >= 50) ---")
    _print_top(qual, 10)

    # 3. Balanced: best PnL * PF product with trades >= 80
    bal = robust[robust["trades"] >= 80].copy()
    if not bal.empty:
        bal["score"] = bal["pnl"] * bal["profit_factor"]
        bal = bal.sort_values("score", ascending=False)
    print("\n  --- BALANCED (PnL*PF, trades >= 80) ---")
    _print_top(bal, 10)

    # 4. Sniper: best avg_pnl_trade with trades >= 30
    snp = robust.sort_values("avg_pnl_trade", ascending=False)
    print("\n  --- SNIPER (max avg PnL/trade, trades >= 30) ---")
    _print_top(snp, 10)

    # 5. PropFirm: PF > 1.5, trades >= 40, high WR
    pf = robust[(robust["profit_factor"] > 1.5) & (robust["trades"] >= 40)].copy()
    pf = pf.sort_values("win_rate", ascending=False)
    print("\n  --- PROPFIRM (PF > 1.5, trades >= 40, max WR) ---")
    _print_top(pf, 10)

    # ================================================================
    # SWEET SPOT: ols_window x z_entry
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  SWEET SPOT: ols_window x z_entry (avg PF, min 20 configs)")
    print(f"{'='*120}")

    pivot = df.groupby(["ols_window", "z_entry"]).agg(
        avg_pf=("profit_factor", "mean"),
        count=("pnl", "size"),
    )
    pivot = pivot[pivot["count"] >= 20]

    if not pivot.empty:
        piv_pf = pivot["avg_pf"].unstack(level="z_entry")
        ols_vals = sorted(piv_pf.index.unique())
        entries = sorted(piv_pf.columns)

        header = f"  {'OLS':>7}"
        for ze in entries:
            header += f" {ze:>6.2f}"
        print(header)
        print("  " + "-" * (7 + 7 * len(entries)))

        for ow in ols_vals:
            row_str = f"  {ow:>7}"
            for ze in entries:
                val = piv_pf.loc[ow, ze] if ze in piv_pf.columns and ow in piv_pf.index else np.nan
                if pd.isna(val):
                    row_str += f" {'--':>6}"
                else:
                    row_str += f" {val:>6.3f}"
            print(row_str)

    # ================================================================
    # SWEET SPOT: ols_window x zscore_window
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  SWEET SPOT: ols_window x zscore_window (avg PF, min 20 configs)")
    print(f"{'='*120}")

    pivot2 = df.groupby(["ols_window", "zscore_window"]).agg(
        avg_pf=("profit_factor", "mean"),
        count=("pnl", "size"),
    )
    pivot2 = pivot2[pivot2["count"] >= 20]

    if not pivot2.empty:
        piv2 = pivot2["avg_pf"].unstack(level="zscore_window")
        ols_vals2 = sorted(piv2.index.unique())
        zw_vals = sorted(piv2.columns)

        header2 = f"  {'OLS':>7}"
        for zw in zw_vals:
            header2 += f" {zw:>6}"
        print(header2)
        print("  " + "-" * (7 + 7 * len(zw_vals)))

        for ow in ols_vals2:
            row_str = f"  {ow:>7}"
            for zw in zw_vals:
                val = piv2.loc[ow, zw] if zw in piv2.columns and ow in piv2.index else np.nan
                if pd.isna(val):
                    row_str += f" {'--':>6}"
                else:
                    row_str += f" {val:>6.3f}"
            print(row_str)

    # ================================================================
    # FINAL TOP 5 SELECTION
    # ================================================================
    print(f"\n\n{'='*120}")
    print("  FINAL TOP 5 SELECTION FOR VALIDATION")
    print(f"{'='*120}")

    selected = []

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
            key = (row.get("ols_window", 0), row.get("zscore_window", 0),
                   row["z_entry"], row["z_exit"], row["z_stop"],
                   row["min_confidence"], row.get("profile", ""), row.get("window", ""))
            if key not in used_keys:
                used_keys.add(key)
                selected.append((profile_name, row))
                break

    print(f"\n  Selected {len(selected)} configs:\n")
    for i, (pname, row) in enumerate(selected):
        print(f"  [{i+1}] {pname}")
        print(f"      OLS={row.get('ols_window',0):.0f}, ZW={row.get('zscore_window',0):.0f}, "
              f"profile={row.get('profile','?')}, window={row.get('window','?')}")
        print(f"      ze={row['z_entry']:.2f}, zx={row['z_exit']:.2f}, "
              f"zs={row['z_stop']:.2f}, conf={row['min_confidence']:.0f}")
        print(f"      trades={row['trades']:.0f}, WR={row['win_rate']:.1f}%, "
              f"PnL=${row['pnl']:,.0f}, PF={row['profit_factor']:.3f}")
        if "avg_pnl_trade" in row.index:
            print(f"      avg_pnl=${row['avg_pnl_trade']:,.0f}")
        print()

    # Print as Python dict for validate script
    print("\n  --- PYTHON CONFIG DICT (for validate script) ---\n")
    print("CONFIGS_OLS_NQ_RTY = {")
    for _i, (pname, row) in enumerate(selected):
        window_str = row.get("window", "05:00-12:00")
        print(f'    "OLS_{pname}": {{')
        print(f'        "ols_window": {row.get("ols_window", 0):.0f},')
        print(f'        "zscore_window": {row.get("zscore_window", 0):.0f},')
        print(f'        "profile": "{row.get("profile", "tres_court")}",')
        print(f'        "window": "{window_str}",')
        print(f'        "z_entry": {row["z_entry"]:.2f},')
        print(f'        "z_exit": {row["z_exit"]:.2f},')
        print(f'        "z_stop": {row["z_stop"]:.2f},')
        print(f'        "min_confidence": {row["min_confidence"]:.0f},')
        print('    },')
    print("}")


def _print_top(df_top, n=10):
    if df_top is None or df_top.empty:
        print("  (aucun)")
        return

    print(f"  {'#':>3} {'OLS':>6} {'ZW':>4} {'profile':<10} {'window':<14} "
          f"{'ze':>5} {'zx':>4} {'zs':>4} {'conf':>4} | "
          f"{'Trd':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'Avg$':>7}")
    print("  " + "-" * 105)

    for rank, (_, row) in enumerate(df_top.head(n).iterrows(), 1):
        prof_str = str(row.get("profile", "?"))
        win_str = str(row.get("window", "?"))
        print(f"  {rank:>3} {row.get('ols_window', 0):>6.0f} {row.get('zscore_window', 0):>4.0f} "
              f"{prof_str:<10} {win_str:<14} "
              f"{row.get('z_entry', 0):>5.2f} {row.get('z_exit', 0):>4.2f} "
              f"{row.get('z_stop', 0):>4.1f} {row.get('min_confidence', 0):>4.0f} | "
              f"{row.get('trades', 0):>5.0f} {row.get('win_rate', 0):>5.1f}% "
              f"${row.get('pnl', 0):>9,.0f} {row.get('profit_factor', 0):>6.3f} "
              f"${row.get('avg_pnl_trade', 0):>6,.0f}")


if __name__ == "__main__":
    main()
