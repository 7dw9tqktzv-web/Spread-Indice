"""
Find (timeframe, z-score period) combination that gives z-score = target at a specific time.

Grid search over timeframe resampling x z-score rolling window.
Uses fixed Kalman alpha/beta to compute spread = log(A) - alpha - beta * log(B).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PARAMETERS
# ============================================================================
ALPHA = 3.3379
BETA = 0.9135
TARGET_Z = 1.93
TARGET_DATE = "2026-03-04"
TARGET_TIME = "00:35:00"
SESSION_START = "17:30"
SESSION_END = "15:30"

# Grid search ranges
TIMEFRAMES = [1, 2, 3, 5, 10, 15, 30]  # minutes
ZSCORE_PERIODS = list(range(5, 201))  # 5 to 200

# Data files
RAW_DIR = Path("raw")
FILE_A = RAW_DIR / "CLJ26_FUT_CME.scid_BarData.txt"
FILE_B = RAW_DIR / "NGJ26_FUT_CME.scid_BarData.txt"


# ============================================================================
# DATA LOADING
# ============================================================================
def load_sierra_csv(filepath):
    """Load Sierra Chart exported bar data."""
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.set_index("datetime")
    df = df[["Last"]].rename(columns={"Last": "close"})
    df = df[df["close"] > 0]  # Remove bad data
    return df


def filter_session(df, start="17:30", end="15:30"):
    """Filter to session hours. Wraps midnight via OR logic."""
    t = df.index.time
    start_t = pd.Timestamp(f"2000-01-01 {start}").time()
    end_t = pd.Timestamp(f"2000-01-01 {end}").time()
    # Overnight session: t >= 17:30 OR t < 15:30
    mask = (t >= start_t) | (t < end_t)
    return df[mask]


def resample_ohlc(df, minutes):
    """Resample 1min data to N-minute bars."""
    if minutes == 1:
        return df
    return df.resample(f"{minutes}min").last().dropna()


# ============================================================================
# SPREAD & Z-SCORE
# ============================================================================
def compute_spread(df_a, df_b, alpha, beta):
    """Compute log spread with fixed alpha/beta."""
    aligned = df_a.join(df_b, lsuffix="_a", rsuffix="_b", how="inner")
    aligned = aligned.dropna()
    log_a = np.log(aligned["close_a"])
    log_b = np.log(aligned["close_b"])
    aligned["spread"] = log_a - alpha - beta * log_b
    return aligned


def compute_zscore(spread_series, period):
    """Rolling z-score = (spread - SMA) / StdDev."""
    sma = spread_series.rolling(period).mean()
    std = spread_series.rolling(period).std()
    zscore = (spread_series - sma) / std
    return zscore


# ============================================================================
# GRID SEARCH
# ============================================================================
def find_target_bar(index, target_date, target_time, tf_minutes):
    """Find the bar containing the target datetime for a given timeframe."""
    target_dt = pd.Timestamp(f"{target_date} {target_time}")

    if tf_minutes == 1:
        if target_dt in index:
            return target_dt
        # Find closest bar before target
        mask = index <= target_dt
        if mask.any():
            return index[mask][-1]
        return None

    # For higher timeframes, find the bar that contains the target time
    mask = index <= target_dt
    if mask.any():
        return index[mask][-1]
    return None


def main():
    print("Loading data...")
    df_a = load_sierra_csv(FILE_A)
    df_b = load_sierra_csv(FILE_B)

    print(f"CL: {len(df_a)} bars, {df_a.index[0]} -> {df_a.index[-1]}")
    print(f"NG: {len(df_b)} bars, {df_b.index[0]} -> {df_b.index[-1]}")

    # Filter session
    df_a = filter_session(df_a, SESSION_START, SESSION_END)
    df_b = filter_session(df_b, SESSION_START, SESSION_END)
    print(f"After session filter: CL={len(df_a)}, NG={len(df_b)}")

    target_dt = pd.Timestamp(f"{TARGET_DATE} {TARGET_TIME}")
    print(f"\nTarget: z-score = {TARGET_Z} at {target_dt}")
    print(f"Alpha = {ALPHA}, Beta = {BETA}")
    print(f"Grid: {len(TIMEFRAMES)} timeframes x {len(ZSCORE_PERIODS)} periods\n")

    results = []

    for tf in TIMEFRAMES:
        # Resample
        a_resampled = resample_ohlc(df_a, tf)
        b_resampled = resample_ohlc(df_b, tf)

        # Compute spread
        aligned = compute_spread(a_resampled, b_resampled, ALPHA, BETA)

        if len(aligned) == 0:
            continue

        # Find target bar
        bar = find_target_bar(aligned.index, TARGET_DATE, TARGET_TIME, tf)
        if bar is None:
            continue

        bar_idx = aligned.index.get_loc(bar)

        for zp in ZSCORE_PERIODS:
            if bar_idx < zp:
                continue  # Not enough data for this period

            zscore = compute_zscore(aligned["spread"], zp)

            if bar in zscore.index:
                z_val = zscore.loc[bar]
                if np.isnan(z_val):
                    continue

                diff = abs(z_val - TARGET_Z)
                results.append({
                    "timeframe": tf,
                    "z_period": zp,
                    "z_score": z_val,
                    "diff": diff,
                    "bar_time": bar,
                    "spread": aligned.loc[bar, "spread"],
                })

    if not results:
        print("No results found!")
        return

    # Sort by closest to target
    results_df = pd.DataFrame(results).sort_values("diff")

    print("=" * 75)
    print(f"TOP 20 combos closest to z-score = {TARGET_Z}")
    print("=" * 75)
    print(f"{'TF':>4} {'Z_Period':>8} {'Z-Score':>10} {'Diff':>8} {'Spread':>10} {'Bar Time'}")
    print("-" * 75)

    for _, row in results_df.head(20).iterrows():
        print(
            f"{row['timeframe']:>4}m {row['z_period']:>7} {row['z_score']:>10.4f} "
            f"{row['diff']:>8.4f} {row['spread']:>10.6f}  {row['bar_time']}"
        )

    # Exact matches (diff < 0.01)
    exact = results_df[results_df["diff"] < 0.01]
    if len(exact) > 0:
        print(f"\n=== {len(exact)} EXACT MATCHES (diff < 0.01) ===")
        for _, row in exact.iterrows():
            print(f"  TF={row['timeframe']}min  Z_Period={row['z_period']}  z={row['z_score']:.4f}")


if __name__ == "__main__":
    main()
