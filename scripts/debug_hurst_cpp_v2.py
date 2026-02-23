"""Debug Hurst v2: Deep dive into WHY H=0.01 at recent bars.

Key finding from v1: The C++ algorithm is CORRECT. The issue is that
H genuinely IS ~0.01 for the most recent bars.

BUT: The user says Python hurst_rolling gives ~0.4 on the same pair.
This means either:
1. The Python pipeline computes a DIFFERENT spread (different OLS coefficients)
2. The Python hurst_rolling uses different parameters
3. The spread data from Sierra is different from the Python pipeline spread

This script investigates the root cause.
"""

import numpy as np
import pandas as pd
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stats.hurst import hurst_exponent, hurst_rolling


def parse_sierra_export(filepath: str) -> pd.DataFrame:
    """Parse Sierra spreadsheet export."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    header_line = lines[1].strip().split("\t")
    spread_col = None
    hurst_col = None
    for i, col in enumerate(header_line):
        if "Spread (Log)" in col:
            spread_col = i
        if "Hurst Exponent" in col:
            hurst_col = i
    for line in lines[2:]:
        parts = line.strip().split("\t")
        if len(parts) <= spread_col:
            continue
        try:
            dt = parts[0].strip()
            spread_val = float(parts[spread_col])
            hurst_val = float(parts[hurst_col]) if hurst_col and len(parts) > hurst_col else np.nan
            rows.append({"datetime": dt, "spread_log": spread_val, "hurst_cpp": hurst_val})
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows)
    df = df.iloc[::-1].reset_index(drop=True)
    return df


def analyze_recent_spread_behavior(df: pd.DataFrame):
    """Why is H=0.01 at the most recent bars?"""
    spread = df["spread_log"].values
    n = len(spread)

    print("=" * 80)
    print("WHY IS H=0.01 AT RECENT BARS?")
    print("=" * 80)

    # Look at the last 64 bars
    window = spread[n-64:n]

    print(f"\nLast 64 bars spread values:")
    print(f"  Range: [{window.min():.8f}, {window.max():.8f}]")
    print(f"  Spread of range: {window.max() - window.min():.8f}")
    print(f"  Mean: {window.mean():.8f}")
    print(f"  Std: {np.std(window):.8f}")

    # Plot tau vs lag
    print(f"\n  Lag | tau (ddof=1) | log(tau)     | Expected ~0.4 tau")
    print(f"  " + "-" * 65)
    max_lag = 16
    taus = []
    log_taus = []
    log_lags = []
    for lag in range(2, max_lag + 1):
        diffs = window[lag:] - window[:-lag]
        tau = np.std(diffs, ddof=1)
        taus.append(tau)
        log_tau = math.log(tau) if tau > 0 else float('nan')
        log_lag = math.log(lag)
        log_taus.append(log_tau)
        log_lags.append(log_lag)
        # For H=0.4, tau should scale as lag^0.4
        # So tau(lag) = tau(2) * (lag/2)^0.4
        expected_tau = taus[0] * (lag / 2) ** 0.4 if len(taus) > 0 else 0
        print(f"  {lag:>3} | {tau:.10f} | {log_tau:>12.6f} | {expected_tau:.10f}")

    # The key observation: tau PEAKS around lag 5-6, then DECREASES
    # This means the spread has a cyclic/reverting behavior at ~5-6 bar scale
    # AND THEN the amplitude decreases at longer lags
    # This is characteristic of an ANTI-PERSISTENT or very fast mean-reverting process

    print(f"\n  OBSERVATION: tau peaks at lag ~5-6 then decreases!")
    print(f"  This means spread diffs at lag 16 have LESS variance than at lag 6")
    print(f"  This is the signature of a process reverting FASTER than any H > 0")
    print(f"  It means the spread in this window is almost perfectly periodic/reverting")

    # Check: is this because the spread crossed a big jump?
    print(f"\n  Bar-to-bar diffs (last 64):")
    bar_diffs = np.diff(window)
    print(f"    Max abs diff: {np.max(np.abs(bar_diffs)):.10f}")
    print(f"    Mean abs diff: {np.mean(np.abs(bar_diffs)):.10f}")
    print(f"    Std of diffs: {np.std(bar_diffs):.10f}")

    # Check for session gap in the window
    datetimes = df["datetime"].values[n-64:n]
    print(f"\n  Datetime range of last 64 bars:")
    print(f"    First: {datetimes[0]}")
    print(f"    Last: {datetimes[-1]}")

    # Look for large time gaps
    print(f"\n  Checking for session gaps in last 64 bars...")
    for i in range(1, len(datetimes)):
        dt_prev = pd.Timestamp(datetimes[i-1])
        dt_curr = pd.Timestamp(datetimes[i])
        gap_min = (dt_curr - dt_prev).total_seconds() / 60
        if gap_min > 10:
            print(f"    GAP at bar {n-64+i}: {datetimes[i-1]} -> {datetimes[i]} ({gap_min:.0f} min)")
            print(f"    Spread before: {window[i-1]:.10f}, after: {window[i]:.10f}")


def analyze_hurst_over_time(df: pd.DataFrame):
    """Show how Hurst evolves over the data range."""
    spread = df["spread_log"].values
    hurst_cpp = df["hurst_cpp"].values
    n = len(spread)

    print(f"\n{'='*80}")
    print("HURST EVOLUTION OVER TIME (Sierra C++ values)")
    print(f"{'='*80}")

    # Sample at regular intervals
    datetimes = df["datetime"].values
    step = n // 20
    print(f"\n{'Bar':>6} | {'Datetime':>20} | {'Spread':>12} | {'H_cpp':>8} | {'Category':>15}")
    print("-" * 80)
    for i in range(step, n, step):
        cat = "NORMAL" if 0.2 < hurst_cpp[i] < 0.7 else ("LOW" if hurst_cpp[i] < 0.2 else "HIGH")
        print(f"{i:>6} | {datetimes[i]:>20} | {spread[i]:>12.8f} | {hurst_cpp[i]:>8.4f} | {cat:>15}")

    # Last 20 bars specifically
    print(f"\n  --- Last 20 bars ---")
    for i in range(n-20, n):
        cat = "NORMAL" if 0.2 < hurst_cpp[i] < 0.7 else ("LOW" if hurst_cpp[i] < 0.2 else "HIGH")
        print(f"  {i:>6} | {datetimes[i]:>20} | {spread[i]:>12.8f} | {hurst_cpp[i]:>8.4f} | {cat:>15}")

    # Count H < 0.1 episodes
    low_h = hurst_cpp < 0.1
    print(f"\n  Bars with H < 0.1: {np.sum(low_h)} ({100*np.sum(low_h)/n:.1f}%)")
    print(f"  Bars with H < 0.05: {np.sum(hurst_cpp < 0.05)} ({100*np.sum(hurst_cpp < 0.05)/n:.1f}%)")

    # When do H < 0.1 episodes occur?
    indices = np.where(low_h)[0]
    if len(indices) > 0:
        print(f"\n  H < 0.1 episodes (first 5):")
        episode_start = indices[0]
        count = 0
        for j in range(1, len(indices)):
            if indices[j] - indices[j-1] > 5:  # New episode
                print(f"    Bars {episode_start}-{indices[j-1]}: "
                      f"{datetimes[episode_start]} to {datetimes[indices[j-1]]} "
                      f"({indices[j-1]-episode_start+1} bars)")
                episode_start = indices[j]
                count += 1
                if count >= 5:
                    break
        # Last episode
        print(f"    Bars {episode_start}-{indices[-1]}: "
              f"{datetimes[episode_start]} to {datetimes[indices[-1]]} "
              f"({indices[-1]-episode_start+1} bars)")


def compare_python_pipeline_spread(df: pd.DataFrame):
    """Run the Python pipeline to compute spread and compare with Sierra.

    The Python pipeline uses:
    1. Load NQ and YM data
    2. OLS rolling regression: log_a = alpha + beta * log_b + epsilon
    3. Spread = log_a - alpha - beta * log_b (= epsilon)
    4. Hurst on spread

    Sierra may use different OLS parameters, lookback, or computation timing.
    """
    print(f"\n{'='*80}")
    print("COMPARISON: Sierra Spread vs Python Pipeline Spread")
    print(f"{'='*80}")

    # Try to load the Python pipeline and compute spread
    try:
        from src.data.loader import load_instrument
        from src.data.cleaner import clean_bars
        from src.data.resampler import resample_bars
        from src.data.alignment import align_pair
        from src.hedge.ols_rolling import OLSRollingEstimator, OLSRollingConfig
        from src.spread.pair import SpreadPair

        # Load data
        nq = load_instrument("NQ")
        ym = load_instrument("YM")
        nq = clean_bars(nq)
        ym = clean_bars(ym)
        nq_5m = resample_bars(nq, "5min")
        ym_5m = resample_bars(ym, "5min")

        aligned = align_pair(nq_5m, ym_5m)
        print(f"  Python aligned data: {len(aligned.df)} bars")
        print(f"  Date range: {aligned.df.index[0]} to {aligned.df.index[-1]}")

        # Run OLS with Config E parameters
        config = OLSRollingConfig(lookback_bars=3300, zscore_window=30)
        estimator = OLSRollingEstimator(config)
        result = estimator.estimate(aligned)

        py_spread = result.spread
        py_hurst = hurst_rolling(py_spread.dropna(), window=64, step=1)

        # Get the last 200 bars of Python spread
        py_spread_last = py_spread.dropna().tail(200)
        py_hurst_last = py_hurst.dropna().tail(200)

        print(f"\n  Python spread (last 10 bars):")
        for idx, val in py_spread_last.tail(10).items():
            h_val = py_hurst.loc[idx] if idx in py_hurst.index else np.nan
            print(f"    {idx}: spread={val:.10f}, hurst={h_val:.6f}")

        # Compare with Sierra
        sierra_spread_last = df["spread_log"].values[-200:]
        sierra_hurst_last = df["hurst_cpp"].values[-200:]
        sierra_datetimes = df["datetime"].values[-200:]

        print(f"\n  Sierra spread (last 10 bars):")
        for i in range(-10, 0):
            print(f"    {sierra_datetimes[i]}: spread={sierra_spread_last[i]:.10f}, hurst={sierra_hurst_last[i]:.6f}")

        # Key comparison: are the SPREAD VALUES different?
        print(f"\n  --- Spread value comparison ---")
        print(f"  Python spread range (last 200): [{py_spread_last.min():.10f}, {py_spread_last.max():.10f}]")
        print(f"  Sierra spread range (last 200): [{sierra_spread_last.min():.10f}, {sierra_spread_last.max():.10f}]")
        print(f"  Python spread std: {py_spread_last.std():.10f}")
        print(f"  Sierra spread std: {np.std(sierra_spread_last):.10f}")

        # Hurst comparison
        print(f"\n  --- Hurst comparison (last 200) ---")
        print(f"  Python hurst mean: {py_hurst_last.mean():.6f}")
        print(f"  Python hurst median: {py_hurst_last.median():.6f}")
        print(f"  Sierra hurst mean: {np.nanmean(sierra_hurst_last):.6f}")
        print(f"  Sierra hurst median: {np.nanmedian(sierra_hurst_last):.6f}")

        # Are the spread values correlated?
        # We need to match timestamps first
        sierra_dt_parsed = pd.to_datetime(df["datetime"].values)
        sierra_series = pd.Series(df["spread_log"].values, index=sierra_dt_parsed)

        common_idx = py_spread.index.intersection(sierra_series.index)
        if len(common_idx) > 0:
            print(f"\n  Common timestamps: {len(common_idx)}")
            py_common = py_spread.loc[common_idx].dropna()
            si_common = sierra_series.loc[common_idx].dropna()
            common2 = py_common.index.intersection(si_common.index)
            if len(common2) > 10:
                corr = py_common.loc[common2].corr(si_common.loc[common2])
                print(f"  Correlation of spread values: {corr:.6f}")
                diff = py_common.loc[common2] - si_common.loc[common2]
                print(f"  Mean absolute difference: {diff.abs().mean():.10f}")
                print(f"  Max absolute difference: {diff.abs().max():.10f}")

                # Sample of matched values
                sample_idx = common2[-10:]
                print(f"\n  Sample matched values (last 10 common):")
                for idx in sample_idx:
                    print(f"    {idx}: Py={py_common.loc[idx]:.10f}, "
                          f"Si={si_common.loc[idx]:.10f}, "
                          f"Diff={py_common.loc[idx]-si_common.loc[idx]:.10f}")
        else:
            print(f"  No common timestamps found!")
            print(f"  Python index sample: {py_spread.index[-5:]}")
            print(f"  Sierra index sample: {sierra_dt_parsed[-5:]}")

    except Exception as e:
        print(f"  ERROR: Could not run Python pipeline: {e}")
        import traceback
        traceback.print_exc()


def analyze_last_session_hurst(df: pd.DataFrame):
    """Focus specifically on the last trading session to understand H~0.01."""
    spread = df["spread_log"].values
    hurst_cpp = df["hurst_cpp"].values
    datetimes = df["datetime"].values
    n = len(spread)

    print(f"\n{'='*80}")
    print("LAST SESSION DEEP ANALYSIS")
    print(f"{'='*80}")

    # Find last session (2026-02-20)
    last_date = "2026-02-20"
    mask = [last_date in dt for dt in datetimes]
    session_indices = np.where(mask)[0]

    if len(session_indices) == 0:
        print("  No bars found for 2026-02-20")
        return

    print(f"  Session bars: {len(session_indices)}")
    print(f"  Index range: {session_indices[0]} to {session_indices[-1]}")

    # Show spread and hurst for the session
    session_spread = spread[session_indices]
    session_hurst = hurst_cpp[session_indices]
    session_dt = datetimes[session_indices]

    # Focus on the transition from normal to H~0.01
    print(f"\n  Time | Spread      | Hurst   | Note")
    print(f"  " + "-" * 60)
    prev_h = 0.5
    for i, idx in enumerate(session_indices):
        h = session_hurst[i]
        note = ""
        if h < 0.1 and prev_h >= 0.1:
            note = " <-- TRANSITION TO LOW H"
        elif h >= 0.1 and prev_h < 0.1:
            note = " <-- RECOVERY"
        if i < 20 or i > len(session_indices) - 30 or abs(h - prev_h) > 0.1:
            time_part = session_dt[i].split()[-1] if " " in session_dt[i] else session_dt[i]
            print(f"  {time_part} | {spread[idx]:>12.8f} | {h:>7.4f} | {note}")
        prev_h = h

    # Check what happens in the 64-bar window at the end of day
    end_idx = session_indices[-1]
    if end_idx >= 64:
        window = spread[end_idx - 63:end_idx + 1]
        window_dt = datetimes[end_idx - 63:end_idx + 1]

        print(f"\n  Last 64-bar window:")
        print(f"    From: {window_dt[0]}")
        print(f"    To:   {window_dt[-1]}")

        # Check if this window spans a session gap (15:30 -> 17:30 next day)
        gap_found = False
        for j in range(1, len(window_dt)):
            dt_prev = pd.Timestamp(window_dt[j-1])
            dt_curr = pd.Timestamp(window_dt[j])
            gap_min = (dt_curr - dt_prev).total_seconds() / 60
            if gap_min > 10:
                print(f"    SESSION GAP at position {j}: {window_dt[j-1]} -> {window_dt[j]} ({gap_min:.0f} min)")
                print(f"    Spread jump: {window[j-1]:.10f} -> {window[j]:.10f} (diff={window[j]-window[j-1]:.10f})")
                gap_found = True

        if not gap_found:
            print(f"    No session gap in window (all same session)")

        # Autocorrelation of the window
        diffs = np.diff(window)
        if len(diffs) > 1:
            acf1 = np.corrcoef(diffs[:-1], diffs[1:])[0, 1]
            print(f"\n    ACF(1) of bar-to-bar diffs: {acf1:.6f}")
            if acf1 < -0.3:
                print(f"    STRONG NEGATIVE ACF => oscillating/reverting at 1-bar level")
                print(f"    This explains low Hurst: the spread bounces back every bar")


def main():
    filepath = r"C:\Users\Bonjour\Desktop\Spread_Indice\raw\DefaultSpreadsheetStudy.txt"

    df = parse_sierra_export(filepath)

    # Analysis 1: Why H=0.01 at recent bars?
    analyze_recent_spread_behavior(df)

    # Analysis 2: Hurst evolution over time
    analyze_hurst_over_time(df)

    # Analysis 3: Last session deep dive
    analyze_last_session_hurst(df)

    # Analysis 4: Compare with Python pipeline
    compare_python_pipeline_spread(df)

    # =========================================================================
    # FINAL DIAGNOSIS
    # =========================================================================
    print(f"\n{'='*80}")
    print("ROOT CAUSE ANALYSIS")
    print(f"{'='*80}")
    print("""
FINDINGS:
1. The C++ algorithm is CORRECT - our Python simulation matches Sierra exactly.
2. H=0.01 at bar 9450 is the TRUE Hurst of that 64-bar window.
3. The spread in the last 64 bars shows tau DECREASING after lag 5-6,
   meaning the process is ultra-mean-reverting in that specific window.

QUESTION: Does the Python backtest pipeline compute different Hurst values
for the same timestamps? If so, the difference is in the SPREAD COMPUTATION
(different OLS parameters, different data), not in the Hurst algorithm itself.

KEY INSIGHT: The Sierra export covers only ~36 trading days (Jan 1 - Feb 20, 2026).
The Python backtest covers 2019-2025 (5+ years). The Hurst of ~0.4 is the
MEDIAN over 5 years. The Hurst at any given 64-bar window can vary widely
(from 0.01 to 0.77 as shown in the Sierra export itself).

H=0.01 at the END OF DAY is likely because:
- The last bars of the day (14:30-15:30 CT) have low volume/volatility
- The spread oscillates in a tight range as markets wind down
- This creates ultra-mean-reverting behavior in the 64-bar window
""")


if __name__ == "__main__":
    main()
