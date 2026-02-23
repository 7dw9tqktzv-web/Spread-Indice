"""Debug Hurst exponent discrepancy between C++ Sierra and Python.

C++ gives H~0.01, Python gives H~0.4 for the same NQ/YM spread.
This script:
1. Parses Sierra spreadsheet export to extract Spread (Log) values
2. Implements the EXACT C++ Hurst algorithm in Python
3. Compares with Python hurst_rolling from the project
4. Identifies the root cause step by step
"""

import numpy as np
import pandas as pd
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stats.hurst import hurst_exponent, hurst_rolling


# =============================================================================
# 1. Parse Sierra Export
# =============================================================================
def parse_sierra_export(filepath: str) -> pd.DataFrame:
    """Parse Sierra spreadsheet export (tab-separated, reverse chronological)."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip 2 header lines
    header_line = lines[1].strip().split("\t")
    print(f"Header columns count: {len(header_line)}")

    # Find column indices
    spread_col = None
    hurst_col = None
    zscore_col = None
    for i, col in enumerate(header_line):
        if "Spread (Log)" in col:
            spread_col = i
            print(f"  Spread (Log) column: {i} -> '{col}'")
        if "Hurst Exponent" in col:
            hurst_col = i
            print(f"  Hurst Exponent column: {i} -> '{col}'")
        if "Z-Score OLS" in col:
            zscore_col = i
            print(f"  Z-Score OLS column: {i} -> '{col}'")

    if spread_col is None:
        raise ValueError("Could not find Spread (Log) column")

    for line in lines[2:]:
        parts = line.strip().split("\t")
        if len(parts) <= spread_col:
            continue
        try:
            dt = parts[0].strip()
            spread_val = float(parts[spread_col])
            hurst_val = float(parts[hurst_col]) if hurst_col and len(parts) > hurst_col else np.nan
            zscore_val = float(parts[zscore_col]) if zscore_col and len(parts) > zscore_col else np.nan
            rows.append({
                "datetime": dt,
                "spread_log": spread_val,
                "hurst_cpp": hurst_val,
                "zscore_ols": zscore_val,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    print(f"\nParsed {len(df)} data rows")
    print(f"Date range: {df['datetime'].iloc[-1]} to {df['datetime'].iloc[0]} (originally reverse chrono)")

    # REVERSE to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    print(f"After reversal: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    return df


# =============================================================================
# 2. EXACT C++ Hurst Algorithm (simulated in Python)
# =============================================================================
def calculate_hurst_vr_cpp(spread_array: np.ndarray, end_index: int, period: int = 64) -> float:
    """Exact replica of C++ CalculateHurstVR function.

    Parameters
    ----------
    spread_array : np.ndarray
        Full spread array (chronological order).
    end_index : int
        Index of the current bar (inclusive).
    period : int
        Lookback period.

    Returns
    -------
    float
        Hurst exponent.
    """
    if period < 8 or end_index < period:
        return 0.0

    max_lag = period // 4
    if max_lag < 2:
        max_lag = 2
    if max_lag > 50:
        max_lag = 50

    sum_log_lag = 0.0
    sum_log_tau = 0.0
    sum_log_lag2 = 0.0
    sum_log_lag_log_tau = 0.0
    valid_count = 0

    start_idx = end_index - period + 1
    if start_idx < 0:
        start_idx = 0

    for lag in range(2, max_lag + 1):
        total = 0.0
        total_sq = 0.0
        cnt = 0

        for i in range(start_idx, end_index - lag + 1):
            if i >= 0:
                diff = spread_array[i + lag] - spread_array[i]
                total += diff
                total_sq += diff * diff
                cnt += 1

        if cnt < 5:
            continue

        mean = total / cnt
        var = (total_sq - cnt * mean * mean) / (cnt - 1)
        if var <= 0.0:
            continue
        tau = math.sqrt(var)

        log_lag = math.log(lag)
        log_tau = math.log(tau)

        sum_log_lag += log_lag
        sum_log_tau += log_tau
        sum_log_lag2 += log_lag * log_lag
        sum_log_lag_log_tau += log_lag * log_tau
        valid_count += 1

    if valid_count < 3:
        return 0.0

    n = float(valid_count)
    denom = n * sum_log_lag2 - sum_log_lag * sum_log_lag
    if abs(denom) < 1e-10:
        return 0.0

    H = (n * sum_log_lag_log_tau - sum_log_lag * sum_log_tau) / denom
    if H < 0.01:
        H = 0.01
    if H > 0.99:
        H = 0.99

    return H


# =============================================================================
# 3. Python hurst_exponent (single-shot, for comparison)
# =============================================================================
def python_hurst_on_window(spread_array: np.ndarray, end_index: int, period: int = 64) -> float:
    """Run Python hurst_exponent on the same window as C++."""
    start_idx = max(0, end_index - period + 1)
    window_data = spread_array[start_idx:end_index + 1]
    return hurst_exponent(pd.Series(window_data), max_lag=period // 4)


# =============================================================================
# 4. Detailed tau comparison for a single bar
# =============================================================================
def compare_tau_detail(spread_array: np.ndarray, end_index: int, period: int = 64):
    """Print tau values for each lag, comparing C++ and Python approaches."""
    max_lag = period // 4
    if max_lag > 50:
        max_lag = 50

    start_idx = max(0, end_index - period + 1)
    window = spread_array[start_idx:end_index + 1]
    n_window = len(window)

    print(f"\n{'='*80}")
    print(f"DETAILED TAU COMPARISON at bar index {end_index}")
    print(f"  period={period}, max_lag={max_lag}")
    print(f"  start_idx={start_idx}, end_index={end_index}")
    print(f"  window length: {n_window}")
    print(f"  spread range: [{window.min():.10f}, {window.max():.10f}]")
    print(f"  spread mean: {window.mean():.10f}")
    print(f"  spread std: {np.std(window):.10f}")
    print(f"{'='*80}")

    print(f"\n{'Lag':>4} | {'C++ tau':>14} | {'Py tau':>14} | {'C++ log(tau)':>14} | {'Py log(tau)':>14} | {'C++ cnt':>7} | {'Py cnt':>7} | {'Match':>6}")
    print("-" * 100)

    cpp_lags = []
    cpp_log_lags = []
    cpp_log_taus = []
    py_lags = []
    py_log_lags = []
    py_log_taus = []

    for lag in range(2, max_lag + 1):
        # C++ approach: iterate over full array indices
        total = 0.0
        total_sq = 0.0
        cnt = 0
        for i in range(start_idx, end_index - lag + 1):
            if i >= 0:
                diff = spread_array[i + lag] - spread_array[i]
                total += diff
                total_sq += diff * diff
                cnt += 1

        cpp_tau = np.nan
        if cnt >= 5:
            mean = total / cnt
            var = (total_sq - cnt * mean * mean) / (cnt - 1)
            if var > 0:
                cpp_tau = math.sqrt(var)
                cpp_lags.append(lag)
                cpp_log_lags.append(math.log(lag))
                cpp_log_taus.append(math.log(cpp_tau))

        # Python approach: numpy on window
        diffs_py = window[lag:] - window[:-lag]
        py_std = np.std(diffs_py)  # NOTE: np.std uses ddof=0 by default!
        py_std_ddof1 = np.std(diffs_py, ddof=1)
        py_cnt = len(diffs_py)

        py_tau = np.nan
        if py_std > 1e-12 and py_cnt >= 5:
            py_tau = py_std
            py_lags.append(lag)
            py_log_lags.append(math.log(lag))
            py_log_taus.append(math.log(py_tau))

        match = "OK" if (np.isnan(cpp_tau) and np.isnan(py_tau)) else ""
        if not np.isnan(cpp_tau) and not np.isnan(py_tau):
            if abs(cpp_tau - py_tau) < 1e-10:
                match = "OK"
            else:
                match = f"DIFF"

        print(f"{lag:>4} | {cpp_tau:>14.10f} | {py_tau:>14.10f} | "
              f"{math.log(cpp_tau) if not np.isnan(cpp_tau) and cpp_tau > 0 else 0:>14.6f} | "
              f"{math.log(py_tau) if not np.isnan(py_tau) and py_tau > 0 else 0:>14.6f} | "
              f"{cnt:>7} | {py_cnt:>7} | {match:>6}")

    # Compute H from both
    if len(cpp_lags) >= 3:
        cpp_log_lags = np.array(cpp_log_lags)
        cpp_log_taus = np.array(cpp_log_taus)
        n = len(cpp_lags)
        denom = n * np.sum(cpp_log_lags**2) - np.sum(cpp_log_lags)**2
        H_cpp = (n * np.sum(cpp_log_lags * cpp_log_taus) - np.sum(cpp_log_lags) * np.sum(cpp_log_taus)) / denom
        print(f"\n  C++ H (from manual OLS): {H_cpp:.10f}")
        print(f"  C++ H (clamped): {np.clip(H_cpp, 0.01, 0.99):.10f}")

    if len(py_lags) >= 3:
        py_log_lags_arr = np.array(py_log_lags)
        py_log_taus_arr = np.array(py_log_taus)
        H_py = np.polyfit(py_log_lags_arr, py_log_taus_arr, 1)[0]
        print(f"  Python H (from polyfit, ddof=0): {H_py:.10f}")
        print(f"  Python H (clamped): {np.clip(H_py, 0.01, 0.99):.10f}")

    # KEY DIAGNOSTIC: What's the difference between ddof=0 and ddof=1?
    print(f"\n  --- ddof comparison ---")
    print(f"  C++ uses sample variance (ddof=1): var = (sumSq - n*mean^2) / (n-1)")
    print(f"  Python np.std() uses ddof=0 by default: var = sum((x-mean)^2) / n")
    print(f"  This means C++ tau = sqrt(ddof1_var), Python tau = sqrt(ddof0_var)")
    print(f"  Ratio: tau_cpp/tau_py = sqrt(n/(n-1)) for same data")

    # Show the actual impact
    if len(cpp_lags) >= 3:
        # Recompute Python with ddof=1
        py_taus_ddof1 = []
        py_lags_ddof1 = []
        for lag in range(2, max_lag + 1):
            diffs_py = window[lag:] - window[:-lag]
            py_std_ddof1 = np.std(diffs_py, ddof=1)
            if py_std_ddof1 > 1e-12 and len(diffs_py) >= 5:
                py_taus_ddof1.append(py_std_ddof1)
                py_lags_ddof1.append(lag)

        if len(py_lags_ddof1) >= 3:
            log_lags_d1 = np.log(py_lags_ddof1)
            log_taus_d1 = np.log(py_taus_ddof1)
            H_py_ddof1 = np.polyfit(log_lags_d1, log_taus_d1, 1)[0]
            print(f"  Python H with ddof=1: {H_py_ddof1:.10f}")
            print(f"  (ddof difference alone doesn't explain the gap)")


# =============================================================================
# 5. Check spread value magnitude and distribution
# =============================================================================
def analyze_spread_values(df: pd.DataFrame):
    """Analyze the spread values from Sierra export."""
    spread = df["spread_log"].values
    hurst_cpp = df["hurst_cpp"].values

    print(f"\n{'='*80}")
    print("SPREAD VALUE ANALYSIS")
    print(f"{'='*80}")
    print(f"  Total bars: {len(spread)}")
    print(f"  Spread range: [{spread.min():.10f}, {spread.max():.10f}]")
    print(f"  Spread mean: {spread.mean():.10f}")
    print(f"  Spread std: {np.std(spread):.10f}")
    print(f"  Spread abs range: {spread.max() - spread.min():.10f}")

    # Look at typical 64-bar windows
    print(f"\n  --- 64-bar window stats (last 500 bars) ---")
    for idx in [len(spread)-1, len(spread)-100, len(spread)-200, len(spread)-500]:
        if idx < 64:
            continue
        w = spread[idx-63:idx+1]
        print(f"  Bar {idx}: window range = {w.max()-w.min():.10f}, "
              f"window std = {np.std(w):.10f}, "
              f"C++ Hurst = {hurst_cpp[idx]:.6f}")

    # Distribution of C++ Hurst values
    valid_hurst = hurst_cpp[~np.isnan(hurst_cpp) & (hurst_cpp > 0)]
    print(f"\n  --- C++ Hurst distribution ---")
    print(f"  Valid values: {len(valid_hurst)}")
    if len(valid_hurst) > 0:
        print(f"  Min: {valid_hurst.min():.6f}")
        print(f"  Max: {valid_hurst.max():.6f}")
        print(f"  Mean: {valid_hurst.mean():.6f}")
        print(f"  Median: {np.median(valid_hurst):.6f}")
        # Count how many are at the clamp floor
        at_floor = np.sum(valid_hurst < 0.02)
        print(f"  At floor (< 0.02): {at_floor} ({100*at_floor/len(valid_hurst):.1f}%)")
        at_half = np.sum((valid_hurst > 0.3) & (valid_hurst < 0.7))
        print(f"  Near 0.5 (0.3-0.7): {at_half} ({100*at_half/len(valid_hurst):.1f}%)")


# =============================================================================
# 6. CRITICAL CHECK: Is the spread from C++ the raw residual or log-spread?
# =============================================================================
def check_spread_interpretation(df: pd.DataFrame):
    """Check what the C++ 'Spread (Log)' actually represents."""
    spread = df["spread_log"].values

    print(f"\n{'='*80}")
    print("SPREAD INTERPRETATION CHECK")
    print(f"{'='*80}")

    # The spread values range from -0.64 to +0.005
    # This looks like log(price_a) - beta * log(price_b) - alpha
    # OR it could be something else

    # Check if these are price-level spreads (would be in thousands)
    # or log-spreads (would be small decimals)
    print(f"  Spread values sample (first 10 chrono):")
    for i in range(min(10, len(spread))):
        print(f"    [{i}] = {spread[i]:.10f}")
    print(f"  Spread values sample (last 10 chrono):")
    for i in range(max(0, len(spread)-10), len(spread)):
        print(f"    [{i}] = {spread[i]:.10f}")

    # KEY: The old bars have spread ~ -0.64, the new bars have spread ~ 0.005
    # This is a MASSIVE range change. The spread drifts from -0.64 to +0.005
    # That means the spread is NOT stationary at this scale!
    # The 64-bar window would see a tiny fraction of this range.

    print(f"\n  Spread evolution over time:")
    n = len(spread)
    for pct in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        idx = min(int(n * pct / 100), n - 1)
        print(f"    {pct:3d}% (bar {idx:>5d}): spread = {spread[idx]:.10f}")

    # Check differences at the scale of 64-bar windows
    print(f"\n  Typical bar-to-bar differences:")
    diffs = np.diff(spread)
    print(f"    Mean: {diffs.mean():.12f}")
    print(f"    Std:  {np.std(diffs):.12f}")
    print(f"    Max:  {diffs.max():.12f}")
    print(f"    Min:  {diffs.min():.12f}")

    # Check the SCALE of tau values at different time periods
    print(f"\n  --- Scale of tau at different time points ---")
    for pos_pct in [10, 50, 90]:
        idx = min(int(n * pos_pct / 100), n - 1)
        if idx < 64:
            continue
        w = spread[idx-63:idx+1]
        for lag in [2, 8, 16]:
            d = w[lag:] - w[:-lag]
            tau = np.std(d, ddof=1)
            print(f"    pos={pos_pct}%, lag={lag}: tau = {tau:.12f}")


# =============================================================================
# 7. THE CORE TEST: Run both algorithms side by side
# =============================================================================
def run_comparison(df: pd.DataFrame, period: int = 64):
    """Run C++ simulated and Python hurst on the Sierra spread data."""
    spread = df["spread_log"].values
    hurst_cpp_export = df["hurst_cpp"].values
    n = len(spread)

    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON: C++ simulated vs Python vs Sierra export")
    print(f"{'='*80}")

    # Test at several positions
    test_positions = [
        n - 1,       # most recent bar
        n - 50,      # 50 bars ago
        n - 100,     # 100 bars ago
        n - 500,     # 500 bars ago
        n - 1000,    # 1000 bars ago
        n - 5000,    # 5000 bars ago
    ]

    print(f"\n{'Position':>8} | {'Sierra H':>10} | {'C++ sim H':>10} | {'Py hurst':>10} | {'Py rolling':>10} | {'Match?':>8}")
    print("-" * 75)

    # Also compute full rolling for comparison
    spread_series = pd.Series(spread)
    py_rolling = hurst_rolling(spread_series, window=period, step=1)

    for pos in test_positions:
        if pos < period:
            continue

        h_sierra = hurst_cpp_export[pos]
        h_cpp_sim = calculate_hurst_vr_cpp(spread, pos, period)
        h_py_single = python_hurst_on_window(spread, pos, period)
        h_py_rolling = py_rolling.iloc[pos] if pos < len(py_rolling) else np.nan

        match = "YES" if abs(h_cpp_sim - h_sierra) < 0.001 else "NO"

        print(f"{pos:>8} | {h_sierra:>10.6f} | {h_cpp_sim:>10.6f} | {h_py_single:>10.6f} | {h_py_rolling:>10.6f} | {match:>8}")

    # Detailed comparison at the most recent bar
    compare_tau_detail(spread, n - 1, period)

    # Also at a mid-point
    compare_tau_detail(spread, n - 500, period)


# =============================================================================
# 8. HYPOTHESIS: Is the issue the SPREAD SCALE?
# =============================================================================
def test_scale_hypothesis(df: pd.DataFrame, period: int = 64):
    """Test if the spread values are so small that numerical issues arise."""
    spread = df["spread_log"].values
    n = len(spread)

    print(f"\n{'='*80}")
    print("SCALE HYPOTHESIS TEST")
    print(f"{'='*80}")

    # The C++ Hurst is ~0.01. The Python Hurst is ~0.4.
    # Let's check if the C++ algorithm has an issue with small values.

    # Test 1: Run C++ on the actual spread
    idx = n - 1
    h_actual = calculate_hurst_vr_cpp(spread, idx, period)
    print(f"\n  Actual spread (last bar): H = {h_actual:.6f}")

    # Test 2: Run C++ on scaled-up spread (x1000)
    spread_scaled = spread * 1000
    h_scaled = calculate_hurst_vr_cpp(spread_scaled, idx, period)
    print(f"  Scaled x1000 spread: H = {h_scaled:.6f}")

    # Test 3: Run C++ on a known mean-reverting series (OU process)
    np.random.seed(42)
    ou = np.zeros(1000)
    for i in range(1, 1000):
        ou[i] = 0.95 * ou[i-1] + np.random.randn() * 0.1
    h_ou = calculate_hurst_vr_cpp(ou, 999, period)
    h_ou_py = hurst_exponent(pd.Series(ou[-period:]), max_lag=period // 4)
    print(f"  OU process: C++ H = {h_ou:.6f}, Python H = {h_ou_py:.6f}")

    # Test 4: Run C++ on random walk
    rw = np.cumsum(np.random.randn(1000) * 0.001)
    h_rw = calculate_hurst_vr_cpp(rw, 999, period)
    h_rw_py = hurst_exponent(pd.Series(rw[-period:]), max_lag=period // 4)
    print(f"  Random walk: C++ H = {h_rw:.6f}, Python H = {h_rw_py:.6f}")

    # Test 5: Check if Sierra spread bars iterate differently
    # In Sierra, bars are stored in array order. The spreadsheet export is REVERSE.
    # But we reversed it. Let's check if the C++ iterates in the WRONG direction.
    print(f"\n  --- Direction test ---")
    h_forward = calculate_hurst_vr_cpp(spread, idx, period)
    spread_rev = spread[::-1].copy()
    h_reverse = calculate_hurst_vr_cpp(spread_rev, idx, period)
    print(f"  Forward (chrono): H = {h_forward:.6f}")
    print(f"  Reverse: H = {h_reverse:.6f}")


# =============================================================================
# 9. CHECK: float vs double precision
# =============================================================================
def test_float_precision(df: pd.DataFrame, period: int = 64):
    """Test if float32 precision in C++ causes the issue."""
    spread = df["spread_log"].values
    n = len(spread)
    idx = n - 1

    print(f"\n{'='*80}")
    print("FLOAT32 vs FLOAT64 PRECISION TEST")
    print(f"{'='*80}")

    # C++ uses SCFloatArrayRef which is float (32-bit)
    # Our Python simulation uses float64 (double)
    # Let's test with float32

    spread_f32 = spread.astype(np.float32)
    spread_f64 = spread.astype(np.float64)

    max_lag = period // 4
    start_idx = max(0, idx - period + 1)

    print(f"\n  Spread value comparison (float32 vs float64):")
    for i in [start_idx, start_idx + 10, start_idx + 30, idx - 16, idx]:
        print(f"    [{i}] f32={spread_f32[i]:.10f}  f64={spread_f64[i]:.10f}  diff={abs(spread_f64[i]-spread_f32[i]):.2e}")

    print(f"\n  Tau comparison for each lag (float32 vs float64):")
    for lag in range(2, max_lag + 1):
        # float32 computation
        total_f32 = np.float32(0.0)
        total_sq_f32 = np.float32(0.0)
        cnt = 0
        for i in range(start_idx, idx - lag + 1):
            diff = spread_f32[i + lag] - spread_f32[i]
            total_f32 += diff
            total_sq_f32 += diff * diff
            cnt += 1

        if cnt >= 5:
            mean_f32 = total_f32 / cnt
            var_f32 = (total_sq_f32 - cnt * mean_f32 * mean_f32) / (cnt - 1)
            tau_f32 = math.sqrt(max(0, float(var_f32)))
        else:
            tau_f32 = 0

        # float64 computation
        total_f64 = 0.0
        total_sq_f64 = 0.0
        cnt = 0
        for i in range(start_idx, idx - lag + 1):
            diff = float(spread_f64[i + lag]) - float(spread_f64[i])
            total_f64 += diff
            total_sq_f64 += diff * diff
            cnt += 1

        if cnt >= 5:
            mean_f64 = total_f64 / cnt
            var_f64 = (total_sq_f64 - cnt * mean_f64 * mean_f64) / (cnt - 1)
            tau_f64 = math.sqrt(max(0, var_f64))
        else:
            tau_f64 = 0

        if tau_f32 > 0 and tau_f64 > 0:
            ratio = tau_f32 / tau_f64
            print(f"    lag={lag:>2}: tau_f32={tau_f32:.12f}  tau_f64={tau_f64:.12f}  ratio={ratio:.6f}")
        else:
            print(f"    lag={lag:>2}: tau_f32={tau_f32:.12f}  tau_f64={tau_f64:.12f}  (one is zero)")

    # Compute H with float32 input
    h_f32 = calculate_hurst_vr_cpp(spread_f32.astype(np.float64), idx, period)
    h_f64 = calculate_hurst_vr_cpp(spread_f64, idx, period)
    print(f"\n  H from float32 spread: {h_f32:.6f}")
    print(f"  H from float64 spread: {h_f64:.6f}")

    # NOW test with actual float32 arithmetic throughout
    print(f"\n  --- Full float32 arithmetic simulation ---")
    h_full_f32 = calculate_hurst_vr_cpp_float32(spread_f32, idx, period)
    print(f"  H with full float32 arithmetic: {h_full_f32:.6f}")


def calculate_hurst_vr_cpp_float32(spread_f32: np.ndarray, end_index: int, period: int = 64) -> float:
    """Exact C++ replica with float32 for spread input (C++ internal is double)."""
    # Note: The C++ code uses `double` for all internal accumulators!
    # Only the spread array is float (SCFloatArrayRef)
    # So let's simulate: spread[i] is float32, but diff/sum/etc are float64
    if period < 8 or end_index < period:
        return 0.0

    max_lag = period // 4
    if max_lag < 2:
        max_lag = 2
    if max_lag > 50:
        max_lag = 50

    sum_log_lag = 0.0
    sum_log_tau = 0.0
    sum_log_lag2 = 0.0
    sum_log_lag_log_tau = 0.0
    valid_count = 0

    start_idx = end_index - period + 1
    if start_idx < 0:
        start_idx = 0

    for lag in range(2, max_lag + 1):
        total = 0.0  # double in C++
        total_sq = 0.0  # double in C++
        cnt = 0

        for i in range(start_idx, end_index - lag + 1):
            if i >= 0:
                # spread[i] is float32, but diff is computed as double
                diff = float(spread_f32[i + lag]) - float(spread_f32[i])
                total += diff
                total_sq += diff * diff
                cnt += 1

        if cnt < 5:
            continue

        mean = total / cnt
        var = (total_sq - cnt * mean * mean) / (cnt - 1)
        if var <= 0.0:
            continue
        tau = math.sqrt(var)

        log_lag = math.log(lag)
        log_tau = math.log(tau)

        sum_log_lag += log_lag
        sum_log_tau += log_tau
        sum_log_lag2 += log_lag * log_lag
        sum_log_lag_log_tau += log_lag * log_tau
        valid_count += 1

    if valid_count < 3:
        return 0.0

    n = float(valid_count)
    denom = n * sum_log_lag2 - sum_log_lag * sum_log_lag
    if abs(denom) < 1e-10:
        return 0.0

    H = (n * sum_log_lag_log_tau - sum_log_lag * sum_log_tau) / denom
    if H < 0.01:
        H = 0.01
    if H > 0.99:
        H = 0.99

    return H


# =============================================================================
# 10. CRITICAL: Check if spread[i+lag] - spread[i] goes FORWARD (future leak)
#     vs BACKWARD (correct for rolling)
# =============================================================================
def check_direction_semantics(df: pd.DataFrame, period: int = 64):
    """Check if the C++ code computes diffs in the correct direction.

    In the C++ code:
        for (int i = startIdx; i <= endIndex - lag; i++)
            diff = spread[i + lag] - spread[i];

    This means: for each position i in the window, look AHEAD by 'lag' bars.
    The difference is between a FUTURE bar and CURRENT bar.
    This is correct for variance-ratio Hurst on a chronological array.

    BUT: In Sierra Charts, array index 0 could be the OLDEST or NEWEST bar.
    If index 0 = newest (reverse), then spread[i+lag] is OLDER, not newer.
    This changes the sign of diff but NOT the variance/std!
    So the direction shouldn't matter for Hurst.
    """
    spread = df["spread_log"].values
    n = len(spread)
    idx = n - 1

    print(f"\n{'='*80}")
    print("DIRECTION SEMANTICS CHECK")
    print(f"{'='*80}")
    print("  C++ code: diff = spread[i + lag] - spread[i]")
    print("  This looks FORWARD by 'lag' bars from each position i.")
    print("  For Hurst, sign doesn't matter (we compute variance of diffs).")
    print("  But array ordering matters for WHICH bars are in the window!")


# =============================================================================
# 11. THE REAL SMOKING GUN: What if the Sierra spread array
#     is NOT what we think?
# =============================================================================
def check_sierra_array_indexing():
    """
    In Sierra Charts, SCFloatArrayRef uses 0-based indexing where:
    - Index 0 = OLDEST bar in the chart
    - Index sc.ArraySize-1 = NEWEST bar

    So endIndex = sc.Index (current bar being processed)
    startIdx = endIndex - period + 1

    The export is in REVERSE order (newest first), which we reversed.
    So our Python array should match Sierra's indexing after reversal.

    HOWEVER: The C++ function receives `spread` which is [ID3.SG1].
    If SG1 is populated from the study's perspective, the indexing should match.
    """
    print(f"\n{'='*80}")
    print("SIERRA ARRAY INDEXING NOTE")
    print(f"{'='*80}")
    print("  Sierra SCFloatArrayRef: index 0 = oldest bar")
    print("  Export: row 0 = newest bar (REVERSE)")
    print("  After Python reversal: index 0 = oldest = matches Sierra")
    print("  This should be correct.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    filepath = r"C:\Users\Bonjour\Desktop\Spread_Indice\raw\DefaultSpreadsheetStudy.txt"

    print("=" * 80)
    print("HURST EXPONENT DEBUG: C++ vs Python")
    print("=" * 80)

    # Step 1: Parse
    df = parse_sierra_export(filepath)

    # Step 2: Analyze spread values
    analyze_spread_values(df)

    # Step 3: Check spread interpretation
    check_spread_interpretation(df)

    # Step 4: Run comparison
    run_comparison(df, period=64)

    # Step 5: Test scale hypothesis
    test_scale_hypothesis(df, period=64)

    # Step 6: Float precision
    test_float_precision(df, period=64)

    # Step 7: Direction check
    check_direction_semantics(df, period=64)
    check_sierra_array_indexing()

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY OF FINDINGS")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
