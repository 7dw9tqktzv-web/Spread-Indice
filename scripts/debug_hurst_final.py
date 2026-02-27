"""Debug Hurst exponent: definitive root cause analysis.

C++ Sierra gives H~0.01 while Python pipeline gives H~0.4.
This script does a surgical comparison:

1. Loads Sierra spreadsheet export (has C++ spread + C++ Hurst)
2. Implements EXACT C++ algorithm in Python (bit-faithful)
3. Implements EXACT Python algorithm (hurst_exponent with ddof=0)
4. Compares tau-by-tau, lag-by-lag, intermediate values
5. Identifies the EXACT mathematical divergence point

KEY DIFFERENCES TO CHECK:
- np.std(ddof=0) vs C++ sample std (ddof=1)  [tau computation]
- polyfit vs manual OLS  [slope computation]
- hurst_exponent max_lag=20 default vs C++ max_lag=period//4
- hurst_rolling window=256 default vs C++ period=64
- hurst_rolling uses pd.rolling().std() = ddof=1 vs hurst_exponent np.std() = ddof=0

Author: debug session 2026-02-23
"""

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.stats.hurst import hurst_rolling


# =============================================================================
# 1. Parse Sierra Export
# =============================================================================
def parse_sierra_export(filepath: str) -> pd.DataFrame:
    """Parse Sierra spreadsheet export (tab-separated, reverse chronological)."""
    rows = []
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    header_line = lines[1].strip().split("\t")

    # Find column indices
    spread_col = None
    hurst_col = None
    for i, col in enumerate(header_line):
        if "Spread (Log)" in col:
            spread_col = i
        if "Hurst Exponent" in col:
            hurst_col = i

    if spread_col is None:
        raise ValueError("Could not find 'Spread (Log)' column in export")

    print(f"[PARSE] Spread column index: {spread_col}")
    print(f"[PARSE] Hurst column index: {hurst_col}")

    for line in lines[2:]:
        parts = line.strip().split("\t")
        if len(parts) <= spread_col:
            continue
        try:
            dt = parts[0].strip()
            spread_val = float(parts[spread_col])
            hurst_val = float(parts[hurst_col]) if hurst_col and len(parts) > hurst_col else np.nan
            rows.append({
                "datetime": dt,
                "spread_log": spread_val,
                "hurst_cpp": hurst_val,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    # Sierra export is newest-first -> reverse to chronological
    df = df.iloc[::-1].reset_index(drop=True)

    print(f"[PARSE] {len(df)} bars, from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return df


# =============================================================================
# 2. EXACT C++ Algorithm (faithful replica)
# =============================================================================
def hurst_cpp_exact(spread: np.ndarray, end_index: int, period: int) -> dict:
    """Exact replica of C++ CalculateHurstVR.

    Returns dict with H and all intermediate values for diagnosis.
    """
    result = {"H": 0.0, "lags": [], "taus": [], "log_lags": [], "log_taus": [],
              "counts": [], "valid_count": 0, "start_idx": 0, "max_lag": 0}

    if period < 8 or end_index < period:
        result["skip_reason"] = f"period={period} < 8 or end_index={end_index} < period"
        return result

    max_lag = period // 4
    if max_lag < 2:
        max_lag = 2
    if max_lag > 50:
        max_lag = 50
    result["max_lag"] = max_lag

    start_idx = end_index - period + 1
    if start_idx < 0:
        start_idx = 0
    result["start_idx"] = start_idx

    sum_log_lag = 0.0
    sum_log_tau = 0.0
    sum_log_lag2 = 0.0
    sum_log_lag_log_tau = 0.0
    valid_count = 0

    for lag in range(2, max_lag + 1):
        total = 0.0
        total_sq = 0.0
        cnt = 0

        # C++: for (int i = startIdx; i <= endIndex - lag; i++)
        for i in range(start_idx, end_index - lag + 1):
            if i >= 0:
                diff = float(spread[i + lag]) - float(spread[i])
                total += diff
                total_sq += diff * diff
                cnt += 1

        result["counts"].append(cnt)

        if cnt < 5:
            result["lags"].append(lag)
            result["taus"].append(None)
            result["log_lags"].append(None)
            result["log_taus"].append(None)
            continue

        mean = total / cnt
        # C++ uses sample variance: (sumSq - cnt * mean^2) / (cnt - 1)  =  ddof=1
        var = (total_sq - cnt * mean * mean) / (cnt - 1)
        if var <= 0.0:
            result["lags"].append(lag)
            result["taus"].append(None)
            result["log_lags"].append(None)
            result["log_taus"].append(None)
            continue

        tau = math.sqrt(var)
        log_lag = math.log(lag)
        log_tau = math.log(tau)

        result["lags"].append(lag)
        result["taus"].append(tau)
        result["log_lags"].append(log_lag)
        result["log_taus"].append(log_tau)

        sum_log_lag += log_lag
        sum_log_tau += log_tau
        sum_log_lag2 += log_lag * log_lag
        sum_log_lag_log_tau += log_lag * log_tau
        valid_count += 1

    result["valid_count"] = valid_count

    if valid_count < 3:
        result["skip_reason"] = f"valid_count={valid_count} < 3"
        return result

    n = float(valid_count)
    denom = n * sum_log_lag2 - sum_log_lag * sum_log_lag
    if abs(denom) < 1e-10:
        result["skip_reason"] = f"denom={denom} < 1e-10"
        return result

    H_raw = (n * sum_log_lag_log_tau - sum_log_lag * sum_log_tau) / denom
    result["H_raw"] = H_raw
    H = max(0.01, min(0.99, H_raw))
    result["H"] = H
    return result


# =============================================================================
# 3. EXACT Python hurst_exponent (single-shot) - annotated
# =============================================================================
def hurst_python_exact(spread: np.ndarray, end_index: int, period: int,
                       max_lag_override: int = None) -> dict:
    """Replica of Python hurst_exponent() applied on the same window as C++.

    Returns dict with H and all intermediate values for diagnosis.
    """
    result = {"H": np.nan, "lags": [], "taus": [], "log_lags": [], "log_taus": [],
              "counts": [], "valid_count": 0, "start_idx": 0, "max_lag": 0}

    start_idx = max(0, end_index - period + 1)
    window = spread[start_idx:end_index + 1]
    n = len(window)
    result["start_idx"] = start_idx
    result["window_len"] = n

    if max_lag_override is not None:
        max_lag = max_lag_override
    else:
        max_lag = 20  # DEFAULT in hurst_exponent()

    upper = min(max_lag, n // 2)
    result["max_lag"] = upper

    if n < upper + 2:
        result["skip_reason"] = f"n={n} < max_lag+2={upper+2}"
        return result

    valid_lags = []
    tau_list = []

    for lag in range(2, upper + 1):
        diffs = window[lag:] - window[:-lag]
        cnt = len(diffs)
        # Python hurst_exponent uses np.std() which is ddof=0 (population std)
        s = np.std(diffs)  # ddof=0 !!

        result["counts"].append(cnt)
        result["lags"].append(lag)

        if s > 1e-12:
            tau_list.append(s)
            valid_lags.append(lag)
            result["taus"].append(float(s))
            result["log_lags"].append(math.log(lag))
            result["log_taus"].append(math.log(float(s)))
        else:
            result["taus"].append(None)
            result["log_lags"].append(None)
            result["log_taus"].append(None)

    result["valid_count"] = len(valid_lags)

    if len(valid_lags) < 3:
        result["skip_reason"] = f"valid_lags={len(valid_lags)} < 3"
        return result

    log_lags = np.log(valid_lags)
    log_tau = np.log(tau_list)
    H_raw = np.polyfit(log_lags, log_tau, 1)[0]
    result["H_raw"] = float(H_raw)
    H = float(np.clip(H_raw, 0.01, 0.99))
    result["H"] = H
    return result


# =============================================================================
# 4. Compare Python hurst_rolling behavior
# =============================================================================
def hurst_rolling_exact(spread: np.ndarray, end_index: int, period: int) -> dict:
    """Replica of Python hurst_rolling at a specific position.

    hurst_rolling uses:
    - max_lag = min(window // 4, 50)
    - pd.Series.rolling().std() which is ddof=1
    - analytical slope formula (not polyfit)

    Returns dict with H and intermediate values.
    """
    result = {"H": np.nan, "lags": [], "taus": [], "log_lags": [], "log_taus": [],
              "counts": [], "valid_count": 0, "start_idx": 0, "max_lag": 0}

    n = len(spread)
    max_lag = min(period // 4, 50)
    result["max_lag"] = max_lag

    if n < period or max_lag < 3:
        result["skip_reason"] = "n < period or max_lag < 3"
        return result

    # For position end_index, hurst_rolling computes:
    # For each lag, the rolling std of diffs over window=(period-lag) bars
    # The tau at position end_index = rolling_std at that position

    lags = list(range(2, max_lag + 1))
    log_lags_arr = np.log(lags)

    tau_vals = []
    valid_mask = []

    for lag in lags:
        # diffs[t] = spread[t] - spread[t-lag]  (note: backward direction)
        # rolling_std over (period - lag) bars, ddof=1
        diffs = spread[lag:] - spread[:-lag]
        roll_win = period - lag
        if end_index - lag < roll_win - 1:
            tau_vals.append(np.nan)
            valid_mask.append(False)
            continue

        # pd.rolling with min_periods = window = (period-lag), ddof=1
        # At position (end_index - lag) in the diffs array,
        # which corresponds to position end_index in the original
        diffs_series = pd.Series(diffs)
        roll_std = diffs_series.rolling(roll_win, min_periods=roll_win).std()
        # The diffs array is shifted by 'lag' positions
        # diffs[j] = spread[j+lag] - spread[j], for j in [0, n-lag-1]
        # Wait -- in hurst_rolling, it's: diffs = ts[lag:] - ts[:-lag]
        # diffs has length (n - lag)
        # diffs[j] corresponds to original position (j + lag)
        # So at original position end_index, the diff index is (end_index - lag)
        diff_idx = end_index - lag
        if diff_idx < 0 or diff_idx >= len(roll_std):
            tau_vals.append(np.nan)
            valid_mask.append(False)
            continue

        tau = roll_std.iloc[diff_idx]
        tau_vals.append(tau)
        valid_mask.append(not np.isnan(tau) and tau > 1e-12)

    result["lags"] = lags
    result["taus"] = tau_vals

    tau_arr = np.array(tau_vals)
    valid = np.array(valid_mask)
    n_valid = valid.sum()
    result["valid_count"] = int(n_valid)

    if n_valid < 3:
        result["skip_reason"] = f"n_valid={n_valid} < 3"
        return result

    if n_valid == len(lags):
        # Fast path: all valid
        log_tau = np.log(tau_arr)
        sum_x = log_lags_arr.sum()
        sum_x2 = (log_lags_arr ** 2).sum()
        sum_y = log_tau.sum()
        sum_xy = (log_lags_arr * log_tau).sum()
        nv = len(lags)
        H_raw = (nv * sum_xy - sum_x * sum_y) / (nv * sum_x2 - sum_x ** 2)
    else:
        log_tau = np.log(tau_arr[valid])
        lx = log_lags_arr[valid]
        nv = n_valid
        sx = lx.sum()
        sy = log_tau.sum()
        sxy = (lx * log_tau).sum()
        sx2 = (lx ** 2).sum()
        H_raw = (nv * sxy - sx * sy) / (nv * sx2 - sx ** 2)

    result["H_raw"] = float(H_raw)
    result["H"] = float(np.clip(H_raw, 0.01, 0.99))
    result["log_lags"] = [math.log(l) for l in lags]
    result["log_taus"] = [math.log(t) if not np.isnan(t) and t > 0 else None for t in tau_vals]

    return result


# =============================================================================
# 5. Side-by-side comparison at a single bar
# =============================================================================
def compare_at_bar(spread: np.ndarray, end_index: int, period: int,
                   cpp_hurst_from_export: float):
    """Full comparison of all 3 methods at a single bar."""

    print(f"\n{'='*100}")
    print(f"BAR INDEX {end_index}  |  C++ period={period}  |  Sierra export Hurst = {cpp_hurst_from_export:.6f}")
    print(f"{'='*100}")

    # Run all three
    r_cpp = hurst_cpp_exact(spread, end_index, period)
    r_py_single = hurst_python_exact(spread, end_index, period, max_lag_override=period // 4)
    r_py_default = hurst_python_exact(spread, end_index, period, max_lag_override=20)
    r_py_rolling = hurst_rolling_exact(spread, end_index, period)

    print("\n--- RESULTS SUMMARY ---")
    print(f"  Sierra export (C++ actual):     H = {cpp_hurst_from_export:.6f}")
    print(f"  C++ simulation (Python):        H = {r_cpp['H']:.6f}  (raw: {r_cpp.get('H_raw', 'N/A')})")
    print(f"  Python hurst_exponent(maxlag={period//4}):  H = {r_py_single['H']:.6f}  (raw: {r_py_single.get('H_raw', 'N/A')})")
    print(f"  Python hurst_exponent(maxlag=20): H = {r_py_default['H']:.6f}  (raw: {r_py_default.get('H_raw', 'N/A')})")
    print(f"  Python hurst_rolling(w={period}): H = {r_py_rolling['H']:.6f}  (raw: {r_py_rolling.get('H_raw', 'N/A')})")

    # Check if C++ sim matches Sierra export
    cpp_sim_match = abs(r_cpp["H"] - cpp_hurst_from_export) < 0.005
    print(f"\n  C++ sim matches Sierra export: {'YES' if cpp_sim_match else 'NO (MISMATCH!)'}")
    if not cpp_sim_match:
        print(f"    Delta = {r_cpp['H'] - cpp_hurst_from_export:.6f}")
        print("    This suggests float32 precision or data ordering issue in export")

    # Key diagnostic: ddof=0 vs ddof=1
    print("\n--- KEY DIFFERENCE: ddof=0 (Python np.std) vs ddof=1 (C++ sample std) ---")
    print(f"  C++ max_lag = period//4 = {r_cpp['max_lag']}")
    print("  Python hurst_exponent default max_lag = 20")
    print(f"  Python hurst_rolling max_lag = min(period//4, 50) = {r_py_rolling['max_lag']}")

    # Tau-by-tau comparison
    max_lags = max(len(r_cpp["lags"]), len(r_py_single["lags"]))
    print("\n--- TAU COMPARISON (lag by lag) ---")
    print(f"  {'Lag':>4} | {'C++ tau(ddof1)':>16} | {'Py tau(ddof0)':>16} | {'Roll tau(ddof1)':>16} | {'C++ cnt':>7} | {'Py cnt':>7} | {'Ratio C/P':>10}")
    print(f"  {'-'*95}")

    n_compared = 0
    for i in range(max(r_cpp["max_lag"], r_py_single["max_lag"]) - 1):
        lag = i + 2
        cpp_tau = r_cpp["taus"][i] if i < len(r_cpp["taus"]) else None
        py_tau = r_py_single["taus"][i] if i < len(r_py_single["taus"]) else None
        roll_tau = r_py_rolling["taus"][i] if i < len(r_py_rolling["taus"]) else None
        cpp_cnt = r_cpp["counts"][i] if i < len(r_cpp["counts"]) else "-"
        py_cnt = r_py_single["counts"][i] if i < len(r_py_single["counts"]) else "-"

        cpp_s = f"{cpp_tau:.12f}" if cpp_tau is not None else "N/A"
        py_s = f"{py_tau:.12f}" if py_tau is not None else "N/A"
        roll_s = f"{roll_tau:.12f}" if (roll_tau is not None and not (isinstance(roll_tau, float) and np.isnan(roll_tau))) else "N/A"

        ratio = ""
        if cpp_tau is not None and py_tau is not None and py_tau > 0:
            ratio = f"{cpp_tau / py_tau:.6f}"
            n_compared += 1

        print(f"  {lag:>4} | {cpp_s:>16} | {py_s:>16} | {roll_s:>16} | {str(cpp_cnt):>7} | {str(py_cnt):>7} | {ratio:>10}")

    # Explain the tau ratio
    if n_compared > 0:
        # For the same data, std(ddof=1) = std(ddof=0) * sqrt(n/(n-1))
        # where n = count of diffs
        start_idx = max(0, end_index - period + 1)
        n_window = end_index - start_idx + 1
        n_diffs_lag2 = n_window - 2
        expected_ratio = math.sqrt(n_diffs_lag2 / (n_diffs_lag2 - 1))
        print(f"\n  Expected tau ratio (ddof=1/ddof=0) for n={n_diffs_lag2}: {expected_ratio:.6f}")
        print(f"  For period=64, this is sqrt(62/61) = {math.sqrt(62/61):.6f}")
        print(f"  This is a {(expected_ratio - 1) * 100:.2f}% difference in tau")
        print("  Impact on H: minimal (constant multiplier on all taus => same slope)")


# =============================================================================
# 6. The CRITICAL test: What does the Python BACKTEST pipeline produce?
# =============================================================================
def compare_with_backtest_pipeline(df: pd.DataFrame):
    """Run the actual Python backtest pipeline and compare Hurst values."""
    print(f"\n{'='*100}")
    print("COMPARISON WITH PYTHON BACKTEST PIPELINE")
    print(f"{'='*100}")

    try:
        from src.data.alignment import align_pair
        from src.data.cleaner import clean_bars
        from src.data.loader import load_instrument
        from src.data.resampler import resample_bars
        from src.hedge.ols_rolling import OLSRollingConfig, OLSRollingEstimator
        from src.metrics.dashboard import MetricsConfig, compute_all_metrics

        nq = load_instrument("NQ")
        ym = load_instrument("YM")
        nq = clean_bars(nq)
        ym = clean_bars(ym)
        nq_5m = resample_bars(nq, "5min")
        ym_5m = resample_bars(ym, "5min")
        aligned = align_pair(nq_5m, ym_5m)

        print(f"  Python data: {len(aligned.df)} bars, {aligned.df.index[0]} to {aligned.df.index[-1]}")

        # OLS with Config E
        config = OLSRollingConfig(lookback_bars=3300, zscore_window=30)
        estimator = OLSRollingEstimator(config)
        result = estimator.estimate(aligned)

        py_spread = result.spread

        # Compute Hurst with the DASHBOARD config (tres_court: hurst=64)
        metrics_config = MetricsConfig(
            adf_window=12, hurst_window=64, halflife_window=12, corr_window=6
        )
        metrics_df = compute_all_metrics(result, metrics_config)

        print(f"  Metrics computed: {len(metrics_df)} bars")
        print(f"  Hurst column: {'hurst' in metrics_df.columns}")

        # Get Hurst values at the same timestamps as Sierra export
        sierra_datetimes = pd.to_datetime(df["datetime"].values)
        sierra_hurst = df["hurst_cpp"].values
        sierra_spread = df["spread_log"].values

        # Also compute hurst_rolling directly
        hurst_roll_64 = hurst_rolling(py_spread.dropna(), window=64, step=1)

        # Match timestamps
        common_idx = py_spread.index.intersection(pd.DatetimeIndex(sierra_datetimes))
        print(f"  Common timestamps with Sierra: {len(common_idx)}")

        if len(common_idx) > 0:
            # Sample comparison
            sample_idx = common_idx[-20:]
            print("\n  --- Matched comparison (last 20 common timestamps) ---")
            print(f"  {'Datetime':>20} | {'Sierra Spr':>12} | {'Py Spr':>12} | {'Spr Diff':>12} | {'Sierra H':>9} | {'Py H roll':>9} | {'Py H dash':>9}")
            print(f"  {'-'*100}")

            for idx in sample_idx:
                si_pos = np.where(sierra_datetimes == idx)[0]
                si_h = sierra_hurst[si_pos[0]] if len(si_pos) > 0 else np.nan
                si_s = sierra_spread[si_pos[0]] if len(si_pos) > 0 else np.nan
                py_s = py_spread.loc[idx] if idx in py_spread.index else np.nan
                py_h_r = hurst_roll_64.loc[idx] if idx in hurst_roll_64.index else np.nan
                py_h_d = metrics_df.loc[idx, "hurst"] if idx in metrics_df.index else np.nan
                s_diff = py_s - si_s if not (np.isnan(py_s) or np.isnan(si_s)) else np.nan

                print(f"  {str(idx):>20} | {si_s:>12.8f} | {py_s:>12.8f} | {s_diff:>12.8f} | {si_h:>9.5f} | {py_h_r:>9.5f} | {py_h_d:>9.5f}")

            # Overall stats
            si_series = pd.Series(sierra_spread, index=sierra_datetimes)
            common2 = py_spread.index.intersection(si_series.dropna().index)
            if len(common2) > 10:
                py_c = py_spread.loc[common2].dropna()
                si_c = si_series.loc[common2].dropna()
                common3 = py_c.index.intersection(si_c.index)

                if len(common3) > 10:
                    corr = py_c.loc[common3].corr(si_c.loc[common3])
                    mean_diff = (py_c.loc[common3] - si_c.loc[common3]).mean()
                    print(f"\n  Spread correlation (Python vs Sierra): {corr:.6f}")
                    print(f"  Spread mean difference: {mean_diff:.10f}")
                    print(f"  Python spread std: {py_c.loc[common3].std():.10f}")
                    print(f"  Sierra spread std: {si_c.loc[common3].std():.10f}")

            # Hurst distribution comparison
            si_h_series = pd.Series(sierra_hurst, index=sierra_datetimes)
            common_h = hurst_roll_64.dropna().index.intersection(si_h_series.dropna().index)
            if len(common_h) > 10:
                py_h = hurst_roll_64.loc[common_h]
                si_h = si_h_series.loc[common_h]
                print(f"\n  --- Hurst distribution comparison ({len(common_h)} matched bars) ---")
                print(f"  Sierra Hurst: mean={si_h.mean():.4f}, median={si_h.median():.4f}, std={si_h.std():.4f}")
                print(f"  Python Hurst: mean={py_h.mean():.4f}, median={py_h.median():.4f}, std={py_h.std():.4f}")
                print(f"  Hurst correlation: {py_h.corr(si_h):.6f}")
                print(f"  Mean absolute Hurst difference: {(py_h - si_h).abs().mean():.6f}")

                # Percentile comparison
                for pct in [10, 25, 50, 75, 90]:
                    print(f"    P{pct}: Sierra={si_h.quantile(pct/100):.4f}, Python={py_h.quantile(pct/100):.4f}")

        else:
            print("  No common timestamps found -- checking index formats")
            print(f"  Python index sample: {py_spread.index[-3:]}")
            print(f"  Sierra index sample: {sierra_datetimes[-3:]}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# 7. Test on synthetic data: verify both algorithms agree
# =============================================================================
def test_on_synthetic():
    """Verify C++ and Python algorithms on controlled synthetic data."""
    print(f"\n{'='*100}")
    print("SYNTHETIC DATA VALIDATION")
    print(f"{'='*100}")

    np.random.seed(42)
    period = 64

    # Test 1: OU process (mean-reverting, H ~ 0.3-0.4)
    ou = np.zeros(200)
    for i in range(1, 200):
        ou[i] = 0.95 * ou[i-1] + np.random.randn() * 0.1
    end_idx = 199

    r_cpp = hurst_cpp_exact(ou, end_idx, period)
    r_py = hurst_python_exact(ou, end_idx, period, max_lag_override=period // 4)
    r_roll = hurst_rolling_exact(ou, end_idx, period)

    print("\n  OU process (theta=0.05, sigma=0.1):")
    print(f"    C++ (ddof=1):          H = {r_cpp['H']:.6f} (raw: {r_cpp.get('H_raw', 'N/A')})")
    print(f"    Python single (ddof=0): H = {r_py['H']:.6f} (raw: {r_py.get('H_raw', 'N/A')})")
    print(f"    Python rolling (ddof=1): H = {r_roll['H']:.6f} (raw: {r_roll.get('H_raw', 'N/A')})")
    print(f"    DELTA C++ vs Py single: {abs(r_cpp['H'] - r_py['H']):.6f}")

    # Test 2: Random walk (H ~ 0.5)
    rw = np.cumsum(np.random.randn(200))
    r_cpp = hurst_cpp_exact(rw, end_idx, period)
    r_py = hurst_python_exact(rw, end_idx, period, max_lag_override=period // 4)
    r_roll = hurst_rolling_exact(rw, end_idx, period)

    print("\n  Random Walk:")
    print(f"    C++ (ddof=1):          H = {r_cpp['H']:.6f}")
    print(f"    Python single (ddof=0): H = {r_py['H']:.6f}")
    print(f"    Python rolling (ddof=1): H = {r_roll['H']:.6f}")
    print(f"    DELTA C++ vs Py single: {abs(r_cpp['H'] - r_py['H']):.6f}")

    # Test 3: Small spread values (like the real data, ~0.005 scale)
    small_ou = ou * 0.001
    r_cpp = hurst_cpp_exact(small_ou, end_idx, period)
    r_py = hurst_python_exact(small_ou, end_idx, period, max_lag_override=period // 4)

    print("\n  Small-scale OU (x0.001, range ~0.001):")
    print(f"    C++ (ddof=1):          H = {r_cpp['H']:.6f}")
    print(f"    Python single (ddof=0): H = {r_py['H']:.6f}")
    print(f"    DELTA: {abs(r_cpp['H'] - r_py['H']):.6f}")
    print("    (Scale doesn't matter for Hurst -- log(c*tau) = log(c) + log(tau), constant shifts out)")


# =============================================================================
# 8. Find the ACTUAL root cause
# =============================================================================
def find_root_cause(df: pd.DataFrame):
    """Systematic root cause identification."""
    print(f"\n{'='*100}")
    print("ROOT CAUSE IDENTIFICATION")
    print(f"{'='*100}")

    spread = df["spread_log"].values
    hurst_cpp_export = df["hurst_cpp"].values
    n = len(spread)

    # HYPOTHESIS 1: The C++ simulation matches Sierra export
    # (i.e., the algorithm is correct, but parameters differ)
    print("\n--- HYPOTHESIS 1: Does C++ simulation match Sierra export? ---")
    mismatches = 0
    matches = 0
    for pos in range(64, n, 100):
        h_export = hurst_cpp_export[pos]
        h_sim = hurst_cpp_exact(spread, pos, 64)["H"]
        if abs(h_sim - h_export) > 0.01:
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH at bar {pos}: export={h_export:.6f}, sim={h_sim:.6f}")
        else:
            matches += 1

    print(f"  Matches: {matches}, Mismatches: {mismatches}")
    if mismatches == 0:
        print("  --> C++ simulation MATCHES Sierra export perfectly")
        print("  --> The issue is NOT in the C++ algorithm")
    else:
        print("  --> Some mismatches found -- may be float32 precision")

    # HYPOTHESIS 2: Python hurst_rolling uses DIFFERENT parameters
    print("\n--- HYPOTHESIS 2: Python vs C++ parameter differences ---")
    print("  C++ CalculateHurstVR:")
    print("    - period (window) = 64 (from InHurstWindow input)")
    print("    - max_lag = period // 4 = 16")
    print("    - std uses ddof=1 (sample variance)")
    print("    - slope via manual OLS (sum of logs)")
    print("")
    print("  Python hurst_exponent (single-shot):")
    print("    - max_lag = 20 (default), NOT period // 4 = 16")
    print("    - std uses ddof=0 (population variance) <-- DIFFERENT!")
    print("    - slope via np.polyfit")
    print("")
    print("  Python hurst_rolling (used in backtest):")
    print("    - window = 64 (from MetricsConfig.hurst_window)")
    print("    - max_lag = min(window // 4, 50) = 16")
    print("    - std uses ddof=1 (pd.rolling().std())")
    print("    - slope via analytical formula (same as manual OLS)")
    print("")
    print("  KEY PARAMETER DIFFERENCES:")
    print("    1. max_lag: C++ = 16, Python single = 20, Python rolling = 16")
    print("    2. ddof: C++ = 1, Python single = 0, Python rolling = 1")
    print("    3. But ddof doesn't change the slope (constant factor on all taus)")
    print("    4. max_lag difference affects which lags are in the regression")

    # HYPOTHESIS 3: The spread values are DIFFERENT
    print("\n--- HYPOTHESIS 3: Are the spread values different between Sierra and Python? ---")
    print("  Sierra spread = log(NQ) - beta * log(YM) - alpha  (computed per-bar, OLS lookback=3300)")
    print("  Python spread = same formula, same OLS lookback=3300")
    print("  BUT: Sierra uses float32 for spread storage, Python uses float64")
    print("  AND: Sierra OLS may use slightly different data alignment")

    # HYPOTHESIS 4: Hurst_rolling produces different values because of WINDOW semantics
    print("\n--- HYPOTHESIS 4: Window semantics in hurst_rolling ---")
    print("  Python hurst_rolling with window=64:")
    print("    - At position i, computes Hurst on bars [i-63, i] (64 bars)")
    print("    - But the rolling std for lag k uses rolling window = (64 - k)")
    print("    - So for lag=2, rolling std is over 62 bars ENDING at position i")
    print("    - This means each lag uses a SLIGHTLY DIFFERENT set of bars!")
    print("")
    print("  C++ CalculateHurstVR with period=64:")
    print("    - startIdx = endIndex - 63")
    print("    - For each lag, iterates i from startIdx to endIndex - lag")
    print("    - Count of diffs = period - lag = 64 - lag")
    print("    - All diffs start from startIdx (same starting point)")
    print("")
    print("  BOTH use the same starting point. The only difference is:")
    print("    - C++: std(spread[i+lag] - spread[i] for i in [start, end-lag])")
    print("    - Python rolling: std(spread[t] - spread[t-lag] for t in rolling window)")
    print("    - These are the SAME diffs, just different notation")

    # ACTUAL quantitative comparison
    print("\n--- QUANTITATIVE TEST: C++ sim vs Python rolling on Sierra data ---")
    deltas = []
    test_bars = list(range(200, n, 50))
    for pos in test_bars:
        h_cpp = hurst_cpp_exact(spread, pos, 64)["H"]
        h_roll = hurst_rolling_exact(spread, pos, 64)["H"]
        delta = h_cpp - h_roll
        deltas.append(delta)

    deltas = np.array(deltas)
    print(f"  Tested {len(deltas)} bars")
    print(f"  Mean delta (C++ - Python rolling): {np.nanmean(deltas):.6f}")
    print(f"  Std delta:  {np.nanstd(deltas):.6f}")
    print(f"  Max |delta|: {np.nanmax(np.abs(deltas)):.6f}")
    print("  Correlation: check below")

    # The C++ sim uses same ddof=1 as hurst_rolling.
    # If they match closely, the discrepancy is in the SPREAD, not the algorithm.


# =============================================================================
# 9. DEFINITIVE: What does the user ACTUALLY see?
# =============================================================================
def definitive_comparison(df: pd.DataFrame):
    """The user says C++ gives ~0.01 and Python gives ~0.4.
    Let's check: where does the Python pipeline compute ~0.4?
    Is it the SAME bars, or different bars?
    """
    spread = df["spread_log"].values
    hurst_cpp_export = df["hurst_cpp"].values
    n = len(spread)

    print(f"\n{'='*100}")
    print("DEFINITIVE: What does the user actually observe?")
    print(f"{'='*100}")

    # Distribution of C++ Hurst from Sierra
    valid_h = hurst_cpp_export[~np.isnan(hurst_cpp_export) & (hurst_cpp_export > 0)]
    print(f"\n  Sierra C++ Hurst distribution ({len(valid_h)} valid bars):")
    print(f"    Mean:   {valid_h.mean():.4f}")
    print(f"    Median: {np.median(valid_h):.4f}")
    print(f"    Std:    {valid_h.std():.4f}")
    print(f"    P10:    {np.percentile(valid_h, 10):.4f}")
    print(f"    P25:    {np.percentile(valid_h, 25):.4f}")
    print(f"    P50:    {np.percentile(valid_h, 50):.4f}")
    print(f"    P75:    {np.percentile(valid_h, 75):.4f}")
    print(f"    P90:    {np.percentile(valid_h, 90):.4f}")

    # How many bars have H < 0.1 vs H > 0.3?
    print(f"\n    H < 0.05: {np.sum(valid_h < 0.05)} ({100*np.sum(valid_h < 0.05)/len(valid_h):.1f}%)")
    print(f"    H < 0.10: {np.sum(valid_h < 0.10)} ({100*np.sum(valid_h < 0.10)/len(valid_h):.1f}%)")
    print(f"    H < 0.20: {np.sum(valid_h < 0.20)} ({100*np.sum(valid_h < 0.20)/len(valid_h):.1f}%)")
    print(f"    0.3 < H < 0.5: {np.sum((valid_h > 0.3) & (valid_h < 0.5))} ({100*np.sum((valid_h > 0.3) & (valid_h < 0.5))/len(valid_h):.1f}%)")
    print(f"    H > 0.5: {np.sum(valid_h > 0.5)} ({100*np.sum(valid_h > 0.5)/len(valid_h):.1f}%)")

    # Now compute Python hurst_rolling on the SAME Sierra spread
    print("\n  Computing Python hurst_rolling(window=64) on Sierra spread...")
    spread_series = pd.Series(spread)
    py_hurst = hurst_rolling(spread_series, window=64, step=1)
    valid_py = py_hurst.dropna().values

    print(f"  Python hurst_rolling distribution ({len(valid_py)} valid bars):")
    print(f"    Mean:   {valid_py.mean():.4f}")
    print(f"    Median: {np.median(valid_py):.4f}")
    print(f"    Std:    {valid_py.std():.4f}")
    print(f"    P10:    {np.percentile(valid_py, 10):.4f}")
    print(f"    P50:    {np.percentile(valid_py, 50):.4f}")
    print(f"    P90:    {np.percentile(valid_py, 90):.4f}")

    # Direct comparison at same indices
    print("\n  --- Direct bar-by-bar comparison ---")
    # Both arrays are indexed from 0 (chronological)
    valid_both = ~np.isnan(py_hurst.values) & ~np.isnan(hurst_cpp_export) & (hurst_cpp_export > 0)
    if valid_both.sum() > 0:
        cpp_vals = hurst_cpp_export[valid_both]
        py_vals = py_hurst.values[valid_both]
        corr = np.corrcoef(cpp_vals, py_vals)[0, 1]
        mean_diff = np.mean(py_vals - cpp_vals)
        abs_diff = np.mean(np.abs(py_vals - cpp_vals))

        print(f"    Bars with both valid: {valid_both.sum()}")
        print(f"    Correlation: {corr:.6f}")
        print(f"    Mean diff (Python - C++): {mean_diff:.6f}")
        print(f"    Mean |diff|: {abs_diff:.6f}")
        print(f"    Max |diff|: {np.max(np.abs(py_vals - cpp_vals)):.6f}")

        # If the correlation is high and mean_diff is small,
        # then the algorithms agree and the issue is elsewhere
        if corr > 0.95 and abs_diff < 0.05:
            print("\n    ** ALGORITHMS AGREE on the same data **")
            print("    The 'H~0.01 vs H~0.4' discrepancy is NOT an algorithm bug.")
            print("    It's either:")
            print("    a) Different SPREAD values (different OLS params or data)")
            print("    b) The user is comparing different time periods")
            print("    c) The Python 'H~0.4' is a median over 5+ years, not the current bar")
        elif corr > 0.8:
            print("\n    ** ALGORITHMS MOSTLY AGREE but systematic offset **")
            print(f"    Mean shift = {mean_diff:.6f}")
        else:
            print("\n    ** ALGORITHMS DISAGREE -- likely a real bug **")

            # Find largest disagreements
            diffs = py_vals - cpp_vals
            worst_idx = np.argsort(np.abs(diffs))[-5:]
            print("\n    Worst 5 disagreements:")
            bar_indices = np.where(valid_both)[0]
            for wi in worst_idx:
                bi = bar_indices[wi]
                print(f"      Bar {bi}: C++={cpp_vals[wi]:.6f}, Py={py_vals[wi]:.6f}, diff={diffs[wi]:.6f}")

    # Also test: what if we run Python hurst_rolling with DIFFERENT window?
    print("\n  --- Python hurst_rolling with different windows ---")
    for window in [64, 128, 256]:
        py_h_w = hurst_rolling(spread_series, window=window, step=1)
        valid_w = py_h_w.dropna().values
        if len(valid_w) > 0:
            print(f"    window={window}: mean={valid_w.mean():.4f}, median={np.median(valid_w):.4f}")


# =============================================================================
# 10. Float32 impact test
# =============================================================================
def test_float32_impact(df: pd.DataFrame):
    """Test how float32 storage affects Hurst computation."""
    print(f"\n{'='*100}")
    print("FLOAT32 IMPACT TEST")
    print(f"{'='*100}")

    spread_f64 = df["spread_log"].values
    spread_f32 = spread_f64.astype(np.float32)
    n = len(spread_f64)

    # The C++ spread array is SCFloatArrayRef (float32)
    # spread[i] = LogNQ[i] - beta * LogYM[i] - alpha
    # where LogNQ, LogYM are also float32
    # beta and alpha are float

    print("\n  Spread precision loss (float64 -> float32):")
    for i in [n-1, n-100, n-500, n-1000]:
        if i < 0:
            continue
        f64 = spread_f64[i]
        f32 = spread_f32[i]
        print(f"    Bar {i}: f64={f64:.15f}, f32={float(f32):.15f}, loss={abs(f64 - f32):.2e}")

    # Compute Hurst on both
    period = 64
    print("\n  Hurst comparison (float64 vs float32 spread):")
    for pos in [n-1, n-100, n-500, n-2000, n-5000]:
        if pos < period:
            continue
        h_f64 = hurst_cpp_exact(spread_f64, pos, period)["H"]
        h_f32 = hurst_cpp_exact(spread_f32.astype(np.float64), pos, period)["H"]
        h_export = df["hurst_cpp"].values[pos]
        print(f"    Bar {pos}: f64={h_f64:.6f}, f32={h_f32:.6f}, export={h_export:.6f}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    filepath = r"C:\Users\Bonjour\Desktop\Spread_Indice\raw\DefaultSpreadsheetStudy.txt"

    print("=" * 100)
    print("HURST EXPONENT DEBUG -- DEFINITIVE ROOT CAUSE ANALYSIS")
    print("=" * 100)

    # 1. Parse Sierra export
    df = parse_sierra_export(filepath)

    spread = df["spread_log"].values
    hurst_cpp = df["hurst_cpp"].values
    n = len(spread)

    # 2. Verify on synthetic data first
    test_on_synthetic()

    # 3. Single-bar detailed comparison at several positions
    test_positions = [n - 1, n - 100, n - 500, n - 2000, n - 5000]
    for pos in test_positions:
        if pos >= 64:
            compare_at_bar(spread, pos, 64, hurst_cpp[pos])

    # 4. Float32 impact
    test_float32_impact(df)

    # 5. Root cause analysis
    find_root_cause(df)

    # 6. Definitive distribution comparison
    definitive_comparison(df)

    # 7. Compare with Python backtest pipeline
    compare_with_backtest_pipeline(df)

    # =========================================================================
    # FINAL DIAGNOSIS
    # =========================================================================
    print(f"\n{'='*100}")
    print("FINAL DIAGNOSIS")
    print(f"{'='*100}")
    print("""
    The analysis above should reveal one of these root causes:

    A) ALGORITHM IS CORRECT, SAME DATA => C++ and Python agree
       The user's perception of "0.01 vs 0.4" comes from comparing
       recent bars (which genuinely have low H due to end-of-day
       oscillation) vs the 5-year median from the backtest.

    B) ALGORITHM IS CORRECT, DIFFERENT SPREAD DATA
       Sierra and Python compute different spread values due to:
       - float32 vs float64 precision
       - Different data alignment (bar-matching between NQ and YM)
       - Different OLS lookback (if Sierra config differs from Python)

    C) ALGORITHM HAS A BUG
       If C++ simulation diverges from Sierra export, or if Python
       hurst_rolling diverges from C++ simulation on the same data,
       then there's an implementation bug to fix.
    """)


if __name__ == "__main__":
    main()
