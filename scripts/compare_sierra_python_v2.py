"""
Compare Sierra C++ indicator output vs Python pipeline, bar-by-bar.

Sierra export: DefaultSpreadsheetStudy.txt (5min bars, NQ/YM, reverse chrono)
Python: Full pipeline from raw 1min data -> 5min -> OLS -> metrics -> signals

Config E parameters:
  OLS window=3300, zscore_window=30
  Metrics: adf=12, hurst=64, halflife=12, correlation=6
  Signals: z_entry=3.15, z_exit=1.00, z_stop=4.50
  Confidence: min_confidence=67%
  Entry window: 02:00-14:00 CT, flat 15:30 CT
"""

import sys
import os
import warnings

# Setup project root
PROJECT_ROOT = "C:/Users/Bonjour/Desktop/Spread_Indice"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================================
# PART 1: Load Sierra C++ export
# =====================================================================

def load_sierra_export(filepath: str) -> pd.DataFrame:
    """Parse the Sierra Charts spreadsheet export.

    Handles:
    - Reverse chronological order (newest first)
    - Double-space in datetime format
    - Metadata strings in data cells (e.g. 'Bid (read-only): 25054')
    - NaN strings for Kalman Beta
    - Tab-separated with many empty columns
    """
    print("=" * 80)
    print("LOADING SIERRA C++ EXPORT")
    print("=" * 80)

    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    print(f"  Total lines: {len(lines)}")

    # Line 1 = sheet descriptions (skip)
    # Line 2 = column headers
    headers_raw = lines[1].split("\t")

    # Data lines (3+)
    data_lines = lines[2:]
    print(f"  Data lines: {len(data_lines)}")

    # Parse each data line
    records = []
    parse_errors = 0

    for line_no, line in enumerate(data_lines, start=3):
        fields = line.split("\t")
        if len(fields) < 46:
            parse_errors += 1
            continue

        # Column 0: DateTime (format: "2026-02-20  15:30:00" -- double space)
        dt_str = fields[0].strip()
        try:
            dt = pd.Timestamp(dt_str.replace("  ", " "))
        except Exception:
            parse_errors += 1
            continue

        # Helper to safely parse float from a cell that might contain metadata
        def safe_float(idx):
            if idx >= len(fields):
                return np.nan
            val = fields[idx].strip()
            if not val:
                return np.nan
            # Handle "NaN" string
            if val.lower() == "nan":
                return np.nan
            # Handle metadata strings like "Bid (read-only): 25054"
            # These contain letters that aren't just "nan/NaN/inf"
            try:
                return float(val)
            except ValueError:
                return np.nan

        record = {
            "datetime": dt,
            "nq_close": safe_float(4),     # NQ Last (Close)
            "spread_log": safe_float(26),   # [ID3.SG1] Spread (Log)
            "zscore_ols": safe_float(27),   # [ID3.SG5] Z-Score OLS
            "adf_stat": safe_float(28),     # [ID3.SG13] ADF Statistic
            "adf_crit": safe_float(29),     # [ID3.SG14] ADF Critical -2.86
            "hurst": safe_float(30),        # [ID3.SG15] Hurst Exponent (VR)
            "correlation": safe_float(32),  # [ID3.SG17] Correlation
            "half_life": safe_float(33),    # [ID3.SG18] Half-Life (bars)
            "confidence": safe_float(34),   # [ID3.SG19] Confidence Score
            "signal_long": safe_float(36),  # [ID3.SG21] Signal LONG
            "signal_short": safe_float(37), # [ID3.SG22] Signal SHORT
            "signal_exit": safe_float(38),  # [ID3.SG23] Signal EXIT
            "kalman_beta": safe_float(39),  # [ID3.SG24] Kalman Beta
            "kalman_z": safe_float(40),     # [ID3.SG25] Kalman Z-Inn
            "kalman_conf": safe_float(41),  # [ID3.SG26] Kalman Conf
            "ols_beta": safe_float(42),     # [ID3.SG27] OLS Beta
            "ols_alpha": safe_float(43),    # [ID3.SG28] OLS Alpha
            "spread_std": safe_float(44),   # [ID3.SG29] Spread StdDev
            "trade_state": safe_float(45),  # [ID3.SG30] Trade State
        }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.set_index("datetime").sort_index()  # chronological order

    print(f"  Parse errors: {parse_errors}")
    print(f"  Parsed records: {len(df)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Summary stats
    print(f"\n  Sierra data summary:")
    print(f"    Spread (log) range: [{df['spread_log'].min():.6f}, {df['spread_log'].max():.6f}]")
    print(f"    Z-Score OLS range: [{df['zscore_ols'].min():.4f}, {df['zscore_ols'].max():.4f}]")
    print(f"    OLS Beta range: [{df['ols_beta'].min():.6f}, {df['ols_beta'].max():.6f}]")
    print(f"    Hurst range: [{df['hurst'].min():.6f}, {df['hurst'].max():.6f}]")
    print(f"    Half-Life range: [{df['half_life'].min():.2f}, {df['half_life'].max():.2f}]")
    print(f"    Confidence range: [{df['confidence'].min():.2f}, {df['confidence'].max():.2f}]")
    print(f"    Kalman Beta NaN%: {df['kalman_beta'].isna().mean()*100:.1f}%")
    print(f"    Trade State unique: {sorted(df['trade_state'].dropna().unique())}")

    return df


# =====================================================================
# PART 2: Run Python pipeline
# =====================================================================

def run_python_pipeline() -> dict:
    """Run the full Python pipeline on raw NQ/YM data and return results."""
    print("\n" + "=" * 80)
    print("RUNNING PYTHON PIPELINE")
    print("=" * 80)

    from src.data.loader import load_sierra_csv, BarData
    from src.data.cleaner import clean
    from src.data.resampler import resample_to_5min
    from src.data.alignment import align_pair, AlignedPair
    from src.hedge.ols_rolling import OLSRollingEstimator, OLSRollingConfig
    from src.metrics.dashboard import compute_all_metrics, MetricsConfig
    from src.signals.generator import SignalGenerator, SignalConfig
    from src.signals.filters import (
        compute_confidence, ConfidenceConfig,
        apply_confidence_filter, apply_entry_flat_filter,
    )
    from src.spread.pair import SpreadPair
    from src.utils.constants import Instrument
    from src.utils.time_utils import SessionConfig
    from datetime import time

    # Session config (buffer_minutes=0 for full Globex)
    session = SessionConfig(
        session_start=time(17, 30),
        session_end=time(15, 30),
        buffer_minutes=0,
        trading_start=time(2, 0),
        trading_end=time(14, 0),
    )

    # Load raw 1-min data
    nq_path = Path(PROJECT_ROOT) / "raw" / "NQH26_FUT_CME_1mn.scid_BarData.txt"
    ym_path = Path(PROJECT_ROOT) / "raw" / "YMH26_FUT_CME_1mn.scid_BarData.txt"

    print("  Loading NQ 1min...")
    nq_raw = load_sierra_csv(nq_path, Instrument.NQ)
    print(f"    {len(nq_raw.df)} bars, {nq_raw.df.index.min()} to {nq_raw.df.index.max()}")

    print("  Loading YM 1min...")
    ym_raw = load_sierra_csv(ym_path, Instrument.YM)
    print(f"    {len(ym_raw.df)} bars, {ym_raw.df.index.min()} to {ym_raw.df.index.max()}")

    # Clean
    print("  Cleaning...")
    nq_clean = clean(nq_raw, session)
    ym_clean = clean(ym_raw, session)
    print(f"    NQ clean: {len(nq_clean.df)} bars")
    print(f"    YM clean: {len(ym_clean.df)} bars")

    # Resample to 5min
    print("  Resampling to 5min...")
    nq_5m = resample_to_5min(nq_clean)
    ym_5m = resample_to_5min(ym_clean)
    print(f"    NQ 5min: {len(nq_5m.df)} bars")
    print(f"    YM 5min: {len(ym_5m.df)} bars")

    # Align pair
    print("  Aligning NQ/YM...")
    pair = SpreadPair(Instrument.NQ, Instrument.YM)
    aligned = align_pair(nq_5m, ym_5m, pair)
    print(f"    Aligned: {len(aligned.df)} bars")
    print(f"    Range: {aligned.df.index.min()} to {aligned.df.index.max()}")

    # OLS Rolling (Config E)
    print("  Computing OLS Rolling (window=3300, zscore_window=30)...")
    ols_config = OLSRollingConfig(window=3300, zscore_window=30)
    estimator = OLSRollingEstimator(ols_config)
    result = estimator.estimate(aligned)

    first_valid_ols = result.beta.first_valid_index()
    print(f"    First valid OLS output: {first_valid_ols}")
    print(f"    Beta range: [{result.beta.min():.6f}, {result.beta.max():.6f}]")

    # Compute spread std (same as Python: rolling std over zscore_window)
    spread_mu = result.spread.rolling(30).mean()
    spread_std = result.spread.rolling(30).std()

    # Compute alpha (already in HedgeResult? No -- we need to compute it)
    log_a = np.log(aligned.df["close_a"])
    log_b = np.log(aligned.df["close_b"])
    mean_a = log_a.rolling(3300).mean()
    mean_b = log_b.rolling(3300).mean()
    alpha = mean_a - result.beta * mean_b

    # Metrics (tres_court profile)
    print("  Computing metrics (adf=12, hurst=64, halflife=12, corr=6)...")
    metrics_config = MetricsConfig(
        adf_window=12,
        hurst_window=64,
        halflife_window=12,
        correlation_window=6,
    )
    metrics = compute_all_metrics(
        result.spread,
        aligned.df["close_a"],
        aligned.df["close_b"],
        metrics_config,
    )
    print(f"    ADF range: [{metrics['adf_stat'].min():.4f}, {metrics['adf_stat'].max():.4f}]")
    print(f"    Hurst range: [{metrics['hurst'].min():.4f}, {metrics['hurst'].max():.4f}]")
    print(f"    Half-life range: [{metrics['half_life'].min():.2f}, {metrics['half_life'].max():.2f}]")
    print(f"    Correlation range: [{metrics['correlation'].min():.4f}, {metrics['correlation'].max():.4f}]")

    # Confidence scoring
    print("  Computing confidence scores...")
    conf_config = ConfidenceConfig(min_confidence=67.0)
    confidence = compute_confidence(metrics, conf_config)
    print(f"    Confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")

    # Signals (raw, no window filter)
    print("  Generating signals (z_entry=3.15, z_exit=1.00, z_stop=4.50)...")
    sig_config = SignalConfig(z_entry=3.15, z_exit=1.00, z_stop=4.50)
    sig_gen = SignalGenerator(sig_config)
    raw_signals = sig_gen.generate(result.zscore)

    # Apply confidence filter
    filtered_signals = apply_confidence_filter(raw_signals, metrics, conf_config)

    # Apply entry/flat filter (02:00-14:00, flat 15:30)
    final_signals = apply_entry_flat_filter(
        filtered_signals,
        entry_start=time(2, 0),
        entry_end=time(14, 0),
        flat_time=time(15, 30),
    )

    print(f"    Raw signals: {(raw_signals != 0).sum()} non-zero")
    print(f"    After confidence: {(filtered_signals != 0).sum()} non-zero")
    print(f"    After window: {(final_signals != 0).sum()} non-zero")

    return {
        "aligned": aligned,
        "beta": result.beta,
        "alpha": alpha,
        "spread": result.spread,
        "zscore": result.zscore,
        "spread_std": spread_std,
        "metrics": metrics,
        "confidence": confidence,
        "raw_signals": raw_signals,
        "final_signals": final_signals,
    }


# =====================================================================
# PART 3: Compare bar-by-bar
# =====================================================================

def compare_metrics(sierra: pd.DataFrame, python: dict):
    """Bar-by-bar comparison of Sierra C++ vs Python pipeline."""
    print("\n" + "=" * 80)
    print("BAR-BY-BAR COMPARISON")
    print("=" * 80)

    # Find overlapping timestamps
    sierra_idx = sierra.index
    python_idx = python["beta"].dropna().index

    overlap = sierra_idx.intersection(python_idx)
    print(f"\n  Sierra bars: {len(sierra_idx)}")
    print(f"  Python valid bars (OLS computed): {len(python_idx)}")
    print(f"  Overlapping bars: {len(overlap)}")

    if len(overlap) == 0:
        print("  ERROR: No overlapping bars found!")
        return

    print(f"  Overlap range: {overlap.min()} to {overlap.max()}")

    # Filter to overlap and where Sierra has non-zero data (past warmup)
    # Sierra warmup: early bars have spread_log near -0.64 (initial) and
    # all metrics at 0 or 0.5 (defaults)
    # We need bars where OLS Beta is not 0 and not 1 (initial state)
    sierra_valid = sierra.loc[overlap].copy()
    python_beta = python["beta"].loc[overlap]

    # Both must have valid (non-NaN, non-zero for beta) values
    valid_mask = (
        sierra_valid["ols_beta"].notna()
        & (sierra_valid["ols_beta"] != 0)
        & (sierra_valid["ols_beta"] != 1)
        & python_beta.notna()
    )
    valid_idx = overlap[valid_mask.values]
    print(f"  Valid comparison bars (both have OLS data): {len(valid_idx)}")

    if len(valid_idx) == 0:
        print("  ERROR: No valid comparison bars!")
        # Show some data to debug
        print("\n  Sierra OLS Beta sample (first 20 overlap bars):")
        print(sierra_valid["ols_beta"].head(20))
        print("\n  Python Beta sample (first 20 overlap bars):")
        print(python_beta.head(20))
        return

    print(f"  Valid range: {valid_idx.min()} to {valid_idx.max()}")

    # Extract aligned data
    s = sierra.loc[valid_idx]
    p_beta = python["beta"].loc[valid_idx]
    p_alpha = python["alpha"].loc[valid_idx]
    p_spread = python["spread"].loc[valid_idx]
    p_zscore = python["zscore"].loc[valid_idx]
    p_spread_std = python["spread_std"].loc[valid_idx]
    p_metrics = python["metrics"].loc[valid_idx]
    p_confidence = python["confidence"].loc[valid_idx]
    p_signals = python["final_signals"].loc[valid_idx]

    # Define comparison pairs: (name, sierra_col, python_series, tolerance_pct)
    comparisons = [
        ("OLS Beta",       s["ols_beta"],      p_beta,                     0.01),
        ("OLS Alpha",      s["ols_alpha"],      p_alpha,                   0.05),
        ("Spread (Log)",   s["spread_log"],     p_spread,                  0.05),
        ("Z-Score OLS",    s["zscore_ols"],      p_zscore,                 0.05),
        ("Spread StdDev",  s["spread_std"],      p_spread_std,             0.05),
        ("ADF Statistic",  s["adf_stat"],        p_metrics["adf_stat"],    0.10),
        ("Hurst",          s["hurst"],            p_metrics["hurst"],       0.10),
        ("Correlation",    s["correlation"],      p_metrics["correlation"], 0.05),
        ("Half-Life",      s["half_life"],        p_metrics["half_life"],   0.10),
        ("Confidence",     s["confidence"],       p_confidence,             0.10),
    ]

    print("\n" + "-" * 80)
    print(f"  {'Metric':<20} {'N valid':>8} {'MAE':>12} {'MaxAE':>12} "
          f"{'MAPE%':>8} {'Corr':>8} {'Status':>10}")
    print("-" * 80)

    results = {}
    for name, sc, py, tol in comparisons:
        # Align on common valid indices
        mask = sc.notna() & py.notna()
        if name in ("ADF Statistic", "Hurst", "Half-Life", "Confidence"):
            # Also skip where Sierra has default/zero values (warmup)
            if name == "Hurst":
                mask = mask & (sc != 0.5)  # 0.5 is the default hidden line
            elif name == "Half-Life":
                mask = mask & (sc != 0)
            elif name == "Confidence":
                mask = mask & (sc != 0) & (py != 0)

        n = mask.sum()
        if n < 10:
            print(f"  {name:<20} {n:>8} {'N/A':>12} {'N/A':>12} "
                  f"{'N/A':>8} {'N/A':>8} {'SKIP':>10}")
            results[name] = {"n": n, "status": "SKIP"}
            continue

        sc_v = sc[mask].values.astype(float)
        py_v = py[mask].values.astype(float)

        # Compute metrics
        diff = np.abs(sc_v - py_v)
        mae = np.mean(diff)
        max_ae = np.max(diff)

        # MAPE (avoid div by zero)
        denom = np.abs(py_v)
        denom[denom < 1e-10] = 1e-10
        mape = np.mean(diff / denom) * 100

        # Pearson correlation
        if np.std(sc_v) > 1e-12 and np.std(py_v) > 1e-12:
            corr = np.corrcoef(sc_v, py_v)[0, 1]
        else:
            corr = np.nan

        # Status
        if corr > 0.999 and mape < 1.0:
            status = "MATCH"
        elif corr > 0.99 and mape < 5.0:
            status = "CLOSE"
        elif corr > 0.95:
            status = "DRIFT"
        elif np.isnan(corr) or corr < 0.5:
            status = "FAIL"
        else:
            status = "CHECK"

        print(f"  {name:<20} {n:>8} {mae:>12.8f} {max_ae:>12.8f} "
              f"{mape:>7.2f}% {corr:>8.6f} {status:>10}")

        results[name] = {
            "n": n, "mae": mae, "max_ae": max_ae, "mape": mape,
            "corr": corr, "status": status,
            "sc": sc_v, "py": py_v, "idx": sc[mask].index,
        }

    # =====================================================================
    # Detailed analysis for each metric
    # =====================================================================

    print("\n" + "=" * 80)
    print("DETAILED METRIC ANALYSIS")
    print("=" * 80)

    for name, info in results.items():
        if info["status"] == "SKIP":
            continue

        print(f"\n--- {name} ---")
        sc_v = info["sc"]
        py_v = info["py"]
        idx = info["idx"]
        diff = sc_v - py_v

        # Distribution of differences
        print(f"  Difference stats: mean={np.mean(diff):.8f}, std={np.std(diff):.8f}")
        print(f"  Percentiles: P5={np.percentile(diff,5):.8f}, "
              f"P50={np.percentile(diff,50):.8f}, P95={np.percentile(diff,95):.8f}")

        # Show 10 sample bars (spread across time)
        n = len(idx)
        sample_indices = np.linspace(0, n - 1, min(10, n), dtype=int)

        print(f"\n  {'Timestamp':<22} {'Sierra':>14} {'Python':>14} {'Diff':>14} {'Rel%':>8}")
        print("  " + "-" * 74)
        for si in sample_indices:
            ts = idx[si]
            sv = sc_v[si]
            pv = py_v[si]
            d = sv - pv
            rel = abs(d / pv) * 100 if abs(pv) > 1e-10 else 0
            print(f"  {str(ts):<22} {sv:>14.8f} {pv:>14.8f} {d:>14.8f} {rel:>7.3f}%")

    # =====================================================================
    # PART 4: Signal/Trade State comparison
    # =====================================================================

    print("\n" + "=" * 80)
    print("SIGNAL / TRADE STATE COMPARISON")
    print("=" * 80)

    # Sierra trade_state: 0=FLAT, 1=LONG(?), -1=SHORT(?) -- inspect
    # Sierra signal_long/signal_short/signal_exit: separate subgraphs
    # Python: single signal series {-1, 0, +1}

    # Reconstruct Sierra signal from trade_state
    s_state = s["trade_state"]
    p_sig = p_signals

    # Compare trade states
    state_mask = s_state.notna() & p_sig.notna()
    if state_mask.sum() > 0:
        s_sv = s_state[state_mask].values.astype(float)
        p_sv = p_sig[state_mask].values.astype(float)

        match_count = np.sum(s_sv == p_sv)
        total = len(s_sv)
        print(f"\n  Trade State comparison: {match_count}/{total} match ({match_count/total*100:.1f}%)")

        # Show mismatches
        mismatches = np.where(s_sv != p_sv)[0]
        if len(mismatches) > 0:
            print(f"  Mismatches: {len(mismatches)}")
            print(f"\n  {'Timestamp':<22} {'Sierra':>8} {'Python':>8} {'S_ZScore':>10} {'P_ZScore':>10}")
            print("  " + "-" * 60)
            show = min(20, len(mismatches))
            for i in range(show):
                mi = mismatches[i]
                ts = s_state[state_mask].index[mi]
                print(f"  {str(ts):<22} {s_sv[mi]:>8.0f} {p_sv[mi]:>8.0f} "
                      f"{s.loc[ts, 'zscore_ols']:>10.4f} {p_zscore.loc[ts]:>10.4f}")
        else:
            print("  All trade states match perfectly!")
    else:
        print("  No valid trade state data for comparison")

    # =====================================================================
    # PART 5: Known Issues Investigation
    # =====================================================================

    print("\n" + "=" * 80)
    print("KNOWN ISSUES INVESTIGATION")
    print("=" * 80)

    # Issue 1: Hurst on recent bars ~0.012 in Sierra vs ~0.35-0.50 in Python
    print("\n--- ISSUE 1: Hurst Exponent Divergence ---")
    hurst_mask = s["hurst"].notna() & (s["hurst"] != 0.5) & p_metrics["hurst"].notna()
    if hurst_mask.sum() > 0:
        s_hurst = s.loc[hurst_mask, "hurst"]
        p_hurst = p_metrics.loc[hurst_mask, "hurst"]

        # Look at last 50 bars
        recent = min(50, len(s_hurst))
        s_recent = s_hurst.iloc[-recent:]
        p_recent = p_hurst.iloc[-recent:]

        print(f"  Last {recent} bars:")
        print(f"    Sierra Hurst: mean={s_recent.mean():.6f}, "
              f"min={s_recent.min():.6f}, max={s_recent.max():.6f}")
        print(f"    Python Hurst: mean={p_recent.mean():.6f}, "
              f"min={p_recent.min():.6f}, max={p_recent.max():.6f}")
        print(f"    Ratio (Sierra/Python): {s_recent.mean()/p_recent.mean():.4f}")

        # Check distribution of very low Hurst in Sierra
        very_low = (s_hurst < 0.05).sum()
        low = ((s_hurst >= 0.05) & (s_hurst < 0.20)).sum()
        normal = ((s_hurst >= 0.20) & (s_hurst < 0.55)).sum()
        high = (s_hurst >= 0.55).sum()
        print(f"\n  Sierra Hurst distribution:")
        print(f"    < 0.05 (suspicious): {very_low} ({very_low/len(s_hurst)*100:.1f}%)")
        print(f"    0.05-0.20 (very low): {low} ({low/len(s_hurst)*100:.1f}%)")
        print(f"    0.20-0.55 (normal):  {normal} ({normal/len(s_hurst)*100:.1f}%)")
        print(f"    >= 0.55 (trending):  {high} ({high/len(s_hurst)*100:.1f}%)")

        # Show some recent comparisons
        print(f"\n  Recent Hurst comparison (last 10 bars):")
        print(f"  {'Timestamp':<22} {'Sierra':>10} {'Python':>10} {'Diff':>10}")
        print("  " + "-" * 54)
        for i in range(-min(10, len(s_hurst)), 0):
            ts = s_hurst.index[i]
            print(f"  {str(ts):<22} {s_hurst.iloc[i]:>10.6f} {p_hurst.iloc[i]:>10.6f} "
                  f"{s_hurst.iloc[i] - p_hurst.iloc[i]:>10.6f}")

    # Issue 2: Half-Life = 0 or 1 in Sierra vs ~10-60 in Python
    print("\n--- ISSUE 2: Half-Life Divergence ---")
    hl_mask = s["half_life"].notna() & p_metrics["half_life"].notna()
    if hl_mask.sum() > 0:
        s_hl = s.loc[hl_mask, "half_life"]
        p_hl = p_metrics.loc[hl_mask, "half_life"]

        zero_hl = (s_hl == 0).sum()
        one_hl = (s_hl == 1).sum()
        normal_hl = ((s_hl > 1) & (s_hl < 200)).sum()
        huge_hl = (s_hl >= 200).sum()
        print(f"  Sierra Half-Life distribution:")
        print(f"    = 0: {zero_hl} ({zero_hl/len(s_hl)*100:.1f}%)")
        print(f"    = 1: {one_hl} ({one_hl/len(s_hl)*100:.1f}%)")
        print(f"    1-200 (normal): {normal_hl} ({normal_hl/len(s_hl)*100:.1f}%)")
        print(f"    >= 200 (huge): {huge_hl} ({huge_hl/len(s_hl)*100:.1f}%)")

        # Recent comparison
        print(f"\n  Recent Half-Life comparison (last 10 bars):")
        print(f"  {'Timestamp':<22} {'Sierra':>10} {'Python':>10} {'Diff':>10}")
        print("  " + "-" * 54)
        for i in range(-min(10, len(s_hl)), 0):
            ts = s_hl.index[i]
            print(f"  {str(ts):<22} {s_hl.iloc[i]:>10.2f} {p_hl.iloc[i]:>10.2f} "
                  f"{s_hl.iloc[i] - p_hl.iloc[i]:>10.2f}")

    # Issue 3: Kalman Beta = NaN everywhere
    print("\n--- ISSUE 3: Kalman Beta Status ---")
    kalman_total = len(s["kalman_beta"])
    kalman_nan = s["kalman_beta"].isna().sum()
    kalman_zero = (s["kalman_beta"] == 0).sum()
    kalman_valid = kalman_total - kalman_nan - kalman_zero
    print(f"  Total bars: {kalman_total}")
    print(f"  NaN: {kalman_nan} ({kalman_nan/kalman_total*100:.1f}%)")
    print(f"  Zero: {kalman_zero} ({kalman_zero/kalman_total*100:.1f}%)")
    print(f"  Valid (non-NaN, non-zero): {kalman_valid} ({kalman_valid/kalman_total*100:.1f}%)")

    if kalman_valid > 0:
        valid_kb = s["kalman_beta"].dropna()
        valid_kb = valid_kb[valid_kb != 0]
        print(f"  Valid Kalman Beta range: [{valid_kb.min():.6f}, {valid_kb.max():.6f}]")
    else:
        print("  CONFIRMED: Kalman Beta is NaN everywhere -- Kalman filter likely not initialized")

    # Issue 4: Kalman Conf = same as OLS Confidence?
    print("\n--- ISSUE 4: Kalman Conf vs OLS Confidence ---")
    both_valid = s["kalman_conf"].notna() & s["confidence"].notna()
    if both_valid.sum() > 0:
        kc = s.loc[both_valid, "kalman_conf"]
        oc = s.loc[both_valid, "confidence"]
        identical = (np.abs(kc.values - oc.values) < 1e-6).sum()
        print(f"  Bars with both valid: {both_valid.sum()}")
        print(f"  Identical values: {identical} ({identical/both_valid.sum()*100:.1f}%)")
        if identical / both_valid.sum() > 0.95:
            print("  CONFIRMED: Kalman Conf is a copy of OLS Confidence (code bug)")
        else:
            print("  Kalman Conf differs from OLS Confidence -- may be independently computed")

    # =====================================================================
    # PART 6: ROOT CAUSE ANALYSIS -- Data alignment + OLS parity
    # =====================================================================

    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS -- DATA ALIGNMENT")
    print("=" * 80)

    # Step 1: Compare NQ close prices to verify both systems see the same data
    nq_close_py = python["aligned"].df["close_a"]
    nq_close_sierra = sierra["nq_close"]
    price_overlap = nq_close_sierra.index.intersection(nq_close_py.index)

    if len(price_overlap) > 0:
        s_prices = nq_close_sierra.loc[price_overlap]
        p_prices = nq_close_py.loc[price_overlap]
        price_match = (s_prices == p_prices).sum()
        price_close = (np.abs(s_prices - p_prices) < 0.50).sum()
        print(f"\n  NQ Close price comparison ({len(price_overlap)} overlapping bars):")
        print(f"    Exact match: {price_match}/{len(price_overlap)} "
              f"({price_match/len(price_overlap)*100:.1f}%)")
        print(f"    Within $0.50: {price_close}/{len(price_overlap)} "
              f"({price_close/len(price_overlap)*100:.1f}%)")

        # Show sample mismatches
        price_diff = np.abs(s_prices - p_prices)
        big_diffs = price_diff[price_diff > 0.50]
        if len(big_diffs) > 0:
            print(f"    Large diffs (> $0.50): {len(big_diffs)} bars")
            print(f"\n    {'Timestamp':<22} {'Sierra':>12} {'Python':>12} {'Diff':>10}")
            print("    " + "-" * 58)
            for ts in big_diffs.index[:10]:
                print(f"    {str(ts):<22} {s_prices.loc[ts]:>12.2f} "
                      f"{p_prices.loc[ts]:>12.2f} {price_diff.loc[ts]:>10.2f}")
        else:
            print("    No large price differences -- data alignment is GOOD")

    # Step 2: Temporal convergence analysis -- does Beta converge over time?
    print("\n  OLS Beta temporal convergence analysis:")
    if "OLS Beta" in results and results["OLS Beta"]["status"] != "SKIP":
        info = results["OLS Beta"]
        idx = info["idx"]
        sc_v = info["sc"]
        py_v = info["py"]

        # Split into weekly chunks
        n = len(idx)
        chunk_size = n // 7 if n > 70 else max(10, n)
        print(f"\n    {'Period':<30} {'N':>6} {'MAE':>10} {'Corr':>8} {'MAPE%':>8}")
        print("    " + "-" * 64)
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk_sc = sc_v[i:end]
            chunk_py = py_v[i:end]
            chunk_diff = np.abs(chunk_sc - chunk_py)
            chunk_mae = np.mean(chunk_diff)
            denom = np.abs(chunk_py)
            denom[denom < 1e-10] = 1e-10
            chunk_mape = np.mean(chunk_diff / denom) * 100
            if np.std(chunk_sc) > 1e-12 and np.std(chunk_py) > 1e-12:
                chunk_corr = np.corrcoef(chunk_sc, chunk_py)[0, 1]
            else:
                chunk_corr = np.nan
            ts_start = str(idx[i].date())
            ts_end = str(idx[min(end - 1, n - 1)].date())
            print(f"    {ts_start} to {ts_end:<14} {end-i:>6} "
                  f"{chunk_mae:>10.6f} {chunk_corr:>8.4f} {chunk_mape:>7.2f}%")

    # Step 3: Check Sierra chart data length hypothesis
    print("\n  DATA HISTORY HYPOTHESIS:")
    print("    Python uses raw 1min data from 2020-12-07 (5+ years)")
    print("    Sierra chart data length depends on chart configuration")
    print("    OLS window = 3300 bars = ~12.5 trading days")
    print("    If Sierra chart has LESS historical data loaded, the OLS lookback")
    print("    window would start at a different point, producing different Beta.")
    print("    This is the most likely root cause of the Beta divergence.")
    print()
    print("    The SPREAD and Z-SCORE are computed FROM Beta, so Beta divergence")
    print("    cascades to all downstream metrics. ADF, Hurst, Half-Life operate")
    print("    on the SPREAD, so they inherit the divergence too.")
    print()
    print("    Correlation operates on raw log-prices (not spread), which is why")
    print("    it has near-perfect parity (r=0.996).")

    # Step 4: Check if Sierra has the same OLS window
    print("\n  OLS WINDOW CHECK:")
    # Sierra first non-zero Beta is at which bar index?
    sierra_first_beta = sierra.loc[
        (sierra["ols_beta"].notna()) & (sierra["ols_beta"] != 0) & (sierra["ols_beta"] != 1)
    ]
    if len(sierra_first_beta) > 0:
        first_ts = sierra_first_beta.index.min()
        # Count bars from Sierra start to first valid Beta
        sierra_bars_before = (sierra.index < first_ts).sum()
        print(f"    Sierra first valid Beta at: {first_ts}")
        print(f"    Sierra bars before first Beta: {sierra_bars_before}")
        print(f"    Expected (OLS window): 3300")
        if abs(sierra_bars_before - 3300) < 100:
            print("    MATCH -- Sierra appears to use window=3300")
        else:
            print(f"    MISMATCH -- Sierra uses ~{sierra_bars_before} bar warmup, not 3300")
            print("    This suggests Sierra chart has different data depth")

    # Step 5: Restricted comparison on LAST 2 WEEKS only (where OLS should converge)
    print("\n" + "=" * 80)
    print("RESTRICTED COMPARISON: LAST 2 WEEKS ONLY")
    print("=" * 80)

    from datetime import datetime
    cutoff = pd.Timestamp("2026-02-05")
    late_idx = valid_idx[valid_idx >= cutoff]
    print(f"\n  Bars from {cutoff.date()} onward: {len(late_idx)}")

    if len(late_idx) > 50:
        s_late = sierra.loc[late_idx]
        comparisons_late = [
            ("OLS Beta",       s_late["ols_beta"],      python["beta"].loc[late_idx]),
            ("OLS Alpha",      s_late["ols_alpha"],      python["alpha"].loc[late_idx]),
            ("Spread (Log)",   s_late["spread_log"],     python["spread"].loc[late_idx]),
            ("Z-Score OLS",    s_late["zscore_ols"],      python["zscore"].loc[late_idx]),
            ("Spread StdDev",  s_late["spread_std"],      python["spread_std"].loc[late_idx]),
            ("ADF Statistic",  s_late["adf_stat"],        python["metrics"].loc[late_idx, "adf_stat"]),
            ("Hurst",          s_late["hurst"],            python["metrics"].loc[late_idx, "hurst"]),
            ("Correlation",    s_late["correlation"],      python["metrics"].loc[late_idx, "correlation"]),
            ("Half-Life",      s_late["half_life"],        python["metrics"].loc[late_idx, "half_life"]),
            ("Confidence",     s_late["confidence"],       python["confidence"].loc[late_idx]),
        ]

        print(f"\n  {'Metric':<20} {'N':>6} {'MAE':>12} {'Corr':>8} {'MAPE%':>8}")
        print("  " + "-" * 56)
        for name, sc, py in comparisons_late:
            mask = sc.notna() & py.notna()
            if name == "Half-Life":
                mask = mask & (sc != 0)
            if name == "Confidence":
                mask = mask & (sc != 0) & (py != 0)
            n = mask.sum()
            if n < 10:
                print(f"  {name:<20} {n:>6} {'N/A':>12} {'N/A':>8} {'N/A':>8}")
                continue
            sc_v = sc[mask].values.astype(float)
            py_v = py[mask].values.astype(float)
            diff = np.abs(sc_v - py_v)
            mae = np.mean(diff)
            denom = np.abs(py_v)
            denom[denom < 1e-10] = 1e-10
            mape = np.mean(diff / denom) * 100
            if np.std(sc_v) > 1e-12 and np.std(py_v) > 1e-12:
                corr = np.corrcoef(sc_v, py_v)[0, 1]
            else:
                corr = np.nan
            print(f"  {name:<20} {n:>6} {mae:>12.8f} {corr:>8.4f} {mape:>7.2f}%")

    # =====================================================================
    # PART 7: Confidence scoring breakdown
    # =====================================================================

    print("\n" + "=" * 80)
    print("CONFIDENCE SCORING BREAKDOWN")
    print("=" * 80)

    # The confidence depends on ADF, Hurst, Correlation, Half-Life
    # If Hurst and Half-Life have C++ bugs, confidence will diverge
    conf_mask = (
        s["confidence"].notna()
        & (s["confidence"] != 0)
        & p_confidence.notna()
        & (p_confidence != 0)
    )
    if conf_mask.sum() > 10:
        s_conf = s.loc[conf_mask, "confidence"]
        p_conf = p_confidence[conf_mask]

        # Check which component drives the difference
        # If ADF+Corr match but Hurst+HL differ, the error source is clear
        for metric_name, s_col, p_col in [
            ("ADF", "adf_stat", "adf_stat"),
            ("Hurst", "hurst", "hurst"),
            ("Correlation", "correlation", "correlation"),
            ("Half-Life", "half_life", "half_life"),
        ]:
            sm = s.loc[conf_mask, s_col]
            pm = p_metrics.loc[conf_mask, p_col]
            both_ok = sm.notna() & pm.notna()
            if both_ok.sum() > 10:
                r = np.corrcoef(sm[both_ok].values, pm[both_ok].values)[0, 1]
                mae = np.mean(np.abs(sm[both_ok].values - pm[both_ok].values))
                print(f"  {metric_name}: corr={r:.6f}, MAE={mae:.6f} "
                      f"(S mean={sm[both_ok].mean():.4f}, P mean={pm[both_ok].mean():.4f})")
            else:
                print(f"  {metric_name}: insufficient data for comparison")

    # =====================================================================
    # PART 8: Summary and recommendations
    # =====================================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n  Status by metric:")
    for name, info in results.items():
        status = info["status"]
        marker = {
            "MATCH": "[OK]  ",
            "CLOSE": "[~]   ",
            "DRIFT": "[!]   ",
            "FAIL":  "[FAIL]",
            "CHECK": "[??]  ",
            "SKIP":  "[SKIP]",
        }.get(status, "[??]  ")
        if status != "SKIP":
            print(f"    {marker} {name}: corr={info['corr']:.6f}, MAPE={info['mape']:.2f}%")
        else:
            print(f"    {marker} {name}: insufficient data")

    # Count issues
    fails = sum(1 for v in results.values() if v["status"] in ("FAIL", "CHECK"))
    matches = sum(1 for v in results.values() if v["status"] in ("MATCH", "CLOSE"))
    print(f"\n  Matches: {matches}, Issues: {fails}")

    if fails > 0:
        print("\n  RECOMMENDATIONS:")
        if "Hurst" in results and results["Hurst"]["status"] in ("FAIL", "CHECK", "DRIFT"):
            print("    - Hurst: Check C++ variance-ratio implementation")
            print("      - Ensure using std(diffs) not var(diffs)")
            print("      - Check polyfit(log_lags, log_tau) slope computation")
            print("      - Verify max_lag = min(window/4, 50)")
            print("      - Clip result to [0.01, 0.99]")
        if "Half-Life" in results and results["Half-Life"]["status"] in ("FAIL", "CHECK", "DRIFT"):
            print("    - Half-Life: Check C++ AR(1) implementation")
            print("      - b = Cov(Z(t), Z(t-1)) / Var(Z(t-1))")
            print("      - HL = -ln(2) / ln(b), valid only for 0 < b < 1")
            print("      - Check rolling window alignment")

    return results


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("Sierra C++ vs Python Pipeline -- Bar-by-Bar Comparison")
    print(f"Config E: OLS=3300, ZW=30, z_entry=3.15, z_exit=1.00, z_stop=4.50")
    print(f"Metrics: adf=12, hurst=64, halflife=12, corr=6")
    print()

    # Load Sierra export
    sierra_path = os.path.join(PROJECT_ROOT, "raw", "DefaultSpreadsheetStudy.txt")
    sierra = load_sierra_export(sierra_path)

    # Run Python pipeline
    python = run_python_pipeline()

    # Compare
    results = compare_metrics(sierra, python)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
