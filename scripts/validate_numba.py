"""Validate numba-compiled functions produce identical results to original Python loops."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_grid, run_backtest_vectorized
from src.signals.filters import (
    ConfidenceConfig,
    _apply_conf_filter_numba,
    apply_time_stop,
    apply_window_filter_numba,
    compute_confidence,
)
from src.signals.generator import generate_signals_numba

# ============================================================
# Original Python implementations (reference)
# ============================================================

def generate_signals_python(z_vals, z_entry, z_exit, z_stop):
    """Original Python signal generator loop."""
    n = len(z_vals)
    signals = np.zeros(n, dtype=np.int8)
    state = 0
    for t in range(n):
        zt = z_vals[t]
        if np.isnan(zt):
            state = 0
            signals[t] = 0
            continue
        if state == 0:
            if zt < -z_entry:
                state = 1
            elif zt > z_entry:
                state = -1
        elif state == 1:
            if zt > -z_exit:
                state = 0
            elif zt < -z_stop:
                state = 2
        elif state == -1:
            if zt < z_exit:
                state = 0
            elif zt > z_stop:
                state = 2
        elif state == 2:
            if abs(zt) < z_exit:
                state = 0
        signals[t] = state if state in (1, -1) else 0
    return signals


def compute_confidence_python(metrics, config):
    """Original Python confidence scoring loop."""
    n = len(metrics)
    confidence = np.zeros(n)
    adf = metrics["adf_stat"].values
    hurst_v = metrics["hurst"].values
    corr = metrics["correlation"].values
    hl = metrics["half_life"].values

    for i in range(n):
        if np.isnan(adf[i]) or adf[i] >= config.adf_gate:
            confidence[i] = 0.0
            continue
        # ADF
        span = config.adf_best - config.adf_worst
        s_adf = max(0, min(1, (adf[i] - config.adf_worst) / span)) if abs(span) > 1e-12 else 0
        # Hurst
        span = config.hurst_best - config.hurst_worst
        s_h = max(0, min(1, (hurst_v[i] - config.hurst_worst) / span)) if abs(span) > 1e-12 and not np.isnan(hurst_v[i]) else 0
        # Corr
        span = config.corr_best - config.corr_worst
        s_c = max(0, min(1, (corr[i] - config.corr_worst) / span)) if abs(span) > 1e-12 and not np.isnan(corr[i]) else 0
        # HL
        h = hl[i]
        if np.isnan(h) or h < config.hl_min or h > config.hl_max:
            s_hl = 0.0
        elif config.hl_sweet_low <= h <= config.hl_sweet_high:
            s_hl = 1.0
        elif h < config.hl_sweet_low:
            s_hl = (h - config.hl_min) / (config.hl_sweet_low - config.hl_min)
        else:
            s_hl = (config.hl_max - h) / (config.hl_max - config.hl_sweet_high)

        confidence[i] = (config.w_adf * s_adf + config.w_hurst * s_h
                         + config.w_corr * s_c + config.w_hl * s_hl) * 100
    return confidence


def apply_conf_filter_python(sig, confidence, min_conf):
    """Original Python confidence filter loop."""
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        curr = out[t]
        if prev == 0 and curr != 0 and confidence[t] < min_conf:
            out[t] = 0
        prev = out[t]
    return out


def window_filter_python(sig, minutes, es, ee, fm):
    """Original Python window filter loop."""
    out = sig.copy()
    prev = 0
    for t in range(len(out)):
        m = minutes[t]
        curr = out[t]
        if m >= fm or m < es:
            out[t] = 0
            prev = 0
            continue
        if not (es <= m < ee):
            if prev == 0 and curr != 0:
                out[t] = 0
        prev = out[t]
    return out


# ============================================================
# Tests
# ============================================================

def test_signal_generator():
    print("=" * 80)
    print("TEST 1: Signal generator — numba vs Python loop")
    print("=" * 80)

    np.random.seed(42)
    n = 100_000
    z = np.cumsum(np.random.randn(n) * 0.1)
    z[0:50] = np.nan
    z[5000:5010] = np.nan

    sig_old = generate_signals_python(z, 3.0, 1.25, 4.0)

    # Warmup numba
    _ = generate_signals_numba(z[:100].astype(np.float64), 3.0, 1.25, 4.0)
    sig_new = generate_signals_numba(z.astype(np.float64), 3.0, 1.25, 4.0)

    match = np.array_equal(sig_old, sig_new)
    print(f"  Signaux identiques: {match}")
    if not match:
        diffs = np.where(sig_old != sig_new)[0]
        print(f"  DIFFERENCES at {len(diffs)} positions: {diffs[:10]}")

    # Speed
    t0 = time.time()
    for _ in range(10):
        generate_signals_python(z, 3.0, 1.25, 4.0)
    t_old = (time.time() - t0) / 10

    t0 = time.time()
    for _ in range(100):
        generate_signals_numba(z.astype(np.float64), 3.0, 1.25, 4.0)
    t_new = (time.time() - t0) / 100

    print(f"  Python: {t_old*1000:.1f}ms | Numba: {t_new*1000:.2f}ms | Speedup: {t_old/t_new:.0f}x")
    return match


def test_confidence_scoring():
    print("\n" + "=" * 80)
    print("TEST 2: Confidence scoring — vectorized vs Python loop")
    print("=" * 80)

    np.random.seed(123)
    n = 50_000
    metrics_df = pd.DataFrame({
        "adf_stat": np.random.uniform(-4, 0, n),
        "hurst": np.random.uniform(0.3, 0.6, n),
        "correlation": np.random.uniform(0.5, 1.0, n),
        "half_life": np.random.uniform(1, 250, n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="5min"))
    metrics_df.iloc[:100] = np.nan

    cfg = ConfidenceConfig()

    old_conf = compute_confidence_python(metrics_df, cfg)
    new_conf = compute_confidence(metrics_df, cfg).values

    max_diff = np.nanmax(np.abs(old_conf - new_conf))
    match = max_diff < 1e-10
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Identiques (tol 1e-10): {match}")
    if not match:
        diffs = np.where(np.abs(old_conf - new_conf) > 1e-10)[0]
        for d in diffs[:5]:
            print(f"    idx={d}: old={old_conf[d]:.6f} new={new_conf[d]:.6f}")

    t0 = time.time()
    compute_confidence_python(metrics_df, cfg)
    t_old = time.time() - t0

    t0 = time.time()
    compute_confidence(metrics_df, cfg)
    t_new = time.time() - t0

    print(f"  Python loop: {t_old*1000:.0f}ms | Vectorized: {t_new*1000:.1f}ms | Speedup: {t_old/t_new:.0f}x")
    return match


def test_conf_filter():
    print("\n" + "=" * 80)
    print("TEST 3: Confidence filter — numba vs Python loop")
    print("=" * 80)

    np.random.seed(42)
    n = 50_000
    sig_test = np.random.choice([-1, 0, 1], n).astype(np.int8)
    conf_test = np.random.uniform(0, 100, n)

    old = apply_conf_filter_python(sig_test, conf_test, 70.0)
    _ = _apply_conf_filter_numba(sig_test[:100], conf_test[:100], 70.0)  # warmup
    new = _apply_conf_filter_numba(sig_test, conf_test, 70.0)

    match = np.array_equal(old, new)
    print(f"  Filtres identiques: {match}")
    if not match:
        diffs = np.where(old != new)[0]
        print(f"  DIFFERENCES at {len(diffs)} positions")
    return match


def test_time_stop():
    print("\n" + "=" * 80)
    print("TEST 4: Time stop")
    print("=" * 80)

    sig = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, 0], dtype=np.int8)

    result3 = apply_time_stop(sig, 3)
    expected3 = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, 0], dtype=np.int8)
    match_3 = np.array_equal(result3, expected3)
    print(f"  time_stop=3: {match_3}")
    print(f"    Input:    {sig}")
    print(f"    Expected: {expected3}")
    print(f"    Got:      {result3}")

    result0 = apply_time_stop(sig, 0)
    match_0 = np.array_equal(result0, sig)
    print(f"  time_stop=0 (identity): {match_0}")

    return match_3 and match_0


def test_backtest_grid():
    print("\n" + "=" * 80)
    print("TEST 5: run_backtest_grid vs run_backtest_vectorized")
    print("=" * 80)

    np.random.seed(99)
    n = 10000
    px_a = 15000 + np.cumsum(np.random.randn(n) * 5)
    px_b = 35000 + np.cumsum(np.random.randn(n) * 3)
    beta = np.full(n, 1.2)
    sig = np.zeros(n, dtype=np.int8)
    for start in range(100, 9000, 500):
        sig[start:start + 20] = 1
        sig[start + 250:start + 270] = -1

    bt_full = run_backtest_vectorized(px_a, px_b, sig, beta, 20.0, 5.0, 0.25, 1.0, 1, 2.50, 100000.0)
    bt_grid = run_backtest_grid(px_a, px_b, sig, beta, 20.0, 5.0, 0.25, 1.0, 1, 2.50)

    checks = {
        "trades": bt_full["trades"] == bt_grid["trades"],
        "pnl": abs(bt_full["pnl"] - bt_grid["pnl"]) < 0.01,
        "pf": bt_full["profit_factor"] == bt_grid["profit_factor"],
        "win_rate": bt_full["win_rate"] == bt_grid["win_rate"],
        "avg_dur": bt_full["avg_duration_bars"] == bt_grid["avg_duration_bars"],
    }

    for k, v in checks.items():
        status = "OK" if v else "FAIL"
        print(f"  {k}: full={bt_full.get(k, bt_full.get('avg_duration_bars'))} "
              f"grid={bt_grid.get(k, bt_grid.get('avg_duration_bars'))} [{status}]")

    return all(checks.values())


def test_window_filter():
    print("\n" + "=" * 80)
    print("TEST 6: Window filter numba vs Python loop")
    print("=" * 80)

    idx = pd.date_range("2024-01-02 00:00", periods=1000, freq="5min")
    sig = np.zeros(1000, dtype=np.int8)
    sig[100:200] = 1
    sig[500:600] = -1
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)

    old = window_filter_python(sig, minutes, 240, 840, 930)
    _ = apply_window_filter_numba(sig[:10], minutes[:10], 240, 840, 930)  # warmup
    new = apply_window_filter_numba(sig, minutes, 240, 840, 930)

    match = np.array_equal(old, new)
    print(f"  Window filter identique: {match}")
    if not match:
        diffs = np.where(old != new)[0]
        for d in diffs[:10]:
            print(f"    idx={d} min={minutes[d]}: old={old[d]} new={new[d]}")
    return match


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = {
        "Signal generator numba": test_signal_generator(),
        "Confidence vectorized": test_confidence_scoring(),
        "Confidence filter numba": test_conf_filter(),
        "Time stop": test_time_stop(),
        "Backtest grid vs full": test_backtest_grid(),
        "Window filter numba": test_window_filter(),
    }

    print("\n" + "=" * 80)
    print(" RESUME VALIDATION")
    print("=" * 80)
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {name:<30} [{status}]")
        if not ok:
            all_ok = False

    print(f"\n  TOUT OK: {all_ok}")
    sys.exit(0 if all_ok else 1)
