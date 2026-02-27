"""Validation complete Kalman v2 NQ/RTY -- IS/OOS + Walk-Forward + Permutation.

Filtre: trades >= 50 AND MaxDD > -$5,000 (propfirm safe).
Configs issues du grid_kalman_v2_NQ_RTY.py avec poids corrects (ADF 50%, H 30%, C 20%, HL 0%).

Usage:
    python scripts/validate_kalman_v2_NQ_RTY.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("validate_kalman_v2")

OUTPUT_DIR = PROJECT_ROOT / "output" / "NQ_RTY"

# ======================================================================
# NQ_RTY correct weights
# ======================================================================

NQ_RTY_CONF = ConfidenceConfig(w_adf=0.50, w_hurst=0.30, w_corr=0.20, w_hl=0.00)

METRIC_PROFILES = {
    "tres_court": MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6),
    "p16_80":     MetricsConfig(adf_window=16, hurst_window=80, halflife_window=16, correlation_window=8),
    "court":      MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "p28_144":    MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p36_96":     MetricsConfig(adf_window=36, hurst_window=96, halflife_window=36, correlation_window=9),
    "moyen":      MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "p48_128":    MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
}

FIXED_WARMUP = 200
FIXED_GAP_P_MULT = 5.0

# ======================================================================
# Candidates: trades >= 50, MaxDD > -$5,000
# ======================================================================

CANDIDATES = [
    {"label": "K1_p16_80",    "alpha": 3e-7, "profil": "p16_80",     "window": "05:00-12:00", "ze": 1.625, "zx": 1.5, "zs": 3.0,  "conf": 60},
    {"label": "K2_p16_80_c55","alpha": 3e-7, "profil": "p16_80",     "window": "05:00-12:00", "ze": 1.625, "zx": 1.5, "zs": 3.0,  "conf": 55},
    {"label": "K3_p16_80_s325","alpha": 3e-7,"profil": "p16_80",     "window": "05:00-12:00", "ze": 1.625, "zx": 1.5, "zs": 3.25, "conf": 60},
    {"label": "K4_tc",        "alpha": 3e-7, "profil": "tres_court", "window": "05:00-12:00", "ze": 1.75,  "zx": 1.5, "zs": 3.0,  "conf": 60},
    {"label": "K5_p36_96",    "alpha": 3e-7, "profil": "p36_96",     "window": "04:00-13:00", "ze": 1.625, "zx": 1.5, "zs": 3.0,  "conf": 60},
    {"label": "K6_p36_s325",  "alpha": 3e-7, "profil": "p36_96",     "window": "04:00-13:00", "ze": 1.625, "zx": 1.5, "zs": 3.25, "conf": 60},
    {"label": "K7_p36_s275",  "alpha": 3e-7, "profil": "p36_96",     "window": "04:00-13:00", "ze": 1.625, "zx": 1.5, "zs": 2.75, "conf": 60},
]


def load_instruments():
    with open(PROJECT_ROOT / "config" / "instruments.yaml") as f:
        return yaml.safe_load(f)


def parse_window(w):
    parts = w.split("-")
    sh, sm = int(parts[0].split(":")[0]), int(parts[0].split(":")[1])
    eh, em = int(parts[1].split(":")[0]), int(parts[1].split(":")[1])
    return sh * 60 + sm, eh * 60 + em


def build_signal(aligned, cfg, mult_a, mult_b, tick_a, tick_b):
    """Build full signal chain for a Kalman config."""
    _px_a = aligned.df["close_a"].values
    _px_b = aligned.df["close_b"].values
    idx = aligned.df.index

    est = create_estimator("kalman", alpha_ratio=cfg["alpha"],
                           warmup=FIXED_WARMUP, gap_P_multiplier=FIXED_GAP_P_MULT)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread
    zscore = np.ascontiguousarray(
        np.nan_to_num(hr.zscore.values, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float64,
    )

    profile_cfg = METRIC_PROFILES[cfg["profil"]]
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)
    confidence = compute_confidence(metrics, NQ_RTY_CONF).values

    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    entry_start, entry_end = parse_window(cfg["window"])
    flat_min = 15 * 60 + 30

    raw = generate_signals_numba(zscore, cfg["ze"], cfg["zx"], cfg["zs"])
    sig_conf = _apply_conf_filter_numba(raw, confidence, cfg["conf"])
    sig = apply_window_filter_numba(sig_conf, minutes, entry_start, entry_end, flat_min)

    return sig, beta, confidence


def run_bt(px_a, px_b, sig, beta, mult_a, mult_b, tick_a, tick_b):
    """Run backtest and return result dict."""
    return run_backtest_vectorized(
        px_a, px_b, sig, beta,
        mult_a=mult_a, mult_b=mult_b, tick_a=tick_a, tick_b=tick_b,
        slippage_ticks=1, commission=2.50,
    )


def compute_maxdd(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


# ======================================================================
# Walk-Forward
# ======================================================================

def walk_forward(aligned, cfg, mult_a, mult_b, tick_a, tick_b,
                 is_years=2, oos_months=6, step_months=6):
    """Walk-forward validation with rolling IS/OOS windows."""
    idx = aligned.df.index
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values

    # Build signal on full data (Kalman is sequential, can't restart per window)
    sig, beta, confidence = build_signal(aligned, cfg, mult_a, mult_b, tick_a, tick_b)

    # Define walk-forward windows
    start_date = idx[0]
    end_date = idx[-1]
    windows = []

    current = start_date
    while True:
        is_end = current + pd.DateOffset(years=is_years)
        oos_end = is_end + pd.DateOffset(months=oos_months)
        if oos_end > end_date:
            break

        is_mask = (idx >= current) & (idx < is_end)
        oos_mask = (idx >= is_end) & (idx < oos_end)

        if is_mask.sum() > 0 and oos_mask.sum() > 0:
            windows.append((current, is_end, oos_end, is_mask, oos_mask))

        current += pd.DateOffset(months=step_months)

    results = []
    for w_start, is_end, oos_end, is_mask, oos_mask in windows:
        # IS backtest
        np.where(is_mask)[0]
        bt_is = run_bt(px_a[is_mask], px_b[is_mask], sig[is_mask], beta[is_mask],
                       mult_a, mult_b, tick_a, tick_b)

        # OOS backtest
        np.where(oos_mask)[0]
        bt_oos = run_bt(px_a[oos_mask], px_b[oos_mask], sig[oos_mask], beta[oos_mask],
                        mult_a, mult_b, tick_a, tick_b)

        is_pf = bt_is["profit_factor"] if bt_is["trades"] >= 5 else 0
        oos_pf = bt_oos["profit_factor"] if bt_oos["trades"] >= 3 else 0
        oos_pnl = bt_oos["pnl"]
        profitable = oos_pnl > 0

        results.append({
            "is_start": w_start.strftime("%Y-%m"),
            "is_end": is_end.strftime("%Y-%m"),
            "oos_end": oos_end.strftime("%Y-%m"),
            "is_trades": bt_is["trades"],
            "is_pf": round(is_pf, 2),
            "oos_trades": bt_oos["trades"],
            "oos_pf": round(oos_pf, 2),
            "oos_pnl": round(oos_pnl, 2),
            "profitable": profitable,
        })

    n_profitable = sum(r["profitable"] for r in results)
    n_total = len(results)
    wf_go = n_profitable >= n_total // 2 if n_total > 0 else False

    return results, n_profitable, n_total, wf_go


# ======================================================================
# Permutation test
# ======================================================================

def permutation_test(aligned, cfg, mult_a, mult_b, tick_a, tick_b, n_perms=1000):
    """Permutation test: shuffle confidence, re-filter, compare PF."""
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index

    sig, beta, confidence = build_signal(aligned, cfg, mult_a, mult_b, tick_a, tick_b)

    # Observed PF
    bt_obs = run_bt(px_a, px_b, sig, beta, mult_a, mult_b, tick_a, tick_b)
    obs_pf = bt_obs["profit_factor"]

    # Raw signals (before confidence filter)
    est = create_estimator("kalman", alpha_ratio=cfg["alpha"],
                           warmup=FIXED_WARMUP, gap_P_multiplier=FIXED_GAP_P_MULT)
    hr = est.estimate(aligned)
    zscore = np.ascontiguousarray(
        np.nan_to_num(hr.zscore.values, nan=0.0, posinf=0.0, neginf=0.0),
        dtype=np.float64,
    )
    raw = generate_signals_numba(zscore, cfg["ze"], cfg["zx"], cfg["zs"])

    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    entry_start, entry_end = parse_window(cfg["window"])
    flat_min = 15 * 60 + 30

    rng = np.random.default_rng(42)
    perm_pfs = []

    for _ in range(n_perms):
        # Shuffle confidence vector
        conf_perm = rng.permutation(confidence)
        sig_perm = _apply_conf_filter_numba(raw, conf_perm, cfg["conf"])
        sig_perm = apply_window_filter_numba(sig_perm, minutes, entry_start, entry_end, flat_min)

        bt_perm = run_bt(px_a, px_b, sig_perm, beta, mult_a, mult_b, tick_a, tick_b)
        perm_pfs.append(bt_perm["profit_factor"])

    perm_pfs = np.array(perm_pfs)
    p_value = float(np.mean(perm_pfs >= obs_pf))
    perm_go = p_value < 0.05

    return obs_pf, perm_pfs, p_value, perm_go


# ======================================================================
# Main
# ======================================================================

def main():
    log.info("=" * 100)
    log.info(" VALIDATION KALMAN v2 NQ/RTY -- trades>=50, MaxDD>-$5K")
    log.info(" Poids: ADF 50%, Hurst 30%, Corr 20%, HL 0%")
    log.info("=" * 100)

    instruments = load_instruments()
    spec_a, spec_b = instruments["NQ"], instruments["RTY"]
    mult_a, mult_b = spec_a["multiplier"], spec_b["multiplier"]
    tick_a, tick_b = spec_a["tick_size"], spec_b["tick_size"]

    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n_total = len(idx)
    is_end = int(n_total * 0.60)

    all_results = []

    for cfg in CANDIDATES:
        label = cfg["label"]
        log.info(f"\n{'='*80}")
        log.info(f" {label}: a={cfg['alpha']:.1e} {cfg['profil']} {cfg['window']} "
                 f"ze={cfg['ze']} zx={cfg['zx']} zs={cfg['zs']} c={cfg['conf']}")
        log.info(f"{'='*80}")

        # --- Full sample ---
        sig, beta, confidence = build_signal(aligned, cfg, mult_a, mult_b, tick_a, tick_b)
        bt_full = run_bt(px_a, px_b, sig, beta, mult_a, mult_b, tick_a, tick_b)
        max_dd = compute_maxdd(bt_full["equity"])

        # L/S
        sides = bt_full["trade_sides"]
        bt_full["trade_pnls"]
        long_pct = float((sides > 0).sum()) / bt_full["trades"] * 100 if bt_full["trades"] > 0 else 50

        log.info(f"  FULL: {bt_full['trades']} trades, WR={bt_full['win_rate']:.1f}%, "
                 f"PF={bt_full['profit_factor']:.2f}, PnL=${bt_full['pnl']:,.0f}, "
                 f"MaxDD=${max_dd:,.0f}, L/S={long_pct:.0f}%/{100-long_pct:.0f}%")

        # --- IS/OOS 60/40 ---
        bt_is = run_bt(px_a[:is_end], px_b[:is_end], sig[:is_end], beta[:is_end],
                       mult_a, mult_b, tick_a, tick_b)
        bt_oos = run_bt(px_a[is_end:], px_b[is_end:], sig[is_end:], beta[is_end:],
                        mult_a, mult_b, tick_a, tick_b)

        is_pf = bt_is["profit_factor"] if bt_is["trades"] >= 10 else 0
        oos_pf = bt_oos["profit_factor"] if bt_oos["trades"] >= 5 else 0
        isoos_go = "GO" if oos_pf > 1.0 and bt_oos["trades"] >= 5 else "FAIL"

        log.info(f"  IS/OOS: IS={bt_is['trades']}t PF={is_pf:.2f} | "
                 f"OOS={bt_oos['trades']}t PF={oos_pf:.2f} [{isoos_go}]")

        # --- Walk-Forward ---
        wf_results, wf_prof, wf_total, wf_go = walk_forward(
            aligned, cfg, mult_a, mult_b, tick_a, tick_b
        )
        wf_pnl = sum(r["oos_pnl"] for r in wf_results)
        wf_verdict = "GO" if wf_go else "FAIL"

        log.info(f"  WF: {wf_prof}/{wf_total} profitable, PnL=${wf_pnl:,.0f} [{wf_verdict}]")
        for r in wf_results:
            mark = "+" if r["profitable"] else "-"
            log.info(f"    [{mark}] {r['is_start']}->{r['oos_end']} | "
                     f"IS: {r['is_trades']}t PF={r['is_pf']:.2f} | "
                     f"OOS: {r['oos_trades']}t PF={r['oos_pf']:.2f} PnL=${r['oos_pnl']:,.0f}")

        # --- Permutation ---
        obs_pf, perm_pfs, p_val, perm_go = permutation_test(
            aligned, cfg, mult_a, mult_b, tick_a, tick_b, n_perms=1000
        )
        perm_verdict = "GO" if perm_go else "FAIL"

        log.info(f"  PERM: observed PF={obs_pf:.2f} vs permuted mean={perm_pfs.mean():.2f} "
                 f"p={p_val:.3f} [{perm_verdict}]")

        # --- Global verdict ---
        global_go = isoos_go == "GO" and wf_go and perm_go
        global_verdict = "GO" if global_go else "FAIL"
        log.info(f"  >>> GLOBAL: IS/OOS={isoos_go} WF={wf_verdict} PERM={perm_verdict} => [{global_verdict}]")

        all_results.append({
            "label": label,
            "alpha": cfg["alpha"],
            "profil": cfg["profil"],
            "window": cfg["window"],
            "ze": cfg["ze"],
            "zx": cfg["zx"],
            "zs": cfg["zs"],
            "conf": cfg["conf"],
            "trades": bt_full["trades"],
            "wr": bt_full["win_rate"],
            "pnl": bt_full["pnl"],
            "pf": bt_full["profit_factor"],
            "max_dd": round(max_dd, 2),
            "long_pct": round(long_pct, 1),
            "is_pf": round(is_pf, 2),
            "oos_pf": round(oos_pf, 2),
            "isoos_go": isoos_go,
            "wf_result": f"{wf_prof}/{wf_total}",
            "wf_pnl": round(wf_pnl, 2),
            "wf_go": wf_verdict,
            "perm_p": round(p_val, 3),
            "perm_go": perm_verdict,
            "global_go": global_verdict,
        })

    # --- Summary ---
    df = pd.DataFrame(all_results)
    log.info(f"\n{'='*100}")
    log.info(f" SUMMARY -- {len(df)} configs tested")
    log.info(f"{'='*100}")

    go_count = (df["global_go"] == "GO").sum()
    log.info(f"Global GO: {go_count}/{len(df)}")

    for _, r in df.iterrows():
        mark = ">>>" if r["global_go"] == "GO" else "   "
        log.info(
            f"  {mark} {r['label']:<18} | {r['profil']:<10} {r['window']:<12} "
            f"Trd={r['trades']:>3} PF={r['pf']:.2f} MaxDD=${r['max_dd']:>,.0f} L/S={r['long_pct']:.0f}% | "
            f"IS/OOS={r['isoos_go']} WF={r['wf_result']} PERM p={r['perm_p']:.3f} | "
            f"[{r['global_go']}]"
        )

    out_path = OUTPUT_DIR / "validate_kalman_v2.csv"
    df.to_csv(out_path, index=False)
    log.info(f"\n[OUTPUT] {out_path}")


if __name__ == "__main__":
    main()
