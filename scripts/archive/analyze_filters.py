"""Approach A: ablation des filtres individuels sur config robuste NQ_YM.

Config robuste: OLS=2640, ZW=36, z_entry=3.0, z_exit=1.5, z_stop=4.0, profil=tres_court
8 combinaisons systématiques pour identifier quel filtre contribue à l'edge.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_vectorized
from src.data.cache import load_aligned_pair_cache
from src.hedge.factory import create_estimator
from src.metrics.dashboard import MetricsConfig, compute_all_metrics
from src.signals.filters import ConfidenceConfig, compute_confidence
from src.signals.generator import SignalConfig, SignalGenerator
from src.spread.pair import SpreadPair
from src.utils.constants import Instrument

# Seuils individuels pour les filtres de régime
ADF_THRESH = -2.86      # stat ADF < seuil = stationnaire
HURST_THRESH = 0.45     # Hurst < seuil = mean-reverting
CORR_THRESH = 0.80      # corrélation > seuil = cointegrated
HL_MIN = 5              # half-life dans [5, 120] bars
HL_MAX = 120


def main():
    # ── Load data ─────────────────────────────────────────────────────
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.YM)
    aligned = load_aligned_pair_cache(pair, "5min")
    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    n = len(px_a)

    # ── Hedge — OLS 2640, ZW 36 ──────────────────────────────────────
    est = create_estimator("ols_rolling", window=2640, zscore_window=36)
    hr = est.estimate(aligned)
    beta = hr.beta.values
    spread = hr.spread

    # ── Z-score (ZW=36 déjà calculé par l'estimateur) ────────────────
    zscore = hr.zscore

    # ── Metrics — profil tres_court ───────────────────────────────────
    cfg = MetricsConfig(adf_window=12, hurst_window=64, halflife_window=12, correlation_window=6)
    metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], cfg)

    # ── Confidence score (composite) ──────────────────────────────────
    base_conf = ConfidenceConfig()
    confidence = compute_confidence(metrics, base_conf).values

    # ── Signals — entry=3.0, exit=1.5, stop=4.0 ──────────────────────
    gen = SignalGenerator(config=SignalConfig(z_entry=3.0, z_exit=1.5, z_stop=4.0))
    raw_signals = gen.generate(zscore).values

    # ── Trading window mask [04:00-14:00) CT ──────────────────────────
    minutes = idx.hour * 60 + idx.minute
    tw_mask = (minutes >= 240) & (minutes < 840)

    # ── Individual metric arrays ──────────────────────────────────────
    adf_vals = metrics["adf_stat"].values
    hurst_vals = metrics["hurst"].values
    corr_vals = metrics["correlation"].values
    hl_vals = metrics["half_life"].values

    # ── Helper: check individual filter pass ──────────────────────────
    def adf_pass(t):
        v = adf_vals[t]
        return not np.isnan(v) and v < ADF_THRESH

    def hurst_pass(t):
        v = hurst_vals[t]
        return not np.isnan(v) and v < HURST_THRESH

    def corr_pass(t):
        v = corr_vals[t]
        return not np.isnan(v) and v > CORR_THRESH

    def hl_pass(t):
        v = hl_vals[t]
        return not np.isnan(v) and HL_MIN <= v <= HL_MAX

    # ── Backtest runner ───────────────────────────────────────────────
    def run_with_filter(name, block_func):
        sig = raw_signals.copy()
        prev = 0
        blocked = 0
        for t in range(n):
            curr = sig[t]
            if (prev == 0) and (curr != 0) and block_func(t):
                sig[t] = 0
                blocked += 1
            prev = sig[t]
        sig[~tw_mask] = 0

        bt = run_backtest_vectorized(
            px_a, px_b, sig, beta,
            mult_a=20.0, mult_b=5.0, tick_a=0.25, tick_b=1.0,
            slippage_ticks=1, commission=2.50, initial_capital=100_000.0,
        )

        pnls = bt["trade_pnls"]
        num = bt["trades"]
        if len(pnls) > 0:
            median_pnl = float(np.median(pnls))
            best = float(pnls.max())
            worst = float(pnls.min())
            ratio_bw = abs(best / worst) if worst != 0 else float("inf")
            max_consec = 0
            consec = 0
            for p in pnls:
                if p <= 0:
                    consec += 1
                    max_consec = max(max_consec, consec)
                else:
                    consec = 0
        else:
            median_pnl = best = worst = 0
            ratio_bw = 0
            max_consec = 0

        avg_dur_h = bt["avg_duration_bars"] * 5 / 60
        eq = bt["equity"]
        running_max = np.maximum.accumulate(eq)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_arr = (running_max - eq) / running_max * 100
        dd_pct = float(np.nan_to_num(dd_arr).max())

        print(
            f" {name:<30} {num:>5} {bt['win_rate']:>5.1f}% "
            f"${bt['pnl']:>10,.0f} {bt['profit_factor']:>5.2f} "
            f"${bt['avg_pnl_trade']:>7,.0f} ${median_pnl:>7,.0f} "
            f"{dd_pct:>5.1f}% {avg_dur_h:>5.1f}h {max_consec:>5} {ratio_bw:>6.2f}"
        )

    # ── Header ────────────────────────────────────────────────────────
    print("=" * 140)
    print("APPROACH A — Ablation des filtres individuels")
    print("Config robuste: NQ_YM | OLS=2640 | ZW=36 | entry=3.0 | exit=1.5 | stop=4.0 | profil tres_court")
    print(f"Seuils: ADF<{ADF_THRESH} | Hurst<{HURST_THRESH} | Corr>{CORR_THRESH} | HL in [{HL_MIN},{HL_MAX}]")
    print("=" * 140)
    print(f" {'Filtre':<30} {'Trd':>5} {'Win%':>5} {'PnL':>10} {'PF':>5} {'AvgPnL':>7} {'MedPnL':>7} {'DD%':>5} {'Dur':>5} {'MCL':>5} {'B/W':>6}")
    print("-" * 140)

    # ── Section 1: Baseline — aucun filtre vs tout OFF/ON ─────────────
    print(" --- BASELINES ---")
    run_with_filter(
        "1. TOUT OFF (z-score pur)",
        lambda t: False
    )
    run_with_filter(
        "2. Confidence >= 70%",
        lambda t: confidence[t] < 70.0
    )
    run_with_filter(
        "3. Regime individuel ALL ON",
        lambda t: not (adf_pass(t) and hurst_pass(t) and corr_pass(t) and hl_pass(t))
    )

    # ── Section 2: Ablation — retirer UN filtre à la fois ─────────────
    print("-" * 140)
    print(" --- ABLATION (all ON sauf un) ---")
    run_with_filter(
        "4. ALL sauf ADF",
        lambda t: not (hurst_pass(t) and corr_pass(t) and hl_pass(t))
    )
    run_with_filter(
        "5. ALL sauf Hurst",
        lambda t: not (adf_pass(t) and corr_pass(t) and hl_pass(t))
    )
    run_with_filter(
        "6. ALL sauf Corr",
        lambda t: not (adf_pass(t) and hurst_pass(t) and hl_pass(t))
    )
    run_with_filter(
        "7. ALL sauf HL",
        lambda t: not (adf_pass(t) and hurst_pass(t) and corr_pass(t))
    )

    # ── Section 3: Isolation — UN seul filtre actif ───────────────────
    print("-" * 140)
    print(" --- ISOLATION (un seul filtre ON) ---")
    run_with_filter(
        "8. ADF seul",
        lambda t: not adf_pass(t)
    )
    run_with_filter(
        "9. Hurst seul",
        lambda t: not hurst_pass(t)
    )
    run_with_filter(
        "10. Corr seule",
        lambda t: not corr_pass(t)
    )
    run_with_filter(
        "11. HL seul",
        lambda t: not hl_pass(t)
    )

    # ── Section 4: Confidence thresholds ──────────────────────────────
    print("-" * 140)
    print(" --- CONFIDENCE THRESHOLDS ---")
    for thresh in [30, 40, 50, 60, 70]:
        run_with_filter(
            f"Confidence >= {thresh}%",
            lambda t, th=thresh: confidence[t] < th
        )

    # ── Section 5a: Combinaisons SANS ADF ─────────────────────────
    print("-" * 140)
    print(" --- COMBINAISONS SANS ADF (Section 5a) ---")
    run_with_filter(
        "5a.1 Hurst + Corr",
        lambda t: not (hurst_pass(t) and corr_pass(t))
    )
    run_with_filter(
        "5a.2 Hurst + HL",
        lambda t: not (hurst_pass(t) and hl_pass(t))
    )
    run_with_filter(
        "5a.3 Corr + HL",
        lambda t: not (corr_pass(t) and hl_pass(t))
    )
    run_with_filter(
        "5a.4 Hurst + Corr + HL",
        lambda t: not (hurst_pass(t) and corr_pass(t) and hl_pass(t))
    )

    # ── Section 5b: Grille de seuils Hurst + Corr + HL ─────────
    print("-" * 140)
    print(" --- GRILLE SEUILS Hurst + Corr + HL (Section 5b) ---")
    print(f"   HL range fixe [{HL_MIN}, {HL_MAX}] | 16 combinaisons")
    hurst_grid = [0.40, 0.45, 0.50, 0.55]
    corr_grid = [0.70, 0.75, 0.80, 0.85]
    for h_thresh in hurst_grid:
        for c_thresh in corr_grid:
            label = f"H<{h_thresh} C>{c_thresh}"
            run_with_filter(
                label,
                lambda t, ht=h_thresh, ct=c_thresh: not (
                    (not np.isnan(hurst_vals[t]) and hurst_vals[t] < ht)
                    and (not np.isnan(corr_vals[t]) and corr_vals[t] > ct)
                    and hl_pass(t)
                )
            )

    # ── Section 5c: ADF kill switch fenetre longue (480 bars) ───
    print("-" * 140)
    print(" --- ADF KILL SWITCH 480 bars (Section 5c) ---")
    cfg_long_adf = MetricsConfig(
        adf_window=480,
        hurst_window=cfg.hurst_window,
        halflife_window=cfg.halflife_window,
        correlation_window=cfg.correlation_window,
    )
    metrics_long_adf = compute_all_metrics(
        spread, aligned.df["close_a"], aligned.df["close_b"], cfg_long_adf
    )
    adf_long_vals = metrics_long_adf["adf_stat"].values

    def adf_long_pass(t):
        v = adf_long_vals[t]
        return not np.isnan(v) and v < ADF_THRESH

    # ADF 480 pass rate (on signal bars)
    sig_mask_tmp = raw_signals != 0
    adf_long_valid = adf_long_vals[sig_mask_tmp & ~np.isnan(adf_long_vals)]
    if len(adf_long_valid) > 0:
        adf_long_pr = (adf_long_valid < ADF_THRESH).mean() * 100
        print(f"   ADF 480 bars: median={np.median(adf_long_valid):.3f}  pass_rate={adf_long_pr:.1f}%")
    else:
        print("   ADF 480 bars: no valid data on signal bars")

    run_with_filter(
        "5c.1 ADF480 seul",
        lambda t: not adf_long_pass(t)
    )
    run_with_filter(
        "5c.2 ADF480 + Hurst+Corr+HL",
        lambda t: not (adf_long_pass(t) and hurst_pass(t) and corr_pass(t) and hl_pass(t))
    )
    # Variante: ADF480 + best combo from 5b (default thresholds)
    run_with_filter(
        "5c.3 ADF480 + H<0.50 C>0.70 HL",
        lambda t: not (
            adf_long_pass(t)
            and (not np.isnan(hurst_vals[t]) and hurst_vals[t] < 0.50)
            and (not np.isnan(corr_vals[t]) and corr_vals[t] > 0.70)
            and hl_pass(t)
        )
    )

    print("=" * 140)

    # ── Statistiques descriptives des métriques ───────────────────────
    print("\n--- DISTRIBUTION DES MÉTRIQUES (sur barres avec signal brut != 0) ---")
    sig_mask = raw_signals != 0
    for col, vals, label in [
        ("ADF", adf_vals, f"< {ADF_THRESH}"),
        ("Hurst", hurst_vals, f"< {HURST_THRESH}"),
        ("Corr", corr_vals, f"> {CORR_THRESH}"),
        ("HL", hl_vals, f"in [{HL_MIN},{HL_MAX}]"),
    ]:
        valid = vals[sig_mask & ~np.isnan(vals)]
        if len(valid) > 0:
            print(f" {col:<8} median={np.median(valid):>8.3f}  mean={np.mean(valid):>8.3f}  "
                  f"std={np.std(valid):>7.3f}  min={np.min(valid):>8.3f}  max={np.max(valid):>8.3f}  "
                  f"pass_rate ({label}): {_pass_rate(col, valid):.1f}%")


def _pass_rate(col, vals):
    if col == "ADF":
        return (vals < ADF_THRESH).mean() * 100
    elif col == "Hurst":
        return (vals < HURST_THRESH).mean() * 100
    elif col == "Corr":
        return (vals > CORR_THRESH).mean() * 100
    elif col == "HL":
        return ((vals >= HL_MIN) & (vals <= HL_MAX)).mean() * 100
    return 0


if __name__ == "__main__":
    main()
