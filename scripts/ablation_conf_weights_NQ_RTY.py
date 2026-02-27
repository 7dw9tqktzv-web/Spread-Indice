"""Ablation des poids de confidence NQ/RTY — 70 configs x 15 combos de poids.

Objectif: determiner si les poids actuels (ADF 50%, Hurst 30%, Corr 20%, HL 0%)
sont optimaux pour NQ/RTY, ou si d'autres combinaisons donnent de meilleurs resultats.

- 50 configs top diversifiees (selection greedy multi-dimensionnelle)
- 10 configs best par profil manquant (couvrir les 16 profils)
- 10 configs "moyennes" (PF ~1.35-1.45, pour reduire le biais de selection)
- 15 combos de poids (12 sans HL + 3 avec HL)

Usage:
    python scripts/ablation_conf_weights_NQ_RTY.py
"""

import sys
import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest_grid
from src.config.instruments import get_pair_specs
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

# ======================================================================
# Constants NQ/RTY
# ======================================================================
_NQ, _RTY = get_pair_specs("NQ", "RTY")
MULT_A, MULT_B = _NQ.multiplier, _RTY.multiplier
TICK_A, TICK_B = _NQ.tick_size, _RTY.tick_size
SLIPPAGE = 1
COMMISSION = 2.50
FLAT_MIN = 930

# ======================================================================
# 17 Metric Profiles (complet)
# ======================================================================
METRIC_PROFILES = {
    "p12_64":    MetricsConfig(adf_window=12, hurst_window=64,  halflife_window=12, correlation_window=6),
    "p16_80":    MetricsConfig(adf_window=16, hurst_window=80,  halflife_window=16, correlation_window=8),
    "p18_96":    MetricsConfig(adf_window=18, hurst_window=96,  halflife_window=18, correlation_window=9),
    "p20_100":   MetricsConfig(adf_window=20, hurst_window=100, halflife_window=20, correlation_window=10),
    "p24_128":   MetricsConfig(adf_window=24, hurst_window=128, halflife_window=24, correlation_window=12),
    "p28_144":   MetricsConfig(adf_window=28, hurst_window=144, halflife_window=28, correlation_window=14),
    "p30_160":   MetricsConfig(adf_window=30, hurst_window=160, halflife_window=30, correlation_window=15),
    "p36_192":   MetricsConfig(adf_window=36, hurst_window=192, halflife_window=36, correlation_window=18),
    "p42_224":   MetricsConfig(adf_window=42, hurst_window=224, halflife_window=42, correlation_window=21),
    "p48_256":   MetricsConfig(adf_window=48, hurst_window=256, halflife_window=48, correlation_window=24),
    "p60_320":   MetricsConfig(adf_window=60, hurst_window=320, halflife_window=60, correlation_window=30),
    "p18_192":   MetricsConfig(adf_window=18, hurst_window=192, halflife_window=18, correlation_window=18),
    "p24_256":   MetricsConfig(adf_window=24, hurst_window=256, halflife_window=24, correlation_window=24),
    "p30_256":   MetricsConfig(adf_window=30, hurst_window=256, halflife_window=30, correlation_window=24),
    "p48_128":   MetricsConfig(adf_window=48, hurst_window=128, halflife_window=48, correlation_window=12),
    "p48_96":    MetricsConfig(adf_window=48, hurst_window=96,  halflife_window=48, correlation_window=9),
    "p36_96":    MetricsConfig(adf_window=36, hurst_window=96,  halflife_window=36, correlation_window=9),
}

WINDOWS_MAP = {
    "02:00-14:00": (120, 840),
    "04:00-14:00": (240, 840),
    "06:00-14:00": (360, 840),
    "08:00-14:00": (480, 840),
    "08:00-12:00": (480, 720),
    "06:00-12:00": (360, 720),
}

# ======================================================================
# 15 Combos de poids
# ======================================================================
WEIGHT_COMBOS = {
    # --- Sans HL (12) ---
    "W01_ref":        {"w_adf": 0.50, "w_hurst": 0.30, "w_corr": 0.20, "w_hl": 0.00},  # Reference actuelle
    "W02_hurst_up":   {"w_adf": 0.40, "w_hurst": 0.40, "w_corr": 0.20, "w_hl": 0.00},
    "W03_adf_dom":    {"w_adf": 0.60, "w_hurst": 0.20, "w_corr": 0.20, "w_hl": 0.00},
    "W04_equal3":     {"w_adf": 0.33, "w_hurst": 0.34, "w_corr": 0.33, "w_hl": 0.00},
    "W05_corr_up":    {"w_adf": 0.40, "w_hurst": 0.30, "w_corr": 0.30, "w_hl": 0.00},
    "W06_adf_corr":   {"w_adf": 0.50, "w_hurst": 0.20, "w_corr": 0.30, "w_hl": 0.00},
    "W07_hurst_dom":  {"w_adf": 0.30, "w_hurst": 0.40, "w_corr": 0.30, "w_hl": 0.00},
    "W08_adf_ultra":  {"w_adf": 0.70, "w_hurst": 0.15, "w_corr": 0.15, "w_hl": 0.00},
    "W09_hurst_ultra": {"w_adf": 0.20, "w_hurst": 0.50, "w_corr": 0.30, "w_hl": 0.00},
    "W10_adf_only":   {"w_adf": 1.00, "w_hurst": 0.00, "w_corr": 0.00, "w_hl": 0.00},
    "W11_hurst_only": {"w_adf": 0.00, "w_hurst": 1.00, "w_corr": 0.00, "w_hl": 0.00},
    "W12_corr_only":  {"w_adf": 0.00, "w_hurst": 0.00, "w_corr": 1.00, "w_hl": 0.00},
    # --- Avec HL (3) ---
    "W13_equal4":     {"w_adf": 0.25, "w_hurst": 0.25, "w_corr": 0.25, "w_hl": 0.25},
    "W14_hl_mod":     {"w_adf": 0.40, "w_hurst": 0.25, "w_corr": 0.20, "w_hl": 0.15},
    "W15_hl_up":      {"w_adf": 0.35, "w_hurst": 0.25, "w_corr": 0.20, "w_hl": 0.20},
}

# ======================================================================
# 70 Configs (50 top diversifiees + 10 profils manquants + 10 moyennes)
# ======================================================================

# 50 top diversifiees (greedy selection from filtered grid)
TOP_50 = [
    {"ols": 9240, "zw": 32, "profile": "p48_128", "window": "02:00-14:00", "z_entry": 3.25, "z_exit": 0.75, "z_stop": 5.5, "conf": 80},
    {"ols": 3960, "zw": 24, "profile": "p16_80", "window": "08:00-14:00", "z_entry": 3.50, "z_exit": 0.00, "z_stop": 4.5, "conf": 75},
    {"ols": 9240, "zw": 32, "profile": "p48_128", "window": "02:00-14:00", "z_entry": 3.25, "z_exit": 0.50, "z_stop": 4.5, "conf": 80},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "04:00-14:00", "z_entry": 3.50, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 9240, "zw": 20, "profile": "p36_96", "window": "06:00-14:00", "z_entry": 3.00, "z_exit": 0.75, "z_stop": 5.0, "conf": 75},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "04:00-14:00", "z_entry": 3.25, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 3960, "zw": 24, "profile": "p16_80", "window": "06:00-14:00", "z_entry": 3.50, "z_exit": 0.50, "z_stop": 5.5, "conf": 70},
    {"ols": 9240, "zw": 20, "profile": "p36_96", "window": "04:00-14:00", "z_entry": 3.00, "z_exit": 0.75, "z_stop": 5.5, "conf": 75},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "06:00-14:00", "z_entry": 3.25, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 9240, "zw": 36, "profile": "p36_96", "window": "04:00-14:00", "z_entry": 3.50, "z_exit": 1.25, "z_stop": 5.5, "conf": 75},
    {"ols": 10560, "zw": 36, "profile": "p16_80", "window": "06:00-12:00", "z_entry": 3.50, "z_exit": 0.50, "z_stop": 6.0, "conf": 55},
    {"ols": 9240, "zw": 32, "profile": "p48_128", "window": "02:00-14:00", "z_entry": 3.25, "z_exit": 1.25, "z_stop": 6.0, "conf": 80},
    {"ols": 9240, "zw": 48, "profile": "p48_128", "window": "06:00-12:00", "z_entry": 2.75, "z_exit": 1.25, "z_stop": 5.5, "conf": 80},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "06:00-14:00", "z_entry": 3.50, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 9240, "zw": 36, "profile": "p48_128", "window": "02:00-14:00", "z_entry": 3.25, "z_exit": 0.75, "z_stop": 4.5, "conf": 80},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "02:00-14:00", "z_entry": 3.25, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 9240, "zw": 36, "profile": "p36_96", "window": "02:00-14:00", "z_entry": 3.50, "z_exit": 1.25, "z_stop": 6.0, "conf": 75},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "02:00-14:00", "z_entry": 3.50, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "06:00-12:00", "z_entry": 3.25, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 9240, "zw": 48, "profile": "p30_160", "window": "06:00-14:00", "z_entry": 2.75, "z_exit": 0.50, "z_stop": 5.5, "conf": 70},
    {"ols": 9240, "zw": 24, "profile": "p18_192", "window": "02:00-14:00", "z_entry": 2.75, "z_exit": 0.75, "z_stop": 6.0, "conf": 65},
    {"ols": 10560, "zw": 48, "profile": "p16_80", "window": "06:00-12:00", "z_entry": 3.50, "z_exit": 1.50, "z_stop": 5.5, "conf": 55},
    {"ols": 10560, "zw": 36, "profile": "p16_80", "window": "06:00-14:00", "z_entry": 3.50, "z_exit": 0.50, "z_stop": 6.0, "conf": 60},
    {"ols": 9240, "zw": 48, "profile": "p30_160", "window": "06:00-12:00", "z_entry": 2.75, "z_exit": 0.50, "z_stop": 5.5, "conf": 70},
    {"ols": 10560, "zw": 36, "profile": "p16_80", "window": "06:00-14:00", "z_entry": 3.50, "z_exit": 0.50, "z_stop": 4.5, "conf": 55},
    {"ols": 9240, "zw": 36, "profile": "p28_144", "window": "08:00-14:00", "z_entry": 3.00, "z_exit": 0.25, "z_stop": 6.0, "conf": 75},
    {"ols": 9240, "zw": 32, "profile": "p28_144", "window": "02:00-14:00", "z_entry": 3.00, "z_exit": 0.00, "z_stop": 5.5, "conf": 75},
    {"ols": 3960, "zw": 24, "profile": "p16_80", "window": "08:00-14:00", "z_entry": 3.50, "z_exit": 0.00, "z_stop": 5.0, "conf": 70},
    {"ols": 9240, "zw": 48, "profile": "p28_144", "window": "08:00-12:00", "z_entry": 2.75, "z_exit": 0.50, "z_stop": 6.0, "conf": 75},
    {"ols": 9240, "zw": 36, "profile": "p28_144", "window": "02:00-14:00", "z_entry": 3.00, "z_exit": 0.25, "z_stop": 5.5, "conf": 75},
    {"ols": 6600, "zw": 60, "profile": "p28_144", "window": "08:00-14:00", "z_entry": 3.25, "z_exit": 0.75, "z_stop": 5.5, "conf": 80},
    {"ols": 9240, "zw": 36, "profile": "p16_80", "window": "02:00-14:00", "z_entry": 3.50, "z_exit": 1.25, "z_stop": 6.0, "conf": 60},
    {"ols": 9240, "zw": 36, "profile": "p16_80", "window": "04:00-14:00", "z_entry": 3.50, "z_exit": 1.25, "z_stop": 6.0, "conf": 60},
    {"ols": 9240, "zw": 36, "profile": "p16_80", "window": "06:00-14:00", "z_entry": 3.50, "z_exit": 1.25, "z_stop": 6.0, "conf": 60},
    {"ols": 9240, "zw": 12, "profile": "p16_80", "window": "06:00-12:00", "z_entry": 2.50, "z_exit": 0.00, "z_stop": 5.0, "conf": 60},
    {"ols": 9240, "zw": 48, "profile": "p30_160", "window": "02:00-14:00", "z_entry": 2.75, "z_exit": 1.25, "z_stop": 5.5, "conf": 65},
    {"ols": 9240, "zw": 28, "profile": "p28_144", "window": "08:00-14:00", "z_entry": 2.75, "z_exit": 0.75, "z_stop": 4.0, "conf": 70},
    {"ols": 10560, "zw": 36, "profile": "p16_80", "window": "04:00-14:00", "z_entry": 3.50, "z_exit": 0.50, "z_stop": 4.5, "conf": 55},
    {"ols": 7920, "zw": 20, "profile": "p36_96", "window": "04:00-14:00", "z_entry": 3.00, "z_exit": 0.75, "z_stop": 3.5, "conf": 75},
    {"ols": 9240, "zw": 48, "profile": "p16_80", "window": "02:00-14:00", "z_entry": 3.25, "z_exit": 1.50, "z_stop": 5.5, "conf": 60},
    {"ols": 9240, "zw": 48, "profile": "p48_128", "window": "02:00-14:00", "z_entry": 2.75, "z_exit": 0.50, "z_stop": 5.5, "conf": 75},
    {"ols": 9240, "zw": 24, "profile": "p18_192", "window": "04:00-14:00", "z_entry": 2.75, "z_exit": 0.75, "z_stop": 5.0, "conf": 65},
    {"ols": 9240, "zw": 24, "profile": "p28_144", "window": "08:00-12:00", "z_entry": 2.75, "z_exit": 0.75, "z_stop": 4.5, "conf": 75},
    {"ols": 10560, "zw": 36, "profile": "p16_80", "window": "04:00-14:00", "z_entry": 3.50, "z_exit": 0.50, "z_stop": 6.0, "conf": 60},
    {"ols": 9240, "zw": 48, "profile": "p30_160", "window": "04:00-14:00", "z_entry": 2.75, "z_exit": 1.25, "z_stop": 5.5, "conf": 65},
    {"ols": 9240, "zw": 48, "profile": "p28_144", "window": "08:00-12:00", "z_entry": 2.75, "z_exit": 1.00, "z_stop": 6.0, "conf": 75},
    {"ols": 9240, "zw": 48, "profile": "p48_128", "window": "06:00-12:00", "z_entry": 2.75, "z_exit": 1.50, "z_stop": 5.5, "conf": 80},
    {"ols": 9240, "zw": 28, "profile": "p28_144", "window": "08:00-12:00", "z_entry": 2.75, "z_exit": 0.75, "z_stop": 4.0, "conf": 70},
    {"ols": 9240, "zw": 12, "profile": "p16_80", "window": "06:00-12:00", "z_entry": 2.50, "z_exit": 0.25, "z_stop": 4.0, "conf": 60},
    {"ols": 9240, "zw": 48, "profile": "p16_80", "window": "06:00-14:00", "z_entry": 3.25, "z_exit": 1.50, "z_stop": 5.5, "conf": 60},
]

# 10 configs best par profil manquant (couvrir les 16 profils)
MISSING_PROFILES = [
    {"ols": 7920, "zw": 20, "profile": "p42_224", "window": "04:00-14:00", "z_entry": 3.00, "z_exit": 0.75, "z_stop": 3.5, "conf": 75},
    {"ols": 7920, "zw": 24, "profile": "p60_320", "window": "06:00-12:00", "z_entry": 3.50, "z_exit": 0.50, "z_stop": 4.0, "conf": 55},
    {"ols": 7920, "zw": 28, "profile": "p48_96", "window": "06:00-12:00", "z_entry": 3.00, "z_exit": 0.75, "z_stop": 3.5, "conf": 80},
    {"ols": 9240, "zw": 24, "profile": "p20_100", "window": "08:00-14:00", "z_entry": 3.50, "z_exit": 0.25, "z_stop": 4.5, "conf": 70},
    {"ols": 9240, "zw": 36, "profile": "p24_256", "window": "02:00-14:00", "z_entry": 3.00, "z_exit": 1.50, "z_stop": 5.5, "conf": 80},
    {"ols": 7920, "zw": 48, "profile": "p30_256", "window": "08:00-12:00", "z_entry": 3.50, "z_exit": 0.25, "z_stop": 5.5, "conf": 80},
    {"ols": 7920, "zw": 48, "profile": "p36_192", "window": "08:00-12:00", "z_entry": 3.50, "z_exit": 1.25, "z_stop": 5.5, "conf": 70},
    {"ols": 3960, "zw": 28, "profile": "p18_96", "window": "02:00-14:00", "z_entry": 3.00, "z_exit": 1.25, "z_stop": 5.0, "conf": 80},
    {"ols": 9240, "zw": 36, "profile": "p24_128", "window": "08:00-14:00", "z_entry": 3.50, "z_exit": 0.25, "z_stop": 5.5, "conf": 80},
    {"ols": 7920, "zw": 48, "profile": "p12_64", "window": "02:00-14:00", "z_entry": 3.50, "z_exit": 1.75, "z_stop": 5.5, "conf": 75},
]

# 10 configs "moyennes" (PF ~1.35-1.45 dans le grid filtre, patterns varies)
MEDIUM_CONFIGS = [
    {"ols": 9240, "zw": 48, "profile": "p16_80", "window": "02:00-14:00", "z_entry": 2.75, "z_exit": 0.75, "z_stop": 5.0, "conf": 55},
    {"ols": 10560, "zw": 36, "profile": "p48_128", "window": "04:00-14:00", "z_entry": 3.00, "z_exit": 1.00, "z_stop": 5.0, "conf": 65},
    {"ols": 6600, "zw": 28, "profile": "p36_96", "window": "06:00-14:00", "z_entry": 3.25, "z_exit": 0.50, "z_stop": 4.5, "conf": 70},
    {"ols": 3960, "zw": 36, "profile": "p28_144", "window": "04:00-14:00", "z_entry": 2.75, "z_exit": 1.00, "z_stop": 5.5, "conf": 65},
    {"ols": 7920, "zw": 48, "profile": "p30_160", "window": "02:00-14:00", "z_entry": 3.00, "z_exit": 0.50, "z_stop": 4.5, "conf": 60},
    {"ols": 9240, "zw": 20, "profile": "p42_224", "window": "06:00-12:00", "z_entry": 3.00, "z_exit": 1.25, "z_stop": 4.0, "conf": 75},
    {"ols": 10560, "zw": 24, "profile": "p18_96", "window": "08:00-12:00", "z_entry": 3.50, "z_exit": 0.75, "z_stop": 5.0, "conf": 70},
    {"ols": 6600, "zw": 60, "profile": "p48_96", "window": "06:00-14:00", "z_entry": 3.00, "z_exit": 0.25, "z_stop": 5.5, "conf": 75},
    {"ols": 7920, "zw": 32, "profile": "p24_128", "window": "02:00-14:00", "z_entry": 2.75, "z_exit": 1.00, "z_stop": 4.5, "conf": 65},
    {"ols": 5280, "zw": 28, "profile": "p20_100", "window": "04:00-14:00", "z_entry": 3.25, "z_exit": 0.50, "z_stop": 5.0, "conf": 60},
]


def build_config_id(cfg, idx):
    """Short ID for a config."""
    return f"C{idx:02d}_{cfg['profile']}_{cfg['ols']}_{cfg['zw']}"


def main():
    t_start = time_mod.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 100)
    print("  ABLATION CONFIDENCE WEIGHTS — NQ/RTY OLS")
    print("  70 configs x 15 weight combos = 1,050 backtests")
    print("=" * 100)

    print("\nLoading NQ/RTY data...")
    pair = SpreadPair(leg_a=Instrument.NQ, leg_b=Instrument.RTY)
    aligned = load_aligned_pair_cache(pair, "5min")
    if aligned is None:
        print("ERREUR: pas de cache NQ_RTY")
        return

    px_a = aligned.df["close_a"].values
    px_b = aligned.df["close_b"].values
    idx = aligned.df.index
    minutes = (idx.hour * 60 + idx.minute).values.astype(np.int32)
    years = (idx[-1] - idx[0]).days / 365.25
    print(f"Data: {len(px_a):,} bars, {years:.1f} years")
    print(f"Range: {idx[0]} -> {idx[-1]}")

    # ------------------------------------------------------------------
    # Build all configs
    # ------------------------------------------------------------------
    all_configs = []
    for i, cfg in enumerate(TOP_50):
        cfg["group"] = "TOP"
        cfg["id"] = build_config_id(cfg, i)
        all_configs.append(cfg)
    for i, cfg in enumerate(MISSING_PROFILES):
        cfg["group"] = "PROFILE"
        cfg["id"] = build_config_id(cfg, 50 + i)
        all_configs.append(cfg)
    for i, cfg in enumerate(MEDIUM_CONFIGS):
        cfg["group"] = "MEDIUM"
        cfg["id"] = build_config_id(cfg, 60 + i)
        all_configs.append(cfg)

    print(f"\nConfigs: {len(all_configs)} (50 top + 10 profils + 10 moyennes)")
    print(f"Weight combos: {len(WEIGHT_COMBOS)}")
    print(f"Total backtests: {len(all_configs) * len(WEIGHT_COMBOS)}")

    # Profile diversity check
    profiles_used = sorted(set(c["profile"] for c in all_configs))
    print(f"Profiles couverts: {len(profiles_used)}/17 -> {profiles_used}")

    # ------------------------------------------------------------------
    # Pre-compute hedge ratios + raw signals + metrics (only once per config)
    # ------------------------------------------------------------------
    print("\nPre-computing hedge ratios, z-scores, metrics...")
    precomputed = {}
    unique_keys = set()

    for cfg in all_configs:
        key = (cfg["ols"], cfg["zw"], cfg["profile"], cfg["window"],
               cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])
        if key in unique_keys:
            precomputed[cfg["id"]] = precomputed[
                next(c["id"] for c in all_configs
                     if (c["ols"], c["zw"], c["profile"], c["window"],
                         c["z_entry"], c["z_exit"], c["z_stop"]) == key
                     and c["id"] in precomputed)
            ]
            continue
        unique_keys.add(key)

        # Hedge ratio
        est = create_estimator("ols_rolling", window=cfg["ols"], zscore_window=cfg["zw"])
        hr = est.estimate(aligned)
        beta = hr.beta.values
        spread = hr.spread

        # Z-score
        zw = cfg["zw"]
        mu = spread.rolling(zw).mean()
        sigma = spread.rolling(zw).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = ((spread - mu) / sigma).replace([np.inf, -np.inf], np.nan).values
        zscore = np.ascontiguousarray(np.nan_to_num(zscore, nan=0.0), dtype=np.float64)

        # Raw signals (no confidence filter yet)
        raw_sig = generate_signals_numba(zscore, cfg["z_entry"], cfg["z_exit"], cfg["z_stop"])

        # Metrics
        profile_cfg = METRIC_PROFILES[cfg["profile"]]
        metrics = compute_all_metrics(spread, aligned.df["close_a"], aligned.df["close_b"], profile_cfg)

        # Window
        entry_start, entry_end = WINDOWS_MAP[cfg["window"]]

        precomputed[cfg["id"]] = {
            "beta": beta,
            "raw_sig": raw_sig,
            "metrics": metrics,
            "entry_start": entry_start,
            "entry_end": entry_end,
        }

    print(f"Pre-computed: {len(unique_keys)} unique hedge/signal combos")

    # ------------------------------------------------------------------
    # Run ablation: for each config x weight combo, apply confidence + backtest
    # ------------------------------------------------------------------
    print("\nRunning ablation...")
    results = []
    total = len(all_configs) * len(WEIGHT_COMBOS)
    done = 0
    t_bt_start = time_mod.time()

    for cfg in all_configs:
        pre = precomputed[cfg["id"]]
        for w_name, w_vals in WEIGHT_COMBOS.items():
            # Build confidence config with these weights
            conf_cfg = ConfidenceConfig(
                w_adf=w_vals["w_adf"],
                w_hurst=w_vals["w_hurst"],
                w_corr=w_vals["w_corr"],
                w_hl=w_vals["w_hl"],
                min_confidence=float(cfg["conf"]),
            )

            # Compute confidence scores
            confidence = compute_confidence(pre["metrics"], conf_cfg).values

            # Apply confidence filter
            sig = _apply_conf_filter_numba(pre["raw_sig"].copy(), confidence, float(cfg["conf"]))

            # Apply window filter
            sig = apply_window_filter_numba(sig, minutes, pre["entry_start"], pre["entry_end"], FLAT_MIN)

            # Backtest
            bt = run_backtest_grid(
                px_a, px_b, sig, pre["beta"],
                MULT_A, MULT_B, TICK_A, TICK_B,
                SLIPPAGE, COMMISSION,
            )

            results.append({
                "config_id": cfg["id"],
                "group": cfg["group"],
                "ols": cfg["ols"],
                "zw": cfg["zw"],
                "profile": cfg["profile"],
                "window": cfg["window"],
                "z_entry": cfg["z_entry"],
                "z_exit": cfg["z_exit"],
                "z_stop": cfg["z_stop"],
                "min_conf": cfg["conf"],
                "weight_combo": w_name,
                "w_adf": w_vals["w_adf"],
                "w_hurst": w_vals["w_hurst"],
                "w_corr": w_vals["w_corr"],
                "w_hl": w_vals["w_hl"],
                "trades": bt["trades"],
                "win_rate": bt["win_rate"],
                "pnl": bt["pnl"],
                "profit_factor": bt["profit_factor"],
                "avg_pnl_trade": bt["avg_pnl_trade"],
            })

            done += 1
            if done % 150 == 0:
                elapsed = time_mod.time() - t_bt_start
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total} backtests ({done/total*100:.0f}%) — {rate:.0f}/s — ETA {eta:.0f}s")

    elapsed_bt = time_mod.time() - t_bt_start
    print(f"\nBacktests complete: {total} en {elapsed_bt:.1f}s ({total/elapsed_bt:.0f}/s)")

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    out_dir = PROJECT_ROOT / "output" / "NQ_RTY"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_conf_weights.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultats sauvegardes: {csv_path}")

    # ------------------------------------------------------------------
    # Analysis 1: Average metrics by weight combo (across all configs)
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  ANALYSE 1: MOYENNE PAR COMBO DE POIDS (70 configs)")
    print(f"{'='*100}")

    pivot = df.groupby("weight_combo").agg(
        avg_trades=("trades", "mean"),
        avg_wr=("win_rate", "mean"),
        avg_pnl=("pnl", "mean"),
        avg_pf=("profit_factor", "mean"),
        median_pf=("profit_factor", "median"),
        avg_avg_pnl=("avg_pnl_trade", "mean"),
        pct_profitable=("pnl", lambda x: (x > 0).mean() * 100),
        total_pnl=("pnl", "sum"),
    ).sort_values("avg_pf", ascending=False)

    print(f"\n  {'Combo':<20} {'Trades':>6} {'WR%':>6} {'PnL moy':>10} {'PF moy':>7} {'PF med':>7} "
          f"{'$/trd':>7} {'%prof':>6} {'PnL tot':>12}")
    print("  " + "-" * 100)
    for wname, r in pivot.iterrows():
        marker = " <<<" if wname == "W01_ref" else ""
        print(f"  {wname:<20} {r.avg_trades:>6.0f} {r.avg_wr:>5.1f}% ${r.avg_pnl:>9,.0f} "
              f"{r.avg_pf:>7.2f} {r.median_pf:>7.2f} ${r.avg_avg_pnl:>6,.0f} "
              f"{r.pct_profitable:>5.1f}% ${r.total_pnl:>11,.0f}{marker}")

    # ------------------------------------------------------------------
    # Analysis 2: By group (TOP vs PROFILE vs MEDIUM)
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  ANALYSE 2: PAR GROUPE x POIDS")
    print(f"{'='*100}")

    for group in ["TOP", "PROFILE", "MEDIUM"]:
        sub = df[df["group"] == group]
        n_cfg = sub["config_id"].nunique()
        pivot_g = sub.groupby("weight_combo").agg(
            avg_pf=("profit_factor", "mean"),
            avg_pnl=("pnl", "mean"),
            avg_trades=("trades", "mean"),
        ).sort_values("avg_pf", ascending=False)

        print(f"\n  --- {group} ({n_cfg} configs) ---")
        print(f"  {'Combo':<20} {'PF moy':>7} {'PnL moy':>10} {'Trades':>7}")
        print(f"  {'-'*50}")
        for wname, r in pivot_g.head(5).iterrows():
            marker = " <<<" if wname == "W01_ref" else ""
            print(f"  {wname:<20} {r.avg_pf:>7.2f} ${r.avg_pnl:>9,.0f} {r.avg_trades:>7.0f}{marker}")

    # ------------------------------------------------------------------
    # Analysis 3: HL impact
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  ANALYSE 3: IMPACT HL (W13-W15 vs W01 reference)")
    print(f"{'='*100}")

    hl_combos = ["W01_ref", "W13_equal4", "W14_hl_mod", "W15_hl_up"]
    sub_hl = df[df["weight_combo"].isin(hl_combos)]
    pivot_hl = sub_hl.groupby("weight_combo").agg(
        avg_pf=("profit_factor", "mean"),
        avg_pnl=("pnl", "mean"),
        avg_trades=("trades", "mean"),
        avg_wr=("win_rate", "mean"),
    ).reindex(hl_combos)

    print(f"\n  {'Combo':<20} {'PF moy':>7} {'PnL moy':>10} {'Trades':>7} {'WR%':>6}")
    print(f"  {'-'*55}")
    for wname, r in pivot_hl.iterrows():
        print(f"  {wname:<20} {r.avg_pf:>7.2f} ${r.avg_pnl:>9,.0f} {r.avg_trades:>7.0f} {r.avg_wr:>5.1f}%")

    # ------------------------------------------------------------------
    # Analysis 4: Solo metrics (W10-W12)
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  ANALYSE 4: METRIQUES SOLO (W10 ADF, W11 Hurst, W12 Corr)")
    print(f"{'='*100}")

    solo_combos = ["W01_ref", "W10_adf_only", "W11_hurst_only", "W12_corr_only"]
    sub_solo = df[df["weight_combo"].isin(solo_combos)]
    pivot_solo = sub_solo.groupby("weight_combo").agg(
        avg_pf=("profit_factor", "mean"),
        avg_pnl=("pnl", "mean"),
        avg_trades=("trades", "mean"),
        pct_profitable=("pnl", lambda x: (x > 0).mean() * 100),
    ).reindex(solo_combos)

    print(f"\n  {'Combo':<20} {'PF moy':>7} {'PnL moy':>10} {'Trades':>7} {'%prof':>6}")
    print(f"  {'-'*55}")
    for wname, r in pivot_solo.iterrows():
        print(f"  {wname:<20} {r.avg_pf:>7.2f} ${r.avg_pnl:>9,.0f} {r.avg_trades:>7.0f} {r.pct_profitable:>5.1f}%")

    # ------------------------------------------------------------------
    # Analysis 5: Best combo per config (who switches from W01?)
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  ANALYSE 5: MEILLEUR COMBO PAR CONFIG")
    print(f"{'='*100}")

    best_per_config = df.loc[df.groupby("config_id")["profit_factor"].idxmax()]
    ref_per_config = df[df["weight_combo"] == "W01_ref"].set_index("config_id")

    switch_count = {}
    for _, row in best_per_config.iterrows():
        w = row["weight_combo"]
        switch_count[w] = switch_count.get(w, 0) + 1

    print(f"\n  {'Combo':<20} {'Configs ou best':>15}")
    print(f"  {'-'*40}")
    for wname, cnt in sorted(switch_count.items(), key=lambda x: -x[1]):
        pct = cnt / len(all_configs) * 100
        print(f"  {wname:<20} {cnt:>5} ({pct:>4.1f}%)")

    # Configs qui changent significativement
    print("\n  Configs ou un autre combo bat W01 de >10% PF:")
    print(f"  {'Config':<35} {'W01 PF':>7} {'Best combo':<20} {'Best PF':>7} {'Delta':>7}")
    print(f"  {'-'*85}")
    improve_count = 0
    for cid in sorted(ref_per_config.index):
        if cid not in ref_per_config.index:
            continue
        ref_pf = ref_per_config.loc[cid, "profit_factor"]
        best_row = best_per_config[best_per_config["config_id"] == cid].iloc[0]
        best_pf = best_row["profit_factor"]
        if ref_pf > 0 and best_pf > ref_pf * 1.10:
            delta_pct = (best_pf - ref_pf) / ref_pf * 100
            print(f"  {cid:<35} {ref_pf:>7.2f} {best_row['weight_combo']:<20} {best_pf:>7.2f} +{delta_pct:>5.1f}%")
            improve_count += 1

    if improve_count == 0:
        print("  (aucune — W01 est optimal ou quasi-optimal partout)")
    else:
        print(f"\n  {improve_count}/{len(all_configs)} configs ({improve_count/len(all_configs)*100:.0f}%) beneficient d'un changement de poids >10%")

    # ------------------------------------------------------------------
    # Analysis 6: Stability — coefficient of variation of PF across weight combos
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  ANALYSE 6: STABILITE — variation PF selon les poids")
    print(f"{'='*100}")

    stability = df.groupby("config_id").agg(
        mean_pf=("profit_factor", "mean"),
        std_pf=("profit_factor", "std"),
        min_pf=("profit_factor", "min"),
        max_pf=("profit_factor", "max"),
    )
    stability["cv"] = stability["std_pf"] / stability["mean_pf"] * 100
    stability["range_pf"] = stability["max_pf"] - stability["min_pf"]

    print(f"\n  CV moyen (coefficient variation PF): {stability['cv'].mean():.1f}%")
    print(f"  Range PF moyen (max-min): {stability['range_pf'].mean():.2f}")

    print("\n  Configs les PLUS sensibles aux poids (top 10 CV):")
    print(f"  {'Config':<35} {'PF moy':>7} {'PF min':>7} {'PF max':>7} {'CV%':>6}")
    print(f"  {'-'*65}")
    for cid, r in stability.nlargest(10, "cv").iterrows():
        print(f"  {cid:<35} {r.mean_pf:>7.2f} {r.min_pf:>7.2f} {r.max_pf:>7.2f} {r.cv:>5.1f}%")

    print("\n  Configs les MOINS sensibles (top 10 plus stables):")
    print(f"  {'Config':<35} {'PF moy':>7} {'PF min':>7} {'PF max':>7} {'CV%':>6}")
    print(f"  {'-'*65}")
    for cid, r in stability.nsmallest(10, "cv").iterrows():
        print(f"  {cid:<35} {r.mean_pf:>7.2f} {r.min_pf:>7.2f} {r.max_pf:>7.2f} {r.cv:>5.1f}%")

    # ------------------------------------------------------------------
    # VERDICT
    # ------------------------------------------------------------------
    print(f"\n\n{'='*100}")
    print("  VERDICT ABLATION CONFIDENCE WEIGHTS")
    print(f"{'='*100}")

    best_overall = pivot.index[0]
    ref_row = pivot.loc["W01_ref"]
    best_row_v = pivot.iloc[0]

    print(f"\n  Reference (W01): PF moy={ref_row.avg_pf:.2f}, PnL moy=${ref_row.avg_pnl:,.0f}")
    print(f"  Meilleur combo:  {best_overall} PF moy={best_row_v.avg_pf:.2f}, PnL moy=${best_row_v.avg_pnl:,.0f}")

    if best_overall == "W01_ref":
        print("\n  >>> W01 (50/30/20/0) CONFIRME comme optimal <<<")
    else:
        delta = (best_row_v.avg_pf - ref_row.avg_pf) / ref_row.avg_pf * 100
        print(f"\n  >>> {best_overall} bat W01 de {delta:+.1f}% PF en moyenne <<<")
        if abs(delta) < 5:
            print("  >>> Difference marginale (<5%) — W01 reste acceptable <<<")
        else:
            print(f"  >>> Difference significative — considerer re-run du grid avec {best_overall} <<<")

    elapsed_total = time_mod.time() - t_start
    print(f"\n  Ablation complete en {elapsed_total:.0f}s")


if __name__ == "__main__":
    main()
