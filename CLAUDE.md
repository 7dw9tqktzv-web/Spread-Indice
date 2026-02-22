# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Système de spread trading intraday sur futures US (NQ, ES, RTY, YM).
- **Phase 1** : Moteur de backtest Python (optimisation & validation)
- **Phase 2** : Indicateur Sierra Charts temps réel (ACSIL C++)

Le biais directionnel journalier est discrétionnaire — le système time l'entrée avec précision statistique sur des billets macroéconomiques. Voir `ARCHITECTURE.md` pour le design détaillé.

## Commands

```bash
# Activate venv (ALWAYS work in venv)
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v --tb=short

# Run unit tests only (skip integration)
python -m pytest tests/ -v --ignore=tests/test_integration

# Run a single test file
python -m pytest tests/test_hedge/test_ols_rolling.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run backtest pipeline
python scripts/run_backtest.py --pair NQ_ES --method ols_rolling
python scripts/run_backtest.py --pair NQ_ES --method kalman --alpha-ratio 1e-6
python scripts/run_backtest.py --prepare-data  # data prep only
python scripts/run_backtest.py --all --method kalman  # all 6 pairs

# CLI overrides
python scripts/run_backtest.py --pair NQ_ES --method ols_rolling --z-entry 2.5 --z-exit 1.5 --z-stop 3.5 --ols-window 3960 --min-confidence 50

# Grid search (43,200 backtests, 6 pairs × all param combos)
python scripts/run_grid.py --workers 20
python scripts/run_grid.py --workers 20 --dry-run  # show counts only
```

**Important**: All scripts must be run from the project root (cache uses relative path `output/cache`).

## Architecture

### Implementation Status
- **Implemented**: `src/data/`, `src/hedge/` (OLS + Kalman + factory), `src/spread/`, `src/sizing/`, `src/stats/` (vectorized hurst+halflife), `src/metrics/` (dashboard + confidence scoring), `src/signals/` (generator numba + filters numba + confidence filter + time stop + window filter), `src/backtest/` (engine bar-by-bar + vectorized + grid-optimized, performance), `src/utils/`, `config/`, `scripts/` (run_backtest.py, run_grid.py, run_refined_grid.py, run_grid_kalman_v3.py, validate_kalman_top.py, find_safe_kalman.py, find_safe_kalman_micro.py, analyze_grid_results.py, validate_top5_configs.py, validate_numba.py), `tests/` (49 tests)
- **Stubs only**: `src/optimisation/`, `sierra/` (reference docs ready)
- **Numba JIT**: signal generator (451x), confidence filter, time stop, window filter — all validated identical to Python originals

### Data Flow
`raw/*.txt` (Sierra CSV 1min) → `loader.py` → `cleaner.py` → `resampler.py` (1min/3min/5min) → `alignment.py` (pair) → `hedge/` (ratio) → `spread/builder.py` → `metrics/` → `signals/` → `backtest/engine.py` → `performance.py`

Dependencies flow strictly downward. Config YAML files are loaded at script level and injected as dataclasses — modules never read config directly.

### Key Modules
- **`src/data/`** — Pipeline: `loader.py` → `cleaner.py` → `resampler.py` → `alignment.py`. Cache via `cache.py` (Parquet)
- **`src/hedge/`** — Hedge ratio estimators behind `HedgeRatioEstimator` ABC (`base.py`). Implemented: `ols_rolling.py`, `kalman.py`. Factory dispatch via `factory.py` (keyed on `HedgeMethod` enum). Config dataclasses: `OLSRollingConfig`, `KalmanConfig`
- **`src/sizing/`** — Dollar-neutral × β position sizing (scalar + vectorized): `N_b = round((Notionnel_A / Notionnel_B) × β × N_a)`
- **`src/stats/`** — Low-level statistical functions: hurst (variance-ratio, vectorized), halflife (AR(1) via rolling cov, vectorized), correlation, stationarity (2 ADF variants)
- **`src/metrics/`** — Aggregation layer (`dashboard.py`): `MetricsConfig` + `compute_all_metrics()` calls `src/stats/` functions, returns DataFrame with `adf_stat, hurst, half_life, correlation`
- **`src/signals/`** — `generator.py`: stateful 4-state machine (numba JIT, 451x speedup). `filters.py`: confidence scoring (vectorized numpy, 138x), `_apply_conf_filter_numba`, `apply_time_stop`, `apply_window_filter_numba` (all numba JIT) + legacy filters
- **`src/backtest/`** — `engine.py`: BacktestEngine (bar-by-bar), `run_backtest_vectorized` (full with equity), `run_backtest_grid` (no equity, grid-optimized). `performance.py`: PerformanceMetrics
- **`src/spread/`** — `SpreadPair` dataclass (`pair.py`)
- **`config/`** — YAML configs: `instruments.yaml`, `pairs.yaml`, `backtest.yaml`, `optimisation.yaml`
- **`sierra/`** — Phase 2 ACSIL C++ indicator (header-only libs, online algorithms). Reference docs: `infos_sierra.md` (config), `specs_actifs.md` (contract specs), `SC_*_REFERENCE.md` (ACSIL docs), example `.cpp` templates

### Non-Obvious Architectural Details

1. **ABC signature**: `HedgeRatioEstimator.estimate(aligned: AlignedPair) -> HedgeResult` — estimators receive the full `AlignedPair` object, not raw series. Log-price conversion happens inside each estimator.

2. **`HedgeResult` bundles spread AND zscore**: The estimator computes both. The z-score method differs by estimator — OLS uses `zscore_window=12` rolling, Kalman uses innovation-based `ν(t)/√F(t)` (no separate window).

3. **Session filter wraps midnight via OR logic**: `t >= buf_start OR t < buf_end` (session is 17:30→15:30 CT overnight). A naive range check would be wrong.

4. **Column naming convention**: `BarData.df` uses `close`, but after `align_pair()` the `AlignedPair.df` uses `close_a` / `close_b`.

5. **Imports**: `pyproject.toml` sets `pythonpath = ["."]` — all imports use `from src.xxx import yyy`.

6. **Kalman specifics**: `Q = alpha_ratio × R × I` (scale-aware); session gaps (>30min) multiply `P` by `gap_P_multiplier=10.0`; Joseph form covariance update. Alpha sweet spot for NQ_YM: 1.5e-7 to 3e-7. Warmup and gap_P_mult have negligible impact (Kalman converges fast). Innovation z-score is N(0,1) by construction — optimal z_entry=1.5-2.0, z_stop=2.5-2.8 (NOT the same as OLS z_entry=3.15, z_stop=4.5).

7. **Two ADF implementations in `stationarity.py`**: `adf_rolling()` (statsmodels) and `adf_statistic_simple()` (custom, no augmentation) — the simple one is designed for Python/C++ parity testing.

8. **Regression convention (OLS + Kalman aligned)**: `log_a = α + β × log_b + ε` — leg_a is dependent, leg_b is explanatory. Both estimators use this same convention. β is directly usable in the sizing formula.

9. **Signal generator has 4 states, not 3**: FLAT → LONG/SHORT → COOLDOWN → FLAT. After a stop loss, the COOLDOWN state blocks re-entry until `|z| < z_exit` (spread must return to neutral). NaN resets to FLAT (clean session start).

10. **`src/stats/` vs `src/metrics/`**: `stats/` contains pure computation functions (rolling hurst, halflife, etc.). `metrics/dashboard.py` is the aggregation layer that calls `stats/` and returns a unified DataFrame. They are separate layers, not duplicates.

11. **Calculate vs Act separation**: Hedge ratio + metrics compute on full Globex session (17:30-15:30 CT, buffer_minutes=0). Signals + trades restricted to [04:00-14:00) CT via `apply_trading_window_filter()`. Position open at 13:55 is force-closed at 14:00 (signal → 0).

12. **ADF in dashboard uses `adf_statistic_simple()`** (statistics ~-3.5, threshold -2.86), NOT `adf_rolling()` (p-values 0-1). The simple variant is Sierra C++ compatible.

13. **Confidence scoring** replaces binary regime filter: each metric produces a score 0→1 via linear interpolation, weighted (ADF 40%, Hurst 25%, Corr 20%, HL 15%), with ADF gate at stat ≥ -1.00 → 0%. `min_confidence` threshold (default 50%) is optimizable.

14. **Session is 17:30-15:30 CT** (already buffered vs Globex 17:00-16:00). `buffer_minutes=0` in config. 264 bars/day. OLS windows: 1320 (5j), 2640 (10j), ..., 7920 (30j).

15. **Hurst uses variance-ratio** (not R/S). R/S gives biased ~0.99 on spread levels (cumsum-on-cumsum). Variance-ratio works on levels directly, gives H~0.41 median for NQ/ES. Vectorized via precomputed rolling std.

16. **Half-life vectorized** via rolling covariance: `b = Cov(Z(t),Z(t-1)) / Var(Z(t-1))`, `HL = -ln(2)/ln(b)`. Instant vs 39s with per-bar lstsq.

### Instruments & Pairs
- **6 instruments** : NQ, ES, RTY, YM (standards) + MNQ, MYM (micros) — contrats continus, Volume Rollover Back-Adjusted
- **7 paires** : NQ/ES, NQ/RTY, NQ/YM, ES/RTY, ES/YM, RTY/YM, MNQ/MYM
- **Micros** : MNQ/MYM = 1/10eme du standard, memes prix, tick identique, commission $0.79/side
- **Sizing** : `find_optimal_multiplier()` minimise l'erreur d'arrondi (max_multiplier configurable, defaut 1)
- **Dollar stop** : `dollar_stop` dans BacktestConfig (defaut 0 = desactive). Non recommande sur NQ/YM standard (coupe les trades trop tot)

## Research Context
**Toujours lire `CHANGELOG.md` en debut de session** pour avoir le contexte des derniers backtests effectues, resultats et decisions. Ce fichier est l'historique de recherche du projet — il evite de refaire des tests deja faits et permet de reprendre la ou on s'est arrete.

Avant de proposer un test ou une config, verifier dans le changelog si ca n'a pas deja ete teste. Mettre a jour le changelog apres chaque serie de tests significative.

## Key Conventions
- Toujours travailler en **venv**
- Données en **Chicago Time (CT)**
- Tous les calculs sur **log-prix** (ln) pour OLS et Kalman
- Session : 17h30–15h30 CT (Globex), fenêtre trading : 4h00–14h00 CT
- **Git** : utiliser l'agent GitHub (`gh`) pour tous les commits, push et opérations de branche — jamais de commandes git manuelles
- Valider chaque étape avec l'utilisateur avant de passer à la suivante
- Paramètres optimisés en Phase 1 avant implémentation Phase 2

## Selected Config — Config E (principale) — NQ_YM 5min

Issue du grid search affine 1,080,000 combos. Validee IS/OOS, Walk-Forward 4/5, Permutation p=0.000.

| Parametre | Valeur | Notes |
|-----------|--------|-------|
| Paire | NQ_YM | Seule paire avec edge robuste OLS |
| Timeframe | 5min | 1min/3min testes, inferieurs |
| OLS lookback | **3300 bars (~12.5j)** | Grid search affine |
| Z-score window | **30 bars (2h30)** | Reactif |
| z_entry | **3.15** | Zone plate 3.05-3.15 |
| z_exit | **1.00** | Sort tot |
| z_stop | **4.50** | Large |
| Profil metrics | tres_court | adf=12, hurst=64, hl=12, corr=6 |
| min_confidence | **67%** | Transition nette a 67%, 70% trop restrictif pour volume |
| Time stop | **none** | Degrade sur cette config |
| Entry window | **02:00-14:00 CT** | |
| Flat time | 15:30 CT | |
| Barres par jour | 264 (22h x 12 bars/h) | Session 17:30-15:30 CT |
| **Resultats** | **225 trades, 67.1% WR, $25,790, PF 1.83, Sharpe 1.26** | |
| **OOS** | **PF 2.35, Sharpe 2.58 (meilleur que IS!)** | |
| **Walk-Forward** | **4/5, $14,370 total** | |

### Config C (backup) — NQ_YM 5min + filtre horaire

| Parametre | Valeur |
|-----------|--------|
| OLS lookback | 2970 bars (~11.25j) |
| Z-score window | 42 bars (3h30) |
| z_entry | 2.95 |
| z_exit | 1.25 |
| z_stop | 4.00 |
| min_confidence | 67% |
| Time stop | none |
| Entry window | 02:00-15:00 CT |
| **Filtre horaire** | **Entrees 8h-11h CT uniquement** |
| **Resultats (avec filtre)** | 151 trades, 68.2% WR, $21,320, PF 1.78, Sharpe 2.59 |
| **OOS (sans filtre)** | PF 1.38, WF 4/5 |

### Config Kalman — K_Balanced (champion, complementaire a OLS)

OLS reste le moteur de signaux. Kalman est affiche en textbox comme biais discretionnaire (beta + z-score + direction). OLS et Kalman perdent sur des jours/heures/regimes differents = complementarite structurelle.

| Parametre | Valeur | Notes |
|-----------|--------|-------|
| Methode | Kalman | Innovation z-score nu/sqrt(F), auto-adaptatif |
| alpha_ratio | **3e-7** | Beta tres stable, adaptation lente |
| Profil metrics | tres_court | adf=12, hurst=64, hl=12, corr=6 |
| z_entry | **1.375** | Innovation z ~N(0,1), plus bas que OLS |
| z_exit | **0.25** | Sort tard (laisse courir) |
| z_stop | **2.75** | Sweet spot universel Kalman |
| min_confidence | **75%** | Strict |
| Entry window | **03:00-12:00 CT** | |
| **Resultats E-mini** | **238 trades, 75.6% WR, $84,825, PF 1.78** | MaxDD $19k = DANGER propfirm |
| **Resultats Micro x2** | **238 trades, 75.6% WR, $15,936, PF 1.67** | MaxDD $3,832 = SAFE propfirm |
| **Validation** | **IS/OOS GO, WF 5/5 ($32k total), Permutation p=0.000** | |
| **Complementarite** | **1 seul jour de perte commun avec OLS sur 6 ans** | |

### Micro Contracts (MNQ/MYM)

Pour le trading propfirm, les configs Kalman E-mini ont des MaxDD $8k-$45k (DANGER pour trailing DD $4,500).
Solution : micro contracts (MNQ/MYM) a x2 = sweet spot propfirm (92.6% configs MaxDD < $4,500).

| Scaling | PnL | MaxDD | Streak | Status |
|---------|-----|-------|--------|--------|
| E-mini x1 | $84,825 | -$19,155 | 3 | DANGER |
| Micro x1 | $7,968 | -$1,916 | 3 | Trop petit |
| **Micro x2** | **$15,936** | **-$3,832** | **3** | **SAFE** |
| Micro x3 | $23,903 | -$5,748 | 3 | Limite |

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, statsmodels, scipy, filterpy, optuna
- **Phase 2** : C++ (ACSIL Sierra Charts API), header-only, online algorithms, no STL in hot path
