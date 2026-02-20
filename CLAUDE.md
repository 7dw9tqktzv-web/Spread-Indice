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
python scripts/run_backtest.py --prepare-data  # data prep only

# Run optimisation
python scripts/run_optimisation.py --pair NQ_ES --method ols_rolling --engine optuna

# Run all pairs in parallel
python scripts/run_all_pairs.py --workers 8
```

**Important**: All scripts must be run from the project root (cache uses relative path `output/cache`).

## Architecture

### Implementation Status
- **Implemented**: `src/data/`, `src/hedge/` (OLS + Kalman + factory), `src/spread/`, `src/sizing/`, `src/stats/`, `src/metrics/` (dashboard), `src/signals/` (generator + filters), `src/utils/`, `config/`, `tests/` (26 tests)
- **Stubs only**: `src/backtest/`, `src/optimisation/`, `scripts/`, `sierra/`

### Data Flow
`raw/*.txt` (Sierra CSV 1min) → `loader.py` → `cleaner.py` → `resampler.py` (5min) → `alignment.py` (pair) → `hedge/` (ratio) → `spread/builder.py` → `metrics/` → `signals/` → `backtest/engine.py` → `performance.py`

Dependencies flow strictly downward. Config YAML files are loaded at script level and injected as dataclasses — modules never read config directly.

### Key Modules
- **`src/data/`** — Pipeline: `loader.py` → `cleaner.py` → `resampler.py` → `alignment.py`. Cache via `cache.py` (Parquet)
- **`src/hedge/`** — Hedge ratio estimators behind `HedgeRatioEstimator` ABC (`base.py`). Implemented: `ols_rolling.py`, `kalman.py`. Factory dispatch via `factory.py` (keyed on `HedgeMethod` enum). Config dataclasses: `OLSRollingConfig`, `KalmanConfig`
- **`src/sizing/`** — Dollar-neutral × β position sizing (scalar + vectorized): `N_b = round((Notionnel_A / Notionnel_B) × β × N_a)`
- **`src/stats/`** — Low-level statistical functions: hurst (R/S), halflife (AR(1)), correlation, stationarity (2 ADF variants)
- **`src/metrics/`** — Aggregation layer (`dashboard.py`): `MetricsConfig` + `compute_all_metrics()` calls `src/stats/` functions, returns DataFrame with `adf_stat, hurst, half_life, correlation`
- **`src/signals/`** — `generator.py`: stateful 4-state machine (FLAT/LONG/SHORT/COOLDOWN) for z-score threshold crossings. `filters.py`: regime filters (ADF/Hurst/correlation/half-life) block entries, never exits
- **`src/spread/`** — `SpreadPair` dataclass (`pair.py`)
- **`config/`** — YAML configs: `instruments.yaml`, `pairs.yaml`, `backtest.yaml`, `optimisation.yaml`
- **`sierra/`** — Phase 2 ACSIL C++ indicator (header-only libs, online algorithms)

### Non-Obvious Architectural Details

1. **ABC signature**: `HedgeRatioEstimator.estimate(aligned: AlignedPair) -> HedgeResult` — estimators receive the full `AlignedPair` object, not raw series. Log-price conversion happens inside each estimator.

2. **`HedgeResult` bundles spread AND zscore**: The estimator computes both. The z-score method differs by estimator — OLS uses `zscore_window=12` rolling, Kalman uses innovation-based `ν(t)/√F(t)` (no separate window).

3. **Session filter wraps midnight via OR logic**: `t >= buf_start OR t < buf_end` (session is 17:30→15:30 CT overnight). A naive range check would be wrong.

4. **Column naming convention**: `BarData.df` uses `close`, but after `align_pair()` the `AlignedPair.df` uses `close_a` / `close_b`.

5. **Imports**: `pyproject.toml` sets `pythonpath = ["."]` — all imports use `from src.xxx import yyy`.

6. **Kalman specifics**: `Q = alpha_ratio × R × I` (scale-aware); session gaps (>30min) multiply `P` by `gap_P_multiplier=10.0`; Joseph form covariance update.

7. **Two ADF implementations in `stationarity.py`**: `adf_rolling()` (statsmodels) and `adf_statistic_simple()` (custom, no augmentation) — the simple one is designed for Python/C++ parity testing.

8. **Regression convention (OLS + Kalman aligned)**: `log_a = α + β × log_b + ε` — leg_a is dependent, leg_b is explanatory. Both estimators use this same convention. β is directly usable in the sizing formula.

9. **Signal generator has 4 states, not 3**: FLAT → LONG/SHORT → COOLDOWN → FLAT. After a stop loss, the COOLDOWN state blocks re-entry until `|z| < z_exit` (spread must return to neutral). NaN resets to FLAT (clean session start).

10. **`src/stats/` vs `src/metrics/`**: `stats/` contains pure computation functions (rolling hurst, halflife, etc.). `metrics/dashboard.py` is the aggregation layer that calls `stats/` and returns a unified DataFrame. They are separate layers, not duplicates.

### Instruments & Pairs
- **4 instruments** : NQ, ES, RTY, YM (contrats continus, Volume Rollover Back-Adjusted)
- **6 paires** : NQ/ES, NQ/RTY, NQ/YM, ES/RTY, ES/YM, RTY/YM

## Key Conventions
- Toujours travailler en **venv**
- Données en **Chicago Time (CT)**
- Tous les calculs sur **log-prix** (ln) pour OLS et Kalman
- Session : 17h30–15h30 CT (Globex), fenêtre trading : 4h00–14h00 CT
- **Git** : utiliser l'agent GitHub (`gh`) pour tous les commits, push et opérations de branche — jamais de commandes git manuelles
- Valider chaque étape avec l'utilisateur avant de passer à la suivante
- Paramètres optimisés en Phase 1 avant implémentation Phase 2

## Validated Parameters (5min)
| Paramètre | Valeur |
|-----------|--------|
| OLS lookback | 7200 bars (30j) |
| Z-score OLS | 12 bars (1h) |
| Corrélation | 12 bars (1h) |
| ADF | 24 bars (2h) |
| Hurst | 64 bars (~5h20) |
| Half-life | 24 bars (2h) |
| Kalman alpha_ratio | 1e-5 (à optimiser: [1e-6, 1e-5, 1e-4]) |
| Kalman warmup | 100 bars |

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, statsmodels, scipy, filterpy, optuna
- **Phase 2** : C++ (ACSIL Sierra Charts API), header-only, online algorithms, no STL in hot path
