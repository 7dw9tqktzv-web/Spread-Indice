# CLAUDE.md

## Project Overview
Systeme de spread trading intraday sur futures US (NQ, ES, RTY, YM).
- **Phase 1** : Moteur de backtest Python (optimisation & validation)
- **Phase 2** : Indicateur Sierra Charts temps reel (ACSIL C++)

Le biais directionnel journalier est discretionnaire -- le systeme time l'entree avec precision statistique. **CLAUDE.md est la source de verite** pour le code. **MEMORY.md est la source de verite** pour les configs validees et resultats de recherche.

## Commands

```bash
source venv/Scripts/activate                              # ALWAYS work in venv
python -m pytest tests/ -v --tb=short                     # All 62 tests
python scripts/run_backtest.py --pair NQ_YM --method ols_rolling
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7
```

Toutes les commandes (grid, validation, analysis) : voir `COMMANDS.md`.
Scripts a executer depuis la racine (cache = `output/cache`). Raw data (`raw/*.txt`) gitignore.

## Architecture

### Implementation Status
Phase 1 complete : `src/data/`, `src/hedge/` (OLS+Kalman), `src/spread/`, `src/sizing/`, `src/stats/`, `src/metrics/`, `src/signals/` (numba JIT 451x), `src/backtest/`, `src/utils/`, `config/` (4 YAML), 40 scripts, 62 tests. Stubs : `src/optimisation/`.
Phase 2a C++ : `sierra/NQ_YM_SpreadMeanReversion_v1.0.cpp` (1537 lignes) -- indicateur visuel valide.

### Data Flow
`raw/*.txt` (Sierra CSV 1min) -> `loader` -> `cleaner` -> `resampler` (1/3/5min) -> `alignment` (pair) -> `hedge/` (ratio) -> `spread/builder` -> `metrics/` -> `signals/` -> `backtest/engine` -> `performance`

Dependencies flow strictly downward. Config YAML loaded at script level, injected as dataclasses.

### Key Modules
- **`src/data/`** -- Pipeline: `loader` -> `cleaner` -> `resampler` -> `alignment`. Cache: `cache.py` (Parquet)
- **`src/hedge/`** -- `HedgeRatioEstimator` ABC (`base.py`). Impl: `ols_rolling.py`, `kalman.py`. Factory: `factory.py` (`HedgeMethod` enum). Configs: `OLSRollingConfig`, `KalmanConfig`
- **`src/sizing/`** -- Dollar-neutral x beta: `N_b = round((Not_A / Not_B) x beta x N_a)`
- **`src/stats/`** -- Pure functions: hurst (variance-ratio), halflife (rolling cov), correlation, stationarity (2 ADF variants)
- **`src/metrics/`** -- Aggregation: `MetricsConfig` + `compute_all_metrics()` -> DataFrame `adf_stat, hurst, half_life, correlation`
- **`src/signals/`** -- `generator.py`: 4-state machine (numba JIT). `filters.py`: confidence, time stop, window filter (all numba)
- **`src/backtest/`** -- `engine.py`: bar-by-bar + vectorized + grid-optimized. `performance.py`: PerformanceMetrics
- **`config/`** -- `instruments.yaml`, `pairs.yaml`, `backtest.yaml`, `optimisation.yaml`
- **`sierra/`** -- Phase 2 ACSIL C++. Production: `NQ_YM_SpreadMeanReversion_v1.0.cpp`. Source aussi: `F:\SierreChart_Spread_Indices\ACS_Source\`

### Non-Obvious Architectural Details

1. **ABC signature**: `HedgeRatioEstimator.estimate(aligned: AlignedPair) -> HedgeResult` -- full AlignedPair, log-price conversion inside.

2. **HedgeResult bundles spread AND zscore**: OLS uses `zscore_window` rolling, Kalman uses innovation nu(t)/sqrt(F(t)).

3. **Session filter wraps midnight via OR**: `t >= 17:30 OR t < 15:30` (overnight). Naive range check = wrong.

4. **Column naming**: `BarData.df` uses `close`, `AlignedPair.df` uses `close_a`/`close_b`.

5. **Imports**: `pyproject.toml` sets `pythonpath=["."]` -> `from src.xxx import yyy`.

6. **Kalman**: `Q = alpha_ratio x R x I`; gap >30min -> P *= 10; Joseph form update. Alpha: 1.5e-7 to 3e-7. Innovation z-score N(0,1) -> z_entry=1.5-2.0 (NOT same as OLS 3.15). `r_ewma_span` and `adaptive_Q` INVALIDATED (MaxDD 2-7x worse, never activate).

7. **Two ADF**: `adf_rolling()` (statsmodels, p-values) vs `adf_statistic_simple()` (custom, statistics). Dashboard uses simple variant (Sierra C++ compatible).

8. **Regression convention**: `log_a = alpha + beta x log_b + epsilon`. Both OLS and Kalman. Beta directly in sizing.

9. **4-state machine**: FLAT -> LONG/SHORT -> COOLDOWN -> FLAT. COOLDOWN blocks re-entry until `|z| < z_exit`. NaN resets to FLAT.

10. **Confidence scoring**: scores 0->1 via linear interpolation, ADF gate at -1.00 -> 0%. Weights pair-specific:
    - **NQ_YM** : ADF 40%, Hurst 25%, Corr 20%, HL 15%
    - **NQ_RTY** : ADF 50%, Hurst 30%, Corr 20%, HL 0%

## Research Context
Lire `CHANGELOG.md` en debut de session. Verifier avant de proposer un test.

## Instruments & Pairs
6 instruments (NQ, ES, RTY, YM + MNQ, MYM). Paires viables : NQ/YM (OLS+Kalman), NQ/RTY (OLS valide, Kalman en cours). Rejetees : NQ/ES, ES/RTY, ES/YM, RTY/YM. Micros : 1/10e standard, commission $0.62 RT.

## Configs Validees
Voir MEMORY.md pour les parametres et resultats detailles de chaque config (Config E, K_Balanced, NQ_RTY Top 3).

## Phase 2a -- Sierra NQ_YM (VALIDE)
Fichier : `sierra/NQ_YM_SpreadMeanReversion_v1.0.cpp` (1537 lignes). DLL 64-bit, VS 2022 Build Tools.
30 subgraphs, 22 inputs, 7 fonctions utilitaires. Kalman via PersistentDouble (precision double requise, Q=3e-13). H centering (`log_ym - center`) pour stabilite numerique.
Parite signaux C++/Python : **99.9%**. Metriques brutes : Spread r=0.996, Z-Score r=0.974, Beta r=0.971.
Phase 2b a venir : auto-trading, dollar stop, regime indicator.

## Key Conventions
- Toujours travailler en **venv**
- Donnees en **Chicago Time (CT)**, calculs sur **log-prix** (ln)
- Session : 17h30-15h30 CT (Globex), fenetre trading configurable
- **Git** : utiliser `gh` pour commits/push -- jamais de commandes git manuelles
- Valider chaque etape avec l'utilisateur avant de passer a la suivante
- **NQ/YM et NQ/RTY sont des paires independantes** -- ne jamais comparer

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, numba, statsmodels, scipy, filterpy, optuna, pyarrow
- **Phase 2** : C++ (ACSIL Sierra Charts API), header-only, online algorithms. VS 2022 Build Tools.
