# CLAUDE.md

## Project Overview
Systeme de spread trading sur futures US base sur un modele de cointegration globale (multi-paires, multi-secteurs).
- **Phase 1** : Moteur de backtest Python (infrastructure reutilisable)
- **Phase 2** : Indicateurs Sierra Charts temps reel (ACSIL C++)

Le biais directionnel journalier est discretionnaire -- le systeme time l'entree avec precision statistique. **CLAUDE.md est la source de verite** pour le code. **MEMORY.md est la source de verite** pour les recherches et preferences.

## Commands

```bash
source venv/Scripts/activate                              # ALWAYS work in venv
python -m pytest tests/ -v --tb=short                     # All tests
python scripts/run_backtest.py --pair NQ_YM --method ols_rolling
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7
```

Scripts a executer depuis la racine. Voir `COMMANDS.md` pour la reference complete.

## Architecture

### Implementation Status
Phase 1 : moteur de backtest Python complet -- `src/data/`, `src/hedge/` (OLS+Kalman), `src/spread/`, `src/sizing/`, `src/stats/`, `src/metrics/`, `src/signals/` (numba JIT 451x), `src/backtest/`, `src/utils/`, `src/validation/` (CPCV, gates, propfirm, neighborhood, DSR), `config/` (3 YAML), tests unitaires.
Phase 2 C++ : indicateurs ACSIL universels (visual + semi-auto trading) dans `sierra/`.

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
- **`src/validation/`** -- `cpcv.py`: CPCV(10,2) 45 chemins. `gates.py`: binary ADF/Hurst/Corr gates. `neighborhood.py`: robustesse L1. `propfirm.py`: metriques $150K. `deflated_sharpe.py`: DSR correction
- **`config/`** -- `instruments.yaml` (21 futures), `pairs.yaml`, `backtest.yaml`
- **`sierra/`** -- Phase 2 ACSIL C++. Indicateurs universels + exemples de reference. Source aussi: `F:\SierreChart_Spread_Indices\ACS_Source\`

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

10. **Confidence scoring**: scores 0->1 via linear interpolation, ADF gate at -1.00 -> 0%. Weights pair-specific.

11. **Binary gates**: ADF/Hurst/Corr thresholds. Toutes doivent passer (AND). `apply_gate_filter_numba()` bloque les ENTREES quand gate=False, ne bloque jamais les sorties.

12. **CPCV(10,2)**: 45 chemins combinatoriaux. Sharpe = mean/std des PnL (PAS annualise, pas de sqrt(N)). Trade attribue si entree ET sortie dans les blocs test. Purge 100 barres (~8h).

13. **Delta sigma**: `z_exit = max(z_entry - delta_tp, 0)`, `z_stop = z_entry + delta_sl`. Zero exclusions logiques.

## Instruments
21 futures dans `config/instruments.yaml` : indices (NQ, ES, RTY, YM), energie (CL, NG, BZ, HO, RB), metaux (GC, SI, HG, PL, PA), grains (ZC, ZW, ZS) + micros associes.

## Sierra C++ Indicators (Phase 2)
Fichiers dans `sierra/` -- indicateurs ACSIL universels (configurables via inputs pour toute paire).
- `UniversalSpreadIndicator.cpp` -- dernier indicateur semi-auto (trading + visual)
- `KalmanFixedSpreadIndicator.cpp` -- semi-auto (version precedente)
- `NQ_YM_SpreadMeanReversion_v1.0.cpp` -- reference historique (visual only)
- Exemples ACSIL de reference (`SpreadOrderEntry.cpp`, `Studies*.cpp`, etc.)

Compilation : `F:\SierreChart_Spread_Indices\ACS_Source\VisualCCompile.Bat`

## Key Conventions
- Toujours travailler en **venv**
- Donnees en **Chicago Time (CT)**, calculs sur **log-prix** (ln)
- Session : 17h30-15h30 CT (Globex), fenetre trading configurable
- **Git** : utiliser `gh` pour commits/push -- jamais de commandes git manuelles
- Valider chaque etape avec l'utilisateur avant de passer a la suivante

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, numba, statsmodels, scipy, pyarrow, matplotlib, yfinance
- **Phase 2** : C++ (ACSIL Sierra Charts API), header-only, online algorithms. VS 2022 Build Tools.
- **Linting** : ruff (py311, line-length=100, rules E/F/W/I/UP/B). Config in `pyproject.toml`.
- **CI** : `.github/workflows/test.yml` -- pytest + ruff check on windows-latest, Python 3.11.
