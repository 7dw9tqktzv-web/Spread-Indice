# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Systeme de spread trading intraday sur futures US (NQ, ES, RTY, YM).
- **Phase 1** : Moteur de backtest Python (optimisation & validation)
- **Phase 2** : Indicateur Sierra Charts temps reel (ACSIL C++)

Le biais directionnel journalier est discretionnaire -- le systeme time l'entree avec precision statistique sur des billets macroeconomiques. **CLAUDE.md est la source de verite** pour l'etat actuel du code.

## Commands

```bash
# Activate venv (ALWAYS work in venv)
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run all tests (62 tests)
python -m pytest tests/ -v --tb=short

# Run unit tests only (skip integration)
python -m pytest tests/ -v --ignore=tests/test_integration

# Run a single test file
python -m pytest tests/test_hedge/test_ols_rolling.py -v

# Run backtest pipeline
python scripts/run_backtest.py --pair NQ_YM --method ols_rolling
python scripts/run_backtest.py --pair NQ_RTY --method kalman --alpha-ratio 3e-7

# Grid search NQ_YM
python scripts/run_refined_grid.py --workers 5          # OLS refined (1,080,000 combos)
python scripts/run_grid_kalman_v3.py --workers 10       # Kalman (1,009,800 combos)
python scripts/run_grid.py --workers 20                 # OLS broad (43,200, 6 pairs)

# Grid search NQ_RTY
python scripts/refine_ols_balanced.py --workers 10      # OLS refined (14,374,656 combos)
python scripts/run_grid_kalman_NQ_RTY.py --workers 10   # Kalman (856,800 combos)
python scripts/run_grid_ols_NQ_RTY.py --workers 10      # OLS broad (15,945,930 combos)

# Validation
python scripts/validate_top5_configs.py                 # OLS NQ_YM validation
python scripts/validate_kalman_top.py                   # Kalman NQ_YM validation
python scripts/validate_NQ_RTY_top6.py                  # OLS NQ_RTY validation (6 configs)
python scripts/validate_top_NQ_RTY.py                   # Kalman NQ_RTY validation

# Analysis
python scripts/analyze_grid_results.py                  # Deep analysis NQ_YM
python scripts/analyze_grid_NQ_RTY.py                   # Kalman NQ_RTY dimensionnel
python scripts/rank_top_configs_NQ_RTY.py               # Multi-criteria ranking NQ_RTY
python scripts/compare_candidates_NQ_RTY.py             # Overlap analysis NQ_RTY
python scripts/check_maxdd_refined_NQ_RTY.py            # MaxDD check NQ_RTY
python scripts/find_safe_kalman.py                      # E-mini safe configs
python scripts/find_safe_kalman_micro.py                # Micro contracts (x1/x2/x3)
python scripts/analyze_2023_losses.py                   # Autopsie 2023
```

**Important**: All scripts must be run from the project root (cache uses relative path `output/cache`). Raw data (`raw/*.txt`, Sierra CSV exports) is gitignored -- must be provided externally.

## Output Structure

```
output/
  cache/          # Parquet data (aligned pairs, all timeframes)
  backtests/      # Individual pair backtests (NQ_ES/, NQ_RTY/, NQ_YM/)
  NQ_YM/          # Grid results NQ_YM
    grid_refined_ols.csv          # Phase 6 OLS (1,080,000 combos)
    grid_kalman_v3.csv            # Phase 7 Kalman (1,009,800 combos)
    grid_kalman_v3_filtered.csv
  NQ_RTY/         # Grid results NQ_RTY
    grid_refined_ols.csv          # Phase 10 OLS (14,374,656 combos)
    grid_refined_ols_filtered.csv
    grid_kalman.csv               # Phase 9 Kalman (856,800 combos)
    grid_kalman_filtered.csv
```

## Architecture

### Implementation Status
- **Implemented**:
  - `src/data/` -- loader, cleaner, resampler, alignment, cache
  - `src/hedge/` -- OLS rolling + Kalman + factory dispatch
  - `src/spread/` -- SpreadPair dataclass
  - `src/sizing/` -- dollar-neutral position sizing + optimal multiplier
  - `src/stats/` -- vectorized hurst (variance-ratio), halflife, correlation, stationarity (2 ADF variants)
  - `src/metrics/` -- dashboard + confidence scoring
  - `src/signals/` -- generator (numba JIT 451x), filters (numba: confidence, time stop, window filter)
  - `src/backtest/` -- engine (bar-by-bar + vectorized + grid-optimized), performance
  - `src/utils/` -- time_utils, constants
  - `config/` -- 4 YAML configs
  - `scripts/` -- 40 scripts (backtest, grid search, validation, analysis, diagnostics)
  - `tests/` -- 62 tests (24 hedge + 9 signals/generator + 9 signals/filters + 8 backtest/engine + 12 sizing/stops)
- **Phase 2a C++ (NQ_YM)**: `sierra/NQ_YM_SpreadMeanReversion_v1.0.cpp` (1537 lines) — visual indicator compiled & validated
- **Stubs only**: `src/optimisation/`
- **Numba JIT**: signal generator (451x), confidence filter, time stop, window filter -- all validated identical to Python originals

### Data Flow
`raw/*.txt` (Sierra CSV 1min) -> `loader.py` -> `cleaner.py` -> `resampler.py` (1min/3min/5min) -> `alignment.py` (pair) -> `hedge/` (ratio) -> `spread/builder.py` -> `metrics/` -> `signals/` -> `backtest/engine.py` -> `performance.py`

Dependencies flow strictly downward. Config YAML files are loaded at script level and injected as dataclasses -- modules never read config directly.

### Key Modules
- **`src/data/`** -- Pipeline: `loader.py` -> `cleaner.py` -> `resampler.py` -> `alignment.py`. Cache via `cache.py` (Parquet)
- **`src/hedge/`** -- Hedge ratio estimators behind `HedgeRatioEstimator` ABC (`base.py`). Implemented: `ols_rolling.py`, `kalman.py`. Factory dispatch via `factory.py` (keyed on `HedgeMethod` enum). Config dataclasses: `OLSRollingConfig`, `KalmanConfig`
- **`src/sizing/`** -- Dollar-neutral x beta position sizing (scalar + vectorized): `N_b = round((Notionnel_A / Notionnel_B) x beta x N_a)`
- **`src/stats/`** -- Low-level statistical functions: hurst (variance-ratio, vectorized), halflife (AR(1) via rolling cov, vectorized), correlation, stationarity (2 ADF variants)
- **`src/metrics/`** -- Aggregation layer (`dashboard.py`): `MetricsConfig` + `compute_all_metrics()` calls `src/stats/` functions, returns DataFrame with `adf_stat, hurst, half_life, correlation`
- **`src/signals/`** -- `generator.py`: stateful 4-state machine (numba JIT, 451x speedup). `filters.py`: confidence scoring (vectorized numpy, 138x), `_apply_conf_filter_numba`, `apply_time_stop`, `apply_window_filter_numba` (all numba JIT)
- **`src/backtest/`** -- `engine.py`: BacktestEngine (bar-by-bar), `run_backtest_vectorized` (full with equity), `run_backtest_grid` (no equity, grid-optimized). `performance.py`: PerformanceMetrics
- **`src/spread/`** -- `SpreadPair` dataclass (`pair.py`)
- **`config/`** -- YAML configs: `instruments.yaml`, `pairs.yaml`, `backtest.yaml`, `optimisation.yaml`
- **`sierra/`** -- Phase 2 ACSIL C++ indicators + reference docs. `NQ_YM_SpreadMeanReversion_v1.0.cpp` is the production indicator (1537 lines). Source also at `F:\SierreChart_Spread_Indices\ACS_Source\`. Compiles to `NQ_YM_SpreadMeanReversion_64.dll` via VS 2022 Build Tools.

### Non-Obvious Architectural Details

1. **ABC signature**: `HedgeRatioEstimator.estimate(aligned: AlignedPair) -> HedgeResult` -- estimators receive the full `AlignedPair` object, not raw series. Log-price conversion happens inside each estimator.

2. **`HedgeResult` bundles spread AND zscore**: The estimator computes both. The z-score method differs by estimator -- OLS uses `zscore_window` rolling, Kalman uses innovation-based nu(t)/sqrt(F(t)) (no separate window).

3. **Session filter wraps midnight via OR logic**: `t >= buf_start OR t < buf_end` (session is 17:30->15:30 CT overnight). A naive range check would be wrong.

4. **Column naming convention**: `BarData.df` uses `close`, but after `align_pair()` the `AlignedPair.df` uses `close_a` / `close_b`.

5. **Imports**: `pyproject.toml` sets `pythonpath = ["."]` -- all imports use `from src.xxx import yyy`.

6. **Kalman specifics**: `Q = alpha_ratio x R x I` (scale-aware); session gaps (>30min) multiply `P` by `gap_P_multiplier=10.0`; Joseph form covariance update. Alpha sweet spot: 1.5e-7 to 3e-7. Warmup and gap_P_mult have negligible impact. Innovation z-score is N(0,1) by construction -- optimal z_entry=1.5-2.0, z_stop=2.5-2.8 (NOT the same as OLS z_entry=3.15). `HedgeResult.diagnostics` contains `P_trace`, `K_beta`, `R_history` Series (cost=0). `KalmanConfig.r_ewma_span` (default 0=off) and `adaptive_Q` (default False) exist but are INVALIDATED -- never activate (MaxDD degrades 2-7x). P_trace is a pure temporal proxy (Spearman rho vs time = -1.000) -- never use as filter.

7. **Two ADF implementations in `stationarity.py`**: `adf_rolling()` (statsmodels) and `adf_statistic_simple()` (custom, no augmentation) -- the simple one is designed for Python/C++ parity testing.

8. **Regression convention (OLS + Kalman aligned)**: `log_a = alpha + beta x log_b + epsilon` -- leg_a is dependent, leg_b is explanatory. Both estimators use this same convention. beta is directly usable in the sizing formula.

9. **Signal generator has 4 states, not 3**: FLAT -> LONG/SHORT -> COOLDOWN -> FLAT. After a stop loss, the COOLDOWN state blocks re-entry until `|z| < z_exit` (spread must return to neutral). NaN resets to FLAT (clean session start).

10. **`src/stats/` vs `src/metrics/`**: `stats/` contains pure computation functions. `metrics/dashboard.py` is the aggregation layer that calls `stats/` and returns a unified DataFrame. They are separate layers, not duplicates.

11. **Calculate vs Act separation**: Hedge ratio + metrics compute on full Globex session (17:30-15:30 CT, buffer_minutes=0). Signals + trades restricted to configurable entry window via `apply_window_filter_numba()`. Position open near flat time is force-closed (signal -> 0).

12. **ADF in dashboard uses `adf_statistic_simple()`** (statistics ~-3.5, threshold -2.86), NOT `adf_rolling()` (p-values 0-1). The simple variant is Sierra C++ compatible.

13. **Confidence scoring** replaces binary regime filter: each metric produces a score 0->1 via linear interpolation, with ADF gate at stat >= -1.00 -> 0%. Weights are **pair-specific**:
    - **NQ_YM** : ADF 40%, Hurst 25%, Corr 20%, HL 15% (standard `ConfidenceConfig()`)
    - **NQ_RTY** : ADF 50%, Hurst 30%, Corr 20%, HL 0% (ablation: HL retire, +155% trades, PnL stable)

14. **Session is 17:30-15:30 CT** (already buffered vs Globex 17:00-16:00). `buffer_minutes=0` in config. 264 bars/day at 5min. OLS windows: 1320 (5j), 2640 (10j), ..., 9240 (35j).

15. **Hurst uses variance-ratio** (not R/S). R/S gives biased ~0.99 on spread levels. Variance-ratio works on levels directly. Vectorized via precomputed rolling std.

16. **Half-life vectorized** via rolling covariance: `b = Cov(Z(t),Z(t-1)) / Var(Z(t-1))`, `HL = -ln(2)/ln(b)`.

17. **2023 : seule annee negative Kalman NQ_YM** : NQ +38.2% vs YM +8.9% (tech rally) -> spread drift +29.2%, ACF(1) = 0.000 (zero mean reversion). FLAT exits = 86% des pertes. Le confidence scoring est aveugle a ce regime daily. Implication Phase 2 : afficher indicateur de regime daily + gestion FLAT.

### Instruments & Pairs
- **6 instruments** : NQ, ES, RTY, YM (standards) + MNQ, MYM (micros)
- **Paires viables** : NQ/YM (OLS + Kalman), NQ/RTY (OLS valide, Kalman en cours)
- **Paires rejetees** : NQ/ES (Kalman marginal), ES/RTY (biais directionnel pur), ES/YM (systematiquement perdant), RTY/YM (pas teste en profondeur)
- **ES/RTY** : exploration legere seulement (90 configs Kalman, 6 OLS) -- pas de grid exhaustif, a retester
- **Micros** : MNQ/MYM = 1/10eme du standard, memes prix, commission $0.62 RT
- **Dollar stop** : `dollar_stop` dans BacktestConfig (defaut 0 = desactive). Non recommande sur standard (coupe les trades trop tot)

## Research Context
**Toujours lire `CHANGELOG.md` en debut de session** pour avoir le contexte des derniers backtests effectues, resultats et decisions. Ce fichier est l'historique de recherche du projet -- il evite de refaire des tests deja faits et permet de reprendre la ou on s'est arrete.

Avant de proposer un test ou une config, verifier dans le changelog si ca n'a pas deja ete teste. Mettre a jour le changelog apres chaque serie de tests significative.

## Key Conventions
- Toujours travailler en **venv**
- Donnees en **Chicago Time (CT)**
- Tous les calculs sur **log-prix** (ln) pour OLS et Kalman
- Session : 17h30-15h30 CT (Globex), fenetre trading configurable par config
- **Git** : utiliser l'agent GitHub (`gh`) pour tous les commits, push et operations de branche -- jamais de commandes git manuelles
- Valider chaque etape avec l'utilisateur avant de passer a la suivante
- Parametres optimises en Phase 1 avant implementation Phase 2
- **NQ/YM et NQ/RTY sont des paires independantes** -- ne jamais comparer directement

---

## Configs Validees

### NQ_YM -- Config E (principale OLS)

Issue du grid search affine 1,080,000 combos. Validee IS/OOS, Walk-Forward 4/5, Permutation p=0.000.

| Parametre | Valeur |
|-----------|--------|
| Paire | NQ_YM (mult 20/5, tick 0.25/1.0) |
| OLS lookback | 3300 bars (~12.5j) |
| Z-score window | 30 bars (2h30) |
| z_entry / z_exit / z_stop | 3.15 / 1.00 / 4.50 |
| Profil metrics | tres_court (adf=12, hurst=64, hl=12, corr=6) |
| min_confidence | 67% (poids standard ADF/H/C/HL = 40/25/20/15) |
| Entry window | 02:00-14:00 CT, flat 15:30 |
| **Resultats** | **225 trades, 67.1% WR, $25,790, PF 1.83, Sharpe 1.26, MaxDD -$5,190** |
| **Validation** | **OOS PF 2.35 > IS, WF 4/5, Permutation p=0.000** |

### NQ_YM -- Kalman K_Balanced (indicateur complementaire)

OLS = moteur de signaux. Kalman = textbox Sierra biais discretionnaire. Complementarite structurelle (1 seul jour de perte commun sur 6 ans).

| Parametre | Valeur |
|-----------|--------|
| alpha_ratio | 3e-7 |
| z_entry / z_exit / z_stop | 1.375 / 0.25 / 2.75 |
| Profil | tres_court, conf 75% |
| Entry window | 03:00-12:00 CT |
| **E-mini** | **238 trades, 75.6% WR, $84,825, PF 1.78, MaxDD -$19,155 (DANGER propfirm)** |
| **Micro x2** | **238 trades, $15,936, PF 1.67, MaxDD -$3,832 (SAFE propfirm)** |
| **Validation** | **IS/OOS GO, WF 5/5 ($32k total), Permutation p=0.000** |

### NQ_RTY -- Top 3 OLS (Phase 10, 14.4M combos)

6/6 configs validees, 3 retenues pour diversite (~20% overlap entre elles).
Toutes utilisent CONF_WEIGHTS = ADF 50%, Hurst 30%, Corr 20%, HL 0%.
Multipliers: NQ mult=20 tick=0.25, RTY mult=50 tick=0.10.

| Config | OLS | ZW | Profil | Window | ze/zx/zs | conf | Trd | WR | PnL | PF | WF | OOS PF |
|--------|-----|----|--------|--------|----------|------|-----|----|-----|----|----|--------|
| **A_RTY** | 9240 | 20 | p36_96 | 06-14 | 3.00/0.75/5.0 | 75 | 163 | 65% | $19,205 | 2.10 | 5/5 | 3.48 |
| **G_sniper** | 3960 | 24 | p16_80 | 06-14 | 3.50/0.50/4.5 | 70 | 152 | 66% | $20,350 | 2.17 | 4/5 | 3.93 |
| **D_court** | 3960 | 28 | p28_144 | 02-14 | 3.00/1.25/5.5 | 80 | 195 | 59% | $14,710 | 1.95 | 5/5 | 2.25 |

Sweet spots NQ/RTY vs NQ/YM : OLS plus long (7920-9240 vs 3300), ZW plus reactif (20 vs 30), profil p36_96 dominant (ADF lent + Hurst rapide), z_exit plus bas (0.75 vs 1.00).

### NQ_RTY -- Kalman (Phase 9 -- EN COURS)

Grid 856,800 combos complete. 5 configs testees, toutes echouent IS/OOS strict (PF OOS < 1.0) + biais long 82%.
**A retester** avec angle different (symetrie L/S, stabilite annuelle).

---

## Phase 2a — Indicateur Sierra NQ_YM (VALIDE)

### Fichier : `sierra/NQ_YM_SpreadMeanReversion_v1.0.cpp` (1537 lignes)

Indicateur visuel ACSIL C++ temps reel. Pas de trading auto (Phase 2b).
DLL : `NQ_YM_SpreadMeanReversion_64.dll`. Source aussi : `F:\SierreChart_Spread_Indices\ACS_Source\`.
Compile avec VS 2022 Build Tools (batch file).

### Architecture C++
- **Mode** : AutoLoop=1, `sc.CalculationPrecedence = LOW_PREC_LEVEL` (multi-chart)
- **Multi-chart** : NQ (chart 5, leg A dependante), YM (chart 3, leg B explicative) via `sc.GetChartBaseData()`
- **30 subgraphs** : Spread+Bandes (0-3), Z-Score+Seuils (4-11), Stats (12-17), Confiance+Signaux (18-22), Kalman (23-25), Internes (26-29)
- **22 inputs** : Config charts, OLS params, Z-Score seuils, Metrics windows, Confidence, Session/Window, Kalman params, Display
- **7 fonctions utilitaires** : CalculateOLSBeta, CalculateStdDev, CalculateCorrelation, CalculateADFSimple, CalculateHurstVR, CalculateHalfLife, CalculateConfidence
- **TextBox unifiee** : OLS + Kalman, fond dynamique (vert/orange/rouge/gris), coloration per-bar (Z-Score, ADF, Hurst, Corr, Confidence)

### Kalman Filter C++ (specifites)
- **Persistent doubles** (pas float) : `sc.GetPersistentDouble(1..8)` pour theta[2], P[4], R, center
- **H centering** : `H = [1, log_ym - center]` pour stabilite numerique (sans centering, P devient quasi-singuliere apres 1er update car H=[1,10.69])
- **Initialisation** : R estime a bar 999 (variance OLS sur 1000 bars), theta=[0,1], P=I. Kalman tourne depuis bar 1000 (avant OLS bar 3299)
- **Q = alpha_ratio * R * I** : avec alpha_ratio=3e-7, Q=3e-13 — necessite double precision (perdu en float32)
- **Gap detection** : gap > 30min entre barres → P *= gap_P_multiplier

### Execution flow par barre
```
Bar 0-998:     Log prices → pas de Kalman → early return (subgraphs = 0)
Bar 999:       Log prices → Kalman INIT (R, theta, P) + 1er update → early return
Bar 1000-1098: Log prices → Kalman warmup (pas d'affichage) → early return
Bar 1099-3298: Log prices → Kalman affiche (beta, z-inn) → early return
Bar 3299+:     Log prices → Kalman → OLS → Stats → Confidence → Signaux → TextBox
```

### Validation parite C++/Python (10,000 barres)

**Metriques brutes** (correlation Pearson) :
| Metrique | r | Status |
|----------|---|--------|
| Spread | 0.996 | PASS |
| Correlation | 0.997 | PASS |
| Z-Score | 0.974 | PASS |
| OLS Beta | 0.971 | PASS |
| Hurst | 0.962 | PASS |
| StdDev | 0.954 | PASS |
| Confidence | 0.791 | Borderline (OK apres filtre) |
| ADF | 0.533 | Diverge (fenetre 12 bars) |
| Half-Life | 0.479 | Diverge (fenetre 12 bars) |

**Cause ADF/HL** : fenetres de 12 bars amplifient les differences de conditionnement des donnees (Python clean/resample vs Sierra interne). Pas un bug algorithmique.

**Parite des signaux** (ce qui compte en production) :
| Niveau | Accord |
|--------|--------|
| Zone entry Z-Score (|z| > 3.15) | **99.8%** |
| Direction combinee (z + confiance) | **100.0%** |
| Signaux apres filtre confiance | **99.6%** |
| **Signaux finaux (window + flat)** | **99.9%** |

**Verdict : VALIDE** — les decisions de trading convergent malgre les ecarts sur metriques intermediaires.

### Scripts de validation Phase 2a
- `scripts/validate_sierra_v3.py` — Validation bar-a-bar metriques brutes (Pearson r)
- `scripts/validate_signal_parity.py` — Validation parite signaux (regime + state machine + trades)
- `scripts/debug_hurst_final.py` — Debug Hurst C++ (confirme correct, ~0.01 = end-of-session bars, median 0.387)

### Bugs corriges pendant Phase 2a
1. **Kalman gain NaN** : F > 1e-12 guard avant division KG0/KG1
2. **Pre-OLS barres** : early return avec subgraphs=0 pour barres < 3300
3. **Kalman float32 underflow** : Q=3e-13 perdu en float → passage en GetPersistentDouble
4. **Kalman P init** : P=R*I (R~1e-6, filtre sur-confiant) → P=I (match Python)
5. **Kalman start trop tard** : bar 3399 insuffisant avec alpha=3e-7 → restructure, Kalman depuis bar 999
6. **Kalman H non centre** : H=[1,10.69] → P quasi-singuliere (det~3e-6) → H=[1, log_ym - center]

### Phase 2b (a venir)
- Auto-trading : `sc.BuyEntry()` / `sc.SellEntry()` avec state machine
- Dollar stop loss intraday (gestion FLAT exits)
- Regime daily indicator (detection periodes type 2023)

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, numba, statsmodels, scipy, filterpy, optuna, pyarrow (cache)
- **Phase 2** : C++ (ACSIL Sierra Charts API), header-only, online algorithms, no STL in hot path. VS 2022 Build Tools.
