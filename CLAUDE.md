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
Phase 1 complete : `src/data/`, `src/hedge/` (OLS+Kalman), `src/spread/`, `src/sizing/`, `src/stats/`, `src/metrics/`, `src/signals/` (numba JIT 451x), `src/backtest/`, `src/utils/`, `src/validation/` (CPCV, gates, propfirm, neighborhood, DSR), `config/` (3 YAML), ~32 scripts actifs + 18 archives, 107 tests.
Phase 2a C++ : `sierra/NQ_YM_SpreadMeanReversion_v1.0.cpp` -- indicateur visuel valide. Phase 2a NQ_RTY a venir.
Phase 2b v2 C++ : meme fichier (~2150 lignes) -- semi-auto trading (BUY/SELL/FLATTEN + auto-exits + auto-entry + scaling). Teste en simulation et replay.
Phase 13c : Config D NQ_YM OLS validee (PF 2.13, 153 trades, CPCV 97.8%). Methodologie reproductible dans `METHODOLOGY.md`.

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
- **`config/`** -- `instruments.yaml`, `pairs.yaml`, `backtest.yaml`
- **`sierra/`** -- Phase 2 ACSIL C++. Production: `NQ_YM_SpreadMeanReversion_v1.0.cpp`. Source aussi: `F:\SierreChart_Spread_Indices\ACS_Source\`
- **`scripts/archive/`** -- 18 scripts superseded (Phases 6-13a). Reference historique uniquement.

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

10. **Confidence scoring** (Phases 6-12): scores 0->1 via linear interpolation, ADF gate at -1.00 -> 0%. Weights pair-specific:
    - **NQ_YM** : ADF 40%, Hurst 25%, Corr 20%, HL 15%
    - **NQ_RTY** : ADF 50%, Hurst 30%, Corr 20%, HL 0%

11. **Binary gates** (Phase 13+): remplacent le scoring continu pour NQ_YM. ADF < -2.86, Hurst < 0.50, Corr > 0.70. Toutes doivent passer (AND). Plus robuste que le scoring continu (pas de compensation). `apply_gate_filter_numba()` bloque les ENTREES quand gate=False, ne bloque jamais les sorties.

12. **CPCV(10,2)**: 45 chemins combinatoriaux. Sharpe = mean/std des PnL (PAS annualise, pas de sqrt(N)). Trade attribue si entree ET sortie dans les blocs test. Purge 100 barres (~8h).

13. **Delta sigma**: `z_exit = max(z_entry - delta_tp, 0)`, `z_stop = z_entry + delta_sl`. Zero exclusions logiques.

## Research Context
Lire `CHANGELOG.md` en debut de session. Verifier avant de proposer un test.

## Instruments & Pairs
6 instruments (NQ, ES, RTY, YM + MNQ, MYM). Paires viables : NQ/YM (OLS+Kalman), NQ/RTY (OLS+Kalman). Rejetees : NQ/ES, ES/RTY, ES/YM, RTY/YM. Micros : 1/10e standard, commission $0.62 RT.

## Configs Validees
Voir MEMORY.md pour les parametres et resultats detailles.

### NQ_YM -- Config D (Phase 13c, PRODUCTION)
OLS=7000, ADF_w=96, ZW=30, 02:00-14:00, ze=3.25, zx=0.50, zs=4.75, ts=0, binary gates.
153 trades, PF 2.13, WR 69%, $24,880, DD -$4,595. CPCV 97.8% paths+. GO.

### NQ_RTY -- Config #8 (Phase 11, E-mini only)
OLS=9240, ZW=20, p36_96, 06:00-14:00, ze=3.00, zx=0.75, zs=5.0, conf=75.
182 trades, PF 2.10, WR 71%, $40,730, DD -$4,605. WF 5/5.

Autres configs (K_Balanced NQ_YM, K4_tc NQ_RTY, etc.) dans MEMORY.md.

## Phase 2a -- Sierra NQ_YM (VALIDE)
Fichier : `sierra/NQ_YM_SpreadMeanReversion_v1.0.cpp` (~1480 lignes). DLL 64-bit, VS 2022 Build Tools.
24 subgraphs (6 dead slots supprimes), 22 inputs, 7 fonctions utilitaires. Kalman via PersistentDouble (precision double requise, Q = alpha_ratio x R ~ 3e-12). H = [1, log_ym] sans centering (parite Python exacte).
Gap detection via GetDate()/GetTimeInSeconds() absolus (gere weekends/holidays).
Parite signaux C++/Python : **99.9%**. Metriques brutes : Spread r=0.996, Z-Score r=0.974, Beta OLS r=0.971.
Compilation : `F:\SierreChart_Spread_Indices\ACS_Source\VisualCCompile.Bat` -> `Data\NQ_YM_SpreadMeanReversion_64.dll`.

## Phase 2b -- Semi-Auto Trading NQ_YM (VALIDE)
Fichier : meme cpp (~2150 lignes). Inputs 22-31, PersistentInt 4-7, PersistentDouble 8-10.
**Control Bar Buttons** : BUY SP / SELL SP / FLAT SP via ACS_BUTTON_1/2/3. Clic → PendingOrderAction → execution au prochain tick.
**Auto-Entry** : Input[31] toggle. Detection FLAT→LONG/SHORT sur chaque barre (!sc.IsFullRecalculation). Fonctionne en replay rapide (960X).
**Auto-Exits** : z_exit, dollar stop ($), time stop (CT). Tous configurables via Inputs. Enable Auto Exit toggle.
**Scaling** : ajout meme direction autorise. Cooldown 10s anti double-clic. Direction opposee bloquee (FLATTEN first).
**Position Sync** : `GetTradePositionForSymbolAndAccount()` = source de verite. Desync auto-corrigee. FLATTEN utilise qty reelles.
**P&L Live** : calcul manuel (AveragePrice × dollarPerPoint × qty). OpenProfitLoss = total compte (inutilisable). Dollar stop utilise tradePnL.
**Panel TRADING** : bleu (P&L>=0), orange (P&L<0), bold 10pt en position, 8pt flat.
**Architecture ordres** : `sc.BuyOrder()` / `sc.SellOrder()` avec `.Symbol` explicite (cross-symbol). Deferred order pattern via `PendingOrderAction` (PersistentInt 6) pour eviter -8998 pendant full recalc.
**Teste en replay** : auto-entry + auto-exit valides. BUY/SELL/FLATTEN/scaling valides sur Teton Sim1.

## Key Conventions
- Toujours travailler en **venv**
- Donnees en **Chicago Time (CT)**, calculs sur **log-prix** (ln)
- Session : 17h30-15h30 CT (Globex), fenetre trading configurable
- **Git** : utiliser `gh` pour commits/push -- jamais de commandes git manuelles
- Valider chaque etape avec l'utilisateur avant de passer a la suivante
- **NQ/YM et NQ/RTY sont des paires independantes** -- ne jamais comparer

## Tech Stack
- **Phase 1** : Python 3.11+, venv, pandas, numpy, numba, statsmodels, scipy, pyarrow, matplotlib
- **Phase 2** : C++ (ACSIL Sierra Charts API), header-only, online algorithms. VS 2022 Build Tools.
- **Linting** : ruff (py311, line-length=100, rules E/F/W/I/UP/B). Config in `pyproject.toml`.
- **CI** : `.github/workflows/test.yml` -- pytest + ruff check on windows-latest, Python 3.11.
