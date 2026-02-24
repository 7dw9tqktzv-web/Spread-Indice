# Spread Trading Indicator -- Architecture Document

**Version**: 2.0
**Updated**: 2026-02-24
**Scope**: Phase 1 (Python backtest) + Phase 2 (Sierra Charts ACSIL C++)
**Source of truth**: CLAUDE.md (this document is a structural reference)

---

## 1. Directory Structure (actual)

```
Spread_Indice/
|
|-- CLAUDE.md                          # Project rules, conventions, source of truth
|-- ARCHITECTURE.md                    # This document (structural reference)
|-- COMMANDS.md                        # Script usage reference
|-- CHANGELOG.md                       # Research phase history
|-- MEMORY.md                          # Validated configs and results
|-- requirements.txt                   # Python dependencies
|-- pyproject.toml                     # Project metadata + ruff + pytest config
|
|-- raw/                               # Raw Sierra Charts CSV exports (1min, gitignored)
|
|-- config/
|   |-- instruments.yaml               # Tick size, multiplier, margin, costs
|   |-- pairs.yaml                     # Pair definitions and default params
|   |-- backtest.yaml                  # Default backtest parameters
|
|-- src/
|   |-- __init__.py
|   |-- data/                          # Pipeline: loader -> cleaner -> resampler -> alignment
|   |   |-- loader.py                  # CSV parsing, dtype enforcement -> BarData
|   |   |-- cleaner.py                 # Session filtering, gap handling
|   |   |-- resampler.py              # 1min -> 5min OHLCV aggregation
|   |   |-- alignment.py              # Multi-instrument time alignment -> AlignedPair
|   |   |-- cache.py                   # Parquet read/write cache layer
|   |
|   |-- hedge/                         # ABC + Factory pattern
|   |   |-- base.py                    # HedgeRatioEstimator ABC, HedgeResult
|   |   |-- ols_rolling.py            # Rolling OLS beta (OLSRollingConfig)
|   |   |-- kalman.py                  # Kalman filter beta (KalmanConfig)
|   |   |-- factory.py                 # create_estimator() factory
|   |
|   |-- spread/
|   |   |-- pair.py                    # SpreadPair dataclass (pair metadata)
|   |
|   |-- stats/                         # Pure functions, no state
|   |   |-- stationarity.py           # ADF tests (statsmodels + custom)
|   |   |-- hurst.py                   # Hurst exponent (variance-ratio)
|   |   |-- halflife.py               # OU half-life via AR(1)
|   |   |-- correlation.py            # Rolling Pearson correlation
|   |
|   |-- metrics/
|   |   |-- dashboard.py              # MetricsConfig + compute_all_metrics()
|   |
|   |-- signals/                       # Numba JIT compiled
|   |   |-- generator.py              # 4-state machine signal generation
|   |   |-- filters.py                # Confidence, time stop, window filters
|   |
|   |-- sizing/
|   |   |-- position.py               # Dollar-neutral beta-weighted sizing
|   |
|   |-- backtest/
|   |   |-- engine.py                  # 3 modes: bar-by-bar, vectorized, grid-optimized
|   |   |-- performance.py            # PerformanceMetrics (Sharpe, MaxDD, etc.)
|   |
|   |-- utils/
|       |-- constants.py              # Enums: Instrument, HedgeMethod
|       |-- time_utils.py             # SessionConfig, session/window filtering
|
|   |-- validation/                   # Phase 13+ validation framework
|       |-- cpcv.py                   # CPCV(10,2): 45 combinatorial paths, purge zones
|       |-- gates.py                  # Binary gates: ADF/Hurst/Corr, apply_gate_filter_numba
|       |-- deflated_sharpe.py        # DSR: correct for multiple testing bias
|       |-- neighborhood.py           # L1 parameter robustness check
|       |-- propfirm.py              # $150K account metrics, daily loss, trailing DD
|
|-- tests/
|   |-- conftest.py                    # Shared fixtures
|   |-- test_backtest/                 # engine + performance tests
|   |-- test_hedge/                    # OLS + Kalman tests
|   |-- test_signals/                  # Signal generation + filter tests
|   |-- test_validation/              # CPCV + gates + DSR + neighborhood tests
|
|-- scripts/                           # ~32 active scripts
|   |-- archive/                       # 18 superseded scripts (Phases 6-13a)
|
|-- sierra/                            # Phase 2 ACSIL C++
|   |-- NQ_YM_SpreadMeanReversion_v1.0.cpp  # Production indicator (~2150 lines)
|
|-- output/
|   |-- cache/                         # Parquet cache for aligned pair data
```

---

## 2. Data Flow Pipeline

```
raw/*.txt (Sierra CSV 1min)
    |
    |  loader.py
    v
[BarData]  columns: open, high, low, close, volume
    |
    |  cleaner.py  (session filter, buffer, gap fill)
    v
[Clean BarData]
    |
    |  resampler.py  (1min -> 5min)
    v
[5min BarData]
    |
    |  alignment.py  (inner join on timestamp)
    |  cache.py  (Parquet I/O)
    v
[AlignedPair]  columns: close_a, close_b
    |
    |  hedge/factory.py -> ols_rolling.py or kalman.py
    v
[HedgeResult]  beta, spread, zscore
    |
    |  stats/ + metrics/dashboard.py
    v
[Metrics DataFrame]  adf_stat, hurst, half_life, correlation
    |
    |  signals/generator.py  (4-state machine, numba JIT)
    |  signals/filters.py  (confidence scoring, time windows)
    v
[Signal Series]  {+1, 0, -1}
    |
    |  backtest/engine.py  (sizing, slippage, dollar stop)
    v
[Trade Results + Equity Curve]
    |
    |  backtest/performance.py
    v
[PerformanceMetrics]
```

Dependencies flow strictly downward. Config YAML loaded at script level, injected as frozen dataclasses.

---

## 3. Key Design Patterns

- **ABC + Factory** (`hedge/`): `HedgeRatioEstimator` ABC -> `create_estimator()` factory
- **Frozen dataclasses**: All configs immutable (`SessionConfig`, `SignalConfig`, `BacktestConfig`, etc.)
- **Numba JIT**: Signal generator + filters compiled for 451x speedup
- **Three backtest modes**: bar-by-bar (reference), vectorized (equity curve), grid (stats only)
- **Shared helpers**: `_detect_and_pair_trades()`, `_compute_trades()`, `_compute_summary_stats()`

---

## 4. Phase 2 C++ (Sierra Charts)

Single file `NQ_YM_SpreadMeanReversion_v1.0.cpp` (~2150 lines) containing:
- OLS + Kalman hedge ratio computation
- Statistical metrics (ADF, Hurst, HalfLife, Correlation)
- Confidence scoring
- 4-state signal machine
- Semi-auto trading (BUY/SELL/FLATTEN buttons, auto-entry, auto-exits)
- Position sync with broker
- 3-panel textbox dashboard

Build: `F:\SierreChart_Spread_Indices\ACS_Source\VisualCCompile.Bat`
Signal parity Python/C++: 99.9%
