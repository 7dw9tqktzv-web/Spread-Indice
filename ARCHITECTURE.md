# Spread Trading Indicator -- Architecture Document

**Version**: 1.0
**Date**: 2026-02-19
**Scope**: Phase 1 (Python backtest) + Phase 2 (Sierra Charts ACSIL C++)

---

## 1. Directory Structure

```
Spread_Indice/
|
|-- CLAUDE.md                          # Project rules and conventions
|-- ARCHITECTURE.md                    # This document
|-- requirements.txt                   # Python dependencies (pinned)
|-- pyproject.toml                     # Project metadata
|-- venv/                              # Virtual environment
|
|-- raw/                               # Raw Sierra Charts CSV exports (1min)
|   |-- ESH26_FUT_CME_1mn.scid_BarData.txt
|   |-- NQH26_FUT_CME_1mn.scid_BarData.txt
|   |-- RTYH26_FUT_CME_1mn.scid_BarData.txt
|   |-- YMH26_FUT_CME_1mn.scid_BarData.txt
|
|-- Data/                              # Reference PDFs
|
|-- doc_sierra/                        # Sierra Charts ACSIL references
|
|-- config/
|   |-- instruments.yaml               # Tick size, multiplier, margin, costs
|   |-- pairs.yaml                     # Pair definitions and default params
|   |-- backtest.yaml                  # Backtest parameters (windows, thresholds)
|   |-- optimisation.yaml              # Grid/optuna search spaces
|
|-- src/
|   |-- __init__.py
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- loader.py                  # CSV parsing, dtype enforcement
|   |   |-- cleaner.py                 # Session filtering, gap handling, outliers
|   |   |-- resampler.py              # 1min -> 5min OHLCV aggregation
|   |   |-- alignment.py              # Multi-instrument time alignment
|   |   |-- cache.py                   # Parquet read/write cache layer
|   |
|   |-- hedge/
|   |   |-- __init__.py
|   |   |-- base.py                    # Abstract HedgeRatioEstimator
|   |   |-- ols_rolling.py            # Rolling OLS beta
|   |   |-- kalman.py                  # Kalman filter beta (filterpy)
|   |   |-- dollar_neutral.py         # Notional-based ratio
|   |   |-- volatility_neutral.py     # Realized vol ratio
|   |   |-- factory.py                 # HedgeRatioFactory (string -> instance)
|   |
|   |-- spread/
|   |   |-- __init__.py
|   |   |-- builder.py                 # Spread = leg1 - ratio * leg2
|   |   |-- pair.py                    # SpreadPair dataclass (pair metadata)
|   |
|   |-- metrics/
|   |   |-- __init__.py
|   |   |-- zscore.py                  # Rolling z-score
|   |   |-- adf.py                     # ADF test (statsmodels)
|   |   |-- hurst.py                   # Hurst exponent (R/S method)
|   |   |-- halflife.py               # OU half-life via AR(1) regression
|   |   |-- correlation.py            # Rolling Pearson correlation
|   |   |-- kalman_filter.py          # Kalman-filtered spread level
|   |   |-- dashboard.py              # Aggregate all metrics into DataFrame
|   |
|   |-- signals/
|   |   |-- __init__.py
|   |   |-- generator.py              # Z-score threshold crossing logic
|   |   |-- filters.py                # ADF/Hurst/half-life regime filters
|   |   |-- sizing.py                 # Contract sizing from hedge ratio
|   |
|   |-- backtest/
|   |   |-- __init__.py
|   |   |-- engine.py                  # Event-driven backtest loop
|   |   |-- portfolio.py              # Position tracking, PnL, margin
|   |   |-- costs.py                   # Transaction cost model (commission + slippage)
|   |   |-- performance.py            # Sharpe, Calmar, Hit Ratio, Profit Factor, drawdown
|   |   |-- report.py                 # HTML/CSV report generation
|   |
|   |-- optimisation/
|   |   |-- __init__.py
|   |   |-- grid.py                    # Brute-force grid search
|   |   |-- optuna_search.py          # Optuna-based Bayesian optimisation
|   |   |-- objective.py              # Objective function (Sharpe or Calmar)
|   |   |-- walk_forward.py           # Walk-forward analysis with anchored/rolling windows
|   |
|   |-- utils/
|       |-- __init__.py
|       |-- time_utils.py             # Chicago time helpers, session boundaries
|       |-- logging_config.py         # Structured logging setup
|       |-- parallel.py               # joblib/multiprocessing wrappers
|       |-- constants.py              # Enums: HedgeMethod, Instrument, SessionTime
|
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_spread_analysis.ipynb
|   |-- 03_optimisation_results.ipynb
|
|-- tests/
|   |-- __init__.py
|   |-- conftest.py                    # Shared fixtures (sample data, pairs)
|   |-- test_data/
|   |   |-- test_loader.py
|   |   |-- test_cleaner.py
|   |   |-- test_resampler.py
|   |-- test_hedge/
|   |   |-- test_ols_rolling.py
|   |   |-- test_kalman.py
|   |   |-- test_dollar_neutral.py
|   |   |-- test_volatility_neutral.py
|   |-- test_metrics/
|   |   |-- test_zscore.py
|   |   |-- test_hurst.py
|   |   |-- test_halflife.py
|   |-- test_backtest/
|   |   |-- test_engine.py
|   |   |-- test_performance.py
|   |-- test_integration/
|       |-- test_full_pipeline.py      # End-to-end: CSV -> backtest results
|
|-- scripts/
|   |-- run_backtest.py                # CLI entry point for single pair
|   |-- run_optimisation.py           # CLI entry point for optimisation
|   |-- run_all_pairs.py              # Parallel run across 6 pairs x 4 methods
|   |-- export_params.py             # Export optimised params to C++ header/JSON
|   |-- generate_report.py           # Comparative report across all configurations
|
|-- output/
|   |-- cache/                         # Parquet cache for cleaned/resampled data
|   |-- results/                       # Backtest results per pair/method
|   |-- reports/                       # HTML/CSV comparative reports
|   |-- params/                        # Exported optimised parameters
|
|-- sierra/                            # Phase 2 -- ACSIL C++ indicator
|   |-- SpreadIndicator.cpp           # Main ACSIL study function
|   |-- SpreadIndicator.h             # Declarations
|   |-- HedgeRatio.h                  # Hedge ratio implementations (header-only)
|   |-- KalmanFilter.h               # Lightweight Kalman filter (no external deps)
|   |-- Metrics.h                     # Z-score, half-life, Hurst (online algorithms)
|   |-- Dashboard.h                   # Textbox rendering helpers
|   |-- Config.h                      # Optimised parameters (generated from Python)
|   |-- Makefile                       # Build instructions for Sierra DLL
```

---

## 2. Module Dependency Graph

```
scripts/run_backtest.py
  |
  v
config/*.yaml  --->  src/utils/constants.py
                          |
                          v
               src/data/loader.py
                    |
                    v
               src/data/cleaner.py  <---  src/utils/time_utils.py
                    |
                    v
               src/data/resampler.py
                    |
                    v
               src/data/alignment.py
                    |
                    +---> src/data/cache.py (Parquet I/O)
                    |
                    v
               src/hedge/factory.py
                    |
                    +--> src/hedge/ols_rolling.py
                    +--> src/hedge/kalman.py
                    +--> src/hedge/dollar_neutral.py
                    +--> src/hedge/volatility_neutral.py
                    |
                    v
               src/spread/builder.py
                    |
                    v
               src/metrics/dashboard.py
                    |
                    +--> src/metrics/zscore.py
                    +--> src/metrics/adf.py
                    +--> src/metrics/hurst.py
                    +--> src/metrics/halflife.py
                    +--> src/metrics/correlation.py
                    +--> src/metrics/kalman_filter.py
                    |
                    v
               src/signals/generator.py
                    |
                    +--> src/signals/filters.py
                    +--> src/signals/sizing.py
                    |
                    v
               src/backtest/engine.py
                    |
                    +--> src/backtest/portfolio.py
                    +--> src/backtest/costs.py
                    |
                    v
               src/backtest/performance.py
                    |
                    v
               src/backtest/report.py  --->  output/
```

Key principle: dependencies flow strictly downward. No module imports from a layer above it. The `config/` YAML files are loaded once at the script level and injected into constructors as dataclasses -- modules never read config files directly.

---

## 3. Data Flow Pipeline

### 3.1 Phase 1: Backtest Pipeline

```
[Sierra Charts CSV]    raw/*.txt
        |
        |  loader.py -- parse Date+Time, enforce float64, set DatetimeIndex (CT)
        v
[Raw 1min DataFrame]   columns: open, high, low, close, volume, num_trades, bid_vol, ask_vol
        |
        |  cleaner.py -- filter 18:00-15:30 CT session, drop 30min start/end buffer,
        |                remove zero-volume bars, forward-fill small gaps (<3 bars)
        v
[Clean 1min DataFrame]
        |
        |  resampler.py -- resample to 5min OHLCV (standard aggregation rules)
        v
[5min DataFrame]
        |
        |  alignment.py -- inner-join on timestamp across instrument pair,
        |                  verify no NaN, cache to Parquet
        v
[Aligned Pair DataFrame]  columns: close_A, close_B, volume_A, volume_B
        |
        |  hedge/factory.py -- compute hedge ratio time series (one method at a time)
        v
[Hedge Ratio Series]  beta_t for each bar
        |
        |  spread/builder.py -- spread_t = close_A - beta_t * close_B
        v
[Spread Series]
        |
        |  metrics/dashboard.py -- compute all metrics in parallel
        v
[Metrics DataFrame]  z_score, adf_pvalue, hurst, half_life, correlation, kalman_spread
        |
        |  signals/generator.py -- threshold crossing + filters
        v
[Signal Series]  {+1, 0, -1} per bar
        |
        |  backtest/engine.py -- simulate fills with costs
        v
[Trade Log + Equity Curve]
        |
        |  backtest/performance.py -- compute all ratios
        v
[Performance Summary]  Sharpe, Calmar, Hit Ratio, Profit Factor, Max DD, etc.
```

### 3.2 Phase 2: Real-Time Pipeline (Sierra Charts)

```
[Sierra Charts tick data]
        |
        |  ACSIL sc.BaseData[]  -- 5min bars for both legs
        v
[Live bar close for leg A and leg B]
        |
        |  HedgeRatio.h -- rolling OLS or Kalman (online update)
        v
[Current hedge ratio]
        |
        |  spread = close_A - ratio * close_B
        v
[Current spread value]
        |
        |  Metrics.h -- online z-score, half-life, Hurst (over rolling window)
        v
[Current metric values]
        |
        |  Signal logic -- compare z-score to thresholds, apply regime filter
        v
[Buy/Sell/Flat signal + contract sizing]
        |
        |  Dashboard.h -- render to sc.AddAndManageSingleTextDrawingForStudy()
        v
[Textbox overlay on chart]
```

---

## 4. Class and Module Design

### 4.1 Data Layer

```python
# src/data/loader.py
@dataclass
class BarData:
    """Immutable container for a single instrument's bar data."""
    instrument: Instrument          # Enum: NQ, ES, RTY, YM
    timeframe: str                  # "1min" or "5min"
    df: pd.DataFrame                # DatetimeIndex (CT), OHLCV columns

def load_sierra_csv(path: Path, instrument: Instrument) -> BarData: ...
```

```python
# src/data/cleaner.py
@dataclass
class SessionConfig:
    session_start: time             # 17:30 CT
    session_end: time               # 15:30 CT
    buffer_minutes: int             # 30
    trading_start: time             # 04:00 CT
    trading_end: time               # 14:00 CT

def clean(data: BarData, session: SessionConfig) -> BarData: ...
```

```python
# src/data/alignment.py
@dataclass
class AlignedPair:
    pair: SpreadPair
    df: pd.DataFrame                # close_A, close_B, volume_A, volume_B
    timeframe: str

def align_pair(a: BarData, b: BarData, pair: SpreadPair) -> AlignedPair: ...
```

### 4.2 Hedge Ratio Layer

```python
# src/hedge/base.py
class HedgeRatioEstimator(ABC):
    """All hedge ratio methods implement this interface."""

    @abstractmethod
    def fit(self, close_a: pd.Series, close_b: pd.Series) -> pd.Series:
        """Return a Series of hedge ratios indexed by timestamp."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

```python
# src/hedge/ols_rolling.py
@dataclass
class OLSRollingConfig:
    window: int = 120               # bars (5min)
    min_periods: int = 60

class OLSRollingEstimator(HedgeRatioEstimator):
    def __init__(self, config: OLSRollingConfig): ...
    def fit(self, close_a, close_b) -> pd.Series: ...
```

```python
# src/hedge/kalman.py
@dataclass
class KalmanConfig:
    delta: float = 1e-4             # State transition covariance
    ve: float = 1e-1                # Observation noise

class KalmanEstimator(HedgeRatioEstimator):
    def __init__(self, config: KalmanConfig): ...
    def fit(self, close_a, close_b) -> pd.Series: ...
```

```python
# src/hedge/factory.py
def create_estimator(method: HedgeMethod, **kwargs) -> HedgeRatioEstimator: ...
```

### 4.3 Metrics Layer

```python
# src/metrics/dashboard.py
@dataclass
class MetricsConfig:
    zscore_window: int = 120
    adf_window: int = 480
    hurst_window: int = 480
    halflife_window: int = 240
    correlation_window: int = 120

@dataclass
class MetricsSnapshot:
    """Point-in-time values of all metrics for a single bar."""
    zscore: float
    adf_pvalue: float
    hurst: float
    half_life: float
    correlation: float
    kalman_spread: float

def compute_all_metrics(
    spread: pd.Series,
    close_a: pd.Series,
    close_b: pd.Series,
    config: MetricsConfig,
) -> pd.DataFrame:
    """Returns DataFrame with one column per metric, indexed by timestamp."""
    ...
```

### 4.4 Signal Layer

```python
# src/signals/generator.py
@dataclass
class SignalConfig:
    z_entry: float = 2.0            # Enter when |z| > z_entry
    z_exit: float = 0.5             # Exit when |z| < z_exit
    z_stop: float = 4.0             # Stop loss threshold
    min_hurst: float = 0.0          # Hurst must be below this (0.5 typical)
    max_hurst: float = 0.45
    max_adf_pvalue: float = 0.05
    max_half_life: int = 120        # bars
    min_half_life: int = 5

class SignalGenerator:
    def __init__(self, config: SignalConfig): ...
    def generate(self, metrics: pd.DataFrame) -> pd.Series: ...
    # Returns: +1 (long spread), -1 (short spread), 0 (flat)
```

### 4.5 Backtest Layer

```python
# src/backtest/engine.py
@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_per_contract: float = 2.50    # per side
    slippage_ticks: int = 1

class BacktestEngine:
    def __init__(self, config: BacktestConfig, cost_model: CostModel): ...

    def run(
        self,
        aligned: AlignedPair,
        signals: pd.Series,
        hedge_ratios: pd.Series,
    ) -> BacktestResult: ...

@dataclass
class BacktestResult:
    trades: pd.DataFrame            # entry_time, exit_time, pnl, duration, ...
    equity_curve: pd.Series
    daily_returns: pd.Series
```

```python
# src/backtest/performance.py
@dataclass
class PerformanceMetrics:
    sharpe: float
    calmar: float
    hit_ratio: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_duration: int      # bars
    total_trades: int
    avg_trade_pnl: float
    avg_trade_duration: float       # bars
    annual_return: float

def compute_performance(result: BacktestResult) -> PerformanceMetrics: ...
```

### 4.6 Optimisation Layer

```python
# src/optimisation/walk_forward.py
@dataclass
class WalkForwardConfig:
    train_days: int = 252           # 1 year
    test_days: int = 63             # 1 quarter
    step_days: int = 63             # roll forward 1 quarter
    anchored: bool = False          # anchored vs rolling

class WalkForwardAnalyser:
    def __init__(self, config: WalkForwardConfig): ...

    def run(
        self,
        aligned: AlignedPair,
        hedge_method: HedgeMethod,
        objective: Callable,        # optimisation objective
    ) -> list[WalkForwardFold]: ...
```

```python
# src/optimisation/objective.py
def sharpe_objective(
    params: dict,
    aligned: AlignedPair,
    hedge_method: HedgeMethod,
) -> float:
    """Negative Sharpe (for minimisation). Single evaluation."""
    ...
```

---

## 5. Phase 1 Python Workflow -- Execution Order

### Step 1: Data Preparation (run once, cached)

```
scripts/run_backtest.py  --prepare-data

1. Load all 4 raw CSVs via loader.py
2. Clean each via cleaner.py (session filter, gap handling)
3. Resample each to 5min via resampler.py
4. For each of the 6 pairs, align via alignment.py
5. Cache all AlignedPair DataFrames to output/cache/ as Parquet
```

### Step 2: Single Pair Backtest (development/debugging)

```
scripts/run_backtest.py  --pair NQ_ES  --method ols_rolling

1. Load cached AlignedPair for NQ/ES
2. Compute hedge ratios (OLS Rolling)
3. Build spread series
4. Compute all metrics
5. Generate signals
6. Run backtest
7. Compute performance
8. Print summary, save to output/results/
```

### Step 3: Optimisation (per pair, per method)

```
scripts/run_optimisation.py  --pair NQ_ES  --method ols_rolling  --engine optuna

1. Load cached AlignedPair
2. Define search space from config/optimisation.yaml:
   - zscore_window: [60, 240]
   - z_entry: [1.5, 3.0]
   - z_exit: [0.0, 1.0]
   - hedge window: [60, 480]
   - hurst/adf filters: on/off and thresholds
3. Run walk-forward optimisation:
   - For each fold: optimise on train, evaluate on test
   - Objective: maximise Sharpe (or Calmar)
4. Aggregate out-of-sample results across folds
5. Save best parameters to output/params/
```

### Step 4: Full Parallel Run (all pairs x all methods)

```
scripts/run_all_pairs.py  --workers 8

1. Generate 24 tasks: 6 pairs x 4 hedge methods
2. Dispatch via joblib Parallel (or multiprocessing.Pool)
3. Each task: load cache -> optimise -> backtest -> save
4. Collect all 24 PerformanceMetrics
5. Generate comparative report
```

### Step 5: Report Generation

```
scripts/generate_report.py

1. Load all 24 result files from output/results/
2. Build comparison table: rows = pairs, columns = methods, cells = Sharpe/Calmar
3. Rank configurations
4. Generate HTML report with equity curves and heatmaps
5. Save to output/reports/
```

### Step 6: Parameter Export

```
scripts/export_params.py  --format cpp

1. Load best parameters per pair from output/params/
2. Generate sierra/Config.h with constexpr values
3. Generate JSON sidecar for validation
```

---

## 6. Phase 2 C++ Design -- ACSIL Integration

### 6.1 Study Function Structure

```cpp
// SpreadIndicator.cpp
#include "SierraChartStudyInterfaceDefinitions.h"
#include "HedgeRatio.h"
#include "Metrics.h"
#include "Dashboard.h"
#include "Config.h"

SCSFExport scsf_SpreadIndicator(SCStudyInterfaceRef sc) {
    if (sc.SetDefaults) {
        sc.GraphName = "Spread Trading Indicator";
        sc.AutoLoop = 0;  // Manual loop for multi-chart access
        sc.GraphRegion = 0;

        // Subgraphs
        sc.Subgraph[0].Name = "Spread";
        sc.Subgraph[1].Name = "Z-Score";
        sc.Subgraph[2].Name = "Hedge Ratio";
        sc.Subgraph[3].Name = "Signal";

        // Inputs (from optimised parameters)
        sc.Input[0].Name  = "Hedge Method";        // 0=OLS, 1=Kalman, 2=Dollar, 3=Vol
        sc.Input[1].Name  = "Z-Score Window";
        sc.Input[2].Name  = "Z Entry Threshold";
        sc.Input[3].Name  = "Z Exit Threshold";
        sc.Input[4].Name  = "OLS Window";
        sc.Input[5].Name  = "Kalman Delta";
        sc.Input[6].Name  = "Chart Number Leg 2";
        // ... additional inputs
        return;
    }

    // Access second leg data from another chart
    SCGraphData leg2_data;
    sc.GetChartBaseData(sc.Input[6].GetChartNumber(), leg2_data);

    // Main computation loop (process new bars only)
    for (int i = sc.UpdateStartIndex; i < sc.ArraySize; i++) {
        float close_a = sc.BaseData[SC_LAST][i];
        float close_b = leg2_data[SC_LAST][i];

        // Update hedge ratio (online)
        float ratio = update_hedge_ratio(close_a, close_b, sc.Input[0].GetInt());

        // Build spread
        float spread = close_a - ratio * close_b;

        // Update z-score (online rolling mean/std)
        float zscore = update_zscore(spread, sc.Input[1].GetInt());

        // Signal logic
        int signal = compute_signal(zscore,
                                     sc.Input[2].GetFloat(),
                                     sc.Input[3].GetFloat());

        // Assign to subgraphs
        sc.Subgraph[0][i] = spread;
        sc.Subgraph[1][i] = zscore;
        sc.Subgraph[2][i] = ratio;
        sc.Subgraph[3][i] = (float)signal;
    }

    // Update dashboard textbox on last bar
    update_dashboard(sc);
}
```

### 6.2 Online Algorithms (no lookback recomputation)

All C++ implementations use incremental/online algorithms to avoid recomputing over the full window on each bar:

- **Rolling Mean/Std (Z-score)**: Welford's online algorithm with a circular buffer.
- **Rolling OLS**: Incremental least squares using running sums (sum_x, sum_y, sum_xy, sum_xx).
- **Kalman Filter**: Naturally online -- single predict/update step per bar.
- **Dollar Neutral**: Trivial ratio of current prices times multipliers.
- **Volatility Neutral**: Incremental variance tracker with circular buffer.
- **Hurst**: Computed periodically (every N bars) rather than every bar to reduce cost.
- **Half-Life**: AR(1) regression over rolling window, recomputed every N bars.

### 6.3 Dashboard Textbox Layout

```
+----------------------------------------------+
| SPREAD: NQ/ES          Method: OLS Rolling   |
|----------------------------------------------|
| Hedge Ratio:    1.847     Z-Score:   -2.13   |
| Half-Life:      47 bars   Hurst:      0.38   |
| ADF p-value:    0.003     Correl:     0.94   |
|----------------------------------------------|
| Signal: SHORT SPREAD     Size: 1x2           |
| Entry Z: -2.00  Exit Z: -0.50  Stop Z: 4.00 |
+----------------------------------------------+
```

Rendered via `sc.AddAndManageSingleTextDrawingForStudy()` with formatted string, updated on every bar close.

### 6.4 Persistent State Across Bars

Use Sierra Charts persistent variables (`sc.GetPersistentInt()`, `sc.GetPersistentFloat()`, `sc.GetPersistentPointer()`) to maintain:
- Circular buffer arrays for rolling computations
- Kalman filter state (mean, covariance)
- Current position state (long/short/flat)
- Running sums for OLS regression

---

## 7. Configuration Management

### 7.1 Configuration Flow

```
config/instruments.yaml          # Static instrument properties
        |
        v
config/pairs.yaml                # Pair definitions
        |
        v
config/backtest.yaml             # Default parameters
        |
        |  (Phase 1: loaded by scripts, injected as dataclasses)
        v
[Python optimisation]
        |
        v
output/params/NQ_ES_ols.json    # Optimised parameters per pair/method
        |
        |  scripts/export_params.py
        v
sierra/Config.h                  # constexpr values for C++
        |
        v
[Sierra Charts Inputs]           # User can override via chart UI
```

### 7.2 instruments.yaml

```yaml
NQ:
  exchange: CME
  multiplier: 20.0
  tick_size: 0.25
  tick_value: 5.0
  commission: 2.50        # per side per contract
  margin: 18000.0

ES:
  exchange: CME
  multiplier: 50.0
  tick_size: 0.25
  tick_value: 12.50
  commission: 2.50
  margin: 13000.0

RTY:
  exchange: CME
  multiplier: 50.0
  tick_size: 0.10
  tick_value: 5.0
  commission: 2.50
  margin: 7500.0

YM:
  exchange: CME
  multiplier: 5.0
  tick_size: 1.0
  tick_value: 5.0
  commission: 2.50
  margin: 9500.0
```

### 7.3 Parameter Dataclass (single source of truth)

```python
# All tunable parameters in one place
@dataclass
class SpreadParams:
    # Hedge ratio
    hedge_method: HedgeMethod
    ols_window: int = 120
    kalman_delta: float = 1e-4
    kalman_ve: float = 1e-1
    vol_window: int = 60

    # Z-score
    zscore_window: int = 120
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0

    # Regime filters
    adf_window: int = 480
    max_adf_pvalue: float = 0.05
    hurst_window: int = 480
    max_hurst: float = 0.45
    halflife_window: int = 240
    min_half_life: int = 5
    max_half_life: int = 120
    correlation_window: int = 120

    def to_dict(self) -> dict: ...
    def to_json(self, path: Path) -> None: ...

    @classmethod
    def from_json(cls, path: Path) -> "SpreadParams": ...
```

### 7.4 Generated C++ Config

```cpp
// sierra/Config.h  (auto-generated -- do not edit manually)
#pragma once

namespace SpreadConfig {
    // NQ/ES -- OLS Rolling (optimised 2026-02-19)
    namespace NQ_ES_OLS {
        constexpr int    OLS_WINDOW       = 144;
        constexpr int    ZSCORE_WINDOW    = 96;
        constexpr float  Z_ENTRY          = 2.15f;
        constexpr float  Z_EXIT           = 0.40f;
        constexpr float  Z_STOP           = 3.80f;
        constexpr float  MAX_HURST        = 0.42f;
        constexpr float  MAX_ADF_PVALUE   = 0.05f;
        constexpr int    MAX_HALF_LIFE    = 110;
    }
    // ... other pairs and methods
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

| Module | What to test | Method |
|---|---|---|
| `loader.py` | Column names, dtypes, timezone, empty file handling | Known CSV snippet |
| `cleaner.py` | Session boundaries exact to the minute, gap fill logic | Synthetic 1min data with known gaps |
| `resampler.py` | OHLCV aggregation correctness (high=max, low=min, close=last, volume=sum) | 10 bars of 1min -> 2 bars of 5min |
| `alignment.py` | Inner join correctness, no NaN in output | Two series with deliberate mismatches |
| `ols_rolling.py` | Known regression output on synthetic linear data | y = 2x + noise, beta should be ~2.0 |
| `kalman.py` | Convergence to true beta on synthetic data | y = 1.5x, verify ratio converges |
| `zscore.py` | Mean 0, std 1 on normally distributed input | Random normal, z should be ~N(0,1) |
| `hurst.py` | H~0.5 for random walk, H~0 for mean-reverting | Generated GBM vs OU process |
| `halflife.py` | Known OU process half-life recovery | Simulate OU with theta=0.05, verify HL |
| `generator.py` | Signal transitions at exact threshold crossings | Synthetic z-score series |
| `engine.py` | PnL matches hand-calculated example | 3-trade scenario with known prices/costs |
| `performance.py` | Sharpe/Calmar match manual calculation | Known equity curve |

### 8.2 Integration Tests

```python
# tests/test_integration/test_full_pipeline.py
def test_full_pipeline_nq_es():
    """End-to-end: load real data subset -> backtest -> verify non-degenerate results."""
    # Use first 1000 5min bars of NQ/ES
    # Verify: trades > 0, Sharpe is finite, equity curve length matches
    ...

def test_all_hedge_methods_produce_valid_ratios():
    """All 4 methods return finite, non-zero ratios on real data."""
    ...

def test_walk_forward_fold_count():
    """Verify correct number of folds for given train/test/step config."""
    ...
```

### 8.3 Validation Tests (Python vs C++ parity)

```python
# tests/test_integration/test_parity.py
def test_ols_rolling_parity():
    """Compare Python OLS rolling output with C++ implementation on same input."""
    # Export aligned data to CSV
    # Run C++ test binary (compiled separately)
    # Load C++ output CSV
    # Assert max absolute difference < 1e-6
    ...
```

### 8.4 Running Tests

```bash
# All unit tests
python -m pytest tests/ -v --tb=short

# Only fast unit tests (exclude integration)
python -m pytest tests/ -v --ignore=tests/test_integration

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Appendix A: Key Design Decisions

1. **Dataclasses over dicts**: All configuration and results use typed dataclasses. This provides IDE autocompletion, catches typos at construction time, and serves as documentation.

2. **Factory pattern for hedge ratios**: The `HedgeRatioFactory` maps a string/enum to a concrete estimator. This makes it trivial to loop over all methods in scripts and to add new methods later.

3. **Cache layer with Parquet**: Data preparation (load, clean, resample, align) is expensive. The cache layer writes Parquet files keyed by `(pair, timeframe, hash_of_session_config)`. Subsequent runs skip all data prep.

4. **Manual loop in ACSIL (`AutoLoop = 0`)**: Required for multi-chart data access (`sc.GetChartBaseData`). The manual loop processes only new bars (`sc.UpdateStartIndex` to `sc.ArraySize`), keeping CPU usage minimal.

5. **Online algorithms in C++**: No STL containers or dynamic allocation in the hot path. Fixed-size circular buffers allocated once via persistent pointers. This meets Sierra Charts' real-time performance requirements.

6. **Walk-forward over simple train/test split**: Prevents overfitting to a single period. Out-of-sample performance across multiple folds gives a realistic estimate of live performance.

7. **Separation of signal generation and regime filtering**: Signals (z-score crossings) and filters (ADF, Hurst, half-life) are independent modules. Filters can be toggled on/off during optimisation without changing signal logic.

## Appendix B: Python Dependencies

```
pandas>=2.0
numpy>=1.24
scipy>=1.11
statsmodels>=0.14
filterpy>=1.4
scikit-learn>=1.3
optuna>=3.4
joblib>=1.3
pyyaml>=6.0
pyarrow>=14.0
matplotlib>=3.8
jinja2>=3.1
pytest>=7.4
pytest-cov>=4.1
```
