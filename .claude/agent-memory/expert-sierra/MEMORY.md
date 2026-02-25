# Expert Sierra -- Persistent Memory

## Phase 2 Status

### Phase 2a VALIDATED: Visual Indicator
- OLS Config E + Kalman K_Balanced textbox overlay
- 24 subgraphs, 22 inputs, 7 utility functions
- Kalman filter with Joseph form, gap detection, persistent double state
- Confidence scoring with Python weights (40/25/20/15) + ADF gate
- Hurst variance-ratio (NOT R/S), Half-life via Cov/Var
- Dynamic per-bar coloring + 3-panel textbox (Signal/Dashboard/Trading)
- **Parity VALIDATED** : signal agreement 99.9% vs Python on 10k bars

### Phase 2b VALIDATED: Semi-Auto Trading (~2150 lines)
- **Control Bar Buttons** : BUY SP / SELL SP / FLAT SP via ACS_BUTTON_1/2/3
- **Auto-Entry** : OLS signal detection on every bar (works with fast replay 960X)
- **Auto-Exits** : z-exit, dollar stop, time stop. All configurable via Inputs 25-30
- **Position Sync** : real broker positions via GetTradePositionForSymbolAndAccount(). Desync auto-corrected
- **Scaling** : same-direction adds with 10s cooldown. Opposite direction blocked (FLATTEN first)
- **P&L Live** : manual calc (AveragePrice x dollarPerPoint). OpenProfitLoss = account total (unusable)
- **One-Legged Protection** : if one leg fails, close surviving leg immediately
- **Architecture** : sc.BuyOrder()/sc.SellOrder() with .Symbol (cross-symbol). Deferred order pattern
- **Inputs 22-31** : symbols, quantities, z-exit, dollar stop, time stop, auto-exit toggle, trading action dropdown, auto-entry toggle
- **PersistentInt 4-7** : TradingPosition, EntryBarIndex, PendingOrderAction, AutoEntryLastBarIndex
- **PersistentDouble 8,10** : EntrySpreadZ, LastOrderTime (cooldown). PersistentDouble(9) freed (EntryTotalPnL removed)
- **Tested in replay** : auto-entry + auto-exit validated. Manual buttons validated on Teton Sim1
- **Code review fixes (v2.1)** : buttons re-enabled on full recalc, float guards fabs()<1e-12, EntryTotalPnL removed

### Phase 2 TODO
1. Daily regime indicator (detect 2023-type correlation breakdown)
2. NQ_RTY indicator (same architecture, different configs)
3. Live simulation test (full trading day)

### Key Learnings from Phase 2a (C++ Development & Validation)

#### Bugs Fixed (Critical for Future Indicators)
1. **Kalman MUST use double precision** : Q = alpha_ratio * R = 3e-7 * 1e-6 = 3e-13 → lost in float32. Use `sc.GetPersistentDouble()` for all Kalman state.
2. **Kalman H vector MUST be centered** : H=[1, log_ym] with log_ym~10.69 → P near-singular after 1st update (det~3e-6). Fix: H=[1, log_ym - center] where center = mean(log_ym) over init bars.
3. **Kalman P init = I** (identity), NOT R*I. Python uses P=eye(2). R*I with R~1e-6 makes filter over-confident.
4. **Kalman needs to start early** : With alpha_ratio=3e-7 (memory ~3.3M bars), starting at bar 3299 gives insufficient convergence. Start at bar 999 with separate R estimation on first 1000 bars.
5. **F > 1e-12 guard** required before Kalman gain division (prevents NaN).
6. **Pre-OLS bars** : return early with subgraphs=0, don't compute fake spread with beta=1.

#### Validation Results
- **Metrics with r > 0.95** : Spread (0.996), Correlation (0.997), Z-Score (0.974), Beta (0.971), Hurst (0.962), StdDev (0.954)
- **Metrics with r < 0.80** : ADF (0.533), Half-Life (0.479) — 12-bar windows amplify tiny data conditioning differences
- **Signal parity** : 99.9% agreement after all filters. 100% direction agreement on combined entry conditions.
- **Root cause of ADF/HL divergence** : Python clean/resample pipeline vs Sierra internal data handling. Not algorithmic.

#### General Learnings
- **FLAT exits = 86% of 2023 losses**: spread diverges, never reverts, held until force-close at 15:30 CT.
- **Confidence scoring blind to daily regime changes**: bar-by-bar metrics miss macro-level regime shifts.
- **R adaptatif EWMA invalidated**: never use r_ewma_span > 0. Keep R fixed.
- **P_trace is a pure temporal proxy**: correlates with time-since-last-gap, not signal quality. Never use as filter.

#### Compilation
- Batch file method: `cmd.exe //c "$(cygpath -w /tmp/compile_nqym.bat)"`
- Output: `F:\SierreChart_Spread_Indices\Data\NQ_YM_SpreadMeanReversion_64.dll`
- Build env: VS 2022 Build Tools, 64-bit

---

## Target Configs

### OLS Config E (Primary -- Signal Generator)

| Parameter | Value |
|-----------|-------|
| Pair | NQ_YM (5min) |
| OLS lookback | 3300 bars (~12.5j) |
| Z-score window | 30 bars (2h30) |
| z_entry | 3.15 |
| z_exit | 1.00 |
| z_stop | 4.50 |
| min_confidence | 67% |
| Profil metrics | tres_court (adf=12, hurst=64, hl=12, corr=6) |
| Entry window | 02:00-14:00 CT |
| Flat time | 15:30 CT |

### Kalman K_Balanced (Discretionary Textbox Overlay)

| Parameter | Value |
|-----------|-------|
| alpha_ratio | 3e-7 |
| z_entry | 1.3125 |
| z_exit | 0.375 |
| z_stop | 2.75 |
| min_confidence | 75% |
| Profil metrics | tres_court |
| Entry window | 03:00-12:00 CT |

Kalman role: displayed as textbox bias (beta direction, innovation z-score, suggested side). OLS remains the actual signal generator for entries/exits.

---

## ACSIL Core API Reference

### Study Function Pattern

```cpp
#include "sierrachart.h"
SCDLLName("MyDLL")
SCSFExport scsf_MyStudy(SCStudyInterfaceRef sc) {
    if (sc.SetDefaults) {
        sc.GraphName = "My Study";
        sc.AutoLoop = 1;  // 1=per-bar auto, 0=manual loop
        sc.GraphRegion = 0;  // 0=price, 1+=sub
        // Subgraphs, Inputs...
        return;  // MUST return
    }
    // Main logic -- sc.Index is current bar (AutoLoop=1)
}
```

### Data Access
- **OHLCV**: `sc.Open[i]`, `sc.High[i]`, `sc.Low[i]`, `sc.Close[i]`, `sc.Volume[i]`
- **Alternative**: `sc.BaseData[SC_LAST][i]` (SC_OPEN=0, SC_HIGH=1, SC_LOW=2, SC_LAST=3, SC_VOLUME=4, SC_NUM_TRADES=5, SC_OHLC_AVG, SC_HLC_AVG)
- **DateTime**: `sc.BaseDateTimeIn[i]` returns SCDateTime
  - `.GetTimeInSeconds()` -> int (0-86399)
  - `.GetDate()` -> int, `.GetHour()`, `.GetMinute()`, `.GetSecond()`
- **Array size**: `sc.ArraySize` (total bars), `sc.UpdateStartIndex` (first bar needing recalc)
- **Last bar**: `sc.ArraySize - 1`

### Subgraphs (up to 60)

```cpp
SCSubgraphRef Sub = sc.Subgraph[0];
Sub.Name = "Signal";
Sub.DrawStyle = DRAWSTYLE_LINE;  // LINE, POINT, ARROW_UP, ARROW_DOWN, BAR, DASH, HIDDEN, COLOR_BAR, STAIR
Sub.PrimaryColor = RGB(0, 200, 0);
Sub.SecondaryColor = RGB(200, 0, 0);
Sub.LineWidth = 2;
Sub.LineStyle = LINESTYLE_SOLID;  // SOLID, DASH, DOT, DASHDOT
```

- **Extra arrays** (12 per subgraph): `sc.Subgraph[i].Arrays[0..11][index]` -- intermediate storage
- **Per-bar coloring**: `Sub.DataColor[index] = RGB(r, g, b);`
- **Output value**: `Sub[sc.Index] = value;`

### Inputs (up to 64)

```cpp
SCInputRef In = sc.Input[0];
In.Name = "Period"; In.SetInt(20); In.SetIntLimits(1, 1000); int v = In.GetInt();
In.SetFloat(2.5f); In.SetFloatLimits(0.1f, 10.0f); float v = In.GetFloat();
In.SetYesNo(0); int v = In.GetYesNo();  // 0=No, 1=Yes
In.SetColor(RGB(255,0,0)); COLORREF v = In.GetColor();
In.SetTime(HMS_TIME(14,0,0)); int v = In.GetTime();  // seconds since midnight
In.SetCustomInputStrings("OLS;Kalman;Hybrid"); In.SetCustomInputIndex(0); int v = In.GetIndex();
```

### Persistent Variables (survive bar updates)

```cpp
int& State = sc.GetPersistentInt(0);     // id = 0..63
float& Level = sc.GetPersistentFloat(0);
double& Sum = sc.GetPersistentDouble(0);
SCString& Label = sc.GetPersistentSCString(0);

// Custom struct via pointer (id = 0..63)
struct MyData { double beta; double P[4]; };
MyData*& pData = (MyData*&)sc.GetPersistentPointer(0);
if (pData == NULL) {
    pData = (MyData*)sc.AllocateMemory(sizeof(MyData));
    memset(pData, 0, sizeof(MyData));
}
if (sc.LastCallToFunction) { sc.FreeMemory(pData); pData = NULL; }
```

### Multi-Chart Access

**REQUIRES** `sc.CalculationPrecedence = LOW_PREC_LEVEL;` in SetDefaults.

```cpp
SCGraphData OtherData;
sc.GetChartBaseData(ChartNumber, OtherData);  // OtherData[SC_LAST][i]
int OtherIndex = sc.GetContainingIndexForDateTimeIndex(ChartNumber, sc.Index);

SCFloatArray StudyData;
sc.GetStudyArraysFromChartUsingID(ChartNumber, StudyID, StudyData);
```

### Built-in Math Functions

```cpp
sc.SimpleMovAvg(In, Out, sc.Index, Period);
sc.ExponentialMovAvg(In, Out, sc.Index, Period);
sc.StdDev(In, Out, sc.Index, Period);
sc.ATR(sc.BaseData, Out, sc.Index, Period, MOVAVGTYPE_SIMPLE);
sc.RSI(In, Out, sc.Index, Period, MOVAVGTYPE_SIMPLE);
sc.BollingerBand(In, Mid, sc.Index, Period, Mult, MOVAVGTYPE_SIMPLE);
float hi = sc.GetHighest(sc.High, sc.Index, Period);
float lo = sc.GetLowest(sc.Low, sc.Index, Period);
int cross = sc.CrossOver(A, B, sc.Index);  // CROSS_FROM_BOTTOM(1), CROSS_FROM_TOP(-1), NO_CROSS(0)
float r = sc.RoundToTickSize(price, sc.TickSize);
```

### Trading Functions

```cpp
// SetDefaults for spread trading:
sc.AllowMultipleEntriesInSameDirection = 1;  // Required for scaling
sc.SupportReversals = 0;
sc.AllowOnlyOneTradePerBar = 0;  // Required for 2-leg spread (else 2nd leg ret=-8997)
sc.MaximumPositionAllowed = 10;
sc.SupportAttachedOrdersForTrading = 0;
sc.SendOrdersToTradeService = !sc.GlobalTradeSimulationIsOn;  // REQUIRED else orders silently ignored

// CROSS-SYMBOL orders (spread trading):
s_SCNewOrder Order;
Order.OrderQuantity = 2;
Order.Price1 = 0;  // REQUIRED for market orders
Order.OrderType = SCT_ORDERTYPE_MARKET;
Order.Symbol = "MNQH26_FUT_CME";  // Explicit symbol for cross-chart
Order.TextTag = "SpreadBuyNQ";
int ret = sc.BuyOrder(Order);  // or sc.SellOrder(Order)
// ret > 0 = success, ret = -1 = rejected, ret = -8998 = skipped (full recalc)

// SINGLE-SYMBOL orders (primary chart only):
int Result = sc.BuyEntry(Order);  // or sc.SellEntry(Order)
sc.BuyExit(Order);  sc.SellExit(Order);  sc.FlattenAndCancelAllOrders();

// Position data:
s_SCPositionData Pos;
sc.GetTradePosition(Pos);  // primary symbol
sc.GetTradePositionForSymbolAndAccount(Pos, "MNQH26_FUT_CME", sc.SelectedTradeAccount);  // any symbol
int qty = Pos.PositionQuantity;  // + long, - short, 0 flat
double avg = Pos.AveragePrice;   // fill price (for manual P&L calc)
// WARNING: Pos.OpenProfitLoss = account cumulative total in simulation, NOT position P&L
```

### Deferred Order Pattern (Spread Trading)

```cpp
// Problem: Changing Input triggers full recalc -> orders return -8998
// Solution: capture action in PersistentInt, execute on next normal tick

int& PendingOrderAction = sc.GetPersistentInt(6);  // 0=none, 1=buy, 2=sell, 3=flatten

// Step 1: Capture action (works during full recalc)
if (sc.MenuEventID == ACS_BUTTON_1) PendingOrderAction = 1;  // Button click
if (tradingAction != 0) PendingOrderAction = tradingAction;    // Dropdown input

// Step 2: Execute only when NOT in full recalc, on last bar
if (sc.Index == sc.ArraySize - 1 && !sc.IsFullRecalculation && PendingOrderAction != 0)
{
    // Submit orders here...
    PendingOrderAction = 0;  // Clear
}
```

### Manual P&L Calculation

```cpp
// OpenProfitLoss is UNUSABLE (returns account total in simulation)
// Manual calc using AveragePrice + dollar-per-point:
double dpp1 = (sym1[0] == 'M') ? 2.0 : 20.0;   // MNQ=$2/pt, NQ=$20/pt
double dpp2 = (sym2[0] == 'M') ? 0.5 : 5.0;     // MYM=$0.50/pt, YM=$5/pt
double pnl = (currentPrice1 - Pos1.AveragePrice) * Pos1.PositionQuantity * dpp1
           + (currentPrice2 - Pos2.AveragePrice) * Pos2.PositionQuantity * dpp2;
// Note: no API to get TickSize/CurrencyValuePerTick for secondary symbol
// s_SCSymbolData does NOT exist in ACSIL
```

### Control Bar Buttons

```cpp
// SetDefaults:
sc.ReceivePointerEvents = ACS_RECEIVE_POINTER_EVENTS_ALWAYS;
sc.UpdateAlways = 1;

// Setup (in full recalc init):
sc.SetCustomStudyControlBarButtonText(ACS_BUTTON_1, "BUY SP");
sc.SetCustomStudyControlBarButtonHoverText(ACS_BUTTON_1, "BUY SPREAD");

// Detection (MUST be on last bar):
if (sc.Index == sc.ArraySize - 1 && sc.MenuEventID >= ACS_BUTTON_1 && sc.MenuEventID <= ACS_BUTTON_3)
{
    sc.SetCustomStudyControlBarButtonEnable(sc.MenuEventID, 0);  // Visual feedback
    PendingOrderAction = sc.MenuEventID - ACS_BUTTON_1 + 1;
}
// User must manually add ACS buttons to Control Bar: Global Settings > Customize Control Bars
// Then show: Window > Control Bars > Control Bar N
```

### Drawing Tools (TextBox)

```cpp
s_UseTool T;
T.Clear();
T.ChartNumber = sc.ChartNumber;
T.DrawingType = DRAWING_TEXT;
T.AddMethod = UTAM_ADD_OR_ADJUST;
T.LineNumber = 1000;  // unique ID -- reuse to update
T.Region = 0;
T.UseRelativeVerticalValues = 1;
T.BeginValue = 90;  // Y position (%)
T.BeginDateTime = sc.BaseDateTimeIn[sc.ArraySize - 1];
T.Text = "Beta: 1.234";
T.FontSize = 12;
T.Color = RGB(255,255,255);
T.FontBackColor = RGB(0,0,128);
T.TextAlignment = DT_RIGHT;
T.TransparentLabelBackground = 0;
sc.UseTool(T);
```

### Debug and Logging

```cpp
sc.AddMessageToLog("Simple message", 0);  // 0=info, 1=error
SCString msg;
msg.Format("Index=%d Close=%.4f Beta=%.6f", sc.Index, sc.Close[sc.Index], beta);
sc.AddMessageToLog(msg, 0);
// SCString: .Format(), .Append(), .GetChars(), .GetLength(). NOT std::string.
```

---

## ACSIL Gotchas (Critical)

### General
1. **SetDefaults MUST return immediately** after configuration. Logic after `if (sc.SetDefaults) { ... return; }` runs on every bar.
2. **Out-of-bounds array access** does not crash -- returns garbage silently. Always bounds-check.
3. **NEVER use new/delete** -- only `sc.AllocateMemory()` / `sc.FreeMemory()`. Free in `if (sc.LastCallToFunction)`.
4. **Multi-chart needs LOW_PREC_LEVEL** or data from other charts may be stale.
5. **Session overnight wrap**: `time >= start OR time < end` (NOT AND).
6. **Last bar index**: `sc.ArraySize - 1` (not `sc.ArraySize`).
7. **No STL in hot paths** -- use SCFloatArray, raw arrays, sc.AllocateMemory.
8. **SCString not std::string** -- `.Format()`, `.GetChars()`, `.Append()`. Do not mix.
9. **Use double accumulators** for OLS -- float drifts over 3000+ bars. Guard `denom == 0`.
10. **`%,.0f` (comma thousands)** = GNU extension, NOT supported by MSVC. Use `%.0f`.
11. **DLL locked (LNK1104)** : Sierra keeps DLL locked. Compile from Sierra (Analysis > Build) or close Sierra entirely.

### Trading-Specific (Phase 2b discoveries)
12. **`sc.BuyEntry()`/`sc.SellEntry()`** trade ONLY the primary chart symbol. For cross-symbol spread: use **`sc.BuyOrder()`/`sc.SellOrder()`** with `.Symbol` explicit.
13. **Changing an Input triggers full recalc** -> orders return -8998. Solution: deferred order pattern (see above).
14. **`sc.SendOrdersToTradeService = !sc.GlobalTradeSimulationIsOn`** REQUIRED in SetDefaults, else orders silently ignored.
15. **`Trade >> Auto Trading Enabled - Global`** must be active, else ret=-1.
16. **Changing SetDefaults properties** (AllowOnlyOneTradePerBar, ReceivePointerEvents, MaximumPositionAllowed, etc.): must **remove and re-add the study** for new values to take effect.
17. **`sc.AllowOnlyOneTradePerBar = 0`** required for 2-leg spread orders (else 2nd leg returns ret=-8997).
18. **`sc.AllowMultipleEntriesInSameDirection = 1`** required for scaling/averaging (else ret=-1).
19. **`Price1 = 0`** required in `s_SCNewOrder` for market orders.
20. **`sc.SubmitOrder()` does NOT exist** in ACSIL. `s_SCNewOrder.IsBuySellOrder` does NOT exist either.
21. **ret=-1** = order rejected. Exact reason in **Trade Service Log** (Window > Trade Service Log), not the general log.
22. **`s_SCPositionData.OpenProfitLoss`** : in simulation, returns ACCOUNT CUMULATIVE total, NOT position P&L. Use manual calc with `AveragePrice`.
23. **`s_SCPositionData`** has NO `TickSize` or `CurrencyValuePerTick` fields. `s_SCSymbolData` does NOT exist in ACSIL.
24. **`SCDateTime` to double** : use `.GetAsDouble()`, cannot assign directly. Diff * 86400.0 = seconds.
25. **One-legged protection** : if one spread leg fails (ret<=0), close surviving leg immediately. Test `ret1 > 0 && ret2 > 0` (not `||`).
26. **Control Bar Buttons** : `SetCustomStudyControlBarButtonText()` sets text only. User must manually add ACS buttons via Global Settings > Customize Control Bars, then Window > Control Bars to show. **MUST re-enable buttons** via `SetCustomStudyControlBarButtonEnable(id, 1)` on init, otherwise `Enable(id, 0)` on click disables them permanently.
27. **Button events** : detect via `sc.MenuEventID` ONLY on last bar (`sc.Index == sc.ArraySize - 1`). Requires `sc.ReceivePointerEvents = ACS_RECEIVE_POINTER_EVENTS_ALWAYS` and `sc.UpdateAlways = 1`.
28. **Replay fast (960X)** : Sierra processes bars in batch. Signal detection must run on EVERY bar (not just last bar) else signal is missed. Guard `!sc.IsFullRecalculation` prevents historical signals.
29. **Cooldown anti double-click** : `sc.CurrentSystemDateTime.GetAsDouble()` in PersistentDouble, diff * 86400.0 = seconds. FLATTEN always exempt.
30. **Float equality guards** : never use `== 0.0` for computed doubles (OLS denominator, ADF ss_x). Use `fabs(x) < 1e-12`. Exact zero comparison may miss near-zero values from floating-point arithmetic.

---

## Reusable C++ Patterns

### State Machine (4 States)

FLAT(0) -> LONG(1)/SHORT(-1) -> COOLDOWN_LONG(2)/COOLDOWN_SHORT(-2) -> FLAT(0)

```cpp
const int STATE_FLAT=0, STATE_LONG=1, STATE_SHORT=-1;
const int STATE_COOLDOWN_LONG=2, STATE_COOLDOWN_SHORT=-2;
int& TradeState = sc.GetPersistentInt(3);
```

Transition order per bar: (1) cooldown reset, (2) EOD flat, (3) exits, (4) entries. After SL -> COOLDOWN. After TP/EOD -> FLAT. Cross-cooldown entry: LONG entry allowed from COOLDOWN_SHORT and vice versa.

### OLS Beta Single-Pass

Convention: `log_a = alpha + beta * log_b`.

```cpp
double sumX=0, sumY=0, sumX2=0, sumXY=0; int count=0;
for (int i = endIndex-period+1; i <= endIndex; i++) {
    double x = arrX[i], y = arrY[i];
    sumX += x; sumY += y; sumX2 += x*x; sumXY += x*y; count++;
}
double n=(double)count, denom = n*sumX2 - sumX*sumX;
double beta = (n*sumXY - sumX*sumY) / denom;
double alpha = (sumY/n) - beta*(sumX/n);
```

### Correlation Single-Pass (Pearson)

```cpp
double sumX=0, sumY=0, sumX2=0, sumY2=0, sumXY=0; int count=0;
for (...) { sumX+=x; sumY+=y; sumX2+=x*x; sumY2+=y*y; sumXY+=x*y; count++; }
double n=(double)count;
double r = (n*sumXY - sumX*sumY) / sqrt((n*sumX2-sumX*sumX)*(n*sumY2-sumY*sumY));
```

### ADF Statistic (Dickey-Fuller Simple)

Regresses `delta_spread = gamma * lag_spread + mu`. Returns t-stat. No augmentation -- matches Python `adf_statistic_simple()`. Critical: -2.86.

```cpp
for (int i = startIdx; i <= endIndex; i++) {
    double deltaS = spread[i] - spread[i-1];
    double lagS   = spread[i-1];
    sumX+=lagS; sumY+=deltaS; sumXY+=lagS*deltaS; sumX2+=lagS*lagS; sumY2+=deltaS*deltaS;
}
double ss_x = sumX2 - n*meanX*meanX;
double ss_xy = sumXY - n*meanX*meanY;
double gamma = ss_xy / ss_x;
double SSR = ss_y - gamma * ss_xy;
double SE_gamma = sqrt((SSR/(n-2.0)) / ss_x);
return gamma / SE_gamma;  // t-stat
```

Guard: `period < 20`, `n < 20`, `variance <= 0`, `ss_x <= 0`.

### Hurst Exponent (R/S -- Sierra Implementation)

R/S over sub-periods {8,16,32,64,128,256}, OLS of log(R/S) vs log(n). Need >= 4 valid points. Clamp [0.01, 0.99].

**Note**: Python backtest uses variance-ratio (not R/S). NQ/YM v1.0 C++ uses variance-ratio to match Python. GC_SI uses R/S.

### Half-Life (AR(1) via Cov/Var)

```cpp
double sumXY=0, sumX2=0;
for (int i = endIndex-period+2; i <= endIndex; i++) {
    sumXY += spread[i] * spread[i-1];
    sumX2 += spread[i-1] * spread[i-1];
}
double phi = sumXY / sumX2;
if (phi <= 0.0 || phi >= 1.0) return 0.0f;
float hl = (float)(-log(2.0) / log(phi));  // clamp [1, 500]
```

### Z-Score Rolling

```cpp
sc.SimpleMovAvg(Spread, SpreadMean, ZScorePeriod);
float stddev = CalculateStdDevOptimized(Spread, sc.Index, ZScorePeriod);
float z = (stddev > 0.0f) ? (spread - SpreadMean[sc.Index]) / stddev : 0.0f;
```

### Confidence Scoring (Two Versions)

**NQ/YM Python-aligned** (v1.0): ADF 40%, Hurst 25%, Corr 20%, HL 15% with ADF gate at stat >= -1.00 -> 0%.

**GC/SI version** (v2.0): ADF 30%, Hurst 30%, Corr 40% (no half-life, no ADF gate).

```cpp
// GC/SI version:
float scoreADF = (adf < -2.86f) ? 30.0f
    : (adf < 0.0f) ? 30.0f * (-2.86f - adf) / -2.86f : 0.0f;
float scoreHurst = (hurst < 0.5f) ? 30.0f * (0.5f - hurst) / 0.5f : 0.0f;
float scoreCorr = (corr > 0.6f) ? fmin(40.0f, 40.0f*(corr-0.6f)/0.4f) : 0.0f;
float total = fmin(100.0f, scoreADF + scoreHurst + scoreCorr);
```

### Dollar-Neutral Sizing

```cpp
float NotA = PriceA * PointValueA;
float NotB = PriceB * PointValueB * (float)ContractsB;
int contractsA = (int)((NotB / NotA) * beta + 0.5f);
contractsA = max(1, min(contractsA, MaxContracts));
```

### Session/EOD/Cooldown Patterns

```cpp
// Session overnight OR logic:
if (SessionStart > SessionEnd)  // overnight
    InSession = (BarTime >= SessionStart || BarTime <= SessionEnd);
else
    InSession = (BarTime >= SessionStart && BarTime <= SessionEnd);

// EOD flat:
if (SessionStart > SessionEnd)
    PastEOD = (BarTime < SessionStart && BarTime >= FlatEODTime);
else
    PastEOD = (BarTime >= FlatEODTime);

// Cooldown after stop loss (directional in GC_SI, single in NQ_YM):
// NQ_YM: COOLDOWN -> FLAT when |z| < z_exit
// GC_SI: COOLDOWN_LONG blocks longs only, shorts can still fire
```

---

## GC_SI v2.0 Micro (Reference Only)

File: `sierra/GC_SI_SpreadMeanReversion_v2.0_micro.cpp` (1733 lines). MGC/SIL auto-trading. 5-state machine (directional cooldown). `spread = log(SIL) - beta * log(MGC) - alpha` (inverted vs NQ_YM). See source file for details.

---

## Instruments and Contract Specifications

### Standard Contracts

| | NQ | ES | RTY | YM |
|---|---|---|---|---|
| **Multiplier** | $20/pt | $50/pt | $50/pt | $5/pt |
| **Tick Size** | 0.25 | 0.25 | 0.10 | 1.00 |
| **Tick Value** | $5.00 | $12.50 | $5.00 | $5.00 |
| **Commission** | $2.50 RT | $2.50 RT | $2.50 RT | $2.50 RT |
| **Margin** | $18,000 | $13,000 | $7,500 | $9,500 |
| **Sierra Symbol** | NQ.D-CME | ES.D-CME | RTY.D-CME | YM.D-CME |

### Micro Contracts

| | MNQ | MYM |
|---|---|---|
| **Multiplier** | $2/pt | $0.50/pt |
| **Tick Size** | 0.25 | 1.00 |
| **Tick Value** | $0.50 | $0.50 |
| **Commission** | $0.79 RT | $0.79 RT |
| **Margin** | $1,800 | $950 |
| **Sierra Symbol** | MNQ.D-CME | MYM.D-CME |

Micros = 1/10th of standard. Same prices, same tick size, lower commission.

### Sierra Chart Numbers (Multi-Chart Access)

| Constant | Value | Instrument |
|----------|-------|------------|
| CHART_ES | 1 | ES |
| CHART_RTY | 2 | RTY |
| CHART_YM | 3 | YM |
| CHART_NQ | 5 | NQ |

All use Continuous Back-adjusted Volume rollover (CBV). Data feed: Denali Exchange Data Feed.

### Micro Scaling for Propfirm ($4,500 trailing DD)

| Scaling | PnL | MaxDD | Safe |
|---------|-----|-------|------|
| E-mini x1 | $84,825 | -$19,155 | NO |
| Micro x1 | $7,968 | -$1,916 | YES (too small) |
| **Micro x2** | **$15,936** | **-$3,832** | **YES (sweet spot)** |
| Micro x3 | $23,903 | -$5,748 | MARGINAL |

---

## Session and Trading Windows

| Window | Time (CT) | Purpose |
|--------|-----------|---------|
| **Globex** | 17:00-16:00 | Full exchange session |
| **Buffered session** | 17:30-15:30 | Hedge/metrics calculation (264 bars/day) |
| **OLS trading** | 02:00-14:00 (Config E) | Entry window |
| **Kalman trading** | 03:00-12:00 (K_Balanced) | Overlay window |
| **Flat time** | 15:30 | Force-close all positions |
| **GC_SI flat** | 14:55 | GC_SI force-flat time |

Session wraps midnight -> OR logic: `t >= 17:30 OR t < 15:30`. 264 bars/day = 22h x 12 bars/h at 5min.

### Regression Convention
`log(NQ) = alpha + beta * log(YM) + epsilon` -- NQ dependent (leg_a), YM explanatory (leg_b). Both OLS and Kalman use same convention. Beta directly usable in sizing.

### Position Sizing
`N_b = round((Notionnel_A / Notionnel_B) * beta * N_a)` where Notionnel = price * multiplier * contracts.

---

## Spread Trading Knowledge

### Spread Creation Methods in Sierra

| Method | Notes |
|--------|-------|
| **Exchange-traded** (`Get Spreads`) | Real bid/ask, SPAN margin. Calendar/ICS only. |
| **Custom Calculated Symbols** | `{ES}-{NQ}`. No continuous futures -- unusable for CBV. |
| **Add Additional Symbol + Difference** | Supports continuous futures. Less flexible. |
| **ACSIL custom** (our approach) | Full control, log spread, multi-chart. Requires C++. |
| **Spreadsheet Study** | Conditional logic, Z-score formulas. Slow on large data. |

### Z-Score Mean Reversion

OLS vs Kalman z-scores are DIFFERENT SCALES:
- **OLS z-score**: rolling (window=30 bars), empirical distribution. Optimal z_entry=3.15, z_stop=4.50
- **Kalman innovation z-score**: `nu(t)/sqrt(F(t))`, N(0,1) by construction. Optimal z_entry=1.375, z_stop=2.75
- Never mix thresholds -- not interchangeable

### Stop Loss Strategies
- **Z-score stop** (primary): OLS z_stop=4.50, Kalman z_stop=2.75
- **Dollar stop** (optional): not recommended on NQ/YM standard (cuts winners too early)
- **Cooldown after stop**: 4-state machine blocks re-entry until |z| < z_exit

### Legging Risk
One leg fills, other doesn't or fills worse. Mitigation: leg less liquid instrument first, use limit on volatile leg + market on other. All orders are market orders in GC_SI.

---

## Python vs C++ Differences to Watch

| Aspect | Python Phase 1 | C++ Sierra |
|--------|----------------|------------|
| **Confidence weights** | ADF 40, Hurst 25, Corr 20, HL 15 | NQ/YM v1.0: matches Python. GC/SI: ADF 30, Hurst 30, Corr 40 (no HL) |
| **ADF gate** | Hard gate: stat >= -1.00 -> 0% | NQ/YM v1.0: matches. GC/SI: no gate |
| **Hurst method** | Variance-ratio (unbiased) | NQ/YM v1.0: variance-ratio. GC/SI: R/S (biased ~0.99 on cumulative) |
| **Half-life in scoring** | 15% weight | NQ/YM v1.0: included. GC/SI: computed but not in score |
| **ADF implementation** | `adf_statistic_simple()` (statistic ~-3.5, threshold -2.86) | Must replicate exact same, NOT statsmodels `adf_rolling()` |
| **Kalman** | Full with innovation z-score | NQ/YM v1.0: implemented. GC/SI: not implemented |
| **Z-score** | OLS rolling + Kalman innovation | GC/SI: OLS rolling only |
| **Sizing** | `(NotA/NotB) * beta * N_a`, multiplier search | GC/SI: fixed SIL qty, compute MGC |
| **TP/SL** | Z-score only | GC/SI: z-score AND dollar targets |
| **Cooldown** | Single: `abs(z) < z_exit` | GC/SI: separate long/short reset levels |
| **Entry window** | Minute-based (02:00-14:00 CT) | GC/SI: hour-based (0-14 CT) |
| **Recalc caching** | No caching (every bar) | Configurable frequency (N bars) |

---

## Architecture Target

- Single DLL, multi-chart setup
- Chart mapping: NQ=chart5, YM=chart3, ES=chart1, RTY=chart2
- OLS engine on NQ/YM pair, Kalman parallel for overlay
- Build: Visual Studio 2022 Build Tools
- Source location: `F:\SierreChart_Spread_Indices\ACS_Source\`
