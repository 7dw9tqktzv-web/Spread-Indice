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

### Universal Spread Indicator + Trading (Mars 2026)
- **File**: `sierra/UniversalSpreadIndicator.cpp` (~1700 lines, 2 studies in 1 DLL)
- **Purpose**: Universal spread analysis + 1-click trading for ANY futures pair.
- **Study 1** (`scsf_UniversalSpreadIndicator`): Z-Score + ZeroLine + Textbox + metric subgraphs → Region 1
- **Study 2** (`scsf_UniversalSpreadLine`): Reads Spread from Study 1 via `GetStudyArrayUsingID()` → Region 2
- **31 Inputs**: ChartB, PointValues, TickSizes, MicroRatios, SymbolNames, OLS/ZScore/Corr/ADF/Hurst/HL periods, ShowTextBox, FontSize, SwapRegression, Z-Score Upper/Lower Line, **EnableTrading(22), LegASymbol(23), LegBSymbol(24), QtyA(25), QtyB(26), DollarTP(27), DollarSL(28), EnableAutoExit(29), ZScoreOnLogRatio(30)**
- **Trading**: BUY SP / SELL SP / FLAT SP via ACS_BUTTON_1/2/3. Dollar TP/SL auto-exits. 5s order cooldown. SwapRegress-aware directions. Position sync via GetTradePositionForSymbolAndAccount()
- **Bug fixes (06/03/2026)**: 1) Button 2s cooldown (ACS buttons are TOGGLES — ON state fires MenuEventID on every study recalc; FLATTEN bypasses cooldown), 2) Micro-aware P&L (strstr detects MicroName in legSym, divides PointValue by MicroRatio), 3) Symbol trim (TrimRight lambda strips trailing spaces from GetString inputs), 4) BUY/SELL allowed in any direction (no more "BLOCKED: FLATTEN first"), 5) FLATTEN silent when already flat (no log spam from shared button state across chartbooks)
- **PersistentVars**: Int(0)=TradingPosition, Int(1)=PendingOrderAction, Int(2)=EntryBarIndex, Double(0)=LastOrderTime, Double(1)=EntrySpreadZ, Double(2)=LastButtonTime(2s button cooldown)
- **15 Subgraphs**: SG0=Spread(ignore), SG1=Z-Score(line), SG2=Zero(line), SG3=LogA(ignore), SG4=LogB(ignore), SG5=SpreadSMA(ignore), SG6=Z Upper(dash), SG7=Z Lower(dash), SG8=ADF Stat(ignore), SG9=Hurst(ignore), SG10=Correlation(ignore), SG11=Half-Life(ignore), SG12=Score(ignore), **SG13=LogRatio(ignore)**, **SG14=LogRatioSMA(ignore)**
- **Defaults**: GC/SI (PV 100/5000, Tick 0.10/0.005, MicroRatio 10/5, Swap ON)
- **Swap Regression Input**: ON = Y=LogB,X=LogA (for GC/SI). OFF = Y=LogA,X=LogB (for NQ/YM)
- **Spread formula**: `LogY - alpha - beta * LogX` (OLS residual centré sur 0)
- **Z-Score toggle (Input 30)**: OFF (default) = z-score on beta-weighted spread. ON = z-score on ln(A/B). Use ln(A/B) when β≈1, beta-weighted when β far from 1. Textbox shows "Z:Beta" or "Z:ln". NOT a dynamic Kalman z-score — just classical SMA/StdDev on the beta-adjusted spread series
- **Scoring**: 40% ADF + 30% Corr + 30% HL (Hurst displayed but out of score)
- **Input limits**: OLS up to 200k, all periods up to 50k (for 1min timeframe testing)
- **Z-Score threshold lines**: configurable via inputs (default ±2.5), no longer hardcoded
- **Compilation**: `F:\SierreChart_Spread_Indices\ACS_Source\`

#### Alert Conditions
- **Syntax**: `=AND(OR(ID1.SG2 > 3.5, ID1.SG2 < -3.5), ID1.SG9 < -2.86, ID1.SG11 > 0.6)`
- **SG mapping**: SG2=Z-Score, SG9=ADF, SG10=Hurst, SG11=Corr, SG12=HL, SG13=Score
- **DRAWSTYLE_IGNORE subgraphs work in alerts** — data is calculated and stored, alerts reference data not visual
- **Optimal alert settings**: Enabled, Reset on New Bar, Once per Bar, **Evaluate On Bar Close** (critical), Full Precision, no Log, no Disable After Trigger
- **ID.SG syntax works ONLY in Alert Conditions** — does NOT work in Spreadsheet Study Formula Source Sheet cells

#### Data Export
- **Spreadsheet Study**: DRAWSTYLE_IGNORE subgraphs do NOT appear in spreadsheet columns (confirmed bug/limitation)
- **Solution**: Use native study **"Write Bar and Study Data To File"** (ID 379) or **Edit > Export Bar and Study Data to Text File** menu command
- DRAWSTYLE_HIDDEN affects Y-axis scale (unlike IGNORE) — avoid for metric subgraphs in z-score region

### KalmanFixedSpreadIndicator (Mars 2026)
- **File**: `sierra/KalmanFixedSpreadIndicator.cpp` (2 studies in 1 DLL)
- **Study 1** (`scsf_KalmanFixedSpreadIndicator`): Fixed alpha/beta spread + z-score + textbox + metrics
- **Study 2** (`scsf_KalmanFixedDollarSpread`): Dollar-weighted spread with reset time
- **24 Inputs Study 1**: ChartB(0), PVs(1-2), Ticks(3-4), MicroRatios(5-6), SymNames(7-8), MicroNames(9-10), KalmanAlpha(11), KalmanBeta(12), ZScorePeriod(13), CorrPeriod(14), ADFPeriod(15), HurstPeriod(16), HLPeriod(17), ShowTextBox(18), FontSize(19), SwapRegress(20), ZUpper(21), ZLower(22), **ZScoreOnWeightedSpread(23)**
- **Z-Score toggle (Input 23)**: OFF = z-score on ln(A/B). ON = z-score on beta-weighted spread (logY - α - β·logX). Classical SMA/StdDev, NOT dynamic Kalman innovation. Textbox shows "Z:Beta" or "Z:ln"
- **Dollar Spread Study**: Inputs 0-10 (ChartB, Lots, PVs, MicroRatios, ResetTime, Invert). PersistentFloat(0) = RefValue for delta calculation

#### Chart Setup (2 charts, same Chartbook)
- Chart A: instrument A + both studies (Region 1 = z-score, Region 2 = spread)
- Chart B: instrument B, data only, **same timeframe + session as Chart A**
- `sc.GetContainingIndexForDateTimeIndex()` maps by datetime — mismatched timeframes = wrong data
- All period inputs are in **bars** — changing timeframe requires recalculating all periods

#### Universal Indicator Gotchas
1. **Early return MUST set all subgraphs to 0.0f** — garbage values corrupt Y-axis scale (-2^32)
2. **Z-score clamp to ±10** — prevents extreme values from blowing up scale
3. **DrawZeros=0 on internal subgraphs** — hides warmup zeros
4. **One ACSIL study = one GraphRegion** — cannot split subgraphs across regions via code
5. **Two-study-one-DLL pattern**: multiple SCSFExport in same .cpp, companion reads via `sc.GetStudyArrayUsingID()`
6. **OLS/Corr/ADF functions must exclude 0.0f values** — warmup zeros pollute regression
7. **Regression direction affects sizing**: β=0.14 (GC on SI) vs β=4.5 (SI on GC)
8. **DLL locked by Sierra** — must close Sierra completely before recompiling externally

#### ACSIL Order Submission Gotchas (CRITICAL)
9. **SCInputRef::GetString() returns "Unset" NOT ""** — `SetString("")` sets StringValue=nullptr. `GetString()` returns "Unset" when nullptr. NEVER check `GetLength()==0` for empty detection. Default to `sc.Symbol`/`sc.GetChartSymbol()` and only override if input is a real symbol.
10. **s_SCNewOrder REQUIRES TimeInForce + TradeAccount** — Without `TimeInForce=SCT_TIF_DAY` and `TradeAccount=sc.SelectedTradeAccount`, orders return -1 (General order error). Reference: `SpreadOrderEntry.cpp` (official Sierra example).
11. **Auto Trading must be enabled** — `Trade >> Auto Trading Enabled - Global` or per-chart. Without it, orders silently fail. Error only visible in **Trade >> Trade Service Log**, not Message Log.
12. **GetTradingErrorTextMessage() returns const char*** — NOT SCString. Use directly in Format `%s` without `.GetChars()`.
13. **BuyOrder/SellOrder return double, not int** — Cast `(int)sc.BuyOrder()`. >0 = success (order ID), -1 = error.
14. **Control Bar Buttons require manual setup** — `SetCustomStudyControlBarButtonText()` sets text only. User must: Global Settings → Customize Control Bars → Add Custom Study Button 1/2/3. Then Window → Control Bars - Chart → Chart Control Bar N.
15. **MenuEventID persists across study calls with UpdateAlways=1** — A single button click sets `sc.MenuEventID` to the button ID, and this value persists across multiple study calls (each tick triggers a call). Without debounce, button handler fires 10-30x per click. Fix: disable button on click, track LastMenuEventID + LastButtonTime, ignore same button within 1s, re-enable only after debounce period. Do NOT reset LastMenuEventID on re-enable.
16. **Control Bar buttons are per-chart-focus** — MenuEventID goes to the chart with keyboard focus. Multiple chartbooks with same study → button events only reach the focused chart. Fix: close and reopen Control Bar if not responding.
17. **Symbol input trailing whitespace causes ret=-5** — `GetString()` preserves whitespace from user input. "MCLJ26_FUT_CME " (with space) fails order submission with ret=-5. Always trim input symbol strings.

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

Full specs in `sierra/specs_futures.md`. Summary below for quick reference.

### Equity Index (Session 17:00-16:00 CT, pause 60 min)

| Symbol | $/pt | Tick | Tick $ | Micro | Micro $/pt |
|--------|------|------|--------|-------|------------|
| NQ | $20 | 0.25 | $5.00 | MNQ | $2 |
| ES | $50 | 0.25 | $12.50 | MES | $5 |
| RTY | $50 | 0.10 | $5.00 | M2K | $5 |
| YM | $5 | 1.00 | $5.00 | MYM | $0.50 |

### Metals COMEX/NYMEX (Globex 17:00-16:00 CT, RTH variable par produit)

RTH : GC 7:20-12:30, SI 7:25-12:25, HG 7:10-12:00, PL 7:20-12:05, PA 7:30-12:00.

#### Standard
| Symbol | Produit | Exchange | Contrat | $/pt | Tick | Tick $ | Mois |
|--------|---------|----------|---------|------|------|--------|------|
| GC | Gold | COMEX | 100 oz | $100 | $0.10/oz | $10.00 | G,J,M,Q,V,Z |
| SI | Silver | COMEX | 5,000 oz | $5,000 | $0.005/oz | $25.00 | H,K,N,U,Z |
| HG | Copper | COMEX | 25,000 lbs | $25,000 | $0.0005/lb | $12.50 | H,K,N,U,Z |
| PL | Platinum | NYMEX | 50 oz | $50 | $0.10/oz | $5.00 | All months (primary F,J,N,V) |
| PA | Palladium | NYMEX | 100 oz | $100 | $0.50/oz | $50.00 | H,M,U,Z |

#### E-mini (cash settled)
| Symbol | Contrat | Ratio | Tick | Tick $ |
|--------|---------|-------|------|--------|
| QO | 50 oz | 1/2 GC | $0.25/oz | $12.50 |
| QI | 2,500 oz | 1/2 SI | $0.0125/oz | $31.25 |

#### Micro
| Symbol | Produit | Contrat | Ratio | Tick | Tick $ | Settlement |
|--------|---------|---------|-------|------|--------|------------|
| MGC | Micro Gold | 10 oz | 1/10 GC | $0.10/oz | $1.00 | Physical |
| SIL | Micro Silver | 1,000 oz | 1/5 SI | $0.005/oz | $5.00 | Physical |
| MHG | Micro Copper | 2,500 lbs | 1/10 HG | $0.0005/lb | $1.25 | Cash settled |
| PLM | Micro Platinum | 10 oz | 1/5 PL | $0.10/oz | $1.00 | Cash settled |
| PAM | Micro Palladium | 10 oz | 1/10 PA | $0.50/oz | $5.00 | Physical |

Note : SIL et PLM sont 1/5 du standard (pas 1/10). Barchart symbols: PLM=YL, PAM=GP.
Toutes specs vérifiées sur Barchart profiles + Ironbeam + QuantVPS + Lincoln Park Financial.
Symboles métaux Rithmic : GC, SI, HG, PL, PA (std) | QO, QI (e-mini) | MGC, SIL, MHG, PLM, PAM (micro).

### Energy NYMEX (Globex 17:00-16:00 CT, RTH 8:00-13:30 CT)

#### Standard
| Symbol | Produit | Contrat | $/pt | Tick | Tick $ |
|--------|---------|---------|------|------|--------|
| CL | Crude Oil WTI | 1,000 barils | $1,000 | 0.01 | $10.00 |
| HO | Heating Oil (ULSD) | 42,000 gallons | $42,000 | 0.0001 | $4.20 |
| RB | RBOB Gasoline | 42,000 gallons | $42,000 | 0.0001 | $4.20 |
| NG | Natural Gas | 10,000 MMBtu | $10,000 | 0.001 | $10.00 |

#### E-mini
| Symbol | Produit | Contrat | Ratio | Tick | Tick $ | Settlement |
|--------|---------|---------|-------|------|--------|------------|
| QM | E-mini Crude Oil | 500 barils | 1/2 CL | 0.025 | $12.50 | Financial |
| QH | E-mini ULSD (HO) | 21,000 gallons | 1/2 HO | 0.001 | $21.00 | Physical |
| QG | E-mini Nat Gas | 2,500 MMBtu | 1/4 NG | 0.005 | $12.50 | Financial |

#### Micro
| Symbol | Produit | Contrat | Ratio | Tick | Tick $ | Settlement |
|--------|---------|---------|-------|------|--------|------------|
| MCL | Micro Crude Oil | 100 barils | 1/10 CL | 0.01 | $1.00 | Physical |
| MRB | Micro RBOB | 4,200 gallons | 1/10 RB | 0.0001 | $0.42 | Physical |
| MNG | Micro Nat Gas | 1,000 MMBtu | 1/10 NG | 0.001 | $1.00 | Financial |

Note : HO n'a PAS de micro, seulement un e-mini (QH). RB n'a PAS d'e-mini, seulement un micro (MRB).
Toutes les specs vérifiées sur Barchart (source CME/NYMEX).

### Convention symboles Denali vs Rithmic

**Denali/Teton** : `[ROOT][MOIS][AA]_FUT_CME` (ex: CLJ26_FUT_CME). `_CME` pour tout CME Group.
**Rithmic** : `[ROOT][MOIS][A]` (ex: CLJ6). Pas de suffixe, exchange séparé.
**Mois** : F(Jan) G(Feb) H(Mar) J(Apr) K(May) M(Jun) N(Jul) Q(Aug) U(Sep) V(Oct) X(Nov) Z(Dec).
`Edit >> Translate Symbols to Current Service` fait la conversion auto entre les deux formats.

Symboles énergie Rithmic : CL, HO, RB, NG (standard) | QM, QH, QG (e-mini) | MCL, MRB, MNG (micro).
Symboles indices Rithmic : ES, NQ, YM (CBOT), RTY | MES, MNQ, MYM (CBOT), M2K.

### Grains CBOT (Globex 19:00-7:45 + 8:30-13:20 CT SPLIT, RTH 8:30-13:20 CT)

**Session SPLIT** : pause 45 min entre 7:45 et 8:30 CT (contrairement indices/énergie/métaux = continu).

#### Standard
| Symbol | Produit | Contrat | $/pt | Tick | Tick $ | Mois |
|--------|---------|---------|------|------|--------|------|
| ZC | Corn | 5,000 bu | $50/¢ | 1/4¢ ($0.0025/bu) | $12.50 | H,K,N,U,Z |
| ZW | SRW Wheat | 5,000 bu | $50/¢ | 1/4¢ ($0.0025/bu) | $12.50 | H,K,N,U,Z |
| ZS | Soybeans | 5,000 bu | $50/¢ | 1/4¢ ($0.0025/bu) | $12.50 | F,H,K,N,Q,U,X |
| ZL | Soybean Oil | 60,000 lbs | $600/¢ | 0.01¢ ($0.0001/lb) | $6.00 | F,H,K,N,Q,U,V,Z |
| ZM | Soybean Meal | 100 tons | $100/$ | $0.10/ton | $10.00 | F,H,K,N,Q,U,V,Z |

#### Micro (lancés 24 février 2025, CBOT, financial settlement)
| Symbol | Produit | Contrat | Ratio | Tick | Tick $ | $/pt |
|--------|---------|---------|-------|------|--------|------|
| MZC | Micro Corn | 500 bu | 1/10 ZC | 1/2¢ ($0.005/bu) | $2.50 | $5/¢ |
| MZW | Micro Wheat | 500 bu | 1/10 ZW | 1/2¢ ($0.005/bu) | $2.50 | $5/¢ |
| MZS | Micro Soybeans | 500 bu | 1/10 ZS | 1/2¢ ($0.005/bu) | $2.50 | $5/¢ |
| MZL | Micro Soy Oil | 6,000 lbs | 1/10 ZL | 0.02¢ ($0.0002/lb) | $1.20 | $60/¢ |
| MZM | Micro Soy Meal | 10 tons | 1/10 ZM | $0.20/ton | $2.00 | $10/$ |

Note : tick micro ag = 2x le tick standard (particularité ag). Tick value micro = 1/5 standard.
Mêmes mois et mêmes horaires que le standard. Source : CME Group FAQ officiel.

#### Mini (legacy, faible liquidité -- remplacés de facto par micro)
| Symbol | Contrat | Ratio | Tick | Tick $ | Settlement |
|--------|---------|-------|------|--------|------------|
| XC | 1,000 bu | 1/5 ZC | 1/8¢ | $1.25 | Financial |

Symboles grains Rithmic : ZC, ZW, ZS, ZL, ZM (std) | MZC, MZW, MZS, MZL, MZM (micro).
Denali : ZCN26_FUT_CME, MZCN26_FUT_CME, etc. (_FUT_CME pour tout CME Group incl. CBOT).

### Interest Rates (Session 17:00-16:00 CT, pause 60 min)

| Symbol | Face | $/pt | Tick | Tick $ |
|--------|------|------|------|--------|
| ZT | $200K | $2,000 | 1/128 | $15.625 |
| ZF | $100K | $1,000 | 1/128 | $7.8125 |
| ZN | $100K | $1,000 | 1/64 | $15.625 |
| ZB | $100K | $1,000 | 1/32 | $31.25 |

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
| **Globex equity/metals/energy/bonds** | 17:00-16:00 | Full session (pause 16:00-17:00, 60 min) |
| **Globex grains (ZW/ZC/ZS/ZL/ZM)** | 19:00-7:45 + 8:30-13:20 | SPLIT session (pause 45min 7:45-8:30, puis 13:20-19:00) |
| **Buffered session** | 17:30-15:30 | Hedge/metrics calculation (264 bars/day) |
| **OLS trading** | 02:00-14:00 (Config E) | Entry window |
| **Kalman trading** | 03:00-12:00 (K_Balanced) | Overlay window |
| **Flat time NQ/YM** | 15:30 | Force-close all positions |
| **GC_SI flat** | 14:55 | GC_SI force-flat time |

Session wraps midnight -> OR logic: `t >= 17:30 OR t < 15:30`. 264 bars/day = 22h x 12 bars/h at 5min.

### RTH (Regular Trading Hours) par produit

| Produit | RTH (CT) | Globex (CT) | Source |
|---------|----------|-------------|--------|
| NQ, ES, YM, RTY | 8:30 - 15:00 | 17:00 - 16:00 | NYSE/NASDAQ cash |
| CL, HO, RB, NG | 8:00 - 13:30 | 17:00 - 16:00 | Ancien pit NYMEX |
| GC (Gold) | 7:20 - 12:30 | 17:00 - 16:00 | COMEX |
| SI (Silver) | 7:25 - 12:25 | 17:00 - 16:00 | COMEX |
| HG (Copper) | 7:10 - 12:00 | 17:00 - 16:00 | COMEX |
| PL (Platinum) | 7:20 - 12:05 | 17:00 - 16:00 | NYMEX |
| PA (Palladium) | 7:30 - 12:00 | 17:00 - 16:00 | NYMEX |
| ZW, ZC, ZS, ZL, ZM | 8:30 - 13:20 | 19:00-7:45 + 8:30-13:20 (SPLIT) | Ancien pit CBOT |

Vérifié via Barchart (données CME/NYMEX). Pour charts cash Sierra, utiliser les horaires RTH.

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

## Chartbook Configuration & Built-in Studies

### Chart Header
- **DChg%** uses **previous settlement price** (CME) as reference, NOT session open. Known discrepancy with Percent Change Since Open study.
- **DV** = session volume. Both DChg% and DV are Globex session values.
- **Labels (DChg, DV, etc.) are hardcoded** -- cannot be renamed.
- Customize visible fields: `Global Settings >> Customize Chart Header - Standard`
- Font: `Global Settings >> Graphics Settings >> Fonts tab >> Chart Text`
- Colors: `Global Settings >> Graphics Settings >> Colors and Widths >> Net Change Up/Down`

### Daily OHLC (Study ID=137)
Draws Open/High/Low/Close horizontal lines on intraday chart.
- **Use this Intraday Chart** = Yes (uses own chart data, no external ref needed)
- **Reference Days Back** = 0 (current day only) or 1+ for previous days
- **Graph High Low Historically** = Yes (show HH/LL on past days too)
- **Display on Day Session Only** = Yes (requires Evening Session config)
- **Value label only (no dash)**: change Draw Style to **Text** in Subgraphs tab (shows price label on Values Scale without drawing a line)

### Horizontal Line at Time (Study ID=306)
Draws horizontal line at price level at a specific time. Good for Cash Open (8:30 CT).
- **Start Time** = 08:30:00 (time of the level)
- **Use Stop Time** = Yes, **Stop Time** = 16:00:00 (line stops extending)
- **Limit Horizontal Line From Time To 1 Day** = Yes (prevents extension to next day)
- **Ignore Weekends** = Yes (no phantom lines on weekends)

### Percent Change Since Open (Study ID=325)
Formula: `v * (Current - Open_bar1) / Open_bar1`. **Multiplier must = 100** for true percent.
- Uses **first bar of trading day** as reference (session open at 17:00 CT)
- Does NOT match header DChg% (different reference: settlement vs session open)

### Text Display For Study from Chart (Study ID=334)
Displays a study value from another chart as text overlay on current chart.
- Use to show cash session values (%, volume) on Globex chart
- Source chart must be open (can be minimized as floating window)

### Study/Price Overlay
- **Fill Blanks With Last Value** = No (prevents line extension across non-data periods)
- Discontinuities if cash session overlaid on Globex chart (vertical jumps between days)

### Hiding Charts (Window >> Hide Window)
- **Menu**: `Window >> Hide Window` -- chart disappears from screen entirely
- **Still active**: receives data, calculates studies, cross-chart references work (confirmed by SC Engineering)
- **Restore**: via **CW menu** (top bar) -- hidden charts listed there
- **Tab indicator**: `Global Settings >> General Settings >> Windows >> Show Hidden Windows on MDI Tabs` shows hidden charts with "H:" prefix
- **CRITICAL**: `General Settings >> GUI >> Destroy Chart Windows When Hidden` must be **No** -- otherwise charts are destroyed on chartbook switch and cross-chart refs break
- **MDI Minimize Method**: `General Settings >> GUI >> MDI Child Window Minimize Method` can be set to "Hide Window" so minimize button = hide

### VWAP Weekly (Study ID=108)
- **Time Period Type** = Weeks, **Time Period Length** = 1
- **Base on Underlying Data** = Yes (more accurate on 5min)
- Bands: VWAP Variance method, multipliers 0.5/1/1.5/2
- **Do NOT enable "Use Monday as Start of Week"** for futures — Globex week starts Sunday 17:00 CT, default Sunday behavior is correct
- Chart must load 7+ days of data

### Time Range Highlight - Transparent (Study ID=316)
Shades chart background for a time range. Semi-transparent, doesn't obscure candles.
- **Start Time** = 08:30:00, **End Time** = 15:00:00 (cash session highlight)
- Color set in Subgraphs tab

### Volume (Study ID=8)
- **Chart Region** = 2 (separate region below price). No inputs.
- Primary color = up bars, Secondary color = down bars (auto based on Close vs Open)

### Countdown Timer (Study ID=201)
- **Display Continuous Time Countdown Based on Real-Time Clock** = Yes (otherwise only advances on trades)
- Position inputs only work on first add — after that, right-click >> Move Drawing
- Increases CPU usage slightly — use on few charts only

### Practical Tips
- **Set Defaults** button in study settings saves inputs for all future instances of that study type
- Cash session chart: create separate chart with Session Start=08:30, Session End=15:30 (or 16:00)
- Cross-chart refs (Text Display, Study/Price Overlay, ACSIL) require charts in **same chartbook** and **open** (hidden = OK, closed = broken)
- **Draw Style "Text"** = shows value label only, no line/dash on chart

### Alert Condition Formula (Study Alerts)
- **Syntaxe** : fonctions spreadsheet-style, PAS d'opérateurs infixe
- Commence par `=`
- `OR()` et `AND()` sont des **fonctions** avec virgules : `=OR(cond1, cond2)`
- `OR` / `AND` seul en texte ne marche PAS, `||` / `&&` non plus
- Référence subgraph : `SG1`, `SG2`, etc. Pour un autre study : `ID#.SG#` (ex: `ID2.SG1`)
- Comparateurs : `=`, `<>`, `<`, `<=`, `>`, `>=`
- Valeurs négatives supportées directement (ex: `-2.5`)
- Fonctions utiles : `CROSSOVER(SG1, value)`, `CROSSUNDER(SG1, value)`
- **Exemple z-score ±2.5** : `=OR(SG1 >= 2.5, SG1 <= -2.5)`
- Settings recommandés : Enabled ✓, Alert Only Once per Bar ✓
- Source : sierrachart.com/index.php?page=doc/StudyChartAlertsAndScanning.php
- **Data corruption** : si chart affiche données aberrantes, `Edit > Delete All Data and Download` résout 90% des cas. Sinon ouvrir un nouveau chart frais.

### Continuous Futures Contract (Réglages & Recommandations SC Engineering)

#### Options disponibles dans le dropdown
| Option | Back Adj | Rollover |
|--------|----------|----------|
| None | - | - |
| Date Rule Rollover | Non | Date fixe |
| Volume Based Rollover | Non | Volume |
| Date Rule Rollover, Back Adjusted | Oui | Date fixe |
| Volume Based Rollover, Back Adjusted | Oui | Volume |
| Rollover Each Year, Same Month | - | Annuel |
| Forward Curve | - | Multi-mois |

#### Position SC Engineering (Support Board)
- **Thread #77164** : *"We do not recommend using the Back Adjusted option."* Position officielle claire.
- **Thread #68299** : Valeurs back-adjustment instables (changent jour en jour après rollover, ex: -43 → -29). Corrompt chart drawings.
- **Thread #64384** : *"Don't expect any data provider to keep historical data for contract months that are not widely traded at all."* Mois illiquides (ex: V/October sur GC/SI) = pas de data Denali.
- **Thread #84455** : Gaps sur continuous → fix : Intraday Data Storage Time Unit = **1 Minute**.
- **Thread #90754** : Gaps après rollover → fix : mettre à jour Sierra Chart à la dernière version.
- **Thread #91218** : Traders pro n'utilisent PAS continuous. Roll manuel + exclusion semaine de rollover.

#### Comportement par produit (testé empiriquement)

| Actif | Volume Based Rollover (non-back adj) | Gap au rollover |
|-------|--------------------------------------|-----------------|
| **GC** | Fonctionne — très liquide (~300-400k/j) | ~40 pts (petit, acceptable) |
| **SI** | **Gaps importants** — moins liquide (~50-80k/j), contango marqué | Steps/artefacts sévères |
| **HG** | Probablement OK (liquide comme GC) | À vérifier |
| **PL** | Probablement gaps (thin) | À vérifier |
| **PA** | Probablement gaps (thin) | À vérifier |

**Observation empirique SI** : le gap commence le **mercredi 17:00 CT** (ouverture Globex) avant le FND. Seuls **2-3 jours** avant le FND sont affectés, pas la semaine entière.

#### Config Sierra Chart (affichage, par produit)

**Pour tous les métaux COMEX :**
| Setting | Valeur |
|---------|--------|
| Continuous Contract | **Volume Based Rollover** (sans Back Adjusted) |
| Automatically Rollover Futures Symbol | **Yes** |
| Dates to Exclude | **Aucun** (même pour SI — voir Rollover Gate ci-dessous) |

**Pourquoi pas de Date Exclude pour les spreads :**
GC et SI ont des **calendriers de rollover décalés** :
- GC : GJMQZ (fév, avr, jun, aoû, déc)
- SI : HKNUZ (mar, mai, jul, sep, déc)
- Seul **décembre** coïncide

Si on exclut 3 jours sur SI mais pas GC, le beta rolling OLS est calculé sur des données asymétriques (manque de barres sur un leg). Le Date Exclude **casse les calculs de spread**. Le petit gap visuel sur le chart est acceptable — c'est le **rollover gate** qui protège les signaux.

#### Rollover Gate pour Spread Trading (SOLUTION VALIDÉE)

**Principe** : même pattern que les binary gates (ADF, Hurst, Corr). Ne PAS supprimer de data — bloquer les **entrées** pendant les périodes de roll de l'un OU l'autre leg. Les sorties ne sont JAMAIS bloquées.

**Pourquoi c'est la bonne approche :**
1. Données intactes → OLS beta, z-score, rolling averages restent corrects
2. L'artefact de roll (2-3 barres) est dilué dans la fenêtre OLS (7000 barres = 0.04% d'impact)
3. Pattern identique aux gates existantes → facile à implémenter
4. Pas de problème d'asymétrie entre legs

**Impact sur 1 an (GC/SI spread) :**
```
GC rolls: G→J(jan), J→M(mar), M→Q(mai), Q→Z(jul), Z→G(nov) = 5 rolls
SI rolls: Z→H(fév), H→K(fév), K→N(avr), N→U(jun), U→Z(aoû) = 5 rolls
Chevauchement: Z (dec) seulement
Total: ~9 périodes de blocage distinctes × 2-3 jours = ~20-27 jours bloqués/an
Coût: ~5-10 trades perdus/an, zéro artefact dans les calculs
```

**Implémentation Python (backtest) :**
```python
# rollover_gate = not (is_roll_period(leg_a) or is_roll_period(leg_b))
# Bloque entrées, jamais sorties (même logique que apply_gate_filter_numba)
```

**Implémentation C++ (Sierra, Phase 2) :**
```cpp
// sc.ContractRolloverDate disponible en ACSIL
// Supprimer auto-entry pendant rollover, permettre manual BUY/SELL/FLATTEN
// Permettre auto-exits normalement
```

**Note NQ/YM et NQ/RTY** : les indices equity (NQ, ES, RTY, YM) partagent le MÊME cycle trimestriel (H, M, U, Z) et rollent la même semaine (3ème vendredi de mars/jun/sep/déc). Le problème d'asynchronisme n'existe quasi pas. Pour les cross-sector (NQ vs GC, ES vs CL), le décalage est majeur → rollover gate indispensable.

#### Config Global Symbol Settings pour les mois liquides
Indispensable pour continuous contract. Évite de charger des mois sans data.
- **SI** : Contract Months = `HKNUZ` (Mar, Mai, Jul, Sep, Dec)
- **GC** : Contract Months = `GJMQZ` (Fév, Avr, Jun, Aoû, Déc)
- **HG** : Contract Months = `HKNUZ`
- **PL** : Contract Months = `FJNV` (Jan, Avr, Jul, Oct -- primary months)
- **PA** : Contract Months = `HMUZ`

#### Date Exclude : usage chart solo uniquement
Le Date Exclude reste utile pour un **chart individuel** (pas un spread) :
- Format : `YYYY-MM-DD` séparés par virgules
- Emplacement : **Global Symbol Settings** (supporte copier/coller) OU **Chart Settings > Chart Data**
- Ajouter 2-3 jours avant le FND pour nettoyer le chart solo
- **Ne PAS utiliser pour des paires/spreads** (asymétrie entre legs)

#### Réglages complémentaires
- **Global Settings > Data/Trade Service > Common Settings** : `Download Total Volume for All Contracts for Futures Daily Data` = **No** (requis pour continuous)
- **Chart > Show Rollover Dates** : activer pour visualiser les transitions
- **Intraday Data Storage Time Unit = 1 Minute** (évite certains gaps, thread #84455)
- `Edit > Delete All Data and Download` → Select All pour re-télécharger toute la chaîne si problème
- **ACSIL** : `sc.AddDateToExclude()` existe mais bug connu (thread #94759, composante time). `sc.ContractRolloverDate` fonctionne pour détecter les rolls.

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

---

## Denali Exchange Data Feed & Teton Order Routing

### Abonnement & Compte actifs
- **Full CME Group** (CME, CBOT, COMEX, NYMEX) nonprofessional -- **$6/mois**, auto-renew
- Couvre : NQ, ES, RTY, YM, MNQ, MYM, MES, M2K + métaux (GC, SI, MGC, SIL) + énergie (CL, NG)
- **Order routing** : Teton CME Routing [trading]
- **Clearing firm** : Dorman/Stage5 (compte funded, PAS pour trading live -- uniquement pour autoriser Denali real-time)
- Requiert un compte trading funded actif (sinon Denali retombe en delayed)
- **Trading live futur** : propfirm (Apex, TopStep, etc.) via **Rithmic order routing only**
- Sierra supporte dual connexion single-instance : Denali (data) + Rithmic (order routing propfirm)
- **Config critique** : Rithmic connexion → Common Settings → "Allow Support for Sierra Chart Data Feeds" = **Yes**
- Laisser Market Data / Historical Data vides côté Rithmic (Denali gère tout)
- Si symboles cassés : `Edit → Translate Symbols to Current Service`
- Pas de frais data en double (1 seul abonnement Denali suffit)
- Doc officielle : sierrachart.com/DenaliExchangeDataFeed.php#IntegrationWithTradingServices

### Guide : Denali data + Rithmic order routing propfirm (single instance)

**Prérequis** : Sierra Package 10+, compte propfirm avec identifiants Rithmic, Denali déjà actif.

1. `Global Settings >> Data/Trade Service Settings`
2. **Current Selected Service** → "Rithmic Direct - DTC [Trading]" → OK → attendre 5-10s
3. Rouvrir settings, dans **Service Settings** :
   - **Server** : celui indiqué par la propfirm
   - **Trading Username/Password** : identifiants propfirm (case-sensitive)
   - **Market Data Username** : **VIDE** (Denali gère)
   - **Market Data Password** : **VIDE**
   - **Historical Data Username/Password** : **VIDE**
4. Onglet **[Common Settings]** → **"Allow Support for Sierra Chart Data Feeds" = Yes** (CRITIQUE)
5. OK → `File >> Reconnect`
6. `Edit >> Translate Symbols to Current Service` (convertit symboles au format Rithmic)
7. Vérifier : **[M]** après symboles = Denali actif, compte propfirm visible dans Trade Window

**Pour revenir à Teton** : même procédure, re-sélectionner "Teton CME Routing".

### Instances multiples Sierra Chart

- **1 licence suffit** pour N installations sur le même PC
- Installer dans des dossiers séparés (ex: `C:\SierraChart\`, `C:\SierraChart2\`)
- Chaque instance = Data Files Folder différent (éviter conflits)
- Changer titre fenêtre : `Global Settings >> General Settings >> GUI`
- **Max 3 connexions Denali par PC** (limite CME, pas Sierra)
- Exemples de setup :
  - Instance 1 : Teton/Dorman (analyse, simulation)
  - Instance 2 : Rithmic propfirm #1 (trading live)
  - Instance 3 : Rithmic propfirm #2 (trading live)
- 4e instance : doit désactiver Denali (utiliser data Rithmic) ou utiliser un 2e PC
- Toutes partagent le même abonnement Denali, pas de frais supplémentaires

### Denali = DATA (Exchange → Toi)
- Data feed propriétaire Sierra Chart, connexion **directe CME via FIX** (pas d'intermédiaire Rithmic/CQG)
- Serveurs au **Equinix Cermak Road, Chicago** -- datacenter de référence marchés US
- **Tick-by-tick non filtré**, timestamp milliseconde
- **Market Depth** complet non filtré : **500 niveaux** CME (vs ~10-20 retail typique)
- **Bid/Ask trade volume** 100% précis sur CME (attribution correcte bid/ask)
- **Historique profond** : ticks depuis 2011 (CME), barres 1min depuis 2008-2010, daily 15+ ans
- Futures spreads (sept 2015+), options futures CME (sept 2017+)
- Protocole DTC, traitement en background thread, stockage SSD
- Coût : exchange fees nonpro $1.50-$30/mois CME ; pro $115-$462/mois

### Teton = ORDRES (Toi → Exchange)
- Order routing propriétaire Sierra Chart, **colocalisé Aurora IL avec moteurs matching CME**
- Connexion **directe CME/CBOT/NYMEX/COMEX** sans provider intermédiaire
- Latence : **< 500 microsecondes** (0.5ms) standard
- **Quad connections** + triple redondance internet (Zayo, Verizon, Cogent)
- **Zéro frais de transaction** Sierra (uniquement commissions clearing firm)
- **OCO et brackets server-side** (survivent à une déconnexion)
- 1400 niveaux market depth supportés
- Historique fills 2+ ans, Web Trading Panel mobile 24/7
- **Broker-agnostique** : le choix du clearing firm n'affecte PAS la vitesse (confirmé SC Engineering)
- Clearing firms : Edge Clear, Ironbeam, AMP, Optimus, Dorman, Phillip Capital, Stage 5, etc. (~17)

### Pourquoi Denali est performant vs alternatives (Rithmic, CQG)
1. **Pas d'intermédiaire** : CME → Sierra direct. Chaque intermédiaire ajoute latence + peut filtrer/agréger
2. **Données non filtrées** : chaque tick individuel transmis (certains feeds agrègent les ticks rapides)
3. **Profondeur 500 niveaux** vs 10-20 typique -- essentiel pour orderflow
4. **Historique massif** : Rithmic = quelques jours, Denali = depuis 2011
5. **Stabilité** : zero issues reportées vs problèmes connectivité Rithmic signalés
6. **Recommandé par SC** : "utilisez Denali sauf si on n'a pas l'exchange"

### Denali + Teton = Stack intégré pour Spread Trading
```
CME (Aurora IL) --data--> Denali (Equinix Chicago) --> Sierra/ACSIL --> signal
CME (Aurora IL) <--orders-- Teton (Coloc Aurora) <-- sc.BuyOrder()
```
- Denali fournit ticks NQ+YM pour calcul spread/z-score temps réel dans ACSIL
- Teton route les ordres des 2 legs directement au CME (< 500μs)
- Colocation Teton essentielle pour spread : 2 legs quasi-simultanées = minimise legging risk
- Stack 100% Sierra = aucun tiers entre code ACSIL et CME (data ET ordres)

## DTC Protocol -- Recherche complète (07/03/2026)

### Qu'est-ce que DTC ?
- **Data and Trading Communications Protocol** : protocole ouvert (dtcprotocol.org) pour market data + trading
- Conçu par Sierra Chart, adopté par quelques autres plateformes
- Serveur intégré dans Sierra Chart : `Global Settings → Sierra Chart Server Settings → DTC Protocol Server`

### Configuration du serveur DTC dans Sierra Chart
- **Port par défaut** : 11099 (TCP), configurable
- **Port historique** : séparé (11098 typiquement)
- **WebSocket** : auto-détecté quand le client se connecte via `ws://` ou `wss://`
- **TLS** : fichiers `TLSCertificate.key` + `TLSCertificate.crt` dans `/ServerCertificate`
- **Auth** : optionnelle, configurable dans les settings (username/password)
- **Activation** : `Global Settings → Sierra Chart Server Settings → Enable DTC Protocol Server = Yes`
- **Encodages** : Binary (défaut historique), Binary VLS, JSON, Compact JSON, Google Protocol Buffers (GPB)
  - L'encodage se négocie via `ENCODING_REQUEST`/`ENCODING_RESPONSE` au début de la connexion
  - GPB recommandé pour performance, JSON pour debug/prototypage

### Messages DTC principaux
- **Connexion** : `ENCODING_REQUEST` → `ENCODING_RESPONSE` → `LOGON_REQUEST` → `LOGON_RESPONSE`
- **Heartbeat** : `HEARTBEAT` (obligatoire, sinon déconnexion après timeout)
- **Market Data** : `MARKET_DATA_REQUEST` (type 101), `MARKET_DATA_REJECT`, `MARKET_DATA_SNAPSHOT`
- **Market Depth** : `MARKET_DEPTH_REQUEST` (type 102), `MARKET_DEPTH_REJECT`
- **Historical** : `HISTORICAL_PRICE_DATA_REQUEST`, `HISTORICAL_PRICE_DATA_RECORD_RESPONSE`
- **Trading** : `SUBMIT_NEW_SINGLE_ORDER`, `CANCEL_ORDER`, `CANCEL_REPLACE_ORDER`
- **Account** : `TRADE_ACCOUNTS_REQUEST`, `ACCOUNT_BALANCE_REQUEST`
- **Security** : `SECURITY_DEFINITION_FOR_SYMBOL_REQUEST`

### RESTRICTION CRITIQUE : CME Market Data = BLOQUÉ
- **Depuis Sierra Chart v2351+** : les requêtes `MARKET_DATA_REQUEST` pour symboles CME sont rejetées
- Réponse : `MARKET_DATA_REJECT` avec `RejectText = "Market data request not allowed"`
- S'applique AUSSI en localhost (127.0.0.1:11099) — pas seulement remote
- **Raison** : accords de redistribution CME Group. Sierra Chart n'a pas le droit de redistribuer les données CME via DTC
- Sierra Chart Engineering : *"This is no longer allowed due to exchange rules"*
- `MARKET_DEPTH_REQUEST` également rejeté pour CME
- **Avant v2351** : fonctionnait (confirmé par utilisateurs). Bloqué ensuite par Sierra
- **Article Hunt Gather Trade (sept 2025)** : confirme le blocage, projet abandonné

### Ce qui FONCTIONNE via DTC
| Fonctionnalité | Statut | Notes |
|----------------|--------|-------|
| Logon/Auth | ✅ | Username/password optionnel |
| Trading (ordres) | ✅ | BUY/SELL/CANCEL via DTC, Allow Trading doit être activé |
| Sierra-to-Sierra (sub-instance) | ✅ | Même machine uniquement pour market data |
| Historical Data | ⚠️ | Fonctionne mais bugs connus (payload tronqué >1000 bars) |
| CME Real-time Market Data | ❌ | BLOQUÉ par exchange rules |
| CME Market Depth | ❌ | BLOQUÉ par exchange rules |
| Non-CME data | ⚠️ | Possiblement OK si pas de restriction exchange |

### Prérequis pour market data DTC (quand autorisé)
- Le symbole doit avoir un chart intraday ouvert dans Sierra pour que le serveur réponde
- Le `SymbolID` dans la réponse peut différer de celui envoyé dans la requête
- Heartbeat obligatoire sinon déconnexion

### Clients Python DTC existants
| Repo | Stars | Transport | Encoding | Statut |
|------|-------|-----------|----------|--------|
| `jseparovic/python-ws-dtc-client` | 41 | WebSocket | JSON Compact | Le plus complet, REST API incluse |
| `Queeq/pydtc` | 23 | TCP | Binary/Protobuf | Basique, historical data only |
| `john-yan/SierraChartConnect` | 17 | TCP | Protobuf | Historical downloader |
| `puneat/SierraCharts.DTC.NET` | 2 | TCP | .NET | C# uniquement |

### Alternative retenue : yfinance (pas DTC)
- DTC bloqué pour CME data → on utilise `yfinance` (Python) pour le grid search quotidien
- Données delayed ~15min, suffisant pour analyse de régimes/z-scores
- Installé dans le venv du projet, prêt à l'emploi
- Tickers : `CL=F`, `NG=F`, `BZ=F`, `HO=F`, `RB=F`, `GC=F`, `SI=F`, `HG=F`, `PL=F`, `NQ=F`, `ES=F`, `YM=F`, `RTY=F`

### Alternatives DTC potentielles (non implémentées)
1. **ACSIL File Bridge** : study C++ écrit JSON périodiquement, MCP server Python lit le fichier
2. **ACSIL sc.HTTPRequest()** : POST async vers serveur Python local
3. **SC-Py** : bridge tiers (peu maintenu)
4. **Intraday Data File Format** : Sierra recommande d'écrire directement dans les fichiers .scid
5. **DTC pour trading seulement** : ordres via DTC (fonctionnel), data via yfinance
