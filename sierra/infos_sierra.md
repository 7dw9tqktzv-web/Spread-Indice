# Infos Sierra Chart — Configuration utilisateur

Relevé le 2026-02-20 depuis screenshots.

## Version & Data Feed
- **Sierra Chart** : v2867 64-bit (Read Only)
- **Data Feed** : Denali Exchange Data Feed
- **Installation** : `F:\SierreChart_Spread_Indices`

## Symboles (contrats continus CBV)

| Instrument | Symbole | Chart # | Continuous |
|---|---|---|---|
| ES | `ESH26_FUT_CME` | #1 | [CBV] |
| RTY | `RTYH26_FUT_CME` | #2 | [CBV] |
| YM | `YMH26_FUT_CME` | #3 | [CBV] |
| NQ | `NQH26_FUT_CME` | #5 | [CBV] |

Front month : H26 (Mars 2026) pour les 4 instruments.
Les 4 charts sont dans le **même Chartbook**.

## Chart Settings (identique sur les 4 charts)

### Bar Period
- Chart Data Type : **Intraday Chart**
- Bar Period Type : Days-Mins-Secs-Milliseconds Per Bar
- Bar Period Value : **0-5-0** (5 minutes)
- Gap Fill : None
- Graph Draw Type : HL Bars

### Session Times
- Session Start Time : **17:30:00**
- Session End Time : **15:30:00**
- Use Evening Session : **No**
- Evening Start Time : 15:15:00
- Evening End Time : 08:29:59
- New Bar At Session Start : **Yes**
- Weekend Data : **Load All Weekend Data**
- Apply Intraday Session Times To Intraday Chart : **Yes**

### Time Zone
- Global Time Zone : **Chicago (-6 CST/-5 CDT)**
- Per-chart Time Zone : Chicago (-6 CST/-5 CDT) (hérité du global)

### Chart Bar Settings
- Combine Sunday-Monday Daily Bars : No
- Do Not Draw Columns With No Data : No
- Include Columns With No Data : No
- Include Saturday-Sunday Bars : No
- Include Weekend Columns Before First Trade : No

## Build Environment
- **Compilateur** : Visual Studio 2022 Build Tools (v17.14.17)
- **Path** : `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build`
- Build via : `Analysis >> Build Advanced Custom Studies`
- Source files : `F:\SierreChart_Spread_Indices\ACS_Source\`
- **En Phase 2, écrire le .cpp directement dans `F:\SierreChart_Spread_Indices\ACS_Source\`** pour compilation directe sans copie

### Paths vérifiés (General Settings >> Paths)
- Data Files : `F:\SierreChart_Spread_Indices\Data\` ✅
- Market Depth : `F:\SierreChart_Spread_Indices\Data\MarketDepthData\` ✅
- Chart Image : `F:\SierreChart_Teton\Images\` (résidu ancien setup, non bloquant)
- Editor : `C:\Program Files\Notepad++\notepad++.exe`

### ⚠️ Fichiers hérités à adapter en Phase 2
- `VisualCCompile.Bat` — paths hardcodés vers `F:\SierreChart_Backtest_GC_SI_micro\` (ancien projet GC/SI). À réécrire pour pointer vers `F:\SierreChart_Spread_Indices\` quand on créera le study ACSIL.
- `BuildProgramOutput.txt` — vieux log de `SierreChart_TWS_IB`, sera écrasé au prochain build.
- `GC_SI_SpreadMeanReversion_v2.0_micro.cpp`, `PairsZscore.cpp` — anciens projets, ne pas toucher.

## Cross-Chart ACSIL (Phase 2)

Pour accéder aux prix d'un autre instrument depuis un study :
```cpp
SCFloatArray other_close;
sc.GetStudyArrayFromChart(ChartNumber, 0, SC_LAST, other_close);
```

Chart numbers pour le code :
```cpp
#define CHART_ES  1
#define CHART_RTY 2
#define CHART_YM  3
#define CHART_NQ  5
```

## Cohérence Phase 1 ↔ Phase 2

| Paramètre | Phase 1 (Python) | Phase 2 (Sierra) | Match |
|---|---|---|---|
| Timeframe | 5min (resampled) | 0-5-0 natif | ✅ |
| Session | 17:30-15:30 CT | 17:30-15:30 CT | ✅ |
| Buffer | 30min (18:00-15:00) | À implémenter en ACSIL | ⬜ |
| Time Zone | Chicago CT | Chicago (-6 CST/-5 CDT) | ✅ |
| Continuous | Volume Rollover BA | [CBV] | ✅ |
| Log-prices | np.log() | log() C++ | ✅ |
