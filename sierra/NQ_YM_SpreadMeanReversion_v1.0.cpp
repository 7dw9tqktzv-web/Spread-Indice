// ============================================================================
// NQ_YM_SpreadMeanReversion_v1.0.cpp
//
// ETUDE ACSIL - SPREAD MEAN REVERSION NQ/YM
// Phase 2a: Visual indicator + Phase 2b v1: Semi-auto trading
// OLS Config E + Kalman K_Balanced overlay
//
// Description:
//   Indicateur visuel pour le pair trading NQ (Nasdaq 100) / YM (Dow Jones)
//   base sur la cointegration et la mean reversion.
//   OLS Config E: 3300 bars lookback, z_entry=3.15, z_exit=1.00, z_stop=4.50
//   Kalman K_Balanced: alpha=3e-7, innovation z-score, textbox overlay
//   Tous les indicateurs harmonises avec le backtest Python.
//
// Differences critiques vs GC_SI:
//   - Confidence scoring: ADF 40%, Hurst 25%, Corr 20%, HL 15% (pas 30/30/40)
//   - ADF gate: stat >= -1.00 -> confidence = 0%
//   - Hurst: variance-ratio (pas R/S)
//   - Kalman filter: textbox overlay (biais discretionnaire)
//   - 4-state machine: FLAT -> LONG/SHORT -> COOLDOWN -> FLAT
//   - Z-score: Entry + Exit + Stop lines
//   - Dynamic textbox background: green/orange/red/gray
//   - Phase 2b: Semi-auto spread trading (BUY/SELL/FLATTEN + auto-exits)
//
// Auteur: Assistant IA - Developpeur ACSIL Senior
// Date: Fevrier 2026
// ============================================================================

#include "sierrachart.h"
#include <cmath>

SCDLLName("NQ_YM_SpreadMeanReversion")

// ============================================================================
// CONSTANTES DES CONTRATS (CME)
// ============================================================================
const float NQ_MULTIPLIER = 20.0f;    // NQ: $20 par point
const float NQ_TICK_SIZE = 0.25f;     // $0.25 minimum tick
const float NQ_TICK_VALUE = 5.0f;     // $5 par tick

const float YM_MULTIPLIER = 5.0f;     // YM: $5 par point
const float YM_TICK_SIZE = 1.0f;      // $1 minimum tick
const float YM_TICK_VALUE = 5.0f;     // $5 par tick

// Seuils statistiques
const float ADF_CRITICAL_5PCT = -2.86f;

// ============================================================================
// STATE MACHINE (4 etats)
// ============================================================================
const int STATE_FLAT = 0;
const int STATE_LONG = 1;             // Long spread = Buy NQ, Sell YM
const int STATE_SHORT = -1;           // Short spread = Sell NQ, Buy YM
const int STATE_COOLDOWN = 2;         // Post-SL, blocks re-entry until |z| < z_exit

// ============================================================================
// CONFIDENCE SCORING (poids Python)
// ============================================================================
const float W_ADF = 0.40f;
const float W_HURST = 0.25f;
const float W_CORR = 0.20f;
const float W_HL = 0.15f;

// Bornes de scoring
const float ADF_WORST = -1.00f;
const float ADF_BEST = -3.50f;
const float HURST_WORST = 0.52f;
const float HURST_BEST = 0.38f;
const float CORR_WORST = 0.65f;
const float CORR_BEST = 0.92f;
const float HL_MIN = 3.0f;
const float HL_SWEET_LOW = 10.0f;
const float HL_SWEET_HIGH = 60.0f;
const float HL_MAX = 200.0f;

// ============================================================================
// FONCTIONS UTILITAIRES
// ============================================================================

// ----------------------------------------------------------------------------
// 1. CalculateStdDev - Ecart-type (une seule passe)
// ----------------------------------------------------------------------------
float CalculateStdDev(SCFloatArrayRef arr, int endIndex, int period)
{
    if (period <= 1 || endIndex < period - 1)
        return 0.0f;

    double sum = 0.0;
    double sumSq = 0.0;
    int count = 0;

    int startIdx = endIndex - period + 1;

    for (int i = startIdx; i <= endIndex; i++)
    {
        if (i >= 0)
        {
            double val = arr[i];
            sum += val;
            sumSq += val * val;
            count++;
        }
    }

    if (count <= 1)
        return 0.0f;

    double n = (double)count;
    double variance = (sumSq - (sum * sum) / n) / (n - 1.0);

    if (variance <= 0.0)
        return 0.0f;

    return (float)sqrt(variance);
}

// ----------------------------------------------------------------------------
// 2. CalculateCorrelation - Pearson sur log-prix
// ----------------------------------------------------------------------------
float CalculateCorrelation(SCFloatArrayRef logX, SCFloatArrayRef logY,
                           int endIndex, int period)
{
    if (period <= 1 || endIndex < period - 1)
        return 0.0f;

    double sumX = 0.0, sumY = 0.0;
    double sumX2 = 0.0, sumY2 = 0.0;
    double sumXY = 0.0;
    int count = 0;

    int startIdx = endIndex - period + 1;

    for (int i = startIdx; i <= endIndex; i++)
    {
        if (i >= 0)
        {
            double x = logX[i];
            double y = logY[i];

            sumX += x;
            sumY += y;
            sumX2 += x * x;
            sumY2 += y * y;
            sumXY += x * y;
            count++;
        }
    }

    if (count < 2)
        return 0.0f;

    double n = (double)count;
    double numerator = n * sumXY - sumX * sumY;
    double denomX = n * sumX2 - sumX * sumX;
    double denomY = n * sumY2 - sumY * sumY;

    if (denomX <= 0.0 || denomY <= 0.0)
        return 0.0f;

    double correlation = numerator / sqrt(denomX * denomY);

    if (correlation > 1.0) correlation = 1.0;
    if (correlation < -1.0) correlation = -1.0;

    return (float)correlation;
}

// ----------------------------------------------------------------------------
// 3. CalculateOLSBeta - Regression OLS
//    Convention: Y = log(NQ), X = log(YM)
//    log(NQ) = alpha + beta * log(YM) + epsilon
// ----------------------------------------------------------------------------
float CalculateOLSBeta(SCFloatArrayRef arrX, SCFloatArrayRef arrY,
                       int endIndex, int period, float& outAlpha)
{
    if (period < 2 || endIndex < period - 1)
    {
        outAlpha = 0.0f;
        return 1.0f;
    }

    double sumX = 0.0, sumY = 0.0;
    double sumX2 = 0.0;
    double sumXY = 0.0;
    int count = 0;

    int startIdx = endIndex - period + 1;

    for (int i = startIdx; i <= endIndex; i++)
    {
        if (i >= 0)
        {
            double x = arrX[i];
            double y = arrY[i];

            sumX += x;
            sumY += y;
            sumX2 += x * x;
            sumXY += x * y;
            count++;
        }
    }

    if (count < 2)
    {
        outAlpha = 0.0f;
        return 1.0f;
    }

    double n = (double)count;
    double meanX = sumX / n;
    double meanY = sumY / n;

    double denominator = n * sumX2 - sumX * sumX;

    if (denominator == 0.0)
    {
        outAlpha = 0.0f;
        return 1.0f;
    }

    double beta = (n * sumXY - sumX * sumY) / denominator;
    double alpha = meanY - beta * meanX;

    outAlpha = (float)alpha;
    return (float)beta;
}

// ----------------------------------------------------------------------------
// 4. CalculateADFSimple - Test Dickey-Fuller simple (Python-compatible)
//    Minimum count reduced for small windows (window=12)
// ----------------------------------------------------------------------------
float CalculateADFSimple(SCFloatArrayRef spread, int endIndex, int period)
{
    // For small windows like 12, minimum count = max(8, period-2)
    int minCount = period - 2;
    if (minCount < 8) minCount = 8;

    if (period < 5 || endIndex < period)
        return 0.0f;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumXY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    int n = 0;

    int startIdx = endIndex - period + 2;

    for (int i = startIdx; i <= endIndex; i++)
    {
        if (i >= 1)
        {
            double deltaSpread = spread[i] - spread[i-1];
            double lagSpread = spread[i-1];

            sumX += lagSpread;
            sumY += deltaSpread;
            sumXY += lagSpread * deltaSpread;
            sumX2 += lagSpread * lagSpread;
            sumY2 += deltaSpread * deltaSpread;
            n++;
        }
    }

    if (n < minCount)
        return 0.0f;

    double nf = (double)n;
    double meanX = sumX / nf;
    double meanY = sumY / nf;

    double ss_x = sumX2 - nf * meanX * meanX;
    double ss_xy = sumXY - nf * meanX * meanY;
    double ss_y = sumY2 - nf * meanY * meanY;

    if (ss_x == 0.0)
        return 0.0f;

    double gamma = ss_xy / ss_x;
    double SSR = ss_y - gamma * ss_xy;

    if (n <= 2)
        return 0.0f;

    double variance = SSR / (nf - 2.0);

    if (variance <= 0.0 || ss_x <= 0.0)
        return 0.0f;

    double SE_gamma = sqrt(variance / ss_x);

    if (SE_gamma == 0.0)
        return 0.0f;

    return (float)(gamma / SE_gamma);
}

// ----------------------------------------------------------------------------
// 5. CalculateHurstVR - Hurst via variance-ratio (NOT R/S)
//    Matches Python src/stats/hurst.py
// ----------------------------------------------------------------------------
float CalculateHurstVR(SCFloatArrayRef spread, int endIndex, int period)
{
    if (period < 8 || endIndex < period)
        return 0.0f;

    int maxLag = period / 4;
    if (maxLag < 2) maxLag = 2;
    if (maxLag > 50) maxLag = 50;

    double sumLogLag = 0.0, sumLogTau = 0.0;
    double sumLogLag2 = 0.0, sumLogLagLogTau = 0.0;
    int validCount = 0;

    int startIdx = endIndex - period + 1;
    if (startIdx < 0) startIdx = 0;

    for (int lag = 2; lag <= maxLag; lag++)
    {
        // Compute std of differences at this lag
        double sum = 0.0, sumSq = 0.0;
        int cnt = 0;

        for (int i = startIdx; i <= endIndex - lag; i++)
        {
            if (i >= 0)
            {
                double diff = spread[i + lag] - spread[i];
                sum += diff;
                sumSq += diff * diff;
                cnt++;
            }
        }

        if (cnt < 5) continue;

        double mean = sum / cnt;
        double var = (sumSq - cnt * mean * mean) / (cnt - 1);
        if (var <= 0.0) continue;
        double tau = sqrt(var);

        double logLag = log((double)lag);
        double logTau = log(tau);

        sumLogLag += logLag;
        sumLogTau += logTau;
        sumLogLag2 += logLag * logLag;
        sumLogLagLogTau += logLag * logTau;
        validCount++;
    }

    if (validCount < 3) return 0.0f;

    double n = (double)validCount;
    double denom = n * sumLogLag2 - sumLogLag * sumLogLag;
    if (fabs(denom) < 1e-10) return 0.0f;

    double H = (n * sumLogLagLogTau - sumLogLag * sumLogTau) / denom;

    if (H < 0.01) H = 0.01;
    if (H > 0.99) H = 0.99;

    return (float)H;
}

// ----------------------------------------------------------------------------
// 6. CalculateHalfLife - AR(1) via Cov/Var (Python-compatible)
//    phi = Cov(Z(t), Z(t-1)) / Var(Z(t-1)), HL = -ln(2)/ln(phi)
// ----------------------------------------------------------------------------
float CalculateHalfLife(SCFloatArrayRef spread, int endIndex, int period)
{
    if (period < 3 || endIndex < period)
        return 0.0f;

    int startIdx = endIndex - period + 2;
    if (startIdx < 1) startIdx = 1;

    // Compute means first
    double sumX = 0.0, sumY = 0.0;
    int count = 0;
    for (int i = startIdx; i <= endIndex; i++)
    {
        sumX += spread[i-1];  // x = Z(t-1)
        sumY += spread[i];    // y = Z(t)
        count++;
    }
    if (count < 3) return 0.0f;

    double meanX = sumX / count;
    double meanY = sumY / count;

    // Compute Cov(Y,X) and Var(X)
    double covXY = 0.0, varX = 0.0;
    for (int i = startIdx; i <= endIndex; i++)
    {
        double dx = spread[i-1] - meanX;
        double dy = spread[i] - meanY;
        covXY += dx * dy;
        varX += dx * dx;
    }

    if (varX <= 0.0) return 0.0f;

    double phi = covXY / varX;

    if (phi <= 0.0 || phi >= 1.0) return 0.0f;

    double hl = -log(2.0) / log(phi);
    if (hl > 500.0) hl = 500.0;

    return (float)hl;
}

// ----------------------------------------------------------------------------
// 7. CalculateConfidence - Python weights + ADF gate
//    ADF 40%, Hurst 25%, Corr 20%, HL 15%
//    ADF gate: stat >= adfGate -> confidence = 0%
// ----------------------------------------------------------------------------
float CalculateConfidence(float adfStat, float hurst, float correlation,
                          float halfLife, float adfGate)
{
    // ADF gate: if stat >= gate -> confidence = 0%
    if (adfStat >= adfGate || adfStat == 0.0f)
        return 0.0f;

    // ADF score: linear interp, more negative = better
    float s_adf = (adfStat - ADF_WORST) / (ADF_BEST - ADF_WORST);
    if (s_adf < 0.0f) s_adf = 0.0f;
    if (s_adf > 1.0f) s_adf = 1.0f;

    // Hurst score: lower = better
    float s_hurst = (hurst - HURST_WORST) / (HURST_BEST - HURST_WORST);
    if (s_hurst < 0.0f) s_hurst = 0.0f;
    if (s_hurst > 1.0f) s_hurst = 1.0f;

    // Corr score: higher = better
    float s_corr = (correlation - CORR_WORST) / (CORR_BEST - CORR_WORST);
    if (s_corr < 0.0f) s_corr = 0.0f;
    if (s_corr > 1.0f) s_corr = 1.0f;

    // Half-life score: trapezoid (sweet spot 10-60)
    float s_hl = 0.0f;
    if (halfLife >= HL_MIN && halfLife <= HL_MAX)
    {
        if (halfLife >= HL_SWEET_LOW && halfLife <= HL_SWEET_HIGH)
            s_hl = 1.0f;
        else if (halfLife < HL_SWEET_LOW)
            s_hl = (halfLife - HL_MIN) / (HL_SWEET_LOW - HL_MIN);
        else
            s_hl = (HL_MAX - halfLife) / (HL_MAX - HL_SWEET_HIGH);
    }

    float confidence = 100.0f * (W_ADF * s_adf + W_HURST * s_hurst
                                 + W_CORR * s_corr + W_HL * s_hl);

    if (confidence < 0.0f) confidence = 0.0f;
    if (confidence > 100.0f) confidence = 100.0f;

    return confidence;
}

// ============================================================================
// FONCTION PRINCIPALE DE L'ETUDE
// ============================================================================

SCSFExport scsf_NQ_YM_SpreadMeanReversion(SCStudyInterfaceRef sc)
{
    // ========================================================================
    // DEFINITION DES SUBGRAPHS (0-29)
    // ========================================================================

    // --- Spread & Bandes ---
    SCSubgraphRef Spread         = sc.Subgraph[0];    // LINE 2px Yellow
    SCSubgraphRef SpreadMean     = sc.Subgraph[1];    // LINE 1px Gray
    SCSubgraphRef UpperBand      = sc.Subgraph[2];    // LINE 1px DOT Red light
    SCSubgraphRef LowerBand      = sc.Subgraph[3];    // LINE 1px DOT Green light

    // --- Z-Score OLS ---
    SCSubgraphRef ZScore         = sc.Subgraph[4];    // LINE 2px Cyan (dynamic)
    SCSubgraphRef ZEntryPlus     = sc.Subgraph[5];    // LINE 1px DASH Red (+3.15)
    SCSubgraphRef ZEntryMinus    = sc.Subgraph[6];    // LINE 1px DASH Green (-3.15)
    SCSubgraphRef ZZeroLine      = sc.Subgraph[11];   // LINE 1px Gray

    // --- Statistics ---
    SCSubgraphRef ADFStat        = sc.Subgraph[12];   // LINE 1px Purple (dynamic)
    SCSubgraphRef ADFCritical    = sc.Subgraph[13];   // LINE 1px DASH White (-2.86)
    SCSubgraphRef HurstExp       = sc.Subgraph[14];   // LINE 1px Green (dynamic)
    SCSubgraphRef CorrLine       = sc.Subgraph[16];   // LINE 1px Orange (dynamic)
    SCSubgraphRef HalfLifeLine   = sc.Subgraph[17];   // LINE 1px Orange light

    // --- Confidence & Signals ---
    SCSubgraphRef ConfScore      = sc.Subgraph[18];   // BAR 3px dynamic
    SCSubgraphRef SignalLong     = sc.Subgraph[20];   // ARROW_UP 3px Green
    SCSubgraphRef SignalShort    = sc.Subgraph[21];   // ARROW_DOWN 3px Red
    SCSubgraphRef SignalExit     = sc.Subgraph[22];   // DIAMOND 2px White

    // --- Kalman overlay (internal, textbox only) ---
    SCSubgraphRef KalmanBeta     = sc.Subgraph[23];   // IGNORE
    SCSubgraphRef KalmanZInn     = sc.Subgraph[24];   // IGNORE
    SCSubgraphRef KalmanConf     = sc.Subgraph[25];   // IGNORE

    // --- Internal storage ---
    SCSubgraphRef OLSBetaSG      = sc.Subgraph[26];   // IGNORE (also: Arrays[0] = log_NQ)
    SCSubgraphRef OLSAlphaSG     = sc.Subgraph[27];   // IGNORE (also: Arrays[0] = log_YM)
    SCSubgraphRef SpreadStdDev   = sc.Subgraph[28];   // IGNORE
    SCSubgraphRef TradeStateSG   = sc.Subgraph[29];   // IGNORE

    // ========================================================================
    // DEFINITION DES INPUTS (0-21)
    // ========================================================================

    SCInputRef InYMChartNumber   = sc.Input[0];
    SCInputRef InOLSLookback     = sc.Input[1];
    SCInputRef InZScoreWindow    = sc.Input[2];
    SCInputRef InBandMult        = sc.Input[3];
    SCInputRef InZEntry          = sc.Input[4];
    SCInputRef InZExit           = sc.Input[5];
    SCInputRef InZStop           = sc.Input[6];
    SCInputRef InADFWindow       = sc.Input[7];
    SCInputRef InHurstWindow     = sc.Input[8];
    SCInputRef InHLWindow        = sc.Input[9];
    SCInputRef InCorrWindow      = sc.Input[10];
    SCInputRef InMinConfidence   = sc.Input[11];
    SCInputRef InADFGate         = sc.Input[12];
    SCInputRef InSessionStart    = sc.Input[13];
    SCInputRef InSessionEnd      = sc.Input[14];
    SCInputRef InEntryStart      = sc.Input[15];
    SCInputRef InEntryEnd        = sc.Input[16];
    SCInputRef InFlatTime        = sc.Input[17];
    SCInputRef InKalmanAlpha     = sc.Input[18];
    SCInputRef InKalmanWarmup    = sc.Input[19];
    SCInputRef InGapPMult        = sc.Input[20];
    SCInputRef InShowTextBox     = sc.Input[21];

    // --- Trading (Phase 2b) ---
    SCInputRef InTradingAction   = sc.Input[22];
    SCInputRef InLeg1Symbol      = sc.Input[23];
    SCInputRef InLeg2Symbol      = sc.Input[24];
    SCInputRef InLeg1Qty         = sc.Input[25];
    SCInputRef InLeg2Qty         = sc.Input[26];
    SCInputRef InTradingZExit    = sc.Input[27];
    SCInputRef InDollarStop      = sc.Input[28];
    SCInputRef InTimeStop         = sc.Input[29];
    SCInputRef InEnableAutoExit  = sc.Input[30];

    // ========================================================================
    // CONFIGURATION PAR DEFAUT - Config E + K_Balanced + Trading
    // ========================================================================

    if (sc.SetDefaults)
    {
        sc.GraphName = "NQ/YM Spread Mean Reversion v1.0";
        sc.StudyDescription = "OLS Config E + Kalman K_Balanced overlay. "
                              "Phase 2a visual + Phase 2b semi-auto trading.";

        sc.AutoLoop = 1;
        sc.GraphRegion = 1;      // Separate region for spread
        sc.CalculationPrecedence = LOW_PREC_LEVEL;

        // Trading setup (Phase 2b)
        sc.AllowMultipleEntriesInSameDirection = 0;
        sc.MaximumPositionAllowed = 10;
        sc.SupportAttachedOrdersForTrading = 0;
        sc.AllowOnlyOneTradePerBar = 1;
        sc.SupportReversals = 0;
        sc.SendOrdersToTradeService = !sc.GlobalTradeSimulationIsOn;

        // ================================================================
        // SUBGRAPHS
        // ================================================================

        // --- Spread & Bandes ---
        Spread.Name = "Spread (Log)";
        Spread.DrawStyle = DRAWSTYLE_LINE;
        Spread.PrimaryColor = RGB(255, 255, 0);
        Spread.LineWidth = 2;
        Spread.DrawZeros = false;

        SpreadMean.Name = "Spread Mean";
        SpreadMean.DrawStyle = DRAWSTYLE_LINE;
        SpreadMean.PrimaryColor = RGB(128, 128, 128);
        SpreadMean.LineWidth = 1;
        SpreadMean.DrawZeros = false;

        UpperBand.Name = "Upper Band";
        UpperBand.DrawStyle = DRAWSTYLE_LINE;
        UpperBand.PrimaryColor = RGB(255, 100, 100);
        UpperBand.LineWidth = 1;
        UpperBand.LineStyle = LINESTYLE_DOT;
        UpperBand.DrawZeros = false;

        LowerBand.Name = "Lower Band";
        LowerBand.DrawStyle = DRAWSTYLE_LINE;
        LowerBand.PrimaryColor = RGB(100, 255, 100);
        LowerBand.LineWidth = 1;
        LowerBand.LineStyle = LINESTYLE_DOT;
        LowerBand.DrawZeros = false;

        // --- Z-Score OLS ---
        ZScore.Name = "Z-Score OLS";
        ZScore.DrawStyle = DRAWSTYLE_LINE;
        ZScore.PrimaryColor = RGB(0, 200, 255);
        ZScore.SecondaryColorUsed = 1;
        ZScore.LineWidth = 2;
        ZScore.DrawZeros = false;

        ZEntryPlus.Name = "Z Entry +";
        ZEntryPlus.DrawStyle = DRAWSTYLE_LINE;
        ZEntryPlus.PrimaryColor = RGB(255, 0, 0);
        ZEntryPlus.LineWidth = 1;
        ZEntryPlus.LineStyle = LINESTYLE_DASH;
        ZEntryPlus.DrawZeros = false;

        ZEntryMinus.Name = "Z Entry -";
        ZEntryMinus.DrawStyle = DRAWSTYLE_LINE;
        ZEntryMinus.PrimaryColor = RGB(0, 255, 0);
        ZEntryMinus.LineWidth = 1;
        ZEntryMinus.LineStyle = LINESTYLE_DASH;
        ZEntryMinus.DrawZeros = false;

        ZZeroLine.Name = "Z Zero";
        ZZeroLine.DrawStyle = DRAWSTYLE_LINE;
        ZZeroLine.PrimaryColor = RGB(128, 128, 128);
        ZZeroLine.LineWidth = 1;
        ZZeroLine.DrawZeros = false;

        // --- Statistics ---
        ADFStat.Name = "ADF Statistic";
        ADFStat.DrawStyle = DRAWSTYLE_LINE;
        ADFStat.PrimaryColor = RGB(200, 100, 255);
        ADFStat.SecondaryColorUsed = 1;
        ADFStat.LineWidth = 1;
        ADFStat.DrawZeros = false;

        ADFCritical.Name = "ADF Critical -2.86";
        ADFCritical.DrawStyle = DRAWSTYLE_LINE;
        ADFCritical.PrimaryColor = RGB(255, 255, 255);
        ADFCritical.LineWidth = 1;
        ADFCritical.LineStyle = LINESTYLE_DASH;
        ADFCritical.DrawZeros = false;

        HurstExp.Name = "Hurst Exponent (VR)";
        HurstExp.DrawStyle = DRAWSTYLE_LINE;
        HurstExp.PrimaryColor = RGB(100, 200, 100);
        HurstExp.SecondaryColorUsed = 1;
        HurstExp.LineWidth = 1;
        HurstExp.DrawZeros = false;

        CorrLine.Name = "Correlation";
        CorrLine.DrawStyle = DRAWSTYLE_LINE;
        CorrLine.PrimaryColor = RGB(255, 165, 0);
        CorrLine.SecondaryColorUsed = 1;
        CorrLine.LineWidth = 1;
        CorrLine.DrawZeros = false;

        HalfLifeLine.Name = "Half-Life (bars)";
        HalfLifeLine.DrawStyle = DRAWSTYLE_LINE;
        HalfLifeLine.PrimaryColor = RGB(255, 200, 100);
        HalfLifeLine.LineWidth = 1;
        HalfLifeLine.DrawZeros = false;

        // --- Confidence & Signals ---
        ConfScore.Name = "Confidence Score (0-100)";
        ConfScore.DrawStyle = DRAWSTYLE_BAR;
        ConfScore.PrimaryColor = RGB(0, 255, 0);
        ConfScore.SecondaryColorUsed = 1;
        ConfScore.LineWidth = 3;
        ConfScore.DrawZeros = false;

        SignalLong.Name = "Signal LONG (Buy NQ, Sell YM)";
        SignalLong.DrawStyle = DRAWSTYLE_ARROW_UP;
        SignalLong.PrimaryColor = RGB(0, 255, 0);
        SignalLong.LineWidth = 3;
        SignalLong.DrawZeros = false;

        SignalShort.Name = "Signal SHORT (Sell NQ, Buy YM)";
        SignalShort.DrawStyle = DRAWSTYLE_ARROW_DOWN;
        SignalShort.PrimaryColor = RGB(255, 0, 0);
        SignalShort.LineWidth = 3;
        SignalShort.DrawZeros = false;

        SignalExit.Name = "Signal EXIT";
        SignalExit.DrawStyle = DRAWSTYLE_DIAMOND;
        SignalExit.PrimaryColor = RGB(255, 255, 255);
        SignalExit.LineWidth = 2;
        SignalExit.DrawZeros = false;

        // --- Kalman overlay (internal) ---
        KalmanBeta.Name = "Kalman Beta (internal)";
        KalmanBeta.DrawStyle = DRAWSTYLE_IGNORE;
        KalmanBeta.DrawZeros = false;

        KalmanZInn.Name = "Kalman Z-Inn (internal)";
        KalmanZInn.DrawStyle = DRAWSTYLE_IGNORE;
        KalmanZInn.DrawZeros = false;

        KalmanConf.Name = "Kalman Conf (internal)";
        KalmanConf.DrawStyle = DRAWSTYLE_IGNORE;
        KalmanConf.DrawZeros = false;

        // --- Internal storage ---
        OLSBetaSG.Name = "OLS Beta (internal)";
        OLSBetaSG.DrawStyle = DRAWSTYLE_IGNORE;
        OLSBetaSG.DrawZeros = false;

        OLSAlphaSG.Name = "OLS Alpha (internal)";
        OLSAlphaSG.DrawStyle = DRAWSTYLE_IGNORE;
        OLSAlphaSG.DrawZeros = false;

        SpreadStdDev.Name = "Spread StdDev (internal)";
        SpreadStdDev.DrawStyle = DRAWSTYLE_IGNORE;
        SpreadStdDev.DrawZeros = false;

        TradeStateSG.Name = "Trade State (internal)";
        TradeStateSG.DrawStyle = DRAWSTYLE_IGNORE;
        TradeStateSG.DrawZeros = false;

        // ================================================================
        // INPUTS - Config E defaults + K_Balanced
        // ================================================================

        InYMChartNumber.Name = "1. YM Chart Number";
        InYMChartNumber.SetInt(3);
        InYMChartNumber.SetIntLimits(1, 100);
        InYMChartNumber.SetDescription("Number of the YM chart for secondary data.");

        InOLSLookback.Name = "2. OLS Lookback (bars)";
        InOLSLookback.SetInt(3300);
        InOLSLookback.SetIntLimits(100, 20000);
        InOLSLookback.SetDescription("OLS regression window. Config E: 3300 (~12.5 days).");

        InZScoreWindow.Name = "3. Z-Score Window (bars)";
        InZScoreWindow.SetInt(30);
        InZScoreWindow.SetIntLimits(5, 500);
        InZScoreWindow.SetDescription("Rolling window for Z-score mean and stddev. Config E: 30.");

        InBandMult.Name = "4. Band Std Dev Multiplier";
        InBandMult.SetFloat(2.0f);
        InBandMult.SetFloatLimits(1.0f, 5.0f);
        InBandMult.SetDescription("Multiplier for spread bands.");

        InZEntry.Name = "5. Z-Score Entry Threshold";
        InZEntry.SetFloat(3.15f);
        InZEntry.SetFloatLimits(1.0f, 6.0f);
        InZEntry.SetDescription("Z-score entry threshold (symmetric). Config E: 3.15.");

        InZExit.Name = "6. Z-Score Exit Threshold";
        InZExit.SetFloat(1.00f);
        InZExit.SetFloatLimits(0.0f, 3.0f);
        InZExit.SetDescription("Z-score exit threshold (symmetric). Config E: 1.00.");

        InZStop.Name = "7. Z-Score Stop Threshold";
        InZStop.SetFloat(4.50f);
        InZStop.SetFloatLimits(2.0f, 8.0f);
        InZStop.SetDescription("Z-score stop-loss threshold (symmetric). Config E: 4.50.");

        InADFWindow.Name = "8. ADF Window (bars)";
        InADFWindow.SetInt(12);
        InADFWindow.SetIntLimits(5, 500);
        InADFWindow.SetDescription("Window for ADF test. Profile tres_court: 12.");

        InHurstWindow.Name = "9. Hurst Window (bars)";
        InHurstWindow.SetInt(64);
        InHurstWindow.SetIntLimits(8, 500);
        InHurstWindow.SetDescription("Window for Hurst exponent (variance-ratio). Profile tres_court: 64.");

        InHLWindow.Name = "10. Half-Life Window (bars)";
        InHLWindow.SetInt(12);
        InHLWindow.SetIntLimits(3, 500);
        InHLWindow.SetDescription("Window for half-life calculation. Profile tres_court: 12.");

        InCorrWindow.Name = "11. Correlation Window (bars)";
        InCorrWindow.SetInt(6);
        InCorrWindow.SetIntLimits(3, 500);
        InCorrWindow.SetDescription("Window for Pearson correlation on log-prices. Profile tres_court: 6.");

        InMinConfidence.Name = "12. Min Confidence (0-100)";
        InMinConfidence.SetInt(67);
        InMinConfidence.SetIntLimits(0, 100);
        InMinConfidence.SetDescription("Minimum confidence score for signal generation. Config E: 67.");

        InADFGate.Name = "13. ADF Gate Threshold";
        InADFGate.SetFloat(-1.00f);
        InADFGate.SetFloatLimits(-5.0f, 0.0f);
        InADFGate.SetDescription("ADF gate: stat >= this value -> confidence = 0%. Default: -1.00.");

        InSessionStart.Name = "14. Session Start (CT)";
        InSessionStart.SetTime(HMS_TIME(17, 30, 0));
        InSessionStart.SetDescription("Session start time (Chicago Time). Default: 17:30.");

        InSessionEnd.Name = "15. Session End (CT)";
        InSessionEnd.SetTime(HMS_TIME(15, 30, 0));
        InSessionEnd.SetDescription("Session end time (Chicago Time). Default: 15:30.");

        InEntryStart.Name = "16. Entry Window Start (CT)";
        InEntryStart.SetTime(HMS_TIME(2, 0, 0));
        InEntryStart.SetDescription("Entry window start (Chicago Time). Config E: 02:00.");

        InEntryEnd.Name = "17. Entry Window End (CT)";
        InEntryEnd.SetTime(HMS_TIME(14, 0, 0));
        InEntryEnd.SetDescription("Entry window end (Chicago Time). Config E: 14:00.");

        InFlatTime.Name = "18. Flat Time (CT)";
        InFlatTime.SetTime(HMS_TIME(15, 30, 0));
        InFlatTime.SetDescription("Force flat at this time. Config E: 15:30.");

        InKalmanAlpha.Name = "19. Kalman Alpha Ratio";
        InKalmanAlpha.SetFloat(3e-7f);
        InKalmanAlpha.SetFloatLimits(1e-9f, 1e-3f);
        InKalmanAlpha.SetDescription("Kalman Q = alpha_ratio * R * I. K_Balanced: 3e-7.");

        InKalmanWarmup.Name = "20. Kalman Warmup (bars)";
        InKalmanWarmup.SetInt(100);
        InKalmanWarmup.SetIntLimits(10, 1000);
        InKalmanWarmup.SetDescription("Kalman filter warmup period (bars).");

        InGapPMult.Name = "21. Kalman Gap P Multiplier";
        InGapPMult.SetFloat(10.0f);
        InGapPMult.SetFloatLimits(1.0f, 100.0f);
        InGapPMult.SetDescription("Multiply P by this when session gap > 30min. Default: 10.0.");

        InShowTextBox.Name = "22. Show Info Text Box";
        InShowTextBox.SetYesNo(1);
        InShowTextBox.SetDescription("Show the information textbox overlay.");

        // ================================================================
        // TRADING INPUTS (Phase 2b)
        // ================================================================

        InTradingAction.Name = "23. Trading: Action";
        InTradingAction.SetCustomInputStrings("Off;BUY SPREAD;SELL SPREAD;FLATTEN");
        InTradingAction.SetCustomInputIndex(0);
        InTradingAction.SetDescription("Off=idle, BUY=Long spread, SELL=Short spread, FLATTEN=close all.");

        InLeg1Symbol.Name = "24. Trading: Leg 1 Symbol";
        InLeg1Symbol.SetString("MNQH26");
        InLeg1Symbol.SetDescription("Symbol for leg 1 (NQ side). Change for rollover.");

        InLeg2Symbol.Name = "25. Trading: Leg 2 Symbol";
        InLeg2Symbol.SetString("MYMH26");
        InLeg2Symbol.SetDescription("Symbol for leg 2 (YM side). Change for rollover.");

        InLeg1Qty.Name = "26. Trading: Leg 1 Qty";
        InLeg1Qty.SetInt(2);
        InLeg1Qty.SetIntLimits(1, 50);
        InLeg1Qty.SetDescription("Number of contracts for leg 1 (NQ/MNQ).");

        InLeg2Qty.Name = "27. Trading: Leg 2 Qty";
        InLeg2Qty.SetInt(4);
        InLeg2Qty.SetIntLimits(1, 100);
        InLeg2Qty.SetDescription("Number of contracts for leg 2 (YM/MYM).");

        InTradingZExit.Name = "28. Trading: Z Exit";
        InTradingZExit.SetFloat(1.00f);
        InTradingZExit.SetFloatLimits(0.0f, 5.0f);
        InTradingZExit.SetDescription("Z-score threshold for auto-exit. 0 = disabled.");

        InDollarStop.Name = "29. Trading: Dollar Stop ($)";
        InDollarStop.SetFloat(500.0f);
        InDollarStop.SetFloatLimits(0.0f, 10000.0f);
        InDollarStop.SetDescription("Max loss in $ before auto-flatten. 0 = disabled.");

        InTimeStop.Name = "30. Trading: Time Stop (CT)";
        InTimeStop.SetTime(HMS_TIME(15, 30, 0));
        InTimeStop.SetDescription("Force flatten at this time (Chicago Time).");

        InEnableAutoExit.Name = "31. Trading: Enable Auto Exit";
        InEnableAutoExit.SetYesNo(1);
        InEnableAutoExit.SetDescription("Enable automatic exits (z-exit, dollar stop, time stop).");

        return;
    }

    // ========================================================================
    // RECUPERATION DES PARAMETRES
    // ========================================================================

    int YMChartNumber = InYMChartNumber.GetInt();
    int OLSLookback = InOLSLookback.GetInt();
    int ZScoreWin = InZScoreWindow.GetInt();
    float BandMultVal = InBandMult.GetFloat();
    float zEntry = InZEntry.GetFloat();
    float zExit = InZExit.GetFloat();
    float zStop = InZStop.GetFloat();
    int ADFWindow = InADFWindow.GetInt();
    int HurstWindow = InHurstWindow.GetInt();
    int HLWindow = InHLWindow.GetInt();
    int CorrWindow = InCorrWindow.GetInt();
    int MinConf = InMinConfidence.GetInt();
    float ADFGateVal = InADFGate.GetFloat();
    int SessionStartSecs = InSessionStart.GetTime();
    int SessionEndSecs = InSessionEnd.GetTime();
    int EntryStartSecs = InEntryStart.GetTime();
    int EntryEndSecs = InEntryEnd.GetTime();
    int FlatTimeSecs = InFlatTime.GetTime();
    float KalmanAlphaRatio = InKalmanAlpha.GetFloat();
    int KalmanWarmupBars = InKalmanWarmup.GetInt();
    float GapPMultVal = InGapPMult.GetFloat();

    // ========================================================================
    // PERSISTENT VARIABLES - Kalman filter state
    // ========================================================================

    double& K_theta0 = sc.GetPersistentDouble(1);   // Kalman alpha (intercept)
    double& K_theta1 = sc.GetPersistentDouble(2);   // Kalman beta (slope)
    double& K_P00 = sc.GetPersistentDouble(3);
    double& K_P01 = sc.GetPersistentDouble(4);
    double& K_P10 = sc.GetPersistentDouble(5);
    double& K_P11 = sc.GetPersistentDouble(6);
    double& K_R = sc.GetPersistentDouble(7);
    int& K_Initialized = sc.GetPersistentInt(1);
    int& K_WarmupCount = sc.GetPersistentInt(2);

    // State machine
    int& TradeState = sc.GetPersistentInt(3);

    // Trading state (Phase 2b)
    int& TradingPosition = sc.GetPersistentInt(4);  // 0=FLAT, 1=LONG_SPREAD, -1=SHORT_SPREAD
    int& EntryBarIndex = sc.GetPersistentInt(5);
    int& PendingOrderAction = sc.GetPersistentInt(6);  // 0=none, 1=buy, 2=sell, 3=flatten
    double& EntrySpreadZ = sc.GetPersistentDouble(8);
    double& EntryTotalPnL = sc.GetPersistentDouble(9);  // P&L baseline at entry

    // ========================================================================
    // INITIALISATION (full recalc)
    // ========================================================================

    if (sc.IsFullRecalculation && sc.Index == 0)
    {
        // Kalman state reset
        K_theta0 = 0.0;
        K_theta1 = 1.0;
        K_P00 = 1.0; K_P01 = 0.0; K_P10 = 0.0; K_P11 = 1.0;
        K_R = 1e-5;
        K_Initialized = 0;
        K_WarmupCount = 0;

        // State machine reset
        TradeState = STATE_FLAT;

        // Trading state: do NOT reset on full recalc (preserves live position)
        // TradingPosition, EntryBarIndex, EntrySpreadZ, EntryTotalPnL kept as-is
    }

    // ========================================================================
    // RECUPERATION DES DONNEES YM
    // ========================================================================

    SCGraphData YMData;
    sc.GetChartBaseData(YMChartNumber, YMData);

    if (YMData[SC_LAST].GetArraySize() == 0)
    {
        if (sc.Index == 0)
            sc.AddMessageToLog("ERROR: YM chart data not available. Check chart number.", 1);
        return;
    }

    int YMIdx = sc.GetContainingIndexForDateTimeIndex(YMChartNumber, sc.Index);

    if (YMIdx < 0 || YMIdx >= YMData[SC_LAST].GetArraySize())
        return;

    float NQClose = sc.Close[sc.Index];
    float YMClose = YMData[SC_LAST][YMIdx];

    if (NQClose <= 0.0f || YMClose <= 0.0f)
        return;

    // ========================================================================
    // LOG PRICES (store in subgraph extra arrays for reuse)
    // ========================================================================

    // Arrays[0] of OLSBetaSG -> log(NQ)  (leg A, dependent)
    // Arrays[0] of OLSAlphaSG -> log(YM) (leg B, explanatory)
    OLSBetaSG.Arrays[0][sc.Index] = (float)log((double)NQClose);
    OLSAlphaSG.Arrays[0][sc.Index] = (float)log((double)YMClose);

    SCFloatArrayRef LogNQ = OLSBetaSG.Arrays[0];
    SCFloatArrayRef LogYM = OLSAlphaSG.Arrays[0];

    // ========================================================================
    // KALMAN FILTER (independent, runs from bar 999+)
    // Matches Python: R from OLS on first 1000 bars, theta=[0,1], P=I
    // Runs BEFORE OLS section so Kalman values exist even for pre-OLS bars
    // ========================================================================

    // R initialization from OLS on first 1000 bars (Python parity)
    int kalmanRBars = 1000;
    if (K_Initialized == 0 && sc.Index >= kalmanRBars - 1)
    {
        // Quick OLS on first 1000 bars for R estimation
        // Convention: log(NQ) = alpha + beta * log(YM) + epsilon
        float initAlpha = 0.0f;
        float initBeta = CalculateOLSBeta(LogYM, LogNQ, kalmanRBars - 1, kalmanRBars, initAlpha);

        // Compute residual variance (population variance like Python np.var)
        double sumR = 0.0, sumR2 = 0.0;
        for (int i = 0; i < kalmanRBars; i++)
        {
            double r = (double)LogNQ[i] - (double)initAlpha - (double)initBeta * (double)LogYM[i];
            sumR += r;
            sumR2 += r * r;
        }
        double meanR = sumR / kalmanRBars;
        K_R = (sumR2 / kalmanRBars) - meanR * meanR;  // population variance (ddof=0)
        if (K_R < 1e-8) K_R = 1e-5;

        // Initial state: theta from OLS (pre-converged)
        // Python runs 999 extra bars from bar 0; C++ starts at bar 999.
        // Using OLS estimate avoids slow convergence with tiny Q.
        // Model: log_nq = theta0 + theta1 * log_ym, H = [1, log_ym]
        K_theta0 = (double)initAlpha;
        K_theta1 = (double)initBeta;
        K_P00 = 1.0;  K_P01 = 0.0;
        K_P10 = 0.0; K_P11 = 1.0;
        K_Initialized = 1;
    }

    // Kalman update (every bar after initialization) — all in double precision
    float kalZInn = 0.0f;

    if (K_Initialized == 1)
    {
        // Full double precision (bypass float storage for Kalman)
        double log_nq = log((double)NQClose);
        double log_ym = log((double)YMClose);

        double qScalar = (double)KalmanAlphaRatio * K_R;

        // Gap detection: absolute datetime difference (handles weekends/holidays)
        if (sc.Index > 0)
        {
            SCDateTime prevDT = sc.BaseDateTimeIn[sc.Index - 1];
            SCDateTime currDT = sc.BaseDateTimeIn[sc.Index];
            int64_t prevAbs = (int64_t)prevDT.GetDate() * 86400 + prevDT.GetTimeInSeconds();
            int64_t currAbs = (int64_t)currDT.GetDate() * 86400 + currDT.GetTimeInSeconds();
            int gapSeconds = (currAbs > prevAbs) ? (int)(currAbs - prevAbs) : 0;
            if (gapSeconds > 1800)  // > 30 minutes
            {
                double gapMult = (double)GapPMultVal;
                K_P00 *= gapMult;
                K_P01 *= gapMult;
                K_P10 *= gapMult;
                K_P11 *= gapMult;
            }
        }

        // Predict: P += q * I
        K_P00 += qScalar;
        K_P11 += qScalar;

        // H = [1, log_ym] (matches Python, no centering)
        double H0 = 1.0;
        double H1 = log_ym;

        // Innovation
        double y_pred = K_theta0 * H0 + K_theta1 * H1;
        double nu = log_nq - y_pred;

        // F = H @ P @ H^T + R
        double F = H0 * (K_P00 * H0 + K_P01 * H1)
                 + H1 * (K_P10 * H0 + K_P11 * H1) + K_R;

        // Z-score (innovation-based)
        kalZInn = (F > 1e-12) ? (float)(nu / sqrt(F)) : 0.0f;

        // Kalman gain: K = P @ H^T / F
        double KG0 = 0.0, KG1 = 0.0;
        if (F > 1e-12)
        {
            KG0 = (K_P00 * H0 + K_P01 * H1) / F;
            KG1 = (K_P10 * H0 + K_P11 * H1) / F;
        }

        // State update: theta += K * nu
        K_theta0 += KG0 * nu;
        K_theta1 += KG1 * nu;

        // Covariance update (Joseph form): P = (I-KH) P (I-KH)^T + K R K^T
        double IKH00 = 1.0 - KG0 * H0;
        double IKH01 = -KG0 * H1;
        double IKH10 = -KG1 * H0;
        double IKH11 = 1.0 - KG1 * H1;

        // Save old P for Joseph form (need original P)
        double oldP00 = K_P00, oldP01 = K_P01;
        double oldP10 = K_P10, oldP11 = K_P11;

        // Temp = (I-KH) @ P_old
        double T00 = IKH00 * oldP00 + IKH01 * oldP10;
        double T01 = IKH00 * oldP01 + IKH01 * oldP11;
        double T10 = IKH10 * oldP00 + IKH11 * oldP10;
        double T11 = IKH10 * oldP01 + IKH11 * oldP11;

        // P = Temp @ (I-KH)^T + K*K^T*R
        K_P00 = T00 * IKH00 + T01 * IKH01 + KG0 * KG0 * K_R;
        K_P01 = T00 * IKH10 + T01 * IKH11 + KG0 * KG1 * K_R;
        K_P10 = T10 * IKH00 + T11 * IKH01 + KG1 * KG0 * K_R;
        K_P11 = T10 * IKH10 + T11 * IKH11 + KG1 * KG1 * K_R;

        K_WarmupCount++;

        // Store in subgraphs (only after warmup)
        if (K_WarmupCount >= KalmanWarmupBars)
        {
            KalmanBeta[sc.Index] = (float)K_theta1;
            KalmanZInn[sc.Index] = kalZInn;
            // Confidence not yet computed (OLS section below), set 0 for now
            // Will be updated after OLS confidence computation for bars >= OLSLookback-1
            KalmanConf[sc.Index] = 0.0f;
        }
        else
        {
            KalmanBeta[sc.Index] = 0.0f;
            KalmanZInn[sc.Index] = 0.0f;
            KalmanConf[sc.Index] = 0.0f;
        }
    }
    else
    {
        KalmanBeta[sc.Index] = 0.0f;
        KalmanZInn[sc.Index] = 0.0f;
        KalmanConf[sc.Index] = 0.0f;
    }

    // ========================================================================
    // OLS REGRESSION: log(NQ) = alpha + beta * log(YM) + epsilon
    // X = log(YM), Y = log(NQ)
    // ========================================================================

    float alpha = 0.0f;
    float beta = 0.0f;

    // Fixed window only — no expanding warmup (Python parity)
    // Pre-OLS bars: zero OLS subgraphs only (Kalman already set above)
    if (sc.Index < OLSLookback - 1)
    {
        Spread[sc.Index] = 0.0f;
        SpreadMean[sc.Index] = 0.0f;
        SpreadStdDev[sc.Index] = 0.0f;
        ZScore[sc.Index] = 0.0f;
        UpperBand[sc.Index] = 0.0f;
        LowerBand[sc.Index] = 0.0f;
        ZEntryPlus[sc.Index] = 0.0f;
        ZEntryMinus[sc.Index] = 0.0f;
        ZZeroLine[sc.Index] = 0.0f;
        ADFStat[sc.Index] = 0.0f;
        ADFCritical[sc.Index] = 0.0f;
        HurstExp[sc.Index] = 0.0f;
        HalfLifeLine[sc.Index] = 0.0f;
        CorrLine[sc.Index] = 0.0f;
        ConfScore[sc.Index] = 0.0f;
        OLSBetaSG[sc.Index] = 0.0f;
        OLSAlphaSG[sc.Index] = 0.0f;
        TradeStateSG[sc.Index] = 0.0f;
        // KalmanBeta, KalmanZInn, KalmanConf already set by Kalman section above
        return;
    }

    beta = CalculateOLSBeta(LogYM, LogNQ, sc.Index, OLSLookback, alpha);

    OLSBetaSG[sc.Index] = beta;
    OLSAlphaSG[sc.Index] = alpha;

    // ========================================================================
    // SPREAD = log(NQ) - beta * log(YM) - alpha
    // ========================================================================

    float spreadVal = LogNQ[sc.Index] - beta * LogYM[sc.Index] - alpha;
    Spread[sc.Index] = spreadVal;

    // ========================================================================
    // Z-SCORE (rolling mean & stddev)
    // ========================================================================

    sc.SimpleMovAvg(Spread, SpreadMean, ZScoreWin);

    float stdDev = CalculateStdDev(Spread, sc.Index, ZScoreWin);
    SpreadStdDev[sc.Index] = stdDev;

    float zScore = 0.0f;
    if (stdDev > 1e-10f)
        zScore = (spreadVal - SpreadMean[sc.Index]) / stdDev;

    ZScore[sc.Index] = zScore;

    // Bands
    UpperBand[sc.Index] = SpreadMean[sc.Index] + BandMultVal * stdDev;
    LowerBand[sc.Index] = SpreadMean[sc.Index] - BandMultVal * stdDev;

    // Threshold lines (constant)
    ZEntryPlus[sc.Index] = zEntry;
    ZEntryMinus[sc.Index] = -zEntry;
    // (Z Exit, Z Stop lines removed)
    ZZeroLine[sc.Index] = 0.0f;

    // ========================================================================
    // STATISTICS
    // ========================================================================

    float adfStat = CalculateADFSimple(Spread, sc.Index, ADFWindow);
    ADFStat[sc.Index] = adfStat;
    ADFCritical[sc.Index] = ADF_CRITICAL_5PCT;

    float hurst = CalculateHurstVR(Spread, sc.Index, HurstWindow);
    HurstExp[sc.Index] = hurst;

    float halfLife = CalculateHalfLife(Spread, sc.Index, HLWindow);
    HalfLifeLine[sc.Index] = halfLife;

    float correlation = CalculateCorrelation(LogYM, LogNQ, sc.Index, CorrWindow);
    CorrLine[sc.Index] = correlation;

    // ========================================================================
    // CONFIDENCE SCORING
    // ========================================================================

    float confidence = CalculateConfidence(adfStat, hurst, correlation,
                                           halfLife, ADFGateVal);
    ConfScore[sc.Index] = confidence;

    // Update KalmanConf with OLS confidence (now that it's computed)
    if (K_Initialized == 1 && K_WarmupCount >= KalmanWarmupBars)
        KalmanConf[sc.Index] = confidence;

    // ========================================================================
    // STATE MACHINE (4 states: FLAT -> LONG/SHORT -> COOLDOWN -> FLAT)
    // ========================================================================

    // Time checks
    SCDateTime barDT = sc.BaseDateTimeIn[sc.Index];
    int barTimeSecs = barDT.GetTimeInSeconds();

    // Session check (overnight OR logic: 17:30 -> 15:30 CT)
    bool inSession;
    if (SessionStartSecs > SessionEndSecs)
        inSession = (barTimeSecs >= SessionStartSecs || barTimeSecs < SessionEndSecs);
    else
        inSession = (barTimeSecs >= SessionStartSecs && barTimeSecs < SessionEndSecs);

    // Entry window check
    bool inEntryWindow;
    if (EntryStartSecs < EntryEndSecs)
        inEntryWindow = (barTimeSecs >= EntryStartSecs && barTimeSecs < EntryEndSecs);
    else  // wraps midnight
        inEntryWindow = (barTimeSecs >= EntryStartSecs || barTimeSecs < EntryEndSecs);

    // Flat time check
    // For 15:30 CT flat with overnight session (17:30-15:30):
    //   After midnight (barTimeSecs < SessionStartSecs): flat if barTimeSecs >= FlatTimeSecs
    //   Before midnight (barTimeSecs >= SessionStartSecs): never flat (still early in session)
    bool mustFlat = false;
    if (SessionStartSecs > SessionEndSecs)
    {
        // Overnight session
        if (barTimeSecs < SessionStartSecs && barTimeSecs >= FlatTimeSecs)
            mustFlat = true;
    }
    else
    {
        // Daytime session
        if (barTimeSecs >= FlatTimeSecs)
            mustFlat = true;
    }

    float minConf = (float)MinConf;

    // Reset signals
    SignalLong[sc.Index] = 0.0f;
    SignalShort[sc.Index] = 0.0f;
    SignalExit[sc.Index] = 0.0f;

    // NaN check on z-score
    bool zScoreValid = (zScore == zScore);  // false if NaN

    if (!zScoreValid)
    {
        TradeState = STATE_FLAT;
        TradeStateSG[sc.Index] = (float)TradeState;
        return;
    }

    int prevState = TradeState;

    // State transitions
    switch (TradeState)
    {
    case STATE_FLAT:
        if (inSession && inEntryWindow && !mustFlat && confidence >= minConf)
        {
            if (zScore < -zEntry)
            {
                TradeState = STATE_LONG;
                SignalLong[sc.Index] = LowerBand[sc.Index];
            }
            else if (zScore > zEntry)
            {
                TradeState = STATE_SHORT;
                SignalShort[sc.Index] = UpperBand[sc.Index];
            }
        }
        break;

    case STATE_LONG:
        if (mustFlat || !inSession)
        {
            TradeState = STATE_FLAT;
            SignalExit[sc.Index] = Spread[sc.Index];
        }
        else if (zScore < -zStop)
        {
            TradeState = STATE_COOLDOWN;  // SL hit
            SignalExit[sc.Index] = Spread[sc.Index];
        }
        else if (zScore > -zExit)
        {
            TradeState = STATE_FLAT;      // TP (mean reversion complete)
            SignalExit[sc.Index] = Spread[sc.Index];
        }
        break;

    case STATE_SHORT:
        if (mustFlat || !inSession)
        {
            TradeState = STATE_FLAT;
            SignalExit[sc.Index] = Spread[sc.Index];
        }
        else if (zScore > zStop)
        {
            TradeState = STATE_COOLDOWN;  // SL hit
            SignalExit[sc.Index] = Spread[sc.Index];
        }
        else if (zScore < zExit)
        {
            TradeState = STATE_FLAT;      // TP (mean reversion complete)
            SignalExit[sc.Index] = Spread[sc.Index];
        }
        break;

    case STATE_COOLDOWN:
        if (mustFlat || !inSession)
            TradeState = STATE_FLAT;
        else if (fabs(zScore) < zExit)
            TradeState = STATE_FLAT;      // Spread returned to neutral
        break;
    }

    TradeStateSG[sc.Index] = (float)TradeState;

    // ========================================================================
    // PHASE 2b — TRADING LOGIC (last bar only)
    // ========================================================================
    //
    // Architecture: Changing an Input triggers a full recalc in Sierra.
    // Orders submitted during full recalc return -8998 (SCT_SKIPPED_FULL_RECALC).
    // Solution: capture the action into PendingOrderAction during full recalc,
    // then execute the order on the next normal (non-recalc) tick update.
    //
    // PendingOrderAction: 0=none, 1=buy_spread, 2=sell_spread, 3=flatten

    if (sc.Index == sc.ArraySize - 1)
    {
        // --- Step 1: Capture Input action (works during full recalc too) ---
        int tradingAction = InTradingAction.GetIndex();
        if (tradingAction != 0)
        {
            PendingOrderAction = tradingAction;
            InTradingAction.SetCustomInputIndex(0);  // Reset dropdown to Off
        }

        // --- Step 2: Execute orders only when NOT in full recalc ---
        if (!sc.IsFullRecalculation)
        {
            SCString leg1Sym = InLeg1Symbol.GetString();
            SCString leg2Sym = InLeg2Symbol.GetString();
            int leg1Qty = InLeg1Qty.GetInt();
            int leg2Qty = InLeg2Qty.GetInt();
            float tradingZExit = InTradingZExit.GetFloat();
            float dollarStop = InDollarStop.GetFloat();
            int timeStopSecs = InTimeStop.GetTime();
            bool autoExitOn = (InEnableAutoExit.GetYesNo() != 0);

            bool leg1Valid = (leg1Sym.GetLength() > 0);
            bool leg2Valid = (leg2Sym.GetLength() > 0);

            // --- Position tracking ---
            s_SCPositionData Leg1Pos, Leg2Pos;
            double totalPnL = 0.0;

            if (TradingPosition != 0 && leg1Valid && leg2Valid)
            {
                sc.GetTradePositionForSymbolAndAccount(Leg1Pos, leg1Sym, sc.SelectedTradeAccount);
                sc.GetTradePositionForSymbolAndAccount(Leg2Pos, leg2Sym, sc.SelectedTradeAccount);
                totalPnL = Leg1Pos.OpenProfitLoss + Leg2Pos.OpenProfitLoss;
            }

            // --- Execute pending order ---
            if (PendingOrderAction != 0 && leg1Valid && leg2Valid)
            {
                SCString logMsg;

                // --- BUY SPREAD (Long NQ, Short YM) ---
                if (PendingOrderAction == 1 && TradingPosition == 0)
                {
                    s_SCNewOrder BuyLeg1;
                    BuyLeg1.OrderQuantity = leg1Qty;
                    BuyLeg1.Price1 = 0;
                    BuyLeg1.OrderType = SCT_ORDERTYPE_MARKET;
                    BuyLeg1.Symbol = leg1Sym;
                    BuyLeg1.TextTag = "SpreadBuyNQ";
                    int ret1 = sc.BuyOrder(BuyLeg1);

                    s_SCNewOrder SellLeg2;
                    SellLeg2.OrderQuantity = leg2Qty;
                    SellLeg2.Price1 = 0;
                    SellLeg2.OrderType = SCT_ORDERTYPE_MARKET;
                    SellLeg2.Symbol = leg2Sym;
                    SellLeg2.TextTag = "SpreadSellYM";
                    int ret2 = sc.SellOrder(SellLeg2);

                    logMsg.Format("BUY SPREAD: Leg1(%s) ret=%d | Leg2(%s) ret=%d",
                        leg1Sym.GetChars(), ret1, leg2Sym.GetChars(), ret2);
                    sc.AddMessageToLog(logMsg, 0);

                    if (ret1 > 0 || ret2 > 0)
                    {
                        TradingPosition = 1;
                        EntryBarIndex = sc.Index;
                        EntrySpreadZ = (double)zScore;
                        EntryTotalPnL = 0.0;
                    }
                }

                // --- SELL SPREAD (Short NQ, Long YM) ---
                else if (PendingOrderAction == 2 && TradingPosition == 0)
                {
                    s_SCNewOrder SellLeg1;
                    SellLeg1.OrderQuantity = leg1Qty;
                    SellLeg1.Price1 = 0;
                    SellLeg1.OrderType = SCT_ORDERTYPE_MARKET;
                    SellLeg1.Symbol = leg1Sym;
                    SellLeg1.TextTag = "SpreadSellNQ";
                    int ret1 = sc.SellOrder(SellLeg1);

                    s_SCNewOrder BuyLeg2;
                    BuyLeg2.OrderQuantity = leg2Qty;
                    BuyLeg2.Price1 = 0;
                    BuyLeg2.OrderType = SCT_ORDERTYPE_MARKET;
                    BuyLeg2.Symbol = leg2Sym;
                    BuyLeg2.TextTag = "SpreadBuyYM";
                    int ret2 = sc.BuyOrder(BuyLeg2);

                    logMsg.Format("SELL SPREAD: Leg1(%s) ret=%d | Leg2(%s) ret=%d",
                        leg1Sym.GetChars(), ret1, leg2Sym.GetChars(), ret2);
                    sc.AddMessageToLog(logMsg, 0);

                    if (ret1 > 0 || ret2 > 0)
                    {
                        TradingPosition = -1;
                        EntryBarIndex = sc.Index;
                        EntrySpreadZ = (double)zScore;
                        EntryTotalPnL = 0.0;
                    }
                }

                // --- FLATTEN (manual exit) ---
                else if (PendingOrderAction == 3 && TradingPosition != 0)
                {
                    s_SCNewOrder CloseLeg1;
                    CloseLeg1.OrderQuantity = leg1Qty;
                    CloseLeg1.Price1 = 0;
                    CloseLeg1.OrderType = SCT_ORDERTYPE_MARKET;
                    CloseLeg1.Symbol = leg1Sym;
                    CloseLeg1.TextTag = "SpreadFlatNQ";
                    // Was LONG = bought NQ -> sell. Was SHORT = sold NQ -> buy.
                    int ret1;
                    if (TradingPosition == 1)
                        ret1 = sc.SellOrder(CloseLeg1);
                    else
                        ret1 = sc.BuyOrder(CloseLeg1);

                    s_SCNewOrder CloseLeg2;
                    CloseLeg2.OrderQuantity = leg2Qty;
                    CloseLeg2.Price1 = 0;
                    CloseLeg2.OrderType = SCT_ORDERTYPE_MARKET;
                    CloseLeg2.Symbol = leg2Sym;
                    CloseLeg2.TextTag = "SpreadFlatYM";
                    // Was LONG = sold YM -> buy back. Was SHORT = bought YM -> sell.
                    int ret2;
                    if (TradingPosition == 1)
                        ret2 = sc.BuyOrder(CloseLeg2);
                    else
                        ret2 = sc.SellOrder(CloseLeg2);

                    logMsg.Format("FLATTEN: Leg1 ret=%d | Leg2 ret=%d | P&L=$%.0f",
                        ret1, ret2, totalPnL);
                    sc.AddMessageToLog(logMsg, 0);

                    TradingPosition = 0;
                    EntryBarIndex = 0;
                    EntrySpreadZ = 0.0;
                    EntryTotalPnL = 0.0;
                }

                PendingOrderAction = 0;  // Clear pending flag
            }

            // --- AUTO-EXITS ---
            if (autoExitOn && TradingPosition != 0 && leg1Valid && leg2Valid)
            {
                bool shouldFlatten = false;
                SCString exitReason;

                // A. Z-Exit
                if (tradingZExit > 0.0f)
                {
                    if (TradingPosition == 1 && zScore >= -tradingZExit)
                    {
                        shouldFlatten = true;
                        exitReason.Format("Z-EXIT: z=%.2f crossed -%.2f", zScore, tradingZExit);
                    }
                    else if (TradingPosition == -1 && zScore <= tradingZExit)
                    {
                        shouldFlatten = true;
                        exitReason.Format("Z-EXIT: z=%.2f crossed +%.2f", zScore, tradingZExit);
                    }
                }

                // B. Dollar Stop
                if (!shouldFlatten && dollarStop > 0.0f)
                {
                    if (totalPnL <= -dollarStop)
                    {
                        shouldFlatten = true;
                        exitReason.Format("DOLLAR STOP: P&L=$%.0f hit -$%.0f", totalPnL, dollarStop);
                    }
                }

                // C. Time Stop
                if (!shouldFlatten)
                {
                    bool timeToFlat = false;
                    if (SessionStartSecs > SessionEndSecs)
                    {
                        if (barTimeSecs < SessionStartSecs && barTimeSecs >= timeStopSecs)
                            timeToFlat = true;
                    }
                    else
                    {
                        if (barTimeSecs >= timeStopSecs)
                            timeToFlat = true;
                    }
                    if (timeToFlat)
                    {
                        shouldFlatten = true;
                        exitReason = "TIME STOP: flat time reached";
                    }
                }

                // Execute auto-flatten
                if (shouldFlatten)
                {
                    s_SCNewOrder CloseLeg1;
                    CloseLeg1.OrderQuantity = leg1Qty;
                    CloseLeg1.Price1 = 0;
                    CloseLeg1.OrderType = SCT_ORDERTYPE_MARKET;
                    CloseLeg1.Symbol = leg1Sym;
                    CloseLeg1.TextTag = "AutoFlatNQ";
                    if (TradingPosition == 1)
                        sc.SellOrder(CloseLeg1);
                    else
                        sc.BuyOrder(CloseLeg1);

                    s_SCNewOrder CloseLeg2;
                    CloseLeg2.OrderQuantity = leg2Qty;
                    CloseLeg2.Price1 = 0;
                    CloseLeg2.OrderType = SCT_ORDERTYPE_MARKET;
                    CloseLeg2.Symbol = leg2Sym;
                    CloseLeg2.TextTag = "AutoFlatYM";
                    if (TradingPosition == 1)
                        sc.BuyOrder(CloseLeg2);
                    else
                        sc.SellOrder(CloseLeg2);

                    SCString autoMsg;
                    autoMsg.Format("AUTO-EXIT: %s. P&L: $%.0f",
                                   exitReason.GetChars(), totalPnL);
                    sc.AddMessageToLog(autoMsg, 0);

                    TradingPosition = 0;
                    EntryBarIndex = 0;
                    EntrySpreadZ = 0.0;
                    EntryTotalPnL = 0.0;
                }
            }
        }  // end !IsFullRecalculation
    }  // end trading logic (last bar only)

    // ========================================================================
    // DYNAMIC COLORING (per-bar via DataColor)
    // ========================================================================

    // Z-Score dynamic color
    if (fabs(zScore) >= zStop)
        ZScore.DataColor[sc.Index] = RGB(255, 50, 50);       // Red (danger)
    else if (fabs(zScore) >= zEntry)
        ZScore.DataColor[sc.Index] = RGB(0, 255, 0);          // Green (signal zone)
    else if (fabs(zScore) < zExit)
        ZScore.DataColor[sc.Index] = RGB(180, 180, 180);      // Gray (neutral)
    else
        ZScore.DataColor[sc.Index] = RGB(0, 200, 255);        // Cyan (default)

    // ADF dynamic color
    if (adfStat < ADF_CRITICAL_5PCT)
        ADFStat.DataColor[sc.Index] = RGB(0, 255, 0);         // Green (stationary)
    else if (adfStat < -1.00f)
        ADFStat.DataColor[sc.Index] = RGB(255, 165, 0);       // Orange (marginal)
    else
        ADFStat.DataColor[sc.Index] = RGB(255, 0, 0);         // Red (gate active)

    // Hurst dynamic color
    if (hurst < 0.45f)
        HurstExp.DataColor[sc.Index] = RGB(0, 255, 0);        // Green
    else if (hurst < 0.50f)
        HurstExp.DataColor[sc.Index] = RGB(255, 165, 0);      // Orange
    else
        HurstExp.DataColor[sc.Index] = RGB(255, 0, 0);        // Red

    // Correlation dynamic color
    if (correlation >= 0.85f)
        CorrLine.DataColor[sc.Index] = RGB(0, 255, 0);        // Green
    else if (correlation >= 0.65f)
        CorrLine.DataColor[sc.Index] = RGB(255, 165, 0);      // Orange
    else
        CorrLine.DataColor[sc.Index] = RGB(255, 0, 0);        // Red

    // Confidence BAR dynamic color
    if (confidence >= (float)MinConf)
        ConfScore.DataColor[sc.Index] = RGB(0, 255, 0);       // Green
    else if (confidence >= 50.0f)
        ConfScore.DataColor[sc.Index] = RGB(255, 165, 0);     // Orange
    else
        ConfScore.DataColor[sc.Index] = RGB(255, 0, 0);       // Red

    // ========================================================================
    // TEXTBOX — 2 PANELS (last bar only)
    // Panel 1: SIGNAL (large, bold, colored) — execution decision
    // Panel 2: DASHBOARD (compact, dark) — detailed metrics
    // ========================================================================

    if (InShowTextBox.GetYesNo() && sc.Index == sc.ArraySize - 1)
    {
        // --- Kalman data ---
        float kalBeta = KalmanBeta[sc.Index];
        float kalZInnVal = KalmanZInn[sc.Index];
        float kalConf = KalmanConf[sc.Index];
        bool kalLong = (kalZInnVal < -1.3125f);
        bool kalShort = (kalZInnVal > 1.3125f);

        // --- OLS signal conditions ---
        bool olsLong = (zScore < -zEntry && confidence >= minConf);
        bool olsShort = (zScore > zEntry && confidence >= minConf);

        // --- Agreement ---
        bool agree = (olsLong && kalLong) || (olsShort && kalShort);
        bool disagree = (olsLong && kalShort) || (olsShort && kalLong);

        // ============================================================
        // PANEL 1 — SIGNAL (clean, bold)
        // ============================================================
        SCString SignalText;
        COLORREF sigBgColor;

        if (TradeState == STATE_LONG)
        {
            SignalText.Format(
                "  LONG  -  Buy NQ / Sell YM\n"
                "  Z: %+.2f   Conf: %.0f%%   Kalman: %s",
                zScore, confidence,
                agree ? "Agree" : (disagree ? "Disagree" : "Neutral"));
            sigBgColor = agree ? RGB(0, 110, 0) : RGB(0, 75, 0);
        }
        else if (TradeState == STATE_SHORT)
        {
            SignalText.Format(
                "  SHORT  -  Sell NQ / Buy YM\n"
                "  Z: %+.2f   Conf: %.0f%%   Kalman: %s",
                zScore, confidence,
                agree ? "Agree" : (disagree ? "Disagree" : "Neutral"));
            sigBgColor = agree ? RGB(150, 0, 0) : RGB(100, 0, 0);
        }
        else if (TradeState == STATE_COOLDOWN)
        {
            SignalText.Format(
                "  COOLDOWN\n"
                "  Z: %+.2f   Waiting |Z| < %.2f",
                zScore, zExit);
            sigBgColor = RGB(130, 70, 0);
        }
        else // FLAT
        {
            bool confReady = (confidence >= minConf);
            bool zNearEntry = (fabs(zScore) >= zEntry * 0.85f);

            if (confReady && zNearEntry)
            {
                SignalText.Format(
                    "  APPROACHING %s\n"
                    "  Z: %+.2f / %.2f   Conf: %.0f%%",
                    (zScore < 0) ? "Long" : "Short",
                    zScore, zEntry, confidence);
                sigBgColor = RGB(75, 75, 0);
            }
            else
            {
                SignalText.Format(
                    "  NO SIGNAL\n"
                    "  Z: %+.2f   Conf: %.0f%%",
                    zScore, confidence);
                sigBgColor = RGB(35, 35, 50);
            }
        }

        s_UseTool SignalBox;
        SignalBox.Clear();
        SignalBox.ChartNumber = sc.ChartNumber;
        SignalBox.DrawingType = DRAWING_TEXT;
        SignalBox.LineNumber = 10001;
        SignalBox.BeginDateTime = 5;
        SignalBox.BeginValue = 95;
        SignalBox.UseRelativeVerticalValues = 1;
        SignalBox.Region = sc.GraphRegion;
        SignalBox.Text = SignalText;
        SignalBox.FontSize = 10;
        SignalBox.FontBold = 1;
        SignalBox.Color = RGB(255, 255, 255);
        SignalBox.FontBackColor = sigBgColor;
        SignalBox.TransparentLabelBackground = 0;
        SignalBox.TextAlignment = DT_LEFT;
        SignalBox.AddMethod = UTAM_ADD_OR_ADJUST;
        sc.UseTool(SignalBox);

        // ============================================================
        // PANEL 2 — DASHBOARD (metrics + sizing + Kalman)
        // ============================================================

        // Sizing: dollar-neutral
        int N_NQ = 1;
        int N_YM = 0;
        if (YMClose > 0.0f)
            N_YM = (int)(((NQClose * NQ_MULTIPLIER) / (YMClose * YM_MULTIPLIER))
                         * beta * N_NQ + 0.5f);
        if (N_YM < 1) N_YM = 1;

        SCString AdfTag = (adfStat < ADF_CRITICAL_5PCT) ? "Stat" : "Non-Stat";
        SCString HurstTag = (hurst < 0.5f) ? "MR" : "Trend";
        SCString CorrTag = (correlation >= 0.85f) ? "OK" :
                           (correlation >= 0.65f) ? "~" : "Low";

        SCString KalDir = "---";
        if (kalLong) KalDir = "Long";
        else if (kalShort) KalDir = "Short";

        SCString DashText;
        DashText.Format(
            "  NQ x%d  /  YM x%d       Beta: %.4f\n"
            "  ADF: %.2f [%s]   Hurst: %.2f [%s]   HL: %.0f\n"
            "  Corr: %.2f [%s]   Conf: %.0f%%\n"
            "  Kalman   Beta: %.4f   Z: %+.2f   Dir: %s",
            N_NQ, N_YM, beta,
            adfStat, AdfTag.GetChars(), hurst, HurstTag.GetChars(), halfLife,
            correlation, CorrTag.GetChars(), confidence,
            kalBeta, kalZInnVal, KalDir.GetChars()
        );

        // Confidence gradient background
        COLORREF dashBg;
        if (confidence >= 75.0f)
            dashBg = RGB(12, 50, 12);
        else if (confidence >= 67.0f)
            dashBg = RGB(18, 45, 18);
        else if (confidence >= 55.0f)
            dashBg = RGB(50, 42, 8);
        else if (confidence >= 40.0f)
            dashBg = RGB(50, 25, 8);
        else
            dashBg = RGB(45, 12, 12);

        s_UseTool DashBox;
        DashBox.Clear();
        DashBox.ChartNumber = sc.ChartNumber;
        DashBox.DrawingType = DRAWING_TEXT;
        DashBox.LineNumber = 10002;
        DashBox.BeginDateTime = 5;
        DashBox.BeginValue = 78;
        DashBox.UseRelativeVerticalValues = 1;
        DashBox.Region = sc.GraphRegion;
        DashBox.Text = DashText;
        DashBox.FontSize = 8;
        DashBox.FontBold = 0;
        DashBox.Color = RGB(190, 195, 210);
        DashBox.FontBackColor = dashBg;
        DashBox.TransparentLabelBackground = 0;
        DashBox.TextAlignment = DT_LEFT;
        DashBox.AddMethod = UTAM_ADD_OR_ADJUST;
        sc.UseTool(DashBox);

        // ============================================================
        // PANEL 3 — SIZING (exact ratios, standard + micro)
        // ============================================================

        // OLS sizing: N_YM = (NQ_not / YM_not) * beta_ols
        float olsRatioExact = (NQClose * NQ_MULTIPLIER) / (YMClose * YM_MULTIPLIER) * beta;
        int olsRoundStd = (int)(olsRatioExact + 0.5f);
        if (olsRoundStd < 1) olsRoundStd = 1;

        // Kalman sizing: N_YM = (NQ_not / YM_not) * beta_kalman
        float kalRatioExact = 0.0f;
        if (kalBeta > 0.0f)
            kalRatioExact = (NQClose * NQ_MULTIPLIER) / (YMClose * YM_MULTIPLIER) * kalBeta;
        int kalRoundStd = (int)(kalRatioExact + 0.5f);
        if (kalRoundStd < 1 && kalBeta > 0.0f) kalRoundStd = 1;

        // Micro: MNQ=$2/pt, MYM=$0.50/pt — ratio identique (mult/10)
        // Mx2: 2 MNQ -> 2 * ratio MYM (meilleure granularite)
        float mx2OlsExact = olsRatioExact * 2.0f;
        int mx2OlsRound = (int)(mx2OlsExact + 0.5f);
        float mx2KalExact = kalRatioExact * 2.0f;
        int mx2KalRound = (int)(mx2KalExact + 0.5f);

        // Notionals
        float notNQ = NQClose * NQ_MULTIPLIER;
        float notYM = YMClose * YM_MULTIPLIER;

        SCString SizingText;
        SizingText.Format(
            "  SIZING        OLS (%.4f)       Kalman (%.4f)\n"
            "  Std    1 NQ / %d YM           |  1 NQ / %d YM\n"
            "  Mx2    2 MNQ / %d MYM      |  2 MNQ / %d MYM\n"
            "  Not:  NQ $%.0f   YM $%.0f",
            beta, kalBeta,
            olsRoundStd, kalRoundStd,
            mx2OlsRound, mx2KalRound,
            notNQ, notYM
        );

        s_UseTool SizingBox;
        SizingBox.Clear();
        SizingBox.ChartNumber = sc.ChartNumber;
        SizingBox.DrawingType = DRAWING_TEXT;
        SizingBox.LineNumber = 10003;
        SizingBox.BeginDateTime = 5;
        SizingBox.BeginValue = 55;
        SizingBox.UseRelativeVerticalValues = 1;
        SizingBox.Region = sc.GraphRegion;
        SizingBox.Text = SizingText;
        SizingBox.FontSize = 8;
        SizingBox.FontBold = 0;
        SizingBox.Color = RGB(170, 180, 200);
        SizingBox.FontBackColor = RGB(25, 25, 35);
        SizingBox.TransparentLabelBackground = 0;
        SizingBox.TextAlignment = DT_LEFT;
        SizingBox.AddMethod = UTAM_ADD_OR_ADJUST;
        sc.UseTool(SizingBox);

        // ============================================================
        // PANEL 4 — TRADING STATUS (Phase 2b)
        // ============================================================

        // Read trading params for display
        SCString leg1SymDisp = InLeg1Symbol.GetString();
        SCString leg2SymDisp = InLeg2Symbol.GetString();
        int leg1QtyDisp = InLeg1Qty.GetInt();
        int leg2QtyDisp = InLeg2Qty.GetInt();
        float tradingZExitDisp = InTradingZExit.GetFloat();
        float dollarStopDisp = InDollarStop.GetFloat();
        bool autoExitDisp = (InEnableAutoExit.GetYesNo() != 0);

        // Get live P&L if in position
        double dispPnL = 0.0;
        if (TradingPosition != 0)
        {
            s_SCPositionData L1P, L2P;
            sc.GetTradePositionForSymbolAndAccount(L1P, leg1SymDisp, sc.SelectedTradeAccount);
            sc.GetTradePositionForSymbolAndAccount(L2P, leg2SymDisp, sc.SelectedTradeAccount);
            dispPnL = L1P.OpenProfitLoss + L2P.OpenProfitLoss;
        }

        SCString TradingText;
        COLORREF tradeBg;

        if (TradingPosition == 0)
        {
            TradingText.Format(
                "  TRADING  [FLAT]   Auto Exit: %s\n"
                "  Leg1: %s x%d   |   Leg2: %s x%d\n"
                "  Z Exit: %.2f   Dollar Stop: $%.0f",
                autoExitDisp ? "ON" : "OFF",
                leg1SymDisp.GetChars(), leg1QtyDisp,
                leg2SymDisp.GetChars(), leg2QtyDisp,
                tradingZExitDisp, dollarStopDisp
            );
            tradeBg = RGB(40, 40, 50);
        }
        else
        {
            const char* posLabel = (TradingPosition == 1) ? "LONG SPREAD" : "SHORT SPREAD";
            int barsInTrade = sc.Index - EntryBarIndex;

            TradingText.Format(
                "  TRADING  [%s]   Bars: %d\n"
                "  P&L: $%.0f   |   Entry Z: %.2f\n"
                "  Z now: %+.2f   Z exit: %.2f   $Stop: -$%.0f",
                posLabel, barsInTrade,
                dispPnL, (float)EntrySpreadZ,
                zScore, tradingZExitDisp, dollarStopDisp
            );

            if (dispPnL > 0.0)
                tradeBg = RGB(10, 60, 10);
            else if (dispPnL < -dollarStopDisp * 0.5)
                tradeBg = RGB(80, 10, 10);
            else
                tradeBg = RGB(60, 30, 10);
        }

        s_UseTool TradeBox;
        TradeBox.Clear();
        TradeBox.ChartNumber = sc.ChartNumber;
        TradeBox.DrawingType = DRAWING_TEXT;
        TradeBox.LineNumber = 10004;
        TradeBox.BeginDateTime = 5;
        TradeBox.BeginValue = 35;
        TradeBox.UseRelativeVerticalValues = 1;
        TradeBox.Region = sc.GraphRegion;
        TradeBox.Text = TradingText;
        TradeBox.FontSize = 8;
        TradeBox.FontBold = 0;
        TradeBox.Color = RGB(200, 210, 230);
        TradeBox.FontBackColor = tradeBg;
        TradeBox.TransparentLabelBackground = 0;
        TradeBox.TextAlignment = DT_LEFT;
        TradeBox.AddMethod = UTAM_ADD_OR_ADJUST;
        sc.UseTool(TradeBox);
    }
}
