// ============================================================================
// KalmanFixedSpreadIndicator.cpp
//
// ETUDE ACSIL - INDICATEUR VISUEL DE SPREAD A PARAMETRES KALMAN FIXES
// Applicable a toute paire de futures via Inputs configurables
//
// Description:
//   Indicateur visuel pour analyser les spreads entre paires de futures.
//   Alpha et Beta sont fournis en entree (issus d'un modele Kalman externe).
//   Affiche le log-spread et le z-score sur le chart, avec une textbox
//   contenant les metriques statistiques (ADF, Hurst, Correlation, Half-Life),
//   un score composite, et le sizing dollar-neutral (standard + micro).
//
//   Visual-only: pas de trading, pas de Kalman interne, pas de state machine.
//
// Spread: log(Y) - alpha - beta * log(X)   (alpha/beta fixes)
// Z-score: (ln(A/B) - SMA) / StdDev        (rolling on log price ratio)
//
// Metriques:
//   - ADF simplifie (Dickey-Fuller sans augmentation)
//   - Hurst variance-ratio (NOT R/S)
//   - Half-Life AR(1) via Cov/Var
//   - Correlation Pearson sur log-prix
//
// Scoring: 40% ADF + 30% Corr + 30% HL (Hurst hors score)
//
// Date: Mars 2026
// ============================================================================

#include "sierrachart.h"
#include <cmath>

SCDLLName("KalmanFixedSpreadIndicator")

// ============================================================================
// CONSTANTES DE SCORING
// ============================================================================

const float W_ADF  = 0.40f;
const float W_CORR = 0.30f;
const float W_HL   = 0.30f;

const float ADF_SCORE_BEST  = -3.00f;
const float ADF_SCORE_WORST = -2.50f;

const float CORR_SCORE_BEST  = 0.85f;
const float CORR_SCORE_WORST = 0.50f;

const float HL_SCORE_BEST  = 24.0f;
const float HL_SCORE_WORST = 144.0f;

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
        if (i >= 0 && logX[i] != 0.0f && logY[i] != 0.0f)
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
// 3. CalculateADFSimple - Dickey-Fuller simple (sans augmentation)
// ----------------------------------------------------------------------------
float CalculateADFSimple(SCFloatArrayRef spread, int endIndex, int period)
{
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
        if (i >= 1 && spread[i] != 0.0f && spread[i-1] != 0.0f)
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

    if (fabs(ss_x) < 1e-12)
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
// 4. CalculateHurstVR - Hurst via variance-ratio (NOT R/S)
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
// 5. CalculateHalfLife - AR(1) via Cov/Var
// ----------------------------------------------------------------------------
float CalculateHalfLife(SCFloatArrayRef spread, int endIndex, int period)
{
    if (period < 3 || endIndex < period)
        return 0.0f;

    int startIdx = endIndex - period + 2;
    if (startIdx < 1) startIdx = 1;

    double sumX = 0.0, sumY = 0.0;
    int count = 0;
    for (int i = startIdx; i <= endIndex; i++)
    {
        sumX += spread[i-1];
        sumY += spread[i];
        count++;
    }
    if (count < 3) return 0.0f;

    double meanX = sumX / count;
    double meanY = sumY / count;

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
// 6. CalculateScore - Score composite 0-100
// ----------------------------------------------------------------------------
float CalculateScore(float adfStat, float correlation, float halfLife,
                     SCString& outLabel, COLORREF& outColor)
{
    float s_adf = 0.0f;
    if (adfStat <= ADF_SCORE_BEST)
        s_adf = 100.0f;
    else if (adfStat >= ADF_SCORE_WORST)
        s_adf = 0.0f;
    else
        s_adf = 100.0f * (adfStat - ADF_SCORE_WORST) / (ADF_SCORE_BEST - ADF_SCORE_WORST);

    float s_corr = 0.0f;
    if (correlation >= CORR_SCORE_BEST)
        s_corr = 100.0f;
    else if (correlation <= CORR_SCORE_WORST)
        s_corr = 0.0f;
    else
        s_corr = 100.0f * (correlation - CORR_SCORE_WORST) / (CORR_SCORE_BEST - CORR_SCORE_WORST);

    float s_hl = 0.0f;
    if (halfLife <= 0.0f)
        s_hl = 0.0f;
    else if (halfLife <= HL_SCORE_BEST)
        s_hl = 100.0f;
    else if (halfLife >= HL_SCORE_WORST)
        s_hl = 0.0f;
    else
        s_hl = 100.0f * (HL_SCORE_WORST - halfLife) / (HL_SCORE_WORST - HL_SCORE_BEST);

    float score = W_ADF * s_adf + W_CORR * s_corr + W_HL * s_hl;
    if (score < 0.0f) score = 0.0f;
    if (score > 100.0f) score = 100.0f;

    if (score >= 75.0f)      { outLabel = "FORT";   outColor = RGB(0, 200, 0); }
    else if (score >= 50.0f) { outLabel = "BON";    outColor = RGB(220, 220, 230); }
    else if (score >= 25.0f) { outLabel = "LEGER";  outColor = RGB(255, 165, 0); }
    else                     { outLabel = "FAIBLE"; outColor = RGB(255, 50, 50); }

    return score;
}

// ----------------------------------------------------------------------------
// 7. GetMetricLabel - Label par metrique (FORT/BON/LEGER/FAIBLE)
// ----------------------------------------------------------------------------
void GetMetricLabel(const char* metricType, float value, SCString& outLabel)
{
    if (strcmp(metricType, "ADF") == 0)
    {
        if (value < -3.00f)       outLabel = "FORT";
        else if (value < -2.86f)  outLabel = "BON";
        else if (value < -2.50f)  outLabel = "LEGER";
        else                      outLabel = "FAIBLE";
    }
    else if (strcmp(metricType, "HURST") == 0)
    {
        if (value < 0.35f)        outLabel = "FORT";
        else if (value < 0.45f)   outLabel = "BON";
        else if (value < 0.50f)   outLabel = "LEGER";
        else                      outLabel = "FAIBLE";
    }
    else if (strcmp(metricType, "CORR") == 0)
    {
        if (value > 0.85f)        outLabel = "FORT";
        else if (value > 0.70f)   outLabel = "BON";
        else if (value > 0.50f)   outLabel = "LEGER";
        else                      outLabel = "FAIBLE";
    }
    else if (strcmp(metricType, "HL") == 0)
    {
        if (value > 0.0f && value <= 24.0f)  outLabel = "FORT";
        else if (value <= 48.0f)             outLabel = "BON";
        else if (value <= 144.0f)            outLabel = "LEGER";
        else                                 outLabel = "FAIBLE";
    }
}

// ============================================================================
// FONCTION PRINCIPALE
// ============================================================================
SCSFExport scsf_KalmanFixedSpreadIndicator(SCStudyInterfaceRef sc)
{
    // ========================================================================
    // SUBGRAPHS
    // ========================================================================
    SCSubgraphRef Spread    = sc.Subgraph[0];
    SCSubgraphRef ZScore    = sc.Subgraph[1];
    SCSubgraphRef ZeroLine  = sc.Subgraph[2];
    SCSubgraphRef LogA      = sc.Subgraph[3];
    SCSubgraphRef LogB      = sc.Subgraph[4];
    SCSubgraphRef SpreadSMA = sc.Subgraph[5];
    SCSubgraphRef ZUpperLine = sc.Subgraph[6];
    SCSubgraphRef ZLowerLine = sc.Subgraph[7];
    SCSubgraphRef SubADF     = sc.Subgraph[8];
    SCSubgraphRef SubHurst   = sc.Subgraph[9];
    SCSubgraphRef SubCorr    = sc.Subgraph[10];
    SCSubgraphRef SubHL      = sc.Subgraph[11];
    SCSubgraphRef SubScore   = sc.Subgraph[12];
    SCSubgraphRef LogRatio   = sc.Subgraph[13];
    SCSubgraphRef LogRatSMA  = sc.Subgraph[14];

    // ========================================================================
    // INPUTS
    // ========================================================================
    SCInputRef InChartB       = sc.Input[0];
    SCInputRef InPointValueA  = sc.Input[1];
    SCInputRef InPointValueB  = sc.Input[2];
    SCInputRef InTickSizeA    = sc.Input[3];
    SCInputRef InTickSizeB    = sc.Input[4];
    SCInputRef InMicroRatioA  = sc.Input[5];
    SCInputRef InMicroRatioB  = sc.Input[6];
    SCInputRef InSymNameA     = sc.Input[7];
    SCInputRef InSymNameB     = sc.Input[8];
    SCInputRef InMicroNameA   = sc.Input[9];
    SCInputRef InMicroNameB   = sc.Input[10];
    SCInputRef InKalmanAlpha  = sc.Input[11];
    SCInputRef InKalmanBeta   = sc.Input[12];
    SCInputRef InZScorePeriod = sc.Input[13];
    SCInputRef InCorrPeriod   = sc.Input[14];
    SCInputRef InADFPeriod    = sc.Input[15];
    SCInputRef InHurstPeriod  = sc.Input[16];
    SCInputRef InHLPeriod     = sc.Input[17];
    SCInputRef InShowTextBox  = sc.Input[18];
    SCInputRef InFontSize     = sc.Input[19];
    SCInputRef InSwapRegress  = sc.Input[20];
    SCInputRef InZUpperThresh = sc.Input[21];
    SCInputRef InZLowerThresh = sc.Input[22];

    // ========================================================================
    // DEFAULTS
    // ========================================================================
    if (sc.SetDefaults)
    {
        sc.GraphName = "Kalman Fixed Spread Indicator";
        sc.StudyDescription = "Spread analysis with fixed Kalman alpha/beta: z-score, ADF, Hurst, HL, Corr, scoring, sizing";
        sc.AutoLoop = 1;
        sc.GraphRegion = 1;
        sc.CalculationPrecedence = LOW_PREC_LEVEL;

        // --- Subgraphs ---
        Spread.Name = "Spread (internal)";
        Spread.DrawStyle = DRAWSTYLE_IGNORE;
        Spread.DrawZeros = 0;

        ZScore.Name = "Z-Score";
        ZScore.DrawStyle = DRAWSTYLE_LINE;
        ZScore.PrimaryColor = RGB(0, 200, 255);
        ZScore.LineWidth = 2;
        ZScore.DrawZeros = 0;

        ZeroLine.Name = "Zero";
        ZeroLine.DrawStyle = DRAWSTYLE_LINE;
        ZeroLine.PrimaryColor = RGB(128, 128, 128);
        ZeroLine.LineWidth = 1;
        ZeroLine.DrawZeros = 1;

        LogA.Name = "LogA (internal)";
        LogA.DrawStyle = DRAWSTYLE_IGNORE;
        LogA.DrawZeros = 0;

        LogB.Name = "LogB (internal)";
        LogB.DrawStyle = DRAWSTYLE_IGNORE;
        LogB.DrawZeros = 0;

        SpreadSMA.Name = "SpreadSMA (internal)";
        SpreadSMA.DrawStyle = DRAWSTYLE_IGNORE;
        SpreadSMA.DrawZeros = 0;

        ZUpperLine.Name = "Z Upper";
        ZUpperLine.DrawStyle = DRAWSTYLE_DASH;
        ZUpperLine.PrimaryColor = RGB(255, 80, 80);
        ZUpperLine.LineWidth = 1;
        ZUpperLine.DrawZeros = 0;

        ZLowerLine.Name = "Z Lower";
        ZLowerLine.DrawStyle = DRAWSTYLE_DASH;
        ZLowerLine.PrimaryColor = RGB(255, 80, 80);
        ZLowerLine.LineWidth = 1;
        ZLowerLine.DrawZeros = 0;

        // --- Metric subgraphs (hidden, for Alert Conditions) ---
        SubADF.Name = "ADF Stat";
        SubADF.DrawStyle = DRAWSTYLE_HIDDEN;
        SubADF.DrawZeros = 0;

        SubHurst.Name = "Hurst";
        SubHurst.DrawStyle = DRAWSTYLE_HIDDEN;
        SubHurst.DrawZeros = 0;

        SubCorr.Name = "Correlation";
        SubCorr.DrawStyle = DRAWSTYLE_HIDDEN;
        SubCorr.DrawZeros = 0;

        SubHL.Name = "Half-Life";
        SubHL.DrawStyle = DRAWSTYLE_HIDDEN;
        SubHL.DrawZeros = 0;

        SubScore.Name = "Score";
        SubScore.DrawStyle = DRAWSTYLE_HIDDEN;
        SubScore.DrawZeros = 0;

        LogRatio.Name = "LogRatio (internal)";
        LogRatio.DrawStyle = DRAWSTYLE_IGNORE;
        LogRatio.DrawZeros = 0;

        LogRatSMA.Name = "LogRatSMA (internal)";
        LogRatSMA.DrawStyle = DRAWSTYLE_IGNORE;
        LogRatSMA.DrawZeros = 0;

        // --- Inputs ---
        InChartB.Name = "Chart Number B (Secondary)";
        InChartB.SetInt(2);
        InChartB.SetIntLimits(1, 100);

        InPointValueA.Name = "Point Value A ($/pt)";
        InPointValueA.SetFloat(1000.0f);   // CL = $1000/pt
        InPointValueA.SetFloatLimits(0.01f, 100000.0f);

        InPointValueB.Name = "Point Value B ($/pt)";
        InPointValueB.SetFloat(10000.0f);  // NG = $10000/pt
        InPointValueB.SetFloatLimits(0.01f, 100000.0f);

        InTickSizeA.Name = "Tick Size A";
        InTickSizeA.SetFloat(0.01f);       // CL tick = 0.01
        InTickSizeA.SetFloatLimits(0.0001f, 100.0f);

        InTickSizeB.Name = "Tick Size B";
        InTickSizeB.SetFloat(0.001f);      // NG tick = 0.001
        InTickSizeB.SetFloatLimits(0.0001f, 100.0f);

        InMicroRatioA.Name = "Micro Ratio A (10=1/10, 5=1/5)";
        InMicroRatioA.SetInt(10);          // MCL = 1/10 CL
        InMicroRatioA.SetIntLimits(1, 100);

        InMicroRatioB.Name = "Micro Ratio B (10=1/10, 5=1/5)";
        InMicroRatioB.SetInt(4);           // QG = 1/4 NG
        InMicroRatioB.SetIntLimits(1, 100);

        InSymNameA.Name = "Symbol Name A";
        InSymNameA.SetString("CL");

        InSymNameB.Name = "Symbol Name B";
        InSymNameB.SetString("NG");

        InMicroNameA.Name = "Micro Name A";
        InMicroNameA.SetString("MCL");

        InMicroNameB.Name = "Micro Name B";
        InMicroNameB.SetString("QG");

        InKalmanAlpha.Name = "Kalman Alpha (Intercept)";
        InKalmanAlpha.SetFloat(3.3379f);
        InKalmanAlpha.SetFloatLimits(-100.0f, 100.0f);

        InKalmanBeta.Name = "Kalman Beta (Hedge Ratio)";
        InKalmanBeta.SetFloat(0.9135f);
        InKalmanBeta.SetFloatLimits(-50.0f, 50.0f);

        InZScorePeriod.Name = "Z-Score Period";
        InZScorePeriod.SetInt(135);
        InZScorePeriod.SetIntLimits(2, 50000);

        InCorrPeriod.Name = "Correlation Period";
        InCorrPeriod.SetInt(96);
        InCorrPeriod.SetIntLimits(2, 50000);

        InADFPeriod.Name = "ADF Period";
        InADFPeriod.SetInt(96);
        InADFPeriod.SetIntLimits(2, 50000);

        InHurstPeriod.Name = "Hurst Period";
        InHurstPeriod.SetInt(128);
        InHurstPeriod.SetIntLimits(2, 50000);

        InHLPeriod.Name = "Half-Life Period";
        InHLPeriod.SetInt(96);
        InHLPeriod.SetIntLimits(2, 50000);

        InShowTextBox.Name = "Show TextBox";
        InShowTextBox.SetYesNo(1);

        InFontSize.Name = "TextBox Font Size";
        InFontSize.SetInt(8);
        InFontSize.SetIntLimits(6, 14);

        InSwapRegress.Name = "Swap Regression (Y=B, for GC/SI)";
        InSwapRegress.SetYesNo(0);  // Default OFF for CL/NG (Y=CL, X=NG)

        InZUpperThresh.Name = "Z-Score Upper Line";
        InZUpperThresh.SetFloat(3.0f);
        InZUpperThresh.SetFloatLimits(-50.0f, 50.0f);

        InZLowerThresh.Name = "Z-Score Lower Line";
        InZLowerThresh.SetFloat(-3.0f);
        InZLowerThresh.SetFloatLimits(-50.0f, 50.0f);

        return;
    }

    // ========================================================================
    // LECTURE DES PARAMETRES
    // ========================================================================
    int ChartB        = InChartB.GetInt();
    float PointValueA = InPointValueA.GetFloat();
    float PointValueB = InPointValueB.GetFloat();
    int MicroRatioA   = InMicroRatioA.GetInt();
    int MicroRatioB   = InMicroRatioB.GetInt();
    float KalmanAlpha = InKalmanAlpha.GetFloat();
    float KalmanBeta  = InKalmanBeta.GetFloat();
    int ZScorePeriod  = InZScorePeriod.GetInt();
    int CorrPeriod    = InCorrPeriod.GetInt();
    int ADFPeriod     = InADFPeriod.GetInt();
    int HurstPeriod   = InHurstPeriod.GetInt();
    int HLPeriod      = InHLPeriod.GetInt();
    int ShowTextBox   = InShowTextBox.GetYesNo();
    int FontSize      = InFontSize.GetInt();
    int SwapRegress   = InSwapRegress.GetYesNo();
    float ZUpperThresh = InZUpperThresh.GetFloat();
    float ZLowerThresh = InZLowerThresh.GetFloat();

    // ========================================================================
    // ACCES DONNEES CHART B
    // ========================================================================
    SCGraphData ChartBData;
    sc.GetChartBaseData(ChartB, ChartBData);

    if (ChartBData[SC_LAST].GetArraySize() == 0)
    {
        if (sc.Index == 0)
            sc.AddMessageToLog("KalmanFixedSpread: Chart B data not available. Check Chart Number B input.", 1);
        Spread[sc.Index]    = 0.0f;
        ZScore[sc.Index]    = 0.0f;
        ZeroLine[sc.Index]  = 0.0f;
        ZUpperLine[sc.Index] = 0.0f;
        ZLowerLine[sc.Index] = 0.0f;
        SpreadSMA[sc.Index] = 0.0f;
        SubADF[sc.Index]    = 0.0f;
        SubHurst[sc.Index]  = 0.0f;
        SubCorr[sc.Index]   = 0.0f;
        SubHL[sc.Index]     = 0.0f;
        SubScore[sc.Index]  = 0.0f;
        return;
    }

    int idxB = sc.GetContainingIndexForDateTimeIndex(ChartB, sc.Index);
    if (idxB < 0 || idxB >= ChartBData[SC_LAST].GetArraySize())
    {
        Spread[sc.Index]    = 0.0f;
        ZScore[sc.Index]    = 0.0f;
        ZeroLine[sc.Index]  = 0.0f;
        ZUpperLine[sc.Index] = 0.0f;
        ZLowerLine[sc.Index] = 0.0f;
        SpreadSMA[sc.Index] = 0.0f;
        SubADF[sc.Index]    = 0.0f;
        SubHurst[sc.Index]  = 0.0f;
        SubCorr[sc.Index]   = 0.0f;
        SubHL[sc.Index]     = 0.0f;
        SubScore[sc.Index]  = 0.0f;
        return;
    }

    float PriceA = sc.Close[sc.Index];
    float PriceB = ChartBData[SC_LAST][idxB];

    if (PriceA <= 0.0f || PriceB <= 0.0f)
    {
        Spread[sc.Index]    = 0.0f;
        ZScore[sc.Index]    = 0.0f;
        ZeroLine[sc.Index]  = 0.0f;
        ZUpperLine[sc.Index] = 0.0f;
        ZLowerLine[sc.Index] = 0.0f;
        SpreadSMA[sc.Index] = 0.0f;
        SubADF[sc.Index]    = 0.0f;
        SubHurst[sc.Index]  = 0.0f;
        SubCorr[sc.Index]   = 0.0f;
        SubHL[sc.Index]     = 0.0f;
        SubScore[sc.Index]  = 0.0f;
        return;
    }

    // ========================================================================
    // LOG-PRIX
    // ========================================================================
    LogA[sc.Index] = (float)log((double)PriceA);
    LogB[sc.Index] = (float)log((double)PriceB);

    // Horizontal lines (toujours)
    ZeroLine[sc.Index]  = 0.0f;
    ZUpperLine[sc.Index] = ZUpperThresh;
    ZLowerLine[sc.Index] = ZLowerThresh;

    // ========================================================================
    // SPREAD avec alpha/beta FIXES (Kalman externe)
    // ========================================================================
    float alpha = KalmanAlpha;
    float beta  = KalmanBeta;
    float spreadVal;

    if (SwapRegress)
    {
        // Y=LogB, X=LogA → Spread = LogB - alpha - beta*LogA
        spreadVal = LogB[sc.Index] - alpha - beta * LogA[sc.Index];
    }
    else
    {
        // Y=LogA, X=LogB → Spread = LogA - alpha - beta*LogB
        spreadVal = LogA[sc.Index] - alpha - beta * LogB[sc.Index];
    }

    Spread[sc.Index] = spreadVal;

    // ========================================================================
    // LOG RATIO ln(A/B) pour z-score (independant du Kalman alpha/beta)
    // ========================================================================
    float logRatioVal;
    if (SwapRegress)
        logRatioVal = LogB[sc.Index] - LogA[sc.Index];  // ln(B/A)
    else
        logRatioVal = LogA[sc.Index] - LogB[sc.Index];  // ln(A/B)

    LogRatio[sc.Index] = logRatioVal;

    // ========================================================================
    // Z-SCORE (rolling on log ratio)
    // ========================================================================
    sc.SimpleMovAvg(LogRatio, LogRatSMA, ZScorePeriod);
    float stdDev = CalculateStdDev(LogRatio, sc.Index, ZScorePeriod);
    float zScore = 0.0f;
    if (stdDev > 1e-10f)
        zScore = (logRatioVal - LogRatSMA[sc.Index]) / stdDev;

    if (zScore > 10.0f)  zScore = 10.0f;
    if (zScore < -10.0f) zScore = -10.0f;

    ZScore[sc.Index] = zScore;

    // ========================================================================
    // METRIQUES STATISTIQUES
    // ========================================================================
    float adfStat    = CalculateADFSimple(Spread, sc.Index, ADFPeriod);
    float hurst      = CalculateHurstVR(Spread, sc.Index, HurstPeriod);
    float halfLife    = CalculateHalfLife(Spread, sc.Index, HLPeriod);
    float correlation = CalculateCorrelation(LogA, LogB, sc.Index, CorrPeriod);

    // ========================================================================
    // SCORING COMPOSITE
    // ========================================================================
    SCString scoreLabel;
    COLORREF scoreColor;
    float score = CalculateScore(adfStat, correlation, halfLife, scoreLabel, scoreColor);

    SubADF[sc.Index]   = adfStat;
    SubHurst[sc.Index] = hurst;
    SubCorr[sc.Index]  = correlation;
    SubHL[sc.Index]    = halfLife;
    SubScore[sc.Index] = score;

    // ========================================================================
    // COULEUR DYNAMIQUE Z-SCORE
    // ========================================================================
    float absZ = (float)fabs(zScore);
    if (absZ >= 3.0f)
        ZScore.DataColor[sc.Index] = RGB(255, 50, 50);      // Rouge -- extreme
    else if (absZ >= 2.0f)
        ZScore.DataColor[sc.Index] = RGB(255, 165, 0);      // Orange -- attention
    else if (absZ >= 1.0f)
        ZScore.DataColor[sc.Index] = RGB(0, 200, 255);      // Cyan -- actif
    else
        ZScore.DataColor[sc.Index] = RGB(180, 180, 180);    // Gris -- neutre

    // ========================================================================
    // TEXTBOX (derniere barre uniquement)
    // ========================================================================
    if (ShowTextBox && sc.Index == sc.ArraySize - 1)
    {
        SCString symA = InSymNameA.GetString();
        SCString symB = InSymNameB.GetString();
        SCString microA = InMicroNameA.GetString();
        SCString microB = InMicroNameB.GetString();

        // Labels par metrique
        SCString adfLabel, hurstLabel, corrLabel, hlLabel;
        GetMetricLabel("ADF", adfStat, adfLabel);
        GetMetricLabel("HURST", hurst, hurstLabel);
        GetMetricLabel("CORR", correlation, corrLabel);
        GetMetricLabel("HL", halfLife, hlLabel);

        // Half-life en temps humain
        SCString hlTimeStr;
        if (halfLife <= 0.0f)
        {
            hlTimeStr = "---";
        }
        else if (sc.SecondsPerBar > 0)
        {
            int totalMinutes = (int)(halfLife * (float)sc.SecondsPerBar / 60.0f + 0.5f);
            int hours = totalMinutes / 60;
            int mins = totalMinutes % 60;
            hlTimeStr.Format("%dh%02d", hours, mins);
        }
        else
        {
            hlTimeStr.Format("%.0f bars", halfLife);
        }

        // ================================================================
        // SIZING DOLLAR-NEUTRAL
        // ================================================================
        float notionalA = PriceA * PointValueA;
        float notionalB = PriceB * PointValueB;
        float microPVA = PointValueA / (float)MicroRatioA;
        float microPVB = PointValueB / (float)MicroRatioB;
        float absBeta = (float)fabs(beta);

        float N_a_std, N_b_std;
        float micro_a_exact, micro_b_exact;

        if (SwapRegress)
        {
            N_b_std = 1.0f;
            N_a_std = 1.0f;
            if (notionalA > 0.0f)
                N_a_std = notionalB / notionalA * absBeta;

            float ratioPerMicroB = 0.0f;
            float microNotA = PriceA * microPVA;
            if (microNotA > 0.0f)
                ratioPerMicroB = PriceB * microPVB / microNotA * absBeta;
            micro_a_exact = ratioPerMicroB;
            micro_b_exact = 1.0f;
        }
        else
        {
            N_a_std = 1.0f;
            N_b_std = 1.0f;
            if (notionalB > 0.0f)
                N_b_std = notionalA / notionalB * absBeta;

            float ratioPerMicroA = 0.0f;
            float microNotB = PriceB * microPVB;
            if (microNotB > 0.0f)
                ratioPerMicroA = PriceA * microPVA / microNotB * absBeta;
            micro_a_exact = 1.0f;
            micro_b_exact = ratioPerMicroA;
        }

        // ================================================================
        // SIZING INVERSE
        // ================================================================
        float inv_a_std, inv_b_std;
        float inv_micro_a, inv_micro_b;

        if (SwapRegress)
        {
            inv_a_std = 1.0f;
            inv_b_std = (N_a_std > 0.0f) ? N_b_std / N_a_std : 0.0f;
            inv_micro_a = 1.0f;
            inv_micro_b = (micro_a_exact > 0.0f) ? micro_b_exact / micro_a_exact : 0.0f;
        }
        else
        {
            inv_b_std = 1.0f;
            inv_a_std = (N_b_std > 0.0f) ? N_a_std / N_b_std : 0.0f;
            inv_micro_b = 1.0f;
            inv_micro_a = (micro_b_exact > 0.0f) ? micro_a_exact / micro_b_exact : 0.0f;
        }

        // ================================================================
        // CONSTRUCTION DU TEXTE
        // ================================================================
        bool hasMicro = (MicroRatioA > 1 && MicroRatioB > 1);

        SCString InfoText;
        InfoText.Format(
            "%s / %s  |  B %.4f  |  a %.4f  |  KALMAN\n"
            "-------------------------------------\n"
            "ADF: %.2f %s    Hurst: %.2f %s\n"
            "Corr: %.2f %s     HL: %.0f (%s) %s\n"
            "-------------------------------------\n"
            "SIGNAL: %.0f %s\n"
            "-------------------------------------\n"
            "STD:  %.2f %s  /  %.2f %s\n"
            "INV:  %.2f %s  /  %.2f %s",
            symA.GetChars(), symB.GetChars(), beta, alpha,
            adfStat, adfLabel.GetChars(), hurst, hurstLabel.GetChars(),
            correlation, corrLabel.GetChars(),
            halfLife, hlTimeStr.GetChars(), hlLabel.GetChars(),
            score, scoreLabel.GetChars(),
            N_a_std, symA.GetChars(), N_b_std, symB.GetChars(),
            inv_a_std, symA.GetChars(), inv_b_std, symB.GetChars()
        );

        if (hasMicro)
        {
            SCString microLines;
            microLines.Format(
                "\nMICRO: %.2f %s / %.2f %s\n"
                "M.INV: %.2f %s / %.2f %s",
                micro_a_exact, microA.GetChars(), micro_b_exact, microB.GetChars(),
                inv_micro_a, microA.GetChars(), inv_micro_b, microB.GetChars()
            );
            InfoText += microLines;
        }

        // Couleur de fond dynamique selon score
        COLORREF bgColor;
        if (score >= 75.0f)
            bgColor = RGB(12, 50, 12);
        else if (score >= 50.0f)
            bgColor = RGB(30, 30, 50);
        else if (score >= 25.0f)
            bgColor = RGB(50, 35, 8);
        else
            bgColor = RGB(45, 12, 12);

        s_UseTool TextBox;
        TextBox.Clear();
        TextBox.ChartNumber = sc.ChartNumber;
        TextBox.DrawingType = DRAWING_TEXT;
        TextBox.LineNumber = 20002;
        TextBox.BeginDateTime = 5;
        TextBox.BeginValue = 95;
        TextBox.UseRelativeVerticalValues = 1;
        TextBox.Region = sc.GraphRegion;
        TextBox.Text = InfoText;
        TextBox.FontSize = FontSize;
        TextBox.FontBold = 0;
        TextBox.Color = RGB(220, 220, 230);
        TextBox.FontBackColor = bgColor;
        TextBox.TransparentLabelBackground = 0;
        TextBox.TextAlignment = DT_LEFT;
        TextBox.AddMethod = UTAM_ADD_OR_ADJUST;
        sc.UseTool(TextBox);
    }
}

// ============================================================================
// COMPANION STUDY: Kalman Fixed Spread Line
//
// Reads Spread from the main Kalman Fixed Spread Indicator study and displays
// it in its own region (Region 2). The main study shows Z-Score in Region 1.
//
// Usage:
//   1. Apply "Kalman Fixed Spread Indicator" on chart (Region 1 = z-score + textbox)
//   2. Apply "Kalman Fixed Spread Line" on same chart (Region 2 = spread)
//   3. Set Input "Study Reference" to point to Kalman Fixed Spread Indicator,
//      Subgraph "Spread (internal)"
// ============================================================================
SCSFExport scsf_KalmanFixedSpreadLine(SCStudyInterfaceRef sc)
{
    // ========================================================================
    // SUBGRAPHS
    // ========================================================================
    SCSubgraphRef SpreadLine = sc.Subgraph[0];
    SCSubgraphRef BollMid    = sc.Subgraph[1];
    SCSubgraphRef BollUpper  = sc.Subgraph[2];
    SCSubgraphRef BollLower  = sc.Subgraph[3];

    // ========================================================================
    // INPUTS
    // ========================================================================
    SCInputRef InStudyRef  = sc.Input[0];
    SCInputRef InBollPeriod = sc.Input[1];
    SCInputRef InBollMult   = sc.Input[2];

    // ========================================================================
    // DEFAULTS
    // ========================================================================
    if (sc.SetDefaults)
    {
        sc.GraphName = "Kalman Fixed Spread Line";
        sc.StudyDescription = "Companion: Spread + Bollinger Bands from Kalman Fixed Spread Indicator";
        sc.AutoLoop = 1;
        sc.GraphRegion = 2;
        sc.CalculationPrecedence = LOW_PREC_LEVEL;

        SpreadLine.Name = "Spread";
        SpreadLine.DrawStyle = DRAWSTYLE_LINE;
        SpreadLine.PrimaryColor = RGB(255, 220, 50);
        SpreadLine.LineWidth = 2;
        SpreadLine.DrawZeros = 0;

        BollMid.Name = "BB Mid";
        BollMid.DrawStyle = DRAWSTYLE_DASH;
        BollMid.PrimaryColor = RGB(128, 128, 128);
        BollMid.LineWidth = 1;
        BollMid.DrawZeros = 0;

        BollUpper.Name = "BB Upper";
        BollUpper.DrawStyle = DRAWSTYLE_DASH;
        BollUpper.PrimaryColor = RGB(255, 80, 80);
        BollUpper.LineWidth = 1;
        BollUpper.DrawZeros = 0;

        BollLower.Name = "BB Lower";
        BollLower.DrawStyle = DRAWSTYLE_DASH;
        BollLower.PrimaryColor = RGB(255, 80, 80);
        BollLower.LineWidth = 1;
        BollLower.DrawZeros = 0;

        InStudyRef.Name = "Study Reference (Kalman Fixed Spread)";
        InStudyRef.SetStudySubgraphValues(0, 0);

        InBollPeriod.Name = "Bollinger Period";
        InBollPeriod.SetInt(30);
        InBollPeriod.SetIntLimits(2, 50000);

        InBollMult.Name = "Bollinger Multiplier";
        InBollMult.SetFloat(2.5f);
        InBollMult.SetFloatLimits(0.5f, 5.0f);

        return;
    }

    // ========================================================================
    // READ SPREAD FROM MAIN STUDY
    // ========================================================================
    SCFloatArray SourceSpread;
    sc.GetStudyArrayUsingID(InStudyRef.GetStudyID(), InStudyRef.GetSubgraphIndex(), SourceSpread);

    if (SourceSpread.GetArraySize() == 0)
    {
        SpreadLine[sc.Index] = 0.0f;
        BollMid[sc.Index]    = 0.0f;
        BollUpper[sc.Index]  = 0.0f;
        BollLower[sc.Index]  = 0.0f;
        return;
    }

    float spread = SourceSpread[sc.Index];
    SpreadLine[sc.Index] = spread;

    // ========================================================================
    // BOLLINGER BANDS
    // ========================================================================
    int period = InBollPeriod.GetInt();
    float mult = InBollMult.GetFloat();

    if (sc.Index < period - 1)
    {
        BollMid[sc.Index]   = 0.0f;
        BollUpper[sc.Index] = 0.0f;
        BollLower[sc.Index] = 0.0f;
        return;
    }

    double sum = 0.0;
    int count = 0;
    int startIdx = sc.Index - period + 1;
    for (int i = startIdx; i <= sc.Index; i++)
    {
        if (i >= 0 && SourceSpread[i] != 0.0f)
        {
            sum += SourceSpread[i];
            count++;
        }
    }

    if (count < 2)
    {
        BollMid[sc.Index]   = 0.0f;
        BollUpper[sc.Index] = 0.0f;
        BollLower[sc.Index] = 0.0f;
        return;
    }

    double mean = sum / count;

    double sumSq = 0.0;
    for (int i = startIdx; i <= sc.Index; i++)
    {
        if (i >= 0 && SourceSpread[i] != 0.0f)
        {
            double diff = SourceSpread[i] - mean;
            sumSq += diff * diff;
        }
    }

    double std = sqrt(sumSq / (count - 1));

    float mid = (float)mean;
    BollMid[sc.Index]   = mid;
    BollUpper[sc.Index] = mid + mult * (float)std;
    BollLower[sc.Index] = mid - mult * (float)std;
}

// ============================================================================
// COMPANION STUDY 2: Dollar Spread
//
// Displays the dollar-weighted spread: LotsA * PV_A * PriceA - LotsB * PV_B * PriceB
// When the line goes UP   -> long A / short B profits
// When the line goes DOWN -> short A / long B profits
//
// Usage:
//   1. Apply on same chart as Kalman Fixed Spread Indicator (Region 3)
//   2. Set Chart Number B, Lots, and Point Values for your sizing
// ============================================================================
SCSFExport scsf_KalmanFixedDollarSpread(SCStudyInterfaceRef sc)
{
    // ========================================================================
    // SUBGRAPHS
    // ========================================================================
    SCSubgraphRef DollarSpread = sc.Subgraph[0];
    SCSubgraphRef ZeroLine     = sc.Subgraph[1];

    // ========================================================================
    // INPUTS
    // ========================================================================
    SCInputRef InChartB       = sc.Input[0];
    SCInputRef InStdLotsA     = sc.Input[1];
    SCInputRef InStdLotsB     = sc.Input[2];
    SCInputRef InMicroLotsA   = sc.Input[3];
    SCInputRef InMicroLotsB   = sc.Input[4];
    SCInputRef InPointValueA  = sc.Input[5];
    SCInputRef InPointValueB  = sc.Input[6];
    SCInputRef InMicroRatioA  = sc.Input[7];
    SCInputRef InMicroRatioB  = sc.Input[8];

    // ========================================================================
    // DEFAULTS
    // ========================================================================
    if (sc.SetDefaults)
    {
        sc.GraphName = "Kalman Fixed Dollar Spread";
        sc.StudyDescription = "Dollar-weighted spread: LotsA x PV_A x PriceA - LotsB x PV_B x PriceB";
        sc.AutoLoop = 1;
        sc.GraphRegion = 3;
        sc.CalculationPrecedence = LOW_PREC_LEVEL;

        DollarSpread.Name = "Dollar Spread";
        DollarSpread.DrawStyle = DRAWSTYLE_LINE;
        DollarSpread.PrimaryColor = RGB(0, 255, 128);
        DollarSpread.LineWidth = 2;
        DollarSpread.DrawZeros = 0;

        ZeroLine.Name = "Zero";
        ZeroLine.DrawStyle = DRAWSTYLE_LINE;
        ZeroLine.PrimaryColor = RGB(128, 128, 128);
        ZeroLine.LineWidth = 1;
        ZeroLine.DrawZeros = 1;

        InChartB.Name = "Chart Number B (Secondary)";
        InChartB.SetInt(5);
        InChartB.SetIntLimits(1, 100);

        InStdLotsA.Name = "Std Lots A";
        InStdLotsA.SetInt(1);
        InStdLotsA.SetIntLimits(0, 100);

        InStdLotsB.Name = "Std Lots B";
        InStdLotsB.SetInt(0);
        InStdLotsB.SetIntLimits(0, 100);

        InMicroLotsA.Name = "Micro Lots A";
        InMicroLotsA.SetInt(0);
        InMicroLotsA.SetIntLimits(0, 100);

        InMicroLotsB.Name = "Micro Lots B";
        InMicroLotsB.SetInt(0);
        InMicroLotsB.SetIntLimits(0, 100);

        InPointValueA.Name = "Point Value A ($/pt)";
        InPointValueA.SetFloat(1000.0f);
        InPointValueA.SetFloatLimits(0.01f, 100000.0f);

        InPointValueB.Name = "Point Value B ($/pt)";
        InPointValueB.SetFloat(10000.0f);
        InPointValueB.SetFloatLimits(0.01f, 100000.0f);

        InMicroRatioA.Name = "Micro Ratio A (Std/Micro)";
        InMicroRatioA.SetInt(10);
        InMicroRatioA.SetIntLimits(1, 100);

        InMicroRatioB.Name = "Micro Ratio B (Std/Micro)";
        InMicroRatioB.SetInt(4);
        InMicroRatioB.SetIntLimits(1, 100);

        return;
    }

    // ========================================================================
    // ACCES DONNEES
    // ========================================================================
    int ChartB = InChartB.GetInt();

    SCGraphData ChartBData;
    sc.GetChartBaseData(ChartB, ChartBData);

    if (ChartBData[SC_LAST].GetArraySize() == 0)
    {
        DollarSpread[sc.Index] = 0.0f;
        ZeroLine[sc.Index] = 0.0f;
        return;
    }

    int idxB = sc.GetContainingIndexForDateTimeIndex(ChartB, sc.Index);
    if (idxB < 0 || idxB >= ChartBData[SC_LAST].GetArraySize())
    {
        DollarSpread[sc.Index] = 0.0f;
        ZeroLine[sc.Index] = 0.0f;
        return;
    }

    float PriceA = sc.Close[sc.Index];
    float PriceB = ChartBData[SC_LAST][idxB];

    if (PriceA <= 0.0f || PriceB <= 0.0f)
    {
        DollarSpread[sc.Index] = 0.0f;
        ZeroLine[sc.Index] = 0.0f;
        return;
    }

    // ========================================================================
    // DOLLAR SPREAD (delta from first bar, so line starts at 0)
    // ========================================================================
    int StdLotsA   = InStdLotsA.GetInt();
    int StdLotsB   = InStdLotsB.GetInt();
    int MicroLotsA = InMicroLotsA.GetInt();
    int MicroLotsB = InMicroLotsB.GetInt();
    float PVA      = InPointValueA.GetFloat();
    float PVB      = InPointValueB.GetFloat();
    int MicroRatA  = InMicroRatioA.GetInt();
    int MicroRatB  = InMicroRatioB.GetInt();

    float notionalA = (StdLotsA * PVA + MicroLotsA * PVA / MicroRatA) * PriceA;
    float notionalB = (StdLotsB * PVB + MicroLotsB * PVB / MicroRatB) * PriceB;
    float rawSpread = notionalA - notionalB;

    // Store reference value (first valid bar) in PersistentFloat
    float& RefValue = sc.GetPersistentFloat(0);
    if (sc.Index == 0)
        RefValue = rawSpread;

    DollarSpread[sc.Index] = rawSpread - RefValue;
    ZeroLine[sc.Index] = 0.0f;
}
