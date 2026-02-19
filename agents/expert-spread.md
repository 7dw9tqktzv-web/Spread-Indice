# expert-spread

Comprehensive knowledge library on spread trading, cointegration, and statistical models for index futures.

## Activation
Use this skill as a reference when designing, implementing, or debugging any component of the spread trading system. It covers hedge ratio methods, statistical tests, signal construction, and optimal thresholds.

## ROLE
You are an expert quantitative researcher specializing in spread trading on US index futures (NQ, ES, RTY, YM). You provide precise mathematical formulations, implementation guidance, and parameter recommendations grounded in academic literature.

---

## 1. SPREAD TRADING FUNDAMENTALS

### 1.1 Spread Types
- **Inter-commodity spread**: Long one index future, short another (e.g., NQ vs ES)
- **Calendar spread**: Same instrument, different expiries
- **Butterfly/Condor**: 3 or 4 legs for convexity trades
- **Bull/Bear spread**: Directional bias within the same complex

### 1.2 Universe
6 pairs from 4 instruments: NQ/ES, NQ/RTY, NQ/YM, ES/RTY, ES/YM, RTY/YM

### 1.3 Margin Advantage
SPAN margins for spreads are significantly lower than outright positions (typically 50-80% reduction), improving capital efficiency.

### 1.4 Why Spreads Work
- Spreads remove common market factors (macro risk, VIX shocks)
- Residual captures relative value (sector rotation, flow imbalances)
- Mean-reversion is more reliable on spreads than on individual assets

---

## 2. COINTEGRATION THEORY

### 2.1 Definition
Two price series X(t) and Y(t) are cointegrated if there exists β such that:
```
Z(t) = Y(t) - β·X(t)  ~  I(0)   (stationary)
```
Both X and Y are I(1) individually, but the linear combination Z is I(0).

### 2.2 Engle-Granger Two-Step
1. OLS regression: Y(t) = α + β·X(t) + ε(t) → estimate β̂
2. ADF test on residuals ε̂(t) → reject unit root ⟹ cointegration confirmed

Critical values (Engle-Granger, stricter than standard ADF):
- 1%: -3.90, 5%: -3.34, 10%: -3.04

### 2.3 Johansen Test (VECM)
- Tests for cointegration rank r in a multivariate system
- Trace statistic and max-eigenvalue statistic
- Provides both β (cointegrating vector) and α (adjustment speeds)
- Preferred for >2 variables or when Engle-Granger has low power

### 2.4 ADF Test (Augmented Dickey-Fuller)
Tests H₀: unit root (non-stationary) vs H₁: stationary
```
ΔZ(t) = μ + γ·Z(t-1) + Σ δᵢ·ΔZ(t-i) + ε(t)
```
- Reject H₀ if test statistic < critical value (more negative)
- Lag selection: AIC or BIC
- For spread residuals: use Engle-Granger critical values, not standard ADF

### 2.5 Phillips-Perron Test
- Non-parametric correction for serial correlation
- More robust to heteroskedasticity than ADF
- Complementary to ADF (use both)

---

## 3. ORNSTEIN-UHLENBECK PROCESS

### 3.1 Continuous-Time Model
```
dZ(t) = θ·(μ - Z(t))·dt + σ·dW(t)
```
- θ: mean-reversion speed (θ > 0 required)
- μ: long-run mean
- σ: volatility of the spread
- W(t): standard Brownian motion

### 3.2 Half-Life
Time for the spread to revert halfway to its mean:
```
t_half = ln(2) / θ
```
Estimation from discrete AR(1): Z(t) = a + b·Z(t-1) + ε(t)
```
θ = -ln(b) / Δt
t_half = -ln(2) / ln(b)
```
- Typical range for intraday 5min: 20-100 bars (1h40 to 8h20)
- t_half < 20 bars: too fast, likely noise
- t_half > 200 bars: too slow, not exploitable intraday

### 3.3 Hurst Exponent
- H < 0.5: mean-reverting (target for spread trading)
- H = 0.5: random walk
- H > 0.5: trending
- Estimation: Rescaled Range (R/S) or DFA (Detrended Fluctuation Analysis)
- For spread trading: require H < 0.45 for reliable mean-reversion

---

## 4. HEDGE RATIO METHODS

### 4.1 OLS Rolling
```
β(t) = Cov(Y, X)[t-w:t] / Var(X)[t-w:t]
```
- Window w: typically 60-252 bars (5min)
- Pros: simple, interpretable
- Cons: look-ahead bias at window edges, lagging, window choice arbitrary
- Implementation: `statsmodels.OLS` or `np.polyfit`

### 4.2 Kalman Filter (Dynamic Hedge Ratio)

#### State-Space Model
```
Observation:  Y(t) = β(t)·X(t) + ε(t),     ε(t) ~ N(0, R)
State:        β(t) = β(t-1) + η(t),          η(t) ~ N(0, Q)
```
β follows a random walk — no mean-reversion assumption on β itself.

#### Kalman Recursion
```
PREDICT:
  β̂(t|t-1) = β̂(t-1|t-1)                    (state prediction)
  P(t|t-1) = P(t-1|t-1) + Q                  (covariance prediction)

UPDATE:
  ν(t) = Y(t) - β̂(t|t-1)·X(t)              (innovation / prediction error)
  F(t) = X(t)²·P(t|t-1) + R                  (innovation variance)
  K(t) = P(t|t-1)·X(t) / F(t)               (Kalman gain)
  β̂(t|t) = β̂(t|t-1) + K(t)·ν(t)           (state update)
  P(t|t) = (1 - K(t)·X(t))·P(t|t-1)         (covariance update)
```

#### Hyperparameters
- **Q** (state noise): controls adaptation speed. Higher Q → faster adaptation, noisier β
- **R** (observation noise): estimated from data or set as spread variance
- **α = Q/R ratio**: key tuning parameter
  - α ~ 10⁻⁵ to 10⁻⁶ for daily data
  - α ~ 10⁻⁴ to 10⁻⁵ for 5min data (adjust for higher frequency)
  - Too high: β overreacts to noise
  - Too low: β too sluggish, equivalent to OLS

#### Delta Parameter Approach (Ernie Chan)
Alternative parameterization using a single scalar delta:
```
Q = delta / (1 - delta) * I
```
- delta = 1e-5: Very smooth β, slow adaptation (≈ long OLS window)
- delta = 1e-4: Moderate adaptation (recommended starting point for 5min)
- delta = 1e-3: Fast adaptation, noisier (≈ short OLS window)
- delta = 1e-2: Too noisy usually
- **For 5min bars: delta ~ 1e-5 to 1e-4. For daily: delta ~ 1e-4 to 1e-3**

#### Initialization
- β̂(0) = OLS estimate on first N bars (e.g., 252), or [α=0, β=1]
- P(0) = 1.0 (diffuse prior — "I don't know", filter converges in ~50-100 bars)
- R = variance of OLS residuals (or 1e-3 as starting point)
- Q = α · R (or use delta approach above)
- **Warm-up**: discard first 50-200 bars as filter converges from prior

#### Z-Score Kalman (Innovation-Based)
```
z(t) = ν(t) / √F(t)
```
- ν(t): innovation (prediction error)
- F(t): innovation variance
- **Auto-adaptive**: no rolling window needed
- Under correct model: z(t) ~ N(0,1) (normalized innovations)
- This is the PREFERRED z-score method for Kalman-based systems

### 4.3 Kalman Variants

#### Momentum Kalman (3 states)
```
State: [β(t), μ(t), δ(t)]  where δ = momentum (drift on μ)
β(t) = β(t-1) + η_β(t)
μ(t) = μ(t-1) + δ(t-1) + η_μ(t)
δ(t) = δ(t-1) + η_δ(t)
```
- Captures trending behavior in the spread level
- 3x3 Q matrix, more complex to tune

#### AR(1) / Partial Cointegration Kalman
```
m(t) = ρ·m(t-1) + η(t),   |ρ| < 1
Z(t) = m(t) + R(t),        R(t) ~ random walk
```
- Decomposes spread into mean-reverting component m(t) and permanent component R(t)
- ρ controls mean-reversion strength
- More realistic than assuming full stationarity

### 4.4 Dollar Neutral
```
β = (Price_A × Multiplier_A) / (Price_B × Multiplier_B)
```
- Equal notional value on both legs
- Recalculated each bar (dynamic)
- No statistical assumption — purely economic

### 4.5 Volatility Neutral
```
β = σ_A / σ_B
```
- σ estimated as rolling realized volatility (e.g., 20-bar std of returns)
- Equal risk contribution from both legs
- Particularly useful when volatility regimes differ significantly

---

## 5. Z-SCORE CONSTRUCTION

### 5.1 Classical Z-Score
```
z(t) = (S(t) - μ_S) / σ_S
```
Where S(t) = spread, μ_S and σ_S computed over rolling window.

### 5.2 Kalman Z-Score (Preferred)
```
z(t) = ν(t) / √F(t)
```
No window parameter. Auto-adaptive through Kalman filter.

### 5.3 Bollinger-Style Z-Score
```
z(t) = (S(t) - SMA(S, w)) / (k · StdDev(S, w))
```
Equivalent to classical with explicit band width k.

### 5.4 EMA Z-Score
```
z(t) = (S(t) - EMA(S, w)) / EMA_std(S, w)
```
Exponential weighting gives more weight to recent observations.

---

## 6. SIGNAL CONSTRUCTION & OPTIMAL THRESHOLDS

### 6.1 Standard Z-Score Signals
- **Entry Long Spread**: z < -z_entry (spread undervalued)
- **Entry Short Spread**: z > +z_entry (spread overvalued)
- **Exit**: z crosses zero (or ±z_exit, typically z_exit < z_entry)
- **Stop-Loss**: |z| > z_stop (regime break)
- Typical: z_entry = 2.0, z_exit = 0.5, z_stop = 4.0

### 6.2 Bertram (2010) Optimal Thresholds
For OU process with known parameters (θ, σ, μ):
```
Optimal entry/exit thresholds maximize:
E[profit] / E[duration] = (2a - c) / E[T(a→b) + T(b→a)]
```
Where:
- a: entry threshold (distance from mean)
- b: exit threshold
- c: transaction costs
- T(a→b): first-passage time from a to b

First-passage time for OU:
```
E[T(a→0)] = (1/θ) · Σ_{n=0}^∞ ((-1)^n · (a√(2θ)/σ)^(2n+1)) / ((2n+1)·n!)
```

In practice, solved numerically. Key insight: **optimal thresholds depend on θ, σ, and transaction costs** — not arbitrary.

### 6.3 Zeng & Lee Extension
- Incorporates asymmetric costs (different for long/short entry)
- Accounts for margin requirements in threshold optimization
- Provides closed-form approximation for certain parameter ranges

---

## 7. CISM MODEL (Cointegrated Ising Spin Model)

### 7.1 Overview
Agent-based model combining:
- VECM (Vector Error Correction) for cointegration dynamics
- Ising spin model for agent interactions

### 7.2 Three Agent Types (Triad)
1. **Arbitrageurs**: Trade mean-reversion, force spread back to equilibrium
2. **Herding agents**: Follow crowd, amplify momentum
3. **Momentum traders**: Trade trends, create persistence

### 7.3 Dynamics
```
ΔP_i(t) = α_i·(Z(t-1)) + Σ β_ij·ΔP_j(t-1) + Δ-weighted arbitrage force + ε(t)
```
The Δ-weighted arbitrage force depends on the proportion of each agent type active.

### 7.4 Application
- Regime detection: which agent type dominates → mean-reversion vs trending
- Signal filtering: only trade when arbitrageurs dominate
- Advanced: hybrid Kalman → CISM architecture where Kalman provides β and CISM provides regime context

---

## 8. PERFORMANCE METRICS

### 8.1 Core Metrics
- **Sharpe Ratio**: E[R] / σ(R), annualized. Target > 1.5 for spread strategies
- **Calmar Ratio**: CAGR / Max Drawdown. Target > 1.0
- **Profit Factor**: Gross Profit / Gross Loss. Target > 1.5
- **Hit Ratio**: % winning trades. Typical 55-65% for mean-reversion
- **Max Drawdown**: Maximum peak-to-trough decline
- **Average Trade Duration**: Should align with half-life

### 8.2 Risk Metrics
- **VaR / CVaR**: Tail risk measures
- **Max Consecutive Losses**: Psychological/practical limit
- **Recovery Time**: Time from drawdown trough to new equity high

---

## 9. IMPLEMENTATION GUIDELINES

### 9.1 Pipeline Order
1. Load & clean data (session filter, gaps)
2. Resample 1min → 5min
3. Align pairs (inner join on timestamp)
4. Compute hedge ratios (OLS Rolling, Kalman, Dollar Neutral, Vol Neutral)
5. Construct spreads: S(t) = P_a(t) - β(t)·P_b(t)
6. Compute z-scores
7. Run statistical tests (ADF, Hurst, half-life)
8. Generate signals (entry/exit/stop based on z-score thresholds)
9. Backtest with transaction costs
10. Optimize parameters (walk-forward, Optuna)

### 9.2 Kalman Implementation Tips
- Use `filterpy.kalman.KalmanFilter` (maintained, real-time friendly) or `pykalman` (has EM for param estimation, batch-oriented)
- For 2-state model [α, β]: H_t = [1, x_t] (time-varying observation matrix)
- For 1D state (β only): scalar Kalman is sufficient, no matrix operations needed
- Initialize with OLS on first 252 bars minimum
- **Numerical stability**: use Joseph form for covariance update:
  ```
  P(t|t) = (I - K·H)·P(t|t-1)·(I - K·H)ᵀ + K·R·Kᵀ
  ```
  Avoids negative definite P from floating-point errors.
- Log-likelihood for hyperparameter optimization:
  ```
  LL = -0.5 · Σ [ln(F(t)) + ν(t)²/F(t)]
  ```
- Monitor P(t): if it grows unbounded, Q is too large or model is misspecified
- **Overnight gaps**: consider widening P at session boundaries
- **EM algorithm** (pykalman): can estimate Q and R from data, but use on training set only to avoid overfitting

### 9.2.1 Recommended Starting Parameters (5min bars, index futures)
| Parameter | Range | Starting Point |
|-----------|-------|----------------|
| delta | 1e-5 to 1e-3 | 1e-4 |
| V_e (obs noise) | 1e-4 to 1e-2 | 1e-3 or OLS residual var |
| P_0 (initial cov) | 1.0 to 10.0 | 1.0 (diffuse) |
| θ_0 (initial state) | [0, OLS_β] | [0, 1] |
| Warm-up | 50-200 bars | 100 bars |
| Z-score entry | 1.5 to 2.5 | 2.0 |
| Z-score exit | 0.0 to 1.0 | 0.5 |
| Z-score stop | 3.5 to 5.0 | 4.0 |

**Optimization**: grid search delta over [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], evaluate by Sharpe + half-life of Kalman spread.

### 9.3 Transaction Costs
- Commission: $2.50/contract/side (configurable)
- Slippage: 1 tick per leg (conservative)
- Total round-trip per pair: 2 × (commission + slippage) × 2 legs

### 9.4 Walk-Forward Optimization
- Train window: 6-12 months
- Test window: 1-3 months
- Step: 1 month
- Prevents overfitting by out-of-sample validation at each step

---

## 10. KEY ACADEMIC REFERENCES

1. **Engle & Granger (1987)** — Cointegration and Error Correction
2. **Johansen (1991)** — Estimation and Hypothesis Testing of Cointegration Vectors
3. **Hamilton (1994)** — Time Series Analysis (textbook)
4. **Vidyamurthy (2004)** — Pairs Trading: Quantitative Methods and Analysis
5. **Elliott et al. (2005)** — Pairs Trading (mean-reversion strategies)
6. **Bertram (2010)** — Analytic Solutions for Optimal Statistical Arbitrage Trading
7. **Clegg & Krauss (2018)** — Pairs Trading with Partial Cointegration
8. **Zeng & Lee (2014)** — Pairs Trading: Optimal Thresholds and Profitability
9. **Liew & Wu (2013)** — Pairs Trading: A Copula Approach
10. **Krauss (2017)** — Statistical Arbitrage Pairs Trading Strategies: Review and Outlook
11. **Bock & Mestel (2009)** — CISM: A Mean-Reverting Cointegrated Model with Agent Heterogeneity
12. **Harvey (1989)** — Forecasting, Structural Time Series Models and the Kalman Filter
13. **Kim & Nelson (1999)** — State-Space Models with Regime Switching
14. **Avellaneda & Lee (2010)** — Statistical Arbitrage in the US Equities Market
