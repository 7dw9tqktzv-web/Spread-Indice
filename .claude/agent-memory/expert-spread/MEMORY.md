la mémoire# Expert Spread — Persistent Memory

## Validated Findings

### Hurst Exponent
- R/S method biased ~0.99 on spread levels (cumsum-on-cumsum effect) — DO NOT USE
- Variance-ratio method works on levels directly: H~0.41 median on NQ/ES spread
- Threshold H < 0.45 for mean-reversion confirmed by literature (Borondo 2024, Ramos-Requena 2017)
- References: Weron (2002) "DFA unanimous winner", Ernie Chan variance-ratio on log-prices
- Bui & Slepaczuk (2022, Physica A): Hurst method on Nasdaq 100 stocks — cannot outperform benchmark, sensitive to rebalancing period

### Cointegration — NQ/ES
- ADF simple stat median ~-1.68 (many bars not stationary)
- Only 11% of bars pass ADF < -2.86 threshold
- Kalman innovation z-score (v/sqrt(F)) with alpha_ratio=1e-6: std=0.48, rarely reaches +/-2.0
- NQ/ES is the only pair with confirmed symmetric (long+short) edge

### Backtest Results
- OLS rolling: systematically losing on all configs (z_entry 2.0-3.0, all OLS windows)
- Kalman a=1e-6, entry=2.0, conf=50%: 24 trades, 79% win, +$8,575, PF 3.45
- Kalman a=1e-7, entry=2.5, conf=50%: 55 trades, 56% win, +$6,123, PF 1.33
- NQ_RTY profit (+$21,650) comes from long bias (NQ outperformance), not mean-reversion

### Confidence Scoring
- Binary regime filter (all 4 conditions AND) gives 0% pass rate
- Continuous scoring (ADF 40%, Hurst 25%, Corr 20%, HL 15%) with ADF gate at -1.00 works
- Sweet spot: min_confidence=50% balances trade count vs quality

### Log Prices vs Raw Prices in Kalman Spread Trading (Feb 2026)

**Ernie Chan (2013 blog post)**: Key distinction:
- **Price spread** hA*yA - hB*yB stationary => keep **number of shares** fixed in ratio hA:hB
- **Log price spread** hA*log(yA) - hB*log(yB) stationary => keep **market values** fixed in ratio hA:hB (requires rebalancing)
- "For most cointegrating pairs, both price and log price spreads are stationary, so it doesn't matter"
- For pairs where ONLY log spread cointegrates: means **returns differences** are mean-reverting (not price differences)
- URL: http://epchan.blogspot.com/2013/11/cointegration-trading-with-log-prices.html

**Theoretical justification for log prices**:
1. Log-returns are additive (not multiplicative) — state-space models assume additive noise
2. Log transformation stabilizes variance (heteroscedasticity reduction) — important for KF Gaussian assumption
3. Cointegration theory (Engle-Granger) works on I(1) series; log-prices are I(1) while raw prices can be I(2) for volatile assets
4. The beta in log-space has a cleaner economic interpretation: elasticity (% change in A per % change in B)
5. For dollar-neutral sizing: log-beta directly gives the dollar ratio (market value ratio)

**Practical implication for our system**:
- Our Kalman model: ln(P_a) = alpha + beta * ln(P_b) + epsilon
- The innovation z-score nu/sqrt(F) is computed in log-space
- The spread S = ln(P_a) - beta*ln(P_b) is in log-units
- This is consistent with Palomar (2025), Chan (2013), and the majority of literature
- Using raw prices would work for strongly cointegrated pairs but log is more theoretically sound

### Kalman alpha_ratio Parameter Ranges (Feb 2026)

**Consolidated from all sources**:
| Source | Parameter | Value | Context |
|--------|-----------|-------|---------|
| Chan (2013) | delta | 1e-4 | ETF daily (EWA/EWC, TLT/IEI) |
| QuantStart | delta | 1e-4 | ETF daily (Chan implementation) |
| PTL (production) | delta | 1e-4 | Equities daily, "very sensitive" |
| QuantInsti (China futures) | delta | 1e-3 | Futures intraday (higher freq) |
| Palomar (2025) basic | alpha | 1e-5 | ETF daily (EWA/EWC) |
| Palomar (2025) momentum | alpha | 1e-6 | ETF daily (smoothest) |
| Our system (NQ_ES) | alpha_ratio | 1e-6, 1e-7 | Index futures 5min |

**Key observations**:
- Literature spans **4 orders of magnitude**: 1e-3 to 1e-7
- **Higher frequency data needs LARGER delta** (more state variation per bar) — QuantInsti uses 1e-3 for intraday
- **BUT our Q = alpha_ratio * R * I** (scale-aware), not Q = delta * I — so our alpha_ratio is not directly comparable to Chan's delta
- Since R = Var[OLS residuals], our effective Q already scales with spread noise level
- Palomar's formula: Q = alpha * Var[eps_LS] * diag(1, 1/Var[y2]) — closest to ours
- **Warning from Palomar**: "If spread variance becomes too small, profit may totally disappear after transaction costs" — alpha cannot be too small
- **Warning from PTL**: "This model is very sensitive to delta parameter"
- Reddit (r/algotrading 2017): "Optimizing delta vs not optimizing delta out of sample perform almost the same" — suggests moderate sensitivity

### Kalman Innovation Z-Score vs Rolling Z-Score

**Key difference for threshold selection**:
- Innovation z-score nu/sqrt(F) is **auto-adaptive** — F includes current state uncertainty
- When KF is well-converged, innovation z-score ~ N(0,1) by construction
- This means z=2.0 in innovation space is more extreme than z=2.0 in rolling z-score
- Chan/QuantStart use entry=1.0, exit=0.0 for innovation z-score
- PTL uses entry=1.0-2.0 for Kalman (lower than OLS entry=1.5-2.5)
- Our OLS grid found z_entry=3.0-3.15 optimal — for Kalman, expect optimal to be LOWER
- **Recommendation**: Test z_entry from 1.0 to 3.5 for Kalman (wider range than OLS)

### Kalman Grid Search on NQ_YM — Key Findings (Feb 2026)

**Grid results (290,304 combos)**: Best Kalman configs have PF 1.86-2.35 but only 50-70 trades over 5 years.

**Innovation z-score threshold confirmation**:
- Optimal z_entry for Kalman on NQ_YM: 1.0-2.25 (vs OLS 3.15) — confirms literature prediction
- Chan/QuantStart entry=1.0, PTL entry=1.0-2.0 — our grid confirms this range
- Innovation z-score ~ N(0,1) when well-converged, so z=1.75-2.25 is already 2+ sigma equivalent

**Alpha sweet spot confirmed**:
- a=1e-7 to 5e-7 optimal (very slow adaptation) for NQ_YM 5min
- a >= 2e-6: rapidly declining quality. a >= 5e-4: ZERO profitable configs
- Consistent with Palomar momentum (1e-6) being smoothest, and our scale-aware Q formula
- The scale-aware Q = alpha * R * I means our 1e-7 is effectively more conservative than Chan's 1e-4

**Tight stop z=2.5 dominates in Kalman**:
- z_stop=2.5 across all top configs (vs OLS z_stop=4.5)
- Expected because innovation z-score is narrower distribution than rolling z-score
- z=2.5 in innovation space corresponds roughly to z=4-5 in rolling z-score space
- Literature (Chan): no explicit stop-loss for Kalman; PTL: max z >10 (very wide)
- Ernie Chan (2011 blog): evolved view — stop-loss IS needed for mean-reversion to handle survivorship bias and black swans

### Minimum Trade Count for Statistical Validity (Feb 2026)

**Literature consensus**:
| Source | Minimum | Notes |
|--------|---------|-------|
| CLT heuristic | 30 | Absolute floor, not a target |
| Van Tharp | 30-50 | "Bare minimum" for position-sizing testing |
| Kevin Davey | 100-300 | Account for walk-forward validation |
| Lopez de Prado (2018) | 200-500 | Institutional-grade, accounts for overfitting |
| BacktestBase review | 100+ with 5yr regime diversity | Trade count AND time period AND regimes needed |

**Key formula**: n = (Z^2 * p * (1-p)) / E^2
- For 80% WR, 95% conf, 5% error: n = (1.96^2 * 0.8 * 0.2) / 0.05^2 = 246 trades
- For 80% WR, 90% conf, 10% error: n = (1.645^2 * 0.8 * 0.2) / 0.10^2 = 43 trades

**Critical nuance**: "500 trades in 6 months (one regime) is less reliable than 100 trades over 5 years (multiple regimes)" — regime diversity matters as much as count.

**For our 50-70 Kalman trades over 5 years**: Statistically borderline (above CLT floor, below institutional grade). Regime coverage is good (5 years spans multiple cycles). Must complement with: permutation test, walk-forward, Monte Carlo. NOT sufficient for standalone deployment.

**URLs**:
- https://www.backtestbase.com/education/how-many-trades-for-backtest
- Bailey & Lopez de Prado (2014): SSRN 2460551

### Stop Loss in Mean-Reversion: Chan's Evolved View (Feb 2026)

**Ernie Chan (2011 blog post)**: "Stop loss, profit cap, survivorship bias, and black swans"
- Original view: no stop-loss for mean-reversion (loss = "too early", not wrong)
- **Evolved view**: stop-loss IS rational because:
  1. **Survivorship bias**: we only trade pairs that historically mean-reverted; some will stop mean-reverting
  2. **Black swans**: events outside backtest period can cause catastrophic loss
- "If you never had a loss of 20% in a single day, a stop-loss of 20% has no effect on backtest... but no one can say it won't happen"
- **Recommendation**: use stop-loss even for mean-reversion, sized to protect against regime breaks

**Ernie Chan (2011 blog post)**: "When cointegration of a pair breaks down"
- Cointegration can break down for 6+ months then mysteriously return
- Need to go beyond technicals into fundamentals to assess if breakdown is temporary
- URL: http://epchan.blogspot.com/2011/06/when-cointegration-of-pair-breaks-down.html

### Combining OLS + Kalman: Literature on Ensemble Methods (Feb 2026)

**No direct literature** found on combining OLS + Kalman for spread trading signals specifically. However:

1. **Kalman as OLS confirmation filter**: Natural approach — use Kalman innovation z-score direction to confirm OLS signal direction. If OLS says LONG but Kalman z-score is not negative, skip.

2. **Dynamic Model Averaging (DMA)**: Risse & Ohl (2017, J. Empirical Finance) combine multiple state-space models with time-varying weights via Kalman filter + dynamic Occam's window. Showed superior performance on stock/gold markets. Could weight OLS and Kalman signals dynamically.

3. **ML + Kalman**: Guliyev (2025, GitHub ml-kalman-pairs-trading) combines Kalman spread with Random Forest trade filtering. Three stages: static OLS -> Kalman -> ML+Kalman. Each improves on previous.

4. **Robust Kalman + HMM**: Ewan KW (GitHub) combines robust Kalman filter with Hidden Markov Model for regime detection. HMM identifies market regimes, Kalman adapts within each regime.

5. **Moerdijk (2025, Leiden thesis)**: Compared Kalman vs Transformer vs Time-MoE for spread prediction. Deep learning had better prediction accuracy but Kalman had better z-score breach detection (better trading performance). "KF was able to more accurately capture z-score breaches, triggering trading signals correctly more often."

**Practical recommendation for our system**: The simplest ensemble is Kalman z-score as confirmation filter on OLS signals. More sophisticated: DMA with time-varying weights. ML filtering adds complexity but Guliyev's results suggest improvement.

### Kalman Warmup and Initialization (Feb 2026)

**Zhao & Huang (2017, ADCONIP)**: "On Initialization of the Kalman Filter"
- Poor initial guess propagates as undiscovered bias through recursions
- Large initial P (high uncertainty) is standard practice but not systematic
- Two initialization methods proposed for large initial uncertainties

**Palomar (2025)**: Initialize from LS estimates on training window — our approach (OLS on first 1000 bars for R estimate)

**Practical observations**:
- Our warmup=100 bars is 8.3 hours at 5min — likely insufficient for full P convergence with very small alpha
- With alpha=1e-7 and Q = 1e-7 * R * I, the state update is extremely slow
- P converges when P_predict ~ P_update, roughly when Q contribution dominates initial P uncertainty
- For alpha=1e-7, convergence could take 500-1000+ bars
- **Recommendation**: warmup should scale inversely with alpha_ratio. For 1e-7, warmup=500-1000. For 1e-6, warmup=200-300.

**Reddit r/statistics**: "P is part of the state, you only choose the initial one, which typically is just a diagonal matrix signifying your lack of information"

### Deflated Sharpe Ratio and Grid Search Overfitting (Feb 2026)

**Bailey & Lopez de Prado (2014, J. Portfolio Management)**:
- DSR corrects for selection bias (multiple testing) and non-normal returns
- Key formula: DSR = PSR(SR*) where SR* = sqrt(V[SR]) * ((1 - gamma) * Z_alpha + gamma * Z_alpha_N)
- **Critical insight for grid search**: The expected max SR from N independent trials on pure noise is:
  `E[max SR] ~ sqrt(2 * ln(N))` (Bonferroni-like correction)
  - N = 100 trials: E[max SR] ~ 3.03
  - N = 1,000 trials: E[max SR] ~ 3.72
  - N = 10,000 trials: E[max SR] ~ 4.29
  - N = 1,500,000 trials: E[max SR] ~ 5.53
- **Implication for our grid**: 1.5M combos means ANY config with Sharpe < ~2.0 could be noise
- **Mitigation**: Walk-forward + permutation test already applied. But increasing grid to 5M+ combos would raise the bar further
- **Recommendation**: Do NOT increase granularity beyond ~2M combos without proportionally stronger OOS validation
- Reference: Bailey & Lopez de Prado (2014), SSRN 2460551

### Topstep 150K Express Funded Account Rules (Feb 2026)

**Express Funded Account ($150K)**:
- Max Loss Limit: $4,500 (trailing, from highest end-of-day balance)
- Daily Loss Limit: -$3,000
- Payout: 5 Benchmark Days ($200+ each), then 50% of balance up to $5,000/payout, 90/10 split
- Scaling plan: contracts limited by balance level
- Account starts at $0 balance — trader must profit to withdraw

**Live Funded Account ($150K)** (after promotion from Express):
- Start with 20% of balance ($30K), unlock more via milestones
- Dynamic risk expansion: daily loss limit adjusts with account balance
- 30 winning Live days of $150+ to unlock full balance + daily payouts

**Key constraint for our system**: MaxDD $4,500 is the HARD ceiling. Our OLS Config E has MaxDD $5,190 — ALREADY OVER THE LIMIT. Only K_RunnerPF ($5,925) and K_BestPF ($7,705) are also over. Need configs with MaxDD < $3,500 to have safety margin.

### Combining Independent Trading Systems — Literature (Feb 2026)

**Nurp (2025, "Uncorrelated Strategies: Holy Grail of Portfolio Construction")**:
- Markowitz + Dalio: portfolio of uncorrelated return streams dramatically reduces risk
- Key formula: portfolio_vol = sqrt(sum(w_i^2 * vol_i^2) + 2*sum(w_i*w_j*cov_ij))
- If correlation = 0: portfolio_vol = sqrt(sum(w_i^2 * vol_i^2)) << sum(w_i * vol_i)
- N uncorrelated strategies with equal vol: portfolio_vol = individual_vol / sqrt(N)

**Reddit r/algotrading (2024, practical consensus)**:
- **Independent parallel**: simplest, run on separate capital allocations
- **Confluence**: trade only when k strategies agree on direction — reduces volume, increases quality
- **Laddering**: more agreeing signals = larger position — risk-weighted ensemble
- **Rollover**: rotate capital to best-performing strategy — momentum allocation

**Moerdijk (2025, Leiden thesis)**:
- Kalman Filter z-score breach detection BETTER than Transformer/MoE for trading signals
- Deep learning better at prediction accuracy but worse at signal timing
- "KF was able to more accurately capture z-score breaches, triggering trading signals correctly more often"

**Yang, Huang & Chen (2023, SSRN 4590815)**:
- Hierarchical pair trading: ML clustering for pair selection + Kalman for signal generation
- KF eliminates noise, generates dynamic trading signals with improved returns/stability

**Guliyev (2025, GitHub ml-kalman-pairs-trading)**:
- Three progressive stages: Static OLS -> Kalman -> ML+Kalman (Random Forest trade filter)
- Each stage improves on previous — RF filters out low-probability Kalman signals

**Key finding for OLS+Kalman combination**:
- Our data shows OLS and Kalman lose on DIFFERENT days (3% overlap on losing days for K_BestPnL)
- OLS loses 8h-9h (post-US-open), Kalman loses 4h+13h (window edges)
- This is near-zero correlation on losing days = ideal for parallel system
- PARALLEL INDEPENDENT is the correct approach (not confirmation filter)
- Confirmation filter would REDUCE volume without adding quality (they signal on different days/times)

## Open Questions
- Engle-Granger critical values (-3.34) stricter than standard ADF (-2.86) — should we switch?
- Bertram (2010) optimal thresholds based on OU parameters — not yet implemented
- Kalman momentum variant (Palomar) with 3-state [mu, gamma, gamma_dot] — worth testing?
- Dynamic Model Averaging between OLS and Kalman — implementation complexity vs benefit?

---

## Literature Review: Inter-Index Spread Trading (Feb 2026)

### Source 1: Palomar (2025) — "Portfolio Optimization" Textbook, Ch.15
**URL**: https://bookdown.org/palomar/portfoliooptimizationbook/15.6-kalman-pairs-trading.html

**Models Used**: OLS Rolling, Kalman Filter (basic + momentum variant)

**Kalman Filter Specifications**:
- State: alpha_t = (mu_t, gamma_t) — mean and hedge ratio as hidden states
- Observation: y1t = [1, y2t] * [mu_t, gamma_t] + epsilon_t
- Transition: T = Identity (random walk on states)
- Noise: epsilon_t ~ N(0, sigma_e^2), eta_t ~ N(0, Q)
- **Q = alpha * Var[epsilon_LS] * diag(1, 1/Var[y2])** — this is the key formula
- **alpha (hyper-parameter) = 1e-5 for basic Kalman, 1e-6 for Kalman with momentum**
- alpha controls ratio of state variability to spread variability
- Initial states from LS estimates on training window

**Kalman with Momentum** (extended model):
- State: alpha_t = (mu_t, gamma_t, gamma_dot_t) — adds velocity of hedge ratio
- Transition matrix: [[1,0,0],[0,1,1],[0,0,1]]
- **alpha = 1e-6** (one order smaller because momentum smooths more)
- Result: "less noisy hedge ratio, better spread"

**Partial Cointegration Extension** (Clegg & Krauss 2018):
- State: alpha_t = (mu_t, gamma_t, epsilon_t) with AR(1) residual
- Transition: rho = 0.9 (autoregressive parameter, |rho| < 1)

**OLS Rolling Parameters**:
- Lookback: **2 years** (daily data)
- Very noisy hedge ratio (oscillates 0.6-1.2 on EWA/EWC)

**Z-score Construction**:
- Rolling window z-score with **6-month lookback** for mean and std
- Threshold strategy: entry at **s0 = 1.0** (1 std deviation)
- Normalized spread: z_t = (y1t - gamma*y2t - mu) / (1 + gamma)

**Results (EWA/EWC, 2013-2022, no transaction costs)**:
- Rolling LS: cumulative return 0.6
- Basic Kalman (alpha=1e-5): cumulative return 2.0
- Kalman momentum (alpha=1e-6): cumulative return 3.2
- Kalman methods: "much better drawdown, less noisy curves"

**Key Insight**: "If spread variance becomes too small [from Kalman smoothing], profit may totally disappear after transaction costs" — alpha cannot be too small.

---

### Source 2: Ernie Chan (2013) — "Algorithmic Trading" Book
**URL**: Referenced in QuantStart, Hudson & Thames, Palomar

**Models Used**: Kalman Filter on ETF pairs (EWA/EWC, TLT/IEI)

**Kalman Filter Specifications (Chan's formulation)**:
- State: x_t = [intercept, slope] (alpha, beta)
- Observation: y_t = [1, price_B] * x_t + epsilon
- **delta (transition covariance) = 1e-4** — Chan's default in book examples
- **Ve (observation covariance) = 1e-3** — measurement noise
- Q = delta * I (identity scaled by delta)
- R = Ve (scalar)
- Innovation z-score: e_t / sqrt(Qt) where Qt is innovation covariance
- Entry: when |z| exceeds sqrt(Qt), i.e., **effectively 1.0 std of innovation**
- Exit: when z crosses zero (mean reversion complete)

**Key Insight**: Chan uses innovation-based z-score (not rolling), which is self-normalizing.

---

### Source 3: QuantStart — Kalman Filter Pairs Trading (TLT/IEI)
**URL**: https://www.quantstart.com/articles/kalman-filter-based-pairs-trading-strategy-in-qstrader/

**Model**: Kalman Filter (Chan's formulation)

**Z-score Construction**:
- Forecast error = y_observed - y_predicted
- Normalized by sqrt(Qt) (Kalman innovation covariance)
- **"Parameterless" approach**: entry/exit at +/- 1 standard deviation of forecast error
- Effectively entry = 1.0, exit = 0.0 (cross zero)
- No explicit stop-loss mentioned

**Key Insight**: "One parameterless approach is to consider a multiple of the standard deviation of the spread and use these as the bounds. For simplicity we can set the coefficient to one."

---

### Source 4: Pair Trading Lab (PTL) Wiki — Production Platform
**URL**: https://wiki.pairtradinglab.com/wiki/Pair_Trading_Models

**Models Implemented**: Ratio, Residual (OLS), Kalman, Kalman-Grid v2, Kalman-Auto

**Ratio Model Parameters**:
- Entry threshold: **typical 1.5-2.5, default 2.0**
- Exit threshold: **typical -0.5 to 0.5, default 0.0**
- MA period: **typical 10-100, default 15** (EMA)
- Std dev period: **typical 10-100, default 15**
- Max z-score filter: **>4** (to filter regime breaks)
- Entry modes: simple, uptick, downtick

**Residual (OLS) Model Parameters**:
- Entry threshold: **typical 1.2-2.5, default 1.5**
- Exit threshold: **typical -0.5 to 0.5, default 0.0**
- **OLS regression window: typical 15-300 bars** (daily)
- Same window used for std dev calculation
- Dollar-neutral only

**Kalman Model Parameters**:
- **delta (transition covariance): typical 0.0001 (= 1e-4)**
- **Ve (observation covariance): typical 0.001 (= 1e-3)**
- Entry threshold: **typical 1.0-2.0**
- Exit threshold: **typical -1.0 to 0.0**
- Max z-score: **>10** (if used)
- **WARNING from PTL: "This model is very sensitive to delta parameter"**
- Supports dollar-neutral AND beta-neutral

**Kalman-Grid v2 (best performing model)**:
- Auto-estimates delta and Ve via grid of Kalman filters + ML ranking
- Entry threshold: **typical 1.0-4.0**
- Exit threshold: **typical -1.0 to 0.0**
- "Always-in-position" mode: entry=1, exit=-1 with reversals
- **Out-of-sample results (300 pairs, Jan-Sep 2016)**:
  - Kalman-Grid v2 normal: **median CAGR 7.2%, mean 9.5%**
  - Kalman-Grid v2 aggressive: **median CAGR 7.1%, mean 11.4%**
  - Kalman-Grid v1: **median CAGR 4.9%, mean 10.8%**
  - Residual(20): **median CAGR 3.6%, mean 6.6%**
  - Ratio(14): **median CAGR 3.6%, mean 9.0%**

**Key Insight**: Kalman-Grid v2 beats all other models. No parameters to optimize except thresholds. Beta-neutral supported.

---

### Source 5: Barthelemy, Chen, Lucyszyn (2024) — Columbia Engineering
**URL**: https://arxiv.org/html/2412.12555v1

**Model**: OLS with Engle-Granger cointegration test, S&P 500 stocks

**Parameters (pre-optimization)**:
- Z-score entry (theta_in): **2.0**
- Z-score exit (theta_out): **1.0**
- Correlation threshold: **0.8** (pre-filter)
- Cointegration p-value: **0.05** (but nearly all corr>0.8 pairs pass)

**Optimized Parameters (100 pairs average, Bayesian via Optuna)**:
- **theta_in_optimal = 1.42** (std=0.30)
- **theta_out_optimal = 0.37** (std=0.13)
- Lower than expected — "strategy can enter trading zone more rapidly"

**Results**:
- Pre-optimization: 5.2% cumulative return (3 months), std=6.2%
- Post-optimization: same 5.2% return but higher variance
- Max return: +45%, min: -62%, std: 13%
- **Conclusion: optimization does NOT improve out-of-sample returns** — recommended NOT optimizing

**Key Insight**: Optimal thresholds are LOWER than conventional 2.0/1.0 on equities. But out-of-sample performance identical, suggesting overfitting risk.

---

### Source 6: Bertram (2010) + Lipton & Lopez de Prado (2020) — OU Optimal Thresholds
**URLs**:
- Bertram: https://ui.adsabs.harvard.edu/abs/2010PhyA..389.2234B
- Lipton/LdP: DOI 10.1142/S0219024920500569
- Holy & Cerny (2021): https://arxiv.org/abs/2102.04160
- Hudson & Thames: https://hudsonthames.org/optimal-trading-thresholds-for-the-o-u-process/

**Model**: OU process dX = theta*(mu - X)*dt + sigma*dW

**Bertram's Approach**:
- Derives analytic formulae for mean/variance of trade return and duration
- Optimal thresholds maximize **expected return per unit time** or **Sharpe ratio**
- Entry = a (distance above/below mean), Exit = m (near mean)
- Closed-form solution for expected return case
- Depends on OU parameters: theta (mean-reversion speed), mu (mean), sigma (volatility)

**Lipton & Lopez de Prado (2020)**:
- Extends Bertram with **3 outcomes**: (1) profit target, (2) stop-loss, (3) max horizon
- Derives optimal profit-taking and stop-out levels
- Maximizes Sharpe ratio in OU context
- Practical formula — Hudson & Thames has implementation

**Holy & Cerny (2021)**:
- Extends Bertram with **bounded risk** constraint
- Shows estimation imprecision severely impacts optimal strategy
- "Critical for practice" — statistical estimation noise can flip optimal sign

**Key Insight**: Optimal thresholds are NOT fixed (like 2.0 sigma) — they depend on the OU parameters (theta, sigma). For our system, we could derive optimal thresholds from half-life and spread volatility.

---

### Source 7: Milstein et al. (2022/2023) — Neural Augmented Kalman with Bollinger Bands
**URL**: https://arxiv.org/abs/2210.15448

**Model**: Kalman Filter + Bollinger Bands + Neural Network (KalmanNet)

**Approach**:
- Standard Kalman Filter for state estimation
- Bollinger Bands on forecast error for entry/exit signals
- Neural network augments Kalman gains when model is mismatched
- "Partial co-integration" modeling

**Key Insight**: Even academic research acknowledges KF model mismatch is a real problem. NN augmentation helps but adds complexity.

---

### Source 8: Bui & Slepaczuk (2022) — Hurst on Nasdaq 100
**URL**: https://www.sciencedirect.com/science/article/pii/S037843712100964X

**Model**: Generalized Hurst Exponent for pair selection

**Key Findings**:
- Hurst method **cannot outperform benchmark** on Nasdaq 100
- Results very sensitive to: number of pairs traded, rebalancing period
- Less sensitive to financial leverage
- **Hurst > Cointegration method but Hurst < Correlation method** for pair selection
- Correlation simplest and most effective pair selection filter

**Key Insight**: Hurst as a regime filter may be less useful than correlation for pair selection. But for spread quality assessment (our use case), still valid at H<0.5.

---

### Source 9: Zhao (2025) — Dynamic Pairs Trading for Constrained Markets
**URL**: https://www.scirp.org/pdf/jfrm_2411000.pdf

**Models**: Static (20-day lookback), Dynamic (volatility-adjusted), Hybrid (cointegration + volatility)

**Parameters**:
- **OLS lookback: 20 days** (for the simple strategy)
- Transaction costs: 0.4% round-trip
- Z-score thresholds not specified (Chinese A-share specific)

**Results**:
- Static 20-day: 8.6% annual return, 10.7% max DD
- Dynamic volatility: 13.6% annual, 7.7% max DD
- **Hybrid cointegration+vol: 10.6% annual, 6% max DD, Sharpe ~1.03, 84% positive return probability**

**Key Insight**: Hybrid approach (cointegration + volatility adjustment) gives best risk-adjusted returns.

---

### Source 10: TradeFundrr / Industry Consensus — Common Parameters
**URL**: https://tradefundrr.com/pair-trading-strategies/

**Widely Cited Defaults**:
- Correlation threshold: **>0.80**
- Z-score entry: **2.0 standard deviations**
- Z-score profit target (exit): **0.5 standard deviations** (some use 0.0)
- Z-score stop-loss: **3.0 standard deviations**
- Half-life acceptable range: **1-252 days** (if >252, too slow; if <1, noise)

---

### Source 11: StatArb Pairs (Pham The Anh, 2024) — Bayesian Optimization
**URL**: https://medium.com/funny-ai-quant/statarb-pairs-optimized-mean-reversion-trading-system-9e23defeec8a

**Model**: OLS + Bayesian optimization (kappa + half-life based)

**Pair Selection Pipeline**:
1. Cointegration test (Engle-Granger)
2. Correlation analysis
3. ADF stationarity test
4. Rank by composite score

**Key Parameters**:
- Kappa (OU mean-reversion speed) and half-life used jointly
- Bayesian optimization for thresholds

---

### Source 12: Robot Wealth (Kris Longmore) — Kalman Filter Practical Guide
**URL**: https://robotwealth.com/kalman-filter-pairs-trading-r/

**Key Observations**:
- "Real financial series don't exhibit truly stable, cointegrating relationships"
- "Relationships are constantly evolving and changing"
- Kalman filter handles non-stationarity better than OLS
- Uses pykalman / dlm library defaults
- Recommends starting with Chan's delta=1e-4, Ve=1e-3 and adjusting

---

## Consolidated Parameter Ranges (All Sources)

### Z-Score Thresholds
| Parameter | Conservative | Standard | Aggressive | Our System |
|-----------|-------------|----------|------------|------------|
| z_entry | 2.0-2.5 | 1.5-2.0 | 1.0-1.5 | 2.0-2.5 |
| z_exit | 0.0 | 0.0-0.5 | -0.5-0.0 | 0.5 (FLAT target) |
| z_stop | 3.0-4.0 | 3.0 | N/A | 3.0-3.5 |
| z_max_filter | >4.0 | >4.0 | >10.0 | N/A |

### Kalman Filter Parameters
| Parameter | Chan (2013) | PTL | Palomar (2025) | Our System |
|-----------|------------|-----|----------------|------------|
| delta/alpha | 1e-4 | 1e-4 (sensitive!) | 1e-5 (basic), 1e-6 (momentum) | 1e-5 to 1e-7 |
| Ve/R | 1e-3 | 1e-3 | Var[resid_LS] | 1.0 (implied) |
| Q formula | delta*I | delta*I | alpha * Var[eps] * diag(1, 1/Var[y2]) | alpha * R * I |
| Warmup | 100+ | "unstable period" | TLS samples for LS init | 100 bars |

### OLS Rolling Parameters
| Parameter | PTL | Palomar | Columbia | Our System |
|-----------|-----|---------|----------|------------|
| Window | 15-300 bars (daily) | 2 years | Full period | 7920 bars (30j, 5min) |
| Z-score window | Same as OLS | 6 months | Full period | 12 bars (1h) |

### Regime / Quality Filters
| Metric | Threshold | Source |
|--------|-----------|--------|
| ADF p-value | < 0.05 | Columbia, standard Engle-Granger |
| ADF statistic | < -2.86 (5%) or < -3.34 (EG) | Our system uses -2.86 |
| Hurst | < 0.5 (mean-reverting) | Borondo 2024, Chan |
| Half-life | 1-252 days (daily) | Industry consensus |
| Correlation | > 0.80 | Columbia, TradeFundrr |
| Stop-loss z | 3.0 sigma | Industry consensus |

### Key Takeaways for Our System

1. **Alpha_ratio (Kalman)**: Literature uses 1e-4 to 1e-6. Our 1e-5 to 1e-7 range is reasonable. Palomar uses 1e-5 (basic) and 1e-6 (momentum). Too small = spread variance too low = no profit after costs.

2. **Z-score thresholds**: Industry standard is entry=2.0, exit=0.0, stop=3.0. Columbia's optimization found 1.42/0.37 optimal but no out-of-sample improvement. Our 2.0-2.5 entry is within range.

3. **OLS lookback**: PTL uses 15-300 days, Palomar uses 2 years. Our 7920 bars (30 days at 5min) seems SHORT compared to daily-data literature. But intraday data has more samples per unit time — 30 days of 5min = 7920 samples vs 30 samples daily. The INFORMATION content is similar.

4. **Kalman innovation z-score**: Chan and QuantStart both use e_t/sqrt(Qt) directly, entry at 1.0 std. This is lower than our z_entry=2.0, but the innovation z-score has different distribution properties than rolling z-score.

5. **Bertram optimal thresholds**: Should be implemented — they compute optimal entry/exit from OU parameters (theta from half-life, sigma from spread). Would replace arbitrary z thresholds with analytically optimal ones.

6. **Regime filtering**: No source uses our exact weighted scoring approach. Most use binary (ADF pass/fail + Hurst < 0.5). Our continuous scoring with ADF gate is more sophisticated than literature. The Columbia paper's correlation > 0.8 filter is widely used.

7. **Stop-loss**: Industry consensus is 3.0-4.0 sigma. Ernie Chan argues against stop-loss in mean-reversion (you may be "too early", not wrong). Lipton/LdP 2020 derives optimal stop-out analytically.

8. **Kalman > OLS**: Every source confirms Kalman outperforms OLS rolling. Palomar shows 2x-5x better cumulative returns. PTL's Kalman-Grid v2 beats all models.

9. **Out-of-sample warning**: Columbia (2024) and Holy/Cerny (2021) both warn that optimized parameters do NOT transfer well out-of-sample. Estimation error in OU parameters severely impacts strategy.

10. **Kalman momentum variant**: Palomar's extended model with hedge ratio velocity (gamma_dot) could be interesting — uses alpha=1e-6 and produces even smoother spread. Worth investigating.
