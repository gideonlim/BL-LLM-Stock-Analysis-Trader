# Quantitative Finance Research — Potential Improvements for Our Trading Bot

> **Purpose:** A curated survey of academic research, advanced math, and quantitative finance techniques that could meaningfully improve each layer of our trading system. Organized by component, with implementation priority and difficulty ratings.

---

## Table of Contents

1. [Portfolio Optimization](#1-portfolio-optimization)
2. [Position Sizing](#2-position-sizing)
3. [Signal Generation & Feature Engineering](#3-signal-generation--feature-engineering)
4. [Risk Management](#4-risk-management)
5. [Covariance & Return Estimation](#5-covariance--return-estimation)
6. [Execution & Market Microstructure](#6-execution--market-microstructure)
7. [Backtesting & Overfitting Prevention](#7-backtesting--overfitting-prevention)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Portfolio Optimization

Our bot currently uses marginal Sharpe ranking (Markowitz-based). There are three major upgrades worth considering.

### 1.1 Hierarchical Risk Parity (HRP)

**What it is:** An alternative to Markowitz optimization developed by Marcos López de Prado (2016). Instead of inverting a covariance matrix (which is numerically unstable with many assets), HRP uses hierarchical clustering to group correlated assets, then allocates risk top-down through the tree.

**Why it matters for us:** Our covariance matrix is estimated from just 60 days of returns for 15+ stocks. With that few observations relative to assets, the sample covariance matrix is poorly conditioned. HRP doesn't require matrix inversion, so it works even with singular or near-singular covariance matrices. Empirical tests show HRP produces lower out-of-sample variance than Markowitz minimum-variance, despite Markowitz explicitly optimizing for it.

**The math:** HRP works in three steps. First, compute a distance matrix from correlations: d(i,j) = sqrt(0.5 * (1 - ρ_ij)). Second, apply single-linkage hierarchical clustering to group assets. Third, allocate inversely proportional to cluster variance using recursive bisection — split the portfolio into two clusters, weight each inversely by its variance, then recurse down each subtree.

**Implementation difficulty:** Medium. The `scipy.cluster.hierarchy` module handles the clustering. We'd replace or complement `rank_intents_by_marginal_sharpe()` in `portfolio_optimizer.py`.

**Key papers:**
- López de Prado, "Building Diversified Portfolios that Outperform Out of Sample" (2016, Journal of Portfolio Management)
- Recent 2025 extensions exploring distance metric variants and reinforcement learning integration (RL-BHRP)

### 1.2 Black-Litterman Model

**What it is:** A Bayesian framework (Goldman Sachs, 1990) that starts from market equilibrium returns (implied by market-cap weights via reverse optimization) and blends in the investor's own views with specified confidence levels. The output is a set of expected returns that can be fed into a mean-variance optimizer without the extreme, unstable weights that raw Markowitz produces.

**Why it matters for us:** Our signals already generate views — each BUY signal is implicitly a view that the stock will outperform. Black-Litterman gives us a principled way to combine those views with market priors, weighted by our confidence scores. A HIGH confidence signal would shift expected returns more than a LOW confidence one. This directly addresses the problem that raw Markowitz is hypersensitive to return estimates.

**The math:** The posterior expected returns are: E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q], where π is the equilibrium return vector, P is the view matrix, Q is the view returns, Ω is the uncertainty of views, Σ is the covariance matrix, and τ is a scalar (typically 0.025-0.05).

**Implementation difficulty:** Medium-High. Need to compute equilibrium returns (market cap weights + covariance) and map our confidence scores to view uncertainty (Ω). The `pypfopt` library has a Black-Litterman implementation.

**Key insight for our bot:** Our confidence scoring (0-6) already encodes view uncertainty. A signal with confidence 5 should produce a tighter Ω (more certain view) than confidence 2.

### 1.3 Risk Parity

**What it is:** Instead of optimizing returns, allocate so each asset contributes equally to total portfolio risk. This prevents concentrated risk in a few volatile positions.

**Why it matters for us:** Currently, position sizes come from Half-Kelly (return-based), which can over-concentrate in volatile stocks. Risk parity would ensure no single position dominates portfolio variance.

**The math:** For each asset i, the risk contribution is: RC_i = w_i * (Σw)_i / sqrt(w'Σw). Risk parity sets RC_i = RC_j for all i, j.

**Implementation difficulty:** Medium. Requires iterative optimization (scipy.optimize) but well-understood.

---

## 2. Position Sizing

Our bot uses Half-Kelly. There are more sophisticated approaches.

### 2.1 Volatility Targeting

**What it is:** Instead of sizing by expected return (Kelly), size each position to contribute a target amount of daily volatility. This automatically reduces exposure in high-vol markets and increases it in low-vol markets.

**Why it matters for us:** Half-Kelly uses backtest win rate and profit factor, which are backward-looking and don't adapt to current market conditions. Volatility targeting is forward-looking — if a stock's realized vol doubles, position size automatically halves.

**The math:** For a target portfolio volatility σ_target: w_i = σ_target / (N * σ_i), where σ_i is the asset's realized volatility and N is the number of positions. More sophisticated versions account for correlation: w = σ_target * (Σ^{-1} * 1) / (1' * Σ^{-1} * 1).

**Implementation:** Calculate 20-day realized volatility for each stock (we already compute `vol_20` in signals), then: `position_size_pct = target_vol / (num_positions * stock_vol_20)`. This replaces or blends with Half-Kelly.

**Key research:** Man Group's research shows volatility targeting consistently improves Sharpe ratios for equity portfolios and reduces the probability of extreme returns.

**Implementation difficulty:** Low. We already have `vol_20` in the signal data. This could be a blended approach: `final_size = alpha * kelly_size + (1 - alpha) * vol_target_size`.

### 2.2 Constant Proportion Portfolio Insurance (CPPI)

**What it is:** A dynamic position sizing method that protects a portfolio floor value. Position size is: Exposure = multiplier * (portfolio_value - floor). As the portfolio drops toward the floor, exposure shrinks to zero. As it rises, exposure increases.

**Why it matters for us:** This directly addresses drawdown control. We set a maximum acceptable drawdown (e.g., floor = 90% of equity), and CPPI automatically scales down positions as we approach it — much more graceful than a binary circuit breaker.

**The math:** At each rebalance: cushion = portfolio_value - floor; exposure = m * cushion (where m = 3-5 typically). The floor can ratchet up with a TIPP (Time Invariant Portfolio Protection) variant: floor = max(floor, drawdown_pct * peak_value).

**Implementation difficulty:** Low-Medium. Replace the binary circuit breaker with a continuous scaling function. Position sizes shrink smoothly rather than going to zero when a threshold is hit.

### 2.3 Optimal f (Ralph Vince)

**What it is:** An empirical method that tests various bet fractions against the actual distribution of historical returns to find the fraction that maximizes terminal wealth. Unlike Kelly (which assumes a simple win/loss binary), Optimal f uses the full return distribution.

**Why it matters for us:** Our strategies produce continuous returns, not binary outcomes. Optimal f is theoretically more appropriate than Kelly for this case.

**Caveat:** Optimal f tends to be aggressive — it maximizes growth but can produce large drawdowns. In practice, use a fraction of Optimal f (similar to how we use Half-Kelly).

---

## 3. Signal Generation & Feature Engineering

### 3.1 Market Regime Detection via Hidden Markov Models (HMMs)

**What it is:** HMMs model the market as switching between hidden states (e.g., bull/bear/sideways), each with different return and volatility characteristics. The model infers which state the market is currently in.

**Why it matters for us:** Our strategies use the same parameters in all market conditions. A mean reversion strategy that works in calm markets can be disastrous during a crash. With HMM regime detection, we can: (a) disable certain strategies in unfavorable regimes, (b) adjust position sizes based on regime, (c) use different stop-loss multipliers per regime.

**The math:** An HMM with K states has: transition matrix A (K×K) defining state switching probabilities, emission distributions (typically Gaussian for returns: μ_k, σ_k per state), and initial state probabilities. The Baum-Welch algorithm (EM) estimates parameters; the forward-backward algorithm computes state probabilities.

**Implementation:** Use `hmmlearn` library. Fit a 2-3 state Gaussian HMM on SPY returns. Use the current state probability to scale position sizes: full size in bull regime, half size in neutral, no new positions in bear.

**Key papers:**
- Regime-Switching Factor Investing with Hidden Markov Models (MDPI, 2020) — achieved Sharpe ratios of 1.9 with 3-state HMM
- Multi-model ensemble HMM voting framework for regime shift detection (2025)

**Implementation difficulty:** Medium. The model itself is straightforward with `hmmlearn`. The challenge is choosing the right number of states and features (returns, vol, breadth).

### 3.2 Fractional Differentiation (López de Prado)

**What it is:** A method to make time series stationary while preserving as much memory (trend information) as possible. Standard integer differentiation (returns = price[t] - price[t-1]) makes the series stationary but destroys all memory. Fractional differentiation with d ≈ 0.2-0.4 achieves stationarity while retaining 90%+ correlation with the original series.

**Why it matters for us:** Our strategies compute indicators on raw prices or simple returns. If we used ML features, fractionally differentiated prices would give us stationary inputs (required for ML) that still contain predictive trend information that returns throw away.

**The math:** The fractional difference operator: (1-B)^d = Σ_{k=0}^{∞} C(d,k) * (-B)^k, where B is the backshift operator and C(d,k) = d!/(k!(d-k)!). For d=0 we get the original series; for d=1, standard returns. The key insight is finding the minimum d (call it d*) where the ADF test indicates stationarity.

**Implementation difficulty:** Low. The `fracdiff` Python library implements this efficiently. Use it as a feature preprocessing step before any ML model.

### 3.3 Triple Barrier Method + Meta-Labeling (López de Prado)

**What it is:** Instead of labeling returns as simply up/down after a fixed period, the triple barrier method uses three dynamic barriers: upper barrier (take profit hit first → label +1), lower barrier (stop loss hit first → label -1), and vertical barrier (time expires → label based on return sign). Meta-labeling then trains a secondary ML model to decide whether to act on the primary model's signal.

**Why it matters for us:** Our strategies already produce signals, and we already use bracket orders (SL/TP). The triple barrier method would label historical trades more realistically (matching how our bracket orders actually work). Meta-labeling would add a learned "bet sizing" model on top — rather than uniform Half-Kelly, an ML model learns when each strategy's signals are more or less reliable.

**The connection to our system:** Our bracket orders (entry → SL/TP/time expiry) are literally the triple barrier in production. We can generate labels from historical bracket-order outcomes and train a meta-model to predict signal reliability.

**Implementation difficulty:** Medium-High. Requires generating proper labels from historical data and training a secondary classifier (Random Forest or Gradient Boosting).

### 3.4 Adaptive Moving Averages (KAMA, FRAMA)

**What it is:** Moving averages that automatically adjust their smoothing period based on market conditions. Kaufman's Adaptive Moving Average (KAMA) uses an efficiency ratio; Ehlers' Fractal Adaptive Moving Average (FRAMA) uses fractal dimension.

**Why it matters for us:** Our SMA/EMA crossover strategies use fixed periods (10/50, 9/21). Adaptive MAs would reduce whipsaws in choppy markets while staying responsive in trends.

**Implementation difficulty:** Low. These are drop-in replacements for existing moving average calculations in `strategies.py`.

### 3.5 Factor-Aware Signal Enhancement

**What it is:** Rather than treating each stock independently, incorporate cross-sectional factor exposures (momentum, value, quality, size) to enrich signals.

**Key research findings:**
- Time-series momentum (past 12-month return predicting future return) achieves Sharpe ratios above 1.20 across asset classes
- Dual momentum (combining time-series and cross-sectional) generates 1.88% per month in academic studies
- Quality factor (profitability + growth + payout + safety) combined with momentum shows persistent alpha

**Why it matters for us:** Our signals are purely technical. Adding factor awareness (is this stock in the top momentum quintile? is it a quality company?) would provide orthogonal information.

**Implementation difficulty:** Medium. Requires fundamental data (earnings, book value) beyond our current Yahoo Finance price data.

### 3.6 GARCH Volatility Forecasting

**What it is:** GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models volatility as time-varying, capturing the empirical fact that volatility clusters — high-vol days tend to follow high-vol days.

**Why it matters for us:** We use 20-day realized volatility (`vol_20`), which is backward-looking. GARCH forecasts tomorrow's volatility, which is more useful for position sizing and stop-loss placement. The GJR-GARCH variant also captures the "leverage effect" (volatility rises more after losses than gains).

**The math:** GARCH(1,1): σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}. GJR-GARCH adds: + γ * ε²_{t-1} * I(ε_{t-1} < 0), where I is an indicator for negative returns.

**Implementation:** The `arch` Python library implements GARCH and its variants. Use forecasted volatility for: ATR-based stop losses, position sizing, and regime detection.

**Implementation difficulty:** Low-Medium. The library handles estimation. The work is integrating forecasts into our sizing/stop-loss pipeline.

---

## 4. Risk Management

### 4.1 Conditional Value-at-Risk (CVaR / Expected Shortfall)

**What it is:** While VaR asks "what's the worst loss at the 95th percentile?", CVaR asks "given that we're in the worst 5%, what's the expected loss?" CVaR captures tail risk that VaR misses.

**Why it matters for us:** Our risk manager uses a simple daily loss circuit breaker (3%). CVaR-based risk management would give us a more nuanced view of tail risk. We could constrain portfolio construction to keep CVaR below a threshold, or use CVaR to size positions.

**The math:** CVaR_α = E[L | L > VaR_α] = (1/(1-α)) * ∫_{α}^{1} VaR_u du. For a portfolio, CVaR is a coherent risk measure (unlike VaR) — it's convex, which makes optimization tractable.

**Implementation:** Use historical simulation or parametric estimation. The `scipy.stats` module can compute CVaR from return distributions. Integrate as an additional risk check: reject orders that push portfolio CVaR above a threshold.

**Implementation difficulty:** Medium. The math is straightforward, but estimating CVaR reliably requires sufficient return history.

### 4.2 Dynamic Stop-Loss via ATR Chandelier Exits

**What it is:** An evolution of ATR trailing stops that anchors the stop to the highest high (for longs) rather than the current price. The Chandelier Exit = Highest High over N periods - ATR × multiplier.

**Why it matters for us:** Our current trailing stop (entry + 50% of gain) is simple but doesn't adapt to volatility. ATR Chandelier exits widen in volatile markets (preventing premature stops) and tighten in calm markets (locking in profits).

**Implementation:** Replace `_calculate_trailing_stop()` in `monitor.py` with: `new_sl = max(current_sl, highest_high_22 - atr_14 * 3.0)`. The 22-day high and 14-day ATR adapt to current conditions.

**Implementation difficulty:** Low. We already have ATR calculations.

### 4.3 Maximum Drawdown Control (CPPI-Based)

**What it is:** Instead of a binary circuit breaker (stop all trading at -3%), use a continuous scaling function. As drawdown increases, position sizes decrease proportionally. The floor ratchets up with portfolio gains (TIPP variant).

**The math:** exposure_multiplier = max(0, (portfolio_value - floor) / portfolio_value * m), where floor = max(initial_floor, peak_value * (1 - max_drawdown_pct)), and m is a risk multiplier (typically 3-5).

**Implementation difficulty:** Low-Medium. Modify the risk manager's `evaluate_order()` to scale `adjusted_notional` by the CPPI multiplier rather than applying a binary circuit breaker.

### 4.4 Correlation Breakdown Detection

**What it is:** Monitor whether the correlation structure of our portfolio is changing. During crises, correlations tend to spike toward 1.0 (everything falls together), destroying diversification exactly when it's needed most.

**Implementation:** Track rolling 20-day average pairwise correlation of held positions. When it exceeds a threshold (e.g., 0.7), reduce overall exposure or halt new positions. This is a form of regime-aware risk management.

**Implementation difficulty:** Low. We already compute covariance matrices in the portfolio optimizer.

---

## 5. Covariance & Return Estimation

### 5.1 Ledoit-Wolf Shrinkage Estimator

**What it is:** The sample covariance matrix from 60 days of 15+ stocks is noisy and ill-conditioned. Ledoit-Wolf shrinkage "shrinks" it toward a structured target (like the identity matrix or a single-factor model), reducing estimation error.

**Why it matters for us:** Our portfolio optimizer estimates a covariance matrix from 60 days of returns. With 15 stocks, that's 105 unique covariance terms estimated from ~60 observations — severely underdetermined. Shrinkage dramatically improves out-of-sample portfolio performance.

**The math:** Σ_shrunk = δ * F + (1 - δ) * S, where S is the sample covariance, F is the structured target, and δ is the optimal shrinkage intensity (computed analytically). For the nonlinear version, each eigenvalue of the sample covariance is individually shrunk toward a structured target.

**Implementation:** `sklearn.covariance.LedoitWolf()` is a one-line replacement. Use it in `portfolio_optimizer.py` when computing the covariance matrix.

**Implementation difficulty:** Very Low. Literally replace `returns.cov()` with `LedoitWolf().fit(returns).covariance_`.

**Key papers:**
- Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance Matrix" (2003, Journal of Portfolio Management)
- Ledoit & Wolf, "Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection" (2017, Review of Financial Studies) — the "Goldilocks" estimator

### 5.2 Exponentially Weighted Covariance

**What it is:** Weight recent observations more heavily when estimating covariance, so the matrix adapts to changing market conditions.

**The math:** Apply exponential weights: w_t = λ^{T-t} / Σ λ^{T-t}, where λ ≈ 0.94-0.97 (RiskMetrics uses 0.94). Then compute weighted covariance.

**Implementation:** `returns.ewm(halflife=30).cov()` in pandas. Can be combined with Ledoit-Wolf shrinkage.

**Implementation difficulty:** Very Low.

### 5.3 Denoised Covariance via Random Matrix Theory

**What it is:** Marchenko-Pastur theory from random matrix theory tells us which eigenvalues of a sample covariance matrix are noise vs. signal. By replacing noise eigenvalues with their average, we get a cleaner covariance estimate.

**The math:** For a matrix with T observations and N assets, the noise eigenvalues follow the Marchenko-Pastur distribution bounded by λ± = σ² * (1 ± sqrt(N/T))². Eigenvalues below λ+ are likely noise. Replace them with their average while preserving the trace.

**Implementation difficulty:** Medium. Requires eigendecomposition and knowledge of RMT. The `pypfopt` library implements this as the "denoised" risk model.

---

## 6. Execution & Market Microstructure

### 6.1 Slippage Modeling

**What it is:** Estimate the expected difference between the intended trade price and the actual fill price. Slippage depends on order size relative to average daily volume, bid-ask spread, and market impact.

**Why it matters for us:** We use market orders for bracket entries. For small retail orders, slippage is minimal, but as position sizes grow, it becomes significant. Incorporating a slippage model into backtest results would give more honest performance metrics.

**Simple model:** slippage_bps = base_spread_bps + impact_bps * sqrt(order_size / avg_daily_volume). A common approximation is 5-10 bps for liquid large-caps.

**Implementation difficulty:** Low. Add slippage as a transaction cost in backtest scoring. Already partially handled by `transaction_cost_bps` in config.

### 6.2 Time-of-Day Execution Optimization

**What it is:** Market microstructure research shows that spreads are widest at the open (first 30 minutes) and narrow throughout the day, with a small widening at close. Intraday volatility follows a U-shape.

**Why it matters for us:** If we submit market orders at market open (as our GitHub Actions cron suggests), we pay maximum spread. Delaying 30-60 minutes could save 5-15 bps per trade.

**Implementation difficulty:** Very Low. Change the cron schedule from 10:00 AM to 10:30 AM or 11:00 AM ET.

### 6.3 Limit Orders for Entry

**What it is:** Instead of market orders, place limit orders at or slightly below the current price for BUY entries. This can capture the bid-ask spread rather than paying it.

**Why it matters for us:** For illiquid mid-caps, the bid-ask spread can be 10-30 bps. Limit orders won't always fill, but when they do, they save the spread.

**Trade-off:** Missed fills mean missed signals. A hybrid approach: use limit orders for LOW confidence signals (where we're less sure) and market orders for HIGH confidence signals (where speed matters).

**Implementation difficulty:** Medium. Requires changes to `broker.py` to support limit order types and handling of unfilled orders.

---

## 7. Backtesting & Overfitting Prevention

### 7.1 Walk-Forward Optimization

**What it is:** Instead of optimizing parameters on the full historical dataset (which overfits), use a rolling or expanding window: optimize on in-sample data, test on out-of-sample data, advance the window, and repeat. Only report out-of-sample results.

**Why it matters for us:** Our backtest windows (3mo/6mo/12mo) are tested on the same data used for scoring. Walk-forward would split each window into train/test, giving more honest performance estimates.

**Implementation:** For each window, use 80% for optimization and 20% for validation. Only count the validation portion for scoring. The anchored variant (expanding in-sample window) provides stability; the rolling variant adapts faster.

**Implementation difficulty:** Medium. Requires restructuring `backtest.py` to split windows.

### 7.2 Combinatorially Symmetric Cross-Validation (CSCV)

**What it is:** A method by Bailey, Borwein, and López de Prado (2014) that estimates the probability a backtest is overfitted. It partitions the data into S subsets, tests all combinations, and measures how often the in-sample optimal strategy is also out-of-sample optimal.

**Why it matters for us:** When we test 11 strategies and pick the best, we're vulnerable to selection bias. CSCV gives us a probability estimate (e.g., "there is a 40% chance this backtest result is overfitted").

**Implementation difficulty:** Medium-High. Requires combinatorial testing across data partitions.

### 7.3 Deflated Sharpe Ratio

**What it is:** Adjusts the Sharpe ratio for: (a) number of strategies tested (multiple testing correction), (b) non-normality of returns (skewness and kurtosis), and (c) sample size.

**Why it matters for us:** We test 11 strategies per stock across 3 timeframes = 33 strategy-window combinations. The probability that the best one has Sharpe > 0 by chance alone is high. The Deflated Sharpe Ratio corrects for this.

**The math:** DSR = PSR(SR* | SR_0), where SR_0 = sqrt(V[SR_k]) * ((1-γ) * z^{-1}(1 - 1/N) + γ * z^{-1}(1 - 1/N * e^{-1})), N = number of trials, γ = Euler's constant. This gives the probability that the observed Sharpe exceeds what we'd expect from the best of N random trials.

**Implementation difficulty:** Low-Medium. The formula is closed-form. Add it as a post-processing step to composite scoring.

### 7.4 Probabilistic Sharpe Ratio (Lo, 2002)

**What it is:** Andrew Lo's seminal 2002 paper shows that the standard Sharpe ratio estimator has significant bias from serial correlation and small sample sizes. The Probabilistic Sharpe Ratio gives confidence intervals and minimum track record length.

**Why it matters for us:** Our 3-month backtest window has ~63 trading days. With that few observations, the standard error of the Sharpe ratio is large. A stock with a measured Sharpe of 1.5 over 63 days might not be statistically different from 0. The Probabilistic Sharpe Ratio quantifies this uncertainty.

**The math:** PSR(SR*) = Φ((SR - SR*) * sqrt(T-1) / sqrt(1 - γ₃*SR + (γ₄-1)/4 * SR²)), where γ₃ is skewness, γ₄ is kurtosis, T is sample size, and SR* is the benchmark Sharpe. The minimum track record length to claim SR > SR* at significance α is: MinTRL = 1 + (1 - γ₃*SR + (γ₄-1)/4 * SR²) * (z_α / (SR - SR*))².

**Implementation difficulty:** Low. It's a single formula applied to backtest output.

---

## 8. Implementation Roadmap

Prioritized by impact-to-effort ratio. Grouped into three phases.

### Phase 1: Quick Wins (Low effort, high impact)

| Improvement | Component | Effort | Impact | Description |
|---|---|---|---|---|
| **Ledoit-Wolf shrinkage** | portfolio_optimizer.py | Very Low | High | Replace `returns.cov()` with `LedoitWolf().fit(returns).covariance_` — dramatically better covariance estimates |
| **Probabilistic Sharpe Ratio** | backtest.py | Low | High | Add confidence intervals to Sharpe estimates — stops us from trusting short-window flukes |
| **Volatility-targeted sizing** | executor.py / signals.py | Low | High | Blend Half-Kelly with vol-targeting: size inversely proportional to realized vol |
| **Deflated Sharpe Ratio** | backtest.py | Low | Medium | Correct for multiple testing (11 strategies × 3 windows = 33 trials) |
| **Execution timing** | cli.py / GitHub Actions | Very Low | Low-Medium | Delay market orders by 30-60 min after open to avoid wide spreads |
| **ATR Chandelier exits** | monitor.py | Low | Medium | Replace simple trailing stop with volatility-adaptive stops |
| **Exponentially weighted covariance** | portfolio_optimizer.py | Very Low | Medium | Weight recent returns more heavily in covariance estimation |

### Phase 2: Meaningful Upgrades (Medium effort, high impact)

| Improvement | Component | Effort | Impact | Description |
|---|---|---|---|---|
| **HRP allocation** | portfolio_optimizer.py | Medium | High | Add Hierarchical Risk Parity as an alternative to marginal-Sharpe ranking |
| **HMM regime detection** | new module | Medium | High | 2-3 state HMM on SPY to detect bull/bear/sideways; scale positions by regime |
| **GARCH volatility forecasting** | signals.py / risk.py | Medium | High | Replace backward-looking vol_20 with forward-looking GARCH forecasts |
| **CPPI drawdown control** | risk.py | Medium | High | Replace binary circuit breaker with continuous position scaling |
| **Walk-forward backtesting** | backtest.py | Medium | High | Split windows into train/test for honest out-of-sample metrics |
| **Fractional differentiation** | new preprocessing | Low-Medium | Medium | Make features stationary while preserving memory — foundation for future ML |
| **Slippage model** | backtest.py | Low | Medium | Add realistic slippage to backtest costs |
| **Correlation breakdown monitoring** | monitor.py / risk.py | Low | Medium | Track portfolio correlation and reduce exposure when correlations spike |

### Phase 3: Advanced Capabilities (Higher effort, potentially transformative)

| Improvement | Component | Effort | Impact | Description |
|---|---|---|---|---|
| **Black-Litterman integration** | portfolio_optimizer.py | Medium-High | High | Use signals as views, confidence as uncertainty, market equilibrium as prior |
| **Meta-labeling** | new module | High | High | Train ML model to predict when strategy signals are reliable |
| **Triple barrier labeling** | backtest.py | Medium-High | High | Label trades by which barrier was hit first (matches our bracket orders) |
| **CVaR portfolio constraints** | risk.py / portfolio_optimizer.py | Medium | Medium-High | Constrain portfolio construction to limit tail risk |
| **CSCV overfitting detection** | backtest.py | Medium-High | Medium | Estimate probability that backtest results are overfitted |
| **Factor-aware signals** | signals.py | Medium-High | Medium | Add momentum, value, quality factor exposures to signal generation |
| **Limit order entries** | broker.py | Medium | Medium | Use limit orders for lower-confidence signals to save spread |
| **Random Matrix Theory denoising** | portfolio_optimizer.py | Medium | Medium | Denoise covariance matrix using Marchenko-Pastur distribution |

---

## Key References

### Books
- López de Prado, M. — *Advances in Financial Machine Learning* (2018) — Triple barrier, meta-labeling, fractional differentiation, HRP, deflated Sharpe ratio
- Ledoit, O. & Wolf, M. — *Honey, I Shrunk the Sample Covariance Matrix* (2003)
- Lo, A. — *The Statistics of Sharpe Ratios* (Financial Analysts Journal, 2002)

### Papers (Portfolio Optimization)
- López de Prado, M. — "Building Diversified Portfolios that Outperform Out of Sample" (2016, JPM)
- Black, F. & Litterman, R. — "Global Portfolio Optimization" (1992, Financial Analysts Journal)
- Ledoit, O. & Wolf, M. — "Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection" (2017, RFS)

### Papers (Risk & Regime)
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. — "The Probability of Backtest Overfitting" (2014, Journal of Computational Finance)
- Ang, A. & Bekaert, G. — "International Asset Allocation with Regime Shifts" (2002, Review of Financial Studies)
- Bollerslev, T. — "Generalized Autoregressive Conditional Heteroskedasticity" (1986, Journal of Econometrics)

### Papers (Momentum & Factors)
- Moskowitz, T., Ooi, Y.H. & Pedersen, L.H. — "Time Series Momentum" (2012, Journal of Financial Economics)
- Antonacci, G. — "Dual Momentum Investing" (2014)
- Ehlers, J. — "Fractal Adaptive Moving Average" (Technical Analysis of Stocks & Commodities)

---

## Recommended Python Libraries

| Library | Use Case | Install |
|---|---|---|
| `sklearn.covariance` | Ledoit-Wolf shrinkage | (included in scikit-learn) |
| `hmmlearn` | Hidden Markov Models for regime detection | `pip install hmmlearn` |
| `arch` | GARCH volatility forecasting | `pip install arch` |
| `fracdiff` | Fractional differentiation | `pip install fracdiff` |
| `pypfopt` | Black-Litterman, HRP, risk models | `pip install pyportfolioopt` |
| `scipy.cluster.hierarchy` | Hierarchical clustering for HRP | (included in scipy) |
| `scipy.optimize` | Risk parity optimization | (included in scipy) |

---

*Last updated: March 13, 2026*
