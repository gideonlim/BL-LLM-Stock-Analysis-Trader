# Quantitative Finance Research — Potential Improvements for Our Trading Bot

> **Purpose:** A curated survey of academic research, advanced math, and quantitative finance techniques that could meaningfully improve each layer of our trading system. Organized by component, with implementation priority and difficulty ratings.

### Implementation Progress

> **13 Fully Implemented** | **4 Partially Implemented** | **17 Not Implemented** (as of March 2026)

| Status | Meaning |
|--------|---------|
| ✅ | Fully implemented |
| 🔶 | Partially implemented |
| ⬜ | Not yet implemented |

---

## Table of Contents

1. [Portfolio Optimization](#1-portfolio-optimization)
2. [Position Sizing](#2-position-sizing)
3. [Signal Generation & Feature Engineering](#3-signal-generation--feature-engineering)
4. [Risk Management](#4-risk-management)
5. [Covariance & Return Estimation](#5-covariance--return-estimation)
6. [Execution & Market Microstructure](#6-execution--market-microstructure)
7. [Backtesting & Overfitting Prevention](#7-backtesting--overfitting-prevention)
8. [Market Sentiment & Alternative Data](#8-market-sentiment--alternative-data)
9. [Implementation Roadmap](#9-implementation-roadmap)

---

## 1. Portfolio Optimization

Our bot currently uses marginal Sharpe ranking (Markowitz-based). There are three major upgrades worth considering.

### 1.1 Hierarchical Risk Parity (HRP) ⬜

**What it is:** An alternative to Markowitz optimization developed by Marcos López de Prado (2016). Instead of inverting a covariance matrix (which is numerically unstable with many assets), HRP uses hierarchical clustering to group correlated assets, then allocates risk top-down through the tree.

**Why it matters for us:** Our covariance matrix is estimated from just 60 days of returns for 15+ stocks. With that few observations relative to assets, the sample covariance matrix is poorly conditioned. HRP doesn't require matrix inversion, so it works even with singular or near-singular covariance matrices. Empirical tests show HRP produces lower out-of-sample variance than Markowitz minimum-variance, despite Markowitz explicitly optimizing for it.

**The math:** HRP works in three steps. First, compute a distance matrix from correlations: d(i,j) = sqrt(0.5 * (1 - ρ_ij)). Second, apply single-linkage hierarchical clustering to group assets. Third, allocate inversely proportional to cluster variance using recursive bisection — split the portfolio into two clusters, weight each inversely by its variance, then recurse down each subtree.

**Implementation difficulty:** Medium. The `scipy.cluster.hierarchy` module handles the clustering. We'd replace or complement `rank_intents_by_marginal_sharpe()` in `portfolio_optimizer.py`.

**Key papers:**
- López de Prado, "Building Diversified Portfolios that Outperform Out of Sample" (2016, Journal of Portfolio Management)
- Recent 2025 extensions exploring distance metric variants and reinforcement learning integration (RL-BHRP)

### 1.2 Black-Litterman Model ✅

> **Implemented in:** `trading_bot_bl/black_litterman.py`, `trading_bot_bl/portfolio_optimizer.py`. Full BL model with posterior return computation, confidence-mapped view uncertainty (Ω), regime-sensitive covariance blending, and LLM-enhanced view generation. Primary optimization method.

**What it is:** A Bayesian framework (Goldman Sachs, 1990) that starts from market equilibrium returns (implied by market-cap weights via reverse optimization) and blends in the investor's own views with specified confidence levels. The output is a set of expected returns that can be fed into a mean-variance optimizer without the extreme, unstable weights that raw Markowitz produces.

**Why it matters for us:** Our signals already generate views — each BUY signal is implicitly a view that the stock will outperform. Black-Litterman gives us a principled way to combine those views with market priors, weighted by our confidence scores. A HIGH confidence signal would shift expected returns more than a LOW confidence one. This directly addresses the problem that raw Markowitz is hypersensitive to return estimates.

**The math:** The posterior expected returns are: E[R] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}π + P'Ω^{-1}Q], where π is the equilibrium return vector, P is the view matrix, Q is the view returns, Ω is the uncertainty of views, Σ is the covariance matrix, and τ is a scalar (typically 0.025-0.05).

**Implementation difficulty:** Medium-High. Need to compute equilibrium returns (market cap weights + covariance) and map our confidence scores to view uncertainty (Ω). The `pypfopt` library has a Black-Litterman implementation.

**Key insight for our bot:** Our confidence scoring (0-6) already encodes view uncertainty. A signal with confidence 5 should produce a tighter Ω (more certain view) than confidence 2.

### 1.3 Risk Parity ⬜

**What it is:** Instead of optimizing returns, allocate so each asset contributes equally to total portfolio risk. This prevents concentrated risk in a few volatile positions.

**Why it matters for us:** Currently, position sizes come from Half-Kelly (return-based), which can over-concentrate in volatile stocks. Risk parity would ensure no single position dominates portfolio variance.

**The math:** For each asset i, the risk contribution is: RC_i = w_i * (Σw)_i / sqrt(w'Σw). Risk parity sets RC_i = RC_j for all i, j.

**Implementation difficulty:** Medium. Requires iterative optimization (scipy.optimize) but well-understood.

---

## 2. Position Sizing

Our bot uses Half-Kelly. There are more sophisticated approaches.

### 2.1 Volatility Targeting 🔶

> **Partially implemented:** Market sentiment module applies VIX-based position sizing multiplier (fear → larger, greed → smaller). However, true per-stock volatility-targeted sizing (inverse of realized vol) is not implemented — sizing still uses Half-Kelly capped by confidence.

**What it is:** Instead of sizing by expected return (Kelly), size each position to contribute a target amount of daily volatility. This automatically reduces exposure in high-vol markets and increases it in low-vol markets.

**Why it matters for us:** Half-Kelly uses backtest win rate and profit factor, which are backward-looking and don't adapt to current market conditions. Volatility targeting is forward-looking — if a stock's realized vol doubles, position size automatically halves.

**The math:** For a target portfolio volatility σ_target: w_i = σ_target / (N * σ_i), where σ_i is the asset's realized volatility and N is the number of positions. More sophisticated versions account for correlation: w = σ_target * (Σ^{-1} * 1) / (1' * Σ^{-1} * 1).

**Implementation:** Calculate 20-day realized volatility for each stock (we already compute `vol_20` in signals), then: `position_size_pct = target_vol / (num_positions * stock_vol_20)`. This replaces or blends with Half-Kelly.

**Key research:** Man Group's research shows volatility targeting consistently improves Sharpe ratios for equity portfolios and reduces the probability of extreme returns.

**Implementation difficulty:** Low. We already have `vol_20` in the signal data. This could be a blended approach: `final_size = alpha * kelly_size + (1 - alpha) * vol_target_size`.

### 2.2 Constant Proportion Portfolio Insurance (CPPI) ⬜

**What it is:** A dynamic position sizing method that protects a portfolio floor value. Position size is: Exposure = multiplier * (portfolio_value - floor). As the portfolio drops toward the floor, exposure shrinks to zero. As it rises, exposure increases.

**Why it matters for us:** This directly addresses drawdown control. We set a maximum acceptable drawdown (e.g., floor = 90% of equity), and CPPI automatically scales down positions as we approach it — much more graceful than a binary circuit breaker.

**The math:** At each rebalance: cushion = portfolio_value - floor; exposure = m * cushion (where m = 3-5 typically). The floor can ratchet up with a TIPP (Time Invariant Portfolio Protection) variant: floor = max(floor, drawdown_pct * peak_value).

**Implementation difficulty:** Low-Medium. Replace the binary circuit breaker with a continuous scaling function. Position sizes shrink smoothly rather than going to zero when a threshold is hit.

### 2.3 Optimal f (Ralph Vince) ⬜

**What it is:** An empirical method that tests various bet fractions against the actual distribution of historical returns to find the fraction that maximizes terminal wealth. Unlike Kelly (which assumes a simple win/loss binary), Optimal f uses the full return distribution.

**Why it matters for us:** Our strategies produce continuous returns, not binary outcomes. Optimal f is theoretically more appropriate than Kelly for this case.

**Caveat:** Optimal f tends to be aggressive — it maximizes growth but can produce large drawdowns. In practice, use a fraction of Optimal f (similar to how we use Half-Kelly).

---

## 3. Signal Generation & Feature Engineering

### 3.1 Market Regime Detection via Hidden Markov Models (HMMs) ⬜

> **Note:** Market regime detection is implemented via SPY 200-day SMA trend filter (`market_sentiment.py`) with BULL/CAUTION/BEAR/SEVERE_BEAR tiers, but uses a deterministic trend-following approach rather than HMMs.

**What it is:** HMMs model the market as switching between hidden states (e.g., bull/bear/sideways), each with different return and volatility characteristics. The model infers which state the market is currently in.

**Why it matters for us:** Our strategies use the same parameters in all market conditions. A mean reversion strategy that works in calm markets can be disastrous during a crash. With HMM regime detection, we can: (a) disable certain strategies in unfavorable regimes, (b) adjust position sizes based on regime, (c) use different stop-loss multipliers per regime.

**The math:** An HMM with K states has: transition matrix A (K×K) defining state switching probabilities, emission distributions (typically Gaussian for returns: μ_k, σ_k per state), and initial state probabilities. The Baum-Welch algorithm (EM) estimates parameters; the forward-backward algorithm computes state probabilities.

**Implementation:** Use `hmmlearn` library. Fit a 2-3 state Gaussian HMM on SPY returns. Use the current state probability to scale position sizes: full size in bull regime, half size in neutral, no new positions in bear.

**Key papers:**
- Regime-Switching Factor Investing with Hidden Markov Models (MDPI, 2020) — achieved Sharpe ratios of 1.9 with 3-state HMM
- Multi-model ensemble HMM voting framework for regime shift detection (2025)

**Implementation difficulty:** Medium. The model itself is straightforward with `hmmlearn`. The challenge is choosing the right number of states and features (returns, vol, breadth).

### 3.2 Fractional Differentiation (López de Prado) ⬜

**What it is:** A method to make time series stationary while preserving as much memory (trend information) as possible. Standard integer differentiation (returns = price[t] - price[t-1]) makes the series stationary but destroys all memory. Fractional differentiation with d ≈ 0.2-0.4 achieves stationarity while retaining 90%+ correlation with the original series.

**Why it matters for us:** Our strategies compute indicators on raw prices or simple returns. If we used ML features, fractionally differentiated prices would give us stationary inputs (required for ML) that still contain predictive trend information that returns throw away.

**The math:** The fractional difference operator: (1-B)^d = Σ_{k=0}^{∞} C(d,k) * (-B)^k, where B is the backshift operator and C(d,k) = d!/(k!(d-k)!). For d=0 we get the original series; for d=1, standard returns. The key insight is finding the minimum d (call it d*) where the ADF test indicates stationarity.

**Implementation difficulty:** Low. The `fracdiff` Python library implements this efficiently. Use it as a feature preprocessing step before any ML model.

### 3.3 Triple Barrier Method + Meta-Labeling (López de Prado) ⬜

**What it is:** Instead of labeling returns as simply up/down after a fixed period, the triple barrier method uses three dynamic barriers: upper barrier (take profit hit first → label +1), lower barrier (stop loss hit first → label -1), and vertical barrier (time expires → label based on return sign). Meta-labeling then trains a secondary ML model to decide whether to act on the primary model's signal.

**Why it matters for us:** Our strategies already produce signals, and we already use bracket orders (SL/TP). The triple barrier method would label historical trades more realistically (matching how our bracket orders actually work). Meta-labeling would add a learned "bet sizing" model on top — rather than uniform Half-Kelly, an ML model learns when each strategy's signals are more or less reliable.

**The connection to our system:** Our bracket orders (entry → SL/TP/time expiry) are literally the triple barrier in production. We can generate labels from historical bracket-order outcomes and train a meta-model to predict signal reliability.

**Implementation difficulty:** Medium-High. Requires generating proper labels from historical data and training a secondary classifier (Random Forest or Gradient Boosting).

### 3.4 Adaptive Moving Averages (KAMA, FRAMA) ⬜

**What it is:** Moving averages that automatically adjust their smoothing period based on market conditions. Kaufman's Adaptive Moving Average (KAMA) uses an efficiency ratio; Ehlers' Fractal Adaptive Moving Average (FRAMA) uses fractal dimension.

**Why it matters for us:** Our SMA/EMA crossover strategies use fixed periods (10/50, 9/21). Adaptive MAs would reduce whipsaws in choppy markets while staying responsive in trends.

**Implementation difficulty:** Low. These are drop-in replacements for existing moving average calculations in `strategies.py`.

### 3.5 Factor-Aware Signal Enhancement ⬜

**What it is:** Rather than treating each stock independently, incorporate cross-sectional factor exposures (momentum, value, quality, size) to enrich signals.

**Key research findings:**
- Time-series momentum (past 12-month return predicting future return) achieves Sharpe ratios above 1.20 across asset classes
- Dual momentum (combining time-series and cross-sectional) generates 1.88% per month in academic studies
- Quality factor (profitability + growth + payout + safety) combined with momentum shows persistent alpha

**Why it matters for us:** Our signals are purely technical. Adding factor awareness (is this stock in the top momentum quintile? is it a quality company?) would provide orthogonal information.

**Implementation difficulty:** Medium. Requires fundamental data (earnings, book value) beyond our current Yahoo Finance price data.

### 3.6 GARCH Volatility Forecasting ⬜

**What it is:** GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models volatility as time-varying, capturing the empirical fact that volatility clusters — high-vol days tend to follow high-vol days.

**Why it matters for us:** We use 20-day realized volatility (`vol_20`), which is backward-looking. GARCH forecasts tomorrow's volatility, which is more useful for position sizing and stop-loss placement. The GJR-GARCH variant also captures the "leverage effect" (volatility rises more after losses than gains).

**The math:** GARCH(1,1): σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}. GJR-GARCH adds: + γ * ε²_{t-1} * I(ε_{t-1} < 0), where I is an indicator for negative returns.

**Implementation:** The `arch` Python library implements GARCH and its variants. Use forecasted volatility for: ATR-based stop losses, position sizing, and regime detection.

**Implementation difficulty:** Low-Medium. The library handles estimation. The work is integrating forecasts into our sizing/stop-loss pipeline.

---

## 4. Risk Management

### 4.1 Conditional Value-at-Risk (CVaR / Expected Shortfall) ⬜

**What it is:** While VaR asks "what's the worst loss at the 95th percentile?", CVaR asks "given that we're in the worst 5%, what's the expected loss?" CVaR captures tail risk that VaR misses.

**Why it matters for us:** Our risk manager uses a simple daily loss circuit breaker (3%). CVaR-based risk management would give us a more nuanced view of tail risk. We could constrain portfolio construction to keep CVaR below a threshold, or use CVaR to size positions.

**The math:** CVaR_α = E[L | L > VaR_α] = (1/(1-α)) * ∫_{α}^{1} VaR_u du. For a portfolio, CVaR is a coherent risk measure (unlike VaR) — it's convex, which makes optimization tractable.

**Implementation:** Use historical simulation or parametric estimation. The `scipy.stats` module can compute CVaR from return distributions. Integrate as an additional risk check: reject orders that push portfolio CVaR above a threshold.

**Implementation difficulty:** Medium. The math is straightforward, but estimating CVaR reliably requires sufficient return history.

### 4.2 Dynamic Stop-Loss via ATR Chandelier Exits 🔶

> **Partially implemented:** ATR-based trailing stops exist in `monitor.py` (2× ATR below current price with breakeven floor). However, this is not the full Chandelier Exit — stops are anchored to current price rather than the highest high over N periods.

**What it is:** An evolution of ATR trailing stops that anchors the stop to the highest high (for longs) rather than the current price. The Chandelier Exit = Highest High over N periods - ATR × multiplier.

**Why it matters for us:** Our current trailing stop (entry + 50% of gain) is simple but doesn't adapt to volatility. ATR Chandelier exits widen in volatile markets (preventing premature stops) and tighten in calm markets (locking in profits).

**Implementation:** Replace `_calculate_trailing_stop()` in `monitor.py` with: `new_sl = max(current_sl, highest_high_22 - atr_14 * 3.0)`. The 22-day high and 14-day ATR adapt to current conditions.

**Implementation difficulty:** Low. We already have ATR calculations.

### 4.3 Maximum Drawdown Control (CPPI-Based) ⬜

**What it is:** Instead of a binary circuit breaker (stop all trading at -3%), use a continuous scaling function. As drawdown increases, position sizes decrease proportionally. The floor ratchets up with portfolio gains (TIPP variant).

**The math:** exposure_multiplier = max(0, (portfolio_value - floor) / portfolio_value * m), where floor = max(initial_floor, peak_value * (1 - max_drawdown_pct)), and m is a risk multiplier (typically 3-5).

**Implementation difficulty:** Low-Medium. Modify the risk manager's `evaluate_order()` to scale `adjusted_notional` by the CPPI multiplier rather than applying a binary circuit breaker.

### 4.4 Correlation Breakdown Detection ⬜

**What it is:** Monitor whether the correlation structure of our portfolio is changing. During crises, correlations tend to spike toward 1.0 (everything falls together), destroying diversification exactly when it's needed most.

**Implementation:** Track rolling 20-day average pairwise correlation of held positions. When it exceeds a threshold (e.g., 0.7), reduce overall exposure or halt new positions. This is a form of regime-aware risk management.

**Implementation difficulty:** Low. We already compute covariance matrices in the portfolio optimizer.

---

## 5. Covariance & Return Estimation

### 5.1 Ledoit-Wolf Shrinkage Estimator ✅

> **Implemented in:** `trading_bot_bl/black_litterman.py`. Custom implementation following Ledoit-Wolf 2004 methodology. Used as long-term covariance estimator, blended with EWMA in regime-sensitive mode.

**What it is:** The sample covariance matrix from 60 days of 15+ stocks is noisy and ill-conditioned. Ledoit-Wolf shrinkage "shrinks" it toward a structured target (like the identity matrix or a single-factor model), reducing estimation error.

**Why it matters for us:** Our portfolio optimizer estimates a covariance matrix from 60 days of returns. With 15 stocks, that's 105 unique covariance terms estimated from ~60 observations — severely underdetermined. Shrinkage dramatically improves out-of-sample portfolio performance.

**The math:** Σ_shrunk = δ * F + (1 - δ) * S, where S is the sample covariance, F is the structured target, and δ is the optimal shrinkage intensity (computed analytically). For the nonlinear version, each eigenvalue of the sample covariance is individually shrunk toward a structured target.

**Implementation:** `sklearn.covariance.LedoitWolf()` is a one-line replacement. Use it in `portfolio_optimizer.py` when computing the covariance matrix.

**Implementation difficulty:** Very Low. Literally replace `returns.cov()` with `LedoitWolf().fit(returns).covariance_`.

**Key papers:**
- Ledoit & Wolf, "Honey, I Shrunk the Sample Covariance Matrix" (2003, Journal of Portfolio Management)
- Ledoit & Wolf, "Nonlinear Shrinkage of the Covariance Matrix for Portfolio Selection" (2017, Review of Financial Studies) — the "Goldilocks" estimator

### 5.2 Exponentially Weighted Covariance ✅

> **Implemented in:** `trading_bot_bl/black_litterman.py`. EWMA covariance with configurable halflife (default 21 days). Used as short-term component in regime-sensitive blending alongside Ledoit-Wolf.

**What it is:** Weight recent observations more heavily when estimating covariance, so the matrix adapts to changing market conditions.

**The math:** Apply exponential weights: w_t = λ^{T-t} / Σ λ^{T-t}, where λ ≈ 0.94-0.97 (RiskMetrics uses 0.94). Then compute weighted covariance.

**Implementation:** `returns.ewm(halflife=30).cov()` in pandas. Can be combined with Ledoit-Wolf shrinkage.

**Implementation difficulty:** Very Low.

### 5.3 Denoised Covariance via Random Matrix Theory ⬜

**What it is:** Marchenko-Pastur theory from random matrix theory tells us which eigenvalues of a sample covariance matrix are noise vs. signal. By replacing noise eigenvalues with their average, we get a cleaner covariance estimate.

**The math:** For a matrix with T observations and N assets, the noise eigenvalues follow the Marchenko-Pastur distribution bounded by λ± = σ² * (1 ± sqrt(N/T))². Eigenvalues below λ+ are likely noise. Replace them with their average while preserving the trace.

**Implementation difficulty:** Medium. Requires eigendecomposition and knowledge of RMT. The `pypfopt` library implements this as the "denoised" risk model.

---

## 6. Execution & Market Microstructure

### 6.1 Slippage Modeling ✅

> **Implemented in:** `trading_bot_bl/broker.py`, `trading_bot_bl/journal.py`. Limit orders with configurable max entry slippage (default 1.0%). Journal tracks entry/exit slippage per trade. Analytics include slippage reporting.

**What it is:** Estimate the expected difference between the intended trade price and the actual fill price. Slippage depends on order size relative to average daily volume, bid-ask spread, and market impact.

**Why it matters for us:** We use market orders for bracket entries. For small retail orders, slippage is minimal, but as position sizes grow, it becomes significant. Incorporating a slippage model into backtest results would give more honest performance metrics.

**Simple model:** slippage_bps = base_spread_bps + impact_bps * sqrt(order_size / avg_daily_volume). A common approximation is 5-10 bps for liquid large-caps.

**Implementation difficulty:** Low. Add slippage as a transaction cost in backtest scoring. Already partially handled by `transaction_cost_bps` in config.

### 6.2 Time-of-Day Execution Optimization ✅

> **Implemented:** Execution scheduled at 10:15 AM ET via GitHub Actions (45 min after market open). Avoids peak opening volatility and wide spreads.

**What it is:** Market microstructure research shows that spreads are widest at the open (first 30 minutes) and narrow throughout the day, with a small widening at close. Intraday volatility follows a U-shape.

**Why it matters for us:** If we submit market orders at market open (as our GitHub Actions cron suggests), we pay maximum spread. Delaying 30-60 minutes could save 5-15 bps per trade.

**Implementation difficulty:** Very Low. Change the cron schedule from 10:00 AM to 10:30 AM or 11:00 AM ET.

### 6.3 Limit Orders for Entry ✅

> **Implemented in:** `trading_bot_bl/broker.py`. Entry uses limit orders at `current_price × (1 + max_entry_slippage_pct)` when slippage > 0. Won't fill if stock gaps up beyond the limit. Configurable via `MAX_ENTRY_SLIPPAGE_PCT`.

**What it is:** Instead of market orders, place limit orders at or slightly below the current price for BUY entries. This can capture the bid-ask spread rather than paying it.

**Why it matters for us:** For illiquid mid-caps, the bid-ask spread can be 10-30 bps. Limit orders won't always fill, but when they do, they save the spread.

**Trade-off:** Missed fills mean missed signals. A hybrid approach: use limit orders for LOW confidence signals (where we're less sure) and market orders for HIGH confidence signals (where speed matters).

**Implementation difficulty:** Medium. Requires changes to `broker.py` to support limit order types and handling of unfilled orders.

---

## 7. Backtesting & Overfitting Prevention

### 7.1 Walk-Forward Optimization ✅

> **Implemented in:** `quant_analysis_bot/backtest.py`. 70/30 train-test split (configurable via `walk_forward_validation_pct`). Metrics scored only on out-of-sample validation portion. Next-bar execution prevents look-ahead bias.

**What it is:** Instead of optimizing parameters on the full historical dataset (which overfits), use a rolling or expanding window: optimize on in-sample data, test on out-of-sample data, advance the window, and repeat. Only report out-of-sample results.

**Why it matters for us:** Our backtest windows (3mo/6mo/12mo) are tested on the same data used for scoring. Walk-forward would split each window into train/test, giving more honest performance estimates.

**Implementation:** For each window, use 80% for optimization and 20% for validation. Only count the validation portion for scoring. The anchored variant (expanding in-sample window) provides stability; the rolling variant adapts faster.

**Implementation difficulty:** Medium. Requires restructuring `backtest.py` to split windows.

### 7.2 Combinatorially Symmetric Cross-Validation (CSCV) ✅

> **Implemented in:** `quant_analysis_bot/cscv.py`. Full PBO (Probability of Backtest Overfitting) implementation per Bailey et al. (2017). Partitions return matrix into S equal sub-periods, evaluates all C(S, S/2) training/testing combinations, computes PBO, logit distribution, Spearman rank correlation, and per-strategy summary. Auto-selects partition count based on data length. Integrated via `--validate` CLI flag. PBO propagated to BacktestResult and DailySignal. 24 unit tests in `test_cscv.py`.

**What it is:** A method by Bailey, Borwein, and López de Prado (2014) that estimates the probability a backtest is overfitted. It partitions the data into S subsets, tests all combinations, and measures how often the in-sample optimal strategy is also out-of-sample optimal.

**Why it matters for us:** When we test 11 strategies and pick the best, we're vulnerable to selection bias. CSCV gives us a probability estimate (e.g., "there is a 40% chance this backtest result is overfitted").

**Implementation difficulty:** Medium-High. Requires combinatorial testing across data partitions.

### 7.3 Deflated Sharpe Ratio ✅

> **Implemented in:** `quant_analysis_bot/backtest.py`. Full Bailey & de Prado (2014) DSR implementation. Uses strategy return stream (not raw stock returns) for accurate skew/kurtosis. Applied as scoring multiplier with confidence weighting.

**What it is:** Adjusts the Sharpe ratio for: (a) number of strategies tested (multiple testing correction), (b) non-normality of returns (skewness and kurtosis), and (c) sample size.

**Why it matters for us:** We test 11 strategies per stock across 3 timeframes = 33 strategy-window combinations. The probability that the best one has Sharpe > 0 by chance alone is high. The Deflated Sharpe Ratio corrects for this.

**The math:** DSR = PSR(SR* | SR_0), where SR_0 = sqrt(V[SR_k]) * ((1-γ) * z^{-1}(1 - 1/N) + γ * z^{-1}(1 - 1/N * e^{-1})), N = number of trials, γ = Euler's constant. This gives the probability that the observed Sharpe exceeds what we'd expect from the best of N random trials.

**Implementation difficulty:** Low-Medium. The formula is closed-form. Add it as a post-processing step to composite scoring.

### 7.4 Probabilistic Sharpe Ratio (Lo, 2002) ✅

> **Implemented in:** `trading_bot_bl/journal_analytics.py`. PSR calculation with 0% null hypothesis for live trading performance assessment. Also includes MinTRL (minimum track record length) computation.

**What it is:** Andrew Lo's seminal 2002 paper shows that the standard Sharpe ratio estimator has significant bias from serial correlation and small sample sizes. The Probabilistic Sharpe Ratio gives confidence intervals and minimum track record length.

**Why it matters for us:** Our 3-month backtest window has ~63 trading days. With that few observations, the standard error of the Sharpe ratio is large. A stock with a measured Sharpe of 1.5 over 63 days might not be statistically different from 0. The Probabilistic Sharpe Ratio quantifies this uncertainty.

**The math:** PSR(SR*) = Φ((SR - SR*) * sqrt(T-1) / sqrt(1 - γ₃*SR + (γ₄-1)/4 * SR²)), where γ₃ is skewness, γ₄ is kurtosis, T is sample size, and SR* is the benchmark Sharpe. The minimum track record length to claim SR > SR* at significance α is: MinTRL = 1 + (1 - γ₃*SR + (γ₄-1)/4 * SR²) * (z_α / (SR - SR*))².

**Implementation difficulty:** Low. It's a single formula applied to backtest output.

---

## 8. Market Sentiment & Alternative Data

Our bot currently uses only price/volume-derived technical indicators. Incorporating market sentiment — from news headlines, social media, options flow, and institutional filings — provides orthogonal information that technical analysis alone cannot capture. Research consistently shows that sentiment signals are most powerful when combined with (not replacing) quantitative signals.

### 8.1 News Headline Sentiment via FinBERT ⬜

**What it is:** FinBERT is a BERT-based language model fine-tuned on financial text (ProsusAI/finBERT). It classifies financial headlines into positive, negative, or neutral with a confidence score. Unlike generic sentiment tools (VADER, TextBlob), FinBERT understands financial language — it correctly interprets "the company beat expectations" as positive and "the stock was downgraded" as negative.

**Why it matters for us:** Our signals are purely technical. A stock might show a perfect SMA crossover while the company just announced an SEC investigation. FinBERT headline sentiment adds a qualitative filter that catches what price action hasn't priced in yet. Research from the S&P 500 (2018–2023) shows FinBERT-enhanced models consistently outperform technical-only baselines in both AUC, F1-score, and simulated trading profitability.

**How to integrate:** Fetch recent headlines for each ticker (via Alpaca News API — we already have Alpaca credentials, or Finnhub's free tier). Run FinBERT on each headline to get a sentiment score (-1 to +1). Aggregate into a daily sentiment score per ticker: `daily_sentiment = mean(headline_scores, weighted_by_recency)`. Use as a signal filter or confidence modifier — a BUY signal with negative sentiment gets its confidence demoted; a BUY with strong positive sentiment gets a boost.

**The math:** FinBERT outputs softmax probabilities for [negative, neutral, positive]. The composite score: `sent = P(positive) - P(negative)`, ranging from -1.0 to +1.0. For aggregation across N headlines: `daily_sent = Σ(sent_i × w_i) / Σ(w_i)`, where w_i = exp(-λ × age_hours_i) (exponential recency decay, λ ≈ 0.1).

**Implementation:** The `transformers` library from Hugging Face provides FinBERT inference in ~5 lines of Python. The model is ~400MB and runs on CPU in ~50ms per headline. For our pipeline (scoring ~20 tickers × ~5 headlines each), total overhead is ~5 seconds.

**Implementation difficulty:** Low-Medium. The NLP inference is trivial. The work is building a reliable headline fetcher and deciding how sentiment modifies the signal pipeline.

**Key papers:**
- Araci (2019), "FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models"
- FinBERT-LSTM integration for stock movement prediction (ACM 2024) — combined sentiment with LSTM price model for market/industry/stock-level news
- Tak & Pele (2025), S&P 500 empirical evaluation showing FinBERT outperforms lexicon-based approaches

### 8.2 LLM-Based Sentiment Scoring (GPT/Claude) ✅

> **Implemented in:** `trading_bot_bl/llm_views.py`, `trading_bot_bl/news_fetcher.py`. LLM-enhanced view generation using Claude or OpenAI API with repeated sampling (N=10 at temp 0.7) for uncertainty estimation. Headlines fetched and used as context. Integrated into Black-Litterman model as view confidence (Ω). Feature-flagged via `LLM_VIEWS_ENABLED`.

**What it is:** Instead of a specialized model like FinBERT, use a general-purpose LLM (GPT-4, Claude, Llama 3) with domain-specific prompting to score financial news. Advanced techniques include Chain-of-Thought (CoT) prompting, Domain Knowledge Chain-of-Thought (DK-CoT), and few-shot examples to improve accuracy.

**Why it matters for us:** We already have LLM infrastructure in `llm_views.py` for Black-Litterman view generation. Extending this to headline sentiment scoring requires minimal new code. LLMs can also provide reasoning for their sentiment scores, which FinBERT cannot. Research shows generative LLMs (Llama 3.1, GPT-4) outperform discriminative models (BERT, FinBERT) on financial sentiment tasks when prompted correctly.

**How to integrate:** Our existing `news_fetcher.py` already fetches headlines. Add a `score_headlines_llm()` function that batches headlines for a ticker and prompts the LLM: "Rate each headline's impact on {ticker}'s stock price from -1.0 (very negative) to +1.0 (very positive). Consider financial context." The LLM returns structured JSON scores. Aggregate identically to FinBERT approach.

**Key techniques from research:**
- **DK-CoT (Domain Knowledge Chain-of-Thought):** Inject domain-specific financial knowledge into the reasoning chain. E.g., "This company is in the semiconductor sector; supply chain disruptions are typically negative for chip stocks."
- **FinDPO (Preference Optimization):** Fine-tune LLMs on human-preference-ranked financial sentiment datasets. Achieves state-of-the-art results on small financial corpora.
- **Repeated sampling:** Generate N sentiment scores per headline and take the mode — reduces noise from LLM stochasticity. (We already use this pattern in `llm_views.py`.)

**Trade-offs vs. FinBERT:** LLMs are 100-1000× slower per headline and cost API credits. FinBERT is free, fast, and deterministic. For our pipeline, a hybrid approach makes sense: FinBERT as the default (fast, free), with LLM scoring as an optional enrichment for high-confidence signals or pre-earnings periods.

**Implementation difficulty:** Low (we already have the LLM and news infrastructure).

**Key papers:**
- FinLlama (ACM ICAIF 2024) — Llama 2 7B fine-tuned for financial sentiment with quantified strength
- Springer (2025) — DK-CoT strategy with knowledge-enhanced LLM sentiment prediction
- FinDPO (ACM 2025) — Preference optimization for financial sentiment

### 8.3 Market-Wide Sentiment Indicators (VIX, Put/Call, Fear & Greed) ✅

> **Implemented in:** `trading_bot_bl/market_sentiment.py`. VIX and put/call ratio with 252-day z-score normalization. Composite MSI = −0.5 × vix_z − 0.5 × pcr_z. Two-tier regime classification (FEAR/GREED/NEUTRAL) with contrarian position sizing. SPY 200-SMA trend regime overlay (BULL/CAUTION/BEAR/SEVERE_BEAR) for bear market protection. Feature-flagged via `MARKET_SENTIMENT_ENABLED` and `SPY_REGIME_ENABLED`.

**What it is:** Quantitative market-wide sentiment gauges that measure aggregate investor emotion: the VIX (CBOE Volatility Index, "fear gauge"), the equity put/call ratio, and the CNN Fear & Greed Index (a composite of 7 sub-indicators).

**Why it matters for us:** These indicators capture regime-level sentiment that individual stock analysis misses. High VIX (>30) and elevated put/call ratios (>1.0) historically coincide with market bottoms — contrarian buying opportunities. Low VIX (<15) with extreme greed readings often precede corrections. Our position sizing and risk management should scale with market-wide fear/greed.

**How to integrate:** Fetch VIX from Yahoo Finance (ticker `^VIX` — we already use yfinance). Put/call ratio from CBOE daily data. Compute a composite market sentiment score: `market_sentiment = normalize(vix_z_score, put_call_z_score)`. Use as a portfolio-level modifier:
- Extreme fear (VIX > 30, P/C > 1.2): increase position sizes by 10-20% (contrarian) or reduce new entries (momentum)
- Extreme greed (VIX < 15, P/C < 0.6): tighten stops, reduce new position sizes
- Neutral: no modification

**The math:** VIX z-score: `z_vix = (VIX_current - VIX_mean_252) / VIX_std_252`. Put/call z-score: similarly normalized against 1-year history. Market sentiment index: `MSI = -0.5 × z_vix - 0.5 × z_pc` (negative because high VIX/P-C = fear = potentially contrarian bullish).

**Dual-use strategy (backed by research):** The most effective practitioners combine momentum and contrarian approaches: ride sentiment-driven momentum in trending markets, flip to contrarian positioning when sentiment hits extremes. Our HMM regime detection (Section 3.1) would determine which mode to use.

**Implementation difficulty:** Very Low. VIX is a free yfinance download. The logic is a few lines of z-score computation added to the risk manager or portfolio optimizer.

### 8.4 Social Media Sentiment (Reddit, Twitter/X) ⬜

**What it is:** Extracting sentiment from social media platforms — particularly finance-focused communities like r/WallStreetBets, StockTwits, and Financial Twitter — to gauge retail investor sentiment and detect momentum buildups before they appear in price.

**Why it matters for us:** Research shows Reddit discussions (particularly r/WallStreetBets) exhibit stronger predictive signals for abrupt volatility shifts than traditional news, while Twitter sentiment aligns more with gradual market reactions. Simpler volume metrics (comment count, Google Trends) often outperform sophisticated NLP sentiment scores.

**Critical caveats from research:**
- Social media sentiment has only a **weak correlation** with actual stock prices when measured directly
- **Volume-based metrics** (number of mentions, comment count, Google Trends) are often more predictive than NLP sentiment scores
- Reddit/WSB attention **increases risk-taking** and **reduces holding-period returns**: positions created during peak WSB attention realize -8.5% holding period returns on average
- Social media sentiment shows **herding behavior** — peer influence drives sentiment contagion, not independent analysis
- Predictive power is strongest for **meme stocks** and decays for large-cap liquid equities

**Practical implementation approach:** Rather than attempting full social media NLP (which is noisy), use volume-based signals: track mention counts for held tickers via StockTwits API or Reddit API. A sudden spike in social mentions for a held position is a **risk signal** (potential meme/pump dynamics) — trigger tighter stops or reduced position size, not a buy signal.

**The math:** Mention z-score: `z_mentions = (mentions_today - mentions_mean_30d) / mentions_std_30d`. If `z_mentions > 3.0` for a held position: flag for review, tighten trailing stop. If `z_mentions > 3.0` for a signal candidate: consider reducing confidence score.

**Implementation difficulty:** Medium. Reddit and Twitter APIs require authentication. StockTwits is easier (public API). The value proposition is lower than news sentiment for our use case (fundamentals-driven large-cap trading).

**Key papers:**
- Li & Li (2024, SSRN) — Google search sentiment predicts meme stock returns at 3-7 day horizons, Bloomberg news at 7-14 day horizons
- ScienceDirect (2024) — WSB attention increases uninformed trading and reduces holding-period returns
- ResearchGate (2025) — Reddit stronger for volatility prediction, Twitter for gradual reactions

### 8.5 Options Flow & Unusual Activity ⬜

**What it is:** Monitoring large or unusual options trades from institutional players — hedge funds, market makers, and proprietary desks that drive the majority of options volume. Unusual options activity (high volume relative to open interest, especially out-of-the-money contracts expiring within 35 days) often precedes significant price moves.

**Why it matters for us:** Options markets price in information before the equity market does. A large block of calls bought on a stock we're considering is a confirmation signal; a surge in puts on a held position is an early warning. Dark pool prints (large block trades executed off-exchange) reveal hidden institutional accumulation or distribution.

**How to integrate:** Use the Unusual Whales API or FlowAlgo API (paid, ~$30-50/month) to fetch daily unusual options activity for our universe. Compute a net options sentiment score: `options_sent = (bullish_premium - bearish_premium) / total_premium`. Integrate as a signal confirmation/filter: a BUY signal confirmed by net bullish options flow gets a confidence boost; a BUY contradicted by heavy put buying gets a confidence penalty.

**Key metrics:**
- **Net premium flow:** `bullish_premium - bearish_premium` — measures directional institutional conviction
- **Put/call volume ratio per ticker:** Elevated put volume on a specific stock vs. its norm = bearish institutional positioning
- **Sweep detection:** Options "sweeps" (aggressive fills across multiple exchanges) indicate urgency — institutions willing to pay up
- **Dark pool net flow:** Net buying vs. selling in dark pools signals hidden accumulation/distribution

**Implementation difficulty:** Medium. Requires a paid data subscription. The analysis logic is straightforward once data is available. Most valuable as a confirmation layer rather than a primary signal.

### 8.6 Earnings Sentiment & Post-Earnings Announcement Drift (PEAD) 🔶

> **Partially implemented:** Full earnings event filter across both signal generation and execution pipelines. **Risk-side (trading_bot_bl):** Earnings blackout in `earnings.py` blocks new entries within 3d pre / 1d post window. Integrated into RiskManager. Monitor warns about held positions approaching earnings AND tightens stops for profitable positions within the blackout window (locks in 50% of unrealised gain). **Signal-side (quant_analysis_bot):** `EarningsContext` in `signals.py` adjusts confidence score based on proximity (-3 on earnings day, -2 within blackout, +1/-1 for strong positive/negative post-earnings surprise). Earnings date and surprise data flow from `pead.py` through `cli.py` to signal generation. New `DailySignal` fields: `days_to_earnings`, `earnings_date`, `last_surprise_pct`, `earnings_confidence_adj`. 26 unit tests in `test_earnings_filter.py`. Full PEAD alpha strategy (SUE-based signal source) not yet implemented.

**What it is:** The well-documented anomaly where stocks continue drifting in the direction of an earnings surprise for 60-90 days after the announcement. A positive earnings surprise (beat estimates) leads to continued positive drift; a negative surprise leads to continued decline. The effect is strongest when combined with investor attention metrics.

**Why it matters for us:** Our backtesting doesn't account for earnings events. A strategy might generate a BUY signal right before a negative earnings report, leading to a large gap-down that blows through the stop loss. Conversely, PEAD provides a well-researched alpha source: buying stocks that beat earnings and riding the 60-day drift.

**Quantified alpha (recent research):**
- A hedge portfolio going long top SUE (Standardized Unexpected Earnings) decile and short bottom decile generates 5.1% risk-adjusted returns over 3 months (~20% annualized) — Garfinkel, Hribar & Hsiao (2024)
- Chinese market evidence: 6.78% quarterly excess return from earnings surprise strategy
- AI-enhanced PEAD: CNN analysis of visual earnings data patterns predicts drift with 3.6% spread over 63 days

**How to integrate (two uses):**

**Use 1 — Earnings event filter (risk management):** Before generating signals, check if any ticker has earnings in the next 3 days (Yahoo Finance earnings calendar via yfinance). If so, either skip the signal (too risky — gaps can exceed SL) or widen the stop loss to accommodate gap risk. After earnings, if the stock beat estimates, boost confidence; if it missed, demote.

**Use 2 — PEAD as a signal source:** After each earnings release, compute the earnings surprise: `SUE = (EPS_actual - EPS_estimate) / std(EPS_surprises_8q)`. If SUE is in the top decile, generate a BUY signal with a 60-day time horizon and wider stops. This would be a new strategy added to `strategies.py`.

**Important caveat:** Recent debate (2025) questions whether PEAD still generates alpha for liquid large-caps after excluding microcaps (t-stat drops from 2.18 to 1.43). The effect may be stronger for mid/small-caps in our universe. AI/LLM-driven analysis may be accelerating information absorption, reducing the drift window.

**The math:** Standardized Unexpected Earnings: `SUE = (EPS_actual - EPS_consensus) / σ(EPS_surprise_history)`. Earnings surprise as a sentiment modifier: `conf_adjustment = clip(SUE * 0.5, -2, +2)` — applied to the signal's confidence score.

**Implementation difficulty:** Low-Medium. Earnings dates and estimates are available from yfinance. The earnings filter is trivial; a full PEAD strategy requires more work.

**Key papers:**
- Garfinkel, Hribar & Hsiao (2024) — CNN-based PEAD prediction, 3.6% 63-day spread
- Lan et al. (2024, ScienceDirect) — Earnings surprise + investor attention + PEAD investing strategy
- CFA Institute (2025) — Discussion of whether generative AI is disrupting PEAD effectiveness

### 8.7 Institutional Ownership Changes (13F Filings) ⬜

**What it is:** SEC Form 13F requires institutional investment managers with >$100M in AUM to disclose their equity holdings quarterly. By computing changes between filings, you can measure net institutional flow — where the biggest players are accumulating or distributing.

**Why it matters for us:** Institutional ownership changes are a slow-moving but high-conviction sentiment signal. Research shows stocks with the highest institutional sentiment scores have outperformed those with the lowest by 12% per annum from 2007–2024, with a Sharpe ratio of 0.83 and low turnover of 1.7%/day.

**Key patterns that generate alpha:**
- **Cluster buying:** Multiple institutions initiating new positions in the same quarter
- **Inflection points:** A stock that was being sold by institutions for 2+ quarters suddenly sees buying
- **Cessation of selling:** Insiders who had been consistently selling stop — often precedes positive catalysts

**How to integrate:** Fetch quarterly 13F data from SEC EDGAR (free) or a provider like WhaleWisdom / Fintel API. Compute a quarterly institutional sentiment score: `inst_sent = (new_positions + increased_positions - decreased_positions - closed_positions) / total_filers`. Use as a slow-moving signal modifier: positive institutional flow → confidence boost for BUY signals on that ticker; negative flow → confidence penalty.

**Important caveat:** 13F data is reported with a 45-day lag (filed 45 days after quarter end). This makes it a confirmation/positioning signal rather than a timing signal. Most useful for filtering out stocks that institutions are actively exiting.

**Implementation difficulty:** Medium. SEC EDGAR provides free XML/JSON 13F data. Parsing and aggregating across filers requires some work. Commercial APIs (ExtractAlpha, WhaleWisdom) simplify this significantly.

**Key research:**
- ExtractAlpha 13F Sentiment Signal — 12% annual outperformance, 0.83 Sharpe (2007–2024)
- Alpha Architect — Cluster insider buying patterns and their predictive power

### 8.8 Composite Sentiment Integration Architecture 🔶

> **Partially implemented:** VIX + put/call ratio are combined into MSI. LLM views are integrated into Black-Litterman. SPY trend regime provides a third signal layer. However, the full multi-source architecture (news + options + earnings + 13F + social media unified into one composite score per ticker) is not built.

**What it is:** Rather than using any single sentiment source, build a multi-source sentiment layer that aggregates news, market-wide, options, earnings, and institutional signals into a unified sentiment score per ticker.

**Why it matters for us:** Each sentiment source has different characteristics — news sentiment is fast but noisy, institutional flow is slow but high-conviction, options flow captures informed positioning. A composite approach captures the best of each while diversifying away noise from any single source.

**Proposed architecture:**

```
Per-ticker sentiment aggregation:
┌──────────────────────────────────────────────────────┐
│ Source              │ Weight │ Update Freq │ Latency  │
├──────────────────────────────────────────────────────┤
│ News (FinBERT)      │ 0.30   │ Hourly      │ Minutes  │
│ Market-wide (VIX)   │ 0.15   │ Daily       │ Real-time│
│ Options flow        │ 0.20   │ Daily       │ Same-day │
│ Earnings surprise   │ 0.20   │ Quarterly   │ Same-day │
│ Institutional (13F) │ 0.15   │ Quarterly   │ 45-day   │
└──────────────────────────────────────────────────────┘

composite_sentiment = Σ(source_score × weight × availability_flag)
                      / Σ(weight × availability_flag)
```

**How it modifies the pipeline:**

1. **Signal generation (quant_analysis_bot):** `composite_sentiment` becomes an additional feature in the confidence scoring function (signals.py). A BUY signal with strongly positive composite sentiment: confidence +1. Strongly negative: confidence -1 (may force HOLD if total drops below threshold).

2. **Portfolio optimization (trading_bot_bl):** Sentiment can inform Black-Litterman view confidence (Ω). Positive sentiment → tighter Ω (more confident view) → larger BL weight.

3. **Risk management:** Market-wide extreme fear → either halt new entries (momentum-preservation) or scale up entries (contrarian). Configurable per strategy.

4. **Position monitoring:** Sudden negative sentiment shift for a held position → tighten trailing stop or trigger early exit review.

**Graceful degradation:** If any source is unavailable (API down, no recent headlines), the composite score is computed from available sources only — the denominator adjusts. No single source is required.

**Implementation difficulty:** Medium overall. Each individual source is Low-Medium; the integration layer is the main work.

---

## 9. Implementation Roadmap

Prioritized by impact-to-effort ratio. Grouped into three phases.

### Phase 1: Quick Wins (Low effort, high impact)

| Improvement | Component | Effort | Impact | Status |
|---|---|---|---|---|
| **Ledoit-Wolf shrinkage** | black_litterman.py | Very Low | High | ✅ Implemented |
| **Probabilistic Sharpe Ratio** | journal_analytics.py | Low | High | ✅ Implemented |
| **Volatility-targeted sizing** | executor.py / signals.py | Low | High | 🔶 Partial (VIX-level only) |
| **Deflated Sharpe Ratio** | backtest.py | Low | Medium | ✅ Implemented |
| **Execution timing** | GitHub Actions | Very Low | Low-Medium | ✅ Implemented (10:15 AM ET) |
| **ATR Chandelier exits** | monitor.py | Low | Medium | 🔶 Partial (ATR trailing, not highest-high anchored) |
| **Exponentially weighted covariance** | black_litterman.py | Very Low | Medium | ✅ Implemented |
| **VIX / market-wide sentiment** | market_sentiment.py | Very Low | Medium | ✅ Implemented (+ SPY regime filter) |
| **Earnings event filter** | signals.py / risk.py | Low | Medium-High | ✅ Implemented |

### Phase 2: Meaningful Upgrades (Medium effort, high impact)

| Improvement | Component | Effort | Impact | Status |
|---|---|---|---|---|
| **HRP allocation** | portfolio_optimizer.py | Medium | High | ⬜ Not implemented |
| **HMM regime detection** | new module | Medium | High | ⬜ Not implemented (SPY SMA-based regime exists) |
| **GARCH volatility forecasting** | signals.py / risk.py | Medium | High | ⬜ Not implemented |
| **CPPI drawdown control** | risk.py | Medium | High | ⬜ Not implemented |
| **Walk-forward backtesting** | backtest.py | Medium | High | ✅ Implemented (70/30 split) |
| **Fractional differentiation** | new preprocessing | Low-Medium | Medium | ⬜ Not implemented |
| **Slippage model** | broker.py / journal.py | Low | Medium | ✅ Implemented (limit orders + tracking) |
| **Correlation breakdown monitoring** | monitor.py / risk.py | Low | Medium | ⬜ Not implemented |
| **FinBERT news sentiment** | new sentiment module | Low-Medium | High | ⬜ Not implemented |
| **LLM headline scoring** | llm_views.py | Low | Medium-High | ✅ Implemented (LLM views + news context) |
| **Social media mention volume** | new module | Medium | Medium | ⬜ Not implemented |

### Phase 3: Advanced Capabilities (Higher effort, potentially transformative)

| Improvement | Component | Effort | Impact | Status |
|---|---|---|---|---|
| **Black-Litterman integration** | black_litterman.py | Medium-High | High | ✅ Implemented (full BL + LLM views) |
| **Meta-labeling** | new module | High | High | ⬜ Not implemented |
| **Triple barrier labeling** | backtest.py | Medium-High | High | ⬜ Not implemented |
| **CVaR portfolio constraints** | risk.py / portfolio_optimizer.py | Medium | Medium-High | ⬜ Not implemented |
| **CSCV overfitting detection** | cscv.py | Medium-High | Medium | ✅ Implemented (PBO + logits + rank corr) |
| **Factor-aware signals** | signals.py | Medium-High | Medium | ⬜ Not implemented |
| **Limit order entries** | broker.py | Medium | Medium | ✅ Implemented |
| **Random Matrix Theory denoising** | portfolio_optimizer.py | Medium | Medium | ⬜ Not implemented |
| **PEAD strategy** | strategies.py / earnings.py | Medium | Medium-High | 🔶 Earnings blackout filter implemented |
| **Institutional 13F sentiment** | new module | Medium | Medium | ⬜ Not implemented |
| **Options flow integration** | new module | Medium | Medium-High | ⬜ Not implemented |
| **Composite sentiment layer** | new sentiment module | Medium-High | High | 🔶 Partial (VIX + P/C + SPY + LLM) |

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

### Papers (Sentiment & Alternative Data)
- Araci, D. — "FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models" (2019, arXiv)
- Tak, R. & Pele, D.T. — "Enhancing Trading Performance Through Sentiment Analysis with LLMs: Evidence from the S&P 500" (2025, PICBE)
- FinLlama — "LLM-Based Financial Sentiment Analysis for Algorithmic Trading" (2024, ACM ICAIF)
- FinDPO — "Financial Sentiment Analysis for Algorithmic Trading through Preference Optimization of LLMs" (2025, ACM)
- Springer (2025) — "Leveraging LLMs as News Sentiment Predictor: A Knowledge-Enhanced Strategy" (Discover Computing)
- Li, J. & Li, Z. — "Sentiment, Social Media, and Meme Stock Return Predictability" (2024, SSRN)
- ScienceDirect (2024) — "Social Media Attention and Retail Investor Behavior: Evidence from r/WallStreetBets"
- ICCS (2025) — "Predicting Stock Prices with ChatGPT-Annotated Reddit Sentiment"
- Garfinkel, J., Hribar, P. & Hsiao, P. — "Can Generative AI Disrupt PEAD?" (2024/2025, CFA Institute)
- Lan, Q. et al. — "Post-Earnings Announcement Drift: Earnings Surprise, Investor Attention, and Strategy" (2024, ScienceDirect)
- ExtractAlpha — "13F Stock Sentiment Signal" (2024, Fact Sheet — 12% annual outperformance, 0.83 Sharpe)
- ScienceDirect (2025) — "GNN-Based Social Media Sentiment Analysis for Stock Market Forecasting"

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
| `transformers` | FinBERT / LLM sentiment inference | `pip install transformers torch` |
| `finnhub-python` | Free financial news API | `pip install finnhub-python` |
| `praw` | Reddit API wrapper (social media sentiment) | `pip install praw` |
| `yfinance` | VIX data, earnings dates, fundamentals | (already installed) |

---

*Last updated: April 3, 2026*
