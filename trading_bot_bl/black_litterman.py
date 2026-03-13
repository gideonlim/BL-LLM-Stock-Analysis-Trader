"""
Black-Litterman portfolio optimization model.

Combines market equilibrium returns (prior) with signal-generated
views (posterior) to produce optimal portfolio weights. Optionally
enriched with LLM-generated views for qualitative insight.

References:
    - Black & Litterman, "Global Portfolio Optimization" (1992)
    - He & Litterman, "The Intuition Behind Black-Litterman" (1999)
    - Idzorek, "A Step-by-Step Guide to the Black-Litterman Model"
    - Young & Bin, "LLM-Enhanced Black-Litterman" (ICLR 2025)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
ANNUALIZATION_FACTOR = np.sqrt(252)
DAILY_RISK_FREE = 0.05 / 252  # ~5% annual, daily


@dataclass
class BLView:
    """A single investor view for the Black-Litterman model."""

    ticker: str
    expected_return: float  # annualized expected return
    confidence: float       # 0-1, higher = more certain
    source: str = "quant"   # "quant", "llm", or "blended"
    reasoning: str = ""


@dataclass
class BLResult:
    """Output of the Black-Litterman optimization."""

    posterior_returns: dict[str, float]    # ticker → annual return
    optimal_weights: dict[str, float]      # ticker → weight (0-1)
    equilibrium_returns: dict[str, float]  # ticker → prior return
    views_used: list[BLView] = field(default_factory=list)
    portfolio_sharpe: float = 0.0
    portfolio_return: float = 0.0
    portfolio_volatility: float = 0.0


# ── Ledoit-Wolf Shrinkage ────────────────────────────────────────

def ledoit_wolf_shrinkage(returns: np.ndarray) -> np.ndarray:
    """
    Compute the Ledoit-Wolf shrinkage estimator for the covariance
    matrix. Shrinks toward a scaled identity target.

    This is a pure-numpy implementation of the analytical formula
    from Ledoit & Wolf (2004) "A well-conditioned estimator for
    large-dimensional covariance matrices."

    Args:
        returns: T×N matrix of asset returns (T observations,
                 N assets).

    Returns:
        N×N shrunk covariance matrix.
    """
    t, n = returns.shape

    # Sample covariance (using T-1 for unbiased estimate)
    sample_cov = np.cov(returns, rowvar=False)

    # Shrinkage target: scaled identity
    # μ = average variance (trace / n)
    mu = np.trace(sample_cov) / n
    target = mu * np.eye(n)

    # Compute optimal shrinkage intensity
    # Based on Ledoit-Wolf (2004) Theorem 3.1
    x = returns - returns.mean(axis=0)
    s = x.T @ x / t  # non-bias-corrected for the formula

    # sum of squared differences from target
    delta = s - target
    sum_sq_delta = np.sum(delta ** 2)

    # Estimate of asymptotic variances
    y = x ** 2
    phi = np.sum(y.T @ y / t - s ** 2)

    # Shrinkage intensity (clamped to [0, 1])
    kappa = (phi / t) / sum_sq_delta if sum_sq_delta > 0 else 1.0
    shrinkage = max(0.0, min(1.0, kappa))

    shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

    log.debug(
        f"Ledoit-Wolf shrinkage intensity: {shrinkage:.4f} "
        f"(N={n}, T={t})"
    )
    return shrunk_cov


# ── Equilibrium Returns ──────────────────────────────────────────

def compute_equilibrium_returns(
    market_caps: dict[str, float],
    cov_matrix: np.ndarray,
    tickers: list[str],
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """
    Compute implied equilibrium returns via reverse optimization.

    π = δ × Σ × w_mkt

    The market "believes" these returns given current
    market-cap weights. This is the Black-Litterman prior.

    Args:
        market_caps: Ticker → market capitalization in dollars.
        cov_matrix: N×N covariance matrix (daily returns).
        tickers: Ordered list of ticker symbols matching cov_matrix.
        risk_aversion: δ parameter (default 2.5).

    Returns:
        N-length array of implied daily equilibrium returns.
    """
    # Market-cap weights
    caps = np.array([market_caps.get(t, 0.0) for t in tickers])
    total_cap = caps.sum()

    if total_cap <= 0:
        log.warning("No valid market caps — using equal weights")
        w_mkt = np.ones(len(tickers)) / len(tickers)
    else:
        w_mkt = caps / total_cap

    # Implied equilibrium returns (daily)
    pi = risk_aversion * cov_matrix @ w_mkt

    return pi


def estimate_risk_aversion(
    market_returns: np.ndarray,
    risk_free_rate: float = DAILY_RISK_FREE,
) -> float:
    """
    Estimate the market's implied risk aversion from historical
    market returns.

    δ = (E[R_m] - R_f) / Var(R_m)

    Args:
        market_returns: Array of daily market returns.
        risk_free_rate: Daily risk-free rate.

    Returns:
        Estimated risk aversion coefficient.
    """
    excess_return = market_returns.mean() - risk_free_rate
    variance = market_returns.var()

    if variance <= 0:
        return 2.5  # sensible default

    delta = excess_return / variance
    # Clamp to reasonable range
    return float(np.clip(delta, 0.5, 10.0))


# ── Signal-to-View Conversion ────────────────────────────────────

def signals_to_views(
    signals: list,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert our quant bot signals into Black-Litterman views.

    Each BUY signal becomes an absolute view: "I expect stock X
    to return Q% annualized." The confidence score maps to view
    uncertainty in the Omega matrix.

    Args:
        signals: List of Signal objects (from models.py).
        tickers: Ordered ticker list matching the covariance matrix.

    Returns:
        P: K×N picking matrix (K views, N assets)
        Q: K-length view returns vector (daily)
        omega_diag: K-length diagonal of uncertainty matrix
    """
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)

    views_p = []
    views_q = []
    views_omega = []

    for sig in signals:
        if sig.ticker not in ticker_idx:
            continue
        if sig.signal_raw != 1:  # only BUY signals are views
            continue

        idx = ticker_idx[sig.ticker]

        # Pick vector: absolute view on this single stock
        p_row = np.zeros(n)
        p_row[idx] = 1.0
        views_p.append(p_row)

        # Expected return: convert annualized to daily
        # Use the backtest Sharpe as a proxy for expected return
        # Sharpe * vol gives excess return
        # Fallback: use a modest view based on confidence
        annual_return = _estimate_view_return(sig)
        daily_return = annual_return / 252
        views_q.append(daily_return)

        # Uncertainty: inversely proportional to confidence
        # confidence_score ranges 0-6
        # High confidence (5-6) → low omega → view dominates
        # Low confidence (0-1) → high omega → prior dominates
        omega = _confidence_to_omega(sig.confidence_score)
        views_omega.append(omega)

    if not views_p:
        return np.array([]).reshape(0, n), np.array([]), np.array([])

    P = np.array(views_p)
    Q = np.array(views_q)
    omega_diag = np.array(views_omega)

    return P, Q, omega_diag


def _estimate_view_return(signal) -> float:
    """
    Estimate the annualized expected return for a signal's view.

    Uses the signal's Sharpe ratio and realized volatility when
    available. Falls back to confidence-based estimates.
    """
    # Best case: Sharpe × implied_vol gives excess return
    if signal.sharpe > 0 and signal.current_price > 0:
        # Estimate vol from stop loss distance as a proxy
        if signal.stop_loss_price > 0:
            sl_distance = abs(
                signal.current_price - signal.stop_loss_price
            ) / signal.current_price
            # ATR-based SL is ~1.5 ATR; daily vol ≈ SL_pct / 1.5
            daily_vol = sl_distance / 1.5
            annual_vol = daily_vol * np.sqrt(252)
            # Expected return = Sharpe × vol + risk-free
            annual_return = signal.sharpe * annual_vol + 0.05
            return float(np.clip(annual_return, 0.02, 0.60))

    # Fallback: confidence-based graduated view
    confidence_returns = {
        6: 0.25,  # HIGH: 25% expected
        5: 0.20,
        4: 0.15,
        3: 0.12,  # MEDIUM: 12%
        2: 0.08,
        1: 0.05,
        0: 0.03,  # LOW: 3%
    }
    return confidence_returns.get(signal.confidence_score, 0.08)


def _confidence_to_omega(confidence_score: int) -> float:
    """
    Map confidence score (0-6) to Black-Litterman view uncertainty.

    Lower omega = more certain view = more weight in posterior.
    Based on He & Litterman (1999) proportionality approach,
    scaled for our 0-6 confidence system.

    The omega values represent the variance of the view's
    distribution. A smaller omega means tighter distribution
    (more confident view).
    """
    # Exponential mapping: confidence → uncertainty
    # HIGH (5-6): omega ≈ 0.0001 (view almost certain)
    # MEDIUM (3-4): omega ≈ 0.001 (moderate uncertainty)
    # LOW (0-2): omega ≈ 0.01 (high uncertainty → prior dominates)
    omega_map = {
        6: 0.00005,
        5: 0.0001,
        4: 0.0005,
        3: 0.001,
        2: 0.005,
        1: 0.01,
        0: 0.05,
    }
    return omega_map.get(confidence_score, 0.005)


# ── LLM View Integration ─────────────────────────────────────────

def integrate_llm_views(
    quant_P: np.ndarray,
    quant_Q: np.ndarray,
    quant_omega: np.ndarray,
    llm_views: list[BLView],
    tickers: list[str],
    llm_weight: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Blend LLM-generated views with quantitative signal views.

    Two strategies:
    1. REPLACE: If LLM has a view for a ticker the quant bot also
       covers, blend the two Q values and take the more conservative
       omega.
    2. ADD: If LLM has views for tickers not covered by quant
       signals, add them as new rows to P, Q, Ω.

    Args:
        quant_P, quant_Q, quant_omega: From signals_to_views().
        llm_views: List of BLView from the LLM module.
        tickers: Ordered ticker list.
        llm_weight: Weight for LLM views when blending (0-1).

    Returns:
        Updated P, Q, omega_diag with LLM views integrated.
    """
    if not llm_views:
        return quant_P, quant_Q, quant_omega

    ticker_idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)

    # Find which quant rows correspond to which tickers
    quant_ticker_rows: dict[str, int] = {}
    for row_i in range(quant_P.shape[0]):
        active_idx = np.argmax(quant_P[row_i])
        quant_ticker_rows[tickers[active_idx]] = row_i

    new_P = list(quant_P)
    new_Q = list(quant_Q)
    new_omega = list(quant_omega)

    for view in llm_views:
        if view.ticker not in ticker_idx:
            continue

        daily_return = view.expected_return / 252
        llm_omega = _confidence_to_omega(
            int(view.confidence * 6)  # map 0-1 to 0-6
        )

        if view.ticker in quant_ticker_rows:
            # BLEND: weighted average of quant and LLM views
            row_i = quant_ticker_rows[view.ticker]
            quant_q = new_Q[row_i]
            new_Q[row_i] = (
                (1 - llm_weight) * quant_q
                + llm_weight * daily_return
            )
            # Take the more conservative (larger) omega
            new_omega[row_i] = min(
                new_omega[row_i], llm_omega
            )
            log.info(
                f"  BL: blended LLM view for {view.ticker} "
                f"(quant={quant_q * 252:.1%}, "
                f"llm={view.expected_return:.1%}, "
                f"blended={new_Q[row_i] * 252:.1%})"
            )
        else:
            # ADD: new view row
            p_row = np.zeros(n)
            p_row[ticker_idx[view.ticker]] = 1.0
            new_P.append(p_row)
            new_Q.append(daily_return)
            new_omega.append(llm_omega)
            log.info(
                f"  BL: added LLM-only view for {view.ticker} "
                f"(return={view.expected_return:.1%}, "
                f"confidence={view.confidence:.2f})"
            )

    return (
        np.array(new_P),
        np.array(new_Q),
        np.array(new_omega),
    )


# ── Core BL Posterior ─────────────────────────────────────────────

def compute_posterior_returns(
    pi: np.ndarray,
    cov_matrix: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    omega_diag: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Compute Black-Litterman posterior expected returns.

    E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]

    Args:
        pi: N-length equilibrium return vector (daily).
        cov_matrix: N×N covariance matrix (daily).
        P: K×N picking matrix.
        Q: K-length view returns vector (daily).
        omega_diag: K-length diagonal of view uncertainty matrix.
        tau: Scalar uncertainty in the prior (typically 0.01-0.1).

    Returns:
        N-length posterior expected returns (daily).
    """
    n = len(pi)

    if P.shape[0] == 0:
        # No views — posterior equals prior
        log.info("  BL: no views provided, using equilibrium returns")
        return pi

    # τΣ and its inverse
    tau_cov = tau * cov_matrix
    tau_cov_inv = np.linalg.inv(
        tau_cov + 1e-10 * np.eye(n)  # regularize
    )

    # Ω (diagonal matrix of view uncertainties)
    Omega = np.diag(omega_diag)
    Omega_inv = np.diag(1.0 / (omega_diag + 1e-12))

    # BL formula
    # M = (τΣ)⁻¹ + P'Ω⁻¹P
    M = tau_cov_inv + P.T @ Omega_inv @ P

    # M⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
    posterior = np.linalg.solve(
        M, tau_cov_inv @ pi + P.T @ Omega_inv @ Q
    )

    return posterior


# ── Weight Optimization ───────────────────────────────────────────

def optimize_weights(
    posterior_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list[str],
    risk_aversion: float = 2.5,
    max_weight: float = 0.15,
    min_weight: float = 0.0,
) -> dict[str, float]:
    """
    Compute optimal portfolio weights from posterior returns.

    Uses the analytical solution: w* = (δΣ)⁻¹ × E[R]
    then clamps and normalizes to respect position limits.

    Args:
        posterior_returns: N-length BL posterior returns (daily).
        cov_matrix: N×N covariance matrix.
        tickers: Ordered ticker list.
        risk_aversion: δ parameter.
        max_weight: Maximum weight per asset.
        min_weight: Minimum weight per asset (0 = allow zero).

    Returns:
        Dict of ticker → optimal weight.
    """
    n = len(tickers)

    # Analytical unconstrained weights
    delta_cov_inv = np.linalg.inv(
        risk_aversion * cov_matrix + 1e-10 * np.eye(n)
    )
    raw_weights = delta_cov_inv @ posterior_returns

    # Remove short positions (long-only constraint)
    raw_weights = np.maximum(raw_weights, min_weight)

    # Cap per-stock weight
    raw_weights = np.minimum(raw_weights, max_weight)

    # Normalize to sum to target exposure (max 80%)
    total = raw_weights.sum()
    if total > 0:
        # Scale to sum to 1, then individual weights serve as
        # relative allocations — executor will apply exposure cap
        raw_weights = raw_weights / total

    weights = {
        tickers[i]: round(float(raw_weights[i]), 6)
        for i in range(n)
    }

    return weights


# ── Main BL Pipeline ──────────────────────────────────────────────

def run_black_litterman(
    signals: list,
    held_positions: dict,
    portfolio_equity: float,
    market_caps: dict[str, float] | None = None,
    llm_views: list[BLView] | None = None,
    tau: float = 0.05,
    risk_aversion: float | None = None,
    max_position_pct: float = 15.0,
    lookback_days: int = 60,
    llm_weight: float = 0.3,
) -> BLResult | None:
    """
    Full Black-Litterman pipeline.

    1. Fetch returns and build covariance matrix (Ledoit-Wolf)
    2. Compute equilibrium returns from market-cap weights
    3. Convert signals to views (P, Q, Ω)
    4. Optionally integrate LLM views
    5. Compute posterior returns
    6. Optimize portfolio weights

    Args:
        signals: List of Signal objects from quant bot.
        held_positions: Current portfolio positions dict.
        portfolio_equity: Total account equity.
        market_caps: Ticker → market cap. If None, uses equal
            weighting as the prior (less ideal but functional).
        llm_views: Optional LLM-generated views.
        tau: Prior uncertainty scalar (0.01-0.1).
        risk_aversion: δ parameter. If None, estimated from data.
        max_position_pct: Max weight per stock (%).
        lookback_days: Days of return history for covariance.
        llm_weight: Weight for LLM views when blending (0-1).

    Returns:
        BLResult with posterior returns and optimal weights,
        or None if insufficient data.
    """
    # Collect all relevant tickers
    buy_signals = [s for s in signals if s.signal_raw == 1]
    if not buy_signals:
        log.info("  BL: no BUY signals to optimize")
        return None

    candidate_tickers = [s.ticker for s in buy_signals]
    held_tickers = list(held_positions.keys())
    all_tickers = sorted(set(candidate_tickers + held_tickers))

    log.info(
        f"  BL: {len(candidate_tickers)} candidates, "
        f"{len(held_tickers)} held, "
        f"{len(all_tickers)} total tickers"
    )

    # ── 1. Fetch returns and build covariance ─────────────────
    returns = _fetch_returns(all_tickers, lookback_days)
    if returns is None or returns.shape[1] < 2:
        log.warning(
            "  BL: insufficient return data — "
            "falling back to original ranking"
        )
        return None

    # Align tickers with available data
    available_tickers = list(returns.columns)
    n = len(available_tickers)

    # Ledoit-Wolf shrinkage for robust covariance
    cov_matrix = ledoit_wolf_shrinkage(returns.values)
    log.info(f"  BL: covariance matrix {n}×{n} (Ledoit-Wolf)")

    # ── 2. Equilibrium returns ────────────────────────────────
    if risk_aversion is None:
        # Estimate from SPY if available, else use default
        risk_aversion = _estimate_market_risk_aversion(
            lookback_days
        )
        log.info(
            f"  BL: estimated risk aversion δ={risk_aversion:.2f}"
        )

    if market_caps is None:
        market_caps = _fetch_market_caps(available_tickers)

    pi = compute_equilibrium_returns(
        market_caps, cov_matrix, available_tickers, risk_aversion
    )

    eq_returns = {
        available_tickers[i]: round(float(pi[i] * 252), 4)
        for i in range(n)
    }
    log.info(
        f"  BL: equilibrium returns computed "
        f"(range: {min(eq_returns.values()):.1%} to "
        f"{max(eq_returns.values()):.1%} annual)"
    )

    # ── 3. Convert signals to views ──────────────────────────
    # Filter to only signals with tickers in our data
    valid_signals = [
        s for s in buy_signals
        if s.ticker in set(available_tickers)
    ]
    P, Q, omega_diag = signals_to_views(
        valid_signals, available_tickers
    )
    log.info(f"  BL: {P.shape[0]} quant views generated")

    # ── 4. Integrate LLM views ────────────────────────────────
    views_used = [
        BLView(
            ticker=s.ticker,
            expected_return=float(Q[i] * 252) if i < len(Q) else 0,
            confidence=s.confidence_score / 6.0,
            source="quant",
        )
        for i, s in enumerate(valid_signals)
        if i < len(Q)
    ]

    if llm_views:
        P, Q, omega_diag = integrate_llm_views(
            P, Q, omega_diag,
            llm_views, available_tickers, llm_weight
        )
        views_used.extend(llm_views)
        log.info(
            f"  BL: {len(llm_views)} LLM views integrated "
            f"(weight={llm_weight})"
        )

    # ── 5. Posterior returns ──────────────────────────────────
    posterior = compute_posterior_returns(
        pi, cov_matrix, P, Q, omega_diag, tau
    )

    posterior_dict = {
        available_tickers[i]: round(float(posterior[i] * 252), 4)
        for i in range(n)
    }

    # Log the shift from prior to posterior
    log.info("  BL: posterior returns (top 10 by return):")
    sorted_post = sorted(
        posterior_dict.items(), key=lambda x: -x[1]
    )
    for ticker, ret in sorted_post[:10]:
        eq_ret = eq_returns.get(ticker, 0)
        shift = ret - eq_ret
        log.info(
            f"    {ticker:<7} prior={eq_ret:>+7.2%}  "
            f"posterior={ret:>+7.2%}  "
            f"shift={shift:>+7.2%}"
        )

    # ── 6. Optimize weights ───────────────────────────────────
    weights = optimize_weights(
        posterior, cov_matrix, available_tickers,
        risk_aversion=risk_aversion,
        max_weight=max_position_pct / 100,
    )

    # Portfolio-level metrics
    w_arr = np.array([weights[t] for t in available_tickers])
    port_ret = float(np.dot(w_arr, posterior) * 252)
    port_vol = float(
        np.sqrt(np.dot(w_arr, cov_matrix @ w_arr) * 252)
    )
    port_sharpe = (
        (port_ret - 0.05) / port_vol if port_vol > 0 else 0.0
    )

    log.info(
        f"  BL: optimal portfolio — "
        f"return={port_ret:.1%}, vol={port_vol:.1%}, "
        f"sharpe={port_sharpe:.2f}"
    )

    # Log top weight allocations
    sorted_weights = sorted(
        weights.items(), key=lambda x: -x[1]
    )
    log.info("  BL: target weights (top 10):")
    for ticker, w in sorted_weights[:10]:
        if w > 0.001:
            log.info(f"    {ticker:<7} {w:>7.1%}")

    return BLResult(
        posterior_returns=posterior_dict,
        optimal_weights=weights,
        equilibrium_returns=eq_returns,
        views_used=views_used,
        portfolio_sharpe=round(port_sharpe, 3),
        portfolio_return=round(port_ret, 4),
        portfolio_volatility=round(port_vol, 4),
    )


# ── Helper Functions ──────────────────────────────────────────────

def _fetch_returns(
    tickers: list[str],
    lookback_days: int = 60,
) -> pd.DataFrame | None:
    """Fetch daily returns for tickers using yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — BL unavailable")
        return None

    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=lookback_days + 30)

    try:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]].rename(
                columns={"Close": tickers[0]}
            )

        returns = prices.pct_change().dropna()
        returns = returns.tail(lookback_days)

        # Drop tickers with too little data
        min_data = int(len(returns) * 0.8)
        returns = returns.dropna(axis=1, thresh=min_data)

        return returns

    except Exception as e:
        log.warning(f"Could not fetch returns: {e}")
        return None


def _estimate_market_risk_aversion(
    lookback_days: int = 60,
) -> float:
    """Estimate risk aversion from SPY returns."""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(days=lookback_days + 30)
        spy = yf.download(
            "SPY", start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False, auto_adjust=True,
        )
        if spy.empty:
            return 2.5

        returns = spy["Close"].pct_change().dropna().values
        delta = estimate_risk_aversion(returns)
        return delta

    except Exception:
        return 2.5


def _fetch_market_caps(
    tickers: list[str],
) -> dict[str, float]:
    """Fetch market caps from yfinance for equilibrium weights."""
    caps: dict[str, float] = {}
    try:
        import yfinance as yf
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                cap = info.get(
                    "marketCap",
                    info.get("totalAssets", 0),
                )
                if cap and cap > 0:
                    caps[ticker] = float(cap)
            except Exception:
                pass
    except ImportError:
        pass

    if not caps:
        log.info(
            "  BL: no market caps available, "
            "using equal-weight prior"
        )
        for t in tickers:
            caps[t] = 1.0

    return caps
