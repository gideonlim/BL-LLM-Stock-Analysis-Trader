"""
Portfolio-aware signal ranking using Modern Portfolio Theory.

Instead of just taking the highest composite-score signals, this
module ranks BUY candidates by their **marginal contribution to
portfolio Sharpe ratio** given your current holdings.

Key insight from Markowitz: a stock with a lower individual Sharpe
but low correlation to your existing positions can improve your
portfolio more than a high-Sharpe stock that's correlated with
what you already hold.

The ranking pipeline:
1. Fetch recent daily returns for all candidates + held positions
2. Build a covariance matrix
3. For each candidate, estimate the new portfolio Sharpe if we
   added that stock at its proposed weight
4. Rank by marginal Sharpe improvement (delta vs. current)
5. Penalize sector/correlation clustering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# How many trading days of returns to use for correlation
LOOKBACK_DAYS = 60
ANNUALIZATION_FACTOR = np.sqrt(252)
RISK_FREE_RATE = 0.05 / 252  # ~5% annual, daily


@dataclass
class RankedIntent:
    """An order intent enriched with portfolio-level metrics."""

    intent: object              # OrderIntent (avoid circular import)
    marginal_sharpe: float = 0.0   # change in portfolio Sharpe
    correlation_to_portfolio: float = 0.0
    individual_sharpe: float = 0.0
    diversification_score: float = 0.0  # 0-1, higher = more diversifying
    rank: int = 0


def fetch_returns_matrix(
    tickers: list[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Fetch daily returns for a list of tickers.

    Returns a DataFrame with tickers as columns and dates as rows.
    Uses yfinance for data, with graceful handling of missing data.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — skipping portfolio optimization")
        return pd.DataFrame()

    if not tickers:
        return pd.DataFrame()

    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=lookback_days + 30)  # buffer for weekends

    try:
        data = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            return pd.DataFrame()

        # Extract Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            # Single ticker returns a flat DataFrame
            prices = data[["Close"]].rename(
                columns={"Close": tickers[0]}
            )

        # Calculate daily returns
        returns = prices.pct_change().dropna()

        # Keep only the last N trading days
        returns = returns.tail(lookback_days)

        # Drop tickers with too little data (< 80% of rows)
        min_data = int(len(returns) * 0.8)
        returns = returns.dropna(axis=1, thresh=min_data)

        return returns

    except Exception as e:
        log.warning(f"Could not fetch returns data: {e}")
        return pd.DataFrame()


def compute_portfolio_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Compute annualized Sharpe ratio for a given portfolio.

    Args:
        weights: Array of portfolio weights (must sum to <= 1).
        mean_returns: Mean daily returns per asset.
        cov_matrix: Covariance matrix of daily returns.

    Returns:
        Annualized Sharpe ratio.
    """
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    if port_vol <= 0:
        return 0.0

    daily_sharpe = (port_return - RISK_FREE_RATE) / port_vol
    return float(daily_sharpe * ANNUALIZATION_FACTOR)


def rank_intents_by_marginal_sharpe(
    intents: list,
    held_positions: dict,
    portfolio_equity: float,
) -> list[RankedIntent]:
    """
    Rank BUY intents by their marginal contribution to portfolio
    Sharpe ratio.

    Steps:
    1. Collect all tickers (held + candidates)
    2. Fetch recent returns
    3. Compute current portfolio Sharpe from held positions
    4. For each candidate, estimate new Sharpe if we add it
    5. Rank by marginal Sharpe improvement

    Args:
        intents: List of OrderIntent objects (BUY signals).
        held_positions: Portfolio positions dict from broker.
        portfolio_equity: Total account equity.

    Returns:
        List of RankedIntent, sorted best-first.
    """
    if not intents:
        return []

    # Collect all tickers we need data for
    candidate_tickers = [i.ticker for i in intents]
    held_tickers = list(held_positions.keys())
    all_tickers = list(set(candidate_tickers + held_tickers))

    log.info(
        f"  Portfolio optimizer: {len(candidate_tickers)} candidates, "
        f"{len(held_tickers)} held positions"
    )

    # Fetch returns
    returns = fetch_returns_matrix(all_tickers)

    if returns.empty or len(returns.columns) < 2:
        log.info(
            "  Not enough return data for optimization — "
            "falling back to composite score ranking"
        )
        return [
            RankedIntent(intent=i, rank=idx + 1)
            for idx, i in enumerate(intents)
        ]

    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    ticker_list = list(returns.columns)
    ticker_idx = {t: i for i, t in enumerate(ticker_list)}
    n = len(ticker_list)

    # ── Build current portfolio weights ──────────────────────────
    current_weights = np.zeros(n)
    for ticker, pos in held_positions.items():
        if ticker in ticker_idx:
            mv = abs(pos.get("market_value", 0.0))
            if portfolio_equity > 0:
                current_weights[ticker_idx[ticker]] = (
                    mv / portfolio_equity
                )

    # Cash weight = 1 - sum of position weights
    # (cash doesn't affect Sharpe calculation directly)

    # Current portfolio Sharpe
    if current_weights.sum() > 0:
        current_sharpe = compute_portfolio_sharpe(
            current_weights, mean_returns, cov_matrix
        )
    else:
        current_sharpe = 0.0

    log.info(f"  Current portfolio Sharpe: {current_sharpe:.2f}")

    # ── Evaluate each candidate ──────────────────────────────────
    ranked: list[RankedIntent] = []

    for intent in intents:
        ticker = intent.ticker

        if ticker not in ticker_idx:
            # No data for this ticker — assign neutral rank
            ranked.append(RankedIntent(
                intent=intent,
                marginal_sharpe=0.0,
                diversification_score=0.5,
            ))
            continue

        idx = ticker_idx[ticker]

        # Individual Sharpe
        std = returns[ticker].std()
        if std > 0:
            ind_sharpe = float(
                (returns[ticker].mean() - RISK_FREE_RATE)
                / std * ANNUALIZATION_FACTOR
            )
        else:
            ind_sharpe = 0.0

        # Correlation to existing portfolio
        if current_weights.sum() > 0:
            port_returns = returns.values @ current_weights
            corr = float(np.corrcoef(
                returns[ticker].values, port_returns
            )[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Hypothetical weights: add this stock at proposed weight
        proposed_weight = intent.notional / portfolio_equity if portfolio_equity > 0 else 0.0
        new_weights = current_weights.copy()
        new_weights[idx] += proposed_weight

        # Normalize if over 1.0 (scale everything down proportionally)
        total = new_weights.sum()
        if total > 1.0:
            new_weights = new_weights / total

        # New portfolio Sharpe
        new_sharpe = compute_portfolio_sharpe(
            new_weights, mean_returns, cov_matrix
        )

        marginal = new_sharpe - current_sharpe

        # Diversification score: lower correlation = higher score
        div_score = max(0.0, min(1.0, (1.0 - corr) / 2.0))

        ranked.append(RankedIntent(
            intent=intent,
            marginal_sharpe=round(marginal, 4),
            correlation_to_portfolio=round(corr, 3),
            individual_sharpe=round(ind_sharpe, 2),
            diversification_score=round(div_score, 3),
        ))

    # ── Sort by marginal Sharpe (best improvement first) ──────────
    # Tiebreaker: diversification score, then composite score
    ranked.sort(
        key=lambda r: (
            -r.marginal_sharpe,
            -r.diversification_score,
            -r.intent.signal.composite_score,
        )
    )

    # Assign ranks
    for i, r in enumerate(ranked):
        r.rank = i + 1

    # Log top candidates
    log.info("  Portfolio-optimized ranking (top 15):")
    log.info(
        f"    {'Rank':>4} {'Ticker':<7} {'Composite':>9} "
        f"{'IndSharpe':>9} {'Corr':>6} {'MargSharpe':>10} "
        f"{'DivScore':>8}"
    )
    for r in ranked[:15]:
        log.info(
            f"    {r.rank:>4} {r.intent.ticker:<7} "
            f"{r.intent.signal.composite_score:>9.1f} "
            f"{r.individual_sharpe:>9.2f} "
            f"{r.correlation_to_portfolio:>+6.2f} "
            f"{r.marginal_sharpe:>+10.4f} "
            f"{r.diversification_score:>8.3f}"
        )

    return ranked
