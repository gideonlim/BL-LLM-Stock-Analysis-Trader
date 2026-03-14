"""
Portfolio-aware signal ranking and optimization.

Supports two optimization modes:

1. **Black-Litterman (default)**: Uses the BL model to compute
   posterior returns from market equilibrium + signal views,
   then derives optimal weights. Optionally enriched with LLM
   views for qualitative insight.

2. **Marginal Sharpe (fallback)**: Ranks BUY candidates by their
   marginal contribution to portfolio Sharpe ratio. Used when
   BL data requirements aren't met.

The optimizer re-ranks and re-sizes order intents so the best
portfolio-level allocations execute first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
LOOKBACK_DAYS = 60
ANNUALIZATION_FACTOR = np.sqrt(252)
RISK_FREE_RATE = 0.05 / 252


@dataclass
class RankedIntent:
    """An order intent enriched with portfolio-level metrics."""

    intent: object              # OrderIntent
    marginal_sharpe: float = 0.0
    correlation_to_portfolio: float = 0.0
    individual_sharpe: float = 0.0
    diversification_score: float = 0.0
    bl_weight: float = 0.0     # BL optimal weight (0-1)
    bl_posterior_return: float = 0.0  # BL posterior annual return
    optimization_method: str = "marginal_sharpe"
    rank: int = 0


# ── Black-Litterman Optimization Path ────────────────────────────

def optimize_with_black_litterman(
    intents: list,
    held_positions: dict,
    portfolio_equity: float,
    config=None,
) -> list[RankedIntent] | None:
    """
    Rank and resize intents using the Black-Litterman model.

    Returns None if BL optimization fails (insufficient data,
    missing dependencies), signaling the caller to fall back
    to marginal Sharpe ranking.

    Args:
        intents: List of OrderIntent objects (BUY signals).
        held_positions: Portfolio positions dict.
        portfolio_equity: Total account equity.
        config: TradingConfig with BL parameters.

    Returns:
        List of RankedIntent sorted by BL weight, or None.
    """
    try:
        from trading_bot_bl.black_litterman import (
            run_black_litterman,
            BLView,
        )
        from trading_bot_bl.llm_views import (
            LLMConfig,
            generate_all_views,
        )
        from trading_bot_bl.news_fetcher import fetch_news_batch
    except ImportError as e:
        log.warning(f"BL imports failed: {e}")
        return None

    if not intents:
        return []

    # Collect signals from intents
    signals = [i.signal for i in intents]
    buy_tickers = [s.ticker for s in signals]

    # ── Generate LLM views if enabled ─────────────────────────
    llm_views: list[BLView] = []
    if config and config.llm_views_enabled:
        llm_config = LLMConfig(
            enabled=True,
            provider=config.llm_provider,
            model=config.llm_model,
            num_samples=config.llm_num_samples,
            temperature=config.llm_temperature,
            llm_weight=config.llm_weight,
            max_tickers=config.llm_max_tickers,
        )

        # Fetch news for LLM context
        news_map = fetch_news_batch(buy_tickers, max_headlines=5)

        llm_views = generate_all_views(
            signals=signals,
            config=llm_config,
            news_map=news_map,
        )

    # ── Run Black-Litterman ───────────────────────────────────
    tau = config.bl_tau if config else 0.05
    risk_aversion = (
        config.bl_risk_aversion if config else None
    )
    lookback = (
        config.bl_lookback_days if config else LOOKBACK_DAYS
    )
    max_pos_pct = (
        config.risk.max_position_pct if config else 15.0
    )
    llm_weight = config.llm_weight if config else 0.3
    regime_sensitive = (
        config.bl_regime_sensitive if config else True
    )
    max_sector_pct = (
        config.bl_max_sector_pct if config else 0.40
    )

    bl_result = run_black_litterman(
        signals=signals,
        held_positions=held_positions,
        portfolio_equity=portfolio_equity,
        llm_views=llm_views if llm_views else None,
        tau=tau,
        risk_aversion=risk_aversion,
        max_position_pct=max_pos_pct,
        lookback_days=lookback,
        llm_weight=llm_weight,
        regime_sensitive=regime_sensitive,
        max_sector_pct=max_sector_pct,
    )

    if bl_result is None:
        return None

    # ── Convert BL weights to ranked intents ──────────────────
    # BL weights are used for RANKING (which stocks to buy first)
    # but the quant bot's suggested notional is the position size
    # floor. BL can scale up to its optimal weight, but never
    # below the original signal's suggested size.
    ranked: list[RankedIntent] = []

    for intent in intents:
        ticker = intent.ticker
        bl_weight = bl_result.optimal_weights.get(ticker, 0.0)
        bl_return = bl_result.posterior_returns.get(ticker, 0.0)
        eq_return = bl_result.equilibrium_returns.get(ticker, 0.0)

        # Use the LARGER of BL's optimal weight or the quant
        # bot's original suggested size (from Kelly sizing).
        # BL spreads weights thinly across many stocks; the
        # quant bot's Kelly size is a better per-position floor.
        bl_notional = bl_weight * portfolio_equity
        original_notional = intent.notional  # from Kelly sizing
        final_notional = max(bl_notional, original_notional)
        intent.notional = round(final_notional, 2)
        intent.reason = (
            f"{intent.reason} "
            f"[BL: weight={bl_weight:.1%}, "
            f"return={bl_return:+.1%}]"
        )

        ranked.append(RankedIntent(
            intent=intent,
            bl_weight=round(bl_weight, 6),
            bl_posterior_return=round(bl_return, 4),
            marginal_sharpe=round(bl_return - eq_return, 4),
            diversification_score=round(
                max(0, 1 - bl_weight * 10), 3
            ),
            optimization_method="black_litterman",
        ))

    # Sort by BL weight (highest allocation first)
    ranked.sort(key=lambda r: -r.bl_weight)

    # Assign ranks and log
    for i, r in enumerate(ranked):
        r.rank = i + 1

    log.info("  BL-optimized ranking (top 15):")
    log.info(
        f"    {'Rank':>4} {'Ticker':<7} {'BLWeight':>8} "
        f"{'PostRet':>8} {'Notional':>10} "
        f"{'Composite':>9}"
    )
    for r in ranked[:15]:
        if r.bl_weight > 0.001:
            log.info(
                f"    {r.rank:>4} {r.intent.ticker:<7} "
                f"{r.bl_weight:>7.1%} "
                f"{r.bl_posterior_return:>+7.1%} "
                f"${r.intent.notional:>9,.2f} "
                f"{r.intent.signal.composite_score:>9.1f}"
            )

    log.info(
        f"  BL portfolio: Sharpe={bl_result.portfolio_sharpe:.2f}, "
        f"Return={bl_result.portfolio_return:.1%}, "
        f"Vol={bl_result.portfolio_volatility:.1%}"
    )

    return ranked


# ── Marginal Sharpe Fallback ──────────────────────────────────────

def fetch_returns_matrix(
    tickers: list[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Fetch daily returns for a list of tickers."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning(
            "yfinance not installed — skipping optimization"
        )
        return pd.DataFrame()

    if not tickers:
        return pd.DataFrame()

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
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
        else:
            prices = data[["Close"]].rename(
                columns={"Close": tickers[0]}
            )

        returns = prices.pct_change().dropna()
        returns = returns.tail(lookback_days)
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
    """Compute annualized Sharpe ratio for a portfolio."""
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )
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
    Rank BUY intents by marginal Sharpe contribution (fallback).

    Used when Black-Litterman optimization is disabled or fails.
    """
    if not intents:
        return []

    candidate_tickers = [i.ticker for i in intents]
    held_tickers = list(held_positions.keys())
    all_tickers = list(set(candidate_tickers + held_tickers))

    log.info(
        f"  Marginal Sharpe: {len(candidate_tickers)} candidates, "
        f"{len(held_tickers)} held positions"
    )

    returns = fetch_returns_matrix(all_tickers)
    if returns.empty or len(returns.columns) < 2:
        log.info(
            "  Not enough data — falling back to composite score"
        )
        return [
            RankedIntent(intent=i, rank=idx + 1)
            for idx, i in enumerate(intents)
        ]

    # Use Ledoit-Wolf if available, else sample cov
    try:
        from trading_bot_bl.black_litterman import (
            ledoit_wolf_shrinkage,
        )
        cov_matrix = ledoit_wolf_shrinkage(returns.values)
    except ImportError:
        cov_matrix = returns.cov().values

    mean_returns = returns.mean().values
    ticker_list = list(returns.columns)
    ticker_idx = {t: i for i, t in enumerate(ticker_list)}
    n = len(ticker_list)

    # Build current portfolio weights
    current_weights = np.zeros(n)
    for ticker, pos in held_positions.items():
        if ticker in ticker_idx:
            mv = abs(pos.get("market_value", 0.0))
            if portfolio_equity > 0:
                current_weights[ticker_idx[ticker]] = (
                    mv / portfolio_equity
                )

    current_sharpe = (
        compute_portfolio_sharpe(
            current_weights, mean_returns, cov_matrix
        )
        if current_weights.sum() > 0 else 0.0
    )

    log.info(f"  Current portfolio Sharpe: {current_sharpe:.2f}")

    # Evaluate each candidate
    ranked: list[RankedIntent] = []

    for intent in intents:
        ticker = intent.ticker
        if ticker not in ticker_idx:
            ranked.append(RankedIntent(
                intent=intent,
                marginal_sharpe=0.0,
                diversification_score=0.5,
                optimization_method="marginal_sharpe",
            ))
            continue

        idx = ticker_idx[ticker]

        # Individual Sharpe
        std = returns[ticker].std()
        ind_sharpe = (
            float(
                (returns[ticker].mean() - RISK_FREE_RATE)
                / std * ANNUALIZATION_FACTOR
            )
            if std > 0 else 0.0
        )

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

        # Hypothetical portfolio with new stock
        proposed_weight = (
            intent.notional / portfolio_equity
            if portfolio_equity > 0 else 0.0
        )
        new_weights = current_weights.copy()
        new_weights[idx] += proposed_weight
        total = new_weights.sum()
        if total > 1.0:
            new_weights = new_weights / total

        new_sharpe = compute_portfolio_sharpe(
            new_weights, mean_returns, cov_matrix
        )
        marginal = new_sharpe - current_sharpe
        div_score = max(0.0, min(1.0, (1.0 - corr) / 2.0))

        ranked.append(RankedIntent(
            intent=intent,
            marginal_sharpe=round(marginal, 4),
            correlation_to_portfolio=round(corr, 3),
            individual_sharpe=round(ind_sharpe, 2),
            diversification_score=round(div_score, 3),
            optimization_method="marginal_sharpe",
        ))

    # Sort by marginal Sharpe
    ranked.sort(
        key=lambda r: (
            -r.marginal_sharpe,
            -r.diversification_score,
            -r.intent.signal.composite_score,
        )
    )

    for i, r in enumerate(ranked):
        r.rank = i + 1

    # Log top candidates
    log.info("  Marginal Sharpe ranking (top 15):")
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


# ── Unified Entry Point ───────────────────────────────────────────

def optimize_intents(
    intents: list,
    held_positions: dict,
    portfolio_equity: float,
    config=None,
) -> list[RankedIntent]:
    """
    Optimize BUY intents using the best available method.

    Tries Black-Litterman first (if enabled), falls back to
    marginal Sharpe ranking.

    Args:
        intents: List of OrderIntent objects (BUY signals).
        held_positions: Portfolio positions dict.
        portfolio_equity: Total account equity.
        config: TradingConfig with optimization parameters.

    Returns:
        List of RankedIntent, sorted best-first.
    """
    if not intents:
        return []

    use_bl = (
        config and config.use_black_litterman
        and len(intents) >= 2
    )

    if use_bl:
        log.info(
            "  Portfolio optimization: Black-Litterman"
            + (" + LLM views" if config.llm_views_enabled else "")
        )
        ranked = optimize_with_black_litterman(
            intents, held_positions, portfolio_equity, config
        )
        if ranked is not None:
            return ranked
        log.info(
            "  BL optimization failed — falling back to "
            "marginal Sharpe"
        )

    if len(intents) >= 2:
        return rank_intents_by_marginal_sharpe(
            intents, held_positions, portfolio_equity
        )

    # Single intent — no optimization needed
    return [
        RankedIntent(intent=intents[0], rank=1)
    ]
