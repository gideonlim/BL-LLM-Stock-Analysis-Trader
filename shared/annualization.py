"""Annualization helpers — single source of truth for trading-day math.

Replaces every hardcoded ``252`` and ``0.05 / 252`` across the codebase.
Both ``trading_bot_bl`` and ``quant_analysis_bot`` import from here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_bot_bl.market_config import MarketConfig


def ann_factor(market: MarketConfig) -> float:
    """√(trading_days_per_year) — used to annualize daily Sharpe/Sortino."""
    return math.sqrt(market.trading_days_per_year)


def daily_risk_free(market: MarketConfig) -> float:
    """Daily risk-free rate derived from the annual rate."""
    return market.risk_free_rate / market.trading_days_per_year


def ann_return(
    total_return: float,
    n_days: int,
    market: MarketConfig,
) -> float:
    """Annualize a cumulative return over *n_days* trading days."""
    if n_days <= 0:
        return 0.0
    return (1 + total_return) ** (
        market.trading_days_per_year / n_days
    ) - 1


def ann_volatility(
    daily_std: float,
    market: MarketConfig,
) -> float:
    """Annualize a daily standard deviation."""
    return daily_std * math.sqrt(market.trading_days_per_year)
