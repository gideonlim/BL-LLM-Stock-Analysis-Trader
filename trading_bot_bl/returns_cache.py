"""Shared yfinance return-data fetcher with in-session caching.

Both black_litterman.py and portfolio_optimizer.py need the same
daily-return matrix from yfinance.  This module downloads once per
(ticker-set, lookback) pair and serves subsequent requests from an
in-memory cache, eliminating duplicate HTTP calls within a single
execution run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger(__name__)

# ── In-memory cache keyed by (frozenset(tickers), lookback_days) ────
_cache: dict[tuple[frozenset[str], int], pd.DataFrame] = {}


def fetch_returns(
    tickers: list[str],
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Fetch daily returns for *tickers* over *lookback_days*.

    Returns an empty DataFrame on failure (missing yfinance,
    empty download, etc.).  Results are cached for the lifetime
    of the process so that a second call with the same arguments
    is free.
    """
    if not tickers:
        return pd.DataFrame()

    key = (frozenset(tickers), lookback_days)
    if key in _cache:
        return _cache[key]

    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not installed — cannot fetch returns")
        return pd.DataFrame()

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

        # Drop tickers with insufficient data (< 80 % of rows)
        min_data = int(len(returns) * 0.8)
        returns = returns.dropna(axis=1, thresh=min_data)

        _cache[key] = returns
        return returns

    except Exception as e:
        log.warning(f"Could not fetch returns: {e}")
        return pd.DataFrame()


def clear_cache() -> None:
    """Reset the in-memory cache (useful in tests)."""
    _cache.clear()
