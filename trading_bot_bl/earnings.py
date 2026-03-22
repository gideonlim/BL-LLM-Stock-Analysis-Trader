"""
Earnings blackout filter — blocks new entries near earnings dates.

Earnings announcements cause 5-10× normal daily volatility. ATR-based
stop losses are calibrated for normal moves and are useless against
overnight gaps. This module fetches upcoming earnings dates from
yfinance and provides a simple check: "is this ticker within N days
of an earnings announcement?"

Data source: yfinance .calendar property (free, no API key needed).
Returns the next earnings date from Yahoo Finance consensus estimates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional

log = logging.getLogger(__name__)

# Module-level cache: {ticker: EarningsInfo} — survives the full
# execution run so we only hit Yahoo once per ticker per session.
_earnings_cache: Dict[str, "EarningsInfo"] = {}


@dataclass(frozen=True)
class EarningsInfo:
    """Earnings date information for a single ticker."""

    ticker: str
    next_earnings_date: Optional[date] = None
    days_until_earnings: Optional[int] = None
    in_blackout: bool = False
    blackout_reason: str = ""


def fetch_earnings_date(ticker: str) -> Optional[date]:
    """
    Fetch the next earnings date for a ticker via yfinance.

    Returns None if unavailable or on any error (graceful degradation).
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        cal = t.calendar

        if cal is None or not isinstance(cal, dict):
            return None

        earnings_dates = cal.get("Earnings Date")
        if not earnings_dates:
            return None

        # .calendar returns a list of datetime.date objects
        # Usually 1 element, sometimes 2 (range estimate)
        # Take the earliest date
        if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
            earliest = min(earnings_dates)
            if hasattr(earliest, "date"):
                # It's a datetime, convert to date
                return earliest.date()
            if isinstance(earliest, date):
                return earliest

        return None

    except Exception as e:
        log.debug(f"Earnings lookup failed for {ticker}: {e}")
        return None


def check_earnings_blackout(
    ticker: str,
    *,
    pre_days: int = 3,
    post_days: int = 1,
    today: Optional[date] = None,
    use_cache: bool = True,
) -> EarningsInfo:
    """
    Check if a ticker is within the earnings blackout window.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    pre_days : int
        Days before earnings to block entry (default: 3).
    post_days : int
        Days after earnings to block entry (default: 1).
    today : date, optional
        Override for testing. Defaults to today's date.
    use_cache : bool
        Use module-level cache to avoid repeated yfinance calls.

    Returns
    -------
    EarningsInfo with blackout status and reason.
    """
    if today is None:
        today = date.today()

    # Check cache first
    if use_cache and ticker in _earnings_cache:
        cached = _earnings_cache[ticker]
        # Recalculate days_until and blackout relative to today
        # in case the function is called on a different day
        if cached.next_earnings_date is not None:
            return _evaluate_blackout(
                ticker, cached.next_earnings_date,
                today, pre_days, post_days,
            )
        return cached

    # Fetch from yfinance
    earnings_date = fetch_earnings_date(ticker)

    if earnings_date is None:
        info = EarningsInfo(ticker=ticker)
        if use_cache:
            _earnings_cache[ticker] = info
        return info

    # Cache the raw date, evaluate blackout
    if use_cache:
        _earnings_cache[ticker] = EarningsInfo(
            ticker=ticker,
            next_earnings_date=earnings_date,
        )

    return _evaluate_blackout(
        ticker, earnings_date, today, pre_days, post_days,
    )


def _evaluate_blackout(
    ticker: str,
    earnings_date: date,
    today: date,
    pre_days: int,
    post_days: int,
) -> EarningsInfo:
    """Evaluate whether today falls in the blackout window."""
    days_until = (earnings_date - today).days

    # Blackout window: [earnings - pre_days, earnings + post_days]
    blackout_start = earnings_date - timedelta(days=pre_days)
    blackout_end = earnings_date + timedelta(days=post_days)

    in_blackout = blackout_start <= today <= blackout_end
    reason = ""

    if in_blackout:
        if days_until > 0:
            reason = (
                f"Earnings in {days_until} day(s) "
                f"({earnings_date.isoformat()}) — "
                f"blackout {pre_days}d pre / {post_days}d post"
            )
        elif days_until == 0:
            reason = (
                f"Earnings TODAY ({earnings_date.isoformat()}) — "
                f"blackout active"
            )
        else:
            reason = (
                f"Earnings was {abs(days_until)} day(s) ago "
                f"({earnings_date.isoformat()}) — "
                f"post-earnings blackout"
            )

    return EarningsInfo(
        ticker=ticker,
        next_earnings_date=earnings_date,
        days_until_earnings=days_until,
        in_blackout=in_blackout,
        blackout_reason=reason,
    )


def batch_check_earnings(
    tickers: list[str],
    *,
    pre_days: int = 3,
    post_days: int = 1,
) -> Dict[str, EarningsInfo]:
    """
    Check earnings blackout for multiple tickers.

    Returns {ticker: EarningsInfo} for all tickers.
    Tickers with no earnings data are included with in_blackout=False.
    """
    results = {}
    for ticker in tickers:
        results[ticker] = check_earnings_blackout(
            ticker, pre_days=pre_days, post_days=post_days,
        )
    return results


def clear_cache() -> None:
    """Clear the module-level earnings cache."""
    _earnings_cache.clear()
