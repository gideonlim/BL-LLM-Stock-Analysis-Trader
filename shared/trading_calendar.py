"""Market session helpers using pandas_market_calendars.

Provides ``is_session_open()`` — the single source of truth for
"is this market open right now?" — used by:
  - ``quant_analysis_bot/prefetch.py`` (data layer, all markets)
  - ``IBKRBroker.is_market_open()`` (non-US live execution)

``AlpacaBroker.is_market_open()`` continues to use Alpaca's own
clock endpoint for US live execution — unchanged.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from trading_bot_bl.market_config import MarketConfig

log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_calendar(calendar_name: str):
    """Lazily import and cache a pandas_market_calendars calendar."""
    import pandas_market_calendars as mcal
    return mcal.get_calendar(calendar_name)


def is_session_open(
    market: "MarketConfig",
    now: datetime | None = None,
) -> bool:
    """Check whether *market* is in regular session at *now*.

    Uses ``pandas_market_calendars`` for authoritative session
    open/close times (including early closes around holidays).
    Handles TSE's lunch break via ``market.lunch_break``.

    Parameters
    ----------
    market : MarketConfig
        The market configuration to check.
    now : datetime, optional
        The moment to check.  Defaults to ``datetime.now(UTC)``.

    Returns
    -------
    bool
        True if the market is in a regular trading session.
    """
    try:
        cal = _get_calendar(market.holiday_calendar_name)
    except Exception as exc:
        log.error(
            f"Calendar '{market.holiday_calendar_name}' unavailable: "
            f"{exc}. Failing closed (returning False) for safety."
        )
        return False

    tz = ZoneInfo(market.timezone)
    local_now = (
        now or datetime.now(ZoneInfo("UTC"))
    ).astimezone(tz)

    today = local_now.date()

    try:
        sched = cal.schedule(
            start_date=str(today),
            end_date=str(today),
        )
    except Exception:
        return False

    if sched.empty:
        return False  # holiday or weekend

    # Extract session open/close in market-local time
    open_ts = sched.iloc[0]["market_open"]
    close_ts = sched.iloc[0]["market_close"]

    # Convert to local timezone for comparison
    open_local = open_ts.tz_convert(tz).time()
    close_local = close_ts.tz_convert(tz).time()

    t = local_now.time()

    if not (open_local <= t < close_local):
        return False

    # Handle split-session lunch break (e.g. TSE 11:30-12:30)
    if market.lunch_break is not None:
        lunch_start, lunch_end = market.lunch_break
        if lunch_start <= t < lunch_end:
            return False

    return True


def is_trading_day(
    market: "MarketConfig",
    date=None,
) -> bool:
    """Check whether *date* is a trading day for *market*.

    Parameters
    ----------
    market : MarketConfig
        The market configuration to check.
    date : date-like, optional
        Defaults to today in the market's timezone.

    Returns
    -------
    bool
    """
    from datetime import date as date_type

    try:
        cal = _get_calendar(market.holiday_calendar_name)
    except Exception as exc:
        log.error(
            f"Calendar '{market.holiday_calendar_name}' unavailable: "
            f"{exc}. Failing closed (returning False) for safety."
        )
        return False

    if date is None:
        tz = ZoneInfo(market.timezone)
        date = datetime.now(tz).date()

    if isinstance(date, datetime):
        date = date.date()

    try:
        sched = cal.schedule(
            start_date=str(date),
            end_date=str(date),
        )
        return not sched.empty
    except Exception as exc:
        log.error(
            f"Calendar schedule lookup failed for {date}: {exc}. "
            f"Failing closed (returning False) for safety."
        )
        return False


def _fallback_time_check(
    market: "MarketConfig",
    now: datetime | None = None,
) -> bool:
    """Simple weekday + time-of-day fallback (no holiday awareness)."""
    tz = ZoneInfo(market.timezone)
    local_now = (
        now or datetime.now(ZoneInfo("UTC"))
    ).astimezone(tz)

    if local_now.weekday() >= 5:
        return False

    t = local_now.time()
    if not (market.market_open <= t < market.market_close):
        return False

    if market.lunch_break is not None:
        lunch_start, lunch_end = market.lunch_break
        if lunch_start <= t < lunch_end:
            return False

    return True
