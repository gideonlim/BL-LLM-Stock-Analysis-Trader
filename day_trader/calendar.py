"""NYSE session-aware time math.

All scheduled events in the day-trader (premarket scan, first ORB
scan, exit-only mode, force-flat, etc.) resolve through this module
rather than against hard-coded wall-clock times. That way:

- Holidays are skipped
- Half-days (early closes at 13:00 ET) shift the EOD windows
- DST transitions are handled by ``zoneinfo``
- Unscheduled halts can be modelled by overriding ``session_for``

Wraps ``pandas_market_calendars`` and exposes a tiny stable surface
to the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Standard (non-half-day) NYSE session is 6h30m wide. Anything
# significantly shorter is treated as a half day.
_FULL_DAY_HOURS = 6.5
_HALF_DAY_THRESHOLD = timedelta(hours=6, minutes=15)

_NYSE = mcal.get_calendar("XNYS")


@dataclass(frozen=True)
class NyseSession:
    """One trading session on the NYSE."""

    date: date
    open_et: datetime  # tz-aware (ET)
    close_et: datetime  # tz-aware (ET)
    is_half_day: bool

    @property
    def length(self) -> timedelta:
        return self.close_et - self.open_et

    def open_minus(self, minutes: int) -> datetime:
        """ET datetime ``minutes`` before the open."""
        return self.open_et - timedelta(minutes=minutes)

    def open_plus(self, minutes: int) -> datetime:
        """ET datetime ``minutes`` after the open."""
        return self.open_et + timedelta(minutes=minutes)

    def close_minus(self, minutes: int) -> datetime:
        """ET datetime ``minutes`` before the close (handles half days)."""
        return self.close_et - timedelta(minutes=minutes)

    def contains(self, et_dt: datetime) -> bool:
        """Is ``et_dt`` within ``[open, close)``?"""
        if et_dt.tzinfo is None:
            raise ValueError("contains() requires a tz-aware datetime")
        return self.open_et <= et_dt.astimezone(ET) < self.close_et


def now_et() -> datetime:
    """Current wall-clock time in ET, tz-aware."""
    return datetime.now(tz=ET)


def session_for(target: Optional[date] = None) -> Optional[NyseSession]:
    """Return the NYSE session for ``target`` (default: today in ET).

    Returns ``None`` when ``target`` is a weekend, holiday, or
    otherwise has no scheduled session.
    """
    d = target or now_et().date()
    return _session_cached(d)


@lru_cache(maxsize=512)
def _session_cached(d: date) -> Optional[NyseSession]:
    """Cached lookup keyed by date (immutable)."""
    schedule = _NYSE.schedule(start_date=d, end_date=d)
    if schedule.empty:
        return None
    row = schedule.iloc[0]

    # mcal returns timezone-aware UTC timestamps as pandas Timestamps.
    open_ts = row["market_open"]
    close_ts = row["market_close"]

    open_dt = open_ts.to_pydatetime()
    close_dt = close_ts.to_pydatetime()

    # Ensure tz-aware: pandas timestamps from mcal *are* tz-aware,
    # but be defensive (older versions can vary).
    if open_dt.tzinfo is None:
        open_dt = open_dt.replace(tzinfo=UTC)
    if close_dt.tzinfo is None:
        close_dt = close_dt.replace(tzinfo=UTC)

    open_et = open_dt.astimezone(ET)
    close_et = close_dt.astimezone(ET)
    is_half = (close_et - open_et) < _HALF_DAY_THRESHOLD

    return NyseSession(
        date=d, open_et=open_et, close_et=close_et, is_half_day=is_half,
    )


def is_market_open(at: Optional[datetime] = None) -> bool:
    """Is the regular-hours session active at ``at`` (default: now)?"""
    at = at or now_et()
    if at.tzinfo is None:
        raise ValueError("is_market_open() requires a tz-aware datetime")
    et = at.astimezone(ET)
    sess = session_for(et.date())
    return sess is not None and sess.contains(et)


def time_until_close(at: Optional[datetime] = None) -> Optional[timedelta]:
    """Time remaining until today's close, or ``None`` if not in-session."""
    at = at or now_et()
    if at.tzinfo is None:
        raise ValueError("time_until_close() requires a tz-aware datetime")
    et = at.astimezone(ET)
    sess = session_for(et.date())
    if sess is None or et >= sess.close_et:
        return None
    return sess.close_et - et


def time_since_open(at: Optional[datetime] = None) -> Optional[timedelta]:
    """Elapsed time in today's session, or ``None`` if not in-session."""
    at = at or now_et()
    if at.tzinfo is None:
        raise ValueError("time_since_open() requires a tz-aware datetime")
    et = at.astimezone(ET)
    sess = session_for(et.date())
    if sess is None or et < sess.open_et:
        return None
    return et - sess.open_et


def next_session(after: Optional[date] = None) -> Optional[NyseSession]:
    """Return the next NYSE session strictly after ``after`` (default: today).

    Looks up to 14 calendar days forward; returns ``None`` if no
    session is scheduled in that window (which would be unusual).
    """
    start = after or now_et().date()
    schedule = _NYSE.schedule(
        start_date=start + timedelta(days=1),
        end_date=start + timedelta(days=14),
    )
    if schedule.empty:
        return None
    first = schedule.iloc[0]
    next_date = first.name.to_pydatetime().date() if hasattr(
        first.name, "to_pydatetime"
    ) else first.name
    return session_for(next_date)


def is_within_eod_flatten_window(
    at: Optional[datetime] = None,
    *,
    minutes_before_close: int = 5,
) -> bool:
    """True iff we're inside the EOD-flatten window for the current session.

    The standalone ``deploy/scripts/eod_flatten.py`` watchdog calls
    this every minute and only acts when it returns True. systemd's
    ``OnCalendar`` is static (doesn't follow half-days), so we rely
    on this gate to handle the early-close case.
    """
    at = at or now_et()
    if at.tzinfo is None:
        raise ValueError("requires tz-aware datetime")
    et = at.astimezone(ET)
    sess = session_for(et.date())
    if sess is None:
        return False
    window_start = sess.close_minus(minutes_before_close)
    return window_start <= et < sess.close_et
