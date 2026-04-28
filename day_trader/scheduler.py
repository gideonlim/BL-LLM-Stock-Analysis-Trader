"""Calendar-driven event scheduler.

All scheduled events are defined as offsets from the NYSE session
open or close, resolved through ``day_trader/calendar.py``. Nothing
is hard-coded to 09:30 or 16:00 — half-days, holidays, and DST are
handled automatically.

The scheduler owns a list of named events with their fire-times.
The executor polls :meth:`due_events` each loop tick (~1 s) and
dispatches whatever's due.

Events are one-shot per session: once fired, they're marked done
and won't re-fire until :meth:`reset_for_session` is called.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from day_trader.calendar import NyseSession

log = logging.getLogger(__name__)


@dataclass
class ScheduledEvent:
    """A named event with its absolute ET fire-time."""

    name: str
    fire_at: datetime         # tz-aware ET
    fired: bool = False
    repeating: bool = False   # if True, fire every ``interval``
    interval: Optional[timedelta] = None  # only for repeating
    _next_fire: Optional[datetime] = field(
        default=None, repr=False,
    )

    def is_due(self, now_et: datetime) -> bool:
        if self.fired and not self.repeating:
            return False
        target = self._next_fire or self.fire_at
        return now_et >= target

    def mark_fired(self, now_et: datetime) -> None:
        if self.repeating and self.interval:
            self._next_fire = (self._next_fire or self.fire_at) + self.interval
        else:
            self.fired = True


def build_session_schedule(
    session: NyseSession,
    *,
    catalyst_refresh_min_before_open: int = 90,
    premarket_scan_min_before_open: int = 60,
    regime_snapshot_min_before_open: int = 5,
    first_scan_min_after_open: int = 5,
    scan_interval_seconds: int = 60,
    exit_only_min_before_close: int = 15,
    force_flat_min_before_close: int = 5,
) -> list[ScheduledEvent]:
    """Build the full event schedule for one session.

    Returns events in chronological order. The executor iterates
    :meth:`due_events` each tick; each event fires at most once
    (except ``scan_tick`` which repeats every ``scan_interval_seconds``
    until ``exit_only``).
    """
    events: list[ScheduledEvent] = []

    # ── Pre-open events ──────────────────────────────────────────
    events.append(ScheduledEvent(
        name="catalyst_refresh",
        fire_at=session.open_minus(catalyst_refresh_min_before_open),
    ))
    events.append(ScheduledEvent(
        name="premarket_scan",
        fire_at=session.open_minus(premarket_scan_min_before_open),
    ))
    events.append(ScheduledEvent(
        name="recovery_reconcile",
        fire_at=session.open_minus(premarket_scan_min_before_open),
    ))
    events.append(ScheduledEvent(
        name="regime_snapshot",
        fire_at=session.open_minus(regime_snapshot_min_before_open),
    ))

    # ── Market-hours events ──────────────────────────────────────
    events.append(ScheduledEvent(
        name="market_open",
        fire_at=session.open_et,
    ))
    events.append(ScheduledEvent(
        name="first_scan",
        fire_at=session.open_plus(first_scan_min_after_open),
    ))
    # Repeating scan tick from first_scan until exit_only
    events.append(ScheduledEvent(
        name="scan_tick",
        fire_at=session.open_plus(first_scan_min_after_open + 1),
        repeating=True,
        interval=timedelta(seconds=scan_interval_seconds),
    ))

    # ── End-of-day events ────────────────────────────────────────
    events.append(ScheduledEvent(
        name="exit_only",
        fire_at=session.close_minus(exit_only_min_before_close),
    ))
    events.append(ScheduledEvent(
        name="force_flat",
        fire_at=session.close_minus(force_flat_min_before_close),
    ))
    events.append(ScheduledEvent(
        name="session_close",
        fire_at=session.close_et,
    ))

    events.sort(key=lambda e: e.fire_at)
    return events


class Scheduler:
    """Stateful wrapper around a session's event schedule."""

    def __init__(self, events: list[ScheduledEvent]) -> None:
        self._events = list(events)

    @classmethod
    def for_session(cls, session: NyseSession, **kwargs) -> "Scheduler":
        """Build the default schedule for a session."""
        return cls(build_session_schedule(session, **kwargs))

    def due_events(self, now_et: datetime) -> list[ScheduledEvent]:
        """Return all events that are due at ``now_et``.

        Events are returned in schedule order. The executor processes
        them sequentially — ``force_flat`` before ``session_close``,
        etc. Each event is marked fired after return.
        """
        due: list[ScheduledEvent] = []
        for event in self._events:
            if event.is_due(now_et):
                due.append(event)
                event.mark_fired(now_et)
        return due

    def reset_for_session(self) -> None:
        """Unfire all events. Called if the daemon stays up across
        sessions (rare — usually we restart)."""
        for e in self._events:
            e.fired = False
            e._next_fire = None

    def all_events(self) -> list[ScheduledEvent]:
        """For logging / diagnostics — the full schedule."""
        return list(self._events)

    def next_event_at(self, now_et: datetime) -> Optional[datetime]:
        """When is the next unfired event? Used by the executor to
        decide sleep duration — no point polling at 1 Hz when the
        next event is 30 min away."""
        candidates: list[datetime] = []
        for e in self._events:
            if e.fired and not e.repeating:
                continue
            target = e._next_fire or e.fire_at
            if target > now_et:
                candidates.append(target)
        return min(candidates) if candidates else None
