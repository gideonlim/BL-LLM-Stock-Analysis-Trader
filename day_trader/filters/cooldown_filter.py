"""CooldownFilter — no-revenge gate.

Reads from the shared :class:`CooldownTracker` populated by
``DayRiskManager.record_close()`` on every losing trade.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from day_trader.filters.base import Filter
from day_trader.filters.cooldown import CooldownTracker
from day_trader.models import FilterContext


class CooldownFilter(Filter):
    """Reject if either ticker or strategy is in a cooldown period."""

    name = "cooldown"

    def __init__(
        self,
        cooldowns: CooldownTracker,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        self.cooldowns = cooldowns
        # Allow tests to inject a fake clock.
        self._clock = clock or datetime.now

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"
        cooled, reason = self.cooldowns.is_cooled_down(
            ticker=ctx.signal.ticker,
            strategy=ctx.signal.strategy,
            now=self._clock(),
        )
        if cooled:
            return False, reason
        return True, ""
