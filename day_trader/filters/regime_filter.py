"""RegimeFilter — global market-state gate.

Halts new entries when:

- VIX is above a hard threshold (default 35)
- SPY is in SEVERE_BEAR per the swing bot's regime cache
- High-vol switch is tripped

Existing positions are NOT touched — this filter only gates new
entries. Position management (stops, trailing) continues normally
under whatever regime.
"""

from __future__ import annotations

from day_trader.config import DayRiskLimits
from day_trader.filters.base import Filter
from day_trader.models import FilterContext


class RegimeFilter(Filter):
    """Reject signals during severe-vol or severe-bear regimes."""

    name = "regime"

    def __init__(self, limits: DayRiskLimits):
        self.limits = limits

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"
        ms = ctx.market_state
        if ms is None:
            # No market state means we don't know — fail open is risky;
            # fail closed is safer. The executor MUST populate market
            # state before scanning; if it didn't, that's a bug.
            return False, "no_market_state"

        if (
            self.limits.halt_above_vix
            and ms.vix > self.limits.halt_above_vix
        ):
            return False, "vix_too_high"

        if self.limits.halt_in_severe_bear and ms.is_severe_bear:
            return False, "spy_severe_bear"

        return True, ""
