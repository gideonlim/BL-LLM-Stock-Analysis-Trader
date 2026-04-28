"""SpreadFilter — bid-ask spread gate.

Rejects signals where the current spread is too wide:

- Default 15 bps for stocks priced > $10
- Default 30 bps for stocks priced ≤ $10 (penny-stock-style names)

Spreads are the truest measure of execution friction — a $0.05 spread
on a $100 stock is 5 bps (cheap), but the same $0.05 on a $5 stock
is 100 bps (expensive). Day-trade entries with wide spreads hand
edge to market makers before the strategy even starts.
"""

from __future__ import annotations

from day_trader.config import DayRiskLimits
from day_trader.filters.base import Filter
from day_trader.models import FilterContext


PRICE_TIER_BREAKPOINT = 10.0


class SpreadFilter(Filter):
    """Reject when spread (bps of mid) exceeds the price-tier threshold."""

    name = "spread"

    def __init__(self, limits: DayRiskLimits):
        self.limits = limits

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"
        if ctx.quote is None:
            return False, "no_quote"

        mid = ctx.quote.mid
        if mid <= 0:
            return False, "invalid_quote"

        spread_bps = ctx.quote.spread_bps
        threshold = (
            self.limits.max_spread_bps_above_10
            if mid > PRICE_TIER_BREAKPOINT
            else self.limits.max_spread_bps_at_or_under_10
        )

        if spread_bps > threshold:
            return False, "spread_too_wide"
        return True, ""
