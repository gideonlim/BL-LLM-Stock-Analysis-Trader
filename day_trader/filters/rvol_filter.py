"""RvolFilter — relative volume gate.

The strategy emits signals on tickers with elevated activity (the
"Stocks in Play" framing). Without an RVOL gate, the same strategy
signals can fire on dead-quiet days when the edge isn't really there.

Two checks:

- **Premarket RVOL**: signal.rvol vs trailing-30-day-average premarket
  volume. Strategy is responsible for populating ``signal.rvol`` with
  the premarket value at scan time.
- **Intraday last-5-min RVOL**: derived from ``ctx.bars`` — last 5
  bars' aggregate volume vs trailing-30-day session-time-of-day
  average. Approximated below from the cached bars.

Both checks must pass.
"""

from __future__ import annotations

from day_trader.config import DayRiskLimits
from day_trader.filters.base import Filter
from day_trader.models import FilterContext


class RvolFilter(Filter):
    """Reject signals without sufficient relative volume."""

    name = "rvol"

    def __init__(self, limits: DayRiskLimits):
        self.limits = limits

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"

        # ── Premarket RVOL ─────────────────────────────────────────
        # The strategy populates this from the premarket scanner.
        # 0 means "not available / not yet computed" — fail closed
        # before market opens, fail open after (signal.rvol is
        # then expected to be intraday RVOL).
        premkt_rvol = ctx.signal.rvol or 0.0
        if premkt_rvol <= 0:
            return False, "rvol_unknown"

        if premkt_rvol < self.limits.min_premkt_rvol:
            return False, "premkt_rvol_too_low"

        # ── Intraday RVOL approximation ────────────────────────────
        # If we have at least 5 minute bars in the cache, compute the
        # last-5-min RVOL. The daily volume average for the last-5
        # window isn't trivially available without a full historical
        # cache; we approximate by checking that recent volume is
        # not collapsing relative to early-session volume.
        bars = ctx.bars or []
        if len(bars) >= 10:
            recent = sum((b.volume or 0) for b in bars[-5:]) / 5.0
            earlier = sum((b.volume or 0) for b in bars[-10:-5]) / 5.0
            if earlier > 0:
                ratio = recent / earlier
                if ratio < self.limits.min_intraday_rvol / 2.0:
                    # Recent volume is collapsing — likely a dead
                    # setup. Threshold is permissive (half the
                    # premkt threshold) because intraday RVOL is
                    # relative to the same session, not 30 days.
                    return False, "intraday_rvol_collapsing"

        return True, ""
