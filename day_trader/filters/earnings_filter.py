"""EarningsFilter — earnings blackout for non-catalyst strategies.

Wraps the existing ``trading_bot_bl/earnings.py`` blackout logic so
the day-trader doesn't open positions in tickers about to report.

Bypassed by ``catalyst_momentum`` (when v2.5 ships) — that strategy
specifically wants to trade catalyst-driven moves.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, Optional

from trading_bot_bl.earnings import check_earnings_blackout

from day_trader.filters.base import Filter
from day_trader.models import FilterContext

log = logging.getLogger(__name__)


# Strategies that LEGITIMATELY want to trade through earnings.
# When their signals come through this filter, the filter passes
# them. Anything else is blocked.
_CATALYST_STRATEGIES = frozenset({
    "catalyst_momentum",
})


class EarningsFilter(Filter):
    """Reject signals on tickers in the earnings blackout window.

    Blackout is currently 3 calendar days before earnings and 1 day
    after, sourced from yfinance via ``trading_bot_bl/earnings.py``.
    """

    name = "earnings"

    def __init__(
        self,
        *,
        pre_days: int = 3,
        post_days: int = 1,
        catalyst_strategies: Iterable[str] = _CATALYST_STRATEGIES,
        today: Optional[date] = None,
    ):
        self.pre_days = pre_days
        self.post_days = post_days
        self.catalyst_strategies = frozenset(catalyst_strategies)
        self._today = today  # for tests

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"
        # Catalyst-momentum strategies bypass the earnings blackout
        if ctx.signal.strategy in self.catalyst_strategies:
            return True, ""

        try:
            info = check_earnings_blackout(
                ctx.signal.ticker,
                pre_days=self.pre_days,
                post_days=self.post_days,
                today=self._today or date.today(),
            )
        except Exception as exc:
            # Earnings calendar fetch can fail (yfinance hiccups).
            # Fail open — better to potentially trade through earnings
            # once than to silently halt all entries on a flaky API.
            # The swing bot uses the same convention.
            log.debug(
                "EarningsFilter: lookup failed for %s: %s — passing",
                ctx.signal.ticker, exc,
            )
            return True, ""

        if info.in_blackout:
            # Reasons from EarningsInfo: "earnings_pre_window",
            # "earnings_post_window". Pass through verbatim.
            return False, info.blackout_reason or "earnings_blackout"
        return True, ""
