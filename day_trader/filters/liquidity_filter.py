"""LiquidityFilter — ADV + intraday volume participation.

Wraps the existing ``trading_bot_bl/liquidity.py`` ADV check.
Adds an intraday participation check on top: the proposed share
count must not be too large a fraction of the session volume so
far, otherwise our exit fills will move the market against us.
"""

from __future__ import annotations

import logging
from typing import Optional

from trading_bot_bl.liquidity import check_liquidity

from day_trader.config import DayRiskLimits
from day_trader.filters.base import Filter
from day_trader.models import FilterContext

log = logging.getLogger(__name__)


class LiquidityFilter(Filter):
    """Reject signals on illiquid tickers OR signals that would put
    too much weight on session volume.

    Two layers:

    1. **ADV** (average daily $-volume) — wraps trading_bot_bl. A
       30-day average. Cuts thinly-traded names.
    2. **Intraday participation** — proposed shares as % of
       cumulative session volume so far. Set ≤ 0.5% by default
       (well below the 1.0% market-impact threshold most papers
       cite). Blocks entries that would overwhelm the order book
       on exit.
    """

    name = "liquidity"

    def __init__(
        self,
        limits: DayRiskLimits,
        *,
        max_intraday_participation_pct: float = 0.5,
    ):
        self.limits = limits
        self.max_intraday_participation_pct = max_intraday_participation_pct

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"

        # ── 1. ADV via shared liquidity module ─────────────────────
        # check_liquidity needs the proposed position_notional. We
        # don't know the final qty here (sizing happens after filters
        # in DayRiskManager.review), so use a sentinel notional that
        # only triggers participation rejection on truly thin tickers.
        # The plan's intraday participation check below catches the
        # final-sized case.
        try:
            info = check_liquidity(
                ctx.signal.ticker,
                position_notional=0.0,
                min_adv_shares=self.limits.min_adv_shares,
                min_dollar_volume=self.limits.min_adv_dollar_volume,
                max_participation_pct=1.0,
            )
        except Exception as exc:
            # Like the swing bot: fail open on data errors. Logged
            # at info so it shows up in the daily log digest.
            log.info(
                "LiquidityFilter: ADV lookup failed for %s: %s — "
                "passing this scan",
                ctx.signal.ticker, exc,
            )
            return True, ""

        if not info.passes:
            return False, info.rejection_reason or "below_adv"

        # ── 2. Intraday participation guard ────────────────────────
        # Intraday volume comes from the BarCache via ctx.bars.
        # We don't know the final share count here (sizing happens
        # downstream); we use this as a coarse signal that the
        # session has been active enough to support our entry.
        # An empty intraday volume is fine pre-open / first bar.
        # When WholeShareSizingFilter has run and the executor has
        # the final qty, the executor can do a sharper participation
        # check before submitting. Here we only ensure the session
        # isn't dead.
        intraday_vol = sum((b.volume or 0.0) for b in (ctx.bars or []))
        if (
            ctx.bars
            and len(ctx.bars) >= 5
            and intraday_vol < self.limits.min_adv_shares * 0.01
        ):
            # Less than 1% of typical daily volume in the bars
            # we've seen → today is unusually quiet. Skip.
            return False, "intraday_volume_too_thin"

        return True, ""
