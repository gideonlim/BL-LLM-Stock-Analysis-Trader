"""WholeShareSizingFilter — reject when whole-share rounding breaks risk.

Alpaca bracket orders require whole-share quantities (fractional
shares are not allowed with bracket/OCO order class). With small
budgets (e.g. the $5k pilot stage where per-trade risk is $12.50),
high-priced tickers can produce share counts where:

- ``shares = floor(notional / price)`` is too small to size around
  the ATR stop properly, OR
- The minimum 1-share position implies a stop-loss risk that
  exceeds ``per_trade_risk_pct``.

Either way: reject explicitly rather than silently size weird.
"""

from __future__ import annotations

import math

from day_trader.config import DayRiskLimits
from day_trader.filters.base import Filter
from day_trader.models import FilterContext


class WholeShareSizingFilter(Filter):
    """Reject signals where whole-share rounding breaks the risk
    constraint or sizes too small to be meaningful."""

    name = "whole_share_sizing"

    def __init__(
        self,
        limits: DayRiskLimits,
        *,
        equity_at_session_start: float,
        budget_pct: float | None = None,
    ):
        """``equity_at_session_start`` is set from the executor after
        recovery succeeds — used to convert per_trade_risk_pct from
        a percentage into a dollar amount.

        ``budget_pct`` is for unit tests; production passes the
        live value from limits.
        """
        self.limits = limits
        self.equity = equity_at_session_start
        # Track for diagnostics — the actual size cap is in DayRiskManager.
        self.budget_pct = budget_pct or limits.budget_pct

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"

        signal = ctx.signal
        if signal.signal_price <= 0:
            return False, "invalid_signal_price"

        # Risk per share = |entry - stop|
        risk_per_share = abs(
            signal.signal_price - signal.stop_loss_price
        )
        if risk_per_share <= 0:
            return False, "no_stop_distance"

        # Max dollar risk allowed by config
        max_risk_dollars = self.equity * (
            self.limits.per_trade_risk_pct / 100
        )
        if max_risk_dollars <= 0:
            return False, "no_risk_budget"

        # Max whole shares we can buy without exceeding the dollar
        # risk cap = floor(max_risk_dollars / risk_per_share)
        max_shares_by_risk = math.floor(max_risk_dollars / risk_per_share)

        if max_shares_by_risk < self.limits.min_qty:
            # Even one share would exceed the risk budget.
            # E.g. AAPL at $400, ATR stop at $5 below → 1 share = $5
            # risk. With per_trade_risk_pct=0.25% on $5k equity,
            # max_risk_dollars = $12.50 → 1 share fits ($5 risk).
            # But on, say, $1000 equity → max_risk = $2.50 → 0 shares.
            return False, "stop_too_wide_for_risk_budget"

        return True, ""
