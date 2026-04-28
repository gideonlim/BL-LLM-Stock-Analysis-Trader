"""CatalystFilter — strategy-specific catalyst-label gating.

Each strategy declares whether it wants tickers with NEWS_HIGH /
NEWS_LOW / NO_NEWS catalysts. The classifier sets the label at
premarket; this filter just gates entries based on the strategy's
declared requirement.

Examples:

- ``vwap_reversion`` requires ``NO_NEWS`` (mean reversion thesis
  fails when news is moving the stock for a real reason).
- ``catalyst_momentum`` requires ``NEWS_HIGH`` (the whole point).
- ``orb_vwap``, ``vwap_pullback`` accept any catalyst — they have
  ``required_catalyst_label = None``.
"""

from __future__ import annotations

from typing import Mapping, Optional

from day_trader.filters.base import Filter
from day_trader.models import FilterContext


# Each strategy declares its required catalyst. None means "any".
DEFAULT_STRATEGY_REQUIREMENTS: Mapping[str, Optional[str]] = {
    "orb_vwap": None,
    "vwap_pullback": None,
    "vwap_reversion": "NO_NEWS",
    "catalyst_momentum": "NEWS_HIGH",
}


class CatalystFilter(Filter):
    """Apply per-strategy catalyst-label requirements."""

    name = "catalyst"

    def __init__(
        self,
        *,
        strategy_requirements: Mapping[str, Optional[str]] = (
            DEFAULT_STRATEGY_REQUIREMENTS
        ),
    ):
        self.strategy_requirements = dict(strategy_requirements)

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"

        required = self.strategy_requirements.get(ctx.signal.strategy)
        if required is None:
            # Strategy doesn't care about catalyst label
            return True, ""

        actual = (ctx.signal.catalyst_label or "").upper()
        if not actual:
            return False, "catalyst_label_unknown"

        if actual != required:
            return False, f"catalyst_mismatch_want_{required}_got_{actual}"

        return True, ""
