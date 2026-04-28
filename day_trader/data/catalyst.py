"""Catalyst classifier — labels each ticker NEWS_HIGH / NEWS_LOW / NO_NEWS.

Used by ``CatalystFilter`` to gate per-strategy entries:

- ``vwap_reversion`` (v2) requires ``NO_NEWS`` — mean-reversion
  thesis fails when news is moving the stock for a real reason.
- ``catalyst_momentum`` (v2.5) requires ``NEWS_HIGH``.
- ``orb_vwap`` and ``vwap_pullback`` (v1) accept any label.

v1 implementation: classify based on **earnings calendar proximity
only**. The shared ``trading_bot_bl/earnings.py`` already fetches
the next earnings date per ticker via yfinance, with caching. That's
enough signal for the v1 strategies, none of which actually require
NEWS_HIGH (the only strategy that does is catalyst_momentum, which
is parked behind a real-time news vendor decision in v2.5+).

v2 adds news-headline ingestion (Alpaca news endpoint or Benzinga)
to upgrade NO_NEWS → NEWS_LOW when there's recent context, and to
populate NEWS_HIGH for strategies that need same-session news.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from trading_bot_bl.earnings import check_earnings_blackout

log = logging.getLogger(__name__)


# Public label constants (used by CatalystFilter requirements)
NEWS_HIGH = "NEWS_HIGH"
NEWS_LOW = "NEWS_LOW"
NO_NEWS = "NO_NEWS"


class CatalystClassifier:
    """Earnings-calendar-driven catalyst labelling.

    Labels:

    - ``NEWS_HIGH`` — earnings today, yesterday, or tomorrow. The
      ticker is reacting to a confirmed material catalyst. Used by
      catalyst_momentum strategies.
    - ``NEWS_LOW`` — earnings within ±3 days but not today/yesterday/
      tomorrow. Some residual catalyst risk; not the centre of the
      announcement. Used as the moderate gating bucket.
    - ``NO_NEWS`` — earnings further than 3 days away (or not on the
      calendar). Mean-reversion strategies require this label.
    """

    def __init__(
        self,
        *,
        high_window_days: int = 1,
        low_window_days: int = 3,
        today: Optional[date] = None,
    ):
        if low_window_days < high_window_days:
            raise ValueError(
                f"low_window_days ({low_window_days}) must be >= "
                f"high_window_days ({high_window_days})"
            )
        self.high_window_days = high_window_days
        self.low_window_days = low_window_days
        self._today = today

    def classify(self, ticker: str) -> str:
        """Return the catalyst label for ``ticker`` for today's session.

        Always returns a valid label string. On lookup error, returns
        ``NO_NEWS`` (fail-open) — the same convention used by other
        wrappers around yfinance which can be flaky."""
        try:
            info = check_earnings_blackout(
                ticker,
                # Use a wide blackout window so check_earnings_blackout
                # always populates days_until_earnings; we make our
                # own decision below.
                pre_days=10,
                post_days=10,
                today=self._today or date.today(),
            )
        except Exception as exc:
            log.debug(
                "CatalystClassifier: lookup failed for %s: %s — "
                "defaulting to NO_NEWS",
                ticker, exc,
            )
            return NO_NEWS

        days = info.days_until_earnings
        if days is None or info.next_earnings_date is None:
            return NO_NEWS

        # days < 0 means earnings already happened (post-window)
        abs_days = abs(days)
        if abs_days <= self.high_window_days:
            return NEWS_HIGH
        if abs_days <= self.low_window_days:
            return NEWS_LOW
        return NO_NEWS

    def classify_many(self, tickers: list[str]) -> dict[str, str]:
        """Classify a watchlist in one go. Helpful at session start
        for the executor: build the ``TickerContext.catalyst_label``
        for each ticker before any scan tick."""
        return {t.upper(): self.classify(t) for t in tickers}
