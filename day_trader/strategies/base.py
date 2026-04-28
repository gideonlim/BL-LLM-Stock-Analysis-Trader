"""DayTradeStrategy ABC.

A strategy's job is to emit ``DayTradeSignal`` candidates each scan
tick. Strategies are stateless across ticks where possible — what
state they do hold (e.g. "have I already fired the ORB on AAPL
today?") is per-ticker, per-session, and reset via
:meth:`reset_for_session`.

Strategies do NOT decide whether a signal becomes an order — that's
the filter pipeline + risk manager's job. They only declare the
setup and let the rest of the stack route it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from day_trader.calendar import NyseSession
from day_trader.data.cache import BarCache
from day_trader.models import (
    DayTradeSignal,
    ExitIntent,
    MarketState,
    OpenDayTrade,
    TickerContext,
)

log = logging.getLogger(__name__)


class DayTradeStrategy(ABC):
    """Base for all day-trade strategies."""

    #: Short snake_case identifier used in logs, journal, filter
    #: histogram keys. Must match the keys in
    #: ``filters.catalyst_filter.DEFAULT_STRATEGY_REQUIREMENTS``.
    name: str = "base"

    #: If set, the CatalystFilter rejects signals whose
    #: ``signal.catalyst_label`` doesn't match this value.
    #: ``None`` = strategy accepts any catalyst label.
    required_catalyst_label: Optional[str] = None

    # ── Per-ticker fired-set (default impl — strategies that fire
    # multiple times per ticker can override) ───────────────────

    def __init__(self) -> None:
        # Tracks tickers we've already fired a signal on this session
        # so single-shot strategies (ORB) don't spam the pipeline.
        self._fired_today: set[str] = set()

    def reset_for_session(self) -> None:
        """Clear per-session state. Called by the executor at session
        start (after recovery succeeds)."""
        self._fired_today.clear()

    def already_fired(self, ticker: str) -> bool:
        return ticker.upper() in self._fired_today

    def mark_fired(self, ticker: str) -> None:
        self._fired_today.add(ticker.upper())

    # ── Scan API ──────────────────────────────────────────────────

    @abstractmethod
    def scan_ticker(
        self,
        ticker: str,
        bar_cache: BarCache,
        ticker_context: TickerContext,
        market_state: MarketState,
        now_et: datetime,
        session: NyseSession,
    ) -> Optional[DayTradeSignal]:
        """Run the strategy on a single ticker.

        Return ``None`` if the setup isn't present, else a
        :class:`DayTradeSignal` with entry / SL / TP populated.

        ``now_et`` is the current ET time (tz-aware). Used to gate
        time-based logic (e.g. ORB fires only after 09:35).
        ``session`` is the resolved NYSE session for ``now_et.date()``,
        already known to be a trading day by the executor — strategies
        don't need to re-check.
        """

    def scan(
        self,
        watchlist: list[str],
        bar_cache: BarCache,
        ticker_contexts: dict[str, TickerContext],
        market_state: MarketState,
        now_et: datetime,
        session: NyseSession,
    ) -> list[DayTradeSignal]:
        """Default: iterate ``watchlist``, calling :meth:`scan_ticker` on
        each. Strategy-level exceptions are logged and the offending
        ticker is skipped — one bad ticker doesn't kill the scan."""
        signals: list[DayTradeSignal] = []
        for ticker in watchlist:
            t = ticker.upper()
            ctx = ticker_contexts.get(t)
            if ctx is None:
                continue
            try:
                sig = self.scan_ticker(
                    t, bar_cache, ctx, market_state, now_et, session,
                )
            except Exception:
                log.exception(
                    "%s: scan_ticker(%s) raised — skipping",
                    self.name, t,
                )
                continue
            if sig is not None:
                signals.append(sig)
        return signals

    # ── Position management API ──────────────────────────────────

    def manage(
        self,
        position: OpenDayTrade,
        bar_cache: BarCache,
        now_et: datetime,
        session: NyseSession,
    ) -> Optional[ExitIntent]:
        """Per-tick check on an open position. Default: no override.

        Strategies that want time-stops, trailing logic, or
        manage-and-exit-on-pattern-break override this. Returning
        ``None`` lets the broker's bracket SL/TP run unmodified.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"<{self.__class__.__name__} name={self.name}>"
