"""Position manager — tracks live day-trade positions.

Sits between the executor and the broker: when a fill arrives, the
executor calls :meth:`open_position`; on every scan tick, the
executor calls :meth:`check_all` which delegates to each position's
strategy ``manage()`` for time-stops and trailing logic; when a
close fires, the executor calls :meth:`close_position`.

The manager does NOT submit orders. It returns :class:`ExitIntent`
objects that the executor routes through ``close_tagged_daytrade_qty``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from day_trader.calendar import NyseSession
from day_trader.data.cache import BarCache
from day_trader.models import ExitIntent, OpenDayTrade

log = logging.getLogger(__name__)


class PositionManager:
    """In-memory registry of open day-trade positions."""

    def __init__(self) -> None:
        self._positions: dict[str, OpenDayTrade] = {}

    # ── Lifecycle ─────────────────────────────────────────────────

    def open_position(self, position: OpenDayTrade) -> None:
        """Register a newly filled day-trade position."""
        key = position.ticker.upper()
        if key in self._positions:
            log.warning(
                "PositionManager: overwriting existing position for %s "
                "(prior entry_price=%.2f, new=%.2f)",
                key,
                self._positions[key].entry_price,
                position.entry_price,
            )
        self._positions[key] = position
        log.info(
            "PositionManager: opened %s %s %d sh @ %.2f "
            "(tag=%s)",
            position.side, key, position.qty,
            position.entry_price, position.parent_client_order_id,
        )

    def close_position(self, ticker: str) -> Optional[OpenDayTrade]:
        """Remove and return the position for ``ticker``.

        Returns ``None`` if no such position (idempotent — safe to
        call on a ticker that was already closed by the broker).
        """
        key = ticker.upper()
        pos = self._positions.pop(key, None)
        if pos is not None:
            log.info("PositionManager: closed %s", key)
        return pos

    def reset_for_session(self) -> None:
        """Clear all positions. Called at session start after recovery
        re-seeds from the journal."""
        self._positions.clear()

    # ── Read accessors ────────────────────────────────────────────

    def get(self, ticker: str) -> Optional[OpenDayTrade]:
        return self._positions.get(ticker.upper())

    def all_positions(self) -> list[OpenDayTrade]:
        return list(self._positions.values())

    def tickers(self) -> list[str]:
        return sorted(self._positions.keys())

    def count(self) -> int:
        return len(self._positions)

    def has(self, ticker: str) -> bool:
        return ticker.upper() in self._positions

    # ── Per-tick management ──────────────────────────────────────

    def check_all(
        self,
        strategies: dict,
        bar_cache: BarCache,
        now_et: datetime,
        session: NyseSession,
    ) -> list[ExitIntent]:
        """Run each position's strategy.manage() and collect exits.

        ``strategies`` maps ``strategy_name → DayTradeStrategy``.
        Returns a list of :class:`ExitIntent` for positions whose
        strategies want to close (time-stop, trailing logic, etc.).
        Positions whose strategies return ``None`` are left alone
        (their broker bracket SL/TP handles exit).
        """
        exits: list[ExitIntent] = []
        for ticker, pos in list(self._positions.items()):
            strat = strategies.get(pos.strategy)
            if strat is None:
                continue
            try:
                intent = strat.manage(pos, bar_cache, now_et, session)
            except Exception:
                log.exception(
                    "PositionManager: %s.manage(%s) raised — skipping",
                    pos.strategy, ticker,
                )
                continue
            if intent is not None:
                exits.append(intent)
        return exits

    # ── Force-close helpers ──────────────────────────────────────

    def all_for_force_close(self) -> list[OpenDayTrade]:
        """Return all positions so the executor can force-flat them
        (exit-only mode / EOD force-close). Does NOT remove them —
        the executor calls ``close_position`` after each successful
        close order."""
        return list(self._positions.values())
