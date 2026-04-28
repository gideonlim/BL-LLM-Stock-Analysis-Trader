"""Symbol locks — prevent swing/day same-symbol overlap.

The day-trader and swing bot share an Alpaca account. Alpaca nets
positions and quantity changes at the **account + ticker** level —
if swing holds 100 AAPL and day buys 50 AAPL, Alpaca shows one
150-share AAPL position. The existing ``AlpacaBroker.close_position``
closes the entire symbol, so a 15:55 day-trade force-flat would
accidentally liquidate the swing AAPL position.

This module is the non-negotiable filter that prevents that:

- :class:`SymbolLock` answers "is this ticker safe for the day-trader
  to enter?" by checking swing positions, swing open orders, and
  outstanding day-trade orders.

- The complementary ``is_held_by_day_trader()`` accessor is consumed
  by the swing risk manager so it never enters a ticker the day-bot
  is currently in.

Long-term path: split into a separate Alpaca subaccount for the
day-trader. Until then, the lock is the safety net.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Optional

from day_trader.order_tags import is_day_trade_id

log = logging.getLogger(__name__)


# ── Lock-result types ─────────────────────────────────────────────


@dataclass(frozen=True)
class LockReason:
    """Why a ticker is locked. Empty ``reason`` means not locked."""

    locked: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.locked


# Sentinel constants for callers that want to switch on reason.
REASON_SWING_POSITION = "swing_position"
REASON_SWING_OPEN_ORDER = "swing_open_order"
REASON_DAYTRADE_OPEN = "daytrade_open"
REASON_DAYTRADE_PENDING = "daytrade_pending_order"


@dataclass
class _Snapshot:
    """One periodic refresh of broker state. Cheap to compare for staleness."""

    captured_at: datetime
    swing_position_tickers: frozenset[str]
    swing_pending_tickers: frozenset[str]
    daytrade_position_tickers: frozenset[str]
    daytrade_pending_tickers: frozenset[str]


class SymbolLock:
    """Tracks which tickers are off-limits to which bot.

    Periodically refreshes from the broker. Filters consult
    :meth:`is_locked` at scan time without re-fetching.

    Usage::

        lock = SymbolLock(broker)
        lock.refresh()  # at session start and every ~60s

        verdict = lock.is_locked("AAPL")
        if verdict.locked:
            log.info(f"Skipping AAPL: {verdict.reason}")

        # Symmetric path for the swing risk manager:
        if lock.is_held_by_day_trader("MSFT"):
            ...  # swing rejects entry

    The lock is read-only over broker state. It never submits orders.
    """

    def __init__(
        self,
        broker,
        *,
        refresh_ttl_seconds: float = 60.0,
    ):
        self._broker = broker
        self._refresh_ttl = timedelta(seconds=refresh_ttl_seconds)
        self._snapshot: Optional[_Snapshot] = None

    # ── Refresh / snapshot ────────────────────────────────────────

    def refresh(self, *, now: Optional[datetime] = None) -> _Snapshot:
        """Re-fetch broker state and rebuild the snapshot.

        Cheap-to-call (a few REST calls) — the executor calls this
        at session start and every ~60 s during market hours.
        """
        now = now or datetime.now()
        portfolio = self._broker.get_portfolio()
        open_orders = self._fetch_open_orders()

        swing_pos: set[str] = set()
        day_pos: set[str] = set()
        for ticker, info in (portfolio.positions or {}).items():
            # Position itself isn't tagged. We use the heuristic:
            # if there is at least one open order on this ticker
            # whose client_order_id is `dt:`-tagged, treat the
            # position as day-trader-owned. This matches what
            # recovery.reconcile() resolves more precisely; here
            # we just need the lock to be defensive.
            if any(
                self._order_ticker(o) == ticker
                and is_day_trade_id(self._order_client_id(o))
                for o in open_orders
            ):
                day_pos.add(ticker)
            else:
                swing_pos.add(ticker)

        swing_pending: set[str] = set()
        day_pending: set[str] = set()
        for o in open_orders:
            t = self._order_ticker(o)
            if not t:
                continue
            if is_day_trade_id(self._order_client_id(o)):
                day_pending.add(t)
            else:
                swing_pending.add(t)

        snap = _Snapshot(
            captured_at=now,
            swing_position_tickers=frozenset(swing_pos),
            swing_pending_tickers=frozenset(swing_pending),
            daytrade_position_tickers=frozenset(day_pos),
            daytrade_pending_tickers=frozenset(day_pending),
        )
        self._snapshot = snap
        return snap

    def _ensure_fresh(self, *, now: Optional[datetime] = None) -> _Snapshot:
        """Refresh if the snapshot is missing or older than the TTL."""
        now = now or datetime.now()
        if (
            self._snapshot is None
            or now - self._snapshot.captured_at > self._refresh_ttl
        ):
            return self.refresh(now=now)
        return self._snapshot

    # ── Public accessors ──────────────────────────────────────────

    def is_locked(
        self,
        ticker: str,
        *,
        now: Optional[datetime] = None,
    ) -> LockReason:
        """Should the day-trader skip ``ticker``?

        Locks if any of:
        - Swing has an open position on ``ticker``
        - Swing has an open order on ``ticker``
        - Day-trader already has an open position on ``ticker``
        - Day-trader has an outstanding open order on ``ticker``

        Last two prevent doubling-up and silent qty drift between
        the strategy's expectation and the broker's reality.
        """
        snap = self._ensure_fresh(now=now)
        t = ticker.upper()
        if t in snap.swing_position_tickers:
            return LockReason(True, REASON_SWING_POSITION)
        if t in snap.swing_pending_tickers:
            return LockReason(True, REASON_SWING_OPEN_ORDER)
        if t in snap.daytrade_position_tickers:
            return LockReason(True, REASON_DAYTRADE_OPEN)
        if t in snap.daytrade_pending_tickers:
            return LockReason(True, REASON_DAYTRADE_PENDING)
        return LockReason(False, "")

    def is_held_by_day_trader(
        self,
        ticker: str,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        """For the swing risk manager: is ``ticker`` in any day-trade
        position or order? Symmetric to :meth:`is_locked` but only
        considers day-trade exposure (so swing can reject entries
        on tickers the day-bot already owns)."""
        snap = self._ensure_fresh(now=now)
        t = ticker.upper()
        return (
            t in snap.daytrade_position_tickers
            or t in snap.daytrade_pending_tickers
        )

    def locked_tickers(self) -> frozenset[str]:
        """All tickers the day-trader is currently locked out of.
        Useful for logging/diagnostics."""
        snap = self._ensure_fresh()
        return (
            snap.swing_position_tickers
            | snap.swing_pending_tickers
            | snap.daytrade_position_tickers
            | snap.daytrade_pending_tickers
        )

    # ── Broker-shape adapters ────────────────────────────────────
    # The Alpaca-py order object exposes ``symbol`` and
    # ``client_order_id``. We accept duck-typed objects for testing.

    @staticmethod
    def _order_ticker(order) -> str:
        for attr in ("symbol", "ticker"):
            t = getattr(order, attr, None)
            if t:
                return str(t).upper()
        return ""

    @staticmethod
    def _order_client_id(order) -> str:
        return str(getattr(order, "client_order_id", "") or "")

    def _fetch_open_orders(self) -> list:
        """Fetch open orders via whichever method the broker exposes.

        ``trading_bot_bl/broker.py`` has a private helper for this;
        we use the public TradingClient if available, else fall
        back to inspecting ``broker._client`` directly.
        """
        # Preferred path: a public method on AlpacaBroker.
        if hasattr(self._broker, "list_open_orders"):
            return list(self._broker.list_open_orders())
        # Fallback: hit alpaca-py via the wrapped client.
        client = getattr(self._broker, "_client", None)
        if client is None:
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
            return list(client.get_orders(filter=req))
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("SymbolLock: open-order fetch failed: %s", exc)
            return []
