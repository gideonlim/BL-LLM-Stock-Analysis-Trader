"""Live market data feed — Alpaca StockDataStream adapter.

Wraps ``alpaca.data.live.StockDataStream`` behind a stable, typed
interface so the executor / strategies never see Alpaca SDK objects
directly. Translates Alpaca's bar/quote/trade types into the
day_trader.models equivalents on the way through.

Lifecycle:

1. Construct with credentials + watchlist
2. Register handlers via ``on_bar``, ``on_quote``, ``on_trade``
3. ``await feed.start()`` to spin up the WebSocket task
4. ``await feed.update_subscriptions(new_watchlist)`` whenever the
   premarket scanner refreshes the day's symbols
5. ``await feed.stop()`` at session end (or in the daemon's crash
   handler)

Reconnect: alpaca-py's StockDataStream has internal reconnect logic.
We add a thin watchdog (heartbeat last_bar_seen_at) so the executor
can detect a stalled stream and restart the daemon.

Testing approach: the WebSocket itself is hard to test without
network. We test:

- Conversion functions (alpaca → day_trader.models) — pure functions
- Subscription tracking (set diff for adds/removes) — pure
- Handler registration + dispatch via a stub stream — easy

Full integration (real WebSocket, real Alpaca) happens in paper-trade
validation, not unit tests.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Optional

from day_trader.models import Bar, Quote, Trade

log = logging.getLogger(__name__)


# ── Handler signatures ───────────────────────────────────────────


OnBar = Callable[[Bar], Awaitable[None]]
OnQuote = Callable[[Quote], Awaitable[None]]
OnTrade = Callable[[Trade], Awaitable[None]]


# ── Conversions (alpaca-py → day_trader.models) ──────────────────


def _to_utc(ts) -> datetime:
    """Coerce Alpaca timestamp (which may be naive UTC, ISO string,
    or pandas Timestamp) into a tz-aware UTC datetime.

    Handles:
    - ``None`` → now(UTC)
    - tz-aware ``datetime`` → ``.astimezone(UTC)``
    - naive ``datetime`` → assumed UTC
    - ISO-format strings: ``"Z"``, ``"+00:00"``, ``"-04:00"``
    - pandas ``Timestamp`` → ``.to_pydatetime()`` then recurse
    - Anything else or malformed → now(UTC) with a warning
    """
    if ts is None:
        return datetime.now(tz=timezone.utc)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(ts, str):
        try:
            # Python 3.11+ fromisoformat handles "Z", "+HH:MM",
            # "-HH:MM" natively after we normalise the Z suffix.
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            # Always convert to UTC — the input may carry a non-UTC
            # offset (e.g. "-04:00" for ET) that needs normalising.
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            log.warning(
                "feed._to_utc: unparseable timestamp string %r "
                "— falling back to now(UTC)", ts,
            )
            return datetime.now(tz=timezone.utc)
    # pandas Timestamp
    if hasattr(ts, "to_pydatetime"):
        return _to_utc(ts.to_pydatetime())
    log.warning(
        "feed._to_utc: unexpected timestamp type %s: %r "
        "— falling back to now(UTC)", type(ts).__name__, ts,
    )
    return datetime.now(tz=timezone.utc)


def alpaca_bar_to_model(alpaca_bar) -> Bar:
    """Convert an alpaca-py Bar to ``day_trader.models.Bar``.

    Falls back gracefully on missing fields — the live feed
    occasionally omits VWAP or trade_count, which we just leave at
    zero for ``BarCache.add_bar`` to compute."""
    return Bar(
        ticker=str(getattr(alpaca_bar, "symbol", "")).upper(),
        timestamp=_to_utc(getattr(alpaca_bar, "timestamp", None)),
        open=float(getattr(alpaca_bar, "open", 0) or 0),
        high=float(getattr(alpaca_bar, "high", 0) or 0),
        low=float(getattr(alpaca_bar, "low", 0) or 0),
        close=float(getattr(alpaca_bar, "close", 0) or 0),
        volume=float(getattr(alpaca_bar, "volume", 0) or 0),
        vwap=float(getattr(alpaca_bar, "vwap", 0) or 0),
        trade_count=int(getattr(alpaca_bar, "trade_count", 0) or 0),
    )


def alpaca_quote_to_model(alpaca_quote) -> Quote:
    """Convert an alpaca-py Quote to ``day_trader.models.Quote``."""
    return Quote(
        ticker=str(getattr(alpaca_quote, "symbol", "")).upper(),
        timestamp=_to_utc(getattr(alpaca_quote, "timestamp", None)),
        bid_price=float(getattr(alpaca_quote, "bid_price", 0) or 0),
        bid_size=float(getattr(alpaca_quote, "bid_size", 0) or 0),
        ask_price=float(getattr(alpaca_quote, "ask_price", 0) or 0),
        ask_size=float(getattr(alpaca_quote, "ask_size", 0) or 0),
    )


def alpaca_trade_to_model(alpaca_trade) -> Trade:
    """Convert an alpaca-py Trade to ``day_trader.models.Trade``."""
    return Trade(
        ticker=str(getattr(alpaca_trade, "symbol", "")).upper(),
        timestamp=_to_utc(getattr(alpaca_trade, "timestamp", None)),
        price=float(getattr(alpaca_trade, "price", 0) or 0),
        size=float(getattr(alpaca_trade, "size", 0) or 0),
        exchange=str(getattr(alpaca_trade, "exchange", "") or ""),
    )


# ── Stream factory (overridable for tests) ───────────────────────


def _create_stream(api_key: str, api_secret: str, feed: str):
    """Default factory: returns an Alpaca-py StockDataStream.

    Tests inject a fake by passing ``stream_factory=...`` to
    :class:`MarketDataFeed`.
    """
    from alpaca.data.live import StockDataStream
    return StockDataStream(
        api_key=api_key,
        secret_key=api_secret,
        feed=feed,
    )


# ── Feed adapter ─────────────────────────────────────────────────


class MarketDataFeed:
    """Async wrapper around Alpaca's live data stream.

    Public API is fully async. The underlying ``StockDataStream``
    runs as an asyncio task started by :meth:`start` and torn down
    by :meth:`stop`. Subscriptions can be updated mid-session via
    :meth:`update_subscriptions` for cases where the watchlist
    changes (e.g. mid-session catalyst spike).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        feed: str = "sip",
        stream_factory: Optional[Callable] = None,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._feed = feed
        self._stream_factory = stream_factory or _create_stream
        self._stream = None
        self._subscribed: set[str] = set()
        self._on_bar: Optional[OnBar] = None
        self._on_quote: Optional[OnQuote] = None
        self._on_trade: Optional[OnTrade] = None
        self._task: Optional[asyncio.Task] = None
        # Heartbeat — last time any bar event arrived
        self._last_bar_at: Optional[datetime] = None
        self._lock = threading.Lock()

    # ── Handler registration ─────────────────────────────────────

    def on_bar(self, callback: OnBar) -> None:
        """Register the bar callback. MUST be set before ``start()``."""
        self._on_bar = callback

    def on_quote(self, callback: OnQuote) -> None:
        self._on_quote = callback

    def on_trade(self, callback: OnTrade) -> None:
        self._on_trade = callback

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, symbols: list[str]) -> None:
        """Start the WebSocket task and subscribe to the initial
        watchlist. Idempotent — calling on a running feed updates
        subscriptions instead."""
        if self.is_running:
            await self.update_subscriptions(symbols)
            return

        self._stream = self._stream_factory(
            self._api_key, self._api_secret, self._feed,
        )
        self._subscribed = set()
        self._subscribe_internal(symbols)

        # Run the stream in its own task so we don't block the executor
        self._task = asyncio.create_task(
            self._run_loop(), name="day_trader.market_data_feed",
        )
        log.info(
            "MarketDataFeed: started, %d symbols subscribed (feed=%s)",
            len(self._subscribed), self._feed,
        )

    async def stop(self) -> None:
        """Stop the WebSocket and cancel the task. Safe to call
        multiple times. Safe to call on a never-started feed."""
        if self._stream is not None:
            try:
                # alpaca-py's stop_ws is async on newer versions, sync
                # on some older ones — be defensive.
                stop_fn = getattr(self._stream, "stop_ws", None)
                if stop_fn is not None:
                    result = stop_fn()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as exc:
                log.warning(
                    "MarketDataFeed: stream stop_ws error (continuing): %s",
                    exc,
                )

        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._stream = None
        self._subscribed.clear()

    async def update_subscriptions(self, symbols: list[str]) -> None:
        """Diff the new watchlist against current subscriptions;
        subscribe to additions, unsubscribe from removals."""
        if self._stream is None:
            raise RuntimeError(
                "MarketDataFeed.update_subscriptions called before start()"
            )
        new = {s.upper() for s in symbols if s and s.strip()}
        with self._lock:
            current = set(self._subscribed)
            to_add = new - current
            to_remove = current - new

        if to_add:
            self._subscribe_internal(sorted(to_add))
        if to_remove:
            self._unsubscribe_internal(sorted(to_remove))

    def subscribed_symbols(self) -> list[str]:
        """Snapshot of currently-subscribed symbols (sorted)."""
        with self._lock:
            return sorted(self._subscribed)

    @property
    def last_bar_at(self) -> Optional[datetime]:
        """Timestamp (UTC) of the most recent bar event we forwarded.

        Watchdogs use this to detect a stalled stream — if no bars
        have arrived for >2 min during market hours, something is
        wrong (network partition, Alpaca outage, our handler hung).
        """
        with self._lock:
            return self._last_bar_at

    # ── Subscription internals ───────────────────────────────────

    def _subscribe_internal(self, symbols: list[str]) -> None:
        """Add subscriptions for ``symbols`` to whichever event
        types have a registered handler. Symbols normalized to
        uppercase before storage and forwarding to alpaca-py."""
        if not symbols:
            return
        normalized = [
            s.strip().upper() for s in symbols if s and s.strip()
        ]
        if not normalized:
            return
        with self._lock:
            self._subscribed.update(normalized)
        # alpaca-py uses *args for the subscribe methods
        if self._on_bar is not None:
            self._stream.subscribe_bars(self._handle_bar, *normalized)
        if self._on_quote is not None:
            self._stream.subscribe_quotes(self._handle_quote, *normalized)
        if self._on_trade is not None:
            self._stream.subscribe_trades(self._handle_trade, *normalized)
        log.debug(
            "MarketDataFeed: subscribed to %d new symbols (total=%d)",
            len(normalized), len(self._subscribed),
        )

    def _unsubscribe_internal(self, symbols: list[str]) -> None:
        if not symbols:
            return
        normalized = [
            s.strip().upper() for s in symbols if s and s.strip()
        ]
        if not normalized:
            return
        with self._lock:
            self._subscribed.difference_update(normalized)
        if self._on_bar is not None:
            unsub = getattr(self._stream, "unsubscribe_bars", None)
            if unsub is not None:
                unsub(*normalized)
        if self._on_quote is not None:
            unsub = getattr(self._stream, "unsubscribe_quotes", None)
            if unsub is not None:
                unsub(*normalized)
        if self._on_trade is not None:
            unsub = getattr(self._stream, "unsubscribe_trades", None)
            if unsub is not None:
                unsub(*normalized)
        log.debug(
            "MarketDataFeed: unsubscribed %d symbols (total=%d)",
            len(normalized), len(self._subscribed),
        )

    # ── Stream main loop ─────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Run the underlying stream until cancelled or stopped.

        alpaca-py's StockDataStream has its own internal reconnect
        loop, so we just await its ``_run_forever`` (or ``run``) and
        let the framework handle transport-level reconnects. Our
        responsibility is to translate cancellation cleanly and to
        propagate fatal errors (which we do via task exception ->
        executor's crash handler)."""
        try:
            run_forever = (
                getattr(self._stream, "_run_forever", None)
                or getattr(self._stream, "run_forever", None)
            )
            if run_forever is None:
                # Last resort: call the synchronous ``run`` in an executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._stream.run)
                return
            result = run_forever()
            if asyncio.iscoroutine(result):
                await result
        except asyncio.CancelledError:
            log.info("MarketDataFeed: run loop cancelled")
            raise
        except Exception:
            log.exception("MarketDataFeed: stream crashed")
            raise

    # ── Alpaca event handlers ─────────────────────────────────────

    async def _handle_bar(self, alpaca_bar) -> None:
        if self._on_bar is None:
            return
        try:
            bar = alpaca_bar_to_model(alpaca_bar)
        except Exception:
            log.exception(
                "MarketDataFeed: bar conversion failed: %r", alpaca_bar,
            )
            return
        with self._lock:
            self._last_bar_at = datetime.now(tz=timezone.utc)
        try:
            await self._on_bar(bar)
        except Exception:
            log.exception(
                "MarketDataFeed: on_bar handler raised on %s", bar.ticker,
            )

    async def _handle_quote(self, alpaca_quote) -> None:
        if self._on_quote is None:
            return
        try:
            q = alpaca_quote_to_model(alpaca_quote)
        except Exception:
            log.exception(
                "MarketDataFeed: quote conversion failed: %r", alpaca_quote,
            )
            return
        try:
            await self._on_quote(q)
        except Exception:
            log.exception(
                "MarketDataFeed: on_quote handler raised on %s", q.ticker,
            )

    async def _handle_trade(self, alpaca_trade) -> None:
        if self._on_trade is None:
            return
        try:
            t = alpaca_trade_to_model(alpaca_trade)
        except Exception:
            log.exception(
                "MarketDataFeed: trade conversion failed: %r", alpaca_trade,
            )
            return
        try:
            await self._on_trade(t)
        except Exception:
            log.exception(
                "MarketDataFeed: on_trade handler raised on %s", t.ticker,
            )
