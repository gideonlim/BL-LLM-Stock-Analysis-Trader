"""Tests for the MarketDataFeed adapter.

We don't connect to real Alpaca. Instead:

- Conversion functions (alpaca_*_to_model) — pure functions, easy to
  test with SimpleNamespace fakes.
- Subscription tracking — MarketDataFeed delegates to a stub stream
  whose subscribe_xxx methods record calls.
- Lifecycle (start/stop/update) — stub stream confirms calls happen
  in expected order.

Real WebSocket integration is verified during paper trading.
"""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from day_trader.data.feed import (
    MarketDataFeed,
    alpaca_bar_to_model,
    alpaca_quote_to_model,
    alpaca_trade_to_model,
)
from day_trader.models import Bar, Quote, Trade


# ── Conversion functions ─────────────────────────────────────────


class TestConversions(unittest.TestCase):
    def test_bar_conversion(self):
        ab = SimpleNamespace(
            symbol="AAPL", timestamp=datetime(2026, 4, 28, 13, 30, tzinfo=timezone.utc),
            open=100.0, high=101.0, low=99.5, close=100.8,
            volume=150_000, vwap=100.4, trade_count=420,
        )
        bar = alpaca_bar_to_model(ab)
        self.assertIsInstance(bar, Bar)
        self.assertEqual(bar.ticker, "AAPL")
        self.assertEqual(bar.open, 100.0)
        self.assertEqual(bar.high, 101.0)
        self.assertEqual(bar.low, 99.5)
        self.assertEqual(bar.close, 100.8)
        self.assertEqual(bar.volume, 150_000)
        self.assertEqual(bar.vwap, 100.4)
        self.assertEqual(bar.trade_count, 420)
        self.assertEqual(bar.timestamp.tzinfo, timezone.utc)

    def test_bar_conversion_naive_timestamp_assumes_utc(self):
        ab = SimpleNamespace(
            symbol="AAPL", timestamp=datetime(2026, 4, 28, 13, 30),
            open=100, high=101, low=99, close=100, volume=100,
        )
        bar = alpaca_bar_to_model(ab)
        self.assertEqual(bar.timestamp.tzinfo, timezone.utc)

    def test_bar_conversion_iso_string_timestamp(self):
        ab = SimpleNamespace(
            symbol="AAPL", timestamp="2026-04-28T13:30:00Z",
            open=100, high=101, low=99, close=100, volume=100,
        )
        bar = alpaca_bar_to_model(ab)
        self.assertEqual(bar.timestamp.tzinfo, timezone.utc)

    def test_bar_conversion_uppercase_ticker(self):
        ab = SimpleNamespace(
            symbol="aapl", timestamp=datetime.now(tz=timezone.utc),
            open=100, high=101, low=99, close=100, volume=100,
        )
        bar = alpaca_bar_to_model(ab)
        self.assertEqual(bar.ticker, "AAPL")

    def test_bar_conversion_missing_optional_fields(self):
        # No vwap or trade_count
        ab = SimpleNamespace(
            symbol="AAPL", timestamp=datetime.now(tz=timezone.utc),
            open=100, high=101, low=99, close=100, volume=100,
        )
        bar = alpaca_bar_to_model(ab)
        self.assertEqual(bar.vwap, 0.0)
        self.assertEqual(bar.trade_count, 0)

    def test_quote_conversion(self):
        aq = SimpleNamespace(
            symbol="MSFT",
            timestamp=datetime(2026, 4, 28, 13, 30, tzinfo=timezone.utc),
            bid_price=200.0, bid_size=10, ask_price=200.05, ask_size=15,
        )
        q = alpaca_quote_to_model(aq)
        self.assertIsInstance(q, Quote)
        self.assertEqual(q.ticker, "MSFT")
        self.assertEqual(q.bid_price, 200.0)
        self.assertEqual(q.ask_price, 200.05)

    def test_trade_conversion(self):
        at = SimpleNamespace(
            symbol="TSLA",
            timestamp=datetime(2026, 4, 28, 13, 30, tzinfo=timezone.utc),
            price=250.0, size=100, exchange="V",
        )
        t = alpaca_trade_to_model(at)
        self.assertIsInstance(t, Trade)
        self.assertEqual(t.ticker, "TSLA")
        self.assertEqual(t.price, 250.0)
        self.assertEqual(t.size, 100)
        self.assertEqual(t.exchange, "V")


# ── MarketDataFeed lifecycle + subscription tracking ─────────────


class _StubStream:
    """Records subscribe/unsubscribe calls. ``_run_forever`` returns
    a coroutine that sleeps until cancelled (mimics a long-running
    WebSocket task)."""

    def __init__(self):
        self.subscribed_bars: set[str] = set()
        self.subscribed_quotes: set[str] = set()
        self.subscribed_trades: set[str] = set()
        self.bar_handler = None
        self.quote_handler = None
        self.trade_handler = None
        self.stopped = False
        self.run_called = False

    def subscribe_bars(self, handler, *symbols):
        self.bar_handler = handler
        self.subscribed_bars.update(symbols)

    def subscribe_quotes(self, handler, *symbols):
        self.quote_handler = handler
        self.subscribed_quotes.update(symbols)

    def subscribe_trades(self, handler, *symbols):
        self.trade_handler = handler
        self.subscribed_trades.update(symbols)

    def unsubscribe_bars(self, *symbols):
        self.subscribed_bars.difference_update(symbols)

    def unsubscribe_quotes(self, *symbols):
        self.subscribed_quotes.difference_update(symbols)

    def unsubscribe_trades(self, *symbols):
        self.subscribed_trades.difference_update(symbols)

    async def _run_forever(self):
        self.run_called = True
        try:
            # Sleep "forever" until cancelled
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    async def stop_ws(self):
        self.stopped = True


def _factory(stream: _StubStream):
    """Returns a stream factory that yields ``stream`` on call."""
    def make(api_key, api_secret, feed):
        return stream
    return make


class TestFeedLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_initial_subscribe_only_triggers_registered_types(self):
        stream = _StubStream()
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(stream),
        )

        async def on_bar(bar):
            pass

        feed.on_bar(on_bar)
        # Note: no on_quote / on_trade registered

        await feed.start(["AAPL", "MSFT"])
        self.assertTrue(feed.is_running)

        # Only bars subscribed; quotes/trades not
        self.assertEqual(stream.subscribed_bars, {"AAPL", "MSFT"})
        self.assertEqual(stream.subscribed_quotes, set())
        self.assertEqual(stream.subscribed_trades, set())

        await feed.stop()

    async def test_subscribed_symbols_snapshot(self):
        stream = _StubStream()
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(stream),
        )

        async def on_bar(bar):
            pass

        feed.on_bar(on_bar)
        await feed.start(["aapl", "MSFT"])  # mixed case
        self.assertEqual(feed.subscribed_symbols(), ["AAPL", "MSFT"])
        await feed.stop()

    async def test_update_subscriptions_diffs_correctly(self):
        stream = _StubStream()
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(stream),
        )

        async def on_bar(bar):
            pass

        feed.on_bar(on_bar)
        await feed.start(["AAPL", "MSFT"])

        # Now switch to a partially overlapping list
        await feed.update_subscriptions(["MSFT", "TSLA", "NVDA"])

        # AAPL removed, TSLA + NVDA added
        self.assertEqual(stream.subscribed_bars, {"MSFT", "TSLA", "NVDA"})
        await feed.stop()

    async def test_stop_idempotent(self):
        stream = _StubStream()
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(stream),
        )

        async def on_bar(bar):
            pass

        feed.on_bar(on_bar)
        await feed.start(["AAPL"])
        await feed.stop()
        # Second stop must not raise
        await feed.stop()
        self.assertFalse(feed.is_running)

    async def test_start_when_already_running_updates_subscriptions(self):
        stream = _StubStream()
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(stream),
        )

        async def on_bar(bar):
            pass

        feed.on_bar(on_bar)
        await feed.start(["AAPL"])
        # Re-starting should diff to the new list
        await feed.start(["MSFT"])
        self.assertEqual(stream.subscribed_bars, {"MSFT"})
        await feed.stop()

    async def test_handler_dispatch(self):
        """Stub stream's bar_handler is invoked → our on_bar runs
        with a converted Bar instance."""
        stream = _StubStream()
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(stream),
        )

        received: list[Bar] = []

        async def on_bar(bar):
            received.append(bar)

        feed.on_bar(on_bar)
        await feed.start(["AAPL"])

        # Simulate Alpaca delivering a bar
        alpaca_bar = SimpleNamespace(
            symbol="AAPL",
            timestamp=datetime.now(tz=timezone.utc),
            open=100, high=101, low=99, close=100.5,
            volume=10_000, vwap=100.2, trade_count=42,
        )
        await stream.bar_handler(alpaca_bar)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].ticker, "AAPL")
        self.assertEqual(received[0].close, 100.5)
        # Heartbeat updated
        self.assertIsNotNone(feed.last_bar_at)

        await feed.stop()

    async def test_handler_exception_doesnt_break_feed(self):
        stream = _StubStream()
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(stream),
        )

        async def on_bar(bar):
            raise RuntimeError("strategy crashed")

        feed.on_bar(on_bar)
        await feed.start(["AAPL"])

        alpaca_bar = SimpleNamespace(
            symbol="AAPL", timestamp=datetime.now(tz=timezone.utc),
            open=100, high=101, low=99, close=100, volume=100,
        )
        # Must not raise out of the feed
        await stream.bar_handler(alpaca_bar)
        await feed.stop()

    async def test_update_before_start_raises(self):
        feed = MarketDataFeed(
            "k", "s", stream_factory=_factory(_StubStream()),
        )
        with self.assertRaises(RuntimeError):
            await feed.update_subscriptions(["AAPL"])


if __name__ == "__main__":
    unittest.main()
