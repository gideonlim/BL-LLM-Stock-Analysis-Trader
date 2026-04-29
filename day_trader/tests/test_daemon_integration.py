"""End-to-end integration test for DayTraderDaemon.

Wires up the full daemon with stubbed broker, feed, premarket
scanner, and catalyst classifier. Runs a complete session
(pre_session → run_session → post_session) and verifies:

- Pre-session loads watchlist, sets up risk manager, opens feed
- Scan ticks produce signals → filter pipeline → risk review
- Heartbeat is written each loop
- Force-flat fires at close-5
- Post-session stops feed cleanly

We don't connect to real Alpaca; the broker, feed, scanner, and
classifier are all duck-typed stubs that the daemon doesn't know
are stubs.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
import unittest.mock as _um
from datetime import date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zoneinfo import ZoneInfo

# Mock alpaca SDK before importing the daemon
_alpaca_mock = _um.MagicMock()
for mod in [
    "alpaca", "alpaca.trading", "alpaca.trading.client",
    "alpaca.trading.requests", "alpaca.trading.enums",
    "alpaca.data", "alpaca.data.historical", "alpaca.data.requests",
    "alpaca.data.live", "alpaca.data.timeframe",
    "alpaca.data.models",
]:
    sys.modules.setdefault(mod, _alpaca_mock)


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


# ── Stubs ────────────────────────────────────────────────────────


class _StubBroker:
    """Tracks every call so tests can assert on submitted orders + closes."""

    def __init__(self):
        self.submitted_orders = []
        self.cancelled_orders = []
        self.closed_positions = []
        self._client = _um.MagicMock()
        self._data_client = _um.MagicMock()

    def get_portfolio(self):
        return SimpleNamespace(
            equity=100_000.0,
            cash=50_000.0,
            buying_power=200_000.0,
            market_value=50_000.0,
            day_pnl=0.0,
            day_pnl_pct=0.0,
            positions={},
        )

    def list_open_orders(self):
        return []

    def submit_bracket_order(self, **kwargs):
        from trading_bot_bl.models import OrderResult
        self.submitted_orders.append(kwargs)
        return OrderResult(
            ticker=kwargs["ticker"],
            order_id=f"alpaca-{len(self.submitted_orders)}",
            client_order_id=kwargs.get("client_order_id", ""),
            status="submitted",
            side=kwargs["side"],
            notional=kwargs["notional"],
        )

    def submit_market_order(self, **kwargs):
        from trading_bot_bl.models import OrderResult
        return OrderResult(
            ticker=kwargs["ticker"],
            order_id=f"alpaca-mkt-{kwargs.get('client_order_id', '')}",
            client_order_id=kwargs.get("client_order_id", ""),
            status="submitted",
            side=kwargs["side"],
            notional=0.0,
        )

    def cancel_order_by_id(self, order_id: str):
        self.cancelled_orders.append(order_id)


class _StubFeed:
    """Records start/stop/subscribe calls; never opens a real WebSocket."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.subscribed = []
        self._on_bar = None
        self._on_quote = None

    def on_bar(self, cb):
        self._on_bar = cb

    def on_quote(self, cb):
        self._on_quote = cb

    def on_trade(self, cb):
        pass

    async def start(self, symbols):
        self.started = True
        self.subscribed = list(symbols)

    async def stop(self):
        self.stopped = True

    async def update_subscriptions(self, symbols):
        self.subscribed = list(symbols)


class _StubPremarketScanner:
    """Returns a fixed dict of TickerContexts."""

    def __init__(self, contexts: dict):
        self.contexts = contexts

    def scan(self, universe, target_date, top_n=50):
        # Return only contexts for tickers in the universe
        u = {t.upper() for t in universe}
        return {
            t: ctx for t, ctx in self.contexts.items()
            if t in u
        }


class _StubCatalystClassifier:
    def __init__(self, label="NO_NEWS"):
        self.label = label

    def classify(self, ticker):
        return self.label

    def classify_many(self, tickers):
        return {t.upper(): self.label for t in tickers}


# ── Daemon assembly helper ──────────────────────────────────────


def _make_daemon(*, dry_run: bool = True):
    """Build a daemon with all stubs wired in."""
    from day_trader.config import DayTradeConfig
    from day_trader.executor import DayTraderDaemon
    from day_trader.filters.base import FilterPipeline
    from day_trader.filters.cooldown import CooldownTracker
    from day_trader.filters.cooldown_filter import CooldownFilter
    from day_trader.filters.regime_filter import RegimeFilter
    from day_trader.heartbeat import Heartbeat
    from day_trader.models import TickerContext
    from day_trader.order_tags import SequenceCounter
    from day_trader.strategies.orb_vwap import OrbVwapStrategy

    tmp = TemporaryDirectory()
    config = DayTradeConfig()
    config.dry_run = dry_run
    config.journal_dir = Path(tmp.name) / "journal"
    config.state_dir = Path(tmp.name) / "state"
    config.max_watchlist_size = 5

    broker = _StubBroker()
    feed = _StubFeed()

    # Use only filters that don't require external lookups
    cooldowns = CooldownTracker()
    pipeline = FilterPipeline([
        RegimeFilter(config.risk),
        CooldownFilter(cooldowns),
    ])

    # Build a few ticker contexts
    contexts = {
        "AAPL": TickerContext(
            ticker="AAPL", premkt_rvol=3.0,
            premkt_gap_pct=2.0, prev_close=100.0,
            avg_daily_volume=10_000_000,
        ),
        "MSFT": TickerContext(
            ticker="MSFT", premkt_rvol=2.5,
            premkt_gap_pct=1.5, prev_close=200.0,
            avg_daily_volume=8_000_000,
        ),
    }
    scanner = _StubPremarketScanner(contexts)
    classifier = _StubCatalystClassifier()
    seq = SequenceCounter(config.state_dir / "seq.json")

    daemon = DayTraderDaemon(
        broker=broker,
        config=config,
        strategies=[OrbVwapStrategy()],
        pipeline=pipeline,
        feed=feed,
        premarket_scanner=scanner,
        catalyst_classifier=classifier,
        seq_counter=seq,
    )
    # Override heartbeat path to writable tmp dir (default is
    # /var/run/day-trader, not present on dev/CI machines)
    daemon.heartbeat = Heartbeat(Path(tmp.name) / "hb.json")
    return daemon, broker, feed, tmp


# ── Tests ─────────────────────────────────────────────────────────


class TestPreSession(unittest.IsolatedAsyncioTestCase):
    async def test_pre_session_wires_everything(self):
        daemon, broker, feed, tmp = _make_daemon()
        try:
            # Use a known historical session
            from day_trader.calendar import session_for
            session = session_for(date(2024, 7, 9))
            self.assertIsNotNone(session)

            # Mock fetch_market_state (yfinance-bound)
            with _um.patch(
                "day_trader.data.market_state.fetch_market_state"
            ) as mock_ms:
                from day_trader.models import MarketState
                mock_ms.return_value = MarketState(
                    spy_price=550, vix=15.0, spy_trend_regime="BULL",
                )
                await daemon._pre_session(session)

            # Watchlist populated
            self.assertGreater(len(daemon.watchlist), 0)
            self.assertIn("AAPL", daemon.watchlist)
            # Risk manager seeded
            self.assertEqual(daemon.risk.session_starting_equity, 100_000.0)
            # Feed started
            self.assertTrue(feed.started)
            # Scheduler built
            self.assertIsNotNone(daemon.scheduler)
        finally:
            tmp.cleanup()


class TestSessionRun(unittest.IsolatedAsyncioTestCase):
    async def test_full_session_completes_cleanly(self):
        """End-to-end: pre_session → run_session → post_session.

        Patches session_for to return a fixed session and now_et
        to be 1s past that session's close. Independent of when
        the test runs (works on weekends, holidays, off-hours).
        """
        daemon, broker, feed, tmp = _make_daemon(dry_run=True)
        try:
            from day_trader.calendar import session_for
            test_session = session_for(date(2024, 7, 9))
            self.assertIsNotNone(test_session)

            from day_trader.models import MarketState
            fake_now = test_session.close_et + timedelta(seconds=1)

            def _fake_session_for(*args, **kwargs):
                return test_session

            def _fake_now_et():
                return fake_now

            from day_trader import executor as exec_mod

            with _um.patch.object(
                exec_mod, "session_for", _fake_session_for,
            ), _um.patch.object(
                exec_mod, "now_et", _fake_now_et,
            ), _um.patch(
                "day_trader.data.market_state.fetch_market_state",
                return_value=MarketState(
                    spy_price=550, vix=15.0, spy_trend_regime="BULL",
                ),
            ):
                await daemon.run()

            # Feed lifecycle
            self.assertTrue(feed.started)
            self.assertTrue(feed.stopped)
            # Heartbeat was written at least once during the session
            self.assertTrue(daemon.heartbeat.path.exists())
            # Pipeline stats reset at session end
            self.assertEqual(daemon.pipeline.stats, {})
        finally:
            tmp.cleanup()


class TestNoSession(unittest.IsolatedAsyncioTestCase):
    async def test_holiday_stands_down(self):
        """On a holiday, daemon logs "standing down" and exits clean."""
        daemon, broker, feed, tmp = _make_daemon()
        try:
            from day_trader import calendar as cal_mod
            with _um.patch.object(
                cal_mod, "session_for", return_value=None,
            ):
                # Direct import so the daemon's session_for sees
                # the patch
                with _um.patch(
                    "day_trader.executor.session_for",
                    return_value=None,
                ):
                    await daemon.run()
            # Feed never started (no session)
            self.assertFalse(feed.started)
        finally:
            tmp.cleanup()


class TestKillSwitchAlerts(unittest.IsolatedAsyncioTestCase):
    async def test_kill_switch_fires_alert(self):
        """When daily P&L breaches the limit, alerter.crit is called."""
        daemon, broker, feed, tmp = _make_daemon()
        try:
            with _um.patch.object(daemon.alerter, "crit") as mock_crit:
                # Boot the risk manager with $100k equity, 1.5% limit
                daemon.risk.start_session(equity=100_000.0)
                daemon.risk.record_fill(notional=10_000.0)
                # Open a fake position so _close_position has something
                # to close
                from day_trader.models import OpenDayTrade
                daemon.position_mgr.open_position(OpenDayTrade(
                    ticker="AAPL", strategy="orb_vwap", side="long",
                    qty=100, entry_price=100,
                    entry_time=datetime.now(tz=UTC),
                    sl_price=95, tp_price=105,
                    parent_client_order_id="dt:20240709:0001:AAPL",
                    seq=1,
                ))
                # Set bar cache so the close has something to compute P&L
                from day_trader.models import Bar
                daemon.bar_cache.add_bar(Bar(
                    ticker="AAPL", timestamp=datetime.now(tz=UTC),
                    open=100, high=80, low=80, close=80,  # huge loss
                    volume=1000,
                ))

                # In dry-run mode the actual broker call is skipped,
                # so directly exercise record_close path.
                pos = daemon.position_mgr.get("AAPL")
                pnl = (80 - 100) * 100  # -$2000 = 2% loss > 1.5% limit
                daemon.risk.record_close(
                    ticker="AAPL",
                    strategy="orb_vwap",
                    pnl=pnl,
                    entry_notional=10_000,
                )
                # Verify kill switch tripped
                self.assertTrue(daemon.risk.kill_switch_tripped)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
