"""Tests for SymbolLock — swing/day same-symbol exclusion.

These tests use a duck-typed FakeBroker (SimpleNamespace-based) so
we don't depend on alpaca-py being installed. The lock only reads
broker state — never submits orders — so this is sufficient.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from day_trader.symbol_locks import (
    REASON_DAYTRADE_OPEN,
    REASON_DAYTRADE_PENDING,
    REASON_SWING_OPEN_ORDER,
    REASON_SWING_POSITION,
    SymbolLock,
)


def _portfolio(positions: dict) -> SimpleNamespace:
    """Build a duck-typed PortfolioSnapshot."""
    return SimpleNamespace(
        equity=100_000.0,
        cash=50_000.0,
        positions=positions,
    )


def _order(symbol: str, client_order_id: str = "", order_id: str = "x"):
    """Build a duck-typed Alpaca order object."""
    return SimpleNamespace(
        id=order_id, symbol=symbol, client_order_id=client_order_id,
    )


def _make_broker(positions: dict, open_orders: list) -> SimpleNamespace:
    broker = SimpleNamespace()
    broker.get_portfolio = lambda: _portfolio(positions)
    broker.list_open_orders = lambda: open_orders
    return broker


# ── Locking by source ─────────────────────────────────────────────


class TestLocking(unittest.TestCase):
    def test_swing_position_locks(self):
        broker = _make_broker(
            positions={"AAPL": {"qty": 100, "avg_entry": 150}},
            open_orders=[],
        )
        lock = SymbolLock(broker)
        v = lock.is_locked("AAPL")
        self.assertTrue(v.locked)
        self.assertEqual(v.reason, REASON_SWING_POSITION)

    def test_swing_open_order_locks(self):
        broker = _make_broker(
            positions={},
            open_orders=[_order("MSFT", client_order_id="alpaca-auto-1")],
        )
        lock = SymbolLock(broker)
        v = lock.is_locked("MSFT")
        self.assertTrue(v.locked)
        self.assertEqual(v.reason, REASON_SWING_OPEN_ORDER)

    def test_daytrade_pending_order_locks(self):
        broker = _make_broker(
            positions={},
            open_orders=[
                _order("TSLA", client_order_id="dt:20260428:0001:TSLA"),
            ],
        )
        lock = SymbolLock(broker)
        v = lock.is_locked("TSLA")
        self.assertTrue(v.locked)
        self.assertEqual(v.reason, REASON_DAYTRADE_PENDING)

    def test_daytrade_position_locks(self):
        # Position exists AND there's a tagged order on the same
        # ticker → treated as daytrade-owned.
        broker = _make_broker(
            positions={"NVDA": {"qty": 50}},
            open_orders=[
                _order("NVDA", client_order_id="dt:20260428:0001:NVDA"),
            ],
        )
        lock = SymbolLock(broker)
        v = lock.is_locked("NVDA")
        self.assertTrue(v.locked)
        # Either the position-lock or pending-order lock is fine here;
        # both are correct outcomes.
        self.assertIn(v.reason, (REASON_DAYTRADE_OPEN, REASON_DAYTRADE_PENDING))

    def test_unrelated_ticker_unlocked(self):
        broker = _make_broker(
            positions={"AAPL": {"qty": 100}},
            open_orders=[_order("MSFT", client_order_id="alpaca-1")],
        )
        lock = SymbolLock(broker)
        v = lock.is_locked("GOOG")
        self.assertFalse(v.locked)
        self.assertEqual(v.reason, "")

    def test_lowercase_input_normalized(self):
        broker = _make_broker(
            positions={"AAPL": {"qty": 100}},
            open_orders=[],
        )
        lock = SymbolLock(broker)
        self.assertTrue(lock.is_locked("aapl").locked)


# ── Symmetric direction (swing rejects day-trader's tickers) ─────


class TestIsHeldByDayTrader(unittest.TestCase):
    def test_swing_position_returns_false(self):
        # Swing's own position should NOT be flagged as day-trader-held.
        broker = _make_broker(
            positions={"AAPL": {"qty": 100}},
            open_orders=[],
        )
        lock = SymbolLock(broker)
        self.assertFalse(lock.is_held_by_day_trader("AAPL"))

    def test_daytrade_pending_returns_true(self):
        broker = _make_broker(
            positions={},
            open_orders=[
                _order("TSLA", client_order_id="dt:20260428:0001:TSLA"),
            ],
        )
        lock = SymbolLock(broker)
        self.assertTrue(lock.is_held_by_day_trader("TSLA"))

    def test_unrelated_returns_false(self):
        broker = _make_broker(
            positions={"AAPL": {"qty": 100}},
            open_orders=[
                _order("TSLA", client_order_id="dt:20260428:0001:TSLA"),
            ],
        )
        lock = SymbolLock(broker)
        self.assertFalse(lock.is_held_by_day_trader("GOOG"))


# ── Refresh / staleness ──────────────────────────────────────────


class TestRefresh(unittest.TestCase):
    def test_refresh_picks_up_new_swing_position(self):
        positions = {}
        broker = _make_broker(positions, [])
        lock = SymbolLock(broker, refresh_ttl_seconds=1)
        self.assertFalse(lock.is_locked("AAPL").locked)
        # Simulate swing bot adding a position
        positions["AAPL"] = {"qty": 100}
        # Force a refresh (TTL is short but we want determinism)
        lock.refresh()
        self.assertTrue(lock.is_locked("AAPL").locked)

    def test_locked_tickers_returns_union(self):
        broker = _make_broker(
            positions={"AAPL": {"qty": 100}, "MSFT": {"qty": 50}},
            open_orders=[
                _order("TSLA", client_order_id="dt:20260428:0001:TSLA"),
                _order("GOOG", client_order_id="alpaca-auto-1"),
            ],
        )
        lock = SymbolLock(broker)
        locked = lock.locked_tickers()
        self.assertEqual(locked, frozenset({"AAPL", "MSFT", "TSLA", "GOOG"}))


# ── Critical safety contracts ────────────────────────────────────


class TestSafetyContracts(unittest.TestCase):
    """The plan's review explicitly required these scenarios."""

    def test_swing_holds_blocks_day_entry(self):
        # "test_symbol_lock_swing_holds_blocks_day" from the plan
        broker = _make_broker(
            positions={"AAPL": {"qty": 100}},
            open_orders=[],
        )
        lock = SymbolLock(broker)
        v = lock.is_locked("AAPL")
        self.assertTrue(v.locked)
        self.assertEqual(v.reason, REASON_SWING_POSITION)

    def test_day_holds_blocks_swing_entry(self):
        # "test_symbol_lock_day_holds_blocks_swing" from the plan.
        # Swing risk manager would call is_held_by_day_trader("MSFT")
        # and reject the entry.
        broker = _make_broker(
            positions={},
            open_orders=[
                _order("MSFT", client_order_id="dt:20260428:0007:MSFT"),
            ],
        )
        lock = SymbolLock(broker)
        self.assertTrue(lock.is_held_by_day_trader("MSFT"))


if __name__ == "__main__":
    unittest.main()
