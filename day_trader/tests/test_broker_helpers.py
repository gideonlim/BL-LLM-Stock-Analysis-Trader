"""Tests for close_tagged_daytrade_qty.

This is the critical safety helper. Failures here would cause the
EOD watchdog to either close swing positions accidentally, or fail
to close day-trade positions at all.
"""

from __future__ import annotations

import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

from trading_bot_bl.models import OrderResult

from day_trader.broker_helpers import (
    close_tagged_daytrade_qty,
    list_tagged_daytrade_orders,
)


def _order(
    symbol: str,
    client_order_id: str = "",
    order_id: str = "alpaca-id-1",
    status: str = "accepted",
    parent_id: str = "",
):
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        client_order_id=client_order_id,
        status=status,
        parent_id=parent_id,
    )


class FakeBroker:
    """Tracks all interactions so tests can assert on mechanics."""

    def __init__(self, open_orders=None):
        self.open_orders = list(open_orders or [])
        self.cancelled: list[str] = []
        self.market_orders: list[dict] = []
        self.close_position_calls: list[str] = []
        self.order_status: dict[str, str] = {
            o.id: o.status for o in self.open_orders
        }
        self.next_market_id = 1

    def list_open_orders(self):
        return self.open_orders

    def cancel_order_by_id(self, order_id: str):
        self.cancelled.append(order_id)
        # Simulate the cancel taking effect immediately
        self.order_status[order_id] = "canceled"

    def get_order_by_id(self, order_id: str):
        return SimpleNamespace(
            id=order_id, status=self.order_status.get(order_id, "unknown"),
        )

    def submit_market_order(self, **kwargs):
        # Match AlpacaBroker.submit_market_order's signature
        oid = f"market-{self.next_market_id}"
        self.next_market_id += 1
        self.market_orders.append(kwargs)
        return OrderResult(
            ticker=kwargs["ticker"],
            order_id=oid,
            client_order_id=kwargs.get("client_order_id", ""),
            status="submitted",
            side=kwargs["side"],
            notional=0.0,
        )

    def close_position(self, ticker: str):
        # If this gets called, the safety helper has FAILED.
        self.close_position_calls.append(ticker)
        raise AssertionError(
            f"close_position({ticker!r}) called — must never be invoked "
            f"by close_tagged_daytrade_qty"
        )


# ── list_tagged_daytrade_orders ──────────────────────────────────


class TestListTaggedOrders(unittest.TestCase):
    def test_filters_by_dt_prefix(self):
        broker = FakeBroker(open_orders=[
            _order("AAPL", "dt:20260428:0001:AAPL", "id1"),
            _order("MSFT", "alpaca-auto-1", "id2"),
            _order("TSLA", "dt:20260428:0002:TSLA", "id3"),
        ])
        tagged = list_tagged_daytrade_orders(broker)
        ids = sorted(o.id for o in tagged)
        self.assertEqual(ids, ["id1", "id3"])

    def test_filters_by_ticker(self):
        broker = FakeBroker(open_orders=[
            _order("AAPL", "dt:20260428:0001:AAPL", "id1"),
            _order("MSFT", "dt:20260428:0002:MSFT", "id2"),
        ])
        tagged = list_tagged_daytrade_orders(broker, ticker="AAPL")
        self.assertEqual([o.id for o in tagged], ["id1"])


# ── close_tagged_daytrade_qty ────────────────────────────────────


class TestCloseTaggedDaytradeQty(unittest.TestCase):
    """The plan-required safety tests."""

    def test_cancels_legs_first_then_submits_close(self):
        """test_close_tagged_cancels_legs_first from the plan."""
        parent = "dt:20260428:0001:AAPL"
        # Parent + SL leg + TP leg, all tagged
        broker = FakeBroker(open_orders=[
            _order("AAPL", parent, "parent-id"),
            _order("AAPL", "", "sl-id", parent_id=parent),
            _order("AAPL", "", "tp-id", parent_id=parent),
        ])
        result = close_tagged_daytrade_qty(
            broker, "AAPL", qty=10,
            side="long", parent_client_order_id=parent,
        )
        # All three orders should be cancelled
        self.assertEqual(
            sorted(broker.cancelled), ["parent-id", "sl-id", "tp-id"],
        )
        # A single market close was submitted
        self.assertEqual(len(broker.market_orders), 1)
        mkt = broker.market_orders[0]
        # Side flips: we held long, so close is sell
        self.assertEqual(mkt["side"], "sell")
        self.assertEqual(mkt["qty"], 10)
        self.assertEqual(mkt["ticker"], "AAPL")
        # The close-order is tagged with :exit suffix
        self.assertEqual(mkt["client_order_id"], parent + ":exit")
        # Result reflects success
        self.assertTrue(result.succeeded)
        # close_position must NEVER be called
        self.assertEqual(broker.close_position_calls, [])

    def test_does_not_touch_untagged_swing_orders(self):
        """test_close_tagged_does_not_touch_untagged from the plan."""
        parent = "dt:20260428:0001:AAPL"
        broker = FakeBroker(open_orders=[
            _order("AAPL", parent, "parent-id"),
            _order("AAPL", "", "sl-id", parent_id=parent),
            # An untagged swing-bracket on the SAME ticker — must be untouched
            _order("AAPL", "swing-bracket-1", "swing-id"),
        ])
        close_tagged_daytrade_qty(
            broker, "AAPL", qty=10,
            side="long", parent_client_order_id=parent,
        )
        # Only the dt:-tagged orders are cancelled
        self.assertIn("parent-id", broker.cancelled)
        self.assertIn("sl-id", broker.cancelled)
        self.assertNotIn("swing-id", broker.cancelled)
        # And close_position is never called
        self.assertEqual(broker.close_position_calls, [])

    def test_short_position_close_buys(self):
        parent = "dt:20260428:0002:TSLA"
        broker = FakeBroker(open_orders=[
            _order("TSLA", parent, "p"),
        ])
        close_tagged_daytrade_qty(
            broker, "TSLA", qty=5,
            side="short", parent_client_order_id=parent,
        )
        self.assertEqual(broker.market_orders[0]["side"], "buy")

    def test_invalid_qty_returns_error(self):
        broker = FakeBroker()
        result = close_tagged_daytrade_qty(
            broker, "AAPL", qty=0,
            side="long",
            parent_client_order_id="dt:20260428:0001:AAPL",
        )
        self.assertIn("qty must be positive", result.error)
        self.assertEqual(broker.market_orders, [])

    def test_invalid_side_returns_error(self):
        broker = FakeBroker()
        result = close_tagged_daytrade_qty(
            broker, "AAPL", qty=10,
            side="bogus",
            parent_client_order_id="dt:20260428:0001:AAPL",
        )
        self.assertIn("side must be", result.error)
        self.assertEqual(broker.market_orders, [])

    def test_non_dt_parent_id_returns_error(self):
        broker = FakeBroker()
        result = close_tagged_daytrade_qty(
            broker, "AAPL", qty=10,
            side="long",
            parent_client_order_id="alpaca-auto-12345",
        )
        self.assertIn("must be a day-trade tag", result.error)
        self.assertEqual(broker.market_orders, [])

    def test_no_open_legs_still_submits_close(self):
        # Edge case: by the time the EOD watchdog runs, the bracket
        # legs may have already filled (e.g. SL hit during the day).
        # We should still submit the qty-close to ensure flat.
        parent = "dt:20260428:0001:AAPL"
        broker = FakeBroker(open_orders=[])  # nothing left
        result = close_tagged_daytrade_qty(
            broker, "AAPL", qty=10,
            side="long", parent_client_order_id=parent,
        )
        # Nothing to cancel
        self.assertEqual(broker.cancelled, [])
        # But a market close should still have been attempted
        self.assertEqual(len(broker.market_orders), 1)
        self.assertTrue(result.succeeded)


if __name__ == "__main__":
    unittest.main()
