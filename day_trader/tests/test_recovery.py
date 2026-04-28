"""Tests for recovery.reconcile() — the journal/orders/positions
three-way reconciliation. Failures here mean the daemon either
miss-counts open exposure or wrongly enters incident mode."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from trading_bot_bl.models import JournalEntry

from day_trader.recovery import ReconcileIncident, reconcile


def _portfolio(positions: dict) -> SimpleNamespace:
    return SimpleNamespace(
        equity=100_000.0,
        cash=50_000.0,
        positions=positions,
    )


def _order(symbol: str, client_order_id: str, order_id: str = "alpaca-1"):
    return SimpleNamespace(
        id=order_id, symbol=symbol, client_order_id=client_order_id,
        status="accepted",
    )


def _make_broker(positions: dict, open_orders: list):
    broker = SimpleNamespace()
    broker.get_portfolio = lambda: _portfolio(positions)
    broker.list_open_orders = lambda: open_orders
    return broker


def _write_open_daytrade(
    journal_dir: Path,
    ticker: str,
    qty: float,
    notional: float,
    parent_id: str,
    status: str = "open",
):
    """Write a journal entry simulating an open day-trade.

    NOTE: trade_id must NOT contain colons — those are reserved
    characters in Windows filenames (alternate data stream
    syntax) and would silently break the test on Windows hosts.
    """
    journal_dir.mkdir(parents=True, exist_ok=True)
    # Strip colons from parent_id when used in trade_id
    safe_id_suffix = parent_id.replace(":", "_")
    entry = JournalEntry(
        trade_id=f"{ticker}_{safe_id_suffix}",
        ticker=ticker,
        strategy="ORB_VWAP",
        side="long",
        entry_order_id=parent_id,
        entry_signal_price=100.0,
        entry_fill_price=100.0,
        entry_qty=qty,
        entry_notional=notional,
        status=status,
        opened_at="2026-04-28T09:35:00",
        trade_type="daytrade",
    )
    from dataclasses import asdict
    path = journal_dir / f"{entry.trade_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(entry), f)


class TestCleanReconcile(unittest.TestCase):
    """Happy path — journal, orders, and positions all agree."""

    def test_no_open_anything_is_clean(self):
        broker = _make_broker(positions={}, open_orders=[])
        with TemporaryDirectory() as tmp:
            r = reconcile(broker, Path(tmp))
        self.assertTrue(r.is_clean)
        self.assertEqual(r.open_journal_entries, [])
        self.assertEqual(r.open_notional, 0.0)
        self.assertEqual(r.incidents, [])

    def test_swing_position_only_is_clean(self):
        # The day-trader doesn't care about swing-only state.
        broker = _make_broker(
            positions={"AAPL": {"qty": 100, "avg_entry": 150}},
            open_orders=[
                _order("AAPL", "alpaca-swing-1"),  # untagged → swing
            ],
        )
        with TemporaryDirectory() as tmp:
            r = reconcile(broker, Path(tmp))
        self.assertTrue(r.is_clean)

    def test_journal_open_with_matching_position_clean(self):
        parent = "dt:20260428:0001:AAPL"
        broker = _make_broker(
            positions={"AAPL": {"qty": 10}},
            # Bracket legs may or may not still be open — we test
            # the case where they're filled (so no tagged orders left)
            # but the journal still says open. Position match → clean.
            open_orders=[],
        )
        with TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "journal"
            _write_open_daytrade(jdir, "AAPL", qty=10, notional=1000.0,
                                 parent_id=parent)
            r = reconcile(broker, jdir)
        self.assertTrue(r.is_clean,
                        f"expected clean, got incidents: {r.incidents}")
        self.assertEqual(len(r.open_journal_entries), 1)
        self.assertAlmostEqual(r.open_notional, 1000.0)

    def test_journal_open_with_tagged_orders_clean(self):
        parent = "dt:20260428:0002:MSFT"
        broker = _make_broker(
            positions={"MSFT": {"qty": 5}},
            open_orders=[_order("MSFT", parent, "p1")],
        )
        with TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "journal"
            _write_open_daytrade(jdir, "MSFT", qty=5, notional=500.0,
                                 parent_id=parent)
            r = reconcile(broker, jdir)
        self.assertTrue(r.is_clean)


class TestIncidents(unittest.TestCase):
    """Mismatch cases — recovery enters INCIDENT MODE."""

    def test_orphan_journal_entry(self):
        # Journal says we have an open AAPL day-trade, but Alpaca
        # has nothing for that ticker. Likely: position was closed
        # externally while daemon was down.
        parent = "dt:20260428:0001:AAPL"
        broker = _make_broker(positions={}, open_orders=[])
        with TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "journal"
            _write_open_daytrade(jdir, "AAPL", qty=10, notional=1000.0,
                                 parent_id=parent)
            r = reconcile(broker, jdir)
        self.assertFalse(r.is_clean)
        self.assertEqual(len(r.incidents), 1)
        self.assertEqual(r.incidents[0].kind, "orphan_journal")
        self.assertEqual(r.incidents[0].ticker, "AAPL")

    def test_orphan_tagged_order(self):
        # Tagged order on Alpaca with no journal counterpart. Could
        # mean a previous daemon crashed after submitting an order
        # but before writing the journal entry.
        parent = "dt:20260428:0007:TSLA"
        broker = _make_broker(
            positions={},
            open_orders=[_order("TSLA", parent, "p1")],
        )
        with TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "journal"
            jdir.mkdir(parents=True, exist_ok=True)
            r = reconcile(broker, jdir)
        self.assertFalse(r.is_clean)
        kinds = {i.kind for i in r.incidents}
        self.assertIn("orphan_order", kinds)

    def test_qty_mismatch_journal_exceeds_position(self):
        # Journal claims 100 shares of AAPL day-trade, but broker
        # only shows net position of 50. Something has been closed
        # without the daemon's knowledge.
        parent = "dt:20260428:0001:AAPL"
        broker = _make_broker(
            positions={"AAPL": {"qty": 50}},
            open_orders=[],
        )
        with TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "journal"
            _write_open_daytrade(jdir, "AAPL", qty=100, notional=10000.0,
                                 parent_id=parent)
            r = reconcile(broker, jdir)
        self.assertFalse(r.is_clean)
        self.assertEqual(r.incidents[0].kind, "qty_mismatch")

    def test_swing_only_does_not_trigger_incident(self):
        # A swing position + untagged swing order on AAPL with NO
        # day-trade journal entries should be clean — recovery
        # only cares about day-trade exposure.
        broker = _make_broker(
            positions={"AAPL": {"qty": 100}},
            open_orders=[_order("AAPL", "alpaca-swing-1")],
        )
        with TemporaryDirectory() as tmp:
            r = reconcile(broker, Path(tmp))
        self.assertTrue(r.is_clean)


class TestOpenNotional(unittest.TestCase):
    def test_open_notional_sums_journal(self):
        broker = _make_broker(
            positions={"AAPL": {"qty": 10}, "MSFT": {"qty": 5}},
            open_orders=[],
        )
        with TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "journal"
            _write_open_daytrade(jdir, "AAPL", qty=10, notional=1500.0,
                                 parent_id="dt:20260428:0001:AAPL")
            _write_open_daytrade(jdir, "MSFT", qty=5, notional=2000.0,
                                 parent_id="dt:20260428:0002:MSFT")
            r = reconcile(broker, jdir)
        self.assertTrue(r.is_clean)
        self.assertAlmostEqual(r.open_notional, 3500.0)


class TestSummary(unittest.TestCase):
    def test_clean_summary(self):
        broker = _make_broker(positions={}, open_orders=[])
        with TemporaryDirectory() as tmp:
            r = reconcile(broker, Path(tmp))
        self.assertIn("clean", r.summary())

    def test_incident_summary_mentions_count(self):
        parent = "dt:20260428:0001:AAPL"
        broker = _make_broker(positions={}, open_orders=[])
        with TemporaryDirectory() as tmp:
            jdir = Path(tmp) / "journal"
            _write_open_daytrade(jdir, "AAPL", qty=10, notional=1000.0,
                                 parent_id=parent)
            r = reconcile(broker, jdir)
        s = r.summary()
        self.assertIn("INCIDENT", s)
        self.assertIn("orphan_journal", s)


if __name__ == "__main__":
    unittest.main()
