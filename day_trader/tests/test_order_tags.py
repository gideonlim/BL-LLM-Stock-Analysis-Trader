"""Tests for the dt: client_order_id schema and SequenceCounter."""

from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from day_trader.order_tags import (
    SequenceCounter,
    is_day_trade_id,
    make_exit_order_id,
    make_order_id,
    parse_order_id,
)


class TestMakeOrderId(unittest.TestCase):
    def test_basic_format(self):
        oid = make_order_id(7, "AAPL", today=date(2026, 4, 28))
        self.assertEqual(oid, "dt:20260428:0007:AAPL")

    def test_seq_zero_padded(self):
        oid = make_order_id(0, "MSFT", today=date(2026, 1, 1))
        self.assertEqual(oid, "dt:20260101:0000:MSFT")

    def test_seq_at_max(self):
        oid = make_order_id(9999, "TSLA", today=date(2026, 12, 31))
        self.assertEqual(oid, "dt:20261231:9999:TSLA")

    def test_dotted_ticker_allowed(self):
        oid = make_order_id(1, "BRK.B", today=date(2026, 4, 28))
        self.assertEqual(oid, "dt:20260428:0001:BRK.B")

    def test_seq_negative_rejected(self):
        with self.assertRaises(ValueError):
            make_order_id(-1, "AAPL", today=date(2026, 4, 28))

    def test_seq_overflow_rejected(self):
        with self.assertRaises(ValueError):
            make_order_id(10000, "AAPL", today=date(2026, 4, 28))

    def test_lowercase_ticker_rejected(self):
        with self.assertRaises(ValueError):
            make_order_id(1, "aapl", today=date(2026, 4, 28))

    def test_empty_ticker_rejected(self):
        with self.assertRaises(ValueError):
            make_order_id(1, "", today=date(2026, 4, 28))

    def test_under_alpaca_length_limit(self):
        # Alpaca client_order_id limit is 128 chars. Worst case
        # ticker is ~5 chars; ours come out around 24 chars.
        oid = make_order_id(9999, "BRK.B", today=date(2099, 12, 31))
        self.assertLess(len(oid), 32)


class TestExitOrderId(unittest.TestCase):
    def test_appends_exit_suffix(self):
        parent = "dt:20260428:0007:AAPL"
        self.assertEqual(make_exit_order_id(parent), "dt:20260428:0007:AAPL:exit")

    def test_non_dt_parent_rejected(self):
        with self.assertRaises(ValueError):
            make_exit_order_id("alpaca-server-gen-12345")

    def test_double_exit_rejected(self):
        with self.assertRaises(ValueError):
            make_exit_order_id("dt:20260428:0007:AAPL:exit")


class TestIsDayTradeId(unittest.TestCase):
    def test_dt_prefix_recognized(self):
        self.assertTrue(is_day_trade_id("dt:20260428:0001:AAPL"))

    def test_swing_id_not_recognized(self):
        self.assertFalse(is_day_trade_id("alpaca-auto-12345"))

    def test_empty_returns_false(self):
        self.assertFalse(is_day_trade_id(""))

    def test_none_returns_false(self):
        self.assertFalse(is_day_trade_id(None))


class TestParseOrderId(unittest.TestCase):
    def test_round_trip_parent(self):
        oid = make_order_id(42, "NVDA", today=date(2026, 5, 1))
        parsed = parse_order_id(oid)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.date_str, "20260501")
        self.assertEqual(parsed.seq, 42)
        self.assertEqual(parsed.ticker, "NVDA")
        self.assertFalse(parsed.is_exit)
        self.assertEqual(parsed.parent_id, oid)

    def test_round_trip_exit(self):
        parent = make_order_id(7, "AAPL", today=date(2026, 4, 28))
        exit_id = make_exit_order_id(parent)
        parsed = parse_order_id(exit_id)
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed.is_exit)
        self.assertEqual(parsed.parent_id, parent)

    def test_dotted_ticker_round_trips(self):
        oid = make_order_id(1, "BRK.B", today=date(2026, 4, 28))
        parsed = parse_order_id(oid)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.ticker, "BRK.B")

    def test_non_dt_returns_none(self):
        self.assertIsNone(parse_order_id("alpaca-server-12345"))

    def test_malformed_returns_none(self):
        self.assertIsNone(parse_order_id("dt:not-a-date:0007:AAPL"))
        self.assertIsNone(parse_order_id("dt:20260428:abc:AAPL"))
        self.assertIsNone(parse_order_id("dt:20260428:0007:lowercase"))
        self.assertIsNone(parse_order_id("dt:20260428:0007"))  # missing ticker


class TestSequenceCounter(unittest.TestCase):
    def test_starts_at_zero(self):
        with TemporaryDirectory() as tmp:
            c = SequenceCounter(Path(tmp) / "seq.json")
            self.assertEqual(c.peek(today=date(2026, 4, 28)), 0)

    def test_monotonic_within_day(self):
        with TemporaryDirectory() as tmp:
            c = SequenceCounter(Path(tmp) / "seq.json")
            today = date(2026, 4, 28)
            seqs = [c.next(today=today) for _ in range(5)]
            self.assertEqual(seqs, [0, 1, 2, 3, 4])
            self.assertEqual(c.peek(today=today), 5)

    def test_resets_on_new_day(self):
        with TemporaryDirectory() as tmp:
            c = SequenceCounter(Path(tmp) / "seq.json")
            d1 = date(2026, 4, 28)
            d2 = date(2026, 4, 29)
            self.assertEqual(c.next(today=d1), 0)
            self.assertEqual(c.next(today=d1), 1)
            # New day → counter resets
            self.assertEqual(c.next(today=d2), 0)
            self.assertEqual(c.next(today=d2), 1)

    def test_persists_across_instances(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.json"
            today = date(2026, 4, 28)
            # Process 1
            c1 = SequenceCounter(path)
            c1.next(today=today)
            c1.next(today=today)
            c1.next(today=today)
            # Process 2 (simulated daemon restart mid-session)
            c2 = SequenceCounter(path)
            seq = c2.next(today=today)
            self.assertEqual(seq, 3, "must continue from where prev left off")

    def test_corrupt_state_resets_safely(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.json"
            path.write_text("not valid json {")
            c = SequenceCounter(path)
            # Should NOT crash; should treat as fresh-day reset
            seq = c.next(today=date(2026, 4, 28))
            self.assertEqual(seq, 0)

    def test_state_format_atomic(self):
        # After a write, no .tmp file should remain
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "seq.json"
            c = SequenceCounter(path)
            c.next(today=date(2026, 4, 28))
            self.assertTrue(path.exists())
            tmp_files = list(Path(tmp).glob("seq.json.tmp"))
            self.assertEqual(tmp_files, [], "tmp file leaked after rename")
            data = json.loads(path.read_text())
            self.assertEqual(data, {"date": "20260428", "next": 1})


if __name__ == "__main__":
    unittest.main()
