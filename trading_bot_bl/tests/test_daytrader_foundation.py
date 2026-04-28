"""Foundation tests for the day-trader module.

Covers v1 prerequisites that must land before any day_trader/ code:
- broker.client_order_id plumbing through submit_bracket_order
  and submit_market_order
- OrderResult.client_order_id field
- JournalEntry.trade_type field (defaults to "swing", round-trips,
  graceful default for legacy journal files without the field)

Run with:
    python -m pytest trading_bot_bl/tests/test_daytrader_foundation.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock as _um
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

# Mock alpaca SDK before importing broker.py / models.py
_alpaca_mock = _um.MagicMock()
sys.modules["alpaca"] = _alpaca_mock
sys.modules["alpaca.trading"] = _alpaca_mock.trading
sys.modules["alpaca.trading.client"] = _alpaca_mock.trading.client
sys.modules["alpaca.trading.requests"] = _alpaca_mock.trading.requests
sys.modules["alpaca.trading.enums"] = _alpaca_mock.trading.enums
sys.modules["alpaca.data"] = _alpaca_mock.data
sys.modules["alpaca.data.historical"] = _alpaca_mock.data.historical
sys.modules["alpaca.data.requests"] = _alpaca_mock.data.requests

from trading_bot_bl.models import JournalEntry, OrderResult


def _setup_broker_with_simplenamespace_requests(test_case):
    """Reload trading_bot_bl.broker, install SimpleNamespace
    side_effects on the alpaca request classes, and register cleanup
    so mock state doesn't leak between tests.

    Raw MagicMocks auto-create sub-mocks on attribute access, which
    defeats `getattr(req, "client_order_id", "")` checks. SimpleNamespace
    returns the actual values we passed.

    Returns the (broker_module, broker_instance) pair.
    """
    import importlib

    import trading_bot_bl.broker as broker_module

    importlib.reload(broker_module)

    request_classes = [
        broker_module.MarketOrderRequest,
        broker_module.LimitOrderRequest,
        broker_module.TakeProfitRequest,
        broker_module.StopLossRequest,
    ]
    for cls in request_classes:
        cls.side_effect = lambda **kw: SimpleNamespace(**kw)
        # Reset call_args/side_effect between tests to prevent leaks.
        test_case.addCleanup(
            lambda c=cls: setattr(c, "side_effect", None)
        )
        test_case.addCleanup(cls.reset_mock)

    config = _um.MagicMock()
    config.api_key = "k"
    config.api_secret = "s"
    config.paper = True
    config.validate.return_value = None

    broker = broker_module.AlpacaBroker(config)
    broker._client = _um.MagicMock()
    return broker_module, broker


# ── OrderResult.client_order_id ──────────────────────────────────


class TestOrderResultClientOrderId(unittest.TestCase):
    def test_client_order_id_defaults_to_empty(self):
        result = OrderResult(ticker="AAPL")
        self.assertEqual(result.client_order_id, "")

    def test_client_order_id_round_trips_through_dict(self):
        result = OrderResult(
            ticker="AAPL",
            order_id="abc123",
            client_order_id="dt:20260428:0001:AAPL",
            status="submitted",
        )
        d = asdict(result)
        self.assertEqual(d["client_order_id"], "dt:20260428:0001:AAPL")


# ── JournalEntry.trade_type ──────────────────────────────────────


class TestJournalEntryTradeType(unittest.TestCase):
    def _minimal_entry(self, **overrides) -> JournalEntry:
        kwargs = {
            "trade_id": "AAPL_abc12345",
            "ticker": "AAPL",
            "strategy": "ORB_VWAP",
            "side": "long",
            "entry_order_id": "abc123",
            "entry_signal_price": 100.0,
        }
        kwargs.update(overrides)
        return JournalEntry(**kwargs)

    def test_trade_type_defaults_to_swing(self):
        entry = self._minimal_entry()
        self.assertEqual(entry.trade_type, "swing")

    def test_trade_type_can_be_daytrade(self):
        entry = self._minimal_entry(trade_type="daytrade")
        self.assertEqual(entry.trade_type, "daytrade")

    def test_trade_type_serializes_to_dict(self):
        entry = self._minimal_entry(trade_type="daytrade")
        d = asdict(entry)
        self.assertEqual(d["trade_type"], "daytrade")

    def test_legacy_journal_file_loads_with_swing_default(self):
        # Simulates an existing journal file written before the
        # trade_type field existed: load via journal._dict_to_entry
        # and confirm the default takes effect.
        from trading_bot_bl.journal import _dict_to_entry

        legacy_dict = {
            "trade_id": "MSFT_xyz",
            "ticker": "MSFT",
            "strategy": "EMA_Crossover",
            "side": "long",
            "entry_order_id": "xyz",
            "entry_signal_price": 200.0,
            # No trade_type — pre-day-trader file
        }
        entry = _dict_to_entry(legacy_dict)
        self.assertEqual(entry.trade_type, "swing")

    def test_round_trip_through_json(self):
        from trading_bot_bl.journal import (
            _dict_to_entry,
            _entry_to_dict,
        )

        original = self._minimal_entry(trade_type="daytrade")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trade.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_entry_to_dict(original), f)
            with open(path, encoding="utf-8") as f:
                loaded = _dict_to_entry(json.load(f))
        self.assertEqual(loaded.trade_type, "daytrade")
        self.assertEqual(loaded.ticker, original.ticker)


# ── journal.create_trade trade_type propagation ──────────────────


class TestCreateTradeTradeType(unittest.TestCase):
    """Verify the public lifecycle helper propagates trade_type
    through to the persisted JournalEntry."""

    def _call(self, **overrides) -> "JournalEntry | None":
        from trading_bot_bl.journal import create_trade

        kwargs = {
            "order_id": "alpaca-order-id-1",
            "ticker": "AAPL",
            "strategy": "ORB_VWAP",
            "side": "buy",
            "signal_price": 100.0,
            "notional": 1000.0,
            "sl_price": 95.0,
            "tp_price": 110.0,
            "composite_score": 30.0,
            "confidence": "HIGH",
            "confidence_score": 5,
        }
        kwargs.update(overrides)
        return create_trade(**kwargs)

    def test_default_trade_type_is_swing(self):
        entry = self._call()
        self.assertIsNotNone(entry)
        self.assertEqual(entry.trade_type, "swing")

    def test_explicit_daytrade(self):
        entry = self._call(trade_type="daytrade")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.trade_type, "daytrade")

    def test_daytrade_persists_to_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            journal_dir = Path(tmpdir)
            entry = self._call(
                trade_type="daytrade",
                journal_dir=journal_dir,
            )
            self.assertIsNotNone(entry)
            # Reload from disk and confirm round-trip.
            written = journal_dir / f"{entry.trade_id}.json"
            self.assertTrue(written.exists())
            with open(written, encoding="utf-8") as f:
                d = json.load(f)
            self.assertEqual(d["trade_type"], "daytrade")


# ── journal_analytics.breakdown_by_trade_type ────────────────────


class TestBreakdownByTradeType(unittest.TestCase):
    """Verify analytics partitions trades by trade_type and produces
    independent metrics per type so swing vs day-trade performance
    can be evaluated separately."""

    def _closed(
        self,
        ticker: str,
        trade_type: str,
        pnl: float,
        r: float,
        risk: float = 100.0,
        strategy: str = "ORB_VWAP",
    ) -> JournalEntry:
        # Build a minimal closed entry. Default risk=$100 keeps
        # r_multiple math simple in tests.
        return JournalEntry(
            trade_id=f"{ticker}_{trade_type[:3]}",
            ticker=ticker,
            strategy=strategy,
            side="long",
            entry_order_id=f"order-{ticker}",
            entry_signal_price=100.0,
            entry_fill_price=100.0,
            realized_pnl=pnl,
            r_multiple=r,
            initial_risk_dollars=risk,
            status="closed",
            trade_type=trade_type,
        )

    def test_partitions_swing_and_daytrade(self):
        from trading_bot_bl.journal_analytics import (
            breakdown_by_trade_type,
        )

        trades = [
            self._closed("AAPL", "swing", pnl=200.0, r=2.0),
            self._closed("MSFT", "swing", pnl=-100.0, r=-1.0),
            self._closed("TSLA", "daytrade", pnl=50.0, r=0.5),
            self._closed("NVDA", "daytrade", pnl=75.0, r=0.75),
            self._closed("AMD", "daytrade", pnl=-25.0, r=-0.25),
        ]
        result = breakdown_by_trade_type(trades)

        self.assertEqual(set(result.keys()), {"swing", "daytrade"})
        self.assertEqual(result["swing"].overall.total_trades, 2)
        self.assertEqual(result["daytrade"].overall.total_trades, 3)
        self.assertEqual(result["swing"].overall.total_pnl, 100.0)
        self.assertEqual(result["daytrade"].overall.total_pnl, 100.0)
        # 1W/1L = 50% for swing; 2W/1L = 67% for daytrade
        self.assertAlmostEqual(
            result["swing"].overall.win_rate, 0.5, places=2
        )
        self.assertAlmostEqual(
            result["daytrade"].overall.win_rate, 2 / 3, places=2
        )

    def test_empty_or_missing_trade_type_buckets_to_swing(self):
        from trading_bot_bl.journal_analytics import (
            breakdown_by_trade_type,
        )

        # Simulate a malformed entry whose trade_type slipped through
        # as empty string.
        entry = self._closed("AAPL", "", pnl=100.0, r=1.0)
        result = breakdown_by_trade_type([entry])
        self.assertIn("swing", result)
        self.assertEqual(result["swing"].overall.total_trades, 1)

    def test_skips_open_and_pending_trades(self):
        from trading_bot_bl.journal_analytics import (
            breakdown_by_trade_type,
        )

        closed = self._closed("AAPL", "daytrade", pnl=100.0, r=1.0)
        open_entry = JournalEntry(
            trade_id="MSFT_open",
            ticker="MSFT",
            strategy="ORB_VWAP",
            side="long",
            entry_order_id="o1",
            entry_signal_price=200.0,
            status="open",
            trade_type="daytrade",
        )
        result = breakdown_by_trade_type([closed, open_entry])
        # Only the closed daytrade counts toward metrics.
        self.assertEqual(
            result["daytrade"].overall.total_trades, 1
        )

    def test_format_trade_type_breakdown_text(self):
        from trading_bot_bl.journal_analytics import (
            breakdown_by_trade_type,
            format_trade_type_breakdown,
        )

        trades = [
            self._closed("AAPL", "swing", pnl=200.0, r=2.0),
            self._closed("TSLA", "daytrade", pnl=50.0, r=0.5),
        ]
        breakdown = breakdown_by_trade_type(trades)
        text = format_trade_type_breakdown(breakdown)
        self.assertIn("Performance by Trade Type", text)
        self.assertIn("swing", text)
        self.assertIn("daytrade", text)
        # Stable ordering: swing first.
        swing_idx = text.index("swing")
        day_idx = text.index("daytrade")
        self.assertLess(swing_idx, day_idx)

    def test_format_breakdown_empty_returns_empty_string(self):
        from trading_bot_bl.journal_analytics import (
            format_trade_type_breakdown,
        )

        self.assertEqual(format_trade_type_breakdown({}), "")


# ── PDF report renders with mixed swing + daytrade trades ────────


class TestPDFReportRendersTradeTypes(unittest.TestCase):
    """End-to-end smoke test: build a journal directory with mixed
    swing + daytrade entries, run the PDF report, and confirm the
    output exists. Catches accidental regressions in the trade-type
    section, the Trade Log column, or the analytics integration."""

    def _write_entry(
        self,
        journal_dir: Path,
        ticker: str,
        trade_type: str,
        pnl: float,
        r: float,
    ) -> None:
        entry = JournalEntry(
            trade_id=f"{ticker}_{trade_type[:3]}",
            ticker=ticker,
            strategy="ORB_VWAP" if trade_type == "daytrade" else "EMA",
            side="long",
            entry_order_id=f"order-{ticker}",
            entry_signal_price=100.0,
            entry_fill_price=100.0,
            entry_qty=10,
            entry_notional=1000.0,
            entry_date="2026-04-20",
            exit_price=100.0 + pnl / 10,
            exit_date="2026-04-21",
            exit_reason="take_profit" if pnl > 0 else "stop_loss",
            realized_pnl=pnl,
            r_multiple=r,
            initial_risk_dollars=100.0,
            holding_days=1,
            opened_at="2026-04-20T09:35:00",
            closed_at="2026-04-21T15:55:00",
            status="closed",
            trade_type=trade_type,
        )
        journal_dir.mkdir(parents=True, exist_ok=True)
        path = journal_dir / f"{entry.trade_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(entry), f)

    def test_pdf_renders_with_mixed_trade_types(self):
        # Heavy import — only when this test runs
        from trading_bot_bl.journal_report import generate_pdf_report

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            journal_dir = log_dir / "journal"

            self._write_entry(journal_dir, "AAPL", "swing", 200.0, 2.0)
            self._write_entry(journal_dir, "MSFT", "swing", -100.0, -1.0)
            self._write_entry(journal_dir, "TSLA", "daytrade", 50.0, 0.5)
            self._write_entry(journal_dir, "NVDA", "daytrade", 75.0, 0.75)
            self._write_entry(journal_dir, "AMD", "daytrade", -25.0, -0.25)

            output_path = log_dir / "report.pdf"
            result_path = generate_pdf_report(log_dir, output_path)

            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 1000)


# ── executor.write_execution_log includes client_order_id ────────


class TestWriteExecutionLogClientOrderId(unittest.TestCase):
    """Execution logs must persist client_order_id so audit/recovery
    can reconcile orders by tag, not just by ticker."""

    def test_log_includes_client_order_id_when_present(self):
        from trading_bot_bl.executor import write_execution_log

        results = [
            OrderResult(
                ticker="AAPL",
                order_id="abc123",
                client_order_id="dt:20260428:0001:AAPL",
                status="submitted",
                side="buy",
                notional=1000.0,
            ),
            OrderResult(
                ticker="MSFT",
                order_id="def456",
                client_order_id="",
                status="submitted",
                side="buy",
                notional=2000.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_execution_log(results, Path(tmpdir))
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(len(data["orders"]), 2)
        self.assertEqual(
            data["orders"][0]["client_order_id"],
            "dt:20260428:0001:AAPL",
        )
        # Empty string is preserved (not dropped) so downstream
        # tooling can rely on the field always being present.
        self.assertEqual(data["orders"][1]["client_order_id"], "")


# ── broker.submit_bracket_order client_order_id plumbing ──────────


class TestBracketOrderClientOrderId(unittest.TestCase):
    """Verify client_order_id is forwarded to Alpaca request and
    echoed in the OrderResult."""

    def setUp(self):
        self.broker_module, self.broker = (
            _setup_broker_with_simplenamespace_requests(self)
        )

        # Capture submitted requests and return a stub Alpaca order.
        # The stub echoes the request's client_order_id (real Alpaca
        # behaviour: server returns whatever id you sent).
        self.submitted_requests: list = []

        def _fake_submit(req):
            self.submitted_requests.append(req)
            return SimpleNamespace(
                id="alpaca-order-id-1",
                client_order_id=getattr(req, "client_order_id", ""),
            )

        self.broker._client.submit_order = _fake_submit

    def test_bracket_market_with_client_order_id(self):
        result = self.broker.submit_bracket_order(
            ticker="AAPL",
            side="buy",
            notional=1000.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            current_price=100.0,
            client_order_id="dt:20260428:0001:AAPL",
        )
        self.assertEqual(result.status, "submitted")
        self.assertEqual(
            result.client_order_id, "dt:20260428:0001:AAPL"
        )
        self.assertEqual(len(self.submitted_requests), 1)
        sent = self.submitted_requests[0]
        # The broker forwarded client_order_id to the request object.
        self.assertEqual(
            sent.client_order_id, "dt:20260428:0001:AAPL"
        )
        # And it's reflected in the MagicMock call_args of the class.
        self.assertEqual(
            self.broker_module.MarketOrderRequest.call_args.kwargs.get(
                "client_order_id"
            ),
            "dt:20260428:0001:AAPL",
        )

    def test_bracket_without_client_order_id_omits_kwarg(self):
        # When no client_order_id is passed, the broker MUST NOT set
        # the kwarg on the request — letting Alpaca generate one.
        self.broker.submit_bracket_order(
            ticker="MSFT",
            side="buy",
            notional=1000.0,
            stop_loss_price=180.0,
            take_profit_price=220.0,
            current_price=200.0,
        )
        # MarketOrderRequest is the mocked class — inspect its last
        # call kwargs.
        last_kwargs = (
            self.broker_module.MarketOrderRequest.call_args.kwargs
        )
        self.assertNotIn("client_order_id", last_kwargs)

    def test_bracket_limit_order_passes_client_order_id(self):
        self.broker.submit_bracket_order(
            ticker="TSLA",
            side="buy",
            notional=1000.0,
            stop_loss_price=190.0,
            take_profit_price=240.0,
            current_price=200.0,
            max_entry_slippage_pct=1.5,
            client_order_id="dt:20260428:0002:TSLA",
        )
        last_kwargs = (
            self.broker_module.LimitOrderRequest.call_args.kwargs
        )
        self.assertEqual(
            last_kwargs.get("client_order_id"),
            "dt:20260428:0002:TSLA",
        )

    def test_bracket_rejected_still_echoes_client_order_id(self):
        # Force the underlying client to raise so we hit the except.
        def _raise(req):
            raise RuntimeError("alpaca said no")

        self.broker._client.submit_order = _raise

        result = self.broker.submit_bracket_order(
            ticker="AAPL",
            side="buy",
            notional=1000.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            current_price=100.0,
            client_order_id="dt:20260428:0003:AAPL",
        )
        self.assertEqual(result.status, "rejected")
        self.assertEqual(
            result.client_order_id, "dt:20260428:0003:AAPL"
        )

    def test_bracket_surfaces_server_generated_id_when_caller_passes_none(
        self,
    ):
        """When the caller doesn't pass a client_order_id, the broker
        should still surface the id Alpaca's server generated — that's
        the correlation key the journal needs for follow-up actions
        (e.g. cancel-replace, reconcile)."""
        # Override fake to return a server-generated id regardless of req
        def _fake_submit(req):
            return SimpleNamespace(
                id="alpaca-order-id-1",
                client_order_id="alpaca-server-gen-abcdef",
            )

        self.broker._client.submit_order = _fake_submit

        result = self.broker.submit_bracket_order(
            ticker="AAPL",
            side="buy",
            notional=1000.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            current_price=100.0,
            # No client_order_id passed — Alpaca generates one
        )
        self.assertEqual(result.status, "submitted")
        self.assertEqual(
            result.client_order_id, "alpaca-server-gen-abcdef"
        )


# ── broker.submit_market_order client_order_id plumbing ──────────


class TestMarketOrderClientOrderId(unittest.TestCase):
    def setUp(self):
        self.broker_module, self.broker = (
            _setup_broker_with_simplenamespace_requests(self)
        )

        def _fake_submit(req):
            return SimpleNamespace(
                id="alpaca-mkt-order-1",
                client_order_id=getattr(req, "client_order_id", ""),
            )

        self.broker._client.submit_order = _fake_submit

    def test_market_order_with_client_order_id(self):
        result = self.broker.submit_market_order(
            ticker="AAPL",
            side="sell",
            qty=10,
            client_order_id="dt:20260428:0001:AAPL:exit",
        )
        self.assertEqual(result.status, "submitted")
        self.assertEqual(
            result.client_order_id,
            "dt:20260428:0001:AAPL:exit",
        )
        last_kwargs = (
            self.broker_module.MarketOrderRequest.call_args.kwargs
        )
        self.assertEqual(
            last_kwargs.get("client_order_id"),
            "dt:20260428:0001:AAPL:exit",
        )

    def test_market_order_without_client_order_id_omits_kwarg(self):
        self.broker.submit_market_order(
            ticker="MSFT",
            side="buy",
            notional=500.0,
        )
        last_kwargs = (
            self.broker_module.MarketOrderRequest.call_args.kwargs
        )
        self.assertNotIn("client_order_id", last_kwargs)

    def test_market_order_rejected_echoes_client_order_id(self):
        def _raise(req):
            raise RuntimeError("alpaca said no")

        self.broker._client.submit_order = _raise

        result = self.broker.submit_market_order(
            ticker="AAPL",
            side="buy",
            qty=5,
            client_order_id="dt:20260428:0099:AAPL",
        )
        self.assertEqual(result.status, "rejected")
        self.assertEqual(
            result.client_order_id, "dt:20260428:0099:AAPL"
        )

    def test_market_surfaces_server_generated_id_when_caller_passes_none(
        self,
    ):
        """Mirror of the bracket-order test: server-generated ids
        propagate into OrderResult when the caller doesn't supply one."""

        def _fake_submit(req):
            return SimpleNamespace(
                id="alpaca-mkt-order-1",
                client_order_id="alpaca-server-gen-xyz",
            )

        self.broker._client.submit_order = _fake_submit

        result = self.broker.submit_market_order(
            ticker="MSFT",
            side="buy",
            notional=500.0,
            # No client_order_id passed
        )
        self.assertEqual(result.status, "submitted")
        self.assertEqual(
            result.client_order_id, "alpaca-server-gen-xyz"
        )


if __name__ == "__main__":
    unittest.main()
