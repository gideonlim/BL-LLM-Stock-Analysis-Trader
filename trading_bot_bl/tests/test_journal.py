"""
Tests for the trade journal — lifecycle, analytics, and equity curve.

Covers:
1. Journal entry creation, persistence, and loading
2. Pending → open → closed lifecycle transitions
3. Excursion tracking (MAE/MFE/ETD)
4. SL modification recording
5. Closed trade detection
6. Migration of existing positions
7. Analytics: overall, R-distribution, streaks, excursion, execution
8. Equity curve snapshots and drawdown computation
9. Probabilistic Sharpe and statistical confidence

Run with:
    python -m pytest trading_bot_bl/test_journal.py -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock alpaca dependencies before importing trading_bot_bl
import unittest.mock as _um

_alpaca_mock = _um.MagicMock()
sys.modules["alpaca"] = _alpaca_mock
sys.modules["alpaca.trading"] = _alpaca_mock.trading
sys.modules["alpaca.trading.client"] = _alpaca_mock.trading.client
sys.modules["alpaca.trading.requests"] = (
    _alpaca_mock.trading.requests
)
sys.modules["alpaca.trading.enums"] = (
    _alpaca_mock.trading.enums
)
sys.modules["alpaca.data"] = _alpaca_mock.data
sys.modules["alpaca.data.historical"] = (
    _alpaca_mock.data.historical
)
sys.modules["alpaca.data.requests"] = (
    _alpaca_mock.data.requests
)

from trading_bot_bl.models import (
    EquitySnapshot,
    JournalEntry,
    PortfolioSnapshot,
)
from trading_bot_bl.journal import (
    create_trade,
    close_trade,
    update_trade,
    record_sl_modification,
    load_open_trades,
    load_all_trades,
    detect_closed_trades,
    migrate_existing_positions,
    resolve_pending_trades,
    _entry_to_dict,
    _dict_to_entry,
    _query_exit_details,
    _save_entry,
)
from trading_bot_bl.equity_curve import (
    record_snapshot,
    load_snapshots,
)
from trading_bot_bl.journal_analytics import (
    compute_journal_metrics,
    format_metrics_text,
    _probabilistic_sharpe,
    _min_track_record_length,
    _safe_mean,
    _std,
    _skewness,
    _pearson_corr,
)


class TestJournalSerialization(unittest.TestCase):
    """Test JournalEntry to/from dict/JSON round-trip."""

    def test_round_trip(self):
        entry = JournalEntry(
            trade_id="AAPL_abc12345",
            ticker="AAPL",
            strategy="VWAP Trend",
            side="long",
            entry_order_id="abc12345-full-uuid",
            entry_signal_price=150.0,
        )
        d = _entry_to_dict(entry)
        restored = _dict_to_entry(d)
        self.assertEqual(restored.trade_id, "AAPL_abc12345")
        self.assertEqual(restored.ticker, "AAPL")
        self.assertEqual(restored.strategy, "VWAP Trend")
        self.assertEqual(restored.status, "pending")

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="MSFT_def67890",
                ticker="MSFT",
                strategy="Z-Score",
                side="long",
                entry_order_id="def67890",
                entry_signal_price=400.0,
                status="open",
            )
            _save_entry(entry, jdir)
            loaded = load_open_trades(jdir)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].ticker, "MSFT")

    def test_ignores_extra_json_fields(self):
        """Future-proofing: unknown fields in JSON don't crash."""
        d = {
            "trade_id": "X_123",
            "ticker": "X",
            "strategy": "test",
            "side": "long",
            "entry_order_id": "123",
            "entry_signal_price": 100.0,
            "unknown_future_field": "ignored",
        }
        entry = _dict_to_entry(d)
        self.assertEqual(entry.ticker, "X")


class TestJournalLifecycle(unittest.TestCase):
    """Test the full pending → open → closed lifecycle."""

    def test_create_trade_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = create_trade(
                order_id="order-abc-123-full",
                ticker="AAPL",
                strategy="VWAP Trend",
                side="buy",
                signal_price=150.0,
                notional=1500.0,
                sl_price=142.50,
                tp_price=165.0,
                composite_score=55.0,
                confidence="HIGH",
                confidence_score=5,
                vix=18.5,
                market_regime="NEUTRAL",
                journal_dir=jdir,
            )
            self.assertIsNotNone(entry)
            self.assertEqual(entry.status, "pending")
            self.assertEqual(entry.ticker, "AAPL")
            self.assertEqual(entry.entry_vix, 18.5)
            # Verify file was written
            files = list(jdir.glob("*.json"))
            self.assertEqual(len(files), 1)

    def test_close_trade_computes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="AAPL_abc",
                ticker="AAPL",
                strategy="VWAP",
                side="long",
                entry_order_id="abc",
                entry_signal_price=150.0,
                entry_fill_price=150.50,
                entry_qty=10.0,
                entry_date="2026-03-10T10:00:00",
                original_sl_price=142.50,
                initial_risk_per_share=8.0,
                initial_risk_dollars=80.0,
                max_favorable_excursion=165.0,
                max_adverse_excursion=148.0,
                mfe_pct=9.63,
                mae_pct=1.66,
                status="open",
            )
            _save_entry(entry, jdir)

            close_trade(
                entry,
                exit_price=160.0,
                exit_reason="take_profit",
                journal_dir=jdir,
                expected_exit_price=165.0,
            )

            self.assertEqual(entry.status, "closed")
            # P&L = (160 - 150.50) * 10 = 95
            self.assertAlmostEqual(entry.realized_pnl, 95.0, 1)
            # R-multiple = 95 / 80 = 1.19
            self.assertAlmostEqual(entry.r_multiple, 1.19, 1)
            # ETD = 165 - 160 = 5
            self.assertAlmostEqual(entry.etd, 5.0, 1)
            # Edge ratio = mfe_pct / mae_pct
            self.assertGreater(entry.edge_ratio, 1.0)
            # Exit slippage = 165 - 160 = 5
            self.assertAlmostEqual(entry.exit_slippage, 5.0, 1)

    def test_close_trade_idempotent(self):
        """Closing an already-closed trade is a no-op."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="X_1",
                ticker="X",
                strategy="test",
                side="long",
                entry_order_id="1",
                entry_signal_price=100.0,
                entry_fill_price=100.0,
                status="closed",
                realized_pnl=50.0,
            )
            _save_entry(entry, jdir)
            close_trade(entry, 90.0, "stop_loss", jdir)
            # P&L should NOT change (already closed)
            self.assertEqual(entry.realized_pnl, 50.0)


class TestExcursionTracking(unittest.TestCase):
    """Test MAE/MFE/price sample updates."""

    def test_update_mfe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="AAPL_1",
                ticker="AAPL",
                strategy="test",
                side="long",
                entry_order_id="1",
                entry_signal_price=100.0,
                entry_fill_price=100.0,
                max_favorable_excursion=100.0,
                max_adverse_excursion=100.0,
                status="open",
            )
            _save_entry(entry, jdir)

            update_trade(entry, 110.0, jdir, "2026-03-15")
            self.assertEqual(entry.max_favorable_excursion, 110.0)
            self.assertAlmostEqual(entry.mfe_pct, 10.0, 1)

    def test_update_mae(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="AAPL_2",
                ticker="AAPL",
                strategy="test",
                side="long",
                entry_order_id="2",
                entry_signal_price=100.0,
                entry_fill_price=100.0,
                max_favorable_excursion=100.0,
                max_adverse_excursion=100.0,
                status="open",
            )
            _save_entry(entry, jdir)

            update_trade(entry, 95.0, jdir, "2026-03-15")
            self.assertEqual(entry.max_adverse_excursion, 95.0)
            self.assertAlmostEqual(entry.mae_pct, 5.0, 1)

    def test_price_samples_deduplicate_by_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="X_1",
                ticker="X",
                strategy="test",
                side="long",
                entry_order_id="1",
                entry_signal_price=100.0,
                entry_fill_price=100.0,
                max_favorable_excursion=100.0,
                max_adverse_excursion=100.0,
                status="open",
            )
            _save_entry(entry, jdir)

            update_trade(entry, 101.0, jdir, "2026-03-15")
            update_trade(entry, 102.0, jdir, "2026-03-15")
            # Same date — should only have one sample
            self.assertEqual(len(entry.price_samples), 1)

            update_trade(entry, 103.0, jdir, "2026-03-16")
            self.assertEqual(len(entry.price_samples), 2)


class TestSLModification(unittest.TestCase):
    def test_record_sl_mod(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="X_1",
                ticker="X",
                strategy="test",
                side="long",
                entry_order_id="1",
                entry_signal_price=100.0,
                status="open",
            )
            _save_entry(entry, jdir)

            record_sl_modification(
                entry, 95.0, 100.0, "breakeven", 108.0, jdir
            )
            self.assertEqual(len(entry.sl_modifications), 1)
            self.assertEqual(
                entry.sl_modifications[0]["reason"], "breakeven"
            )


class TestDetectClosedTrades(unittest.TestCase):
    def test_detects_gone_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            # Create open entry for AAPL
            entry = JournalEntry(
                trade_id="AAPL_1",
                ticker="AAPL",
                strategy="test",
                side="long",
                entry_order_id="1",
                entry_signal_price=150.0,
                entry_fill_price=150.0,
                status="open",
                price_samples=[
                    {"date": "2026-03-17", "price": 155.0}
                ],
            )
            _save_entry(entry, jdir)

            # Current positions: no AAPL
            closed = detect_closed_trades(
                current_positions={"MSFT": {}},
                journal_dir=jdir,
            )
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0].ticker, "AAPL")
            self.assertEqual(closed[0].status, "closed")

    def test_no_false_positives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            entry = JournalEntry(
                trade_id="AAPL_1",
                ticker="AAPL",
                strategy="test",
                side="long",
                entry_order_id="1",
                entry_signal_price=150.0,
                status="open",
            )
            _save_entry(entry, jdir)

            # AAPL still in positions
            closed = detect_closed_trades(
                current_positions={"AAPL": {}},
                journal_dir=jdir,
            )
            self.assertEqual(len(closed), 0)


class TestPrematureCloseRevert(unittest.TestCase):
    """Regression tests for _reconcile_premature_closes.

    Background: prior implementation reverted any closed entry to
    open if the ticker was in current_positions. This caused
    bug 2026-04-29: closing FLG, then re-entering FLG weeks later
    caused the original closed entry to revert (losing exit data
    and producing phantom 30-day open holdings).

    Fix: only revert if broker's current avg_entry matches the
    journal entry's entry_fill_price within tolerance.
    """

    def _closed_entry(
        self, trade_id: str, ticker: str, fill: float, jdir: Path,
    ) -> JournalEntry:
        entry = JournalEntry(
            trade_id=trade_id,
            ticker=ticker,
            strategy="test",
            side="long",
            entry_order_id=f"{trade_id}-order",
            entry_signal_price=fill,
            entry_fill_price=fill,
            entry_qty=100,
            entry_notional=fill * 100,
            entry_date="2026-03-30T15:42:00",
            status="closed",
            exit_price=fill * 1.05,
            exit_fill_price=fill * 1.05,
            exit_date="2026-04-10T16:00:00",
            closed_at="2026-04-10T16:00:00",
            exit_reason="take_profit",
            holding_days=11,
            realized_pnl=fill * 100 * 0.05,
            r_multiple=1.5,
        )
        _save_entry(entry, jdir)
        return entry

    def test_does_not_revert_when_avg_entry_differs(self):
        """Different trade with same ticker — must NOT revert.

        FLG closed at $13.13 in March; re-opened at $13.97 in April.
        The April entry is in broker positions, but the March entry
        is a different trade. Reverting it would corrupt the closed
        record.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._closed_entry("FLG_old", "FLG", 13.13, jdir)

            # Broker shows FLG at a DIFFERENT avg_entry (new position)
            detect_closed_trades(
                current_positions={"FLG": {"avg_entry": 13.97, "qty": 616}},
                journal_dir=jdir,
            )

            # Old closed entry should remain closed
            entries = load_all_trades(jdir, lookback_days=365)
            old = next(e for e in entries if e.trade_id == "FLG_old")
            self.assertEqual(old.status, "closed")
            self.assertEqual(old.exit_date, "2026-04-10T16:00:00")
            self.assertEqual(old.holding_days, 11)
            self.assertEqual(old.r_multiple, 1.5)

    def test_reverts_when_avg_entry_matches(self):
        """Same trade prematurely closed — must revert.

        The legitimate use case: bot mistakenly fired close_trade
        but the close order never filled, so the position is still
        held at the same fill price.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._closed_entry("AAPL_premature", "AAPL", 150.0, jdir)

            # Broker shows AAPL at the SAME avg_entry — same trade
            detect_closed_trades(
                current_positions={
                    "AAPL": {"avg_entry": 150.0, "qty": 100},
                },
                journal_dir=jdir,
            )

            entries = load_all_trades(jdir, lookback_days=365)
            entry = next(
                e for e in entries if e.trade_id == "AAPL_premature"
            )
            self.assertEqual(entry.status, "open")
            self.assertEqual(entry.exit_date, "")
            self.assertEqual(entry.holding_days, 0)

    def test_does_not_revert_when_position_gone(self):
        """No premature close to revert — position correctly closed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._closed_entry("MSFT_clean", "MSFT", 200.0, jdir)

            # MSFT not in current positions — closed correctly
            detect_closed_trades(
                current_positions={},
                journal_dir=jdir,
            )

            entries = load_all_trades(jdir, lookback_days=365)
            entry = next(
                e for e in entries if e.trade_id == "MSFT_clean"
            )
            self.assertEqual(entry.status, "closed")

    def test_tolerance_within_threshold_reverts(self):
        """Tiny avg_entry drift (e.g. 0.1%) is treated as same trade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._closed_entry("NVDA_drift", "NVDA", 500.0, jdir)

            # Broker avg 500.10 — 0.02% drift, within 0.5% tolerance
            detect_closed_trades(
                current_positions={
                    "NVDA": {"avg_entry": 500.10, "qty": 50},
                },
                journal_dir=jdir,
            )

            entries = load_all_trades(jdir, lookback_days=365)
            entry = next(
                e for e in entries if e.trade_id == "NVDA_drift"
            )
            self.assertEqual(entry.status, "open")  # reverted

    def test_tolerance_above_threshold_does_not_revert(self):
        """avg_entry drift > 0.5% means different trade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._closed_entry("TSLA_diff", "TSLA", 200.0, jdir)

            # Broker avg 210.0 — 5% drift, well above tolerance
            detect_closed_trades(
                current_positions={
                    "TSLA": {"avg_entry": 210.0, "qty": 50},
                },
                journal_dir=jdir,
            )

            entries = load_all_trades(jdir, lookback_days=365)
            entry = next(
                e for e in entries if e.trade_id == "TSLA_diff"
            )
            self.assertEqual(entry.status, "closed")


class TestDetectClosedOrphan(unittest.TestCase):
    """Regression tests for orphan-close in detect_closed_trades.

    Background: prior implementation skipped any open journal
    entry whose ticker appeared in current_positions, regardless
    of whether the broker's avg_entry matched.  Combined with
    the (now-fixed) premature-close revert bug, this caused
    orphans like INTC_538232b3: closed correctly Apr 13, then
    reverted to open Apr 30 when a NEW INTC trade opened, then
    permanently stuck open because the close loop kept skipping
    on naive ticker match.

    Fix: mirror _reconcile_premature_closes — if avg_entry drifts
    beyond tolerance, the held position is a different trade and
    this journal entry is an orphan (close it).
    """

    def _open_entry(
        self,
        trade_id: str,
        ticker: str,
        fill: float,
        jdir: Path,
        last_sample_price: float | None = None,
    ) -> JournalEntry:
        """Create a status='open' journal entry with a price sample
        so the close-detection fallback path has something to work
        with when no broker is supplied.
        """
        sample_price = (
            last_sample_price
            if last_sample_price is not None
            else fill * 1.05
        )
        entry = JournalEntry(
            trade_id=trade_id,
            ticker=ticker,
            strategy="test",
            side="long",
            entry_order_id=f"{trade_id}-order",
            entry_signal_price=fill,
            entry_fill_price=fill,
            entry_qty=100,
            entry_notional=fill * 100,
            entry_date="2026-04-08T15:44:00",
            opened_at="2026-04-08T15:44:00",
            initial_risk_per_share=fill * 0.1,
            status="open",
            price_samples=[
                {"date": "2026-04-10", "price": sample_price}
            ],
        )
        _save_entry(entry, jdir)
        return entry

    def test_orphan_closes_when_avg_entry_mismatches(self):
        """The INTC scenario.

        Old INTC trade @ $57.41 closed Apr 10 but stuck status=open
        in journal.  New INTC trade opens @ $93.24 — broker now has
        INTC at avg $93.24.  detect_closed_trades must treat the
        old entry as an orphan and close it (not skip on naive
        ticker match).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._open_entry(
                "INTC_orphan",
                "INTC",
                57.41,
                jdir,
                last_sample_price=63.70,
            )

            closed = detect_closed_trades(
                current_positions={
                    "INTC": {"avg_entry": 93.24, "qty": 66},
                },
                journal_dir=jdir,
            )

            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0].ticker, "INTC")
            self.assertEqual(closed[0].status, "closed")
            # Should have used the price-sample fallback
            self.assertGreater(closed[0].exit_price, 0)

    def test_held_position_not_closed_when_avg_entry_matches(self):
        """Same trade, still held — must NOT be closed.

        AAPL fill price $150.00, broker avg_entry $150.00 — this
        is the active position.  The close loop must skip it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._open_entry("AAPL_held", "AAPL", 150.0, jdir)

            closed = detect_closed_trades(
                current_positions={
                    "AAPL": {"avg_entry": 150.0, "qty": 100},
                },
                journal_dir=jdir,
            )
            self.assertEqual(len(closed), 0)

    def test_held_position_not_closed_within_tolerance(self):
        """Tiny avg_entry drift (e.g. 0.1%) is treated as same trade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._open_entry("NVDA_held", "NVDA", 500.0, jdir)

            # Broker avg 500.10 — 0.02% drift, within 0.5% tolerance
            closed = detect_closed_trades(
                current_positions={
                    "NVDA": {"avg_entry": 500.10, "qty": 50},
                },
                journal_dir=jdir,
            )
            self.assertEqual(len(closed), 0)

    def test_safe_default_when_broker_avg_missing(self):
        """If broker_pos lacks avg_entry, fall back to skip (safe).

        Conservative default: never close an entry whose status we
        can't verify.  An orphan staying open one extra cycle is
        cheaper than wrongly closing a live position.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            self._open_entry("MSFT_unknown", "MSFT", 200.0, jdir)

            # Broker dict has no avg_entry — verification impossible
            closed = detect_closed_trades(
                current_positions={"MSFT": {"qty": 50}},
                journal_dir=jdir,
            )
            self.assertEqual(len(closed), 0)


class TestQueryExitDetailsQtyMatch(unittest.TestCase):
    """_query_exit_details must match SELL fills by qty + date.

    Real bug May 2026: an INTC orphan @ $57.41 x 152 (Apr 8 entry)
    was auto-closed using a later sell @ $99.85 x 66 (May 1 close
    of a different INTC trade), yielding bogus +$6,450 P&L.
    Fix: prefer SELL with matching filled_qty and filled_at >=
    entry.entry_date.
    """

    def _mk_order(
        self,
        *,
        qty: float,
        fill_price: float,
        filled_at: str,
        stop_price: float | None = None,
        limit_price: float | None = None,
    ):
        return SimpleNamespace(
            filled_qty=qty,
            filled_avg_price=fill_price,
            filled_at=filled_at,
            stop_price=stop_price,
            limit_price=limit_price,
            order_type="limit",
        )

    def _mk_broker(self, orders):
        client = SimpleNamespace(
            get_orders=lambda **kw: orders,
        )
        return SimpleNamespace(_client=client)

    def _mk_entry(
        self,
        *,
        ticker: str,
        qty: float,
        fill: float,
        entry_date: str,
    ) -> JournalEntry:
        return JournalEntry(
            trade_id=f"{ticker}_test",
            ticker=ticker,
            strategy="test",
            side="long",
            entry_order_id="x",
            entry_signal_price=fill,
            entry_fill_price=fill,
            entry_qty=qty,
            entry_date=entry_date,
            original_sl_price=fill * 0.9,
            original_tp_price=fill * 1.1,
        )

    def test_old_orphan_picks_qty_matching_sell(self):
        """INTC scenario: 2 SELL fills (152 @ $63.70 Apr 10,
        66 @ $99.85 May 1).  Old orphan with qty=152 must pick
        the Apr 10 fill, not the most recent (May 1) fill.
        """
        entry = self._mk_entry(
            ticker="INTC",
            qty=152.0,
            fill=57.41,
            entry_date="2026-04-08 15:44:18+00:00",
        )
        # Alpaca returns most-recent first
        orders = [
            self._mk_order(
                qty=66,
                fill_price=99.85,
                filled_at="2026-05-01 15:19:27+00:00",
                limit_price=99.85,
            ),
            self._mk_order(
                qty=152,
                fill_price=63.70,
                filled_at="2026-04-10 16:00:00+00:00",
                limit_price=63.61,
            ),
        ]
        broker = self._mk_broker(orders)

        fp, reason, exp = _query_exit_details(broker, entry)
        self.assertAlmostEqual(fp, 63.70, places=2)
        self.assertEqual(reason, "take_profit")
        self.assertAlmostEqual(exp, 63.61, places=2)

    def test_new_entry_picks_qty_matching_sell(self):
        """Same broker setup; entry qty=66 must pick the May 1 fill."""
        entry = self._mk_entry(
            ticker="INTC",
            qty=66.0,
            fill=93.24,
            entry_date="2026-04-30 16:00:23+00:00",
        )
        orders = [
            self._mk_order(
                qty=66,
                fill_price=99.85,
                filled_at="2026-05-01 15:19:27+00:00",
                limit_price=99.85,
            ),
            self._mk_order(
                qty=152,
                fill_price=63.70,
                filled_at="2026-04-10 16:00:00+00:00",
                limit_price=63.61,
            ),
        ]
        broker = self._mk_broker(orders)

        fp, reason, _exp = _query_exit_details(broker, entry)
        self.assertAlmostEqual(fp, 99.85, places=2)
        self.assertEqual(reason, "take_profit")

    def test_qty_match_prefers_earliest_after_entry(self):
        """When multiple qty matches exist after entry_date,
        pick the earliest (closest forward in time) — that's
        the actual exit fill, not a later same-qty re-entry exit.
        """
        entry = self._mk_entry(
            ticker="AAPL",
            qty=100.0,
            fill=150.0,
            entry_date="2026-03-01 10:00:00+00:00",
        )
        # Two later same-qty SELLs — the FIRST after entry is correct
        orders = [
            self._mk_order(
                qty=100,
                fill_price=180.0,
                filled_at="2026-04-15 14:00:00+00:00",
                limit_price=180.0,
            ),
            self._mk_order(
                qty=100,
                fill_price=160.0,
                filled_at="2026-03-15 14:00:00+00:00",
                limit_price=160.0,
            ),
        ]
        broker = self._mk_broker(orders)

        fp, _reason, _exp = _query_exit_details(broker, entry)
        # Should pick the Mar 15 fill ($160), not Apr 15 ($180)
        self.assertAlmostEqual(fp, 160.0, places=2)

    def test_falls_back_to_most_recent_when_no_qty_match(self):
        """When no SELL has matching qty (entry qty is unusual),
        fall back to the most recent SELL — preserves legacy
        behavior for orphans whose actual close is unidentifiable.
        """
        entry = self._mk_entry(
            ticker="TSLA",
            qty=200.0,
            fill=200.0,
            entry_date="2026-04-01 10:00:00+00:00",
        )
        orders = [
            self._mk_order(
                qty=66,
                fill_price=210.0,
                filled_at="2026-05-01 14:00:00+00:00",
                limit_price=210.0,
            ),
            self._mk_order(
                qty=152,
                fill_price=190.0,
                filled_at="2026-04-15 14:00:00+00:00",
                limit_price=190.0,
            ),
        ]
        broker = self._mk_broker(orders)

        fp, _reason, _exp = _query_exit_details(broker, entry)
        self.assertAlmostEqual(fp, 210.0, places=2)

    def test_returns_unknown_when_no_filled_orders(self):
        """Empty broker response → (0.0, 'unknown', 0.0)."""
        entry = self._mk_entry(
            ticker="MSFT",
            qty=50.0,
            fill=400.0,
            entry_date="2026-04-01 10:00:00+00:00",
        )
        broker = self._mk_broker([])

        fp, reason, exp = _query_exit_details(broker, entry)
        self.assertEqual(fp, 0.0)
        self.assertEqual(reason, "unknown")
        self.assertEqual(exp, 0.0)


class TestMigration(unittest.TestCase):
    def test_migrates_new_position(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            positions = {
                "AAPL": {
                    "avg_entry": 150.0,
                    "qty": 10.0,
                    "side": "long",
                    "entry_date": "2026-03-10",
                },
            }
            orders = [
                SimpleNamespace(
                    symbol="AAPL",
                    stop_price=142.50,
                    limit_price=None,
                ),
                SimpleNamespace(
                    symbol="AAPL",
                    stop_price=None,
                    limit_price=165.0,
                ),
            ]
            count = migrate_existing_positions(
                positions, orders, jdir, vix=20.0
            )
            self.assertEqual(count, 1)

            loaded = load_open_trades(jdir)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].ticker, "AAPL")
            self.assertIn("migrated", loaded[0].tags)
            self.assertAlmostEqual(
                loaded[0].original_sl_price, 142.50
            )
            self.assertAlmostEqual(
                loaded[0].original_tp_price, 165.0
            )

    def test_skip_already_tracked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jdir = Path(tmpdir) / "journal"
            # Pre-existing journal entry
            entry = JournalEntry(
                trade_id="AAPL_existing",
                ticker="AAPL",
                strategy="test",
                side="long",
                entry_order_id="existing",
                entry_signal_price=150.0,
                status="open",
            )
            _save_entry(entry, jdir)

            count = migrate_existing_positions(
                {"AAPL": {"avg_entry": 150.0, "qty": 10}},
                [],
                jdir,
            )
            self.assertEqual(count, 0)


class TestEquityCurve(unittest.TestCase):
    def test_snapshot_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            portfolio = PortfolioSnapshot(
                equity=100000.0,
                cash=50000.0,
                market_value=50000.0,
                day_pnl=-200.0,
                day_pnl_pct=-0.2,
                positions={
                    "AAPL": {"unrealized_pnl": -200.0},
                },
            )
            snap = record_snapshot(portfolio, log_dir)
            self.assertIsNotNone(snap)
            self.assertEqual(snap.equity, 100000.0)
            self.assertEqual(snap.num_positions, 1)

            loaded = load_snapshots(log_dir)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].equity, 100000.0)

    def test_drawdown_from_hwm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            # First snapshot: equity = 100k (sets HWM)
            p1 = PortfolioSnapshot(
                equity=100000.0,
                cash=50000.0,
                market_value=50000.0,
                positions={},
            )
            record_snapshot(p1, log_dir)

            # Second snapshot: equity dropped to 98k
            p2 = PortfolioSnapshot(
                equity=98000.0,
                cash=50000.0,
                market_value=48000.0,
                positions={},
            )
            snap2 = record_snapshot(p2, log_dir)
            self.assertIsNotNone(snap2)
            # DD = (100k - 98k) / 100k = 2%
            self.assertAlmostEqual(snap2.drawdown_pct, 2.0, 1)
            self.assertEqual(snap2.high_water_mark, 100000.0)


class TestJournalAnalytics(unittest.TestCase):
    """Test metric computation from synthetic trades."""

    def _make_trade(
        self,
        pnl: float,
        r: float = 0.0,
        strategy: str = "test",
        mae_pct: float = 0.0,
        mfe_pct: float = 0.0,
        etd_pct: float = 0.0,
        edge_ratio: float = 0.0,
        holding_days: int = 5,
        exit_reason: str = "take_profit",
        entry_slippage_pct: float = 0.0,
        entry_slippage: float = 0.0,
        exit_slippage: float = 0.0,
        regime: str = "NEUTRAL",
        entry_date: str = "2026-03-10",
        mfe_date: str = "2026-03-12",
        mae_date: str = "2026-03-11",
    ) -> JournalEntry:
        return JournalEntry(
            trade_id=f"T_{id(pnl)}",
            ticker="X",
            strategy=strategy,
            side="long",
            entry_order_id="o",
            entry_signal_price=100.0,
            entry_fill_price=100.0,
            realized_pnl=pnl,
            realized_pnl_pct=pnl / 100,
            r_multiple=r,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            etd_pct=etd_pct,
            edge_ratio=edge_ratio,
            holding_days=holding_days,
            exit_reason=exit_reason,
            entry_slippage_pct=entry_slippage_pct,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            entry_market_regime=regime,
            entry_date=entry_date,
            mfe_date=mfe_date,
            mae_date=mae_date,
            status="closed",
            closed_at="2026-03-15T10:00:00",
        )

    def test_overall_metrics(self):
        trades = [
            self._make_trade(100, r=1.0),
            self._make_trade(200, r=2.0),
            self._make_trade(-50, r=-0.5),
            self._make_trade(-80, r=-0.8),
        ]
        m = compute_journal_metrics(trades)
        self.assertEqual(m.overall.total_trades, 4)
        self.assertEqual(m.overall.wins, 2)
        self.assertEqual(m.overall.losses, 2)
        self.assertAlmostEqual(m.overall.win_rate, 0.5, 2)
        self.assertAlmostEqual(m.overall.avg_win, 150.0, 1)
        self.assertAlmostEqual(m.overall.avg_loss, -65.0, 1)
        self.assertAlmostEqual(m.overall.total_pnl, 170.0, 1)

    def test_profit_factor(self):
        trades = [
            self._make_trade(300, r=3.0),
            self._make_trade(-100, r=-1.0),
        ]
        m = compute_journal_metrics(trades)
        # PF = 300 / 100 = 3.0
        self.assertAlmostEqual(m.overall.profit_factor, 3.0, 1)

    def test_r_distribution(self):
        trades = [
            self._make_trade(100, r=1.0),
            self._make_trade(300, r=3.0),
            self._make_trade(-100, r=-1.0),
            self._make_trade(50, r=0.5),
        ]
        m = compute_journal_metrics(trades)
        rd = m.r_distribution
        self.assertAlmostEqual(rd.mean_r, 0.875, 2)
        self.assertAlmostEqual(rd.pct_above_2r, 0.25, 2)
        self.assertAlmostEqual(rd.pct_below_neg1r, 0.0, 2)

    def test_streaks(self):
        # W, W, W, L, L, W
        trades = [
            self._make_trade(10, r=0.1),
            self._make_trade(20, r=0.2),
            self._make_trade(30, r=0.3),
            self._make_trade(-10, r=-0.1),
            self._make_trade(-20, r=-0.2),
            self._make_trade(5, r=0.05),
        ]
        # Give them sequential close dates
        for i, t in enumerate(trades):
            t.closed_at = f"2026-03-{10+i:02d}T10:00:00"

        m = compute_journal_metrics(trades)
        self.assertEqual(m.streaks.max_consecutive_wins, 3)
        self.assertEqual(m.streaks.max_consecutive_losses, 2)
        self.assertEqual(m.streaks.current_streak, 1)
        self.assertEqual(m.streaks.current_streak_type, "win")

    def test_holding_analysis(self):
        trades = [
            self._make_trade(
                100, r=1.0, holding_days=3,
                exit_reason="take_profit",
            ),
            self._make_trade(
                -50, r=-0.5, holding_days=10,
                exit_reason="time_exit",
            ),
            self._make_trade(
                200, r=2.0, holding_days=5,
                exit_reason="take_profit",
            ),
        ]
        m = compute_journal_metrics(trades)
        self.assertAlmostEqual(m.holding.avg_hold_all, 6.0, 1)
        self.assertAlmostEqual(
            m.holding.time_exit_rate, 1 / 3, 2
        )
        self.assertIn(
            "time_exit", m.holding.exit_reason_distribution
        )

    def test_strategy_breakdown(self):
        trades = [
            self._make_trade(100, r=1.0, strategy="VWAP"),
            self._make_trade(-50, r=-0.5, strategy="VWAP"),
            self._make_trade(200, r=2.0, strategy="Z-Score"),
        ]
        m = compute_journal_metrics(trades)
        self.assertIn("VWAP", m.by_strategy)
        self.assertIn("Z-Score", m.by_strategy)
        self.assertEqual(m.by_strategy["VWAP"].trade_count, 2)
        self.assertEqual(m.by_strategy["Z-Score"].trade_count, 1)

    def test_regime_breakdown(self):
        trades = [
            self._make_trade(100, r=1.0, regime="NEUTRAL"),
            self._make_trade(-50, r=-0.5, regime="FEAR"),
        ]
        m = compute_journal_metrics(trades)
        self.assertIn("NEUTRAL", m.by_regime)
        self.assertIn("FEAR", m.by_regime)

    def test_format_metrics_text(self):
        trades = [
            self._make_trade(100, r=1.0),
            self._make_trade(-50, r=-0.5),
        ]
        m = compute_journal_metrics(trades)
        text = format_metrics_text(m)
        self.assertIn("Trade Journal Report", text)
        self.assertIn("Win Rate", text)

    def test_empty_trades(self):
        """No trades should not crash."""
        m = compute_journal_metrics([])
        self.assertEqual(m.overall.total_trades, 0)

    def test_all_winners(self):
        trades = [
            self._make_trade(100, r=1.0),
            self._make_trade(200, r=2.0),
        ]
        m = compute_journal_metrics(trades)
        self.assertAlmostEqual(m.overall.win_rate, 1.0, 2)
        # Profit factor should be inf (no losses)
        self.assertEqual(m.overall.profit_factor, float("inf"))


class TestMathHelpers(unittest.TestCase):
    def test_safe_mean_empty(self):
        self.assertEqual(_safe_mean([]), 0.0)

    def test_std(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        s = _std(vals)
        self.assertAlmostEqual(s, 2.0, 0)

    def test_skewness_symmetric(self):
        vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
        sk = _skewness(vals)
        self.assertAlmostEqual(sk, 0.0, 1)

    def test_probabilistic_sharpe(self):
        # With a high observed SR and enough observations,
        # PSR should be > 0.5
        psr = _probabilistic_sharpe(
            observed_sr=0.1,
            benchmark_sr=0.0,
            n=100,
            skew=0.0,
            kurtosis=0.0,
        )
        self.assertGreater(psr, 0.5)

    def test_min_track_record_length(self):
        mtrl = _min_track_record_length(
            observed_sr=0.05,
            benchmark_sr=0.0,
            skew=0.0,
            kurtosis=0.0,
        )
        self.assertGreater(mtrl, 2)

    def test_pearson_corr_perfect(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = _pearson_corr(x, y)
        self.assertAlmostEqual(r, 1.0, 2)


class TestRiskAdjustedWithEquityCurve(unittest.TestCase):
    """Test Sharpe/Sortino/drawdown from equity snapshots."""

    def test_sharpe_from_snapshots(self):
        # Create synthetic equity curve: steady growth
        snapshots = []
        equity = 100000.0
        hwm = equity
        for i in range(30):
            equity += 100  # steady $100/day gain
            hwm = max(hwm, equity)
            dd = (hwm - equity) / hwm * 100
            snapshots.append(EquitySnapshot(
                timestamp=f"2026-03-{i+1:02d}",
                equity=round(equity, 2),
                cash=50000.0,
                market_value=round(equity - 50000, 2),
                drawdown_pct=round(dd, 4),
                high_water_mark=round(hwm, 2),
            ))

        trades = [
            JournalEntry(
                trade_id="T_1",
                ticker="X",
                strategy="test",
                side="long",
                entry_order_id="1",
                entry_signal_price=100.0,
                realized_pnl=100.0,
                r_multiple=1.0,
                status="closed",
                closed_at="2026-03-15",
            )
        ]
        m = compute_journal_metrics(trades, snapshots)
        # Steady gains → high Sharpe
        self.assertGreater(m.risk_adjusted.sharpe_ratio, 0)
        # No drawdown (monotonic increase)
        self.assertAlmostEqual(m.drawdown.max_drawdown_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
