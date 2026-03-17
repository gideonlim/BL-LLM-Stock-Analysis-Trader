"""
Tests for the position monitor -- covers all 7 health check scenarios.

Verifies:
1. Emergency loss detection and auto-close
2. Orphaned positions (no SL/TP) — losing closed, winners rebracketed
3. Partial brackets (one leg missing) — flagged as warning
4. Price gapping below SL — force close
5. Price surging above TP — flagged but not closed
6. Breakeven stop — SL moved to entry once profitable enough
7. Trailing stop — SL tightened to lock in gains
8. Time-based exit — close positions held too long
9. Dry run mode — no actual orders placed
10. Edge cases — no positions, missing price data, etc.

Run with:
    python -m unittest trading_bot_bl.test_monitor -v
"""

from __future__ import annotations

import sys
import os
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(__file__))
)

# Mock alpaca dependencies before importing any trading_bot_bl
# modules, since broker.py raises ImportError without alpaca-py.
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

from trading_bot_bl.config import RiskLimits
from trading_bot_bl.models import (
    OrderResult,
    PortfolioSnapshot,
    PositionAlert,
)
from trading_bot_bl.monitor import (
    MonitorReport,
    monitor_positions,
    _calculate_trailing_stop,
    _calculate_breakeven_stop,
    _check_time_exit,
)


# ── Helpers ─────────────────────────────────────────────────


def _make_order(
    symbol: str,
    stop_price: float | None = None,
    limit_price: float | None = None,
    order_type: str = "",
    order_id: str = "ord-1",
    order_class: str = "",
    legs: list | None = None,
) -> SimpleNamespace:
    """Create a fake Alpaca order object."""
    return SimpleNamespace(
        symbol=symbol,
        stop_price=stop_price,
        limit_price=limit_price,
        order_type=order_type,
        order_class=order_class,
        legs=legs,
        id=order_id,
    )


def _make_portfolio(
    positions: dict[str, dict],
) -> PortfolioSnapshot:
    """Create a portfolio snapshot with given positions."""
    return PortfolioSnapshot(
        equity=100_000.0,
        cash=50_000.0,
        buying_power=50_000.0,
        market_value=50_000.0,
        positions=positions,
    )


def _make_broker(
    orders: list | None = None,
    prices: dict[str, float] | None = None,
    close_result_status: str = "submitted",
) -> MagicMock:
    """Create a mock broker with configurable responses."""
    broker = MagicMock()
    broker.get_open_orders.return_value = orders or []
    broker.get_latest_prices.return_value = prices or {}
    broker.close_position.return_value = OrderResult(
        ticker="", status=close_result_status, side="close",
    )
    broker.cancel_order.return_value = True
    # Mock _client for reattach_bracket
    broker._client = MagicMock()
    return broker


def _default_limits() -> RiskLimits:
    """Default risk limits for tests."""
    return RiskLimits(
        emergency_loss_pct=8.0,
        stale_bracket_pct=5.0,
        auto_close_orphaned_losers=True,
        orphan_max_loss_pct=5.0,
        max_hold_days=10,
    )


# ── Check 1: Emergency loss ────────────────────────────────


class TestEmergencyLoss(unittest.TestCase):
    """Position down more than emergency_loss_pct from entry."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_emergency_close_triggered(self, _atr: MagicMock) -> None:
        """Position down 10% (> 8% threshold) should be closed."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 4500.0,
                "unrealized_pnl": -500.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 90.0},
            orders=[],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.emergency_count, 1)
        self.assertEqual(len(report.actions), 1)
        broker.close_position.assert_called_once_with("AAPL")
        alert = report.alerts[0]
        self.assertEqual(alert.alert_type, "emergency_loss")
        self.assertEqual(alert.severity, "critical")

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_emergency_close_dry_run(self, _atr: MagicMock) -> None:
        """Dry run should NOT call close_position."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 4500.0,
                "unrealized_pnl": -500.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(prices={"AAPL": 90.0})

        report = monitor_positions(
            broker, portfolio, self.limits, dry_run=True
        )

        self.assertEqual(report.emergency_count, 1)
        self.assertEqual(len(report.actions), 0)
        broker.close_position.assert_not_called()
        self.assertIn("DRY RUN", report.alerts[0].action_taken)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_emergency_not_triggered_below_threshold(
        self, _atr: MagicMock
    ) -> None:
        """Position down 5% (< 8%) should NOT trigger emergency."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 4750.0,
                "unrealized_pnl": -250.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 95.0},
            orders=[
                _make_order(
                    "AAPL", stop_price=90.0, order_id="sl-1"
                ),
                _make_order(
                    "AAPL", limit_price=115.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.emergency_count, 0)
        broker.close_position.assert_not_called()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_emergency_skips_remaining_checks(
        self, _atr: MagicMock
    ) -> None:
        """Emergency close should skip orphan/bracket checks."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 4000.0,
                "unrealized_pnl": -1000.0,
                "qty": 50.0,
                "entry_date": (
                    date.today() - timedelta(days=20)
                ).isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 85.0},
            orders=[],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.emergency_count, 1)
        self.assertEqual(report.orphaned_count, 0)
        self.assertEqual(len(report.alerts), 1)


# ── Check 2: Orphaned positions (no SL/TP) ─────────────────


class TestOrphanedPositions(unittest.TestCase):
    """Positions with no protective bracket orders at all."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_orphaned_loser_auto_closed(
        self, _atr: MagicMock
    ) -> None:
        """Orphaned losing > orphan_max_loss_pct is closed."""
        portfolio = _make_portfolio({
            "TSLA": {
                "avg_entry": 200.0,
                "market_value": 9400.0,
                "unrealized_pnl": -600.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"TSLA": 188.0},
            orders=[],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.orphaned_count, 1)
        broker.close_position.assert_called_once_with("TSLA")
        alert = report.alerts[0]
        self.assertEqual(alert.alert_type, "orphaned")
        self.assertEqual(alert.severity, "critical")

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_orphaned_loser_not_closed_when_disabled(
        self, _atr: MagicMock
    ) -> None:
        """auto_close_orphaned_losers=False → reattach."""
        limits = RiskLimits(
            auto_close_orphaned_losers=False,
            orphan_max_loss_pct=5.0,
        )
        portfolio = _make_portfolio({
            "TSLA": {
                "avg_entry": 200.0,
                "market_value": 9400.0,
                "unrealized_pnl": -600.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"TSLA": 188.0},
            orders=[],
        )

        report = monitor_positions(broker, portfolio, limits)

        self.assertEqual(report.orphaned_count, 1)
        broker.close_position.assert_not_called()
        # OCO order: single submit_order call (SL + TP linked)
        self.assertEqual(
            broker._client.submit_order.call_count, 1
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_orphaned_winner_rebracketed(
        self, _atr: MagicMock
    ) -> None:
        """Orphaned winning position gets new SL/TP attached."""
        portfolio = _make_portfolio({
            "MSFT": {
                "avg_entry": 300.0,
                "market_value": 16000.0,
                "unrealized_pnl": 1000.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"MSFT": 320.0},
            orders=[],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.orphaned_count, 1)
        broker.close_position.assert_not_called()
        # OCO order: single submit_order call (SL + TP linked)
        self.assertEqual(
            broker._client.submit_order.call_count, 1
        )
        self.assertEqual(
            report.alerts[0].severity, "warning"
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_orphaned_small_loss_rebracketed(
        self, _atr: MagicMock
    ) -> None:
        """Orphaned with small loss (< threshold) → rebracketed."""
        portfolio = _make_portfolio({
            "GOOG": {
                "avg_entry": 150.0,
                "market_value": 7350.0,
                "unrealized_pnl": -150.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"GOOG": 147.0},
            orders=[],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.orphaned_count, 1)
        broker.close_position.assert_not_called()
        # OCO order: single submit_order call (SL + TP linked)
        self.assertEqual(
            broker._client.submit_order.call_count, 1
        )


# ── Check 2b: Stop-limit orders not misclassified ───────────


class TestStopLimitClassification(unittest.TestCase):
    """Bracket legs with both stop_price and limit_price
    (stop-limit orders) must be recognized as SL legs, not
    misclassified as orphaned."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_stop_limit_order_detected_as_sl(
        self, _atr: MagicMock
    ) -> None:
        """Stop-limit order (both stop_price and limit_price)
        should be classified as SL, not orphaned."""
        portfolio = _make_portfolio({
            "CI": {
                "avg_entry": 350.0,
                "market_value": 15400.0,
                "unrealized_pnl": 0.0,
                "qty": 44.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"CI": 350.0},
            orders=[
                # Stop-limit SL: has BOTH stop_price and
                # limit_price (the limit caps fill after stop
                # triggers). Old code missed this case.
                _make_order(
                    "CI",
                    stop_price=330.0,
                    limit_price=328.0,
                    order_id="sl-1",
                ),
                # Normal limit TP
                _make_order(
                    "CI",
                    limit_price=385.0,
                    order_id="tp-1",
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        # Should NOT be orphaned — both legs are present
        self.assertEqual(report.orphaned_count, 0)
        broker.close_position.assert_not_called()
        # No reattach attempts
        self.assertEqual(
            broker._client.submit_order.call_count, 0
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_stop_limit_with_only_sl_leg(
        self, _atr: MagicMock
    ) -> None:
        """Stop-limit SL with no TP → partial bracket, not
        fully orphaned."""
        portfolio = _make_portfolio({
            "CI": {
                "avg_entry": 350.0,
                "market_value": 15400.0,
                "unrealized_pnl": 0.0,
                "qty": 44.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"CI": 350.0},
            orders=[
                _make_order(
                    "CI",
                    stop_price=330.0,
                    limit_price=328.0,
                    order_id="sl-1",
                ),
                # No TP order
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        # Should be partial (1 leg missing), not fully orphaned
        self.assertEqual(report.orphaned_count, 1)
        partial_alerts = [
            a for a in report.alerts
            if "take profit" in a.message.lower()
        ]
        self.assertEqual(len(partial_alerts), 1)
        broker.close_position.assert_not_called()
        # Partial bracket now triggers OCO reattach
        # (cancels stale SL, places fresh OCO with both legs)
        broker.cancel_order.assert_called()
        self.assertEqual(
            broker._client.submit_order.call_count, 1
        )


# ── Check 2c: OCO orders recognised as complete bracket ──────

@patch(
    "trading_bot_bl.monitor._fetch_atr", return_value=0.0
)
class TestOCOOrderClassification(unittest.TestCase):
    """OCO parent with legs should be recognised as full bracket."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    def test_oco_parent_with_legs_detected(
        self, _atr: MagicMock
    ) -> None:
        """OCO parent order with SL+TP legs is NOT orphaned."""
        portfolio = _make_portfolio({
            "CI": {
                "avg_entry": 268.0,
                "market_value": 11792.0,
                "unrealized_pnl": 0.0,
                "qty": 44.0,
                "entry_date": date.today().isoformat(),
            }
        })
        # OCO parent is a limit sell (TP), with a stop child (SL)
        sl_leg = SimpleNamespace(
            id="sl-leg-1",
            stop_price=254.94,
            limit_price=None,
            order_type="stop",
        )
        tp_leg = SimpleNamespace(
            id="tp-leg-1",
            stop_price=None,
            limit_price=295.20,
            order_type="limit",
        )
        broker = _make_broker(
            prices={"CI": 268.36},
            orders=[
                _make_order(
                    "CI",
                    limit_price=295.20,
                    order_id="oco-parent-1",
                    order_class="oco",
                    legs=[sl_leg, tp_leg],
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.orphaned_count, 0)
        broker.close_position.assert_not_called()
        broker._client.submit_order.assert_not_called()

    def test_oco_parent_no_legs_uses_parent(
        self, _atr: MagicMock
    ) -> None:
        """OCO parent without legs attr falls back to standard
        classification (limit_price only = TP, missing SL)."""
        portfolio = _make_portfolio({
            "CI": {
                "avg_entry": 268.0,
                "market_value": 11792.0,
                "unrealized_pnl": 0.0,
                "qty": 44.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"CI": 268.36},
            orders=[
                _make_order(
                    "CI",
                    limit_price=295.20,
                    order_id="oco-parent-1",
                    order_class="oco",
                    legs=None,
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        # No legs → standard classification → only TP seen
        # → partial bracket → reattach
        self.assertEqual(report.orphaned_count, 1)


# ── Check 3: Partial brackets (one leg missing) ────────────


class TestPartialBrackets(unittest.TestCase):
    """Positions with only SL or only TP — not both."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_missing_stop_loss_reattached(
        self, _atr: MagicMock
    ) -> None:
        """Position with TP but no SL should be reattached via OCO."""
        portfolio = _make_portfolio({
            "NVDA": {
                "avg_entry": 500.0,
                "market_value": 25500.0,
                "unrealized_pnl": 500.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"NVDA": 510.0},
            orders=[
                _make_order(
                    "NVDA", limit_price=550.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.orphaned_count, 1)
        broker.close_position.assert_not_called()
        alert = report.alerts[0]
        self.assertEqual(alert.severity, "warning")
        self.assertIn("stop loss", alert.message.lower())
        # Stale TP cancelled + OCO reattach placed
        broker.cancel_order.assert_called()
        self.assertEqual(
            broker._client.submit_order.call_count, 1
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_missing_take_profit_reattached(
        self, _atr: MagicMock
    ) -> None:
        """Position with SL but no TP should be reattached via OCO."""
        portfolio = _make_portfolio({
            "AMZN": {
                "avg_entry": 180.0,
                "market_value": 9200.0,
                "unrealized_pnl": 200.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AMZN": 184.0},
            orders=[
                _make_order(
                    "AMZN", stop_price=170.0, order_id="sl-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.orphaned_count, 1)
        broker.close_position.assert_not_called()
        self.assertIn(
            "take profit", report.alerts[0].message.lower()
        )
        # Stale SL cancelled + OCO reattach placed
        broker.cancel_order.assert_called()
        self.assertEqual(
            broker._client.submit_order.call_count, 1
        )


# ── Check 4: Price outside bracket range ────────────────────


class TestPriceOutsideBracket(unittest.TestCase):
    """Price gapped beyond SL/TP without triggering."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_price_gapped_below_sl_closes(
        self, _atr: MagicMock
    ) -> None:
        """Price well below SL (>2% gap) should force close."""
        portfolio = _make_portfolio({
            "META": {
                "avg_entry": 400.0,
                "market_value": 18500.0,
                "unrealized_pnl": -1500.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"META": 370.0},
            orders=[
                _make_order(
                    "META", stop_price=380.0, order_id="sl-1"
                ),
                _make_order(
                    "META", limit_price=440.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        broker.close_position.assert_called_once_with("META")
        self.assertEqual(report.emergency_count, 1)
        gap_alerts = [
            a for a in report.alerts
            if a.alert_type == "extreme_move"
        ]
        self.assertTrue(len(gap_alerts) > 0)
        self.assertIn(
            "gapped below SL", gap_alerts[0].message
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_price_surged_above_tp_not_closed(
        self, _atr: MagicMock
    ) -> None:
        """Price above TP (>2%) is flagged but NOT closed."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 150.0,
                "market_value": 9000.0,
                "unrealized_pnl": 1500.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 175.0},
            orders=[
                _make_order(
                    "AAPL", stop_price=140.0, order_id="sl-1"
                ),
                _make_order(
                    "AAPL", limit_price=170.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        broker.close_position.assert_not_called()
        surge_alerts = [
            a for a in report.alerts
            if "surged above TP" in a.message
        ]
        self.assertEqual(len(surge_alerts), 1)


# ── Check 5: Breakeven stop ────────────────────────────────


class TestBreakevenStop(unittest.TestCase):
    """Move SL to entry once position is profitable enough."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=5.0
    )
    def test_breakeven_triggered_with_atr(
        self, _atr: MagicMock
    ) -> None:
        """Position up > 1×ATR should move SL to breakeven."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 5300.0,
                "unrealized_pnl": 300.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 106.0},
            orders=[
                _make_order(
                    "AAPL", stop_price=92.0, order_id="sl-1"
                ),
                _make_order(
                    "AAPL", limit_price=115.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        be_alerts = [
            a for a in report.alerts
            if a.alert_type == "breakeven_stop"
        ]
        self.assertEqual(len(be_alerts), 1)
        self.assertIn(
            "breakeven", be_alerts[0].message.lower()
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_breakeven_triggered_pct_fallback(
        self, _atr: MagicMock
    ) -> None:
        """Without ATR, breakeven triggers at 3% gain."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 5200.0,
                "unrealized_pnl": 200.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 104.0},
            orders=[
                _make_order(
                    "AAPL", stop_price=92.0, order_id="sl-1"
                ),
                _make_order(
                    "AAPL", limit_price=115.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        be_alerts = [
            a for a in report.alerts
            if a.alert_type == "breakeven_stop"
        ]
        self.assertEqual(len(be_alerts), 1)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_breakeven_skipped_when_sl_above_entry(
        self, _atr: MagicMock
    ) -> None:
        """If SL already at/above entry, skip breakeven."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 5500.0,
                "unrealized_pnl": 500.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 110.0},
            orders=[
                _make_order(
                    "AAPL", stop_price=102.0, order_id="sl-1"
                ),
                _make_order(
                    "AAPL", limit_price=120.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        be_alerts = [
            a for a in report.alerts
            if a.alert_type == "breakeven_stop"
        ]
        self.assertEqual(len(be_alerts), 0)


# ── Check 6: Trailing stop ─────────────────────────────────


class TestTrailingStop(unittest.TestCase):
    """Tighten SL to lock in gains on profitable positions."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=2.0
    )
    def test_trailing_stop_tightened_atr(
        self, _atr: MagicMock
    ) -> None:
        """Position up >5%: trail SL to price - 2×ATR."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 5600.0,
                "unrealized_pnl": 600.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 112.0},
            orders=[
                _make_order(
                    "AAPL", stop_price=93.0, order_id="sl-1"
                ),
                _make_order(
                    "AAPL", limit_price=120.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        stale_alerts = [
            a for a in report.alerts
            if a.alert_type == "stale_bracket"
            and "tightening" in a.message.lower()
        ]
        self.assertGreaterEqual(len(stale_alerts), 1)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_trailing_stop_pct_fallback(
        self, _atr: MagicMock
    ) -> None:
        """Without ATR: trail at entry + 50% of gain."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 5600.0,
                "unrealized_pnl": 600.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 112.0},
            orders=[
                _make_order(
                    "AAPL", stop_price=93.0, order_id="sl-1"
                ),
                _make_order(
                    "AAPL", limit_price=120.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        stale_alerts = [
            a for a in report.alerts
            if a.alert_type == "stale_bracket"
            and "tightening" in a.message.lower()
        ]
        self.assertGreaterEqual(len(stale_alerts), 1)


# ── Check 7: Time-based exit ───────────────────────────────


class TestTimeExit(unittest.TestCase):
    """Close positions held longer than max_hold_days."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_time_exit_triggered(
        self, _atr: MagicMock
    ) -> None:
        """Position held 15 days (> 10 max) should be closed."""
        entry_date = (
            date.today() - timedelta(days=15)
        ).isoformat()
        portfolio = _make_portfolio({
            "AMD": {
                "avg_entry": 120.0,
                "market_value": 6100.0,
                "unrealized_pnl": 100.0,
                "qty": 50.0,
                "entry_date": entry_date,
            }
        })
        broker = _make_broker(
            prices={"AMD": 122.0},
            orders=[
                _make_order(
                    "AMD", stop_price=110.0, order_id="sl-1"
                ),
                _make_order(
                    "AMD", limit_price=140.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        broker.close_position.assert_called_once_with("AMD")
        time_alerts = [
            a for a in report.alerts
            if a.alert_type == "time_exit"
        ]
        self.assertEqual(len(time_alerts), 1)
        self.assertIn(
            "exceeded max hold period",
            time_alerts[0].message,
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_time_exit_not_triggered_within_limit(
        self, _atr: MagicMock
    ) -> None:
        """Position held 5 days (< 10 max) should NOT trigger."""
        entry_date = (
            date.today() - timedelta(days=5)
        ).isoformat()
        portfolio = _make_portfolio({
            "AMD": {
                "avg_entry": 120.0,
                "market_value": 6100.0,
                "unrealized_pnl": 100.0,
                "qty": 50.0,
                "entry_date": entry_date,
            }
        })
        broker = _make_broker(
            prices={"AMD": 122.0},
            orders=[
                _make_order(
                    "AMD", stop_price=110.0, order_id="sl-1"
                ),
                _make_order(
                    "AMD", limit_price=140.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        time_alerts = [
            a for a in report.alerts
            if a.alert_type == "time_exit"
        ]
        self.assertEqual(len(time_alerts), 0)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_time_exit_missing_entry_date(
        self, _atr: MagicMock
    ) -> None:
        """No entry_date → skip time check (don't crash)."""
        portfolio = _make_portfolio({
            "AMD": {
                "avg_entry": 120.0,
                "market_value": 6100.0,
                "unrealized_pnl": 100.0,
                "qty": 50.0,
                "entry_date": None,
            }
        })
        broker = _make_broker(
            prices={"AMD": 122.0},
            orders=[
                _make_order(
                    "AMD", stop_price=110.0, order_id="sl-1"
                ),
                _make_order(
                    "AMD", limit_price=140.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        time_alerts = [
            a for a in report.alerts
            if a.alert_type == "time_exit"
        ]
        self.assertEqual(len(time_alerts), 0)


# ── Dry run mode ────────────────────────────────────────────


class TestDryRun(unittest.TestCase):
    """Dry run should detect issues but take no broker actions."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_dry_run_orphaned_loser_not_closed(
        self, _atr: MagicMock
    ) -> None:
        """Dry run: orphaned loser flagged but not closed."""
        # -6% loss: above orphan threshold (5%) but below
        # emergency threshold (8%) so it hits the orphan path.
        portfolio = _make_portfolio({
            "TSLA": {
                "avg_entry": 200.0,
                "market_value": 9400.0,
                "unrealized_pnl": -600.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"TSLA": 188.0},
            orders=[],
        )

        report = monitor_positions(
            broker, portfolio, self.limits, dry_run=True
        )

        self.assertEqual(report.orphaned_count, 1)
        broker.close_position.assert_not_called()
        self.assertIn(
            "DRY RUN", report.alerts[0].action_taken
        )

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_dry_run_time_exit_not_closed(
        self, _atr: MagicMock
    ) -> None:
        """Dry run: stale position flagged but not closed."""
        entry_date = (
            date.today() - timedelta(days=15)
        ).isoformat()
        portfolio = _make_portfolio({
            "AMD": {
                "avg_entry": 120.0,
                "market_value": 6100.0,
                "unrealized_pnl": 100.0,
                "qty": 50.0,
                "entry_date": entry_date,
            }
        })
        broker = _make_broker(
            prices={"AMD": 122.0},
            orders=[
                _make_order(
                    "AMD", stop_price=110.0, order_id="sl-1"
                ),
                _make_order(
                    "AMD", limit_price=140.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits, dry_run=True
        )

        broker.close_position.assert_not_called()
        self.assertEqual(len(report.actions), 0)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_dry_run_gap_close_not_executed(
        self, _atr: MagicMock
    ) -> None:
        """Dry run: gapped position flagged but not closed."""
        portfolio = _make_portfolio({
            "META": {
                "avg_entry": 400.0,
                "market_value": 18500.0,
                "unrealized_pnl": -1500.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"META": 370.0},
            orders=[
                _make_order(
                    "META", stop_price=380.0, order_id="sl-1"
                ),
                _make_order(
                    "META", limit_price=440.0, order_id="tp-1"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits, dry_run=True
        )

        broker.close_position.assert_not_called()
        self.assertEqual(len(report.actions), 0)


# ── Edge cases ──────────────────────────────────────────────


class TestEdgeCases(unittest.TestCase):
    """Edge cases: empty portfolio, missing data, etc."""

    def setUp(self) -> None:
        self.limits = _default_limits()

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_empty_portfolio(self, _atr: MagicMock) -> None:
        """No positions → clean report, no errors."""
        portfolio = _make_portfolio({})
        broker = _make_broker()

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.positions_checked, 0)
        self.assertEqual(len(report.alerts), 0)
        self.assertEqual(len(report.actions), 0)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_missing_price_data_skips(
        self, _atr: MagicMock
    ) -> None:
        """Position with no price data should be skipped."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 5000.0,
                "unrealized_pnl": 0.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(prices={}, orders=[])

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.positions_checked, 1)
        self.assertEqual(len(report.alerts), 0)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_multiple_positions_each_checked(
        self, _atr: MagicMock
    ) -> None:
        """Multiple positions independently checked."""
        entry_date = date.today().isoformat()
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 100.0,
                "market_value": 5000.0,
                "unrealized_pnl": 0.0,
                "qty": 50.0,
                "entry_date": entry_date,
            },
            "TSLA": {
                "avg_entry": 200.0,
                "market_value": 9000.0,
                "unrealized_pnl": -1000.0,
                "qty": 50.0,
                "entry_date": entry_date,
            },
            "MSFT": {
                "avg_entry": 300.0,
                "market_value": 16000.0,
                "unrealized_pnl": 1000.0,
                "qty": 50.0,
                "entry_date": entry_date,
            },
        })
        broker = _make_broker(
            prices={
                "AAPL": 100.0,
                "TSLA": 180.0,
                "MSFT": 320.0,
            },
            orders=[
                _make_order(
                    "AAPL", stop_price=90.0, order_id="sl-a"
                ),
                _make_order(
                    "AAPL", limit_price=115.0, order_id="tp-a"
                ),
            ],
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(report.positions_checked, 3)
        self.assertGreaterEqual(report.emergency_count, 1)
        self.assertGreaterEqual(report.orphaned_count, 1)

    @patch(
        "trading_bot_bl.monitor._fetch_atr", return_value=0.0
    )
    def test_zero_entry_price_skips(
        self, _atr: MagicMock
    ) -> None:
        """Position with 0 entry price should be skipped."""
        portfolio = _make_portfolio({
            "AAPL": {
                "avg_entry": 0.0,
                "market_value": 5000.0,
                "unrealized_pnl": 0.0,
                "qty": 50.0,
                "entry_date": date.today().isoformat(),
            }
        })
        broker = _make_broker(
            prices={"AAPL": 100.0}, orders=[]
        )

        report = monitor_positions(
            broker, portfolio, self.limits
        )

        self.assertEqual(len(report.alerts), 0)


# ── Unit tests for pure calculation functions ───────────────


class TestCalculateTrailingStop(unittest.TestCase):
    """Unit tests for _calculate_trailing_stop."""

    def test_atr_based_trail(self) -> None:
        """ATR-based: new SL = price - 2×ATR, floored at entry."""
        result = _calculate_trailing_stop(
            entry_price=100.0,
            current_price=120.0,
            current_sl=90.0,
            atr=5.0,
        )
        self.assertEqual(result, 110.0)

    def test_atr_trail_floored_at_entry(self) -> None:
        """ATR trail shouldn't go below entry price."""
        result = _calculate_trailing_stop(
            entry_price=100.0,
            current_price=105.0,
            current_sl=90.0,
            atr=4.0,
        )
        self.assertEqual(result, 100.0)

    def test_pct_fallback(self) -> None:
        """Without ATR: trail at entry + 50% of gain."""
        result = _calculate_trailing_stop(
            entry_price=100.0,
            current_price=120.0,
            current_sl=90.0,
            atr=0.0,
        )
        self.assertEqual(result, 110.0)

    def test_no_improvement_returns_none(self) -> None:
        """If new SL <= current SL, return None."""
        result = _calculate_trailing_stop(
            entry_price=100.0,
            current_price=105.0,
            current_sl=104.0,
            atr=3.0,
        )
        self.assertIsNone(result)

    def test_losing_position_returns_none(self) -> None:
        """Losing position: gain <= 0 → return None."""
        result = _calculate_trailing_stop(
            entry_price=100.0,
            current_price=95.0,
            current_sl=90.0,
            atr=2.0,
        )
        self.assertIsNone(result)


class TestCalculateBreakevenStop(unittest.TestCase):
    """Unit tests for _calculate_breakeven_stop."""

    def test_atr_threshold_met(self) -> None:
        """Gain > 1×ATR → return entry as new SL."""
        result = _calculate_breakeven_stop(
            entry_price=100.0,
            current_price=108.0,
            current_sl=92.0,
            atr=5.0,
        )
        self.assertEqual(result, 100.0)

    def test_atr_threshold_not_met(self) -> None:
        """Gain < 1×ATR → return None."""
        result = _calculate_breakeven_stop(
            entry_price=100.0,
            current_price=103.0,
            current_sl=92.0,
            atr=5.0,
        )
        self.assertIsNone(result)

    def test_pct_fallback_triggered(self) -> None:
        """No ATR: gain > 3% → return entry as SL."""
        result = _calculate_breakeven_stop(
            entry_price=100.0,
            current_price=104.0,
            current_sl=92.0,
            atr=0.0,
        )
        self.assertEqual(result, 100.0)

    def test_pct_fallback_not_met(self) -> None:
        """No ATR: gain < 3% → return None."""
        result = _calculate_breakeven_stop(
            entry_price=100.0,
            current_price=102.0,
            current_sl=92.0,
            atr=0.0,
        )
        self.assertIsNone(result)

    def test_already_at_breakeven(self) -> None:
        """SL already >= entry → return None."""
        result = _calculate_breakeven_stop(
            entry_price=100.0,
            current_price=110.0,
            current_sl=100.0,
            atr=5.0,
        )
        self.assertIsNone(result)

    def test_losing_position(self) -> None:
        """Losing position (gain <= 0) → return None."""
        result = _calculate_breakeven_stop(
            entry_price=100.0,
            current_price=95.0,
            current_sl=90.0,
            atr=5.0,
        )
        self.assertIsNone(result)


class TestCheckTimeExit(unittest.TestCase):
    """Unit tests for _check_time_exit."""

    def test_exceeded_max_hold(self) -> None:
        entry = (
            date.today() - timedelta(days=15)
        ).isoformat()
        self.assertTrue(
            _check_time_exit("AAPL", entry, 10)
        )

    def test_within_max_hold(self) -> None:
        entry = (
            date.today() - timedelta(days=5)
        ).isoformat()
        self.assertFalse(
            _check_time_exit("AAPL", entry, 10)
        )

    def test_exactly_at_max_hold(self) -> None:
        """Exactly at max_hold_days → NOT triggered (> not >=)."""
        entry = (
            date.today() - timedelta(days=10)
        ).isoformat()
        self.assertFalse(
            _check_time_exit("AAPL", entry, 10)
        )

    def test_no_entry_date(self) -> None:
        self.assertFalse(
            _check_time_exit("AAPL", None, 10)
        )

    def test_iso_datetime_format(self) -> None:
        """Entry date with time component should work."""
        entry = (
            date.today() - timedelta(days=15)
        ).isoformat() + "T09:30:00"
        self.assertTrue(
            _check_time_exit("AAPL", entry, 10)
        )


# ── MonitorReport ───────────────────────────────────────────


class TestMonitorReport(unittest.TestCase):
    """Tests for the MonitorReport dataclass."""

    def test_has_critical_true(self) -> None:
        report = MonitorReport()
        report.alerts.append(
            PositionAlert(
                ticker="AAPL",
                alert_type="emergency_loss",
                severity="critical",
                message="test",
            )
        )
        self.assertTrue(report.has_critical)

    def test_has_critical_false(self) -> None:
        report = MonitorReport()
        report.alerts.append(
            PositionAlert(
                ticker="AAPL",
                alert_type="orphaned",
                severity="warning",
                message="test",
            )
        )
        self.assertFalse(report.has_critical)

    def test_summary_all_counts(self) -> None:
        report = MonitorReport(
            positions_checked=5,
            orphaned_count=2,
            stale_count=1,
            emergency_count=1,
        )
        summary = report.summary()
        self.assertIn("5 positions", summary)
        self.assertIn("2 orphaned", summary)
        self.assertIn("1 stale", summary)
        self.assertIn("1 emergency", summary)

    def test_summary_clean(self) -> None:
        report = MonitorReport(positions_checked=3)
        summary = report.summary()
        self.assertIn("3 positions", summary)
        self.assertNotIn("orphaned", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
