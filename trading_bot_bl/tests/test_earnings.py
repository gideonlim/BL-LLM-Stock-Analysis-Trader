"""Tests for earnings blackout filter."""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from trading_bot_bl.earnings import (
    EarningsInfo,
    _evaluate_blackout,
    check_earnings_blackout,
    clear_cache,
)


class TestEarningsInfo(unittest.TestCase):
    """Test EarningsInfo dataclass."""

    def test_defaults(self):
        info = EarningsInfo(ticker="TEST")
        self.assertFalse(info.in_blackout)
        self.assertIsNone(info.next_earnings_date)
        self.assertEqual(info.blackout_reason, "")

    def test_frozen(self):
        info = EarningsInfo(ticker="TEST")
        with self.assertRaises(AttributeError):
            info.ticker = "OTHER"


class TestEvaluateBlackout(unittest.TestCase):
    """Test the blackout evaluation logic directly."""

    def test_outside_blackout_before(self):
        """10 days before earnings → not in blackout."""
        earnings = date(2026, 5, 1)
        today = date(2026, 4, 21)  # 10 days before
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertFalse(info.in_blackout)
        self.assertEqual(info.days_until_earnings, 10)

    def test_outside_blackout_after(self):
        """5 days after earnings → not in blackout."""
        earnings = date(2026, 5, 1)
        today = date(2026, 5, 6)  # 5 days after
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertFalse(info.in_blackout)

    def test_in_blackout_3_days_before(self):
        """Exactly 3 days before earnings → in blackout."""
        earnings = date(2026, 5, 1)
        today = date(2026, 4, 28)  # 3 days before
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertTrue(info.in_blackout)
        self.assertEqual(info.days_until_earnings, 3)
        self.assertIn("3 day(s)", info.blackout_reason)

    def test_in_blackout_1_day_before(self):
        """1 day before earnings → in blackout."""
        earnings = date(2026, 5, 1)
        today = date(2026, 4, 30)
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertTrue(info.in_blackout)
        self.assertEqual(info.days_until_earnings, 1)

    def test_in_blackout_earnings_day(self):
        """Earnings day itself → in blackout."""
        earnings = date(2026, 5, 1)
        today = date(2026, 5, 1)
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertTrue(info.in_blackout)
        self.assertEqual(info.days_until_earnings, 0)
        self.assertIn("TODAY", info.blackout_reason)

    def test_in_blackout_1_day_after(self):
        """1 day after earnings → in post-earnings blackout."""
        earnings = date(2026, 5, 1)
        today = date(2026, 5, 2)  # 1 day after
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertTrue(info.in_blackout)
        self.assertIn("ago", info.blackout_reason)

    def test_outside_blackout_2_days_after(self):
        """2 days after earnings → outside blackout (post_days=1)."""
        earnings = date(2026, 5, 1)
        today = date(2026, 5, 3)
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertFalse(info.in_blackout)

    def test_boundary_4_days_before(self):
        """4 days before earnings → outside blackout (pre_days=3)."""
        earnings = date(2026, 5, 1)
        today = date(2026, 4, 27)  # 4 days before
        info = _evaluate_blackout("TEST", earnings, today, 3, 1)
        self.assertFalse(info.in_blackout)
        self.assertEqual(info.days_until_earnings, 4)

    def test_custom_window(self):
        """Custom 5-day pre / 2-day post window."""
        earnings = date(2026, 5, 1)
        # 5 days before → in blackout with 5-day window
        today = date(2026, 4, 26)
        info = _evaluate_blackout("TEST", earnings, today, 5, 2)
        self.assertTrue(info.in_blackout)

        # 2 days after → in blackout with 2-day post window
        today = date(2026, 5, 3)
        info = _evaluate_blackout("TEST", earnings, today, 5, 2)
        self.assertTrue(info.in_blackout)


class TestCheckEarningsBlackout(unittest.TestCase):
    """Test the full check function with mocked yfinance."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    @patch("trading_bot_bl.earnings.fetch_earnings_date")
    def test_with_earnings_date(self, mock_fetch):
        """Should detect blackout when earnings date is available."""
        mock_fetch.return_value = date(2026, 4, 28)
        info = check_earnings_blackout(
            "AAPL", today=date(2026, 4, 26), use_cache=False,
        )
        self.assertTrue(info.in_blackout)
        self.assertEqual(info.days_until_earnings, 2)

    @patch("trading_bot_bl.earnings.fetch_earnings_date")
    def test_no_earnings_date(self, mock_fetch):
        """Should pass through when no earnings data available."""
        mock_fetch.return_value = None
        info = check_earnings_blackout(
            "XYZ", today=date(2026, 4, 26), use_cache=False,
        )
        self.assertFalse(info.in_blackout)
        self.assertIsNone(info.next_earnings_date)

    @patch("trading_bot_bl.earnings.fetch_earnings_date")
    def test_cache_works(self, mock_fetch):
        """Should only call yfinance once per ticker."""
        mock_fetch.return_value = date(2026, 5, 15)

        # First call
        check_earnings_blackout("MSFT", today=date(2026, 5, 1))
        # Second call — should use cache
        check_earnings_blackout("MSFT", today=date(2026, 5, 1))

        self.assertEqual(mock_fetch.call_count, 1)

    @patch("trading_bot_bl.earnings.fetch_earnings_date")
    def test_cache_bypass(self, mock_fetch):
        """use_cache=False should always call yfinance."""
        mock_fetch.return_value = date(2026, 5, 15)

        check_earnings_blackout(
            "GOOG", today=date(2026, 5, 1), use_cache=False,
        )
        check_earnings_blackout(
            "GOOG", today=date(2026, 5, 1), use_cache=False,
        )

        self.assertEqual(mock_fetch.call_count, 2)


class TestRiskManagerEarningsGate(unittest.TestCase):
    """Test earnings blackout integration in RiskManager."""

    @patch("trading_bot_bl.risk.check_earnings_blackout")
    def test_blocks_buy_in_blackout(self, mock_check):
        """Buy orders during earnings blackout should be rejected."""
        from trading_bot_bl.config import RiskLimits
        from trading_bot_bl.models import OrderIntent, Signal
        from trading_bot_bl.risk import RiskManager

        mock_check.return_value = EarningsInfo(
            ticker="AAPL",
            next_earnings_date=date(2026, 5, 1),
            days_until_earnings=2,
            in_blackout=True,
            blackout_reason="Earnings in 2 day(s) (2026-05-01)",
        )

        limits = RiskLimits(earnings_blackout_enabled=True)
        rm = RiskManager(limits=limits)

        result = rm.check_earnings_blackout("AAPL")
        self.assertIsNotNone(result)
        self.assertIn("Earnings", result)

    @patch("trading_bot_bl.risk.check_earnings_blackout")
    def test_passes_outside_blackout(self, mock_check):
        """Buy orders outside earnings blackout should pass."""
        from trading_bot_bl.config import RiskLimits
        from trading_bot_bl.risk import RiskManager

        mock_check.return_value = EarningsInfo(
            ticker="MSFT",
            next_earnings_date=date(2026, 5, 15),
            days_until_earnings=30,
            in_blackout=False,
        )

        limits = RiskLimits(earnings_blackout_enabled=True)
        rm = RiskManager(limits=limits)

        result = rm.check_earnings_blackout("MSFT")
        self.assertIsNone(result)

    @patch("trading_bot_bl.risk.check_earnings_blackout")
    def test_disabled_skips_check(self, mock_check):
        """When disabled, earnings check is not called."""
        from trading_bot_bl.config import RiskLimits
        from trading_bot_bl.models import (
            OrderIntent,
            PortfolioSnapshot,
            Signal,
        )
        from trading_bot_bl.risk import RiskManager

        limits = RiskLimits(
            earnings_blackout_enabled=False,
            # Loosen other limits so the order gets through
            min_composite_score=0,
            min_confidence_score=0,
            min_backtest_trades=0,
            max_pbo=1.0,
        )
        rm = RiskManager(limits=limits)

        signal = Signal(
            ticker="AAPL", signal="BUY", signal_raw=1,
            strategy="SMA", confidence="HIGH",
            confidence_score=5, composite_score=50.0,
            current_price=200.0, stop_loss_price=190.0,
            take_profit_price=220.0,
            suggested_position_size_pct=5.0,
            signal_expires="2026-06-01",
            sharpe=1.5, win_rate=60, total_trades=10,
        )
        intent = OrderIntent(
            ticker="AAPL", side="buy", notional=5000.0,
            stop_loss_price=190.0, take_profit_price=220.0,
            signal=signal,
        )
        portfolio = PortfolioSnapshot(
            equity=100000.0, cash=50000.0,
            buying_power=50000.0, market_value=50000.0,
        )

        verdict = rm.evaluate_order(intent, portfolio)
        # Should NOT have called check_earnings_blackout
        mock_check.assert_not_called()


class TestFetchEarningsDate(unittest.TestCase):
    """Test fetch_earnings_date with mocked yfinance."""

    def test_with_mock_yfinance(self):
        """Should extract date from .calendar dict."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.calendar = {
            "Earnings Date": [date(2026, 5, 1)],
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from trading_bot_bl.earnings import fetch_earnings_date
            result = fetch_earnings_date("AAPL")
            self.assertEqual(result, date(2026, 5, 1))

    def test_with_no_calendar(self):
        """Should return None when calendar is None."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from trading_bot_bl.earnings import fetch_earnings_date
            result = fetch_earnings_date("XYZ")
            self.assertIsNone(result)

    def test_with_multiple_dates(self):
        """Should return earliest date when multiple dates given."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.calendar = {
            "Earnings Date": [date(2026, 5, 3), date(2026, 5, 1)],
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from trading_bot_bl.earnings import fetch_earnings_date
            result = fetch_earnings_date("AAPL")
            self.assertEqual(result, date(2026, 5, 1))


if __name__ == "__main__":
    unittest.main()
