"""Tests for ADV liquidity filter."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from trading_bot_bl.liquidity import (
    LiquidityInfo,
    _evaluate_liquidity,
    check_liquidity,
    clear_cache,
)


class TestLiquidityInfo(unittest.TestCase):
    """Test LiquidityInfo dataclass."""

    def test_defaults(self):
        info = LiquidityInfo(ticker="TEST")
        self.assertTrue(info.passes)
        self.assertIsNone(info.avg_daily_volume)
        self.assertEqual(info.rejection_reason, "")

    def test_frozen(self):
        info = LiquidityInfo(ticker="TEST")
        with self.assertRaises(AttributeError):
            info.ticker = "OTHER"


class TestEvaluateLiquidity(unittest.TestCase):
    """Test the liquidity evaluation logic directly."""

    def test_passes_all_checks(self):
        """Liquid stock with small position → passes."""
        info = _evaluate_liquidity(
            "AAPL",
            adv_shares=50_000_000,
            adv_dollars=5_000_000_000.0,
            position_notional=10_000.0,
            min_adv_shares=500_000,
            min_dollar_volume=5_000_000.0,
            max_participation_pct=1.0,
        )
        self.assertTrue(info.passes)
        self.assertEqual(info.rejection_reason, "")
        self.assertIsNotNone(info.position_pct_of_adv)
        self.assertLess(info.position_pct_of_adv, 0.01)

    def test_fails_min_shares(self):
        """Low share volume → rejected."""
        info = _evaluate_liquidity(
            "TINY",
            adv_shares=100_000,
            adv_dollars=1_000_000.0,
            position_notional=5_000.0,
            min_adv_shares=500_000,
            min_dollar_volume=5_000_000.0,
            max_participation_pct=1.0,
        )
        self.assertFalse(info.passes)
        self.assertIn("100,000 shares", info.rejection_reason)
        self.assertIn("500,000", info.rejection_reason)

    def test_fails_min_dollar_volume(self):
        """High share count but low-priced stock → rejected."""
        info = _evaluate_liquidity(
            "PENNY",
            adv_shares=2_000_000,  # plenty of shares
            adv_dollars=2_000_000.0,  # but only $2M daily
            position_notional=5_000.0,
            min_adv_shares=500_000,
            min_dollar_volume=5_000_000.0,
            max_participation_pct=1.0,
        )
        self.assertFalse(info.passes)
        self.assertIn("dollar volume", info.rejection_reason)

    def test_fails_participation_rate(self):
        """Position too large relative to ADV → rejected."""
        info = _evaluate_liquidity(
            "SMALL",
            adv_shares=1_000_000,
            adv_dollars=10_000_000.0,  # $10M daily
            position_notional=200_000.0,  # $200K = 2% of ADV
            min_adv_shares=500_000,
            min_dollar_volume=5_000_000.0,
            max_participation_pct=1.0,
        )
        self.assertFalse(info.passes)
        self.assertIn("2.00%", info.rejection_reason)
        self.assertIn("max 1.0%", info.rejection_reason)

    def test_passes_at_exact_participation_limit(self):
        """Position at exactly max participation → passes."""
        info = _evaluate_liquidity(
            "MID",
            adv_shares=1_000_000,
            adv_dollars=10_000_000.0,
            position_notional=100_000.0,  # exactly 1%
            min_adv_shares=500_000,
            min_dollar_volume=5_000_000.0,
            max_participation_pct=1.0,
        )
        self.assertTrue(info.passes)

    def test_no_dollar_volume_skips_dollar_checks(self):
        """When dollar volume is None, skip dollar-based checks."""
        info = _evaluate_liquidity(
            "NODOLLAR",
            adv_shares=1_000_000,
            adv_dollars=None,
            position_notional=5_000.0,
            min_adv_shares=500_000,
            min_dollar_volume=5_000_000.0,
            max_participation_pct=1.0,
        )
        self.assertTrue(info.passes)

    def test_share_check_runs_before_dollar_check(self):
        """Share volume check should fail first if both fail."""
        info = _evaluate_liquidity(
            "BOTH_FAIL",
            adv_shares=100_000,  # fails shares
            adv_dollars=1_000_000.0,  # also fails dollars
            position_notional=50_000.0,
            min_adv_shares=500_000,
            min_dollar_volume=5_000_000.0,
            max_participation_pct=1.0,
        )
        self.assertFalse(info.passes)
        # Should mention shares, not dollars (shares checked first)
        self.assertIn("shares", info.rejection_reason)


class TestCheckLiquidity(unittest.TestCase):
    """Test the full check function with mocked yfinance."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    @patch("trading_bot_bl.liquidity.fetch_avg_daily_volume")
    def test_liquid_stock_passes(self, mock_fetch):
        mock_fetch.return_value = (50_000_000, 8_000_000_000.0)
        info = check_liquidity(
            "AAPL", 10_000.0, use_cache=False,
        )
        self.assertTrue(info.passes)
        self.assertEqual(info.avg_daily_volume, 50_000_000)

    @patch("trading_bot_bl.liquidity.fetch_avg_daily_volume")
    def test_illiquid_stock_rejected(self, mock_fetch):
        mock_fetch.return_value = (200_000, 3_000_000.0)
        info = check_liquidity(
            "TINY", 10_000.0, use_cache=False,
        )
        self.assertFalse(info.passes)
        self.assertIn("200,000", info.rejection_reason)

    @patch("trading_bot_bl.liquidity.fetch_avg_daily_volume")
    def test_no_data_passes_through(self, mock_fetch):
        """Graceful degradation: no data → let it through."""
        mock_fetch.return_value = (None, None)
        info = check_liquidity(
            "UNKNOWN", 10_000.0, use_cache=False,
        )
        self.assertTrue(info.passes)
        self.assertIsNone(info.avg_daily_volume)

    @patch("trading_bot_bl.liquidity.fetch_avg_daily_volume")
    def test_cache_works(self, mock_fetch):
        mock_fetch.return_value = (10_000_000, 1_000_000_000.0)

        check_liquidity("MSFT", 5_000.0)
        check_liquidity("MSFT", 5_000.0)

        self.assertEqual(mock_fetch.call_count, 1)

    @patch("trading_bot_bl.liquidity.fetch_avg_daily_volume")
    def test_cache_bypass(self, mock_fetch):
        mock_fetch.return_value = (10_000_000, 1_000_000_000.0)

        check_liquidity("GOOG", 5_000.0, use_cache=False)
        check_liquidity("GOOG", 5_000.0, use_cache=False)

        self.assertEqual(mock_fetch.call_count, 2)

    @patch("trading_bot_bl.liquidity.fetch_avg_daily_volume")
    def test_cache_reevaluates_with_new_notional(self, mock_fetch):
        """Cached volume should be re-evaluated with new position size."""
        mock_fetch.return_value = (1_000_000, 10_000_000.0)

        # Small position → passes (0.05% of ADV)
        info1 = check_liquidity("MID", 5_000.0)
        self.assertTrue(info1.passes)

        # Large position → fails (2% of ADV, over 1% limit)
        info2 = check_liquidity("MID", 200_000.0)
        self.assertFalse(info2.passes)

        # Only one yfinance call
        self.assertEqual(mock_fetch.call_count, 1)

    @patch("trading_bot_bl.liquidity.fetch_avg_daily_volume")
    def test_custom_thresholds(self, mock_fetch):
        """Custom thresholds should be respected."""
        mock_fetch.return_value = (300_000, 3_000_000.0)

        # Default thresholds → rejected
        info = check_liquidity(
            "SMALL", 10_000.0, use_cache=False,
        )
        self.assertFalse(info.passes)

        # Relaxed thresholds → passes
        info = check_liquidity(
            "SMALL", 10_000.0,
            min_adv_shares=100_000,
            min_dollar_volume=1_000_000.0,
            max_participation_pct=5.0,
            use_cache=False,
        )
        self.assertTrue(info.passes)


class TestRiskManagerLiquidityGate(unittest.TestCase):
    """Test ADV liquidity integration in RiskManager."""

    @patch("trading_bot_bl.risk.check_liquidity")
    def test_blocks_illiquid_buy(self, mock_check):
        """Buy orders for illiquid stocks should be rejected."""
        from trading_bot_bl.config import RiskLimits
        from trading_bot_bl.risk import RiskManager

        mock_check.return_value = LiquidityInfo(
            ticker="TINY",
            avg_daily_volume=100_000,
            passes=False,
            rejection_reason="ADV 100,000 shares below minimum 500,000",
        )

        limits = RiskLimits(adv_liquidity_enabled=True)
        rm = RiskManager(limits=limits)

        result = rm.check_adv_liquidity("TINY", 10_000.0)
        self.assertIsNotNone(result)
        self.assertIn("ADV", result)

    @patch("trading_bot_bl.risk.check_liquidity")
    def test_passes_liquid_stock(self, mock_check):
        """Buy orders for liquid stocks should pass."""
        from trading_bot_bl.config import RiskLimits
        from trading_bot_bl.risk import RiskManager

        mock_check.return_value = LiquidityInfo(
            ticker="AAPL",
            avg_daily_volume=50_000_000,
            passes=True,
        )

        limits = RiskLimits(adv_liquidity_enabled=True)
        rm = RiskManager(limits=limits)

        result = rm.check_adv_liquidity("AAPL", 10_000.0)
        self.assertIsNone(result)

    @patch("trading_bot_bl.risk.check_liquidity")
    def test_disabled_skips_check(self, mock_check):
        """When disabled, liquidity check is not called."""
        from trading_bot_bl.config import RiskLimits
        from trading_bot_bl.models import (
            OrderIntent,
            PortfolioSnapshot,
            Signal,
        )
        from trading_bot_bl.risk import RiskManager

        limits = RiskLimits(
            adv_liquidity_enabled=False,
            # Loosen other limits so the order gets through
            min_composite_score=0,
            min_confidence_score=0,
            min_backtest_trades=0,
            max_pbo=1.0,
            earnings_blackout_enabled=False,
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
        mock_check.assert_not_called()


class TestFetchAvgDailyVolume(unittest.TestCase):
    """Test fetch_avg_daily_volume with mocked yfinance."""

    def test_with_10day_adv(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "averageDailyVolume10Day": 5_000_000,
            "averageVolume": 4_000_000,
            "currentPrice": 150.0,
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from trading_bot_bl.liquidity import (
                fetch_avg_daily_volume,
            )
            shares, dollars = fetch_avg_daily_volume("AAPL")
            self.assertEqual(shares, 5_000_000)
            self.assertAlmostEqual(dollars, 750_000_000.0)

    def test_fallback_to_average_volume(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "averageDailyVolume10Day": None,
            "averageVolume": 3_000_000,
            "currentPrice": 100.0,
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from trading_bot_bl.liquidity import (
                fetch_avg_daily_volume,
            )
            shares, dollars = fetch_avg_daily_volume("MSFT")
            self.assertEqual(shares, 3_000_000)
            self.assertAlmostEqual(dollars, 300_000_000.0)

    def test_no_volume_data(self):
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from trading_bot_bl.liquidity import (
                fetch_avg_daily_volume,
            )
            shares, dollars = fetch_avg_daily_volume("XYZ")
            self.assertIsNone(shares)
            self.assertIsNone(dollars)

    def test_price_fallback_chain(self):
        """Should try currentPrice → regularMarketPrice → previousClose."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "averageVolume": 1_000_000,
            "currentPrice": None,
            "regularMarketPrice": None,
            "previousClose": 50.0,
        }
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            from trading_bot_bl.liquidity import (
                fetch_avg_daily_volume,
            )
            shares, dollars = fetch_avg_daily_volume("OLD")
            self.assertEqual(shares, 1_000_000)
            self.assertAlmostEqual(dollars, 50_000_000.0)


if __name__ == "__main__":
    unittest.main()
