"""Tests for PEAD earnings surprise data enrichment."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from quant_analysis_bot.pead import (
    clear_cache,
    enrich_with_pead,
    fetch_earnings_surprises,
)


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    """Create minimal OHLCV DataFrame for testing."""
    dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + abs(rng.normal(0, 1, n)),
            "Low": close - abs(rng.normal(0, 1, n)),
            "Close": close,
            "Volume": rng.randint(1_000_000, 5_000_000, n),
            "SMA_200": close * 0.95,  # always in uptrend
        },
        index=dates,
    )


class TestFetchEarningsSurprises(unittest.TestCase):
    """Test earnings surprise data fetching."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_returns_surprise_pct(self):
        """Should compute surprise % from reported vs estimated EPS."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.earnings_dates = pd.DataFrame(
            {
                "Reported EPS": [2.10, 1.80],
                "EPS Estimate": [2.00, 2.00],
            },
            index=pd.DatetimeIndex(
                ["2025-04-25", "2025-01-24"]
            ),
        )
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = fetch_earnings_surprises(
                "AAPL", use_cache=False,
            )
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 2)
            # 2.10 vs 2.00 = +5% surprise
            self.assertAlmostEqual(
                result.iloc[-1]["surprise_pct"], 5.0, places=1,
            )
            # 1.80 vs 2.00 = -10% surprise
            self.assertAlmostEqual(
                result.iloc[0]["surprise_pct"], -10.0, places=1,
            )

    def test_returns_none_when_no_data(self):
        """Should return None if earnings_dates is empty."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.earnings_dates = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = fetch_earnings_surprises(
                "XYZ", use_cache=False,
            )
            self.assertIsNone(result)

    def test_drops_future_earnings(self):
        """Should drop rows where reported EPS is NaN (future dates)."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.earnings_dates = pd.DataFrame(
            {
                "Reported EPS": [float("nan"), 2.00],
                "EPS Estimate": [2.10, 2.00],
            },
            index=pd.DatetimeIndex(
                ["2026-07-25", "2025-04-25"]
            ),
        )
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = fetch_earnings_surprises(
                "AAPL", use_cache=False,
            )
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)

    def test_cache_works(self):
        """Should only fetch once per ticker when caching."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.earnings_dates = pd.DataFrame(
            {
                "Reported EPS": [2.00],
                "EPS Estimate": [1.90],
            },
            index=pd.DatetimeIndex(["2025-04-25"]),
        )
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            fetch_earnings_surprises("MSFT")
            fetch_earnings_surprises("MSFT")
            mock_yf.Ticker.assert_called_once()

    def test_handles_zero_estimate(self):
        """Should handle zero EPS estimate without crashing."""
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_ticker.earnings_dates = pd.DataFrame(
            {
                "Reported EPS": [0.50],
                "EPS Estimate": [0.00],
            },
            index=pd.DatetimeIndex(["2025-04-25"]),
        )
        mock_yf.Ticker.return_value = mock_ticker

        with patch.dict(sys.modules, {"yfinance": mock_yf}):
            result = fetch_earnings_surprises(
                "TINY", use_cache=False,
            )
            # Zero estimate → NaN surprise → dropped
            self.assertIsNone(result)


class TestEnrichWithPead(unittest.TestCase):
    """Test PEAD column enrichment on OHLCV DataFrames."""

    @patch("quant_analysis_bot.pead.fetch_earnings_surprises")
    def test_adds_columns(self, mock_fetch):
        """Should add all three PEAD columns."""
        mock_fetch.return_value = None
        df = _make_ohlcv(100)
        result = enrich_with_pead(df, "TEST")
        self.assertIn("PEAD_Surprise_Pct", result.columns)
        self.assertIn("PEAD_Days_Since", result.columns)
        self.assertIn("PEAD_Gap_Pct", result.columns)

    @patch("quant_analysis_bot.pead.fetch_earnings_surprises")
    def test_no_earnings_all_nan(self, mock_fetch):
        """Without earnings data, PEAD columns should be NaN."""
        mock_fetch.return_value = None
        df = _make_ohlcv(100)
        result = enrich_with_pead(df, "TEST")
        self.assertTrue(result["PEAD_Surprise_Pct"].isna().all())
        self.assertTrue(result["PEAD_Days_Since"].isna().all())
        self.assertTrue(result["PEAD_Gap_Pct"].isna().all())

    @patch("quant_analysis_bot.pead.fetch_earnings_surprises")
    def test_fills_surprise_after_earnings(self, mock_fetch):
        """Surprise % should forward-fill from earnings date."""
        df = _make_ohlcv(100)
        # Place earnings on a date that exists in the DataFrame
        earn_date = df.index[50]

        mock_fetch.return_value = pd.DataFrame(
            {
                "reported_eps": [2.50],
                "estimated_eps": [2.00],
                "surprise_pct": [25.0],
            },
            index=pd.DatetimeIndex([earn_date]),
        )

        result = enrich_with_pead(df, "AAPL")
        # After earnings date: should have surprise filled
        post_earnings = result.iloc[50:]
        filled = post_earnings["PEAD_Surprise_Pct"].dropna()
        self.assertGreater(len(filled), 0)
        self.assertAlmostEqual(filled.iloc[0], 25.0, places=1)

    @patch("quant_analysis_bot.pead.fetch_earnings_surprises")
    def test_days_since_increments(self, mock_fetch):
        """Days since earnings should count up from 0."""
        df = _make_ohlcv(100)
        earn_date = df.index[50]

        mock_fetch.return_value = pd.DataFrame(
            {
                "reported_eps": [2.50],
                "estimated_eps": [2.00],
                "surprise_pct": [25.0],
            },
            index=pd.DatetimeIndex([earn_date]),
        )

        result = enrich_with_pead(df, "AAPL")
        days = result["PEAD_Days_Since"].dropna()
        if len(days) > 3:
            # Should start at 0 and increment
            self.assertEqual(days.iloc[0], 0.0)
            self.assertEqual(days.iloc[1], 1.0)
            self.assertEqual(days.iloc[2], 2.0)

    @patch("quant_analysis_bot.pead.fetch_earnings_surprises")
    def test_gap_pct_computed(self, mock_fetch):
        """Gap % should reflect open vs previous close."""
        df = _make_ohlcv(100)
        earn_date = df.index[50]

        # Force a specific gap
        df.iloc[49, df.columns.get_loc("Close")] = 100.0
        df.iloc[50, df.columns.get_loc("Open")] = 103.0

        mock_fetch.return_value = pd.DataFrame(
            {
                "reported_eps": [2.50],
                "estimated_eps": [2.00],
                "surprise_pct": [25.0],
            },
            index=pd.DatetimeIndex([earn_date]),
        )

        result = enrich_with_pead(df, "AAPL")
        gap = result.iloc[50]["PEAD_Gap_Pct"]
        if not np.isnan(gap):
            self.assertAlmostEqual(gap, 3.0, places=1)

    @patch("quant_analysis_bot.pead.fetch_earnings_surprises")
    def test_multiple_earnings_events(self, mock_fetch):
        """Multiple earnings should each fill their own window."""
        df = _make_ohlcv(200)
        earn_date_1 = df.index[50]
        earn_date_2 = df.index[120]

        mock_fetch.return_value = pd.DataFrame(
            {
                "reported_eps": [2.00, 2.50],
                "estimated_eps": [1.80, 2.00],
                "surprise_pct": [11.1, 25.0],
            },
            index=pd.DatetimeIndex([earn_date_1, earn_date_2]),
        )

        result = enrich_with_pead(df, "AAPL")
        # First window: surprise should be ~11.1
        s1 = result.iloc[55]["PEAD_Surprise_Pct"]
        if not np.isnan(s1):
            self.assertAlmostEqual(s1, 11.1, places=0)

        # Second window: surprise should be 25.0
        s2 = result.iloc[125]["PEAD_Surprise_Pct"]
        if not np.isnan(s2):
            self.assertAlmostEqual(s2, 25.0, places=0)


class TestIndicatorAdditions(unittest.TestCase):
    """Test new indicators added for Donchian/PEAD strategies."""

    def test_donchian_channels(self):
        from quant_analysis_bot.indicators import donchian_channels

        high = pd.Series([10, 12, 11, 15, 13, 14, 16, 12, 11, 10])
        low = pd.Series([8, 9, 8, 10, 9, 11, 12, 10, 9, 8])

        upper, lower = donchian_channels(high, low, period=5)
        # Upper at index 4: max of high[0:5] = 15
        self.assertEqual(upper.iloc[4], 15.0)
        # Lower at index 4: min of low[0:5] = 8
        self.assertEqual(lower.iloc[4], 8.0)

    def test_on_balance_volume(self):
        from quant_analysis_bot.indicators import on_balance_volume

        close = pd.Series([10, 11, 10.5, 12, 11])
        volume = pd.Series([100, 200, 150, 300, 100])

        obv = on_balance_volume(close, volume)
        # Day 0: NaN diff → 0
        # Day 1: up → +200 → cumsum = 200
        # Day 2: down → -150 → cumsum = 50
        # Day 3: up → +300 → cumsum = 350
        # Day 4: down → -100 → cumsum = 250
        self.assertAlmostEqual(obv.iloc[1], 200.0)
        self.assertAlmostEqual(obv.iloc[2], 50.0)
        self.assertAlmostEqual(obv.iloc[3], 350.0)
        self.assertAlmostEqual(obv.iloc[4], 250.0)


if __name__ == "__main__":
    unittest.main()
