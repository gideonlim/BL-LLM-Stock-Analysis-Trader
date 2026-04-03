"""Tests for the earnings event filter in signal generation.

Covers:
  - EarningsContext dataclass behaviour
  - compute_earnings_confidence_adj() adjustment rules
  - generate_daily_signal() integration with earnings context
  - build_earnings_context() from enriched DataFrames
"""

from __future__ import annotations

import math
import unittest
from dataclasses import asdict
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from quant_analysis_bot.signals import (
    EarningsContext,
    compute_earnings_confidence_adj,
    generate_daily_signal,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_minimal_df(n: int = 300) -> pd.DataFrame:
    """Create minimal enriched DataFrame for signal generation."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.randint(1_000_000, 5_000_000, n),
        },
        index=dates,
    )
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["RSI_14"] = 50 + rng.normal(0, 10, n)
    df["Volatility_20"] = 0.2
    df["Vol_Ratio"] = 1.0
    df["ZScore_20"] = 0.0
    df["ADX_14"] = 20.0
    df["ATR_14"] = close * 0.02
    return df


def _make_backtest_result():
    """Create a minimal BacktestResult for signal generation."""
    from quant_analysis_bot.models import BacktestResult

    result = BacktestResult(
        strategy_name="SMA_Crossover",
        ticker="TEST",
        timeframe="12mo",
        backtest_start="2024-01-01",
        backtest_end="2025-01-01",
        trading_days=252,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        win_rate=0.60,
        total_trades=20,
        profit_factor=2.0,
        avg_holding_days=5.0,
        max_drawdown_pct=-10.0,
        annual_return_pct=15.0,
        annual_excess_pct=5.0,
    )
    # composite_score is set dynamically by the backtest scorer
    result.composite_score = 55.0
    return result


def _make_strategy():
    """Create a stub strategy that always returns BUY."""

    class StubStrategy:
        name = "StubBuy"

        def generate_signals(self, df):
            s = pd.Series(0, index=df.index)
            s.iloc[-1] = 1  # BUY on last bar
            return s

    return StubStrategy()


def _default_config():
    return {
        "risk_profile": "moderate",
        "min_sharpe": 0.5,
        "long_only": True,
    }


# ── EarningsContext Tests ────────────────────────────────────────────


class TestEarningsContext(unittest.TestCase):
    """Test EarningsContext dataclass."""

    def test_defaults_are_neutral(self):
        ctx = EarningsContext()
        self.assertEqual(ctx.days_to_earnings, -1)
        self.assertEqual(ctx.earnings_date, "")
        self.assertTrue(math.isnan(ctx.last_surprise_pct))
        self.assertFalse(ctx.is_available)

    def test_is_available_when_days_set(self):
        ctx = EarningsContext(days_to_earnings=5)
        self.assertTrue(ctx.is_available)

    def test_is_available_at_zero(self):
        ctx = EarningsContext(days_to_earnings=0)
        self.assertTrue(ctx.is_available)

    def test_frozen(self):
        ctx = EarningsContext()
        with self.assertRaises(AttributeError):
            ctx.days_to_earnings = 10


# ── Confidence Adjustment Tests ──────────────────────────────────────


class TestComputeEarningsConfidenceAdj(unittest.TestCase):
    """Test the earnings confidence adjustment logic."""

    def test_neutral_when_unavailable(self):
        """No adjustment when earnings data is unavailable."""
        ctx = EarningsContext()
        self.assertEqual(compute_earnings_confidence_adj(ctx), 0)

    def test_earnings_today_max_penalty(self):
        """Earnings TODAY should give -3 penalty."""
        ctx = EarningsContext(
            days_to_earnings=0,
            earnings_date="2026-05-01",
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), -3)

    def test_earnings_1_day_away_penalty(self):
        """1 day before earnings should give -2."""
        ctx = EarningsContext(
            days_to_earnings=1,
            earnings_date="2026-05-02",
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), -2)

    def test_earnings_3_days_away_penalty(self):
        """Exactly at blackout boundary (3d) should give -2."""
        ctx = EarningsContext(
            days_to_earnings=3,
            earnings_date="2026-05-04",
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), -2)

    def test_earnings_4_days_away_no_penalty(self):
        """4 days away (outside default 3d blackout) = no penalty."""
        ctx = EarningsContext(
            days_to_earnings=4,
            earnings_date="2026-05-05",
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), 0)

    def test_positive_surprise_boost(self):
        """Strong positive surprise (>5%) should give +1."""
        ctx = EarningsContext(
            days_to_earnings=30,
            last_surprise_pct=10.0,
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), 1)

    def test_negative_surprise_penalty(self):
        """Strong negative surprise (<-5%) should give -1."""
        ctx = EarningsContext(
            days_to_earnings=30,
            last_surprise_pct=-8.0,
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), -1)

    def test_mild_surprise_no_adjustment(self):
        """Small surprise (within ±5%) should give 0."""
        ctx = EarningsContext(
            days_to_earnings=30,
            last_surprise_pct=2.0,
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), 0)

    def test_combined_blackout_and_negative_surprise(self):
        """Near earnings + negative surprise = -2 + -1 = -3."""
        ctx = EarningsContext(
            days_to_earnings=2,
            earnings_date="2026-05-03",
            last_surprise_pct=-10.0,
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), -3)

    def test_custom_blackout_window(self):
        """Custom 5-day blackout should catch 5d away."""
        ctx = EarningsContext(
            days_to_earnings=5,
            earnings_date="2026-05-06",
        )
        adj = compute_earnings_confidence_adj(
            ctx, blackout_pre_days=5
        )
        self.assertEqual(adj, -2)

    def test_nan_surprise_is_neutral(self):
        """NaN surprise should not contribute to adjustment."""
        ctx = EarningsContext(
            days_to_earnings=30,
            last_surprise_pct=float("nan"),
        )
        self.assertEqual(compute_earnings_confidence_adj(ctx), 0)


# ── Signal Generation Integration Tests ──────────────────────────────


class TestSignalGenerationWithEarnings(unittest.TestCase):
    """Test that earnings context flows through to DailySignal."""

    def setUp(self):
        self.df = _make_minimal_df()
        self.result = _make_backtest_result()
        self.strategy = _make_strategy()
        self.config = _default_config()

    def test_no_earnings_context_defaults(self):
        """Without earnings context, defaults are neutral."""
        sig = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
        )
        self.assertEqual(sig.days_to_earnings, -1)
        self.assertEqual(sig.earnings_date, "")
        self.assertIsNone(sig.last_surprise_pct)
        self.assertEqual(sig.earnings_confidence_adj, 0)

    def test_earnings_context_flows_through(self):
        """Earnings context should populate DailySignal fields."""
        ctx = EarningsContext(
            days_to_earnings=5,
            earnings_date="2026-05-06",
            last_surprise_pct=7.5,
        )
        sig = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
            earnings_ctx=ctx,
        )
        self.assertEqual(sig.days_to_earnings, 5)
        self.assertEqual(sig.earnings_date, "2026-05-06")
        self.assertAlmostEqual(sig.last_surprise_pct, 7.5)
        self.assertEqual(sig.earnings_confidence_adj, 1)

    def test_earnings_penalty_reduces_confidence(self):
        """Near-earnings penalty should reduce confidence score."""
        # Without earnings context
        sig_no_earn = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
        )

        # With earnings tomorrow
        ctx = EarningsContext(
            days_to_earnings=1,
            earnings_date="2026-05-02",
        )
        sig_earn = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
            earnings_ctx=ctx,
        )

        # Confidence score should be lower with earnings
        self.assertLess(
            sig_earn.confidence_score,
            sig_no_earn.confidence_score,
        )
        self.assertEqual(sig_earn.earnings_confidence_adj, -2)

    def test_earnings_day_severe_penalty(self):
        """Earnings day should apply -3 penalty."""
        ctx = EarningsContext(
            days_to_earnings=0,
            earnings_date="2026-05-01",
        )
        sig = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
            earnings_ctx=ctx,
        )
        self.assertEqual(sig.earnings_confidence_adj, -3)

    def test_earnings_notes_added(self):
        """Earnings info should appear in signal notes."""
        ctx = EarningsContext(
            days_to_earnings=2,
            earnings_date="2026-05-03",
            last_surprise_pct=12.0,
        )
        sig = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
            earnings_ctx=ctx,
        )
        self.assertIn("Earnings in 2d", sig.notes)
        self.assertIn("Last surprise: +12.0%", sig.notes)
        # -2 (blackout) + 1 (positive surprise) = -1
        self.assertIn("Earnings conf adj: -1", sig.notes)

    def test_json_serializable(self):
        """DailySignal with earnings should serialize to JSON."""
        import json

        ctx = EarningsContext(
            days_to_earnings=10,
            earnings_date="2026-05-11",
            last_surprise_pct=3.0,
        )
        sig = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
            earnings_ctx=ctx,
        )
        d = asdict(sig)
        # Should not raise
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["days_to_earnings"], 10)
        self.assertEqual(parsed["earnings_date"], "2026-05-11")
        self.assertAlmostEqual(parsed["last_surprise_pct"], 3.0)

    def test_json_serializable_no_surprise(self):
        """DailySignal with None surprise should serialize to JSON."""
        import json

        sig = generate_daily_signal(
            self.df, "TEST", self.strategy,
            self.result, self.config,
        )
        d = asdict(sig)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        self.assertIsNone(parsed["last_surprise_pct"])

    def test_confidence_score_never_negative(self):
        """Confidence score should be floored at 0."""
        # Use a weak backtest result (low base confidence = 0)
        from quant_analysis_bot.models import BacktestResult

        weak_result = BacktestResult(
            strategy_name="Weak",
            ticker="TEST",
            sharpe_ratio=0.1,
            win_rate=0.40,
            profit_factor=0.8,
            total_trades=5,
            avg_holding_days=5.0,
            max_drawdown_pct=-10.0,
        )
        weak_result.composite_score = 10.0

        ctx = EarningsContext(
            days_to_earnings=0,
            earnings_date="2026-05-01",
        )
        sig = generate_daily_signal(
            self.df, "TEST", self.strategy,
            weak_result, self.config,
            earnings_ctx=ctx,
        )
        self.assertGreaterEqual(sig.confidence_score, 0)


# ── build_earnings_context Tests ─────────────────────────────────────


class TestBuildEarningsContext(unittest.TestCase):
    """Test build_earnings_context from enriched DataFrame."""

    def _make_df_with_pead(self, surprise: float = 5.0):
        """DataFrame with PEAD columns populated."""
        dates = pd.bdate_range("2024-01-01", periods=100, freq="B")
        df = pd.DataFrame(
            {"Close": 100.0, "Open": 99.5},
            index=dates,
        )
        df["PEAD_Surprise_Pct"] = np.nan
        df.iloc[-10:, df.columns.get_loc("PEAD_Surprise_Pct")] = (
            surprise
        )
        df["PEAD_Days_Since"] = np.nan
        df["PEAD_Gap_Pct"] = np.nan
        return df

    @patch.dict(
        "sys.modules",
        {"yfinance": MagicMock()},
    )
    def test_builds_context_with_forward_date(self):
        """Should extract next earnings date from yfinance."""
        import sys

        mock_yf = sys.modules["yfinance"]
        mock_ticker = MagicMock()
        mock_ticker.calendar = {
            "Earnings Date": [date.today() + timedelta(days=10)],
        }
        mock_yf.Ticker.return_value = mock_ticker

        from quant_analysis_bot.pead import build_earnings_context

        df = self._make_df_with_pead(surprise=8.0)
        ctx = build_earnings_context(df, "AAPL")

        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.days_to_earnings, 10)
        self.assertAlmostEqual(ctx.last_surprise_pct, 8.0)

    @patch.dict(
        "sys.modules",
        {"yfinance": MagicMock()},
    )
    def test_extracts_surprise_from_pead_columns(self):
        """Should get last surprise from PEAD_Surprise_Pct."""
        import sys

        mock_yf = sys.modules["yfinance"]
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        mock_yf.Ticker.return_value = mock_ticker

        from quant_analysis_bot.pead import build_earnings_context

        df = self._make_df_with_pead(surprise=-3.5)
        ctx = build_earnings_context(df, "TEST")

        # Should still get surprise even without forward date
        self.assertIsNotNone(ctx)
        self.assertAlmostEqual(ctx.last_surprise_pct, -3.5)

    @patch.dict(
        "sys.modules",
        {"yfinance": MagicMock()},
    )
    def test_returns_none_when_no_data(self):
        """Should return None when no earnings data at all."""
        import sys

        mock_yf = sys.modules["yfinance"]
        mock_ticker = MagicMock()
        mock_ticker.calendar = None
        mock_yf.Ticker.return_value = mock_ticker

        from quant_analysis_bot.pead import build_earnings_context

        dates = pd.bdate_range("2024-01-01", periods=50, freq="B")
        df = pd.DataFrame({"Close": 100.0}, index=dates)
        ctx = build_earnings_context(df, "NODATA")

        self.assertIsNone(ctx)


if __name__ == "__main__":
    unittest.main()
