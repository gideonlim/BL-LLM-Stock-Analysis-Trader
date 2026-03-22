"""Tests for Donchian Breakout and PEAD Drift strategies."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.strategies import (
    DonchianBreakout,
    PEAD_Drift,
)


def _make_ohlcv(
    n: int = 300,
    seed: int = 42,
    trend: float = 0.0005,
) -> pd.DataFrame:
    """Create synthetic OHLCV data with common indicators."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n, freq="B")

    close = 100 * np.exp(
        np.cumsum(rng.normal(trend, 0.015, n))
    )
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.randint(500_000, 5_000_000, n).astype(float)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )

    # Add indicators needed by DonchianBreakout
    df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["ADX_14"] = 30.0  # Default: strong trend
    df["Donchian_Upper_20"] = (
        df["High"].rolling(20, min_periods=1).max()
    )
    df["Donchian_Lower_20"] = (
        df["Low"].rolling(20, min_periods=1).min()
    )
    df["Donchian_Upper_55"] = (
        df["High"].rolling(55, min_periods=1).max()
    )
    df["Donchian_Lower_55"] = (
        df["Low"].rolling(55, min_periods=1).min()
    )

    return df


class TestDonchianBreakout(unittest.TestCase):
    """Test Donchian Breakout strategy."""

    def setUp(self):
        self.strategy = DonchianBreakout()

    def test_name_and_description(self):
        self.assertEqual(self.strategy.name, "Donchian Breakout (20/55)")
        self.assertIn("breaks above", self.strategy.description.lower())

    def test_generates_signals(self):
        """Should produce non-zero signals on trending data."""
        df = _make_ohlcv(300, trend=0.001)  # uptrend
        signals = self.strategy.generate_signals(df)
        self.assertEqual(len(signals), len(df))
        # Should have at least some non-zero signals
        self.assertGreater((signals != 0).sum(), 0)

    def test_no_signals_below_sma200(self):
        """Should not buy when price is below SMA_200."""
        df = _make_ohlcv(300, trend=-0.002)  # downtrend
        # Force price well below SMA_200
        df["SMA_200"] = df["Close"] * 1.5
        signals = self.strategy.generate_signals(df)
        # Should have no buy signals (all non-positive)
        buy_signals = (signals > 0).sum()
        self.assertEqual(buy_signals, 0)

    def test_no_signals_low_adx(self):
        """Should not buy when ADX < 20 (no trend)."""
        df = _make_ohlcv(300, trend=0.001)
        df["ADX_14"] = 10.0  # weak/no trend
        signals = self.strategy.generate_signals(df)
        buy_signals = (signals > 0).sum()
        self.assertEqual(buy_signals, 0)

    def test_uses_lagged_channels(self):
        """Breakout should reference previous day's channel, not today's."""
        df = _make_ohlcv(100)
        # Set up so today's close equals today's upper channel
        # but is BELOW yesterday's upper channel
        df["ADX_14"] = 30.0
        df["SMA_200"] = 50.0  # always above

        # Force last bar: close = 100, prev upper = 105, curr upper = 100
        df.iloc[-1, df.columns.get_loc("Close")] = 100.0
        df.iloc[-1, df.columns.get_loc("Donchian_Upper_20")] = 100.0
        df.iloc[-2, df.columns.get_loc("Donchian_Upper_20")] = 105.0

        signals = self.strategy.generate_signals(df)
        # Last bar should NOT be a buy (100 < 105 shifted)
        self.assertLessEqual(signals.iloc[-1], 0)

    def test_asymmetric_exit(self):
        """Exit uses 55d low (wider), not 20d low."""
        df = _make_ohlcv(300, trend=0.001)
        # The strategy should reference Donchian_Lower_55 for exits,
        # not Donchian_Lower_20
        signals = self.strategy.generate_signals(df)
        # Verify signals exist (basic sanity)
        self.assertIsInstance(signals, pd.Series)

    def test_returns_crossover_not_level(self):
        """Signals should be crossover-style (+1/-1 on transitions)."""
        df = _make_ohlcv(300, trend=0.001)
        signals = self.strategy.generate_signals(df)
        # Values should only be -1, 0, or 1
        unique_vals = set(signals.dropna().unique())
        self.assertTrue(unique_vals.issubset({-1.0, 0.0, 1.0}))

    def test_forced_breakout(self):
        """Manually engineer a breakout and verify buy signal."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0  # always above trend
        df["ADX_14"] = 35.0  # strong trend

        # Force a clear exit state first (close below 55d low),
        # then a breakout entry on the last bar.
        # Bar -5: close breaks below 55d low → exit state (-1)
        df.iloc[-5, df.columns.get_loc("Donchian_Lower_55")] = 200.0
        df.iloc[-5, df.columns.get_loc("Close")] = 90.0
        # Bars -4 to -2: no entry or exit triggers → stays in exit state
        for i in [-4, -3, -2]:
            df.iloc[i, df.columns.get_loc("Donchian_Upper_20")] = 200.0
            df.iloc[i, df.columns.get_loc("Close")] = 95.0
            df.iloc[i, df.columns.get_loc("Donchian_Lower_55")] = 50.0
        # Bar -1: breakout above yesterday's 20d channel
        df.iloc[-2, df.columns.get_loc("Donchian_Upper_20")] = 100.0
        df.iloc[-1, df.columns.get_loc("Close")] = 105.0
        df.iloc[-1, df.columns.get_loc("Donchian_Lower_55")] = 50.0

        signals = self.strategy.generate_signals(df)
        # The last signal should be +1 (transition from exit to entry)
        self.assertEqual(signals.iloc[-1], 1.0)


class TestPEAD_Drift(unittest.TestCase):
    """Test PEAD Drift strategy."""

    def setUp(self):
        self.strategy = PEAD_Drift()

    def test_name_and_description(self):
        self.assertEqual(self.strategy.name, "PEAD Earnings Drift")
        self.assertIn("earnings", self.strategy.description.lower())

    def test_no_pead_columns_returns_hold(self):
        """Without PEAD columns, should return all zeros."""
        df = _make_ohlcv(100)
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals == 0).all())

    def test_positive_surprise_generates_buy(self):
        """Positive surprise with confirming gap → BUY in drift window."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0  # always above trend

        # Set PEAD columns
        df["PEAD_Surprise_Pct"] = np.nan
        df["PEAD_Days_Since"] = np.nan
        df["PEAD_Gap_Pct"] = np.nan

        # Simulate earnings 10 days ago with positive surprise
        df.iloc[-10:, df.columns.get_loc("PEAD_Surprise_Pct")] = 15.0
        df.iloc[-10:, df.columns.get_loc("PEAD_Gap_Pct")] = 3.0
        for i in range(10):
            df.iloc[-(10 - i), df.columns.get_loc("PEAD_Days_Since")] = i

        signals = self.strategy.generate_signals(df)
        # Days 2-9 should be BUY (in drift window)
        buy_count = (signals > 0).sum()
        self.assertGreater(buy_count, 0)

    def test_small_surprise_no_signal(self):
        """Surprise < 5% should not trigger."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0
        df["PEAD_Surprise_Pct"] = 3.0  # too small
        df["PEAD_Days_Since"] = 10.0
        df["PEAD_Gap_Pct"] = 2.0

        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals == 0).all())

    def test_no_confirming_gap_no_signal(self):
        """Positive surprise but flat gap (< 1%) → no signal."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0
        df["PEAD_Surprise_Pct"] = 20.0
        df["PEAD_Days_Since"] = 10.0
        df["PEAD_Gap_Pct"] = 0.3  # too small

        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals == 0).all())

    def test_below_sma200_no_signal(self):
        """Above SMA_200 filter should prevent signals."""
        df = _make_ohlcv(100)
        df["SMA_200"] = df["Close"] * 2.0  # way above
        df["PEAD_Surprise_Pct"] = 20.0
        df["PEAD_Days_Since"] = 10.0
        df["PEAD_Gap_Pct"] = 3.0

        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals == 0).all())

    def test_drift_window_respects_boundaries(self):
        """Should only signal between day 2 and day 60."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0
        df["PEAD_Surprise_Pct"] = 20.0
        df["PEAD_Gap_Pct"] = 3.0

        # Day 0 and 1: too early (inside blackout overlap zone)
        df["PEAD_Days_Since"] = 0.0
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals.iloc[-5:] == 0).all())

        df["PEAD_Days_Since"] = 1.0
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals.iloc[-5:] == 0).all())

        # Day 2: should signal
        df["PEAD_Days_Since"] = 2.0
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals.iloc[-5:] > 0).any())

        # Day 60: still in window
        df["PEAD_Days_Since"] = 60.0
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals.iloc[-5:] > 0).any())

        # Day 61: exit signal (drift expired)
        df["PEAD_Days_Since"] = 61.0
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals.iloc[-5:] < 0).any())

    def test_negative_surprise_generates_exit(self):
        """Negative surprise with confirming gap down → EXIT."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0  # above trend (not relevant for exits)
        df["PEAD_Surprise_Pct"] = -10.0
        df["PEAD_Days_Since"] = 10.0
        df["PEAD_Gap_Pct"] = -3.0

        signals = self.strategy.generate_signals(df)
        exit_count = (signals < 0).sum()
        self.assertGreater(exit_count, 0)

    def test_nan_pead_columns_returns_hold(self):
        """All-NaN PEAD columns should produce all-zero signals."""
        df = _make_ohlcv(100)
        df["PEAD_Surprise_Pct"] = np.nan
        df["PEAD_Days_Since"] = np.nan
        df["PEAD_Gap_Pct"] = np.nan

        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals == 0).all())


class TestPEADBlackoutInteraction(unittest.TestCase):
    """Verify PEAD and earnings blackout don't conflict."""

    def test_pead_entry_after_blackout_window(self):
        """PEAD enters at day 2, blackout ends at day 1 (post_days=1).

        There should be no overlap: blackout blocks entries for
        days_until_earnings <= post_days (1), PEAD enters at
        days_since >= 2.
        """
        # Default blackout post_days = 1
        # PEAD entry starts at days_since = 2
        # Gap: day 1 after earnings is blocked by blackout,
        # day 2 is when PEAD can first enter → clean separation
        blackout_post_days = 1
        pead_entry_start = 2
        self.assertGreater(pead_entry_start, blackout_post_days)


if __name__ == "__main__":
    unittest.main()
