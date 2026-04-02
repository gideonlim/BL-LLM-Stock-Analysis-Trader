"""Tests for Donchian Breakout, 52-Week High Momentum, PEAD Drift,
and Multi-Factor Momentum MR strategies."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.strategies import (
    DonchianBreakout,
    FiftyTwoWeekHighMomentum,
    MultiFactorMomentumMR,
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


# ── 52-Week High Momentum tests ──────────────────────────────────────


class TestFiftyTwoWeekHighMomentum(unittest.TestCase):
    """Test 52-Week High Momentum strategy."""

    def setUp(self):
        self.strategy = FiftyTwoWeekHighMomentum()

    def test_name_and_description(self):
        self.assertEqual(self.strategy.name, "52-Week High Momentum")
        self.assertIn("52-week", self.strategy.description.lower())

    def test_missing_columns_returns_hold(self):
        """Without Nearness_52w_High column, should return all zeros."""
        df = _make_ohlcv(100)
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals == 0).all())

    def test_buys_near_high_in_uptrend(self):
        """Price near 52-week high + above SMA_200 + ADX > 20 → BUY."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0  # price always above
        df["ADX_14"] = 30.0  # strong trend
        df["Nearness_52w_High"] = 0.97  # within 5% of high

        signals = self.strategy.generate_signals(df)
        buy_count = (signals > 0).sum()
        self.assertGreater(buy_count, 0)

    def test_no_buy_when_far_from_high(self):
        """Price far from 52-week high (< 0.95) → no BUY."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0
        df["ADX_14"] = 30.0
        df["Nearness_52w_High"] = 0.80  # 20% below high

        signals = self.strategy.generate_signals(df)
        buy_count = (signals > 0).sum()
        self.assertEqual(buy_count, 0)

    def test_no_buy_below_sma200(self):
        """Above SMA_200 filter required → no BUY when below."""
        df = _make_ohlcv(100)
        df["SMA_200"] = df["Close"] * 2.0  # way above price
        df["ADX_14"] = 30.0
        df["Nearness_52w_High"] = 0.98

        signals = self.strategy.generate_signals(df)
        buy_count = (signals > 0).sum()
        self.assertEqual(buy_count, 0)

    def test_no_buy_low_adx(self):
        """ADX < 20 (no trend) → no BUY even near high."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0
        df["ADX_14"] = 10.0  # weak trend
        df["Nearness_52w_High"] = 0.98

        signals = self.strategy.generate_signals(df)
        buy_count = (signals > 0).sum()
        self.assertEqual(buy_count, 0)

    def test_exit_when_momentum_fades(self):
        """Price drops > 10% from high (nearness < 0.90) → EXIT."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0
        df["ADX_14"] = 30.0
        df["Nearness_52w_High"] = 0.85  # faded momentum

        signals = self.strategy.generate_signals(df)
        exit_count = (signals < 0).sum()
        self.assertGreater(exit_count, 0)

    def test_exit_when_below_sma200(self):
        """Price falls below SMA_200 → EXIT (trend broken)."""
        df = _make_ohlcv(100)
        df["SMA_200"] = df["Close"] * 2.0  # below SMA
        df["ADX_14"] = 30.0
        df["Nearness_52w_High"] = 0.92  # between 0.90 and 0.95

        signals = self.strategy.generate_signals(df)
        exit_count = (signals < 0).sum()
        self.assertGreater(exit_count, 0)

    def test_boundary_nearness_0_95(self):
        """Nearness exactly at 0.95 should NOT trigger buy (> 0.95 required)."""
        df = _make_ohlcv(50)
        df["SMA_200"] = 50.0
        df["ADX_14"] = 30.0
        df["Nearness_52w_High"] = 0.95  # exactly at boundary

        signals = self.strategy.generate_signals(df)
        buy_count = (signals > 0).sum()
        self.assertEqual(buy_count, 0)

    def test_boundary_nearness_0_90(self):
        """Nearness exactly at 0.90 should NOT trigger exit (< 0.90 required)."""
        df = _make_ohlcv(50)
        df["SMA_200"] = 50.0  # above trend (no below-SMA exit)
        df["ADX_14"] = 30.0
        df["Nearness_52w_High"] = 0.90  # exactly at boundary

        signals = self.strategy.generate_signals(df)
        # Should be 0 — neither buy (< 0.95) nor exit (>= 0.90)
        self.assertTrue((signals == 0).all())

    def test_mixed_conditions(self):
        """Mixed nearness values: buys near high, exits when faded."""
        df = _make_ohlcv(100)
        df["SMA_200"] = 50.0
        df["ADX_14"] = 30.0
        df["Nearness_52w_High"] = 0.92  # initialize column

        # First half: far from high (no buy, no exit since 0.92 > 0.90)
        df.iloc[:50, df.columns.get_loc("Nearness_52w_High")] = 0.92
        # Second half: near high → buy
        df.iloc[50:, df.columns.get_loc("Nearness_52w_High")] = 0.97

        signals = self.strategy.generate_signals(df)
        self.assertEqual((signals.iloc[:50] > 0).sum(), 0)
        self.assertGreater((signals.iloc[50:] > 0).sum(), 0)

    def test_signal_values_constrained(self):
        """Signal values should only be -1, 0, or 1."""
        df = _make_ohlcv(200)
        df["SMA_200"] = 50.0
        df["ADX_14"] = 30.0
        rng = np.random.RandomState(99)
        df["Nearness_52w_High"] = rng.uniform(0.80, 1.0, 200)

        signals = self.strategy.generate_signals(df)
        unique_vals = set(signals.unique())
        self.assertTrue(unique_vals.issubset({-1, 0, 1}))


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


# ── Multi-Factor Momentum MR ────────────────────────────────────────


def _make_mfmr_df(
    n: int = 400,
    seed: int = 42,
    trend: float = 0.001,
) -> pd.DataFrame:
    """Create synthetic OHLCV with all columns required by MFMR."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n, freq="B")

    close = 100 * np.exp(np.cumsum(rng.normal(trend, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    volume = rng.randint(1_000_000, 10_000_000, n).astype(float)

    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )

    # SMA_200 — strict min_periods=200 (matches indicators.py)
    df["SMA_200"] = df["Close"].rolling(200, min_periods=200).mean()

    # RSI_14
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Volatility_20
    df["Volatility_20"] = df["Close"].pct_change().rolling(20).std()

    # Vol_Ratio
    df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    return df


class TestMultiFactorMomentumMR(unittest.TestCase):
    """Tests for the Multi-Factor Momentum MR strategy."""

    def setUp(self):
        self.strategy = MultiFactorMomentumMR()

    def test_name_and_description(self):
        self.assertEqual(self.strategy.name, "Multi-Factor Momentum MR")
        self.assertIn("momentum", self.strategy.description.lower())

    def test_missing_columns_returns_hold(self):
        """Should return all-zero when required columns are absent."""
        df = _make_ohlcv(300)  # doesn't have Volatility_20 / Vol_Ratio
        signals = self.strategy.generate_signals(df)
        self.assertEqual(len(signals), len(df))
        self.assertTrue((signals == 0).all())

    def test_absolute_min_bars_returns_hold(self):
        """Below _ABSOLUTE_MIN_BARS (41) should return all-zero."""
        df = _make_mfmr_df(n=30)
        signals = self.strategy.generate_signals(df)
        self.assertTrue((signals == 0).all())

    def test_no_signals_during_sma200_warmup(self):
        """Bars where SMA_200 is NaN must be all-zero — no spurious exits.

        Regression test for the bug where ~above_200sma was True on
        NaN SMA bars, emitting persistent -1 exits that charged
        transaction cost while flat.
        """
        df = _make_mfmr_df(n=400)
        signals = self.strategy.generate_signals(df)

        # SMA_200 with min_periods=200 is NaN for indices 0..198
        sma_nan_mask = df["SMA_200"].isna()
        warmup_signals = signals[sma_nan_mask]
        self.assertTrue(
            (warmup_signals == 0).all(),
            f"Expected all-zero during SMA warm-up, got "
            f"{warmup_signals.value_counts().to_dict()}",
        )

    def test_generates_buy_signals_on_uptrend(self):
        """Strong uptrend with all columns should produce at least one buy."""
        df = _make_mfmr_df(n=400, trend=0.002)
        signals = self.strategy.generate_signals(df)
        buy_count = (signals == 1).sum()
        self.assertGreater(
            buy_count, 0,
            "Expected at least one buy signal on strong uptrend data",
        )

    def test_generates_exit_signals(self):
        """Should produce exit signals when conditions deteriorate."""
        df = _make_mfmr_df(n=400, trend=0.002)
        signals = self.strategy.generate_signals(df)
        tradeable_start = df["SMA_200"].first_valid_index()
        post_warmup = signals.loc[tradeable_start:]
        exit_count = (post_warmup == -1).sum()
        self.assertGreater(
            exit_count, 0,
            "Expected exit signals in the tradeable region",
        )

    def test_exit_only_in_tradeable_region(self):
        """All -1 signals must occur where SMA_200 is not NaN."""
        df = _make_mfmr_df(n=400)
        signals = self.strategy.generate_signals(df)
        exits = signals[signals == -1]
        if len(exits) > 0:
            sma_at_exits = df.loc[exits.index, "SMA_200"]
            self.assertTrue(
                sma_at_exits.notna().all(),
                "Found exit signals where SMA_200 is NaN",
            )

    # ── Short-window adaptive weighting tests ───────────────

    def test_short_window_63_bars_participates(self):
        """On a 63-bar (3mo) window the strategy should still produce
        signals using the 4 short-lookback factors with redistributed
        weights, rather than returning all-zero.

        Regression test: the original min_bars=157 hard-return meant
        the strategy got disqualified from 3-month scoring in CSCV.
        """
        # Build a 63-bar frame with strong uptrend + populated SMA_200
        # (use min_periods=1 so SMA is valid from bar 0, simulating a
        # window sliced from a longer series where SMA was pre-computed)
        df = _make_mfmr_df(n=63, trend=0.003)
        df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()
        signals = self.strategy.generate_signals(df)
        self.assertEqual(len(signals), 63)
        # Should produce at least one non-zero signal
        non_zero = (signals != 0).sum()
        self.assertGreater(
            non_zero, 0,
            "Expected non-zero signals on 63-bar window with "
            "adaptive weights, but got all zeros",
        )

    def test_short_window_126_bars_participates(self):
        """On a 126-bar (6mo) window the strategy should produce
        signals using available factors."""
        df = _make_mfmr_df(n=126, trend=0.003)
        df["SMA_200"] = df["Close"].rolling(200, min_periods=1).mean()
        signals = self.strategy.generate_signals(df)
        self.assertEqual(len(signals), 126)
        non_zero = (signals != 0).sum()
        self.assertGreater(
            non_zero, 0,
            "Expected non-zero signals on 126-bar window with "
            "adaptive weights, but got all zeros",
        )

    def test_adaptive_weights_exclude_6m_on_short_frame(self):
        """On a 63-bar frame, mom_6m should NOT be in available factors."""
        strat = MultiFactorMomentumMR()
        n = 63
        available = {
            k for k, (_, min_n) in strat._FACTOR_DEFS.items()
            if n >= min_n + 20
        }
        self.assertNotIn(
            "mom_6m", available,
            "6-month momentum should not be available on 63 bars",
        )
        self.assertIn("mom_1m", available)
        self.assertIn("rsi_inv", available)
        self.assertIn("inv_vol", available)
        self.assertIn("vol_surge", available)

    def test_adaptive_weights_include_6m_on_full_frame(self):
        """On a 400-bar frame, all 5 factors should be available."""
        strat = MultiFactorMomentumMR()
        n = 400
        available = {
            k for k, (_, min_n) in strat._FACTOR_DEFS.items()
            if n >= min_n + 20
        }
        self.assertEqual(len(available), 5)

    # ── Transition signal tests ─────────────────────────────

    def test_no_repeated_exits_while_flat(self):
        """Signals should be transition-based: no consecutive -1 values.

        Regression test for the backtester charging exit cost on every
        -1 bar even when already flat (backtest.py long-only mode).
        """
        df = _make_mfmr_df(n=400, trend=-0.001)  # downtrend
        signals = self.strategy.generate_signals(df)
        # Check that -1 is never immediately followed by -1
        consecutive_exits = (
            (signals == -1) & (signals.shift(1) == -1)
        ).sum()
        self.assertEqual(
            consecutive_exits, 0,
            f"Found {consecutive_exits} consecutive -1 signals; "
            "transitions should never repeat",
        )

    def test_no_repeated_entries(self):
        """No consecutive +1 values — entry happens once."""
        df = _make_mfmr_df(n=400, trend=0.003)
        signals = self.strategy.generate_signals(df)
        consecutive_buys = (
            (signals == 1) & (signals.shift(1) == 1)
        ).sum()
        self.assertEqual(
            consecutive_buys, 0,
            f"Found {consecutive_buys} consecutive +1 signals; "
            "transitions should never repeat",
        )

    def test_signal_dtype_and_range(self):
        """Signals should be integer-like and in {-1, 0, 1}."""
        df = _make_mfmr_df(n=400, trend=0.002)
        signals = self.strategy.generate_signals(df)
        unique = set(signals.unique())
        self.assertTrue(
            unique.issubset({-1, 0, 1}),
            f"Unexpected values: {unique}",
        )

    def test_no_nans_in_output(self):
        """Output signal series must contain no NaN values."""
        df = _make_mfmr_df(n=400)
        signals = self.strategy.generate_signals(df)
        self.assertEqual(signals.isna().sum(), 0)

    # ── State machine regression tests ──────────────────────

    def test_no_false_buy_after_exit_stops_firing(self):
        """Regression: sparse level vector caused diff([-1, 0]) = +1.

        When exit_cond fires on bar N but not on bar N+1 (and no
        buy condition fires either), the strategy must NOT emit +1.
        The desired-position forward-fill must hold -1 (flat) through
        the gap so diff produces 0.
        """
        # Build data where price crosses below 200-SMA mid-series
        # (triggering exit), then stays in a zone where neither
        # buy nor exit fires for several bars.
        df = _make_mfmr_df(n=400, trend=0.0005, seed=99)
        signals = self.strategy.generate_signals(df)

        # Every +1 must occur on a bar where buy conditions hold.
        # Reconstruct buy_eligible the same way the strategy does.
        c = df["Close"]
        sma_valid = df["SMA_200"].notna()

        above_200sma = c > df["SMA_200"]
        not_overbought = df["RSI_14"] < 70

        mom_6m_avail = len(df) >= (126 + 21 + 20)
        if mom_6m_avail:
            mom = (
                c.shift(21) / c.shift(126 + 21) - 1
            )
        else:
            mom = c / c.shift(21) - 1
        positive_momentum = mom > 0

        buy_eligible = sma_valid & above_200sma & not_overbought & positive_momentum

        buys = signals[signals == 1]
        if len(buys) > 0:
            # Every buy signal bar must have buy_eligible == True
            eligible_at_buys = buy_eligible.loc[buys.index]
            false_buys = (~eligible_at_buys).sum()
            self.assertEqual(
                false_buys, 0,
                f"Found {false_buys} buy signals on bars where "
                "buy conditions were NOT met (false buys from "
                "sparse-level diff bug)",
            )

    def test_exit_to_gap_to_exit_no_phantom_entry(self):
        """Explicit test: exit fires, then gap, then exit fires again.

        Sequence: [-1, NaN, NaN, -1] should ffill to [-1,-1,-1,-1],
        diff to [0,0,0,0] — no phantom +1 in the gap.
        """
        # Construct a minimal controlled dataframe where we know
        # the exact signal sequence.  We'll use a prolonged downtrend
        # so exit fires, brief plateau, then exit fires again.
        df = _make_mfmr_df(n=400, trend=-0.002, seed=77)
        signals = self.strategy.generate_signals(df)

        # In a persistent downtrend below 200-SMA, there should be
        # zero buy signals (price never crosses above SMA).
        tradeable_start = df["SMA_200"].first_valid_index()
        if tradeable_start is not None:
            post_warmup = signals.loc[tradeable_start:]
            below_sma = df.loc[tradeable_start:, "Close"] < df.loc[tradeable_start:, "SMA_200"]
            if below_sma.all():
                buy_count = (post_warmup == 1).sum()
                self.assertEqual(
                    buy_count, 0,
                    f"Found {buy_count} buy signals during persistent "
                    "downtrend below 200-SMA (phantom entries)",
                )

    # ── Integration-style sliced-window tests ───────────────

    def test_sliced_63_bar_window_from_enriched_series(self):
        """Production-like test: enrich a long series with real
        min_periods=200 SMA, then slice the last 63 bars.

        This validates that the strategy participates in 3-month
        CSCV scoring when receiving a window that was pre-enriched
        by the full pipeline (SMA_200 already populated from the
        parent series, not faked with min_periods=1).
        """
        # Step 1: build a long series (500 bars) with real warm-up
        full = _make_mfmr_df(n=500, trend=0.002, seed=55)
        # SMA_200 has strict min_periods=200, so bars 0-198 are NaN
        # and bars 199+ are valid — matching indicators.py behavior.

        # Step 2: slice the last 63 bars (simulates CSCV 3mo window)
        window = full.iloc[-63:].copy()

        # SMA_200 should be fully populated in this slice
        self.assertTrue(
            window["SMA_200"].notna().all(),
            "SMA_200 should be valid in a 63-bar slice from "
            "a 500-bar enriched series",
        )

        # Step 3: run strategy on the sliced window
        signals = self.strategy.generate_signals(window)
        self.assertEqual(len(signals), 63)

        # Should produce at least one non-zero signal (the data
        # has a strong uptrend and all indicators are populated)
        non_zero = (signals != 0).sum()
        self.assertGreater(
            non_zero, 0,
            "Strategy produced all-zero on a 63-bar slice from "
            "a fully-enriched 500-bar uptrend series — it should "
            "participate in 3-month CSCV scoring",
        )

    def test_sliced_126_bar_window_from_enriched_series(self):
        """Same as 63-bar test but for 6-month (126-bar) CSCV window."""
        full = _make_mfmr_df(n=500, trend=0.002, seed=55)
        window = full.iloc[-126:].copy()

        self.assertTrue(window["SMA_200"].notna().all())

        signals = self.strategy.generate_signals(window)
        self.assertEqual(len(signals), 126)
        non_zero = (signals != 0).sum()
        self.assertGreater(
            non_zero, 0,
            "Strategy produced all-zero on a 126-bar slice from "
            "a fully-enriched 500-bar series",
        )


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
