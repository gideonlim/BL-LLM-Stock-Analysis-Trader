"""Tests for regime detection and state-dependent strategy gating."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_analysis_bot.regime import (
    _build_regime_df,
    clear_cache,
    enrich_with_regime,
    fetch_regime_data,
)
from quant_analysis_bot.strategies import (
    BollingerBand_Reversion,
    RSI_MeanReversion,
    ZScore_MeanReversion,
    _fear_mask,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_regime_df(
    n: int = 300,
    vix_base: float = 20.0,
    spy_above_sma: bool = True,
) -> pd.DataFrame:
    """Create synthetic regime DataFrame."""
    dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)

    vix = vix_base + rng.normal(0, 2, n)
    spy = 450 + np.cumsum(rng.normal(0.1, 1.5, n))
    spy_sma = pd.Series(spy).rolling(200, min_periods=1).mean().values

    if not spy_above_sma:
        spy = spy_sma - 10  # force below SMA

    df = pd.DataFrame(
        {
            "VIX_Close": vix,
            "SPY_Close": spy,
            "SPY_SMA_200": spy_sma,
        },
        index=dates,
    )
    df["SPY_Below_SMA200"] = df["SPY_Close"] < df["SPY_SMA_200"]
    df["VIX_Elevated"] = df["VIX_Close"] > 25.0
    df["Regime_Fear"] = (
        df["VIX_Elevated"] | df["SPY_Below_SMA200"]
    )
    return df


def _make_ticker_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic ticker OHLCV with indicators."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-06-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)

    df = pd.DataFrame(
        {
            "Close": close,
            "High": high,
            "Low": low,
            "Volume": rng.randint(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )

    # Indicators needed by mean-reversion strategies
    df["RSI_14"] = 50.0 + rng.normal(0, 15, n)
    df["BB_Upper"] = close + 5
    df["BB_Lower"] = close - 5
    df["BB_Mid"] = close
    df["ZScore_20"] = rng.normal(0, 1, n)

    return df


# ── Test: _fear_mask helper ──────────────────────────────────────────


class TestFearMask(unittest.TestCase):
    """Test the _fear_mask() helper used by mean-reversion strategies."""

    def test_returns_true_when_column_missing(self):
        """Without Regime_Fear column, all rows should be True."""
        df = pd.DataFrame({"Close": [100, 101, 102]})
        mask = _fear_mask(df)
        self.assertTrue(mask.all())
        self.assertEqual(len(mask), 3)

    def test_returns_column_values_when_present(self):
        """Should use the Regime_Fear column values."""
        df = pd.DataFrame(
            {"Close": [100, 101, 102], "Regime_Fear": [True, False, True]}
        )
        mask = _fear_mask(df)
        self.assertEqual(list(mask), [True, False, True])

    def test_nan_defaults_to_true(self):
        """NaN in Regime_Fear should default to True (unfiltered)."""
        df = pd.DataFrame(
            {
                "Close": [100, 101, 102],
                "Regime_Fear": [True, np.nan, False],
            }
        )
        mask = _fear_mask(df)
        self.assertEqual(list(mask), [True, True, False])


# ── Test: enrich_with_regime ─────────────────────────────────────────


class TestEnrichWithRegime(unittest.TestCase):
    """Test regime data enrichment of ticker DataFrames."""

    def test_adds_regime_columns(self):
        """Should add Regime_Fear and related columns."""
        regime_df = _make_regime_df(300)
        ticker_df = _make_ticker_df(100)
        result = enrich_with_regime(ticker_df, regime_df)

        self.assertIn("Regime_Fear", result.columns)
        self.assertIn("VIX_Close", result.columns)
        self.assertIn("SPY_Below_SMA200", result.columns)
        self.assertIn("VIX_Elevated", result.columns)

    def test_none_regime_df_defaults_to_fear(self):
        """With no regime data, Regime_Fear should be True everywhere."""
        ticker_df = _make_ticker_df(50)
        result = enrich_with_regime(ticker_df, None)
        self.assertTrue(result["Regime_Fear"].all())

    def test_empty_regime_df_defaults_to_fear(self):
        """Empty regime DataFrame → Regime_Fear = True everywhere."""
        ticker_df = _make_ticker_df(50)
        result = enrich_with_regime(ticker_df, pd.DataFrame())
        self.assertTrue(result["Regime_Fear"].all())

    def test_date_alignment(self):
        """Regime data should align by date, not position."""
        dates = pd.bdate_range("2024-09-01", periods=5, freq="B")
        ticker_df = pd.DataFrame(
            {"Close": [100, 101, 102, 103, 104]},
            index=dates,
        )

        regime_dates = pd.bdate_range("2024-08-01", periods=30, freq="B")
        regime_df = pd.DataFrame(
            {
                "VIX_Close": np.full(30, 20.0),
                "SPY_Close": np.full(30, 450.0),
                "SPY_SMA_200": np.full(30, 440.0),
                "SPY_Below_SMA200": np.full(30, False),
                "VIX_Elevated": np.full(30, False),
                "Regime_Fear": np.full(30, False),
            },
            index=regime_dates,
        )

        result = enrich_with_regime(ticker_df, regime_df)
        # All dates in September should have regime data (via ffill)
        self.assertEqual(len(result), 5)
        self.assertFalse(result["Regime_Fear"].any())

    def test_fear_when_vix_elevated(self):
        """High VIX should produce Regime_Fear = True."""
        regime_df = _make_regime_df(300, vix_base=35.0)
        ticker_df = _make_ticker_df(100)
        result = enrich_with_regime(ticker_df, regime_df)
        # Most days should be in fear regime with VIX ~35
        fear_pct = result["Regime_Fear"].mean()
        self.assertGreater(fear_pct, 0.8)

    def test_fear_when_spy_below_sma(self):
        """SPY below SMA_200 should produce Regime_Fear = True."""
        regime_df = _make_regime_df(
            300, vix_base=15.0, spy_above_sma=False
        )
        ticker_df = _make_ticker_df(100)
        result = enrich_with_regime(ticker_df, regime_df)
        # All days should be fear (SPY forced below SMA)
        self.assertTrue(result["Regime_Fear"].all())


# ── Test: fetch_regime_data caching ──────────────────────────────────


class TestFetchRegimeDataCache(unittest.TestCase):
    """Test module-level caching behavior."""

    def setUp(self):
        clear_cache()

    def tearDown(self):
        clear_cache()

    def test_caches_result(self):
        """Second call should return cached result."""
        import quant_analysis_bot.regime as regime_mod

        fake_df = _make_regime_df(50)
        original_build = regime_mod._build_regime_df
        call_count = [0]

        def counting_build(*args, **kwargs):
            call_count[0] += 1
            return fake_df

        regime_mod._build_regime_df = counting_build
        try:
            result1 = fetch_regime_data(lookback_days=100)
            result2 = fetch_regime_data(lookback_days=100)
            self.assertEqual(call_count[0], 1)
            pd.testing.assert_frame_equal(result1, result2)
        finally:
            regime_mod._build_regime_df = original_build

    def test_no_cache_flag(self):
        """use_cache=False should bypass cache."""
        import quant_analysis_bot.regime as regime_mod

        fake_df = _make_regime_df(50)
        original_build = regime_mod._build_regime_df
        call_count = [0]

        def counting_build(*args, **kwargs):
            call_count[0] += 1
            return fake_df

        regime_mod._build_regime_df = counting_build
        try:
            fetch_regime_data(lookback_days=100, use_cache=False)
            fetch_regime_data(lookback_days=100, use_cache=False)
            self.assertEqual(call_count[0], 2)
        finally:
            regime_mod._build_regime_df = original_build

    def test_clear_cache(self):
        """clear_cache() should reset module state."""
        from quant_analysis_bot import regime

        regime._regime_cache = "sentinel"
        clear_cache()
        self.assertIsNone(regime._regime_cache)


# ── Test: Strategy gating with regime ────────────────────────────────


class TestRSIMeanReversionRegimeGate(unittest.TestCase):
    """RSI Mean Reversion should only buy during fear regime."""

    def setUp(self):
        self.strategy = RSI_MeanReversion()

    def test_buys_during_fear(self):
        """Should generate buy signals when Regime_Fear is True."""
        df = _make_ticker_df(50)
        df["RSI_14"] = 20.0  # oversold
        df["Regime_Fear"] = True

        signals = self.strategy.generate_signals(df)
        self.assertGreater((signals > 0).sum(), 0)

    def test_no_buys_during_greed(self):
        """Should NOT generate buy signals when Regime_Fear is False."""
        df = _make_ticker_df(50)
        df["RSI_14"] = 20.0  # oversold
        df["Regime_Fear"] = False

        signals = self.strategy.generate_signals(df)
        self.assertEqual((signals > 0).sum(), 0)

    def test_sells_regardless_of_regime(self):
        """Sell signals should fire in any regime (never trap in position)."""
        df = _make_ticker_df(50)
        df["RSI_14"] = 80.0  # overbought
        df["Regime_Fear"] = False  # greed regime

        signals = self.strategy.generate_signals(df)
        self.assertGreater((signals < 0).sum(), 0)

    def test_backward_compatible_no_regime_column(self):
        """Without Regime_Fear column, should buy as before."""
        df = _make_ticker_df(50)
        df["RSI_14"] = 20.0  # oversold
        # No Regime_Fear column

        signals = self.strategy.generate_signals(df)
        self.assertGreater((signals > 0).sum(), 0)


class TestBollingerReversionRegimeGate(unittest.TestCase):
    """Bollinger Band Mean Reversion should only buy during fear regime."""

    def setUp(self):
        self.strategy = BollingerBand_Reversion()

    def test_buys_during_fear(self):
        """Should generate buy signals when Regime_Fear is True."""
        df = _make_ticker_df(50)
        df["Close"] = df["BB_Lower"] - 1  # below lower band
        df["Regime_Fear"] = True

        signals = self.strategy.generate_signals(df)
        self.assertGreater((signals > 0).sum(), 0)

    def test_no_buys_during_greed(self):
        """Should NOT generate buy signals when Regime_Fear is False."""
        df = _make_ticker_df(50)
        df["Close"] = df["BB_Lower"] - 1  # below lower band
        df["Regime_Fear"] = False

        signals = self.strategy.generate_signals(df)
        self.assertEqual((signals > 0).sum(), 0)

    def test_sells_regardless_of_regime(self):
        """Sell signals should fire in any regime."""
        df = _make_ticker_df(50)
        df["Close"] = df["BB_Upper"] + 1  # above upper band
        df["Regime_Fear"] = False

        signals = self.strategy.generate_signals(df)
        self.assertGreater((signals < 0).sum(), 0)


class TestZScoreReversionRegimeGate(unittest.TestCase):
    """Z-Score Mean Reversion should only buy during fear regime."""

    def setUp(self):
        self.strategy = ZScore_MeanReversion()

    def test_buys_during_fear(self):
        """Should generate buy signals when Regime_Fear is True."""
        df = _make_ticker_df(50)
        df["ZScore_20"] = -2.0  # well below mean
        df["Regime_Fear"] = True

        signals = self.strategy.generate_signals(df)
        self.assertGreater((signals > 0).sum(), 0)

    def test_no_buys_during_greed(self):
        """Should NOT generate buy signals when Regime_Fear is False."""
        df = _make_ticker_df(50)
        df["ZScore_20"] = -2.0  # well below mean
        df["Regime_Fear"] = False

        signals = self.strategy.generate_signals(df)
        self.assertEqual((signals > 0).sum(), 0)

    def test_sells_regardless_of_regime(self):
        """Sell signals should fire in any regime."""
        df = _make_ticker_df(50)
        df["ZScore_20"] = 2.0  # well above mean
        df["Regime_Fear"] = False

        signals = self.strategy.generate_signals(df)
        self.assertGreater((signals < 0).sum(), 0)


# ── Test: Mixed regime periods ───────────────────────────────────────


class TestMixedRegimePeriods(unittest.TestCase):
    """Test strategies with alternating fear/greed periods."""

    def test_rsi_only_buys_in_fear_windows(self):
        """RSI buy signals should only appear in fear windows."""
        df = _make_ticker_df(100)
        df["RSI_14"] = 20.0  # oversold everywhere

        # First 50 bars: greed, last 50: fear
        df["Regime_Fear"] = False
        df.iloc[50:, df.columns.get_loc("Regime_Fear")] = True

        strategy = RSI_MeanReversion()
        signals = strategy.generate_signals(df)

        # No buys in first 50 (greed)
        self.assertEqual((signals.iloc[:50] > 0).sum(), 0)
        # Buys in last 50 (fear)
        self.assertGreater((signals.iloc[50:] > 0).sum(), 0)

    def test_bollinger_only_buys_in_fear_windows(self):
        """Bollinger buy signals should only appear in fear windows."""
        df = _make_ticker_df(100)
        df["Close"] = df["BB_Lower"] - 1  # below lower band

        df["Regime_Fear"] = False
        df.iloc[50:, df.columns.get_loc("Regime_Fear")] = True

        strategy = BollingerBand_Reversion()
        signals = strategy.generate_signals(df)

        self.assertEqual((signals.iloc[:50] > 0).sum(), 0)
        self.assertGreater((signals.iloc[50:] > 0).sum(), 0)

    def test_zscore_only_buys_in_fear_windows(self):
        """Z-Score buy signals should only appear in fear windows."""
        df = _make_ticker_df(100)
        df["ZScore_20"] = -2.0

        df["Regime_Fear"] = False
        df.iloc[50:, df.columns.get_loc("Regime_Fear")] = True

        strategy = ZScore_MeanReversion()
        signals = strategy.generate_signals(df)

        self.assertEqual((signals.iloc[:50] > 0).sum(), 0)
        self.assertGreater((signals.iloc[50:] > 0).sum(), 0)


if __name__ == "__main__":
    unittest.main()
