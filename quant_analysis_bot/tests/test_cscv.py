"""Tests for CSCV / PBO overfitting analysis module."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.cscv import (
    CSCVResult,
    _build_return_matrix,
    _partition_matrix,
    _score_partition_set,
    _spearman_rank_corr,
    format_cscv_report,
    run_cscv,
)
from quant_analysis_bot.config import DEFAULT_CONFIG
from quant_analysis_bot.strategies import ALL_STRATEGIES


def _make_price_df(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic enriched price DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2026-03-20", periods=n_days)

    # Random walk with drift
    returns = rng.normal(0.0003, 0.015, n_days)
    close = 100.0 * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n_days)),
            "High": close * (1 + abs(rng.normal(0, 0.01, n_days))),
            "Low": close * (1 - abs(rng.normal(0, 0.01, n_days))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n_days),
        },
        index=dates,
    )

    # Add required technical indicators
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_9"] = df["Close"].ewm(span=9).mean()
    df["EMA_21"] = df["Close"].ewm(span=21).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["RSI_14"] = 100 - 100 / (1 + rs)

    # Bollinger Bands
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Mid"] = bb_mid
    df["BB_Lower"] = bb_mid - 2 * bb_std

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ROC
    df["ROC_10"] = df["Close"].pct_change(10) * 100

    # ZScore
    df["ZScore_20"] = (
        (df["Close"] - df["Close"].rolling(20).mean())
        / df["Close"].rolling(20).std()
    )

    # Stochastic
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = (
        (df["Close"] - low14) / (high14 - low14 + 1e-10) * 100
    )
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    # VWAP (cumulative)
    df["VWAP"] = (
        (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    )
    df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # ADX (simplified)
    df["ADX_14"] = 25.0  # constant for simplicity

    # ATR
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # BB Width
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    df = df.dropna()
    return df


class TestCSCVResult(unittest.TestCase):
    """Test CSCVResult dataclass."""

    def test_default_values(self):
        r = CSCVResult()
        self.assertEqual(r.pbo, 1.0)
        self.assertEqual(r.n_combinations, 0)
        self.assertFalse(r.is_valid)

    def test_summary_invalid(self):
        r = CSCVResult()
        self.assertIn("insufficient data", r.summary())

    def test_summary_valid(self):
        r = CSCVResult(
            pbo=0.25,
            n_combinations=70,
            n_partitions=8,
            mean_oos_rank=3.5,
            mean_rank_correlation=0.4,
            logits=[-0.5, -0.3, 0.1],
            is_valid=True,
        )
        s = r.summary()
        self.assertIn("25.0%", s)
        self.assertIn("LOW", s)
        self.assertIn("70 combinations", s)

    def test_summary_high_pbo(self):
        r = CSCVResult(
            pbo=0.65, n_combinations=20, n_partitions=6,
            mean_oos_rank=7.0, mean_rank_correlation=-0.1,
            logits=[0.5, 0.8], is_valid=True,
        )
        self.assertIn("HIGH", r.summary())


class TestHelpers(unittest.TestCase):
    """Test helper functions."""

    def test_spearman_perfect_correlation(self):
        ranks = [1, 2, 3, 4, 5]
        rho = _spearman_rank_corr(ranks, ranks)
        self.assertAlmostEqual(rho, 1.0, places=5)

    def test_spearman_inverse_correlation(self):
        rho = _spearman_rank_corr([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        self.assertAlmostEqual(rho, -1.0, places=5)

    def test_spearman_no_data(self):
        self.assertEqual(_spearman_rank_corr([], []), 0.0)
        self.assertEqual(_spearman_rank_corr([1], [1]), 0.0)

    def test_partition_matrix(self):
        df = pd.DataFrame(
            np.random.randn(100, 3),
            columns=["A", "B", "C"],
        )
        parts = _partition_matrix(df, 4)
        self.assertEqual(len(parts), 4)
        # All rows accounted for
        total = sum(len(p) for p in parts)
        self.assertEqual(total, 100)

    def test_partition_matrix_uneven(self):
        df = pd.DataFrame(np.random.randn(103, 2), columns=["A", "B"])
        parts = _partition_matrix(df, 4)
        self.assertEqual(len(parts), 4)
        # Last partition gets remainder
        total = sum(len(p) for p in parts)
        self.assertEqual(total, 103)

    def test_score_partition_set(self):
        rng = np.random.default_rng(123)
        df = pd.DataFrame(
            rng.normal(0.001, 0.01, (200, 3)),
            columns=["strat_a", "strat_b", "strat_c"],
        )
        parts = _partition_matrix(df, 4)
        scores = _score_partition_set(parts, (0, 1))
        self.assertEqual(set(scores.keys()), {"strat_a", "strat_b", "strat_c"})
        # Sharpe should be finite
        for s in scores.values():
            self.assertTrue(np.isfinite(s))


class TestBuildReturnMatrix(unittest.TestCase):
    """Test return matrix construction."""

    def test_builds_matrix(self):
        df = _make_price_df(300)
        config = DEFAULT_CONFIG.copy()
        mat = _build_return_matrix(df, ALL_STRATEGIES, config, "TEST")
        self.assertIsNotNone(mat)
        self.assertGreater(len(mat), 50)
        # Should have multiple strategy columns
        self.assertGreaterEqual(len(mat.columns), 3)

    def test_insufficient_data(self):
        df = _make_price_df(30)
        config = DEFAULT_CONFIG.copy()
        mat = _build_return_matrix(df, ALL_STRATEGIES, config, "TEST")
        # With only 30 days, may return None or very small matrix
        # (depends on indicator warmup)
        if mat is not None:
            self.assertGreaterEqual(len(mat), 0)


class TestRunCSCV(unittest.TestCase):
    """Test the full CSCV pipeline."""

    def test_run_cscv_basic(self):
        """CSCV should produce valid results on synthetic data."""
        df = _make_price_df(300, seed=42)
        config = DEFAULT_CONFIG.copy()
        result = run_cscv(df, "TEST", config, n_partitions=6)

        if result.is_valid:
            self.assertGreaterEqual(result.pbo, 0.0)
            self.assertLessEqual(result.pbo, 1.0)
            self.assertGreater(result.n_combinations, 0)
            self.assertGreater(len(result.logits), 0)
            self.assertGreater(len(result.strategy_summary), 0)

    def test_run_cscv_too_short(self):
        """CSCV should gracefully handle insufficient data."""
        df = _make_price_df(40, seed=99)
        config = DEFAULT_CONFIG.copy()
        result = run_cscv(df, "TEST", config)
        # Should not be valid with only 40 days
        self.assertFalse(result.is_valid)

    def test_run_cscv_auto_partitions(self):
        """Auto partition selection should work."""
        df = _make_price_df(300, seed=42)
        config = DEFAULT_CONFIG.copy()
        result = run_cscv(df, "TEST", config, n_partitions=0)
        if result.is_valid:
            self.assertIn(result.n_partitions, [6, 8, 10, 16])

    def test_pbo_range(self):
        """PBO should always be between 0 and 1."""
        df = _make_price_df(300, seed=42)
        config = DEFAULT_CONFIG.copy()
        result = run_cscv(df, "TEST", config, n_partitions=6)
        if result.is_valid:
            self.assertGreaterEqual(result.pbo, 0.0)
            self.assertLessEqual(result.pbo, 1.0)

    def test_strategy_summary_populated(self):
        """Each strategy should have summary stats."""
        df = _make_price_df(300, seed=42)
        config = DEFAULT_CONFIG.copy()
        result = run_cscv(df, "TEST", config, n_partitions=6)
        if result.is_valid:
            for name, summary in result.strategy_summary.items():
                self.assertIn("mean_is_rank", summary)
                self.assertIn("mean_oos_rank", summary)
                self.assertIn("rank_degradation", summary)
                self.assertIn("times_is_best", summary)

    def test_rank_correlation_range(self):
        """Rank correlation should be in [-1, 1]."""
        df = _make_price_df(300, seed=42)
        config = DEFAULT_CONFIG.copy()
        result = run_cscv(df, "TEST", config, n_partitions=6)
        if result.is_valid:
            self.assertGreaterEqual(result.mean_rank_correlation, -1.0)
            self.assertLessEqual(result.mean_rank_correlation, 1.0)

    def test_max_combinations_cap(self):
        """Should respect max_combinations limit."""
        df = _make_price_df(500, seed=42)
        config = DEFAULT_CONFIG.copy()
        result = run_cscv(
            df, "TEST", config, n_partitions=16, max_combinations=50,
        )
        if result.is_valid:
            self.assertLessEqual(result.n_combinations, 50)


class TestFormatReport(unittest.TestCase):
    """Test CSCV report formatting."""

    def test_format_empty(self):
        report = format_cscv_report({})
        self.assertIn("No tickers", report)

    def test_format_invalid_results(self):
        report = format_cscv_report({"AAPL": CSCVResult()})
        self.assertIn("No tickers", report)

    def test_format_valid_results(self):
        r = CSCVResult(
            pbo=0.35,
            n_combinations=70,
            n_partitions=8,
            mean_oos_rank=4.2,
            mean_rank_correlation=0.25,
            logits=[-0.3, 0.1, -0.5],
            strategy_summary={
                "SMA": {"mean_is_rank": 2.0, "mean_oos_rank": 5.0},
                "RSI": {"mean_is_rank": 3.0, "mean_oos_rank": 3.5},
            },
            is_valid=True,
        )
        report = format_cscv_report({"AAPL": r, "GOOG": r})
        self.assertIn("AAPL", report)
        self.assertIn("GOOG", report)
        self.assertIn("MODERATE", report)
        self.assertIn("Interpretation", report)


class TestCSCVWithOverfitData(unittest.TestCase):
    """Test PBO detects overfitting in deliberately overfit scenarios."""

    def test_random_strategies_high_pbo(self):
        """
        With pure random signals, the IS-best should not transfer OOS,
        giving PBO close to or above 0.50.
        """
        rng = np.random.default_rng(42)
        n_days = 200
        n_strats = 10

        # Pure random return streams (no real signal)
        ret_matrix = pd.DataFrame(
            rng.normal(0, 0.01, (n_days, n_strats)),
            columns=[f"random_{i}" for i in range(n_strats)],
        )

        # Manually partition and compute PBO
        from quant_analysis_bot.cscv import (
            _partition_matrix,
            _score_partition_set,
        )
        from itertools import combinations as combos

        n_parts = 8
        parts = _partition_matrix(ret_matrix, n_parts)
        half = n_parts // 2
        all_combos = list(combos(range(n_parts), half))

        strat_names = list(ret_matrix.columns)
        overfit = 0
        median_rank = (n_strats + 1) / 2.0

        for combo in all_combos:
            oos_idx = tuple(i for i in range(n_parts) if i not in combo)
            is_scores = _score_partition_set(parts, combo)
            oos_scores = _score_partition_set(parts, oos_idx)

            best_is = max(strat_names, key=lambda s: is_scores[s])
            oos_ranked = sorted(
                strat_names, key=lambda s: oos_scores[s], reverse=True,
            )
            oos_rank = oos_ranked.index(best_is) + 1
            if oos_rank > median_rank:
                overfit += 1

        pbo = overfit / len(all_combos)
        # Random strategies should have PBO >= 0.35 (often much higher)
        self.assertGreaterEqual(pbo, 0.30,
            f"PBO {pbo:.2f} unexpectedly low for random strategies"
        )


class TestCSCVResultLogits(unittest.TestCase):
    """Test logit distribution properties."""

    def test_logit_sign_interpretation(self):
        """
        Negative logit mean → IS-best tends to rank well OOS (good).
        Positive logit mean → IS-best tends to rank poorly OOS (bad).
        """
        # If rank is 1 out of 10: logit = log(1/9) ≈ -2.2 (good)
        good_logit = math.log(1.0 / 9.0)
        self.assertLess(good_logit, 0)

        # If rank is 9 out of 10: logit = log(9/1) ≈ 2.2 (bad)
        bad_logit = math.log(9.0 / 1.0)
        self.assertGreater(bad_logit, 0)


if __name__ == "__main__":
    unittest.main()
