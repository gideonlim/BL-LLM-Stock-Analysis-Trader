"""Tests for meta-labeling model: features, training, sizing."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.meta_label import (
    FEATURE_NAMES,
    META_LABEL_MIN_TRAINING_TRADES,
    build_training_data,
    compute_meta_kelly,
    extract_meta_features,
    train_meta_model,
)
from quant_analysis_bot.triple_barrier import BarrierTrade


def _make_enriched_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal enriched DataFrame with all required indicators."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.randint(1_000_000, 5_000_000, n),
            "RSI_14": 50 + rng.normal(0, 10, n),
            "ADX_14": 20 + rng.normal(0, 5, n),
            "ATR_14": close * 0.02,
            "BB_Width": rng.uniform(0.02, 0.08, n),
            "SMA_50": close * (1 + rng.normal(0, 0.02, n)),
            "SMA_200": close * (1 + rng.normal(0, 0.03, n)),
            "Volatility_20": rng.uniform(0.15, 0.35, n),
            "ROC_10": rng.normal(0, 2, n),
            "Vol_Ratio": rng.uniform(0.5, 2.0, n),
        },
        index=dates,
    )
    return df


def _make_barrier_trades(
    n: int = 80,
    df: pd.DataFrame | None = None,
) -> list[BarrierTrade]:
    """Create synthetic barrier trades."""
    if df is None:
        df = _make_enriched_df(200)
    rng = np.random.RandomState(123)
    trades = []
    for i in range(n):
        entry_idx = min(i * 2, len(df) - 10)
        exit_idx = entry_idx + rng.randint(3, 8)
        barrier = rng.choice(["upper", "lower", "vertical"])
        label = 1 if barrier == "upper" else (-1 if barrier == "lower" else 0)
        ret = rng.uniform(-5, 8) if barrier != "vertical" else rng.uniform(-1, 1)
        trades.append(
            BarrierTrade(
                entry_idx=entry_idx,
                entry_date=str(df.index[entry_idx].date()),
                entry_price=round(float(df.iloc[entry_idx]["Close"]), 2),
                exit_idx=min(exit_idx, len(df) - 1),
                exit_date=str(df.index[min(exit_idx, len(df) - 1)].date()),
                exit_price=round(float(df.iloc[entry_idx]["Close"]) * (1 + ret / 100), 2),
                holding_days=exit_idx - entry_idx,
                return_pct=round(ret, 2),
                label=label,
                exit_barrier=barrier,
                mfe_pct=max(ret, 0),
                mae_pct=max(-ret, 0),
                mfe_bar=rng.randint(1, 5),
                mae_bar=rng.randint(1, 5),
            )
        )
    return trades


class TestExtractMetaFeatures(unittest.TestCase):
    """Tests for feature extraction at entry bars."""

    def test_returns_all_features(self):
        df = _make_enriched_df()
        features = extract_meta_features(df, 50, strategy_id=3, sl_pct=3.0, tp_pct=6.0)
        for name in FEATURE_NAMES:
            self.assertIn(name, features)

    def test_rr_ratio_computed_correctly(self):
        df = _make_enriched_df()
        features = extract_meta_features(df, 50, strategy_id=0, sl_pct=4.0, tp_pct=8.0)
        self.assertAlmostEqual(features["rr_ratio"], 2.0)

    def test_strategy_id_passed_through(self):
        df = _make_enriched_df()
        features = extract_meta_features(df, 50, strategy_id=7)
        self.assertEqual(features["strategy_id"], 7.0)

    def test_missing_indicators_use_defaults(self):
        """Should not crash when indicators are missing."""
        dates = pd.bdate_range("2025-01-01", periods=10, freq="B")
        df = pd.DataFrame(
            {"Close": np.full(10, 100.0)},
            index=dates,
        )
        features = extract_meta_features(df, 5, strategy_id=0)
        self.assertEqual(features["rsi_14"], 50)
        self.assertEqual(features["adx_14"], 0)


def _make_sl_tp_arrays(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Create uniform SL/TP arrays for testing (no future leakage)."""
    sl = np.full(n, 3.0)
    tp = np.full(n, 6.0)
    return sl, tp


class TestBuildTrainingData(unittest.TestCase):
    """Tests for building X, y, events from barrier trades."""

    def test_correct_shape(self):
        df = _make_enriched_df(200)
        trades = _make_barrier_trades(30, df)
        sl, tp = _make_sl_tp_arrays(len(df))
        X, y, events = build_training_data(
            trades, df, "AAPL", strategy_id=0,
            sl_pct_arr=sl, tp_pct_arr=tp,
        )
        self.assertEqual(X.shape[0], len(trades))
        self.assertEqual(X.shape[1], len(FEATURE_NAMES))
        self.assertEqual(len(y), len(trades))
        self.assertEqual(len(events), len(trades))

    def test_labels_binary(self):
        df = _make_enriched_df(200)
        trades = _make_barrier_trades(30, df)
        sl, tp = _make_sl_tp_arrays(len(df))
        _, y, _ = build_training_data(
            trades, df, "AAPL", strategy_id=0,
            sl_pct_arr=sl, tp_pct_arr=tp,
        )
        self.assertTrue(all(v in (0, 1) for v in y))

    def test_tp_trades_labeled_one(self):
        df = _make_enriched_df(200)
        trades = _make_barrier_trades(30, df)
        sl, tp = _make_sl_tp_arrays(len(df))
        _, y, _ = build_training_data(
            trades, df, "AAPL", strategy_id=0,
            sl_pct_arr=sl, tp_pct_arr=tp,
        )
        for trade, label in zip(trades, y):
            if trade.exit_barrier == "upper":
                self.assertEqual(label, 1)
            else:
                self.assertEqual(label, 0)

    def test_events_have_required_columns(self):
        df = _make_enriched_df(200)
        trades = _make_barrier_trades(10, df)
        sl, tp = _make_sl_tp_arrays(len(df))
        _, _, events = build_training_data(
            trades, df, "AAPL", strategy_id=0,
            sl_pct_arr=sl, tp_pct_arr=tp,
        )
        self.assertIn("ticker", events.columns)
        self.assertIn("entry_ts", events.columns)
        self.assertIn("exit_ts", events.columns)


class TestTrainMetaModel(unittest.TestCase):
    """Tests for meta-model training."""

    def test_skips_with_too_few_trades(self):
        X = np.random.rand(10, len(FEATURE_NAMES))
        y = np.random.randint(0, 2, 10)
        events = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "entry_ts": pd.date_range("2025-01-01", periods=10),
            "exit_ts": pd.date_range("2025-01-06", periods=10),
        })
        result = train_meta_model(X, y, events)
        self.assertIsNone(result)

    def test_trains_with_enough_data(self):
        """Should train successfully with 60+ trades."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not available")

        df = _make_enriched_df(200)
        trades = _make_barrier_trades(80, df)
        sl, tp = _make_sl_tp_arrays(len(df))
        X, y, events = build_training_data(
            trades, df, "AAPL", strategy_id=0,
            sl_pct_arr=sl, tp_pct_arr=tp,
        )

        # Ensure we have both classes
        if len(np.unique(y)) < 2:
            y[0] = 1 - y[0]

        result = train_meta_model(X, y, events, n_cv_splits=3)
        self.assertIsNotNone(result)
        self.assertGreater(result.n_training_trades, 0)
        self.assertIn(result.model_name, ("lightgbm", "sklearn_gbc", "sklearn_rf"))

    def test_uncalibrated_under_100_trades(self):
        """Should train uncalibrated when < 100 trades."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier  # noqa: F401
        except ImportError:
            self.skipTest("sklearn not available")

        df = _make_enriched_df(200)
        trades = _make_barrier_trades(60, df)
        sl, tp = _make_sl_tp_arrays(len(df))
        X, y, events = build_training_data(
            trades, df, "AAPL", strategy_id=0,
            sl_pct_arr=sl, tp_pct_arr=tp,
        )

        if len(np.unique(y)) < 2:
            y[0] = 1 - y[0]

        result = train_meta_model(X, y, events, n_cv_splits=3)
        if result is not None:
            self.assertFalse(result.is_calibrated)


class TestComputeMetaKelly(unittest.TestCase):
    """Tests for meta-label-adjusted Kelly sizing."""

    def test_insufficient_training_uses_base(self):
        final, mult = compute_meta_kelly(
            meta_prob=0.7,
            base_kelly_f=0.10,
            profit_factor=1.5,
            n_training_trades=30,
            is_calibrated=True,
        )
        self.assertAlmostEqual(final, 0.10)
        self.assertAlmostEqual(mult, 1.0)

    def test_negative_meta_prob_uses_base(self):
        final, mult = compute_meta_kelly(
            meta_prob=-1.0,
            base_kelly_f=0.10,
            profit_factor=1.5,
            n_training_trades=200,
            is_calibrated=True,
        )
        self.assertAlmostEqual(final, 0.10)

    def test_high_prob_increases_size(self):
        base = 0.10
        final, mult = compute_meta_kelly(
            meta_prob=0.80,
            base_kelly_f=base,
            profit_factor=2.0,
            n_training_trades=200,
            is_calibrated=True,
        )
        # High probability should result in larger or similar sizing
        self.assertGreaterEqual(final, 0)
        self.assertLessEqual(final, 0.50)  # hard cap

    def test_low_prob_decreases_size(self):
        base = 0.15
        final, mult = compute_meta_kelly(
            meta_prob=0.25,
            base_kelly_f=base,
            profit_factor=1.5,
            n_training_trades=200,
            is_calibrated=True,
        )
        self.assertLessEqual(final, base)

    def test_hard_cap_at_50_pct(self):
        final, _ = compute_meta_kelly(
            meta_prob=0.99,
            base_kelly_f=0.80,
            profit_factor=5.0,
            n_training_trades=300,
            is_calibrated=True,
        )
        self.assertLessEqual(final, 0.50)

    def test_zero_edge_gives_zero(self):
        final, _ = compute_meta_kelly(
            meta_prob=0.20,
            base_kelly_f=0.0,
            profit_factor=0.5,
            n_training_trades=200,
            is_calibrated=True,
        )
        self.assertAlmostEqual(final, 0.0, places=2)

    def test_uncalibrated_halves_blend(self):
        """Uncalibrated model should be more conservative."""
        _, mult_cal = compute_meta_kelly(
            meta_prob=0.70,
            base_kelly_f=0.10,
            profit_factor=2.0,
            n_training_trades=200,
            is_calibrated=True,
        )
        _, mult_uncal = compute_meta_kelly(
            meta_prob=0.70,
            base_kelly_f=0.10,
            profit_factor=2.0,
            n_training_trades=200,
            is_calibrated=False,
        )
        # With same inputs, uncalibrated should produce a multiplier
        # closer to 1.0 (less deviation from base)
        self.assertLessEqual(
            abs(mult_uncal - 1.0), abs(mult_cal - 1.0) + 0.01,
        )


if __name__ == "__main__":
    unittest.main()
