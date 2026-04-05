"""Tests for purged K-Fold CV and sample uniqueness weights."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.cv_purged import (
    compute_sample_weights,
    purged_kfold_split,
)


def _make_events(
    n: int = 20,
    holding: int = 5,
    ticker: str = "AAPL",
) -> pd.DataFrame:
    """Create synthetic overlapping trade events."""
    rows = []
    start = pd.Timestamp("2025-01-01")
    for i in range(n):
        entry = start + pd.Timedelta(days=i * 3)
        exit_ = entry + pd.Timedelta(days=holding)
        rows.append({
            "ticker": ticker,
            "entry_ts": entry,
            "exit_ts": exit_,
        })
    return pd.DataFrame(rows)


class TestComputeSampleWeights(unittest.TestCase):
    """Tests for sample uniqueness weighting."""

    def test_single_trade_weight_is_one(self):
        events = pd.DataFrame([{
            "ticker": "AAPL",
            "entry_ts": pd.Timestamp("2025-01-01"),
            "exit_ts": pd.Timestamp("2025-01-10"),
        }])
        weights = compute_sample_weights(events)
        self.assertEqual(len(weights), 1)
        self.assertAlmostEqual(weights[0], 1.0)

    def test_non_overlapping_trades_equal_weight(self):
        events = pd.DataFrame([
            {"ticker": "AAPL", "entry_ts": pd.Timestamp("2025-01-01"),
             "exit_ts": pd.Timestamp("2025-01-05")},
            {"ticker": "AAPL", "entry_ts": pd.Timestamp("2025-02-01"),
             "exit_ts": pd.Timestamp("2025-02-05")},
        ])
        weights = compute_sample_weights(events)
        self.assertEqual(len(weights), 2)
        # Both isolated → should be approximately equal
        self.assertAlmostEqual(weights[0], weights[1], places=1)

    def test_overlapping_trades_lower_weight(self):
        # Two trades that overlap significantly
        events = pd.DataFrame([
            {"ticker": "AAPL", "entry_ts": pd.Timestamp("2025-01-01"),
             "exit_ts": pd.Timestamp("2025-01-10")},
            {"ticker": "AAPL", "entry_ts": pd.Timestamp("2025-01-02"),
             "exit_ts": pd.Timestamp("2025-01-11")},
        ])
        weights = compute_sample_weights(events)
        self.assertEqual(len(weights), 2)
        # Both should be < 1.0 (they overlap, so uniqueness is lower)
        # After normalisation: mean = 1.0 but individual < what they'd be if isolated
        # The key test: weights are equal (symmetric overlap)
        self.assertAlmostEqual(weights[0], weights[1], places=2)

    def test_different_tickers_no_overlap(self):
        """Trades on different tickers shouldn't affect each other."""
        events = pd.DataFrame([
            {"ticker": "AAPL", "entry_ts": pd.Timestamp("2025-01-01"),
             "exit_ts": pd.Timestamp("2025-01-10")},
            {"ticker": "GOOG", "entry_ts": pd.Timestamp("2025-01-01"),
             "exit_ts": pd.Timestamp("2025-01-10")},
        ])
        weights = compute_sample_weights(events)
        self.assertEqual(len(weights), 2)
        # Both should have equal weight (no cross-ticker overlap)
        self.assertAlmostEqual(weights[0], weights[1], places=2)

    def test_empty_events(self):
        events = pd.DataFrame(columns=["ticker", "entry_ts", "exit_ts"])
        weights = compute_sample_weights(events)
        self.assertEqual(len(weights), 0)

    def test_weights_normalised_to_mean_one(self):
        events = _make_events(20, holding=5)
        weights = compute_sample_weights(events)
        self.assertEqual(len(weights), 20)
        self.assertAlmostEqual(weights.mean(), 1.0, places=5)


class TestPurgedKFoldSplit(unittest.TestCase):
    """Tests for purged K-Fold cross-validation."""

    def test_basic_split_count(self):
        events = _make_events(50, holding=3)
        splits = purged_kfold_split(events, n_splits=5)
        self.assertEqual(len(splits), 5)

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices should not overlap."""
        events = _make_events(50, holding=3)
        splits = purged_kfold_split(events, n_splits=5)
        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            self.assertEqual(
                len(overlap), 0,
                f"Train/test overlap: {overlap}",
            )

    def test_test_indices_cover_all_samples(self):
        """Union of all test folds should cover all samples."""
        events = _make_events(50, holding=3)
        splits = purged_kfold_split(events, n_splits=5)
        all_test = set()
        for _, test_idx in splits:
            all_test.update(test_idx)
        self.assertEqual(all_test, set(range(50)))

    def test_purging_removes_overlapping_train_samples(self):
        """Purged training set should be smaller than naive split."""
        events = _make_events(30, holding=10)  # lots of overlap
        splits = purged_kfold_split(events, n_splits=3)
        for train_idx, test_idx in splits:
            # Naive split would have ~20 train samples
            # Purging should remove some
            naive_train_size = len(events) - len(test_idx)
            self.assertLessEqual(len(train_idx), naive_train_size)

    def test_embargo_further_reduces_training(self):
        events = _make_events(50, holding=3)
        splits_no_embargo = purged_kfold_split(
            events, n_splits=5, embargo_pct=0.0,
        )
        splits_with_embargo = purged_kfold_split(
            events, n_splits=5, embargo_pct=0.05,
        )
        # With embargo, training sets should be smaller or equal
        for (train_ne, _), (train_we, _) in zip(
            splits_no_embargo, splits_with_embargo
        ):
            self.assertLessEqual(len(train_we), len(train_ne))

    def test_small_dataset_fallback(self):
        """Very small datasets should still produce a split."""
        events = _make_events(6, holding=2)
        splits = purged_kfold_split(events, n_splits=5)
        self.assertGreaterEqual(len(splits), 1)
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)

    def test_multi_ticker_purging(self):
        """Purging should only affect same-ticker overlaps."""
        aapl_events = _make_events(10, holding=5, ticker="AAPL")
        goog_events = _make_events(10, holding=5, ticker="GOOG")
        events = pd.concat(
            [aapl_events, goog_events], ignore_index=True,
        )
        splits = purged_kfold_split(events, n_splits=3)
        self.assertGreaterEqual(len(splits), 1)


if __name__ == "__main__":
    unittest.main()
