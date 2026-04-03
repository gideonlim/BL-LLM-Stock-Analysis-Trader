"""Tests for FinBERT news sentiment scoring.

Covers:
  - HeadlineScore dataclass and sentiment property
  - aggregate_ticker_sentiment() with mocked FinBERT
  - adjust_composite_scores() in-place modification
  - Graceful degradation when FinBERT is unavailable
  - Edge cases: empty news, single headline, all neutral
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from trading_bot_bl.news_sentiment import (
    HeadlineScore,
    TickerSentiment,
    aggregate_ticker_sentiment,
    adjust_composite_scores,
    is_available,
    FinBERTScorer,
)


# ── Helpers ───────────────────────────────────────────────────────


def _mock_signal(
    ticker: str, composite_score: float
) -> MagicMock:
    """Create a mock Signal with a mutable composite_score."""
    sig = MagicMock()
    sig.ticker = ticker
    sig.composite_score = composite_score
    return sig


def _make_scores(
    headlines: list[str],
    sentiments: list[tuple[float, float, float]],
) -> list[HeadlineScore]:
    """Build HeadlineScore list from (pos, neg, neu) tuples."""
    return [
        HeadlineScore(
            headline=h, positive=s[0], negative=s[1], neutral=s[2]
        )
        for h, s in zip(headlines, sentiments)
    ]


# ── HeadlineScore ─────────────────────────────────────────────────


class TestHeadlineScore(unittest.TestCase):
    def test_positive_sentiment(self) -> None:
        hs = HeadlineScore("Good news", 0.9, 0.05, 0.05)
        self.assertAlmostEqual(hs.sentiment, 0.85)

    def test_negative_sentiment(self) -> None:
        hs = HeadlineScore("Bad news", 0.1, 0.8, 0.1)
        self.assertAlmostEqual(hs.sentiment, -0.7)

    def test_neutral_sentiment(self) -> None:
        hs = HeadlineScore("Meh", 0.3, 0.3, 0.4)
        self.assertAlmostEqual(hs.sentiment, 0.0)


# ── aggregate_ticker_sentiment ────────────────────────────────────


class TestAggregateSentiment(unittest.TestCase):
    """Tests with mocked FinBERTScorer to avoid model download."""

    def _patch_scorer(self, scores: list[HeadlineScore]):
        """Patch FinBERTScorer.score_headlines to return given scores."""
        return patch.object(
            FinBERTScorer,
            "score_headlines",
            return_value=scores,
        )

    def test_single_ticker_positive(self) -> None:
        headlines = ["Stock surges on earnings beat"]
        scores = _make_scores(headlines, [(0.9, 0.05, 0.05)])

        with self._patch_scorer(scores):
            result = aggregate_ticker_sentiment(
                {"AAPL": headlines}
            )

        self.assertIn("AAPL", result)
        self.assertAlmostEqual(result["AAPL"].score, 0.85)
        self.assertEqual(result["AAPL"].num_headlines, 1)

    def test_multiple_headlines_averaged(self) -> None:
        headlines = ["Great quarter", "Missed guidance"]
        scores = _make_scores(
            headlines,
            [(0.9, 0.05, 0.05), (0.1, 0.8, 0.1)],
        )

        with self._patch_scorer(scores):
            result = aggregate_ticker_sentiment(
                {"MSFT": headlines}
            )

        # (0.85 + (-0.7)) / 2 = 0.075
        self.assertAlmostEqual(result["MSFT"].score, 0.075)

    def test_multiple_tickers(self) -> None:
        news = {
            "AAPL": ["Apple beats"],
            "GOOG": ["Google misses"],
        }
        scores = _make_scores(
            ["Apple beats", "Google misses"],
            [(0.9, 0.05, 0.05), (0.05, 0.9, 0.05)],
        )

        with self._patch_scorer(scores):
            result = aggregate_ticker_sentiment(news)

        self.assertIn("AAPL", result)
        self.assertIn("GOOG", result)
        self.assertGreater(result["AAPL"].score, 0)
        self.assertLess(result["GOOG"].score, 0)

    def test_empty_news_map(self) -> None:
        result = aggregate_ticker_sentiment({})
        self.assertEqual(result, {})

    def test_ticker_with_no_headlines_omitted(self) -> None:
        scores = _make_scores(["Good"], [(0.8, 0.1, 0.1)])
        with self._patch_scorer(scores):
            result = aggregate_ticker_sentiment(
                {"AAPL": ["Good"], "MSFT": []}
            )
        self.assertIn("AAPL", result)
        self.assertNotIn("MSFT", result)

    def test_scorer_unavailable_returns_empty(self) -> None:
        with patch.object(
            FinBERTScorer, "score_headlines", return_value=[]
        ):
            result = aggregate_ticker_sentiment(
                {"AAPL": ["Some headline"]}
            )
        self.assertEqual(result, {})

    def test_decay_weighting(self) -> None:
        """With decay_lambda > 0, first headline (newest) gets more
        weight than later ones."""
        headlines = ["New bullish", "Old bearish"]
        scores = _make_scores(
            headlines,
            [(0.9, 0.05, 0.05), (0.05, 0.9, 0.05)],
        )

        with self._patch_scorer(scores):
            result_equal = aggregate_ticker_sentiment(
                {"X": headlines}, decay_lambda=0.0
            )
            result_decay = aggregate_ticker_sentiment(
                {"X": headlines}, decay_lambda=1.0
            )

        # Equal weighting: (0.85 - 0.85) / 2 = 0.0
        self.assertAlmostEqual(result_equal["X"].score, 0.0, places=1)
        # With decay: first headline (positive, weight=1.0) dominates
        # over second (negative, weight=e^-1 ≈ 0.37)
        self.assertGreater(result_decay["X"].score, 0.2)


# ── adjust_composite_scores ───────────────────────────────────────


class TestAdjustCompositeScores(unittest.TestCase):
    def test_positive_sentiment_boosts_score(self) -> None:
        sig = _mock_signal("AAPL", 30.0)
        sent = {
            "AAPL": TickerSentiment(
                ticker="AAPL",
                score=0.8,
                num_headlines=3,
                headlines=[],
            )
        }
        adj = adjust_composite_scores([sig], sent, weight=5.0)
        self.assertAlmostEqual(sig.composite_score, 34.0)
        self.assertEqual(len(adj), 1)
        self.assertAlmostEqual(adj[0][1], 4.0)

    def test_negative_sentiment_reduces_score(self) -> None:
        sig = _mock_signal("GOOG", 25.0)
        sent = {
            "GOOG": TickerSentiment(
                ticker="GOOG",
                score=-0.6,
                num_headlines=2,
                headlines=[],
            )
        }
        adjust_composite_scores([sig], sent, weight=5.0)
        self.assertAlmostEqual(sig.composite_score, 22.0)

    def test_no_sentiment_no_change(self) -> None:
        sig = _mock_signal("MSFT", 40.0)
        adj = adjust_composite_scores([sig], {}, weight=5.0)
        self.assertAlmostEqual(sig.composite_score, 40.0)
        self.assertEqual(len(adj), 0)

    def test_near_zero_sentiment_skipped(self) -> None:
        sig = _mock_signal("META", 35.0)
        sent = {
            "META": TickerSentiment(
                ticker="META",
                score=0.001,
                num_headlines=1,
                headlines=[],
            )
        }
        adj = adjust_composite_scores([sig], sent, weight=5.0)
        # Delta = 0.005 < 0.01 threshold → skipped
        self.assertAlmostEqual(sig.composite_score, 35.0)
        self.assertEqual(len(adj), 0)

    def test_weight_scales_adjustment(self) -> None:
        sig = _mock_signal("AMZN", 50.0)
        sent = {
            "AMZN": TickerSentiment(
                ticker="AMZN",
                score=1.0,
                num_headlines=5,
                headlines=[],
            )
        }
        adjust_composite_scores([sig], sent, weight=10.0)
        self.assertAlmostEqual(sig.composite_score, 60.0)

    def test_multiple_signals(self) -> None:
        sigs = [
            _mock_signal("AAPL", 30.0),
            _mock_signal("GOOG", 25.0),
            _mock_signal("MSFT", 40.0),
        ]
        sent = {
            "AAPL": TickerSentiment("AAPL", 0.5, 2, []),
            "GOOG": TickerSentiment("GOOG", -0.5, 2, []),
        }
        adj = adjust_composite_scores(sigs, sent, weight=5.0)
        self.assertAlmostEqual(sigs[0].composite_score, 32.5)
        self.assertAlmostEqual(sigs[1].composite_score, 22.5)
        self.assertAlmostEqual(sigs[2].composite_score, 40.0)
        self.assertEqual(len(adj), 2)


# ── Availability ──────────────────────────────────────────────────


class TestAvailability(unittest.TestCase):
    def test_is_available_returns_bool(self) -> None:
        # Just verify it doesn't crash — actual value depends
        # on whether torch/transformers are installed.
        result = is_available()
        self.assertIsInstance(result, bool)


# ── Retry / Cooldown ─────────────────────────────────────────────


class TestLoaderRetry(unittest.TestCase):
    """Verify _ensure_loaded retry-on-failure and cooldown logic."""

    def setUp(self) -> None:
        # Reset class-level state before each test
        FinBERTScorer._tokenizer = None
        FinBERTScorer._model = None
        FinBERTScorer._loaded = False
        FinBERTScorer._last_fail_ts = 0.0

    def tearDown(self) -> None:
        # Clean up
        FinBERTScorer._tokenizer = None
        FinBERTScorer._model = None
        FinBERTScorer._loaded = False
        FinBERTScorer._last_fail_ts = 0.0

    @patch(
        "trading_bot_bl.news_sentiment._TRANSFORMERS_AVAILABLE", True
    )
    @patch(
        "trading_bot_bl.news_sentiment.AutoTokenizer",
        create=True,
    )
    @patch(
        "trading_bot_bl.news_sentiment."
        "AutoModelForSequenceClassification",
        create=True,
    )
    def test_loaded_true_only_on_success(
        self, mock_model_cls, mock_tok_cls
    ) -> None:
        """_loaded should be True only after successful load."""
        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        result = FinBERTScorer._ensure_loaded()
        self.assertTrue(result)
        self.assertTrue(FinBERTScorer._loaded)

    @patch(
        "trading_bot_bl.news_sentiment._TRANSFORMERS_AVAILABLE", True
    )
    @patch(
        "trading_bot_bl.news_sentiment.AutoTokenizer",
        create=True,
    )
    @patch(
        "trading_bot_bl.news_sentiment."
        "AutoModelForSequenceClassification",
        create=True,
    )
    def test_failure_does_not_set_loaded(
        self, mock_model_cls, mock_tok_cls
    ) -> None:
        """After a load failure, _loaded stays False so retry is possible."""
        mock_tok_cls.from_pretrained.side_effect = RuntimeError(
            "boom"
        )

        result = FinBERTScorer._ensure_loaded()
        self.assertFalse(result)
        self.assertFalse(FinBERTScorer._loaded)
        self.assertGreater(FinBERTScorer._last_fail_ts, 0)

    @patch(
        "trading_bot_bl.news_sentiment._TRANSFORMERS_AVAILABLE", True
    )
    def test_cooldown_blocks_retry(self) -> None:
        """Within cooldown window, _ensure_loaded returns False
        without attempting load."""
        import time

        # Simulate a recent failure (within 300s cooldown)
        FinBERTScorer._last_fail_ts = time.monotonic() - 10.0

        result = FinBERTScorer._ensure_loaded()
        self.assertFalse(result)

    @patch(
        "trading_bot_bl.news_sentiment._TRANSFORMERS_AVAILABLE", True
    )
    @patch(
        "trading_bot_bl.news_sentiment.AutoTokenizer",
        create=True,
    )
    @patch(
        "trading_bot_bl.news_sentiment."
        "AutoModelForSequenceClassification",
        create=True,
    )
    def test_retry_after_cooldown_expires(
        self, mock_model_cls, mock_tok_cls
    ) -> None:
        """After cooldown expires, a retry attempt is made."""
        import time as real_time

        mock_tok_cls.from_pretrained.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Set failure timestamp far enough in the past
        FinBERTScorer._last_fail_ts = (
            real_time.monotonic() - 400.0
        )

        result = FinBERTScorer._ensure_loaded()
        self.assertTrue(result)
        self.assertTrue(FinBERTScorer._loaded)
        mock_tok_cls.from_pretrained.assert_called_once()


if __name__ == "__main__":
    unittest.main()
