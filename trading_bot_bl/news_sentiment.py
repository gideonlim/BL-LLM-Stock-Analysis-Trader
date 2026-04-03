"""FinBERT-based news headline sentiment scoring.

Provides deterministic, per-headline sentiment scores using
ProsusAI/finBERT — a BERT model fine-tuned on financial text.
Scores are aggregated per ticker with exponential recency decay
and used to adjust signal composite scores.

Dependencies (optional — install only when enabling this feature):
    pip install transformers torch

The module degrades gracefully when dependencies are missing:
``is_available()`` returns False and all scoring functions return
neutral defaults.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ── Availability check ────────────────────────────────────────────

_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (  # type: ignore[import-untyped]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    import torch  # type: ignore[import-untyped]

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

_FINBERT_MODEL_NAME = "ProsusAI/finbert"


def is_available() -> bool:
    """Return True if FinBERT dependencies are installed."""
    return _TRANSFORMERS_AVAILABLE


# ── Scorer ────────────────────────────────────────────────────────


@dataclass
class HeadlineScore:
    """Sentiment score for a single headline."""

    headline: str
    positive: float  # P(positive)
    negative: float  # P(negative)
    neutral: float  # P(neutral)

    @property
    def sentiment(self) -> float:
        """Net sentiment: P(positive) - P(negative), range [-1, +1]."""
        return self.positive - self.negative


class FinBERTScorer:
    """Lazy-loading FinBERT sentiment scorer.

    The model (~400 MB) is downloaded on first use and cached by
    the ``transformers`` library in ``~/.cache/huggingface/``.
    Subsequent calls reuse the in-memory model.
    """

    _tokenizer = None
    _model = None
    _loaded = False
    _last_fail_ts: float = 0.0  # monotonic timestamp of last failure
    _RETRY_COOLDOWN = 300.0  # seconds before retrying after failure

    @classmethod
    def _ensure_loaded(cls) -> bool:
        """Load model + tokenizer on first call.  Returns False if
        dependencies are missing or loading fails.

        On failure, retries are gated by a 5-minute cooldown to
        avoid hammering a failing download/load on every call.
        """
        # Already loaded successfully — fast path
        if cls._loaded:
            return cls._model is not None

        if not _TRANSFORMERS_AVAILABLE:
            log.warning(
                "FinBERT unavailable — install transformers + "
                "torch to enable news sentiment scoring."
            )
            return False

        # Cooldown after a previous failure
        import time

        if cls._last_fail_ts > 0:
            elapsed = time.monotonic() - cls._last_fail_ts
            if elapsed < cls._RETRY_COOLDOWN:
                log.debug(
                    f"  FinBERT load cooldown: "
                    f"{cls._RETRY_COOLDOWN - elapsed:.0f}s remaining"
                )
                return False

        try:
            log.info(
                f"  Loading FinBERT model ({_FINBERT_MODEL_NAME})..."
            )
            cls._tokenizer = AutoTokenizer.from_pretrained(
                _FINBERT_MODEL_NAME
            )
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                _FINBERT_MODEL_NAME
            )
            cls._model.eval()
            cls._loaded = True  # only set on success
            log.info("  FinBERT model loaded.")
            return True
        except Exception as exc:
            log.warning(f"  FinBERT model load failed: {exc}")
            cls._model = None
            cls._tokenizer = None
            cls._last_fail_ts = time.monotonic()
            return False

    @classmethod
    def score_headlines(
        cls,
        headlines: list[str],
        batch_size: int = 16,
    ) -> list[HeadlineScore]:
        """Score a list of headlines.

        Args:
            headlines: Raw headline strings.
            batch_size: Tokenizer batch size (16 is fine for CPU).

        Returns:
            List of HeadlineScore in the same order as input.
            Returns empty list if model is unavailable.
        """
        if not headlines:
            return []
        if not cls._ensure_loaded():
            return []

        assert cls._tokenizer is not None
        assert cls._model is not None

        results: list[HeadlineScore] = []
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i : i + batch_size]
            tokens = cls._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = cls._model(**tokens)
            probs = torch.nn.functional.softmax(
                outputs.logits, dim=-1
            )
            # FinBERT label order: positive(0), negative(1), neutral(2)
            for j, headline in enumerate(batch):
                p = probs[j].tolist()
                results.append(
                    HeadlineScore(
                        headline=headline,
                        positive=p[0],
                        negative=p[1],
                        neutral=p[2],
                    )
                )

        return results


# ── Aggregation ───────────────────────────────────────────────────


@dataclass
class TickerSentiment:
    """Aggregated sentiment for a single ticker."""

    ticker: str
    score: float  # weighted mean sentiment, [-1, +1]
    num_headlines: int
    headlines: list[HeadlineScore]


def aggregate_ticker_sentiment(
    news_map: dict[str, list[str]],
    decay_lambda: float = 0.0,
) -> dict[str, TickerSentiment]:
    """Score and aggregate headlines per ticker.

    Args:
        news_map: {ticker: [headline_str, ...]} as returned by
            ``news_fetcher.fetch_news_batch()``.
        decay_lambda: Exponential recency decay factor.
            0.0 = equal weighting (default, since our headlines
            are already sorted by recency and cover ~3 days).

    Returns:
        {ticker: TickerSentiment} for tickers with headlines.
        Tickers with no headlines are omitted.
    """
    all_headlines: list[str] = []
    ticker_ranges: list[tuple[str, int, int]] = []

    for ticker, headlines in news_map.items():
        if not headlines:
            continue
        start = len(all_headlines)
        all_headlines.extend(headlines)
        end = len(all_headlines)
        ticker_ranges.append((ticker, start, end))

    if not all_headlines:
        return {}

    scored = FinBERTScorer.score_headlines(all_headlines)
    if not scored:
        return {}

    result: dict[str, TickerSentiment] = {}
    for ticker, start, end in ticker_ranges:
        ticker_scores = scored[start:end]
        if not ticker_scores:
            continue

        if decay_lambda > 0:
            # Weight by recency: first headline is newest
            weights = [
                math.exp(-decay_lambda * idx)
                for idx in range(len(ticker_scores))
            ]
        else:
            weights = [1.0] * len(ticker_scores)

        total_w = sum(weights)
        weighted_sent = sum(
            s.sentiment * w
            for s, w in zip(ticker_scores, weights)
        )
        avg_sent = weighted_sent / total_w if total_w > 0 else 0.0

        result[ticker] = TickerSentiment(
            ticker=ticker,
            score=avg_sent,
            num_headlines=len(ticker_scores),
            headlines=ticker_scores,
        )

    return result


# ── Composite score adjustment ────────────────────────────────────


def adjust_composite_scores(
    signals: list,
    sentiment: dict[str, TickerSentiment],
    weight: float = 5.0,
) -> list[tuple[str, float]]:
    """In-place adjust signal composite scores based on sentiment.

    Args:
        signals: List of Signal objects (modified in-place).
        sentiment: {ticker: TickerSentiment} from
            ``aggregate_ticker_sentiment()``.
        weight: Max points added/removed.  A sentiment of +1.0
            adds ``+weight`` to composite_score; -1.0 subtracts it.

    Returns:
        List of (ticker, delta) for logging — only tickers that
        were actually adjusted.
    """
    adjustments: list[tuple[str, float]] = []

    for sig in signals:
        ts = sentiment.get(sig.ticker)
        if ts is None:
            continue
        delta = ts.score * weight
        if abs(delta) < 0.01:
            continue
        sig.composite_score += delta
        adjustments.append((sig.ticker, delta))

    return adjustments
