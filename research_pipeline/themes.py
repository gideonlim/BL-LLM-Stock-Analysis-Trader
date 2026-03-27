"""Theme detection — finds topics that are suddenly accelerating.

This is the key step from your research report: not "is sentiment
positive?" but "what topics are suddenly getting a lot of attention,
and what entities are being linked together?"

Groups extracted signals by theme tag, measures concentration, and
surfaces clusters that cross the minimum-mentions threshold.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

from research_pipeline.config import ResearchConfig
from research_pipeline.models import ExtractedSignal, ThemeCluster

log = logging.getLogger(__name__)


def _normalise_theme(raw: str) -> str:
    """Lowercase and strip whitespace for grouping."""
    return raw.strip().lower()


def detect_themes(
    signals: list[ExtractedSignal],
    config: ResearchConfig,
) -> list[ThemeCluster]:
    """Group signals by theme and surface accelerating topics.

    Returns ThemeClusters sorted by mention count (descending).
    Only themes with >= min_theme_mentions are included.
    """
    # ── Group by normalised theme ──────────────────────────────
    theme_signals: dict[str, list[ExtractedSignal]] = defaultdict(list)

    for sig in signals:
        for raw_theme in sig.themes:
            key = _normalise_theme(raw_theme)
            if key:
                theme_signals[key].append(sig)

    # ── Build ThemeClusters ────────────────────────────────────
    clusters: list[ThemeCluster] = []

    for theme_key, sigs in theme_signals.items():
        if len(sigs) < config.min_theme_mentions:
            continue

        # Aggregate sentiment (-1 / 0 / +1)
        sentiment_map = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0,
        }
        sentiments = [
            sentiment_map.get(s.sentiment, 0.0) for s in sigs
        ]
        avg_sent = sum(sentiments) / len(sentiments)

        # Collect all related tickers (deduplicated, sorted)
        all_tickers: set[str] = set()
        for s in sigs:
            all_tickers.update(s.tickers)

        # Sample headlines (up to 5)
        sample = tuple(s.headline for s in sigs[:5])

        # Time range
        timestamps = [s.published_at for s in sigs]
        first = min(timestamps)
        last = max(timestamps)

        clusters.append(ThemeCluster(
            theme=theme_key,
            mention_count=len(sigs),
            avg_sentiment=round(avg_sent, 3),
            related_tickers=tuple(sorted(all_tickers)),
            sample_headlines=sample,
            first_seen=first,
            last_seen=last,
        ))

    # Sort by mention count descending
    clusters.sort(key=lambda c: c.mention_count, reverse=True)

    log.info(
        f"Theme detection: {len(clusters)} themes above "
        f"threshold ({config.min_theme_mentions} mentions) "
        f"from {len(theme_signals)} total themes"
    )

    return clusters
