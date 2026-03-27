"""News ingestion — fetches headlines with full provenance.

Every item preserves its original timestamp, source, and ticker
context.  Supports Yahoo Finance (free) and Finnhub (free tier).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta, timezone

from research_pipeline.config import ResearchConfig
from research_pipeline.models import NewsItem

log = logging.getLogger(__name__)


# ── Yahoo Finance ──────────────────────────────────────────────

def _fetch_yahoo(
    ticker: str,
    max_items: int = 10,
) -> list[NewsItem]:
    """Fetch news from Yahoo Finance via yfinance."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        raw = stock.news or []

        items: list[NewsItem] = []
        for entry in raw[:max_items]:
            title = entry.get("title", "")
            if not title:
                continue

            # yfinance timestamps are unix seconds
            ts = entry.get("providerPublishTime", 0)
            pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            items.append(NewsItem(
                headline=title,
                source="yahoo",
                published_at=pub_dt,
                ticker=ticker,
                url=entry.get("link", ""),
            ))

        return items

    except Exception as e:
        log.debug(f"Yahoo fetch failed for {ticker}: {e}")
        return []


# ── Finnhub ────────────────────────────────────────────────────

def _fetch_finnhub(
    ticker: str,
    api_key: str,
    days_back: int = 3,
    max_items: int = 10,
) -> list[NewsItem]:
    """Fetch news from Finnhub API."""
    if not api_key:
        return []

    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (
            datetime.now() - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}"
            f"&from={start}&to={end}"
            f"&token={api_key}"
        )

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        items: list[NewsItem] = []
        for entry in data[:max_items]:
            headline = entry.get("headline", "")
            if not headline:
                continue

            ts = entry.get("datetime", 0)
            pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            items.append(NewsItem(
                headline=headline,
                source="finnhub",
                published_at=pub_dt,
                ticker=ticker,
                url=entry.get("url", ""),
                summary=entry.get("summary", ""),
            ))

        return items

    except Exception as e:
        log.debug(f"Finnhub fetch failed for {ticker}: {e}")
        return []


# ── Finnhub general market news ────────────────────────────────

def _fetch_finnhub_general(
    api_key: str,
    category: str = "general",
    max_items: int = 20,
) -> list[NewsItem]:
    """Fetch general market news (not ticker-specific)."""
    if not api_key:
        return []

    try:
        url = (
            f"https://finnhub.io/api/v1/news"
            f"?category={category}"
            f"&token={api_key}"
        )

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        items: list[NewsItem] = []
        for entry in data[:max_items]:
            headline = entry.get("headline", "")
            if not headline:
                continue

            ts = entry.get("datetime", 0)
            pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            items.append(NewsItem(
                headline=headline,
                source="finnhub_general",
                published_at=pub_dt,
                ticker=None,
                url=entry.get("url", ""),
                summary=entry.get("summary", ""),
            ))

        return items

    except Exception as e:
        log.debug(f"Finnhub general news fetch failed: {e}")
        return []


# ── Public interface ───────────────────────────────────────────

def ingest_news(config: ResearchConfig) -> list[NewsItem]:
    """Fetch news from all available sources.

    Returns a deduplicated, timestamp-sorted list of NewsItems.
    """
    all_items: list[NewsItem] = []

    # 1. General market news (Finnhub)
    general = _fetch_finnhub_general(
        config.finnhub_api_key, max_items=20,
    )
    all_items.extend(general)
    log.info(f"Ingested {len(general)} general market headlines")

    # 2. Per-ticker news from watchlist
    for ticker in config.watchlist_tickers:
        # Try Finnhub first (richer data), fall back to Yahoo
        items = _fetch_finnhub(
            ticker,
            config.finnhub_api_key,
            days_back=config.news_days_back,
            max_items=config.max_headlines_per_ticker,
        )
        if not items:
            items = _fetch_yahoo(
                ticker,
                max_items=config.max_headlines_per_ticker,
            )

        all_items.extend(items)

    # Deduplicate by headline text (case-insensitive)
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for item in all_items:
        key = item.headline.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # Sort by publish time (newest first)
    unique.sort(key=lambda x: x.published_at, reverse=True)

    log.info(
        f"Ingestion complete: {len(unique)} unique headlines "
        f"from {len(config.watchlist_tickers)} tickers"
    )

    return unique
