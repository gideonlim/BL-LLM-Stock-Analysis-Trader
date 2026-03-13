"""
News headline fetcher for LLM view enrichment.

Provides recent news headlines for tickers to give the LLM
qualitative context when generating return predictions.

Supports multiple providers with graceful fallbacks:
    1. Yahoo Finance (free, no API key)
    2. Finnhub (free tier, requires API key)
    3. NewsAPI (free tier, requires API key)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)


def fetch_news_yahoo(
    ticker: str,
    max_headlines: int = 5,
) -> list[str]:
    """
    Fetch recent news headlines from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol.
        max_headlines: Maximum number of headlines.

    Returns:
        List of headline strings.
    """
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return []

        headlines = []
        for item in news[:max_headlines]:
            title = item.get("title", "")
            if title:
                headlines.append(title)

        return headlines

    except Exception as e:
        log.debug(f"Yahoo news fetch failed for {ticker}: {e}")
        return []


def fetch_news_finnhub(
    ticker: str,
    max_headlines: int = 5,
    days_back: int = 3,
) -> list[str]:
    """
    Fetch news from Finnhub API (requires FINNHUB_API_KEY).

    Args:
        ticker: Stock ticker symbol.
        max_headlines: Maximum headlines to return.
        days_back: How many days back to search.

    Returns:
        List of headline strings.
    """
    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        return []

    try:
        import urllib.request
        import json

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

        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        headlines = []
        for item in data[:max_headlines]:
            title = item.get("headline", "")
            if title:
                headlines.append(title)

        return headlines

    except Exception as e:
        log.debug(f"Finnhub news fetch failed for {ticker}: {e}")
        return []


def fetch_news(
    ticker: str,
    max_headlines: int = 5,
) -> list[str]:
    """
    Fetch news headlines using the best available provider.

    Tries providers in order: Yahoo Finance → Finnhub.

    Args:
        ticker: Stock ticker symbol.
        max_headlines: Maximum headlines to return.

    Returns:
        List of headline strings.
    """
    # Try Yahoo first (no API key needed)
    headlines = fetch_news_yahoo(ticker, max_headlines)
    if headlines:
        return headlines

    # Fallback to Finnhub
    headlines = fetch_news_finnhub(ticker, max_headlines)
    if headlines:
        return headlines

    return []


def fetch_news_batch(
    tickers: list[str],
    max_headlines: int = 5,
) -> dict[str, list[str]]:
    """
    Fetch news headlines for multiple tickers.

    Args:
        tickers: List of ticker symbols.
        max_headlines: Max headlines per ticker.

    Returns:
        Dict of ticker → list of headline strings.
    """
    news_map: dict[str, list[str]] = {}

    for ticker in tickers:
        headlines = fetch_news(ticker, max_headlines)
        if headlines:
            news_map[ticker] = headlines
            log.debug(
                f"  News: {ticker} → {len(headlines)} headlines"
            )

    log.info(
        f"  News: fetched headlines for "
        f"{len(news_map)}/{len(tickers)} tickers"
    )
    return news_map
