"""Historical data loader for backtesting.

Fetches 1-min bars from Alpaca's StockHistoricalDataClient and
caches them as parquet files so subsequent runs are instant.

Cache layout::

    day_trader/backtest/data/
      AAPL/2024-07.parquet
      AAPL/2024-08.parquet
      MSFT/2024-07.parquet
      ...

Each parquet has columns: timestamp, open, high, low, close, volume,
vwap, trade_count. One file per (ticker, year-month).

For backtests that span 24 months × 50 tickers, the cold-start
download is ~20 min (Alpaca rate limits). Subsequent runs load from
parquet in seconds.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, time
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from day_trader.calendar import ET
from day_trader.models import Bar

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("day_trader/backtest/data")


def load_bars(
    tickers: list[str],
    start_date: date,
    end_date: date,
    *,
    cache_dir: Optional[Path] = None,
    historical_client=None,
) -> dict[date, dict[str, list[Bar]]]:
    """Load minute bars grouped by date and ticker.

    Returns ``{date: {ticker: [Bar, Bar, ...]}}`` with bars in
    chronological order, suitable for :class:`BacktestEngine.run`.

    Tries cache first (parquet); fetches from Alpaca on miss.
    If ``historical_client`` is None, returns only cached data.
    """
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    result: dict[date, dict[str, list[Bar]]] = {}

    for ticker in tickers:
        bars = _load_ticker(
            ticker, start_date, end_date,
            cache_dir=cache_dir,
            client=historical_client,
        )
        for bar in bars:
            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ZoneInfo("UTC"))
            bar_date = ts.astimezone(ET).date()
            if bar_date not in result:
                result[bar_date] = {}
            if ticker not in result[bar_date]:
                result[bar_date][ticker] = []
            result[bar_date][ticker].append(bar)

    # Sort bars within each (date, ticker) group
    for day_bars in result.values():
        for ticker_bars in day_bars.values():
            ticker_bars.sort(key=lambda b: b.timestamp)

    log.info(
        "Loaded bars: %d tickers, %d dates, %d total bars",
        len(tickers), len(result),
        sum(
            len(bars)
            for day in result.values()
            for bars in day.values()
        ),
    )
    return result


def _load_ticker(
    ticker: str,
    start_date: date,
    end_date: date,
    *,
    cache_dir: Path,
    client,
) -> list[Bar]:
    """Load all minute bars for one ticker across the date range."""
    bars: list[Bar] = []

    # Try cache first
    cached = _load_from_cache(ticker, start_date, end_date, cache_dir)
    if cached:
        log.debug("Cache hit: %s (%d bars)", ticker, len(cached))
        return cached

    # Fetch from Alpaca
    if client is None:
        log.debug("No client, no cache for %s — returning empty", ticker)
        return []

    fetched = _fetch_from_alpaca(ticker, start_date, end_date, client)
    if fetched:
        _save_to_cache(ticker, fetched, cache_dir)
        bars = fetched

    return bars


def _load_from_cache(
    ticker: str,
    start_date: date,
    end_date: date,
    cache_dir: Path,
) -> list[Bar]:
    """Read parquet files for (ticker, date-range) from cache."""
    try:
        import pandas as pd
    except ImportError:
        return []

    ticker_dir = cache_dir / ticker.upper()
    if not ticker_dir.exists():
        return []

    bars: list[Bar] = []
    # Iterate month-files that overlap with [start_date, end_date]
    d = date(start_date.year, start_date.month, 1)
    while d <= end_date:
        month_file = ticker_dir / f"{d.strftime('%Y-%m')}.parquet"
        if month_file.exists():
            try:
                df = pd.read_parquet(month_file)
                for _, row in df.iterrows():
                    ts = row.get("timestamp")
                    if ts is not None and hasattr(ts, "to_pydatetime"):
                        ts = ts.to_pydatetime()
                    if isinstance(ts, datetime):
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
                        bar_date = ts.astimezone(ET).date()
                        if start_date <= bar_date <= end_date:
                            bars.append(Bar(
                                ticker=ticker.upper(),
                                timestamp=ts,
                                open=float(row.get("open", 0) or 0),
                                high=float(row.get("high", 0) or 0),
                                low=float(row.get("low", 0) or 0),
                                close=float(row.get("close", 0) or 0),
                                volume=float(row.get("volume", 0) or 0),
                                vwap=float(row.get("vwap", 0) or 0),
                                trade_count=int(
                                    row.get("trade_count", 0) or 0
                                ),
                            ))
            except Exception as exc:
                log.warning(
                    "Cache read failed for %s: %s", month_file, exc,
                )
        # Next month
        if d.month == 12:
            d = date(d.year + 1, 1, 1)
        else:
            d = date(d.year, d.month + 1, 1)

    return bars


def _fetch_from_alpaca(
    ticker: str,
    start_date: date,
    end_date: date,
    client,
) -> list[Bar]:
    """Fetch minute bars from Alpaca's historical data API."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start_dt = datetime.combine(start_date, time(4, 0), tzinfo=ET)
        end_dt = datetime.combine(end_date, time(20, 0), tzinfo=ET)

        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt,
        )
        log.info("Fetching %s bars from Alpaca (%s → %s)...", ticker, start_date, end_date)
        response = client.get_stock_bars(req)

        bars: list[Bar] = []
        raw_bars = []
        if hasattr(response, "data") and isinstance(response.data, dict):
            raw_bars = list(response.data.get(ticker, []))
        elif hasattr(response, "df"):
            df = response.df
            if df is not None and not df.empty:
                from types import SimpleNamespace
                raw_bars = [
                    SimpleNamespace(**{
                        k: v.item() if hasattr(v, "item") else v
                        for k, v in row.to_dict().items()
                    })
                    for _, row in df.iterrows()
                ]

        for b in raw_bars:
            ts = getattr(b, "timestamp", None)
            if ts is not None and hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if isinstance(ts, datetime) and ts.tzinfo is None:
                ts = ts.replace(tzinfo=ZoneInfo("UTC"))

            bars.append(Bar(
                ticker=ticker.upper(),
                timestamp=ts or datetime.now(tz=ZoneInfo("UTC")),
                open=float(getattr(b, "open", 0) or 0),
                high=float(getattr(b, "high", 0) or 0),
                low=float(getattr(b, "low", 0) or 0),
                close=float(getattr(b, "close", 0) or 0),
                volume=float(getattr(b, "volume", 0) or 0),
                vwap=float(getattr(b, "vwap", 0) or 0),
                trade_count=int(getattr(b, "trade_count", 0) or 0),
            ))

        log.info("Fetched %d bars for %s", len(bars), ticker)
        return bars

    except Exception as exc:
        log.error("Alpaca fetch failed for %s: %s", ticker, exc)
        return []


def _save_to_cache(
    ticker: str,
    bars: list[Bar],
    cache_dir: Path,
) -> None:
    """Write bars to monthly parquet files."""
    try:
        import pandas as pd
    except ImportError:
        log.debug("pandas not available — skipping cache write")
        return

    # Group by year-month
    by_month: dict[str, list[dict]] = {}
    for bar in bars:
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC"))
        key = ts.astimezone(ET).strftime("%Y-%m")
        if key not in by_month:
            by_month[key] = []
        by_month[key].append({
            "timestamp": ts,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "trade_count": bar.trade_count,
        })

    ticker_dir = cache_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)

    for month_key, rows in by_month.items():
        path = ticker_dir / f"{month_key}.parquet"
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        log.debug("Cached %d bars → %s", len(rows), path)
