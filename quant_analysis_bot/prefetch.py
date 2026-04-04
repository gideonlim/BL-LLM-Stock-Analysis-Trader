"""Pre-fetch market data after market close for next-day signal generation.

Downloads all ticker OHLCV data and regime data at 6 PM ET so that
the 7 AM signal generation run can load from cache instead of hitting
yfinance, avoiding rate limits during the time-critical window.

Cache validity is determined by a manifest file that records whether
the market was closed at fetch time.  If the market was open (data
may be incomplete), the next workflow re-fetches.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# US equity market hours (regular session)
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def _et_now() -> datetime:
    """Return the current time in US Eastern."""
    return datetime.now(ET)


def is_market_open() -> bool:
    """Check if the US equity market is currently in regular session.

    Returns ``True`` if it is a weekday and the current ET time is
    between 9:30 AM and 4:00 PM (exclusive of close).  Does not
    account for market holidays — worst case is a false positive that
    causes a re-fetch, which is acceptable.
    """
    now = _et_now()
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return _MARKET_OPEN <= now.time() < _MARKET_CLOSE


def get_previous_trading_day(reference_date: date) -> date:
    """Return the most recent weekday before *reference_date*.

    Mon → Fri, Sat → Fri, Sun → Fri, Tue–Fri → yesterday.
    No holiday awareness — a cache miss after a holiday simply
    triggers a fresh download.
    """
    day = reference_date - timedelta(days=1)
    # Walk back over weekends
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


# ── Manifest I/O ───────────────────────────────────────────────────

def write_manifest(
    cache_dir: str,
    date_str: str,
    tickers: list[str],
    regime_cached: bool,
    market_closed: bool,
) -> str:
    """Write a prefetch manifest to *cache_dir*.

    The manifest records which tickers were successfully prefetched
    and whether the market was closed at fetch time.  Uses atomic
    write (temp file + ``os.replace``) to prevent corruption.
    """
    manifest = {
        "date_str": date_str,
        "market_closed": market_closed,
        "ticker_count": len(tickers),
        "tickers": tickers,
        "regime_cached": regime_cached,
    }

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"prefetch_{date_str}.json")

    fd, tmp_path = tempfile.mkstemp(
        dir=cache_dir, suffix=".tmp", prefix="prefetch_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    log.info(
        f"Prefetch manifest written: {path} "
        f"({len(tickers)} tickers, market_closed={market_closed})"
    )
    return path


def read_manifest(cache_dir: str, date_str: str) -> dict | None:
    """Read a prefetch manifest.  Returns ``None`` if missing or corrupt."""
    path = os.path.join(cache_dir, f"prefetch_{date_str}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"Corrupt prefetch manifest {path}: {exc}")
        return None


# ── Validation ─────────────────────────────────────────────────────

def validate_prefetch_cache(
    cache_dir: str,
    reference_date: date,
) -> tuple[str, list[str]] | None:
    """Check whether a valid prefetch cache exists for *reference_date*.

    Parameters
    ----------
    cache_dir : str
        Path to the cache directory.
    reference_date : date
        Today's date **in ET** (caller must use ``_et_now().date()``).

    Returns
    -------
    (prev_day_str, tickers) if a valid prefetch cache exists, else None.
    ``prev_day_str`` is the YYYYMMDD key for the cached parquet files.
    ``tickers`` is the list of tickers confirmed in the manifest.
    """
    prev_day = get_previous_trading_day(reference_date)
    prev_day_str = prev_day.strftime("%Y%m%d")

    manifest = read_manifest(cache_dir, prev_day_str)
    if manifest is None:
        return None

    if not manifest.get("market_closed", False):
        log.info(
            f"Prefetch cache for {prev_day_str} was fetched while "
            f"market was open — skipping"
        )
        return None

    tickers = manifest.get("tickers", [])
    if not tickers:
        return None

    log.info(
        f"Valid prefetch cache found for {prev_day_str} "
        f"({len(tickers)} tickers)"
    )
    return (prev_day_str, tickers)


# ── Cache pruning ──────────────────────────────────────────────────

def prune_old_cache(cache_dir: str, keep_days: int = 2) -> None:
    """Remove parquet files, regime caches, and manifests older than
    *keep_days* trading days.  Prevents the ``actions/cache`` archive
    from growing unbounded.
    """
    if not os.path.isdir(cache_dir):
        return

    today = _et_now().date()
    # Build set of date strings to keep
    keep_dates: set[str] = set()
    d = today
    kept = 0
    # Walk back up to 2 weeks to find enough trading days
    for _ in range(14):
        if d.weekday() < 5:
            keep_dates.add(d.strftime("%Y%m%d"))
            kept += 1
            if kept > keep_days:
                break
        d -= timedelta(days=1)

    removed = 0
    for fname in os.listdir(cache_dir):
        # Match files like AAPL_20260406.parquet, prefetch_20260406.json,
        # regime_20260406_lb500_vix25.0.parquet
        if not (fname.endswith(".parquet") or fname.endswith(".json")):
            continue

        # Extract date portion — always 8 digits after the first underscore
        parts = fname.split("_")
        file_date_str = None
        for part in parts:
            # Strip extension and check for 8-digit date
            candidate = part.split(".")[0]
            if len(candidate) == 8 and candidate.isdigit():
                file_date_str = candidate
                break

        if file_date_str is None:
            continue

        if file_date_str not in keep_dates:
            try:
                os.unlink(os.path.join(cache_dir, fname))
                removed += 1
            except OSError:
                pass

    if removed:
        log.info(f"Pruned {removed} old cache files")


# ── Main prefetch entry point ──────────────────────────────────────

def run_prefetch(config: dict, tickers: list[str]) -> None:
    """Download all ticker data and regime data, write manifest.

    Designed to run at 6 PM ET after market close.

    Parameters
    ----------
    config : dict
        Standard config from ``load_config()``.
    tickers : list[str]
        Pre-resolved ticker list (from ``fetch_top_us_stocks`` or CLI).
    """
    from quant_analysis_bot.data import batch_fetch_data
    from quant_analysis_bot.regime import prefetch_regime_data

    market_closed = not is_market_open()
    cache_dir = config.get("data_cache_dir", "cache")
    today_et = _et_now().date()
    date_str = today_et.strftime("%Y%m%d")

    log.info(f"{'=' * 60}")
    log.info(f"  Prefetch Mode — {today_et}")
    log.info(f"  Market closed: {market_closed}")
    log.info(f"  Tickers: {len(tickers)}")
    log.info(f"{'=' * 60}")

    # 1. Batch download all ticker data (skip prefetch fallback —
    #    we are the prefetch, so we always want fresh data)
    log.info("Batch downloading ticker data...")
    price_cache = batch_fetch_data(
        tickers,
        config["lookback_days"],
        cache_dir,
        _skip_prefetch=True,
    )

    # 2. Validate each ticker's parquet has today's bar
    verified_tickers: list[str] = []
    for ticker in tickers:
        if ticker not in price_cache:
            continue
        df = price_cache[ticker]
        if df.empty:
            continue
        # Check last bar date matches today's ET date
        last_bar = df.index[-1]
        if hasattr(last_bar, "date"):
            last_date = last_bar.date()
        else:
            last_date = pd.Timestamp(last_bar).date()
        if last_date == today_et:
            verified_tickers.append(ticker)
        else:
            log.debug(
                f"  {ticker}: last bar {last_date} != today {today_et}, "
                f"excluding from manifest"
            )

    log.info(
        f"Verified {len(verified_tickers)}/{len(tickers)} tickers "
        f"have today's bar"
    )

    # 3. Prefetch regime data (VIX + SPY)
    regime_cached = False
    try:
        prefetch_regime_data(
            cache_dir=cache_dir,
            lookback_days=config.get("lookback_days", 500),
            vix_fear_threshold=config.get("vix_fear_threshold", 25.0),
        )
        regime_cached = True
        log.info("Regime data prefetched")
    except Exception as exc:
        log.warning(f"Regime prefetch failed: {exc}")

    # 4. Prune old cache files
    prune_old_cache(cache_dir)

    # 5. Write manifest
    write_manifest(
        cache_dir=cache_dir,
        date_str=date_str,
        tickers=verified_tickers,
        regime_cached=regime_cached,
        market_closed=market_closed,
    )

    log.info(
        f"Prefetch complete: {len(verified_tickers)} tickers, "
        f"regime={'yes' if regime_cached else 'no'}"
    )
