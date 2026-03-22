"""
Post-Earnings Announcement Drift (PEAD) data enrichment.

Fetches historical earnings surprise data from yfinance and adds
columns to the OHLCV DataFrame that the PEAD_Drift strategy reads:

  - PEAD_Surprise_Pct:  (actual - estimate) / |estimate| × 100
  - PEAD_Days_Since:    trading days since the last earnings date
  - PEAD_Gap_Pct:       open-to-previous-close gap on earnings day

The academic evidence for PEAD is deep:
  - Ball & Brown (1968): original discovery
  - Bernard & Thomas (1989): persistence for 60+ trading days
  - Novy-Marx (2015): earnings momentum explains much of
    price momentum
  - NBER 2025: PEAD still linked to underreaction around
    earnings surprises

Data source: yfinance .earnings_dates (free, no API key needed).
Returns a DataFrame with columns: Reported EPS, EPS Estimate.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Module-level cache: {ticker: earnings_df}
_pead_cache: Dict[str, Optional[pd.DataFrame]] = {}
_cache_lock = threading.Lock()


def fetch_earnings_surprises(
    ticker: str,
    use_cache: bool = True,
    retries: int = 2,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical earnings dates and surprises from yfinance.

    Returns a DataFrame indexed by earnings date with columns:
      - reported_eps: actual EPS reported
      - estimated_eps: consensus estimate
      - surprise_pct: (actual - estimate) / |estimate| × 100

    Returns None if data is unavailable.  Retries on empty
    results (Yahoo rate-limits can return empty instead of data).
    """
    with _cache_lock:
        if use_cache and ticker in _pead_cache:
            return _pead_cache[ticker]

    import time

    try:
        import yfinance as yf

        ed = None
        t = yf.Ticker(ticker)  # reuse across retries
        for attempt in range(1 + retries):
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                break
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

        if ed is None or ed.empty:
            log.debug(
                f"{ticker}: No earnings data from yfinance "
                f"(Yahoo coverage gap, not delisted)"
            )
            with _cache_lock:
                if use_cache:
                    _pead_cache[ticker] = None
            return None

        # yfinance .earnings_dates has columns like:
        #   "Reported EPS", "EPS Estimate",
        #   "Surprise(%)" (sometimes present)
        result = pd.DataFrame(index=ed.index)

        if "Reported EPS" in ed.columns:
            result["reported_eps"] = pd.to_numeric(
                ed["Reported EPS"], errors="coerce"
            )
        else:
            with _cache_lock:
                if use_cache:
                    _pead_cache[ticker] = None
            return None

        if "EPS Estimate" in ed.columns:
            result["estimated_eps"] = pd.to_numeric(
                ed["EPS Estimate"], errors="coerce"
            )
        else:
            with _cache_lock:
                if use_cache:
                    _pead_cache[ticker] = None
            return None

        # Compute surprise percentage
        # Guard against zero/NaN estimates
        est_abs = result["estimated_eps"].abs().replace(0, np.nan)
        result["surprise_pct"] = (
            (result["reported_eps"] - result["estimated_eps"])
            / est_abs
            * 100
        )

        # Drop rows where we don't have usable data:
        # - Future dates have NaN for reported_eps
        # - Zero estimates produce NaN surprise
        result = result.dropna(
            subset=["reported_eps", "estimated_eps", "surprise_pct"],
        )

        if result.empty:
            with _cache_lock:
                if use_cache:
                    _pead_cache[ticker] = None
            return None

        # Ensure index is tz-naive for alignment with OHLCV data
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)

        # Sort chronologically
        result = result.sort_index()

        with _cache_lock:
            if use_cache:
                _pead_cache[ticker] = result
        return result

    except Exception as e:
        log.debug(f"PEAD earnings fetch failed for {ticker}: {e}")
        with _cache_lock:
            if use_cache:
                _pead_cache[ticker] = None
        return None


def enrich_with_pead(
    df: pd.DataFrame,
    ticker: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Add PEAD columns to an existing OHLCV + indicators DataFrame.

    Columns added:
      - PEAD_Surprise_Pct: earnings surprise % for the most recent
        earnings announcement (forward-filled until next earnings)
      - PEAD_Days_Since: trading days since the last earnings date
      - PEAD_Gap_Pct: gap % on the earnings day (open vs prev close)

    If earnings data is unavailable, columns are added as NaN
    (the PEAD_Drift strategy handles this gracefully).
    """
    n = len(df)

    # Default: NaN columns (strategy returns HOLD when these are NaN)
    df["PEAD_Surprise_Pct"] = np.nan
    df["PEAD_Days_Since"] = np.nan
    df["PEAD_Gap_Pct"] = np.nan

    earnings = fetch_earnings_surprises(ticker, use_cache=use_cache)
    if earnings is None or earnings.empty:
        return df

    # Align earnings dates to trading days in the DataFrame
    # The earnings_dates index is datetime; the OHLCV index is also
    # datetime (or date). We need to find the nearest trading day.
    df_dates = df.index.normalize() if hasattr(
        df.index, "normalize"
    ) else pd.DatetimeIndex(df.index)

    for earn_date, row in earnings.iterrows():
        earn_dt = pd.Timestamp(earn_date).normalize()
        surprise_pct = row["surprise_pct"]

        if pd.isna(surprise_pct):
            continue

        # Find the first trading day on or after the earnings date
        # (earnings often report after-hours, so the "event" bar
        # is the next trading day)
        mask = df_dates >= earn_dt
        if not mask.any():
            continue

        event_idx = df_dates[mask][0]
        event_pos = df.index.get_loc(event_idx)

        # Compute gap % on the event day
        if event_pos > 0:
            prev_close = df["Close"].iloc[event_pos - 1]
            event_open = df["Open"].iloc[event_pos]
            if prev_close > 0:
                gap_pct = (
                    (event_open - prev_close) / prev_close * 100
                )
            else:
                gap_pct = 0.0
        else:
            gap_pct = 0.0

        # Forward-fill surprise and gap from event day until the
        # next earnings event (or end of data)
        # Find next earnings date after this one
        later_earnings = earnings.index[
            earnings.index > earn_date
        ]
        if len(later_earnings) > 0:
            next_earn_dt = pd.Timestamp(
                later_earnings[0]
            ).normalize()
            end_mask = df_dates < next_earn_dt
            fill_mask = mask & end_mask
        else:
            fill_mask = mask

        # Fill surprise and gap for the drift window
        df.loc[fill_mask, "PEAD_Surprise_Pct"] = surprise_pct
        df.loc[fill_mask, "PEAD_Gap_Pct"] = gap_pct

        # Compute days-since as the count of trading days
        # from the event day
        fill_positions = np.where(fill_mask)[0]
        if len(fill_positions) > 0:
            days_since = fill_positions - event_pos
            df.iloc[
                fill_positions,
                df.columns.get_loc("PEAD_Days_Since"),
            ] = days_since

    return df


def prefetch_earnings_parallel(
    tickers: list[str],
    max_workers: int = 4,
    batch_size: int = 50,
    batch_delay: float = 2.0,
) -> None:
    """Pre-fetch earnings data for many tickers in parallel.

    Uses ThreadPoolExecutor to run yfinance API calls concurrently,
    but processes tickers in batches with cooldown delays between
    batches to avoid Yahoo Finance rate limiting.

    Results are stored in the module-level ``_pead_cache`` so that
    subsequent calls to ``enrich_with_pead()`` hit the cache and
    skip the network call.

    Parameters
    ----------
    tickers : list[str]
        Tickers to pre-fetch earnings for.
    max_workers : int
        Max concurrent threads per batch (default 4).
    batch_size : int
        Tickers per batch before pausing (default 50).
    batch_delay : float
        Seconds to wait between batches (default 2.0).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Only fetch tickers not already cached
    with _cache_lock:
        uncached = [t for t in tickers if t not in _pead_cache]
    if not uncached:
        return

    log.info(
        f"Pre-fetching earnings data for {len(uncached)} tickers "
        f"({max_workers} threads, batches of {batch_size})..."
    )

    def _fetch_one(ticker: str) -> tuple[str, bool]:
        try:
            result = fetch_earnings_surprises(
                ticker, use_cache=True, retries=1,
            )
            return ticker, result is not None
        except Exception:
            return ticker, False

    ok = 0
    failed: list[str] = []
    total = len(uncached)

    # Process in batches to avoid rate limiting
    for batch_start in range(0, total, batch_size):
        batch = uncached[batch_start:batch_start + batch_size]

        try:
            with ThreadPoolExecutor(
                max_workers=max_workers
            ) as pool:
                futures = {
                    pool.submit(_fetch_one, t): t
                    for t in batch
                }
                for future in as_completed(futures):
                    _ticker, success = future.result()
                    if success:
                        ok += 1
                    else:
                        failed.append(_ticker)
        except KeyboardInterrupt:
            log.info("Earnings pre-fetch interrupted by user")
            break

        # Cooldown between batches (skip after last batch)
        if batch_start + batch_size < total:
            time.sleep(batch_delay)

    log.info(
        f"Earnings pre-fetch complete: {ok}/{total} succeeded"
    )
    if failed:
        log.warning(
            f"No earnings data for {len(failed)} tickers: "
            f"{', '.join(sorted(failed))}"
        )


def clear_cache() -> None:
    """Clear the module-level PEAD cache."""
    _pead_cache.clear()
