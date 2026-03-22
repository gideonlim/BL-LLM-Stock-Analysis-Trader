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
from typing import Dict, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Module-level cache: {ticker: earnings_df}
_pead_cache: Dict[str, Optional[pd.DataFrame]] = {}


def fetch_earnings_surprises(
    ticker: str,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical earnings dates and surprises from yfinance.

    Returns a DataFrame indexed by earnings date with columns:
      - reported_eps: actual EPS reported
      - estimated_eps: consensus estimate
      - surprise_pct: (actual - estimate) / |estimate| × 100

    Returns None if data is unavailable.
    """
    if use_cache and ticker in _pead_cache:
        return _pead_cache[ticker]

    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        ed = t.earnings_dates

        if ed is None or ed.empty:
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
            if use_cache:
                _pead_cache[ticker] = None
            return None

        if "EPS Estimate" in ed.columns:
            result["estimated_eps"] = pd.to_numeric(
                ed["EPS Estimate"], errors="coerce"
            )
        else:
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
            if use_cache:
                _pead_cache[ticker] = None
            return None

        # Ensure index is tz-naive for alignment with OHLCV data
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)

        # Sort chronologically
        result = result.sort_index()

        if use_cache:
            _pead_cache[ticker] = result
        return result

    except Exception as e:
        log.debug(f"PEAD earnings fetch failed for {ticker}: {e}")
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


def clear_cache() -> None:
    """Clear the module-level PEAD cache."""
    _pead_cache.clear()
