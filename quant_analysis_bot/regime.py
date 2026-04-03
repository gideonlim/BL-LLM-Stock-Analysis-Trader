"""Market regime detection for state-dependent strategy gating.

Fetches historical VIX and SPY data to compute a daily fear/stress
indicator.  Mean-reversion strategies work best during stressed regimes
(high VIX, SPY below trend), so this module provides the signal that
gates those strategies.

The regime data is fetched *once* per run and cached at module level.
Each ticker's DataFrame is enriched via ``enrich_with_regime()`` which
date-joins the regime columns by index.

Design notes
------------
* **Historical, not just today.**  We need regime state on every
  historical date so backtests properly reflect when mean-reversion
  would have been active.  A single "current VIX" snapshot would
  make backtests unrealistic.
* **Two independent fear signals:**
  1. VIX elevated (> threshold, default 25)
  2. SPY below its 200-day SMA
  Either alone triggers ``Regime_Fear = True``.  Both together
  indicate higher conviction.
* **Graceful degradation:**  If VIX or SPY data is unavailable,
  ``Regime_Fear`` defaults to True (i.e., mean-reversion strategies
  run unfiltered, same as before this feature existed).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Module-level cache — fetched once per process run
_regime_cache: Optional[pd.DataFrame] = None


def fetch_regime_data(
    lookback_days: int = 600,
    *,
    vix_fear_threshold: float = 25.0,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical VIX + SPY data and compute daily regime state.

    Parameters
    ----------
    lookback_days : int
        How many calendar days of history to fetch.
    vix_fear_threshold : float
        VIX level above which the regime is classified as fearful.
    use_cache : bool
        If True, return cached result on subsequent calls.

    Returns
    -------
    pd.DataFrame
        Indexed by date (DatetimeIndex) with columns:
        - ``VIX_Close``: VIX closing level
        - ``SPY_Close``: SPY closing price
        - ``SPY_SMA_200``: SPY 200-day simple moving average
        - ``SPY_Below_SMA200``: bool, SPY < its 200-SMA
        - ``VIX_Elevated``: bool, VIX > threshold
        - ``Regime_Fear``: bool, either fear signal active

    On data failure, returns an empty DataFrame so callers can
    proceed without regime filtering.
    """
    global _regime_cache

    if use_cache and _regime_cache is not None:
        return _regime_cache

    regime_df = _build_regime_df(
        lookback_days=lookback_days,
        vix_fear_threshold=vix_fear_threshold,
    )

    if use_cache:
        _regime_cache = regime_df

    return regime_df


def _build_regime_df(
    lookback_days: int,
    vix_fear_threshold: float,
) -> pd.DataFrame:
    """Internal: fetch and compute regime DataFrame."""
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance not available — regime data disabled")
        return pd.DataFrame()

    from datetime import datetime, timedelta

    end = datetime.now()
    start = end - timedelta(days=lookback_days + 250)
    period_str = f"{lookback_days + 250}d"

    import time

    def _download_with_retry(
        symbol: str, max_retries: int = 3
    ) -> pd.DataFrame:
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    symbol,
                    period=period_str,
                    progress=False,
                    auto_adjust=True,
                )
                if data is not None and not data.empty:
                    return data
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = "RateLimit" in exc_str or (
                    "Too Many Requests" in exc_str
                )
                if is_rate_limit:
                    wait = 30 * (attempt + 1)
                    log.warning(
                        f"{symbol}: Rate limited, "
                        f"waiting {wait}s before retry "
                        f"({attempt + 1}/{max_retries})..."
                    )
                else:
                    wait = 5 * (attempt + 1)
                    log.debug(
                        f"{symbol} fetch attempt "
                        f"{attempt + 1} failed: {exc}"
                    )
                if attempt < max_retries - 1:
                    time.sleep(wait)
        return pd.DataFrame()

    # Fetch VIX
    try:
        vix_data = _download_with_retry("^VIX")
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        vix_close = (
            vix_data["Close"].rename("VIX_Close")
            if not vix_data.empty
            else pd.Series(dtype=float, name="VIX_Close")
        )
    except Exception as exc:
        log.warning(f"VIX fetch failed: {exc}")
        vix_close = pd.Series(dtype=float, name="VIX_Close")

    # Fetch SPY
    try:
        spy_data = _download_with_retry("SPY")
        if isinstance(spy_data.columns, pd.MultiIndex):
            spy_data.columns = spy_data.columns.get_level_values(0)
        spy_close = (
            spy_data["Close"].rename("SPY_Close")
            if not spy_data.empty
            else pd.Series(dtype=float, name="SPY_Close")
        )
    except Exception as exc:
        log.warning(f"SPY fetch failed: {exc}")
        spy_close = pd.Series(dtype=float, name="SPY_Close")

    if vix_close.empty and spy_close.empty:
        log.warning(
            "Both VIX and SPY data unavailable — "
            "regime filtering disabled"
        )
        return pd.DataFrame()

    # Build combined DataFrame
    regime = pd.DataFrame(index=vix_close.index.union(spy_close.index))
    regime["VIX_Close"] = vix_close
    regime["SPY_Close"] = spy_close

    # SPY 200-day SMA
    regime["SPY_SMA_200"] = (
        regime["SPY_Close"]
        .rolling(window=200, min_periods=150)
        .mean()
    )

    # Fear signals
    regime["SPY_Below_SMA200"] = (
        regime["SPY_Close"] < regime["SPY_SMA_200"]
    )
    regime["VIX_Elevated"] = regime["VIX_Close"] > vix_fear_threshold

    # Composite: either signal triggers fear regime
    # Default to True (fear) where data is missing so strategies
    # run unfiltered rather than being silently blocked
    regime["Regime_Fear"] = (
        regime["VIX_Elevated"].fillna(True)
        | regime["SPY_Below_SMA200"].fillna(True)
    )

    # Drop rows where we have no data at all
    regime = regime.dropna(
        subset=["VIX_Close", "SPY_Close"], how="all"
    )

    log.info(
        f"  Regime data: {len(regime)} days, "
        f"fear days: {regime['Regime_Fear'].sum()}"
    )

    return regime


def enrich_with_regime(
    df: pd.DataFrame,
    regime_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add regime columns to a ticker's DataFrame by date-join.

    Parameters
    ----------
    df : pd.DataFrame
        Ticker OHLCV DataFrame (DatetimeIndex).
    regime_df : pd.DataFrame or None
        Pre-fetched regime data from ``fetch_regime_data()``.
        If None or empty, adds ``Regime_Fear = True`` everywhere
        (strategies run unfiltered).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added regime columns.
    """
    if regime_df is None or regime_df.empty:
        # No regime data → default to fear (unfiltered)
        df["Regime_Fear"] = True
        df["VIX_Close"] = np.nan
        df["SPY_Below_SMA200"] = False
        df["VIX_Elevated"] = False
        return df

    # Normalise both indices to date-only for joining
    df_dates = df.index.normalize()
    regime_dates = regime_df.index.normalize()

    # Reindex regime to ticker dates using forward-fill
    # (weekends/holidays get the last trading day's regime)
    regime_aligned = (
        regime_df.set_index(regime_dates)
        .reindex(df_dates)
        .ffill()
    )
    regime_aligned.index = df.index  # restore original index

    # Add columns
    df["Regime_Fear"] = regime_aligned["Regime_Fear"].fillna(True)
    df["VIX_Close"] = regime_aligned["VIX_Close"]
    df["SPY_Below_SMA200"] = regime_aligned[
        "SPY_Below_SMA200"
    ].fillna(False)
    df["VIX_Elevated"] = regime_aligned["VIX_Elevated"].fillna(False)

    return df


def clear_cache() -> None:
    """Clear the module-level regime cache."""
    global _regime_cache
    _regime_cache = None
