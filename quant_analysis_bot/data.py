"""Data fetching and enrichment with technical indicators."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "yfinance is required. Install with: pip install yfinance"
    ) from exc

from quant_analysis_bot.indicators import (
    adx,
    atr,
    bollinger_bands,
    donchian_channels,
    ema,
    macd,
    on_balance_volume,
    rate_of_change,
    rsi,
    sma,
    stochastic,
    vwap,
    zscore,
)

log = logging.getLogger(__name__)


def fetch_data(
    ticker: str, lookback_days: int, cache_dir: str
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance with local caching."""
    os.makedirs(cache_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    cache_file = os.path.join(
        cache_dir, f"{ticker}_{date_str}.parquet"
    )

    if os.path.exists(cache_file):
        log.info(f"  Loading cached data for {ticker}")
        return pd.read_parquet(cache_file)

    log.info(f"  Downloading data for {ticker}...")
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 60)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_parquet(cache_file)
    return df


def batch_fetch_data(
    tickers: list[str],
    lookback_days: int,
    cache_dir: str,
    *,
    _skip_prefetch: bool = False,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV data for multiple tickers in one call.

    Uses ``yf.download(tickers, threads=True)`` which fetches all
    tickers concurrently via Yahoo's batch API — dramatically faster
    than one-at-a-time sequential downloads.

    Returns a dict mapping ticker → DataFrame.  Tickers that are
    already cached on disk are loaded from cache (no download).
    """
    os.makedirs(cache_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    results: dict[str, pd.DataFrame] = {}
    need_download: list[str] = []

    # 1. Load cached tickers, collect uncached ones
    for ticker in tickers:
        cache_file = os.path.join(
            cache_dir, f"{ticker}_{date_str}.parquet"
        )
        if os.path.exists(cache_file):
            results[ticker] = pd.read_parquet(cache_file)
        else:
            need_download.append(ticker)

    if not need_download:
        log.info(f"All {len(tickers)} tickers loaded from cache")
        return results

    # Check prefetch cache from previous trading day
    if not _skip_prefetch:
        from quant_analysis_bot.prefetch import (
            _et_now,
            validate_prefetch_cache,
        )

        prefetch_result = validate_prefetch_cache(
            cache_dir, _et_now().date()
        )
        if prefetch_result is not None:
            prev_date_str, valid_tickers = prefetch_result
            valid_set = set(valid_tickers)
            still_need: list[str] = []
            for ticker in need_download:
                if ticker not in valid_set:
                    still_need.append(ticker)
                    continue
                prefetch_file = os.path.join(
                    cache_dir,
                    f"{ticker}_{prev_date_str}.parquet",
                )
                if not os.path.exists(prefetch_file):
                    still_need.append(ticker)
                    continue
                try:
                    results[ticker] = pd.read_parquet(
                        prefetch_file
                    )
                except Exception as exc:
                    log.debug(
                        f"Prefetch read failed for {ticker}: "
                        f"{exc}"
                    )
                    still_need.append(ticker)

            loaded_from_prefetch = (
                len(need_download) - len(still_need)
            )
            if loaded_from_prefetch:
                log.info(
                    f"Loaded {loaded_from_prefetch} tickers "
                    f"from prefetch cache"
                )
            need_download = still_need

    if not need_download:
        log.info(
            f"All {len(tickers)} tickers loaded from "
            f"cache + prefetch"
        )
        return results

    log.info(
        f"Batch downloading {len(need_download)} tickers "
        f"({len(results)} from cache)..."
    )

    # 2. Single batch download for all uncached tickers
    import time as _time

    end = datetime.now()
    start = end - timedelta(days=lookback_days + 60)

    raw = pd.DataFrame()
    for attempt in range(3):
        try:
            raw = yf.download(
                need_download,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
                threads=True,
                group_by="ticker",
            )
            if raw is not None and not raw.empty:
                break
        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = "RateLimit" in exc_str or (
                "Too Many Requests" in exc_str
            )
            if is_rate_limit:
                wait = 30 * (attempt + 1)
                log.warning(
                    f"Batch download rate limited, "
                    f"waiting {wait}s before retry "
                    f"({attempt + 1}/3)..."
                )
            else:
                wait = 5 * (attempt + 1)
                log.warning(
                    f"Batch download failed: {exc}, "
                    f"retrying in {wait}s..."
                )
            if attempt < 2:
                _time.sleep(wait)

    if raw is None or raw.empty:
        log.warning("Batch download returned empty DataFrame")
        return results

    # 3. Split the multi-ticker result into per-ticker DataFrames
    if len(need_download) == 1:
        # Single ticker: yf.download returns flat columns
        ticker = need_download[0]
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            cache_file = os.path.join(
                cache_dir, f"{ticker}_{date_str}.parquet"
            )
            df.to_parquet(cache_file)
            results[ticker] = df
    else:
        # Multiple tickers: columns are MultiIndex (ticker, field)
        for ticker in need_download:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw[ticker].copy()
                else:
                    df = raw.copy()

                # Drop rows that are all-NaN (ticker had no data)
                df = df.dropna(how="all")

                if df.empty:
                    log.debug(f"No data for {ticker} in batch")
                    continue

                # Flatten if still multi-level
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                cache_file = os.path.join(
                    cache_dir, f"{ticker}_{date_str}.parquet"
                )
                df.to_parquet(cache_file)
                results[ticker] = df
            except (KeyError, Exception) as e:
                log.debug(f"Failed to extract {ticker}: {e}")

    log.info(
        f"Batch download complete: "
        f"{len(results)}/{len(tickers)} tickers ready"
    )
    return results


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Moving averages
    df["SMA_10"] = sma(c, 10)
    df["SMA_20"] = sma(c, 20)
    df["SMA_50"] = sma(c, 50)
    df["SMA_200"] = sma(c, 200)
    df["EMA_9"] = ema(c, 9)
    df["EMA_21"] = ema(c, 21)

    # Momentum
    df["RSI_14"] = rsi(c, 14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(c)
    df["ROC_10"] = rate_of_change(c, 10)
    df["Stoch_K"], df["Stoch_D"] = stochastic(h, l, c)

    # Volatility
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = bollinger_bands(c)
    df["ATR_14"] = atr(h, l, c, 14)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]

    # Trend
    df["ADX_14"] = adx(h, l, c, 14)

    # Breakout channels
    df["Donchian_Upper_20"], df["Donchian_Lower_20"] = (
        donchian_channels(h, l, 20)
    )
    df["Donchian_Upper_55"], df["Donchian_Lower_55"] = (
        donchian_channels(h, l, 55)
    )

    # 52-week high (for nearness-to-high feature)
    df["High_52w"] = h.rolling(window=252, min_periods=126).max()
    df["Nearness_52w_High"] = c / df["High_52w"].replace(0, 1e-10)

    # Statistical
    df["ZScore_20"] = zscore(c, 20)

    # Volume
    df["Vol_SMA_20"] = sma(v, 20)
    df["Vol_Ratio"] = v / df["Vol_SMA_20"]
    df["VWAP"] = vwap(h, l, c, v)
    df["OBV"] = on_balance_volume(c, v)

    # Price features
    df["Daily_Return"] = c.pct_change()
    df["Volatility_20"] = (
        df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    )

    return df
