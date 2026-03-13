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
    ema,
    macd,
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

    # Statistical
    df["ZScore_20"] = zscore(c, 20)

    # Volume
    df["Vol_SMA_20"] = sma(v, 20)
    df["Vol_Ratio"] = v / df["Vol_SMA_20"]
    df["VWAP"] = vwap(h, l, c, v)

    # Price features
    df["Daily_Return"] = c.pct_change()
    df["Volatility_20"] = (
        df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    )

    return df
