#!/usr/bin/env python3
"""
Quant Analysis Bot -- Daily Stock Signal Generator
==================================================
Automatically discovers which trading strategy works best for each stock
via multi-timeframe backtesting, then generates today's signal (BUY / SELL / HOLD).

Features:
  - 11 strategies tested per stock
  - Multi-timeframe analysis (3-month, 6-month, 12-month)
  - Full trade log with every entry/exit
  - Annualized metrics for fair comparison
  - Recent performance weighted higher in scoring

Usage:
    python quant_bot.py                  # Run with default config
    python quant_bot.py --config my.json # Run with custom config
    python quant_bot.py --all-stocks     # Run on top 1000 US stocks by market cap

Output:
    signals/signals_YYYY-MM-DD.csv
    signals/signals_YYYY-MM-DD.json
    reports/backtest_report_YYYY-MM-DD.txt
    trade_logs/trades_TICKER_YYYY-MM-DD.csv
"""

import json
import csv
import os
import sys
import logging
import argparse
import time
import shutil
from contextlib import nullcontext
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────
#  PROGRESS BAR
# ──────────────────────────────────────────────────────────────────────

class ProgressBar:
    """
    A rich terminal progress bar with ETA, speed, and percentage.
    Falls back to tqdm if available, otherwise uses a custom implementation.
    """

    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self._term_width = shutil.get_terminal_size((80, 20)).columns
        self._use_tqdm = HAS_TQDM and total > 0
        self._tqdm_bar = None

        if self._use_tqdm:
            self._tqdm_bar = tqdm(
                total=total, desc=desc, unit=unit,
                bar_format="{l_bar}{bar:30}{r_bar}",
                ncols=min(self._term_width, 120),
            )
        elif total > 0:
            self._print_bar()

    def update(self, n: int = 1, suffix: str = ""):
        """Advance the progress bar by n steps."""
        self.current = min(self.current + n, self.total)
        if self._use_tqdm:
            self._tqdm_bar.update(n)
            if suffix:
                self._tqdm_bar.set_postfix_str(suffix)
        else:
            self._print_bar(suffix)

    def _print_bar(self, suffix: str = ""):
        if self.total == 0:
            return
        elapsed = time.time() - self.start_time
        pct = self.current / self.total
        speed = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / speed if speed > 0 else 0

        # Format timing
        elapsed_str = self._fmt_time(elapsed)
        eta_str = self._fmt_time(eta)

        # Build bar
        bar_width = max(20, min(40, self._term_width - 60))
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build line
        desc_part = f"{self.desc}: " if self.desc else ""
        line = (f"\r{desc_part}{pct:>6.1%} |{bar}| "
                f"{self.current}/{self.total} "
                f"[{elapsed_str}|{eta_str}, {speed:.1f} {self.unit}/s]")
        if suffix:
            line += f" {suffix}"

        # Pad to terminal width to clear previous line
        line = line[:self._term_width].ljust(self._term_width)
        sys.stderr.write(line)
        sys.stderr.flush()

    def _fmt_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.0f}m{seconds % 60:02.0f}s"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"

    def close(self):
        if self._use_tqdm and self._tqdm_bar:
            self._tqdm_bar.close()
        elif self.total > 0:
            self._print_bar()
            sys.stderr.write("\n")
            sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ──────────────────────────────────────────────────────────────────────
#  STOCK UNIVERSE -- fetch top US stocks by market cap
# ──────────────────────────────────────────────────────────────────────

_UNIVERSE_CACHE_DIR = "cache"

# Bundled ticker list location (ships with the bot)
_BUNDLED_TICKERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "us_tickers.json")


def _fetch_wiki_html(url: str) -> str:
    """Fetch HTML from Wikipedia with a proper User-Agent to avoid 403 errors."""
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "QuantAnalysisBot/1.0 "
                "(https://github.com/user/quant-bot; educational use) "
                "Python/pandas"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _fetch_wiki_tickers(name: str, url: str) -> List[str]:
    """Fetch tickers from a Wikipedia S&P index page with proper headers."""
    try:
        from io import StringIO
        html = _fetch_wiki_html(url)
        tables = pd.read_html(StringIO(html))
        df = tables[0]
        # Find the symbol/ticker column
        ticker_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if col_str in ("symbol", "ticker", "ticker symbol"):
                ticker_col = col
                break
        if ticker_col is None:
            ticker_col = df.columns[0]
        tickers = (
            df[ticker_col]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .tolist()
        )
        # Filter out non-ticker strings
        tickers = [t for t in tickers if t and 1 <= len(t) <= 5 and t.replace("-", "").isalpha()]
        log.info(f"  {name}: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(f"  Failed to fetch {name} from Wikipedia: {e}")
        return []


def _load_bundled_tickers() -> List[str]:
    """Load the bundled ticker list shipped with the bot."""
    if os.path.exists(_BUNDLED_TICKERS_FILE):
        with open(_BUNDLED_TICKERS_FILE, "r") as f:
            tickers = json.load(f)
        log.info(f"  Loaded {len(tickers)} tickers from bundled list")
        return tickers
    return []


def _get_market_caps(tickers: List[str], batch_size: int = 50) -> Dict[str, float]:
    """
    Fetch market caps for a list of tickers using yfinance.
    Downloads in batches for efficiency.
    """
    market_caps = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    log.info(f"  Fetching market caps for {len(tickers)} tickers ({total_batches} batches)...")

    with ProgressBar(total=len(tickers), desc="Market caps", unit="stocks") as pbar:
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            batch_str = " ".join(batch)
            try:
                data = yf.Tickers(batch_str)
                for ticker in batch:
                    try:
                        info = data.tickers[ticker].info
                        mcap = info.get("marketCap", 0)
                        if mcap and mcap > 0:
                            market_caps[ticker] = mcap
                    except Exception:
                        pass
                    pbar.update(1, suffix=ticker)
            except Exception as e:
                log.warning(f"  Batch failed: {e}")
                pbar.update(len(batch))

    return market_caps


def fetch_top_us_stocks(n: int = 1000, cache_dir: str = _UNIVERSE_CACHE_DIR) -> List[str]:
    """
    Fetch the top N US stocks by market cap.

    Fallback chain:
      1. Check daily cache
      2. Try Wikipedia (S&P 500 + 400 + 600) with proper User-Agent headers
      3. Fall back to bundled us_tickers.json file
      4. Fetch market caps via yfinance, sort descending, return top N

    Returns:
        List of ticker symbols, ordered by market cap descending.
    """
    os.makedirs(cache_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    cache_file = os.path.join(cache_dir, f"universe_top{n}_{date_str}.json")

    # ── 1. Check daily cache ──────────────────────────────────────────
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = json.load(f)
        log.info(f"  Loaded {len(cached)} tickers from universe cache ({cache_file})")
        return cached[:n]

    log.info(f"  Building top {n} US stock universe...")

    # ── 2. Try Wikipedia with proper headers ──────────────────────────
    all_tickers = set()
    wiki_sources = [
        ("S&P 500",        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
        ("S&P MidCap 400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
        ("S&P SmallCap 600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
    ]

    for name, url in wiki_sources:
        tickers = _fetch_wiki_tickers(name, url)
        all_tickers.update(tickers)
        if len(all_tickers) >= n * 1.2:  # have enough, skip remaining
            break

    # ── 3. Fall back to bundled list if Wikipedia failed ──────────────
    if len(all_tickers) < 100:
        log.warning("  Wikipedia fetch returned too few tickers, using bundled list...")
        bundled = _load_bundled_tickers()
        all_tickers.update(bundled)

    if len(all_tickers) < 50:
        log.error("  Cannot build stock universe: no ticker source available.")
        log.error("  Make sure us_tickers.json exists or that you have internet access.")
        return []

    ticker_list = sorted(all_tickers)
    log.info(f"  Collected {len(ticker_list)} unique tickers")

    # ── 4. Fetch market caps and sort ─────────────────────────────────
    market_caps = _get_market_caps(ticker_list)
    log.info(f"  Successfully got market caps for {len(market_caps)} tickers")

    if len(market_caps) < 50:
        log.error(f"  Only got market caps for {len(market_caps)} stocks -- check your internet")
        return []

    sorted_tickers = sorted(market_caps.keys(), key=lambda t: market_caps[t], reverse=True)
    top_n = sorted_tickers[:n]

    # Cache for today
    with open(cache_file, "w") as f:
        json.dump(top_n, f)
    log.info(f"  Cached {len(top_n)} tickers to {cache_file}")

    if top_n:
        top5_str = ", ".join(f"{t} (${market_caps[t]/1e9:.0f}B)" for t in top_n[:5])
        log.info(f"  Top 5: {top5_str}")
        if len(top_n) >= n:
            bottom = top_n[-1]
            log.info(f"  #{n}: {bottom} (${market_caps[bottom]/1e9:.1f}B)")

    return top_n


# ──────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "tickers": ["AAPL", "GOOG", "NFLX", "GLD", "TEM", "ASTS", "WTI"],
    "lookback_days": 500,           # Historical data to download (extra buffer)
    "long_only": True,              # True = no shorting (SELL means exit to cash)
    "risk_profile": "moderate",     # conservative / moderate / aggressive
    "min_sharpe": 0.5,              # Minimum Sharpe to trust a strategy
    "position_size": 1.0,           # Fraction of capital per trade
    "transaction_cost_bps": 10,     # Cost per trade in basis points
    "output_dir": "signals",
    "report_dir": "reports",
    "trade_log_dir": "trade_logs",
    "data_cache_dir": "cache",
    # Multi-timeframe windows (trading days)
    "backtest_windows": {
        "3mo": 63,
        "6mo": 126,
        "12mo": 252,
    },
    # How much to weight each window in final scoring (recent = higher)
    "window_weights": {
        "3mo": 0.50,
        "6mo": 0.30,
        "12mo": 0.20,
    },
}

RISK_PROFILES = {
    "conservative": {"min_win_rate": 0.55, "max_trades_per_month": 4, "signal_threshold": 0.7},
    "moderate":     {"min_win_rate": 0.48, "max_trades_per_month": 8, "signal_threshold": 0.5},
    "aggressive":   {"min_win_rate": 0.40, "max_trades_per_month": 15, "signal_threshold": 0.3},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("quant_bot")


# ──────────────────────────────────────────────────────────────────────
#  TECHNICAL INDICATORS  (pure numpy/pandas -- no external TA library)
# ──────────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = mid + std_dev * rolling_std
    lower = mid - std_dev * rolling_std
    return upper, mid, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = sma(k, d_period)
    return k, d

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    return (series - mean) / std

def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period) * 100

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index -- measures trend strength."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_val = atr(high, low, close, period)
    plus_di = 100 * ema(plus_dm, period) / atr_val
    minus_di = 100 * ema(minus_dm, period) / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return ema(dx, period)


# ──────────────────────────────────────────────────────────────────────
#  DATA FETCHING
# ──────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, lookback_days: int, cache_dir: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance with local caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{datetime.now().strftime('%Y%m%d')}.parquet")

    if os.path.exists(cache_file):
        log.info(f"  Loading cached data for {ticker}")
        return pd.read_parquet(cache_file)

    log.info(f"  Downloading data for {ticker}...")
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 60)  # extra buffer
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present (yfinance sometimes returns these)
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
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

    return df


# ──────────────────────────────────────────────────────────────────────
#  STRATEGIES -- each returns a Series of signals: +1 (buy), -1 (sell), 0 (hold)
# ──────────────────────────────────────────────────────────────────────

class Strategy:
    """Base class for all strategies."""
    name: str = "base"
    description: str = ""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class SMA_Crossover(Strategy):
    name = "SMA Crossover (10/50)"
    description = "Buy when SMA10 crosses above SMA50, sell on cross below"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["SMA_10"] > df["SMA_50"]] = 1
        signals[df["SMA_10"] < df["SMA_50"]] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class EMA_Crossover(Strategy):
    name = "EMA Crossover (9/21)"
    description = "Buy when EMA9 crosses above EMA21, sell on cross below"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["EMA_9"] > df["EMA_21"]] = 1
        signals[df["EMA_9"] < df["EMA_21"]] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class RSI_MeanReversion(Strategy):
    name = "RSI Mean Reversion"
    description = "Buy when RSI<30 (oversold), sell when RSI>70 (overbought)"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["RSI_14"] < 30] = 1
        signals[df["RSI_14"] > 70] = -1
        return signals


class MACD_Strategy(Strategy):
    name = "MACD Crossover"
    description = "Buy on MACD bullish crossover, sell on bearish crossover"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["MACD_Hist"] > 0] = 1
        signals[df["MACD_Hist"] < 0] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class BollingerBand_Reversion(Strategy):
    name = "Bollinger Band Mean Reversion"
    description = "Buy at lower band, sell at upper band"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["Close"] < df["BB_Lower"]] = 1
        signals[df["Close"] > df["BB_Upper"]] = -1
        return signals


class Momentum_ROC(Strategy):
    name = "Momentum (Rate of Change)"
    description = "Buy on strong positive momentum, sell on strong negative"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        threshold = df["ROC_10"].rolling(50).std()
        signals[df["ROC_10"] > threshold] = 1
        signals[df["ROC_10"] < -threshold] = -1
        return signals


class ZScore_MeanReversion(Strategy):
    name = "Z-Score Mean Reversion"
    description = "Buy when price is >1.5 std below mean, sell >1.5 std above"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        signals[df["ZScore_20"] < -1.5] = 1
        signals[df["ZScore_20"] > 1.5] = -1
        return signals


class Stochastic_Strategy(Strategy):
    name = "Stochastic Oscillator"
    description = "Buy when %K crosses above %D in oversold zone, sell in overbought"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        buy_cond = (df["Stoch_K"] > df["Stoch_D"]) & (df["Stoch_K"] < 25)
        sell_cond = (df["Stoch_K"] < df["Stoch_D"]) & (df["Stoch_K"] > 75)
        signals[buy_cond] = 1
        signals[sell_cond] = -1
        return signals


class VWAP_Strategy(Strategy):
    name = "VWAP Trend"
    description = "Buy when price crosses above VWAP with volume, sell below"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        vol_confirm = df["Vol_Ratio"] > 1.2
        signals[(df["Close"] > df["VWAP"]) & vol_confirm] = 1
        signals[(df["Close"] < df["VWAP"]) & vol_confirm] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class TrendFollowing_ADX(Strategy):
    name = "ADX Trend Following"
    description = "Follow trend when ADX>25 (strong trend), use MA direction"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        strong_trend = df["ADX_14"] > 25
        signals[strong_trend & (df["EMA_9"] > df["SMA_50"])] = 1
        signals[strong_trend & (df["EMA_9"] < df["SMA_50"])] = -1
        return signals.diff().clip(-1, 1).fillna(0)


class CompositeScore(Strategy):
    name = "Composite Multi-Indicator"
    description = "Weighted score across RSI, MACD, BB, and trend indicators"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        score = pd.Series(0.0, index=df.index)

        # RSI component
        score += np.where(df["RSI_14"] < 35, 1, np.where(df["RSI_14"] > 65, -1, 0)) * 0.2

        # MACD component
        score += np.where(df["MACD_Hist"] > 0, 1, np.where(df["MACD_Hist"] < 0, -1, 0)) * 0.2

        # Bollinger component
        bb_pos = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
        score += np.where(bb_pos < 0.2, 1, np.where(bb_pos > 0.8, -1, 0)) * 0.2

        # Trend component (EMA)
        score += np.where(df["EMA_9"] > df["EMA_21"], 1, -1) * 0.2

        # Volume confirmation
        score += np.where(df["Vol_Ratio"] > 1.3, 0.2, 0)

        signals = pd.Series(0, index=df.index)
        signals[score > 0.4] = 1
        signals[score < -0.4] = -1
        return signals


ALL_STRATEGIES = [
    SMA_Crossover(),
    EMA_Crossover(),
    RSI_MeanReversion(),
    MACD_Strategy(),
    BollingerBand_Reversion(),
    Momentum_ROC(),
    ZScore_MeanReversion(),
    Stochastic_Strategy(),
    VWAP_Strategy(),
    TrendFollowing_ADX(),
    CompositeScore(),
]


# ──────────────────────────────────────────────────────────────────────
#  TRADE LOG -- records every individual trade
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """A single completed round-trip trade."""
    trade_num: int
    ticker: str
    strategy: str
    timeframe: str          # "3mo", "6mo", "12mo"
    direction: str          # "LONG" or "SHORT"
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    holding_days: int
    return_pct: float
    outcome: str            # "WIN" or "LOSS"


# ──────────────────────────────────────────────────────────────────────
#  BACKTESTING ENGINE
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    strategy_name: str
    ticker: str
    timeframe: str = ""             # "3mo", "6mo", "12mo"
    backtest_start: str = ""
    backtest_end: str = ""
    trading_days: int = 0
    # Raw returns
    total_return_pct: float = 0.0
    buy_hold_return_pct: float = 0.0
    excess_return_pct: float = 0.0
    # Annualized returns
    annual_return_pct: float = 0.0
    annual_bh_return_pct: float = 0.0
    annual_excess_pct: float = 0.0
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    # Trade stats
    win_rate: float = 0.0
    total_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0


def run_backtest(df: pd.DataFrame, signals: pd.Series, ticker: str,
                 strategy_name: str, timeframe: str,
                 cost_bps: float = 10,
                 long_only: bool = True) -> Tuple[BacktestResult, List[TradeRecord]]:
    """
    Run a full backtest on a signal series.
    Returns performance metrics AND a list of every individual trade.

    If long_only=True, SELL signals close a long position and go to cash
    (no short positions opened). If False, SELL opens a short position.
    """
    result = BacktestResult(strategy_name=strategy_name, ticker=ticker, timeframe=timeframe)
    trade_log: List[TradeRecord] = []

    df = df.copy()
    df["Signal"] = signals
    df = df.dropna(subset=["Close"])

    if len(df) < 30:
        return result, trade_log

    dates = df.index
    close = df["Close"].values
    sig = df["Signal"].values
    cost = cost_bps / 10000

    result.backtest_start = str(dates[0].date())
    result.backtest_end = str(dates[-1].date())
    result.trading_days = len(df)

    # Track positions and returns
    position = 0        # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    trades_raw = []     # raw trade dicts for metric calculation
    daily_returns = []
    equity = [1.0]
    trade_count = 0

    for i in range(1, len(close)):
        daily_ret = 0.0

        if position != 0:
            daily_ret = position * (close[i] / close[i - 1] - 1)

        # Process signals
        if sig[i] == 1 and position <= 0:
            if position == -1 and not long_only:
                # Close short (only in long+short mode)
                trade_ret = entry_price / close[i] - 1 - cost
                holding = (dates[i] - dates[entry_idx]).days
                trade_count += 1
                trade_log.append(TradeRecord(
                    trade_num=trade_count, ticker=ticker, strategy=strategy_name,
                    timeframe=timeframe, direction="SHORT",
                    entry_date=str(dates[entry_idx].date()),
                    entry_price=round(close[entry_idx], 2),
                    exit_date=str(dates[i].date()),
                    exit_price=round(close[i], 2),
                    holding_days=holding,
                    return_pct=round(trade_ret * 100, 2),
                    outcome="WIN" if trade_ret > 0 else "LOSS",
                ))
                trades_raw.append({"return": trade_ret, "holding_days": holding})
            # Open long
            position = 1
            entry_price = close[i]
            entry_idx = i
            daily_ret -= cost

        elif sig[i] == -1 and position >= 0:
            if position == 1:
                # Close long
                trade_ret = close[i] / entry_price - 1 - cost
                holding = (dates[i] - dates[entry_idx]).days
                trade_count += 1
                trade_log.append(TradeRecord(
                    trade_num=trade_count, ticker=ticker, strategy=strategy_name,
                    timeframe=timeframe, direction="LONG",
                    entry_date=str(dates[entry_idx].date()),
                    entry_price=round(close[entry_idx], 2),
                    exit_date=str(dates[i].date()),
                    exit_price=round(close[i], 2),
                    holding_days=holding,
                    return_pct=round(trade_ret * 100, 2),
                    outcome="WIN" if trade_ret > 0 else "LOSS",
                ))
                trades_raw.append({"return": trade_ret, "holding_days": holding})

            if long_only:
                # Exit to cash -- no short position opened
                position = 0
                entry_price = 0.0
                daily_ret -= cost if position == 1 else 0  # cost only if we closed
            else:
                # Open short
                position = -1
                entry_price = close[i]
                entry_idx = i
                daily_ret -= cost

        daily_returns.append(daily_ret)
        equity.append(equity[-1] * (1 + daily_ret))

    equity = np.array(equity)
    daily_returns = np.array(daily_returns)

    if len(daily_returns) == 0:
        return result, trade_log

    n_days = len(daily_returns)

    # -- Core metrics (raw) --
    result.total_return_pct = round((equity[-1] / equity[0] - 1) * 100, 2)
    result.buy_hold_return_pct = round((close[-1] / close[0] - 1) * 100, 2)
    result.excess_return_pct = round(result.total_return_pct - result.buy_hold_return_pct, 2)

    # -- Annualized returns --
    if n_days > 0:
        ann_factor = 252 / n_days
        # Compound annualization: (1+r)^(252/n) - 1
        total_r = equity[-1] / equity[0]
        bh_r = close[-1] / close[0]
        result.annual_return_pct = round((total_r ** ann_factor - 1) * 100, 2)
        result.annual_bh_return_pct = round((bh_r ** ann_factor - 1) * 100, 2)
        result.annual_excess_pct = round(result.annual_return_pct - result.annual_bh_return_pct, 2)

    # -- Sharpe Ratio (annualized) --
    if daily_returns.std() > 0:
        result.sharpe_ratio = round(np.sqrt(252) * daily_returns.mean() / daily_returns.std(), 2)

    # -- Sortino Ratio --
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        result.sortino_ratio = round(np.sqrt(252) * daily_returns.mean() / downside.std(), 2)

    # -- Max Drawdown --
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    result.max_drawdown_pct = round(drawdown.min() * 100, 2)

    # -- Calmar Ratio --
    if result.max_drawdown_pct != 0:
        result.calmar_ratio = round(result.annual_return_pct / abs(result.max_drawdown_pct), 2)

    # -- Trade stats --
    result.total_trades = len(trades_raw)
    if trades_raw:
        wins = [t["return"] for t in trades_raw if t["return"] > 0]
        losses = [t["return"] for t in trades_raw if t["return"] <= 0]
        result.win_rate = round(len(wins) / len(trades_raw), 3)
        result.avg_win_pct = round(np.mean(wins) * 100, 2) if wins else 0.0
        result.avg_loss_pct = round(np.mean(losses) * 100, 2) if losses else 0.0
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0.001
        result.profit_factor = round(total_wins / total_losses, 2)
        result.avg_holding_days = round(np.mean([t["holding_days"] for t in trades_raw]), 1)

    return result, trade_log


# ──────────────────────────────────────────────────────────────────────
#  STRATEGY SELECTOR -- multi-timeframe scoring
# ──────────────────────────────────────────────────────────────────────

def score_single_window(result: BacktestResult, risk_config: dict) -> float:
    """Score a strategy on a single timeframe window."""
    score = 0.0

    # Sharpe ratio (most important) -- 35% weight
    score += min(result.sharpe_ratio, 3.0) * 35

    # Annualized excess return over buy & hold -- 20% weight
    score += np.clip(result.annual_excess_pct, -100, 100) * 0.2

    # Win rate -- 15% weight
    score += (result.win_rate - 0.5) * 150

    # Profit factor -- 15% weight
    score += min(result.profit_factor, 3.0) * 15

    # Drawdown penalty -- 15% weight
    score += result.max_drawdown_pct * 0.5  # negative, so penalizes

    # Penalize too few trades (unreliable)
    if result.total_trades < 3:
        score *= 0.4
    elif result.total_trades < 5:
        score *= 0.7

    # Penalize if win rate below risk threshold
    if result.win_rate < risk_config["min_win_rate"]:
        score *= 0.7

    return round(score, 2)


def select_best_strategy(df: pd.DataFrame, ticker: str, config: dict):
    """
    Test all strategies across multiple timeframes.
    Returns: (best_strategy, best_combined_result, per_window_results, all_trade_logs)

    per_window_results = {
        "3mo":  [(strategy, result, score), ...],
        "6mo":  [...],
        "12mo": [...],
    }
    """
    risk_config = RISK_PROFILES[config["risk_profile"]]
    windows = config["backtest_windows"]
    weights = config["window_weights"]

    # Accumulate: strategy_name -> {window -> (strategy, result, score)}
    strategy_windows: Dict[str, Dict[str, Tuple]] = {}
    per_window_results: Dict[str, list] = {}
    all_trade_logs: List[TradeRecord] = []

    for window_name, n_days in windows.items():
        window_df = df.tail(n_days).copy()
        if len(window_df) < 30:
            log.warning(f"  Skipping {window_name} window for {ticker}: only {len(window_df)} days")
            continue

        window_results = []

        for strategy in ALL_STRATEGIES:
            try:
                signals = strategy.generate_signals(window_df)
                result, trade_log = run_backtest(
                    window_df, signals, ticker, strategy.name,
                    window_name, config["transaction_cost_bps"],
                    long_only=config.get("long_only", True)
                )
                score = score_single_window(result, risk_config)
                result.score = score
                window_results.append((strategy, result, score))
                all_trade_logs.extend(trade_log)

                if strategy.name not in strategy_windows:
                    strategy_windows[strategy.name] = {}
                strategy_windows[strategy.name][window_name] = (strategy, result, score)

            except Exception as e:
                log.warning(f"  {strategy.name} failed on {window_name} for {ticker}: {e}")

        per_window_results[window_name] = sorted(window_results, key=lambda x: x[2], reverse=True)

    # Compute weighted composite score across timeframes
    composite_scores: Dict[str, float] = {}
    strategy_objs: Dict[str, Strategy] = {}

    for strat_name, window_data in strategy_windows.items():
        weighted_score = 0.0
        total_weight = 0.0
        for window_name, weight in weights.items():
            if window_name in window_data:
                _, _, score = window_data[window_name]
                weighted_score += score * weight
                total_weight += weight
        if total_weight > 0:
            composite_scores[strat_name] = round(weighted_score / total_weight, 2)
            # Grab the strategy object from any window
            strategy_objs[strat_name] = list(window_data.values())[0][0]

    if not composite_scores:
        raise ValueError(f"All strategies failed for {ticker}")

    # Best strategy by composite score
    best_name = max(composite_scores, key=composite_scores.get)
    best_strategy = strategy_objs[best_name]

    # Use the longest-window result as the "primary" result for the signal
    longest_window = max(windows.keys(), key=lambda w: windows[w])
    if longest_window in strategy_windows.get(best_name, {}):
        best_result = strategy_windows[best_name][longest_window][1]
    else:
        # Fallback: use whatever window we have
        best_result = list(strategy_windows[best_name].values())[0][1]

    best_result.composite_score = composite_scores[best_name]

    return best_strategy, best_result, per_window_results, composite_scores, all_trade_logs


# ──────────────────────────────────────────────────────────────────────
#  SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DailySignal:
    # ── Identity ──────────────────────────────────────────────────────────
    generated_at: str        # ISO datetime e.g. "2026-03-13T09:31:00"
    date: str                # YYYY-MM-DD  (trading date)
    ticker: str

    # ── Signal ────────────────────────────────────────────────────────────
    signal: str              # BUY / EXIT / HOLD / SELL/SHORT / ERROR
    signal_raw: int          # 1=BUY  -1=SHORT/EXIT  0=HOLD
    strategy: str
    confidence: str          # HIGH / MEDIUM / LOW  (human label)
    confidence_score: int    # 0-6  (raw points behind label)
    composite_score: float   # weighted multi-timeframe backtest score

    # ── Execution ─────────────────────────────────────────────────────────
    current_price: float
    stop_loss_pct: float     # e.g. 5.0 = stop 5% below entry
    stop_loss_price: float   # absolute stop price
    take_profit_pct: float   # e.g. 10.0 = target 10% above entry
    take_profit_price: float # absolute take-profit price
    suggested_position_size_pct: float  # % of portfolio (half-Kelly capped)
    signal_expires: str      # YYYY-MM-DD  (re-evaluate after this date)

    # ── Backtest quality ──────────────────────────────────────────────────
    sharpe: float
    sortino: float
    win_rate: float          # 0-100 scale
    profit_factor: float
    annual_return_pct: float
    annual_excess_pct: float
    max_drawdown_pct: float
    avg_holding_days: float
    total_trades: int
    backtest_period: str     # "2025-03-11 to 2026-03-11 (252 days)"

    # ── Market context ────────────────────────────────────────────────────
    rsi: float
    vol_20: float            # annualised 20-day volatility (float, not label)
    sma_50: float
    sma_200: float
    trend: str               # BULLISH / BEARISH / NEUTRAL
    volatility: str          # LOW / MEDIUM / HIGH  (human label)
    notes: str


def generate_daily_signal(df: pd.DataFrame, ticker: str,
                          strategy: Strategy, result: BacktestResult,
                          config: dict) -> DailySignal:
    """Generate today's actionable signal for a stock."""
    risk_config = RISK_PROFILES[config["risk_profile"]]

    # Generate signal on full data
    signals = strategy.generate_signals(df)
    latest_signal_val = signals.iloc[-1] if len(signals) > 0 else 0

    # Map signal
    long_only = config.get("long_only", True)
    if latest_signal_val > 0:
        signal = "BUY"
        signal_val = 1
    elif latest_signal_val < 0:
        signal = "EXIT" if long_only else "SELL/SHORT"
        signal_val = -1
    else:
        signal = "HOLD"
        signal_val = 0

    # Confidence assessment
    confidence_score = 0
    if result.sharpe_ratio > 1.0:
        confidence_score += 2
    elif result.sharpe_ratio > 0.5:
        confidence_score += 1
    if result.win_rate > 0.55:
        confidence_score += 1
    if result.profit_factor > 1.5:
        confidence_score += 1
    if result.total_trades >= 15:
        confidence_score += 1
    # Bonus: consistent across timeframes (composite score is high)
    if hasattr(result, 'composite_score') and result.composite_score > 50:
        confidence_score += 1

    if confidence_score >= 4:
        confidence = "HIGH"
    elif confidence_score >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # If confidence is too low under risk profile, downgrade to HOLD
    if confidence == "LOW" and signal not in ("HOLD",):
        if result.sharpe_ratio < config["min_sharpe"]:
            signal = "HOLD"

    # Determine trend
    latest = df.iloc[-1]
    if latest.get("SMA_50", 0) > latest.get("SMA_200", 0):
        trend = "BULLISH"
    elif latest.get("SMA_50", 0) < latest.get("SMA_200", 0):
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    # Determine volatility regime
    vol_20 = latest.get("Volatility_20", 0)
    if vol_20 > 0.4:
        volatility = "HIGH"
    elif vol_20 > 0.2:
        volatility = "MEDIUM"
    else:
        volatility = "LOW"

    # Build notes
    notes_parts = []
    rsi_val = latest.get("RSI_14", 50)
    if rsi_val > 70:
        notes_parts.append("RSI overbought")
    elif rsi_val < 30:
        notes_parts.append("RSI oversold")

    if latest.get("Vol_Ratio", 1) > 1.5:
        notes_parts.append("High volume")
    if abs(latest.get("ZScore_20", 0)) > 2:
        notes_parts.append("Price at statistical extreme")

    adx_val = latest.get("ADX_14", 0)
    if adx_val > 30:
        notes_parts.append(f"Strong trend (ADX={adx_val:.0f})")

    backtest_period = f"{result.backtest_start} to {result.backtest_end} ({result.trading_days} days)"

    now = datetime.now()
    current_price = round(float(latest["Close"]), 2)

    # ── Stop loss: ATR-based where available, else drawdown-derived ───────
    atr_val = float(latest.get("ATR_14", 0) or 0)
    if atr_val > 0 and current_price > 0:
        # 1.5× ATR stop
        stop_loss_pct = round(min(max((atr_val * 1.5 / current_price) * 100, 1.5), 12.0), 2)
    else:
        # Fall back to half the max drawdown observed in backtest, capped 2-10%
        stop_loss_pct = round(min(max(abs(result.max_drawdown_pct) * 0.4, 2.0), 10.0), 2)

    if signal_val > 0:   # BUY — stop below entry
        stop_loss_price = round(current_price * (1 - stop_loss_pct / 100), 2)
    elif signal_val < 0: # SHORT — stop above entry
        stop_loss_price = round(current_price * (1 + stop_loss_pct / 100), 2)
    else:
        stop_loss_price = 0.0

    # ── Take profit: 2:1 reward/risk ──────────────────────────────────────
    take_profit_pct = round(stop_loss_pct * 2.0, 2)
    if signal_val > 0:
        take_profit_price = round(current_price * (1 + take_profit_pct / 100), 2)
    elif signal_val < 0:
        take_profit_price = round(current_price * (1 - take_profit_pct / 100), 2)
    else:
        take_profit_price = 0.0

    # ── Position sizing: full Kelly, floored and capped by confidence ──────
    win_rate_f = result.win_rate  # 0-1
    pf = max(result.profit_factor, 0.01)
    # Kelly f* = (win_rate * pf - loss_rate) / pf
    loss_rate = 1 - win_rate_f
    kelly_f = max((win_rate_f * pf - loss_rate) / pf, 0.0)
    kelly_pct = kelly_f * 100  # full Kelly as a percentage
    # Only size if the strategy has a genuine edge (Kelly > 0).
    # Floor: ensure every position with an edge is meaningful (min 3%).
    # Cap: limit by confidence level to prevent over-concentration.
    if kelly_f > 0:
        min_size = 3.0
        max_size = {"HIGH": 20.0, "MEDIUM": 12.0, "LOW": 6.0}.get(confidence, 5.0)
        suggested_position_size_pct = round(
            min(max(kelly_pct, min_size), max_size), 2
        )
    else:
        suggested_position_size_pct = 0.0
    if signal == "HOLD":
        suggested_position_size_pct = 0.0

    # ── Signal expiry ─────────────────────────────────────────────────────
    hold_days = max(int(result.avg_holding_days), 1)
    from datetime import timedelta
    signal_expires = (now + timedelta(days=hold_days)).strftime("%Y-%m-%d")

    composite_score_val = getattr(result, "composite_score", 0.0) or 0.0

    return DailySignal(
        generated_at=now.isoformat(timespec="seconds"),
        date=now.strftime("%Y-%m-%d"),
        ticker=ticker,
        signal=signal,
        signal_raw=signal_val,
        strategy=strategy.name,
        confidence=confidence,
        confidence_score=confidence_score,
        composite_score=round(composite_score_val, 2),
        current_price=current_price,
        stop_loss_pct=stop_loss_pct,
        stop_loss_price=stop_loss_price,
        take_profit_pct=take_profit_pct,
        take_profit_price=take_profit_price,
        suggested_position_size_pct=suggested_position_size_pct,
        signal_expires=signal_expires,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        win_rate=round(result.win_rate * 100, 1),
        profit_factor=round(result.profit_factor, 3),
        annual_return_pct=result.annual_return_pct,
        annual_excess_pct=result.annual_excess_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        avg_holding_days=round(result.avg_holding_days, 1),
        total_trades=result.total_trades,
        backtest_period=backtest_period,
        rsi=round(rsi_val, 1),
        vol_20=round(float(vol_20), 4),
        sma_50=round(float(latest.get("SMA_50", 0) or 0), 2),
        sma_200=round(float(latest.get("SMA_200", 0) or 0), 2),
        trend=trend,
        volatility=volatility,
        notes="; ".join(notes_parts) if notes_parts else "No special conditions",
    )


# ──────────────────────────────────────────────────────────────────────
#  OUTPUT
# ──────────────────────────────────────────────────────────────────────

def write_signals(signals: List[DailySignal], config: dict):
    """Write signals to CSV and JSON, sorted by composite score (highest first)."""
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Sort: highest composite score first, errors sink to bottom
    sorted_signals = sorted(signals, key=lambda s: -s.composite_score)

    # CSV
    csv_path = os.path.join(out_dir, f"signals_{date_str}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(sorted_signals[0]).keys()))
        writer.writeheader()
        for s in sorted_signals:
            writer.writerow(asdict(s))
    log.info(f"Signals written to {csv_path}")

    # JSON
    json_path = os.path.join(out_dir, f"signals_{date_str}.json")
    output = {
        "generated_at": datetime.now().isoformat(),
        "risk_profile": config["risk_profile"],
        "signals": [asdict(s) for s in sorted_signals],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log.info(f"Signals written to {json_path}")

    return csv_path, json_path


def write_trade_logs(all_trade_logs: Dict[str, List[TradeRecord]], config: dict):
    """Write per-ticker trade log CSVs showing every individual trade."""
    trade_dir = config["trade_log_dir"]
    os.makedirs(trade_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    paths = []

    for ticker, trades in all_trade_logs.items():
        if not trades:
            continue
        path = os.path.join(trade_dir, f"trades_{ticker}_{date_str}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(trades[0]).keys()))
            writer.writeheader()
            for t in trades:
                writer.writerow(asdict(t))
        paths.append(path)
        log.info(f"Trade log written: {path} ({len(trades)} trades)")

    return paths


def write_backtest_report(all_window_results: Dict[str, Dict[str, list]],
                          composite_scores: Dict[str, Dict[str, float]],
                          best_strategies: Dict[str, str],
                          config: dict):
    """Write a detailed multi-timeframe backtest report as CSV, sorted by composite score."""
    report_dir = config["report_dir"]
    os.makedirs(report_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(report_dir, f"backtest_report_{date_str}.csv")
    weights = config["window_weights"]
    mode = "LONG-ONLY" if config.get("long_only", True) else "LONG + SHORT"

    # -- Build flat rows for every (ticker, strategy, timeframe) combination --
    CSV_COLUMNS = [
        "ticker", "strategy", "composite_score", "is_best", "timeframe",
        "annual_return_pct", "annual_excess_pct", "sharpe_ratio", "sortino_ratio",
        "win_rate", "max_drawdown_pct", "total_trades", "profit_factor",
        "avg_holding_days", "calmar_ratio", "total_return_pct",
        "buy_hold_return_pct", "excess_return_pct", "window_score",
        "backtest_start", "backtest_end", "trading_days",
    ]

    rows = []
    for ticker in all_window_results:
        best_name = best_strategies[ticker]
        comp = composite_scores[ticker]  # strat_name -> composite_score

        for window_name, window_results in all_window_results[ticker].items():
            for strat, r, sc in window_results:
                rows.append({
                    "ticker": ticker,
                    "strategy": r.strategy_name,
                    "composite_score": round(comp.get(r.strategy_name, 0), 2),
                    "is_best": "Y" if strat.name == best_name else "",
                    "timeframe": window_name,
                    "annual_return_pct": round(r.annual_return_pct, 2),
                    "annual_excess_pct": round(r.annual_excess_pct, 2),
                    "sharpe_ratio": round(r.sharpe_ratio, 3),
                    "sortino_ratio": round(r.sortino_ratio, 3),
                    "win_rate": round(r.win_rate, 4),
                    "max_drawdown_pct": round(r.max_drawdown_pct, 2),
                    "total_trades": r.total_trades,
                    "profit_factor": round(r.profit_factor, 3),
                    "avg_holding_days": round(r.avg_holding_days, 1),
                    "calmar_ratio": round(r.calmar_ratio, 3),
                    "total_return_pct": round(r.total_return_pct, 2),
                    "buy_hold_return_pct": round(r.buy_hold_return_pct, 2),
                    "excess_return_pct": round(r.excess_return_pct, 2),
                    "window_score": round(sc, 2),
                    "backtest_start": r.backtest_start,
                    "backtest_end": r.backtest_end,
                    "trading_days": r.trading_days,
                })

    # Sort: highest composite score first, then by ticker, then timeframe
    rows.sort(key=lambda r: (-r["composite_score"], r["ticker"], r["timeframe"]))

    # -- Write CSV --
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Backtest report written to {csv_path}")

    # -- Build a concise console summary sorted by best composite score --
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append(f"  QUANT ANALYSIS BOT -- BACKTEST SUMMARY  ({date_str})")
    summary_lines.append(f"  Risk Profile: {config['risk_profile'].upper()}  |  Mode: {mode}")
    summary_lines.append(f"  Timeframe Weights: {', '.join(f'{k}={v:.0%}' for k,v in weights.items())}")
    summary_lines.append("=" * 100)

    # One line per ticker, sorted by best composite score desc
    ticker_best = []
    for ticker in all_window_results:
        best_name = best_strategies[ticker]
        best_score = composite_scores[ticker].get(best_name, 0)
        ticker_best.append((ticker, best_name, best_score))
    ticker_best.sort(key=lambda x: -x[2])

    summary_lines.append(
        f"  {'Rank':>4}  {'Ticker':<7} {'Best Strategy':<32} {'Score':>7} "
        f"{'Ann.Ret%':>9} {'Sharpe':>7} {'WinRate':>8} {'MaxDD%':>8}"
    )
    summary_lines.append(f"  {'-' * 92}")

    for rank, (ticker, best_name, best_score) in enumerate(ticker_best, 1):
        # Get the longest-window result for the best strategy for display
        best_row = None
        for row in rows:
            if row["ticker"] == ticker and row["is_best"] == "Y":
                best_row = row
                break
        if best_row:
            summary_lines.append(
                f"  {rank:>4}  {ticker:<7} {best_name:<32} {best_score:>7.1f} "
                f"{best_row['annual_return_pct']:>8.1f}% "
                f"{best_row['sharpe_ratio']:>7.2f} "
                f"{best_row['win_rate']:>7.1%} "
                f"{best_row['max_drawdown_pct']:>8.1f}"
            )

    summary_lines.append("=" * 100)
    summary_lines.append(f"  Full report: {csv_path}")
    summary_lines.append("=" * 100)

    report_text = "\n".join(summary_lines)
    return csv_path, report_text


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> dict:
    """Load config from JSON file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            user_config = json.load(f)
        config.update(user_config)
        log.info(f"Loaded config from {config_path}")
    return config


def run(config: dict):
    """Main execution pipeline."""
    today = datetime.now().strftime("%Y-%m-%d")
    log.info(f"{'=' * 60}")
    mode = "LONG-ONLY" if config.get("long_only", True) else "LONG + SHORT"
    log.info(f"  Quant Analysis Bot -- {today}")
    tickers = config["tickers"]
    if len(tickers) <= 15:
        log.info(f"  Tickers: {', '.join(tickers)}")
    else:
        log.info(f"  Tickers: {len(tickers)} stocks (top by market cap)")
    log.info(f"  Risk Profile: {config['risk_profile']}  |  Mode: {mode}")
    log.info(f"  Timeframes: {', '.join(config['backtest_windows'].keys())}")
    log.info(f"{'=' * 60}")

    all_signals = []
    all_window_results = {}   # ticker -> {window -> [(strat, result, score)]}
    all_composite_scores = {} # ticker -> {strat_name -> composite_score}
    best_strategies = {}
    all_trade_logs = {}       # ticker -> [TradeRecord, ...]

    tickers = config["tickers"]
    n_tickers = len(tickers)
    use_progress = n_tickers > 10  # show progress bar for large runs
    failed_tickers = []

    with ProgressBar(total=n_tickers, desc="Analyzing stocks", unit="stocks") if use_progress else nullcontext() as pbar:
        for i, ticker in enumerate(tickers):
            if not use_progress:
                log.info(f"\n>>> Analyzing {ticker}...")

            try:
                # 1. Fetch & enrich data
                df = fetch_data(ticker, config["lookback_days"], config["data_cache_dir"])
                df = enrich_dataframe(df)

                # 2. Multi-timeframe strategy selection
                best_strat, best_result, per_window, comp_scores, trade_logs = \
                    select_best_strategy(df, ticker, config)

                if not use_progress:
                    log.info(f"  Best strategy: {best_strat.name} "
                             f"(Composite={best_result.composite_score:.1f}, "
                             f"Sharpe={best_result.sharpe_ratio}, "
                             f"Ann.Return={best_result.annual_return_pct}%, "
                             f"WinRate={best_result.win_rate:.1%})")

                all_window_results[ticker] = per_window
                all_composite_scores[ticker] = comp_scores
                best_strategies[ticker] = best_strat.name
                all_trade_logs[ticker] = trade_logs

                # 3. Generate today's signal
                signal = generate_daily_signal(df, ticker, best_strat, best_result, config)
                all_signals.append(signal)
                if not use_progress:
                    log.info(f"  Signal: {signal.signal} (Confidence: {signal.confidence})")
                    log.info(f"  Total trades logged across all timeframes: {len(trade_logs)}")

            except Exception as e:
                if not use_progress:
                    log.error(f"  FAILED for {ticker}: {e}")
                failed_tickers.append(ticker)
                all_signals.append(DailySignal(
                    generated_at=datetime.now().isoformat(timespec="seconds"),
                    date=today, ticker=ticker, signal="ERROR", signal_raw=0,
                    strategy="N/A", confidence="N/A", confidence_score=0,
                    composite_score=0.0,
                    current_price=0.0,
                    stop_loss_pct=0.0, stop_loss_price=0.0,
                    take_profit_pct=0.0, take_profit_price=0.0,
                    suggested_position_size_pct=0.0,
                    signal_expires=today,
                    sharpe=0.0, sortino=0.0, win_rate=0.0, profit_factor=0.0,
                    annual_return_pct=0.0, annual_excess_pct=0.0,
                    max_drawdown_pct=0.0, avg_holding_days=0.0, total_trades=0,
                    backtest_period="N/A",
                    rsi=0.0, vol_20=0.0, sma_50=0.0, sma_200=0.0,
                    trend="N/A", volatility="N/A", notes=str(e),
                ))

            if use_progress and pbar:
                pbar.update(1, suffix=ticker)

    # Report failures summary for large runs
    if use_progress and failed_tickers:
        log.warning(f"\n  {len(failed_tickers)}/{n_tickers} tickers failed: "
                    f"{', '.join(failed_tickers[:20])}"
                    f"{'...' if len(failed_tickers) > 20 else ''}")

    # 4. Write outputs
    csv_path, json_path = "", ""
    if all_signals:
        csv_path, json_path = write_signals(all_signals, config)

    if all_trade_logs:
        trade_paths = write_trade_logs(all_trade_logs, config)

    if all_window_results:
        report_path, report_text = write_backtest_report(
            all_window_results, all_composite_scores, best_strategies, config
        )
        print(f"\n{report_text}")

    # Print signal summary
    print(f"\n{'=' * 70}")
    print(f"  TODAY'S SIGNALS  ({today})")
    print(f"{'=' * 70}")
    for s in all_signals:
        icon = {"BUY": "+", "SELL/SHORT": "-", "EXIT": "x", "HOLD": "=", "ERROR": "!"}
        marker = icon.get(s.signal, "?")
        print(f"  [{marker}] {s.ticker:<6} {s.signal:<5} | {s.strategy}")
        print(f"       Price: ${s.current_price}  RSI: {s.rsi}  "
              f"Trend: {s.trend}  Confidence: {s.confidence}")
        print(f"       Ann.Return: {s.annual_return_pct}%  "
              f"Ann.Excess: {s.annual_excess_pct}%  MaxDD: {s.max_drawdown_pct}%")
        print(f"       Backtest: {s.backtest_period}")
        if s.notes and s.notes != "No special conditions":
            print(f"       Notes: {s.notes}")
    print(f"{'=' * 70}")
    if csv_path:
        print(f"  Signals:    {csv_path}  |  {json_path}")
    if all_trade_logs:
        total_trades = sum(len(t) for t in all_trade_logs.values())
        print(f"  Trade logs: {config['trade_log_dir']}/ ({total_trades} trades total)")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Quant Analysis Bot")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--tickers", type=str, nargs="+", help="Override ticker list")
    parser.add_argument("--risk", type=str, choices=["conservative", "moderate", "aggressive"],
                        help="Override risk profile")
    parser.add_argument("--long-only", action="store_true", default=None,
                        help="Long-only mode: SELL = exit to cash, no shorting (default)")
    parser.add_argument("--long-short", action="store_true", default=None,
                        help="Long+Short mode: SELL = open short position")
    parser.add_argument("--all-stocks", action="store_true",
                        help="Analyze top US stocks by market cap (default: top 1000)")
    parser.add_argument("--top-n", type=int, default=1000,
                        help="Number of top stocks when using --all-stocks (default: 1000)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.tickers:
        config["tickers"] = args.tickers
    if args.risk:
        config["risk_profile"] = args.risk
    if args.long_only:
        config["long_only"] = True
    elif args.long_short:
        config["long_only"] = False

    # Fetch top US stocks if --all-stocks flag is set
    if args.all_stocks:
        log.info(f"Fetching top {args.top_n} US stocks by market cap...")
        config["tickers"] = fetch_top_us_stocks(
            n=args.top_n,
            cache_dir=config.get("data_cache_dir", "cache"),
        )
        if not config["tickers"]:
            log.error("Failed to fetch stock universe. Exiting.")
            sys.exit(1)
        log.info(f"Will analyze {len(config['tickers'])} stocks")
        # For large runs, reduce per-stock log verbosity to keep output clean
        if len(config["tickers"]) > 20:
            logging.getLogger("quant_bot").setLevel(logging.WARNING)

    run(config)


if __name__ == "__main__":
    main()
