"""Stock universe construction -- top US stocks by market cap.

The universe is built from S&P 500/400/600 components, sorted by
market cap.  Extra tickers (ETFs, commodities, etc.) can be added
via the ``EXTRA_TICKERS`` env var or an ``extra_tickers.txt`` file
so they are always included regardless of index membership.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "yfinance is required. Install with: pip install yfinance"
    ) from exc

from quant_analysis_bot.progress import ProgressBar

log = logging.getLogger(__name__)

_UNIVERSE_CACHE_DIR = "cache"
_BUNDLED_TICKERS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "us_tickers.json",
)
# Extra tickers file: checked in bot root and in quant_analysis_bot/
_EXTRA_TICKERS_PATHS = [
    Path(os.path.dirname(os.path.abspath(__file__)), "..", "extra_tickers.txt"),
    Path(os.path.dirname(os.path.abspath(__file__)), "extra_tickers.txt"),
]


def _fetch_wiki_html(url: str) -> str:
    """Fetch HTML from Wikipedia with a proper User-Agent."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "QuantAnalysisBot/1.0 "
                "(https://github.com/user/quant-bot; "
                "educational use) Python/pandas"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _fetch_wiki_tickers(name: str, url: str) -> List[str]:
    """Fetch tickers from a Wikipedia S&P index page."""
    try:
        from io import StringIO

        html = _fetch_wiki_html(url)
        tables = pd.read_html(StringIO(html))
        df = tables[0]

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
        tickers = [
            t
            for t in tickers
            if t and 1 <= len(t) <= 5 and t.replace("-", "").isalpha()
        ]
        log.info(f"  {name}: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.warning(
            f"  Failed to fetch {name} from Wikipedia: {e}"
        )
        return []


def _load_bundled_tickers() -> List[str]:
    """Load the bundled ticker list shipped with the bot."""
    if os.path.exists(_BUNDLED_TICKERS_FILE):
        with open(_BUNDLED_TICKERS_FILE, "r") as f:
            tickers = json.load(f)
        log.info(
            f"  Loaded {len(tickers)} tickers from bundled list"
        )
        return tickers
    return []


def load_extra_tickers() -> List[str]:
    """Load extra tickers from env var and/or text file.

    These are tickers that fall outside the S&P index universe
    (e.g. commodity ETFs like USO, GLD, sector ETFs like XLE,
    treasury ETFs like TLT, or any other instruments you want
    the bot to consider).

    Sources (merged, deduplicated):

    1. ``EXTRA_TICKERS`` env var — comma-separated, e.g.
       ``EXTRA_TICKERS=USO,GLD,XLE,TLT``
    2. ``extra_tickers.txt`` file — one ticker per line,
       ``#`` comments and blank lines ignored.  Searched in
       project root and quant_analysis_bot/ directory.

    Extra tickers are appended to the universe *after* the
    market-cap sort, so they are always included regardless
    of their market cap ranking.  They still go through the
    same backtest pipeline and must produce a strong enough
    signal to pass risk checks.
    """
    extras: set[str] = set()

    # 1. Env var
    env_val = os.getenv("EXTRA_TICKERS", "").strip()
    if env_val:
        for raw in env_val.split(","):
            ticker = raw.strip().upper().replace(".", "-")
            if ticker and 1 <= len(ticker) <= 6:
                extras.add(ticker)
        if extras:
            log.info(
                f"  EXTRA_TICKERS env: {len(extras)} tickers "
                f"({', '.join(sorted(extras))})"
            )

    # 2. Text file
    for path in _EXTRA_TICKERS_PATHS:
        resolved = path.resolve()
        if resolved.exists():
            try:
                with open(resolved, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and blanks
                        if not line or line.startswith("#"):
                            continue
                        # Support inline comments
                        ticker = (
                            line.split("#")[0]
                            .strip()
                            .upper()
                            .replace(".", "-")
                        )
                        if ticker and 1 <= len(ticker) <= 6:
                            extras.add(ticker)
                log.info(
                    f"  Loaded extra tickers from {resolved}"
                )
            except OSError as exc:
                log.debug(
                    f"  Could not read {resolved}: {exc}"
                )
            break  # Only read the first file found

    if extras:
        log.info(
            f"  {len(extras)} extra tickers total: "
            f"{', '.join(sorted(extras))}"
        )

    return sorted(extras)


def _get_market_caps(
    tickers: List[str], batch_size: int = 50
) -> Dict[str, float]:
    """Fetch market caps for a list of tickers using yfinance."""
    market_caps: Dict[str, float] = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    log.info(
        f"  Fetching market caps for {len(tickers)} tickers "
        f"({total_batches} batches)..."
    )

    with ProgressBar(
        total=len(tickers), desc="Market caps", unit="stocks"
    ) as pbar:
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
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


def fetch_top_us_stocks(
    n: int = 1000, cache_dir: str = _UNIVERSE_CACHE_DIR
) -> List[str]:
    """
    Fetch the top N US stocks by market cap.

    Fallback chain:
      1. Check daily cache
      2. Try Wikipedia (S&P 500 + 400 + 600)
      3. Fall back to bundled us_tickers.json
      4. Fetch market caps via yfinance, sort, return top N
    """
    os.makedirs(cache_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    cache_file = os.path.join(
        cache_dir, f"universe_top{n}_{date_str}.json"
    )

    # 1. Daily cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = json.load(f)
        log.info(
            f"  Loaded {len(cached)} tickers from "
            f"universe cache ({cache_file})"
        )
        # Still check for extra tickers — user may have added
        # new ones after the cache was built today.
        extra = load_extra_tickers()
        if extra:
            already = set(cached)
            added = [t for t in extra if t not in already]
            if added:
                cached.extend(added)
                log.info(
                    f"  Appended {len(added)} extra tickers "
                    f"not in cache: {', '.join(added)}"
                )
        return cached[:n] if not extra else cached

    log.info(f"  Building top {n} US stock universe...")

    # 2. Wikipedia
    all_tickers: set[str] = set()
    wiki_sources = [
        (
            "S&P 500",
            "https://en.wikipedia.org/wiki/"
            "List_of_S%26P_500_companies",
        ),
        (
            "S&P MidCap 400",
            "https://en.wikipedia.org/wiki/"
            "List_of_S%26P_400_companies",
        ),
        (
            "S&P SmallCap 600",
            "https://en.wikipedia.org/wiki/"
            "List_of_S%26P_600_companies",
        ),
    ]

    for name, url in wiki_sources:
        tickers = _fetch_wiki_tickers(name, url)
        all_tickers.update(tickers)
        if len(all_tickers) >= n * 1.2:
            break

    # 3. Bundled fallback
    if len(all_tickers) < 100:
        log.warning(
            "  Wikipedia returned too few tickers, "
            "using bundled list..."
        )
        bundled = _load_bundled_tickers()
        all_tickers.update(bundled)

    if len(all_tickers) < 50:
        log.error(
            "  Cannot build stock universe: "
            "no ticker source available."
        )
        return []

    ticker_list = sorted(all_tickers)
    log.info(f"  Collected {len(ticker_list)} unique tickers")

    # 4. Market caps
    market_caps = _get_market_caps(ticker_list)
    log.info(
        f"  Got market caps for {len(market_caps)} tickers"
    )

    if len(market_caps) < 50:
        log.error(
            f"  Only got market caps for {len(market_caps)} "
            f"stocks -- check your internet"
        )
        return []

    sorted_tickers = sorted(
        market_caps.keys(),
        key=lambda t: market_caps[t],
        reverse=True,
    )
    top_n = sorted_tickers[:n]

    # 5. Append extra tickers (ETFs, commodities, etc.)
    #    These bypass the market-cap ranking — they're always
    #    included so the bot can evaluate non-index instruments.
    extra = load_extra_tickers()
    if extra:
        already = set(top_n)
        added = [t for t in extra if t not in already]
        if added:
            top_n.extend(added)
            log.info(
                f"  Appended {len(added)} extra tickers: "
                f"{', '.join(added)}"
            )

    # Cache for today
    with open(cache_file, "w") as f:
        json.dump(top_n, f)
    log.info(f"  Cached {len(top_n)} tickers to {cache_file}")

    if top_n:
        top5_str = ", ".join(
            f"{t} (${market_caps[t] / 1e9:.0f}B)"
            for t in top_n[:5]
            if t in market_caps
        )
        if top5_str:
            log.info(f"  Top 5: {top5_str}")
        # Show the last ticker that was ranked by market cap (not extras)
        ranked = [t for t in top_n if t in market_caps]
        if len(ranked) >= n:
            bottom = ranked[-1]
            log.info(
                f"  #{n}: {bottom} "
                f"(${market_caps[bottom] / 1e9:.1f}B)"
            )

    return top_n
