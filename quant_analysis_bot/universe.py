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
import tempfile
import urllib.request
from datetime import datetime, timedelta
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
# Cache TTL: 14 days, but always expires on Monday so a fresh fetch
# happens at the start of each trading week.
_CACHE_MAX_AGE_DAYS = 14
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


def _cache_is_fresh(cache_path: str) -> bool:
    """Return True if the cache file exists and is still valid.

    The cache expires when **either** condition is met:
      1. The file is older than ``_CACHE_MAX_AGE_DAYS`` (14 days).
      2. A Monday boundary has been crossed since the file was
         written — i.e. at least one Monday 00:00 local time has
         passed.  This ensures a fresh fetch at the start of each
         trading week.
    """
    if not os.path.exists(cache_path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
    now = datetime.now()
    age = now - mtime
    if age > timedelta(days=_CACHE_MAX_AGE_DAYS):
        return False
    # Check whether a Monday 00:00 has passed since mtime.
    # Walk forward from mtime to the next Monday; if that Monday
    # is <= now, the cache spans a week boundary → stale.
    days_until_monday = (7 - mtime.weekday()) % 7  # 0 if mtime is Mon
    if days_until_monday == 0:
        # mtime is a Monday — next boundary is the *following* Monday
        next_monday = (mtime + timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    else:
        next_monday = (mtime + timedelta(days=days_until_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    if now >= next_monday:
        return False
    return True


def fetch_top_us_stocks(
    n: int = 1000,
    cache_dir: str = _UNIVERSE_CACHE_DIR,
    force_refresh: bool = False,
) -> List[str]:
    """Fetch the top N US stocks by market cap.

    Uses a **tiered approach** to minimise yfinance API calls:

    S&P index membership already groups stocks by market-cap tier:
      - S&P 500   → large cap  (~500 stocks)
      - MidCap 400 → mid cap   (~400 stocks)
      - SmallCap 600 → small cap (~600 stocks)

    Rather than fetching all 1500 tickers and sorting, we:
      1. Only fetch indices needed to fill N slots.
      2. Include all tickers from fully-consumed tiers
         without fetching market caps (they're guaranteed to
         rank above the next tier).
      3. Only fetch market caps for the *boundary tier* — the
         one that straddles the N cutoff — so we can sort
         within that tier and pick the top remainder.

    Example: ``--top-n 700`` needs all 500 S&P 500 stocks (no
    market cap fetch needed) + the top 200 from MidCap 400
    (fetch caps for ~400, sort, take 200).  SmallCap 600 is
    never touched.  This cuts API calls from ~1500 to ~400.

    Fallback chain:
      1. Check cache (valid for up to 14 days, resets each Monday)
      2. Try Wikipedia (S&P 500 + 400 + 600, as needed)
      3. Fall back to bundled us_tickers.json
      4. Fetch market caps only for the boundary tier

    Args:
        n: Number of top stocks to return.
        cache_dir: Directory to store cache files.
        force_refresh: If True, ignore the cache and re-fetch
            from Wikipedia / yfinance.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"universe_top{n}.json")

    # 1. Check cache (14-day TTL, resets on Monday)
    if not force_refresh and _cache_is_fresh(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                f"  Cache file corrupt or unreadable ({exc}), "
                f"rebuilding..."
            )
        else:
            age_days = (
                datetime.now()
                - datetime.fromtimestamp(
                    os.path.getmtime(cache_file)
                )
            ).days
            log.info(
                f"  Loaded {len(cached)} tickers from universe "
                f"cache ({cache_file}, {age_days}d old)"
            )
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
    if force_refresh:
        log.info("  Force-refresh requested — bypassing cache.")

    log.info(f"  Building top {n} US stock universe...")

    # 2. Fetch tickers per tier from Wikipedia
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

    # Collect tickers per tier, preserving tier order
    tiers: List[List[str]] = []
    seen: set[str] = set()
    cumulative = 0

    for name, url in wiki_sources:
        raw = _fetch_wiki_tickers(name, url)
        # Deduplicate across tiers (a ticker might appear in
        # multiple indices during rebalancing)
        unique = [t for t in raw if t not in seen]
        seen.update(unique)
        tiers.append(unique)
        cumulative += len(unique)
        # Stop fetching lower tiers if we already have enough
        # candidates.  We need at least N tickers to fill the
        # request.  Add a small buffer for tickers that might
        # fail market-cap lookup.
        if cumulative >= n:
            break

    # 3. Bundled fallback
    total_tickers = sum(len(t) for t in tiers)
    if total_tickers < 100:
        log.warning(
            "  Wikipedia returned too few tickers, "
            "using bundled list..."
        )
        bundled = _load_bundled_tickers()
        # Treat bundled as a single unsorted tier
        unique_bundled = [t for t in bundled if t not in seen]
        tiers.append(unique_bundled)
        total_tickers += len(unique_bundled)

    if total_tickers < 50:
        log.error(
            "  Cannot build stock universe: "
            "no ticker source available."
        )
        return []

    log.info(
        f"  Collected {total_tickers} unique tickers "
        f"across {len(tiers)} tiers"
    )

    # 4. Tiered selection: include full higher tiers, sort
    #    only the boundary tier by market cap
    top_n: List[str] = []
    market_caps: Dict[str, float] = {}
    remaining = n

    for i, tier in enumerate(tiers):
        if remaining <= 0:
            break

        if len(tier) <= remaining:
            # This entire tier fits within N — include all
            # without fetching market caps (they're guaranteed
            # to outrank the next tier by index construction)
            top_n.extend(sorted(tier))
            remaining -= len(tier)
            log.info(
                f"  Tier {i + 1}: included all {len(tier)} "
                f"tickers ({remaining} slots remaining)"
            )
        else:
            # Boundary tier: need to sort by market cap to
            # pick the top `remaining` from this tier
            log.info(
                f"  Tier {i + 1}: sorting {len(tier)} tickers "
                f"by market cap (need top {remaining})..."
            )
            tier_caps = _get_market_caps(tier)
            market_caps.update(tier_caps)

            sorted_tier = sorted(
                tier,
                key=lambda t: tier_caps.get(t, 0),
                reverse=True,
            )
            top_n.extend(sorted_tier[:remaining])
            remaining = 0

    # 5. Append extra tickers (ETFs, commodities, etc.)
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

    # Cache (valid until next Monday or 14 days, whichever is sooner).
    # Atomic write: write to a temp file then rename, so a crash
    # mid-write can't leave a corrupt cache that blocks future runs.
    cache_dir_path = os.path.dirname(cache_file)
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=cache_dir_path, suffix=".tmp"
        )
        with os.fdopen(fd, "w") as f:
            json.dump(top_n, f)
        os.replace(tmp_path, cache_file)
    except OSError as exc:
        log.warning(f"  Failed to write cache: {exc}")
        # Clean up temp file if rename failed
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    else:
        log.info(f"  Cached {len(top_n)} tickers to {cache_file}")

    if top_n and market_caps:
        top5_str = ", ".join(
            f"{t} (${market_caps[t] / 1e9:.0f}B)"
            for t in top_n[:5]
            if t in market_caps
        )
        if top5_str:
            log.info(f"  Top 5: {top5_str}")
        ranked = [t for t in top_n if t in market_caps]
        if ranked:
            bottom = ranked[-1]
            log.info(
                f"  Boundary cutoff: {bottom} "
                f"(${market_caps[bottom] / 1e9:.1f}B)"
            )

    log.info(f"  Final universe: {len(top_n)} tickers")
    return top_n
