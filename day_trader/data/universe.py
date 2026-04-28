"""Universe builder — list of tickers the day-trader will consider.

v1 ships a curated default universe (~60 high-liquidity ETFs and
mega-caps) plus a CSV-override hook. v2 will replace the default
with a live Russell-1000 fetch, ADV-screened.

Why a curated default rather than "the whole market":

- The day-trader's filter pipeline + sub-budget keep capacity small
  (3 max concurrent, $25k budget on a $100k account). There's no
  edge in scanning 5,000 tickers when at most 8 trades fire per day.
- Latency: the WebSocket subscribe-list is bounded. Fewer symbols =
  smaller bandwidth, faster fan-out.
- Slippage: top liquidity is where ATR-bracket fills land where we
  expect them. Thin names blow through stops.

The default is biased to:

- Index ETFs (SPY/QQQ/IWM/DIA + sector slices) — tightest spreads
- Mega-cap leaders (AAPL/MSFT/NVDA/etc.) — academic literature on
  Stocks-in-Play applies primarily to liquid US large-caps
- A few volatility ETFs (UVXY/VXX) to allow regime-aware entries

Override mechanism: pass ``csv_path=...`` to ``load_universe`` to
read a single-column CSV of tickers. Empty / missing rows skipped.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

# Valid ticker shape — same rules as order_tags.py.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]*$")

# Common CSV header-row values that ALSO match _TICKER_RE.
# "TICKER" is a valid ticker shape (all uppercase letters) but is
# obviously a header, not a symbol. This set catches the most common
# column-name overlaps so the shape-based auto-detection doesn't
# misfire.
_COMMON_HEADERS = frozenset({
    "TICKER", "TICKERS", "SYMBOL", "SYMBOLS",
    "STOCK", "STOCKS", "NAME", "COMPANY", "INSTRUMENT",
})


# ── Default v1 universe (~60 symbols) ─────────────────────────────

#: Index + sector ETFs — almost always tightest-spread tickers
_DEFAULT_ETFS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
    "XLRE", "XLC",
    "TLT", "HYG", "GLD", "SLV", "USO",
    "UVXY", "VXX",
    "TQQQ", "SQQQ", "SOXL", "SOXS",
]

#: Mega-cap and high-liquidity individual stocks
_DEFAULT_STOCKS = [
    # Megacaps — the consistent volume leaders
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    # Other top-50-by-volume liquid names
    "AMD", "AVGO", "ORCL", "CRM", "ADBE", "NFLX", "INTC",
    "COIN", "MSTR", "PLTR",
    # Banks
    "JPM", "BAC", "WFC", "GS", "MS",
    # Healthcare / consumer
    "UNH", "JNJ", "LLY", "PFE",
    "HD", "WMT", "COST", "MCD", "NKE",
    # Energy
    "XOM", "CVX",
    # Other large-caps with high intraday volume
    "BABA", "DIS", "BA", "F", "GE", "UBER",
]


DEFAULT_UNIVERSE: tuple[str, ...] = tuple(_DEFAULT_ETFS + _DEFAULT_STOCKS)


# ── Loading ───────────────────────────────────────────────────────


def load_universe(
    *,
    csv_path: Optional[Path] = None,
    extra_symbols: Optional[Iterable[str]] = None,
    excluded_symbols: Optional[Iterable[str]] = None,
) -> list[str]:
    """Build today's tradeable universe.

    Resolution order:

    1. If ``csv_path`` is given and exists, read tickers from it
       (single-column CSV; first column wins; blanks/comments skipped).
    2. Otherwise use ``DEFAULT_UNIVERSE``.

    Then apply ``extra_symbols`` (added) and ``excluded_symbols``
    (removed). Result is normalized to uppercase, deduplicated,
    and sorted alphabetically.

    Returns an empty list only if the CSV exists but contains no
    valid rows — never silently swallow that misconfiguration.
    """
    base: list[str] = []
    if csv_path is not None and Path(csv_path).exists():
        base = _load_csv(Path(csv_path))
        if not base:
            log.warning(
                "Universe CSV %s exists but contains no valid tickers — "
                "returning empty universe (the daemon will refuse to scan)",
                csv_path,
            )
    else:
        base = list(DEFAULT_UNIVERSE)

    extra_set = set(_normalize_symbols(extra_symbols or []))
    exclude_set = set(_normalize_symbols(excluded_symbols or []))

    seen: set[str] = set()
    skipped: list[str] = []
    out: list[str] = []
    for s in base:
        u = s.strip().upper()
        if not u or u in seen or u in exclude_set:
            continue
        if not _TICKER_RE.match(u):
            skipped.append(u)
            continue
        seen.add(u)
        out.append(u)
    for s in extra_set:
        if s and s not in seen and s not in exclude_set:
            if not _TICKER_RE.match(s):
                skipped.append(s)
                continue
            seen.add(s)
            out.append(s)
    if skipped:
        log.warning(
            "Universe: skipped %d invalid ticker(s): %s",
            len(skipped),
            ", ".join(skipped[:10]) + ("…" if len(skipped) > 10 else ""),
        )
    return sorted(out)


def _load_csv(path: Path) -> list[str]:
    """Read tickers from a single-column CSV.

    Tolerant: skips blank lines, comment lines (start with #),
    and any column beyond the first. Strips whitespace.

    Header detection: if the first non-blank, non-comment cell does
    NOT look like a valid ticker (uppercase alpha + digits/dots), we
    treat it as a header and skip it. This handles arbitrary header
    names (``ticker``, ``Symbol``, ``stock_id``, etc.) without
    maintaining a hardcoded allowlist.
    """
    out: list[str] = []
    first_data_line = True
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cell = row[0].strip()
            if not cell or cell.startswith("#"):
                continue
            upper = cell.upper()
            if first_data_line:
                first_data_line = False
                # If the first data cell doesn't match ticker shape
                # OR is a known column-header word (e.g. "TICKER"),
                # treat as header and skip.
                if not _TICKER_RE.match(upper) or upper in _COMMON_HEADERS:
                    continue
            out.append(upper)
    return out


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    return [s.strip().upper() for s in symbols if s and s.strip()]
