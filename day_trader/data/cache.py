"""BarCache — rolling per-ticker bar buffer with on-the-fly indicators.

Bars arrive in chronological order via :meth:`add_bar`. The cache:

- Keeps the last ``ring_size`` bars per ticker (default 390 = full
  regular session at 1-min bars).
- Computes session VWAP cumulatively as bars arrive (typical-price-
  weighted) so strategies don't recompute on every scan.
- Exposes O(1) accessors for latest bar, latest VWAP, ATR, session
  high/low, and recent-window slices.

Concurrency: single-process daemon. We use a lock around mutation
because the feed task and the scan task both read.

Disk snapshots and crash recovery: deferred. The first version is
purely in-memory; we'll add a JSON snapshot path when the executor
crashes-and-recovers behaviour is wired up.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from dataclasses import replace
from typing import Optional

from day_trader.models import Bar

log = logging.getLogger(__name__)


# Default ring size: 390 1-min bars = a full regular NYSE session.
# For 5-min bars this gives 78 sessions of history (much more than
# any strategy needs); strategies should slice via ``get_bars(n=…)``.
DEFAULT_RING_SIZE = 390


class _SessionVwapState:
    """Running cumulative price·volume / volume for one ticker."""

    __slots__ = ("cum_pv", "cum_v")

    def __init__(self) -> None:
        self.cum_pv: float = 0.0
        self.cum_v: float = 0.0

    def update(self, bar: Bar) -> float:
        # Typical price = (H + L + C) / 3 — the standard VWAP input.
        # Falls back to close if H/L are zero (shouldn't happen on
        # real bars, but defensive against malformed inputs).
        if bar.high > 0 and bar.low > 0:
            tp = (bar.high + bar.low + bar.close) / 3.0
        else:
            tp = bar.close
        self.cum_pv += tp * bar.volume
        self.cum_v += bar.volume
        if self.cum_v <= 0:
            return 0.0
        return self.cum_pv / self.cum_v

    def reset(self) -> None:
        self.cum_pv = 0.0
        self.cum_v = 0.0


class BarCache:
    """Rolling per-ticker bar buffer with VWAP and ATR helpers."""

    def __init__(self, ring_size: int = DEFAULT_RING_SIZE):
        if ring_size < 2:
            raise ValueError(f"ring_size must be >= 2, got {ring_size!r}")
        self._ring_size = ring_size
        self._bars: dict[str, deque[Bar]] = defaultdict(
            lambda: deque(maxlen=ring_size)
        )
        self._vwap_state: dict[str, _SessionVwapState] = defaultdict(
            _SessionVwapState
        )
        self._lock = threading.Lock()

    # ── Mutation ──────────────────────────────────────────────────

    def add_bar(self, bar: Bar) -> Bar:
        """Append ``bar`` to the cache, computing session VWAP.

        Returns the bar with ``vwap`` populated. The original frozen
        Bar is replaced with one carrying the computed VWAP — the
        caller can use the returned object for strategy input.

        If ``bar.vwap`` is already non-zero (e.g. the feed is sending
        pre-computed VWAP), we trust it and do not overwrite.
        """
        if not bar.ticker:
            raise ValueError("Bar missing ticker")
        ticker = bar.ticker.upper()
        with self._lock:
            state = self._vwap_state[ticker]
            if bar.vwap > 0:
                # Feed-supplied VWAP takes precedence
                stored = bar
                # But still update our running state so a later bar
                # without supplied VWAP gets the right value.
                state.update(bar)
            else:
                vwap = state.update(bar)
                stored = replace(bar, ticker=ticker, vwap=vwap)
            self._bars[ticker].append(stored)
        return stored

    def reset_session(self, ticker: Optional[str] = None) -> None:
        """Drop bars and VWAP state for one ticker, or all of them.

        Call at session start. Day-trade VWAP is per-session — last
        session's data must NOT bleed into today's calculation."""
        with self._lock:
            if ticker is None:
                self._bars.clear()
                self._vwap_state.clear()
            else:
                self._bars.pop(ticker.upper(), None)
                self._vwap_state.pop(ticker.upper(), None)

    # ── Read accessors ────────────────────────────────────────────

    def has_bars(self, ticker: str) -> bool:
        with self._lock:
            return bool(self._bars.get(ticker.upper()))

    def bar_count(self, ticker: str) -> int:
        with self._lock:
            return len(self._bars.get(ticker.upper(), ()))

    def get_bars(
        self, ticker: str, n: Optional[int] = None,
    ) -> list[Bar]:
        """Return the last ``n`` bars (oldest first), or all if ``n=None``.

        Returns a copy — callers can mutate freely.
        """
        with self._lock:
            buf = self._bars.get(ticker.upper())
            if not buf:
                return []
            bars = list(buf)
        if n is None:
            return bars
        if n <= 0:
            return []
        return bars[-n:]

    def latest(self, ticker: str) -> Optional[Bar]:
        """Most recent bar for ``ticker``, or None."""
        with self._lock:
            buf = self._bars.get(ticker.upper())
            if not buf:
                return None
            return buf[-1]

    def vwap(self, ticker: str) -> float:
        """Current session VWAP for ``ticker``. 0 if no bars yet."""
        bar = self.latest(ticker)
        return bar.vwap if bar else 0.0

    # ── Indicator helpers ─────────────────────────────────────────

    def atr(self, ticker: str, period: int = 14) -> float:
        """Average True Range over the last ``period`` bars.

        Formula (Wilder's TR):
            TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
            ATR  = simple mean of last `period` TRs

        Returns 0.0 if fewer than ``period + 1`` bars are cached.
        """
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period!r}")
        bars = self.get_bars(ticker, n=period + 1)
        if len(bars) < period + 1:
            return 0.0
        trs: list[float] = []
        for i in range(1, len(bars)):
            curr = bars[i]
            prev = bars[i - 1]
            tr = max(
                curr.high - curr.low,
                abs(curr.high - prev.close),
                abs(curr.low - prev.close),
            )
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0

    def session_high_low(
        self, ticker: str, n: Optional[int] = None,
    ) -> tuple[float, float]:
        """Highest high and lowest low of the last ``n`` bars (or all).

        Returns ``(0.0, 0.0)`` if no bars cached.
        """
        bars = self.get_bars(ticker, n=n)
        if not bars:
            return 0.0, 0.0
        return (
            max(b.high for b in bars),
            min(b.low for b in bars),
        )

    def session_volume(
        self, ticker: str, n: Optional[int] = None,
    ) -> float:
        """Sum of bar volume over the last ``n`` bars (or full session)."""
        return sum(b.volume for b in self.get_bars(ticker, n=n))

    def cumulative_typical_price(self, ticker: str) -> float:
        """For diagnostics / external VWAP sanity checks."""
        with self._lock:
            return self._vwap_state.get(
                ticker.upper(), _SessionVwapState()
            ).cum_pv

    # ── Diagnostics ───────────────────────────────────────────────

    def tickers(self) -> list[str]:
        """All tickers currently cached."""
        with self._lock:
            return list(self._bars.keys())

    @property
    def ring_size(self) -> int:
        return self._ring_size
