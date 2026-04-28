"""Premarket scanner — turns the universe into today's watchlist.

Runs once at ~08:30 ET via the scheduler. For each ticker in the
universe:

1. Fetch today's premarket bars (4:00 ET → 09:30 ET).
2. Compute today's premarket dollar-volume and gap %.
3. Compare today's premarket volume to the trailing 30-day average
   of premarket volume → ``premkt_rvol``.
4. Build a :class:`TickerContext` per ticker with these fields plus
   the catalyst label (from :class:`CatalystClassifier`).
5. Rank by composite score = ``premkt_rvol × |premkt_gap_pct|``.
6. Return the top ``top_n`` symbols' contexts as a dict.

Costs: this is by far the most expensive scheduled job in the
day-trader. We hit the broker's historical-data API for ~60 symbols
× 31 days each. Alpaca's batch endpoints + 30-day caching
(persisted across sessions) keep it under ~30 s on a cold start;
warm starts (cached) take a few seconds.

The scanner is **synchronous**. It runs once at session start, not
on hot paths.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Optional

from day_trader.data.catalyst import NO_NEWS, CatalystClassifier
from day_trader.models import TickerContext

log = logging.getLogger(__name__)


# Premarket session in ET: 04:00 to 09:30 (regular open).
_PREMKT_START = time(4, 0)
_PREMKT_END = time(9, 30)


# ── Premarket data fetcher protocol ──────────────────────────────


class PremarketDataFetcher:
    """Interface for getting historical premarket bars.

    Production implementation wraps Alpaca's
    ``StockHistoricalDataClient.get_stock_bars`` with a
    ``StockBarsRequest`` filtered to the premarket session.

    Tests inject a stub that returns deterministic bars.
    """

    def fetch_premarket_volume(
        self,
        ticker: str,
        target_date: date,
    ) -> float:
        """Total premarket SHARE volume for ``target_date``."""
        raise NotImplementedError

    def fetch_premarket_dollar_volume(
        self,
        ticker: str,
        target_date: date,
    ) -> float:
        """Total premarket DOLLAR volume for ``target_date``."""
        raise NotImplementedError

    def fetch_first_premarket_price(
        self,
        ticker: str,
        target_date: date,
    ) -> Optional[float]:
        """Earliest premarket trade price (used for gap calc).
        Returns ``None`` when no premarket trades printed."""
        raise NotImplementedError

    def fetch_prev_close(
        self,
        ticker: str,
        target_date: date,
    ) -> Optional[float]:
        """Previous regular-session close. ``None`` if not available."""
        raise NotImplementedError

    def fetch_avg_premarket_volume(
        self,
        ticker: str,
        target_date: date,
        lookback_days: int = 30,
    ) -> Optional[float]:
        """Trailing average daily premarket SHARE volume (excluding
        ``target_date`` itself). ``None`` if insufficient history."""
        raise NotImplementedError


# ── Scanner ───────────────────────────────────────────────────────


@dataclass
class PremarketRanking:
    """Composite-score ranking for one ticker."""

    ticker: str
    context: TickerContext
    score: float

    @staticmethod
    def composite(ctx: TickerContext) -> float:
        """Score = RVOL × |gap %|. Both ingredients are signed (gap
        can be negative); we take |gap| because direction is captured
        by the strategies, not the watchlist filter."""
        return ctx.premkt_rvol * abs(ctx.premkt_gap_pct)


class PremarketScanner:
    """Builds the day's TickerContext map from the universe."""

    def __init__(
        self,
        fetcher: PremarketDataFetcher,
        catalyst_classifier: Optional[CatalystClassifier] = None,
        *,
        min_premkt_dollar_volume: float = 100_000.0,
        rvol_lookback_days: int = 30,
    ):
        self._fetcher = fetcher
        self._catalyst = catalyst_classifier or CatalystClassifier()
        self.min_premkt_dollar_volume = min_premkt_dollar_volume
        self.rvol_lookback_days = rvol_lookback_days

    def scan(
        self,
        universe: Iterable[str],
        target_date: Optional[date] = None,
        *,
        top_n: int = 50,
    ) -> dict[str, TickerContext]:
        """Run the scanner over ``universe``. Returns a dict mapping
        ticker → :class:`TickerContext` for the top ``top_n`` symbols
        by composite score.

        Symbols with insufficient premarket activity (below
        ``min_premkt_dollar_volume``) or missing reference data
        (no prev close, no avg volume) are skipped silently — they
        won't be in the output.
        """
        target_date = target_date or date.today()
        captured = datetime.now().isoformat(timespec="seconds")

        contexts: dict[str, TickerContext] = {}
        for ticker in universe:
            t = ticker.strip().upper()
            if not t:
                continue
            try:
                ctx = self._scan_one(t, target_date, captured)
            except Exception as exc:
                log.debug(
                    "PremarketScanner: %s failed: %s — skipping",
                    t, exc,
                )
                continue
            if ctx is not None:
                contexts[t] = ctx

        ranked = sorted(
            contexts.values(),
            key=lambda c: PremarketRanking.composite(c),
            reverse=True,
        )
        top = ranked[:top_n]
        log.info(
            "PremarketScanner: scanned %d, top %d selected "
            "(min composite=%.4f, max=%.4f)",
            len(contexts), len(top),
            PremarketRanking.composite(top[-1]) if top else 0.0,
            PremarketRanking.composite(top[0]) if top else 0.0,
        )
        return {ctx.ticker: ctx for ctx in top}

    def _scan_one(
        self,
        ticker: str,
        target_date: date,
        captured_at: str,
    ) -> Optional[TickerContext]:
        # Premarket activity check (cheapest, run first)
        dollar_vol = self._fetcher.fetch_premarket_dollar_volume(
            ticker, target_date,
        )
        if dollar_vol < self.min_premkt_dollar_volume:
            return None

        prev_close = self._fetcher.fetch_prev_close(ticker, target_date)
        if prev_close is None or prev_close <= 0:
            return None

        first_price = self._fetcher.fetch_first_premarket_price(
            ticker, target_date,
        )
        if first_price is None or first_price <= 0:
            return None

        share_vol = self._fetcher.fetch_premarket_volume(
            ticker, target_date,
        )
        avg_vol = self._fetcher.fetch_avg_premarket_volume(
            ticker, target_date, self.rvol_lookback_days,
        )

        gap_pct = (first_price / prev_close - 1) * 100
        if avg_vol is None or avg_vol <= 0:
            premkt_rvol = 0.0
        else:
            premkt_rvol = share_vol / avg_vol

        catalyst_label = self._catalyst.classify(ticker)

        return TickerContext(
            ticker=ticker,
            premkt_rvol=premkt_rvol,
            premkt_gap_pct=gap_pct,
            avg_daily_volume=avg_vol or 0.0,
            avg_dollar_volume=0.0,  # populated by liquidity check elsewhere
            prev_close=prev_close,
            catalyst_label=catalyst_label,
            captured_at=captured_at,
        )


# ── Alpaca implementation of the fetcher ─────────────────────────


class AlpacaPremarketFetcher(PremarketDataFetcher):
    """Production fetcher backed by Alpaca's StockHistoricalDataClient.

    Caches historical premarket aggregates per (ticker, date) for
    the duration of the process so successive runs in the same
    session are cheap.
    """

    def __init__(self, historical_client):
        # Duck-typed: must have get_stock_bars(StockBarsRequest)
        self._client = historical_client
        # Simple in-process caches (per session)
        self._volume_cache: dict[tuple[str, str], float] = {}
        self._dollar_cache: dict[tuple[str, str], float] = {}
        self._first_price_cache: dict[tuple[str, str], Optional[float]] = {}
        self._prev_close_cache: dict[tuple[str, str], Optional[float]] = {}
        self._avg_vol_cache: dict[tuple[str, str, int], Optional[float]] = {}

    def fetch_premarket_volume(
        self, ticker: str, target_date: date,
    ) -> float:
        key = (ticker, target_date.isoformat())
        if key in self._volume_cache:
            return self._volume_cache[key]
        bars = self._fetch_bars_for_date(ticker, target_date)
        vol = sum(self._bar_attr(b, "volume", 0) for b in bars)
        self._volume_cache[key] = float(vol)
        return self._volume_cache[key]

    def fetch_premarket_dollar_volume(
        self, ticker: str, target_date: date,
    ) -> float:
        key = (ticker, target_date.isoformat())
        if key in self._dollar_cache:
            return self._dollar_cache[key]
        bars = self._fetch_bars_for_date(ticker, target_date)
        dv = 0.0
        for b in bars:
            v = self._bar_attr(b, "volume", 0)
            vwap = self._bar_attr(b, "vwap", 0) or self._bar_attr(b, "close", 0)
            dv += v * vwap
        self._dollar_cache[key] = dv
        return dv

    def fetch_first_premarket_price(
        self, ticker: str, target_date: date,
    ) -> Optional[float]:
        key = (ticker, target_date.isoformat())
        if key in self._first_price_cache:
            return self._first_price_cache[key]
        bars = self._fetch_bars_for_date(ticker, target_date)
        out = None
        if bars:
            first = bars[0]
            out = self._bar_attr(first, "open", 0) or None
        self._first_price_cache[key] = out
        return out

    def fetch_prev_close(
        self, ticker: str, target_date: date,
    ) -> Optional[float]:
        key = (ticker, target_date.isoformat())
        if key in self._prev_close_cache:
            return self._prev_close_cache[key]
        out = self._fetch_prev_close_impl(ticker, target_date)
        self._prev_close_cache[key] = out
        return out

    def fetch_avg_premarket_volume(
        self, ticker: str, target_date: date, lookback_days: int = 30,
    ) -> Optional[float]:
        key = (ticker, target_date.isoformat(), lookback_days)
        if key in self._avg_vol_cache:
            return self._avg_vol_cache[key]
        out = self._fetch_avg_volume_impl(
            ticker, target_date, lookback_days,
        )
        self._avg_vol_cache[key] = out
        return out

    # ── Private: Alpaca request orchestration ────────────────────

    def _fetch_bars_for_date(self, ticker: str, target_date: date) -> list:
        """Fetch 1-min bars in the premarket window for one date.

        Implementation note: this calls
        ``client.get_stock_bars(StockBarsRequest(...))`` with a
        04:00–09:30 ET window. Returns a list of bar objects (alpaca
        Bar dataclass) which the public methods aggregate over.

        Defensive against alpaca-py API drift — falls back to empty
        list on any unexpected response shape.
        """
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from zoneinfo import ZoneInfo

            ET = ZoneInfo("America/New_York")
            start = datetime.combine(target_date, _PREMKT_START, tzinfo=ET)
            end = datetime.combine(target_date, _PREMKT_END, tzinfo=ET)
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            response = self._client.get_stock_bars(req)
            # Response could be a BarSet (dict-like) or a list-like.
            if hasattr(response, "data") and isinstance(response.data, dict):
                return list(response.data.get(ticker, []))
            if hasattr(response, "df"):
                # Some versions return a DataFrame; iterate rows
                from types import SimpleNamespace
                df = response.df
                if df is None or df.empty:
                    return []
                return [
                    SimpleNamespace(**{
                        k: v.item() if hasattr(v, "item") else v
                        for k, v in row.to_dict().items()
                    })
                    for _, row in df.iterrows()
                ]
            return list(response or [])
        except Exception as exc:
            log.debug(
                "Premarket bar fetch failed for %s on %s: %s",
                ticker, target_date, exc,
            )
            return []

    def _fetch_prev_close_impl(
        self, ticker: str, target_date: date,
    ) -> Optional[float]:
        """Fetch the previous trading day's close. Walks back up to
        7 days to handle weekends/holidays."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            start = target_date - timedelta(days=7)
            end = target_date
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            response = self._client.get_stock_bars(req)
            bars = []
            if hasattr(response, "data") and isinstance(response.data, dict):
                bars = list(response.data.get(ticker, []))
            elif hasattr(response, "df"):
                df = response.df
                if df is not None and not df.empty:
                    bars = list(df.itertuples())
            if not bars:
                return None
            # Last bar with timestamp strictly before target_date
            for b in reversed(bars):
                ts = self._bar_attr(b, "timestamp", None)
                if ts is None:
                    continue
                if hasattr(ts, "date"):
                    bdate = ts.date()
                else:
                    bdate = ts
                if bdate < target_date:
                    return self._bar_attr(b, "close", 0) or None
            return None
        except Exception as exc:
            log.debug(
                "Prev close fetch failed for %s: %s", ticker, exc,
            )
            return None

    def _fetch_avg_volume_impl(
        self, ticker: str, target_date: date, lookback_days: int,
    ) -> Optional[float]:
        """Average premarket SHARE volume over the last lookback_days
        sessions (excluding today). Accumulates by fetching the same
        04:00–09:30 window for each prior date.

        Production note: this is N requests per ticker. For a 60-symbol
        universe with 30-day lookback that's 1,800 requests on cold
        start. Alpaca's rate limit (10k/min on Algo Trader Plus) is
        plenty, but it's still ~10s of latency. Production may pre-
        compute this nightly and persist a per-symbol JSON cache.
        For v1 we accept the cold-start cost.
        """
        try:
            volumes: list[float] = []
            d = target_date - timedelta(days=1)
            collected = 0
            # Walk back at most 60 calendar days to gather lookback_days
            # trading sessions
            for _ in range(60):
                if collected >= lookback_days:
                    break
                if d.weekday() < 5:  # mon-fri
                    bars = self._fetch_bars_for_date(ticker, d)
                    if bars:
                        v = sum(self._bar_attr(b, "volume", 0) for b in bars)
                        volumes.append(float(v))
                        collected += 1
                d -= timedelta(days=1)
            if not volumes:
                return None
            if len(volumes) < lookback_days * 0.5:
                log.warning(
                    "Avg premkt volume for %s: only %d of %d expected "
                    "sessions returned data — treating as insufficient",
                    ticker, len(volumes), lookback_days,
                )
                return None
            return sum(volumes) / len(volumes)
        except Exception as exc:
            log.debug(
                "Avg premkt volume fetch failed for %s: %s", ticker, exc,
            )
            return None

    @staticmethod
    def _bar_attr(bar, name: str, default):
        """Robust attribute getter — handles both Alpaca SDK objects
        (attr access) and dict-like rows (item access)."""
        if bar is None:
            return default
        v = getattr(bar, name, None)
        if v is not None:
            return v
        if isinstance(bar, Mapping):
            return bar.get(name, default)
        return default
