"""Opening Range Breakout + VWAP filter + RVOL strategy.

Inspired by Zarattini, Barbon, Aziz (2024) "A Profitable Day Trading
Strategy For The U.S. Equity Market" — adapted to retail constraints
(real fills, ATR-bracketed exits, single-shot per ticker).

Setup:

- The "opening range" is the first ``or_minutes`` of the session
  (default 5 → 09:30:00–09:34:59 inclusive on a regular day).
- After ``or_minutes`` have elapsed, scan tickers whose premarket
  RVOL is elevated (filter check; this strategy assumes the filter
  already passed but also checks defensively).
- LONG entry when last close > opening-range high AND last close >
  session VWAP. (Symmetric short logic available but not enabled
  by default — the day-trader is configured long-only at v1.)
- Stop = max(opening-range low, entry − 1×ATR). Whichever is tighter
  (= higher for a long, smaller risk-per-share).
- Take profit = entry + 2R (where R = entry − stop).
- Time stop: ``manage()`` force-flat at ``time_stop_minutes`` after
  open (default 90 → 11:00 ET on a regular day).

One-shot per ticker per session: once the strategy fires for
``AAPL``, it won't re-emit even if price retraces and breaks out
again. This is the published behavior in the source paper.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from day_trader.calendar import NyseSession
from day_trader.data.cache import BarCache
from day_trader.models import (
    Bar,
    DayTradeSignal,
    ExitIntent,
    MarketState,
    OpenDayTrade,
    TickerContext,
)
from day_trader.strategies.base import DayTradeStrategy

log = logging.getLogger(__name__)


class OrbVwapStrategy(DayTradeStrategy):
    """5-min ORB + VWAP confirmation + premkt RVOL gate."""

    name = "orb_vwap"
    required_catalyst_label = None  # any catalyst

    def __init__(
        self,
        *,
        or_minutes: int = 5,
        atr_period: int = 14,
        rr_target: float = 2.0,
        min_premkt_rvol: float = 2.0,
        time_stop_minutes: int = 90,
    ):
        super().__init__()
        if or_minutes < 1:
            raise ValueError(f"or_minutes must be >= 1, got {or_minutes!r}")
        if rr_target <= 0:
            raise ValueError(f"rr_target must be > 0, got {rr_target!r}")
        self.or_minutes = or_minutes
        self.atr_period = atr_period
        self.rr_target = rr_target
        self.min_premkt_rvol = min_premkt_rvol
        self.time_stop_minutes = time_stop_minutes

    # ── Scan ──────────────────────────────────────────────────────

    def scan_ticker(
        self,
        ticker: str,
        bar_cache: BarCache,
        ticker_context: TickerContext,
        market_state: MarketState,
        now_et: datetime,
        session: NyseSession,
    ) -> Optional[DayTradeSignal]:
        # 1. One-shot guard
        if self.already_fired(ticker):
            return None

        # 2. Time gate — must be at or after the OR window close
        or_end = session.open_plus(self.or_minutes)
        if now_et < or_end:
            return None

        # 3. Premkt RVOL gate (defence-in-depth — the filter pipeline
        # also enforces this, but we shouldn't emit signals we know
        # will be rejected)
        if ticker_context.premkt_rvol < self.min_premkt_rvol:
            return None

        # 4. Need enough cached bars: the OR window + at least one
        # post-OR bar to evaluate the breakout.
        bars = bar_cache.get_bars(ticker)
        or_bars = self._select_or_bars(bars, session, self.or_minutes)
        if not or_bars:
            return None
        post_or_bars = self._select_post_or_bars(
            bars, session, self.or_minutes,
        )
        if not post_or_bars:
            return None

        # 5. Compute opening range
        or_high = max(b.high for b in or_bars)
        or_low = min(b.low for b in or_bars)
        if or_high <= or_low:
            return None  # corrupt OR

        # 6. Latest bar must close above OR-high AND above VWAP
        latest = post_or_bars[-1]
        vwap = latest.vwap
        if vwap <= 0:
            return None  # VWAP not ready
        if not (latest.close > or_high and latest.close > vwap):
            return None

        # 7. Stop = max(OR-low, entry − 1×ATR), whichever tighter
        atr = bar_cache.atr(ticker, period=self.atr_period)
        if atr <= 0:
            # Not enough bars for ATR yet — fall back to OR-low only
            stop = or_low
        else:
            atr_stop = latest.close - atr
            stop = max(or_low, atr_stop)

        if stop <= 0 or stop >= latest.close:
            return None  # invalid stop

        risk = latest.close - stop
        if risk <= 0:
            return None
        take_profit = latest.close + self.rr_target * risk

        # 8. Mark fired and emit
        self.mark_fired(ticker)
        log.info(
            "%s: ORB long signal on %s @ %.2f (OR-high=%.2f, "
            "VWAP=%.2f, stop=%.2f, TP=%.2f, RVOL=%.2f)",
            self.name, ticker, latest.close, or_high, vwap,
            stop, take_profit, ticker_context.premkt_rvol,
        )
        return DayTradeSignal(
            ticker=ticker,
            strategy=self.name,
            side="buy",
            signal_price=latest.close,
            stop_loss_price=stop,
            take_profit_price=take_profit,
            atr=atr,
            rvol=ticker_context.premkt_rvol,
            catalyst_label=ticker_context.catalyst_label,
            notes=(
                f"or_high={or_high:.2f} or_low={or_low:.2f} "
                f"vwap={vwap:.2f}"
            ),
        )

    # ── Manage ────────────────────────────────────────────────────

    def manage(
        self,
        position: OpenDayTrade,
        bar_cache: BarCache,
        now_et: datetime,
        session: NyseSession,
    ) -> Optional[ExitIntent]:
        # Time stop: force flat at open + time_stop_minutes
        time_limit = session.open_plus(self.time_stop_minutes)
        if now_et >= time_limit:
            return ExitIntent(
                ticker=position.ticker,
                reason="time_stop",
            )
        return None

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _select_or_bars(
        bars: list[Bar],
        session: NyseSession,
        or_minutes: int,
    ) -> list[Bar]:
        """Bars whose timestamp falls inside the OR window
        ``[session.open_et, session.open_et + or_minutes)``.

        Bar timestamps are expected to be the START of the bar
        interval (Alpaca convention)."""
        if not bars:
            return []
        or_start = session.open_et
        or_end = session.open_plus(or_minutes)
        out: list[Bar] = []
        for b in bars:
            ts = b.timestamp
            if ts.tzinfo is None:
                # Defensive: treat naive timestamps as ET
                continue
            ts_et = ts.astimezone(or_start.tzinfo)
            if or_start <= ts_et < or_end:
                out.append(b)
        return out

    @staticmethod
    def _select_post_or_bars(
        bars: list[Bar],
        session: NyseSession,
        or_minutes: int,
    ) -> list[Bar]:
        """Bars at or after the OR-end timestamp."""
        if not bars:
            return []
        or_end = session.open_plus(or_minutes)
        out: list[Bar] = []
        for b in bars:
            ts = b.timestamp
            if ts.tzinfo is None:
                continue
            ts_et = ts.astimezone(or_end.tzinfo)
            if ts_et >= or_end:
                out.append(b)
        return out
