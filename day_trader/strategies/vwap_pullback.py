"""VWAP pullback continuation.

Trend-continuation entry on tickers already extended off VWAP.
Setup:

- Trending up: at least one higher-high in the last ``trend_lookback``
  bars (default 30) compared to the prior window.
- Price > session VWAP.
- RVOL ≥ ``min_rvol`` (default 1.5).
- Pullback: most recent bar's LOW touched within ``touch_atr_mult``
  × ATR of session VWAP (default 0.25).
- Reversal: latest bar closes ABOVE its open (a bullish reversal
  candle on the pullback).
- Stop = VWAP − ``stop_atr_mult`` × ATR (default 0.5).
- Take profit = prior swing high OR 2R, whichever closer (we want
  the conservative target — gets us out faster on lower-conviction
  setups).

Unlike ORB, this strategy can fire multiple times per ticker per
session because pullbacks recur. The ``SymbolLockFilter`` and
``CooldownFilter`` handle re-entry hygiene.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from day_trader.calendar import NyseSession
from day_trader.data.cache import BarCache
from day_trader.models import (
    DayTradeSignal,
    ExitIntent,
    MarketState,
    OpenDayTrade,
    TickerContext,
)
from day_trader.strategies.base import DayTradeStrategy

log = logging.getLogger(__name__)


class VwapPullbackStrategy(DayTradeStrategy):
    """Pullback-to-VWAP continuation in established intraday uptrends."""

    name = "vwap_pullback"
    required_catalyst_label = None

    def __init__(
        self,
        *,
        atr_period: int = 14,
        trend_lookback: int = 30,
        min_rvol: float = 1.5,
        touch_atr_mult: float = 0.25,
        stop_atr_mult: float = 0.5,
        rr_target: float = 2.0,
        warmup_minutes: int = 30,
    ):
        super().__init__()
        self.atr_period = atr_period
        self.trend_lookback = trend_lookback
        self.min_rvol = min_rvol
        self.touch_atr_mult = touch_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.rr_target = rr_target
        # Don't fire in the first N minutes — too noisy + ORB is
        # supposed to handle that window.
        self.warmup_minutes = warmup_minutes

    def scan_ticker(
        self,
        ticker: str,
        bar_cache: BarCache,
        ticker_context: TickerContext,
        market_state: MarketState,
        now_et: datetime,
        session: NyseSession,
    ) -> Optional[DayTradeSignal]:
        # 1. Warm-up gate — first warmup_minutes belong to ORB
        if now_et < session.open_plus(self.warmup_minutes):
            return None

        # 2. Need sufficient bars for trend detection + ATR
        bars = bar_cache.get_bars(
            ticker, n=max(self.trend_lookback, self.atr_period + 1) + 1,
        )
        if len(bars) < self.trend_lookback + 1:
            return None

        latest = bars[-1]
        vwap = latest.vwap
        if vwap <= 0:
            return None

        # 3. Trend check: at least one higher-high in the last
        # trend_lookback bars vs the prior window of equal size.
        lookback = self.trend_lookback
        recent = bars[-lookback:]
        prior_highs = [b.high for b in bars[-2 * lookback:-lookback]] if len(bars) >= 2 * lookback else []
        if prior_highs:
            recent_max = max(b.high for b in recent)
            prior_max = max(prior_highs)
            if recent_max <= prior_max:
                return None
        else:
            # Not enough history for prior window; require a simpler
            # check — most recent half makes higher highs vs prior half.
            mid = lookback // 2
            recent_max = max(b.high for b in recent[mid:])
            earlier_max = max(b.high for b in recent[:mid])
            if recent_max <= earlier_max:
                return None

        # 4. Price above VWAP
        if latest.close <= vwap:
            return None

        # 5. RVOL gate — uses ticker context's premarket RVOL as a
        # proxy. (Live intraday RVOL via BarCache is checked in
        # the filter pipeline, not here.)
        if ticker_context.premkt_rvol < self.min_rvol:
            return None

        # 6. Pullback touch + reversal candle
        atr = bar_cache.atr(ticker, period=self.atr_period)
        if atr <= 0:
            return None
        touch_distance = self.touch_atr_mult * atr
        # Most recent bar's low must come within ``touch_distance``
        # of VWAP (i.e. the pullback reached close to VWAP)
        if latest.low > vwap + touch_distance:
            return None
        # And the reversal: bar must close above its open
        if latest.close <= latest.open:
            return None

        # 7. Stop and take profit
        stop = vwap - self.stop_atr_mult * atr
        if stop <= 0 or stop >= latest.close:
            return None
        risk = latest.close - stop
        if risk <= 0:
            return None
        rr_target_price = latest.close + self.rr_target * risk
        # Use prior swing high if it gives a CLOSER target
        prior_swing_high = max(b.high for b in recent[:-1]) if len(recent) > 1 else 0.0
        if prior_swing_high > latest.close:
            take_profit = min(rr_target_price, prior_swing_high)
        else:
            take_profit = rr_target_price

        log.info(
            "%s: pullback signal on %s @ %.2f (vwap=%.2f, atr=%.2f, "
            "stop=%.2f, tp=%.2f, rvol=%.2f)",
            self.name, ticker, latest.close, vwap, atr,
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
                f"vwap={vwap:.2f} touch_dist={touch_distance:.2f} "
                f"swing_high={prior_swing_high:.2f}"
            ),
        )

    def manage(
        self,
        position: OpenDayTrade,
        bar_cache: BarCache,
        now_et: datetime,
        session: NyseSession,
    ) -> Optional[ExitIntent]:
        # No strategy-level time stop beyond the executor's force-flat
        # at session close. Bracket SL/TP handle exit.
        return None
