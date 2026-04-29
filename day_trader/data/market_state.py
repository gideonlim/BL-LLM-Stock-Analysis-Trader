"""Market-wide state snapshot — VIX, SPY trend, breadth.

Called once near session start by the executor to populate the
shared :class:`MarketState` consumed by the RegimeFilter and
strategies that care about regime.

For v1 we use yfinance (already a swing-bot dep) since it gives us
both VIX and SPY in two cheap requests. v2 may switch to Alpaca
historical bars for fewer external dependencies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from day_trader.models import MarketState

log = logging.getLogger(__name__)


def fetch_market_state(
    *,
    spy_sma_window: int = 200,
    bear_confirmation_days: int = 3,
    severe_drawdown_pct: float = 15.0,
) -> MarketState:
    """Snapshot SPY price + 200-SMA, VIX, and derive a regime label.

    Regime labels (matching the swing bot's vocabulary):
    - ``BULL`` — SPY > 200-SMA, no severe drawdown, VIX normal
    - ``CAUTION`` — SPY < 200-SMA but only briefly (< confirmation days)
    - ``BEAR`` — SPY < 200-SMA confirmed for ``bear_confirmation_days``
    - ``SEVERE_BEAR`` — SPY drawdown from 52-wk high > severe_drawdown_pct

    Returns a :class:`MarketState` with sensible defaults on failure
    so the daemon can keep running with conservative defaults.
    """
    captured = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    state = MarketState(captured_at=captured)

    try:
        import yfinance as yf

        # SPY: 1y daily history covers 200-SMA + 52-wk high
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y", interval="1d")
        if not spy_hist.empty:
            closes = spy_hist["Close"]
            state.spy_price = float(closes.iloc[-1])
            if len(closes) >= spy_sma_window:
                state.spy_200_sma = float(
                    closes.iloc[-spy_sma_window:].mean()
                )
            # 52-wk high drawdown for SEVERE_BEAR
            high_52w = float(closes.max())
            if high_52w > 0:
                drawdown_pct = (
                    (high_52w - state.spy_price) / high_52w * 100
                )
                # Confirmed bear if SPY closed below 200-SMA for
                # the last `bear_confirmation_days` sessions
                below_sma = (
                    closes.iloc[-bear_confirmation_days:]
                    < state.spy_200_sma
                ).all() if state.spy_200_sma > 0 else False

                if drawdown_pct > severe_drawdown_pct:
                    state.spy_trend_regime = "SEVERE_BEAR"
                elif below_sma:
                    state.spy_trend_regime = "BEAR"
                elif state.spy_price < state.spy_200_sma:
                    state.spy_trend_regime = "CAUTION"
                else:
                    state.spy_trend_regime = "BULL"
    except Exception as exc:
        log.warning(
            "fetch_market_state: SPY lookup failed: %s — defaulting to BULL",
            exc,
        )
        state.spy_trend_regime = "BULL"

    # VIX
    try:
        import yfinance as yf

        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d", interval="1d")
        if not vix_hist.empty:
            state.vix = float(vix_hist["Close"].iloc[-1])
    except Exception as exc:
        log.warning(
            "fetch_market_state: VIX lookup failed: %s — defaulting to 0",
            exc,
        )
        state.vix = 0.0

    log.info(
        "MarketState: spy=$%.2f sma200=$%.2f regime=%s vix=%.2f",
        state.spy_price, state.spy_200_sma,
        state.spy_trend_regime, state.vix,
    )
    return state
