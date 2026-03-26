"""Oil spike detection — composite score boost for fertilizer/ag tickers.

When crude oil (USO ETF proxy) posts a 10%+ gain over a 5-day rolling
window, fertilizer stocks like MOS and CF have historically returned
+5-6% over the next 20 trading days (vs +1% baseline), with a 79%
win rate.  The edge survives beta-adjustment against SPY and XLB, and
persists even when 2022 war-driven events are excluded.

This module:
1. Fetches the last ~30 trading days of USO prices via yfinance.
2. Checks if a 10%+ weekly spike occurred in the last N days.
3. Returns a linearly-decaying boost (peak → 0) that the risk
   manager adds to composite_score for eligible tickers only.

If oil spike detection is disabled (the default) or USO data cannot
be fetched, *zero* code paths in the trading bot are affected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Tickers that receive the composite-score boost after an oil spike.
# Nitrogen fertilizer + phosphate producers most exposed to the
# oil→geopolitical→food-security demand channel.
DEFAULT_OIL_SPIKE_TICKERS: tuple[str, ...] = ("MOS", "CF")

# USO ticker used as the crude oil proxy.
_OIL_TICKER = "USO"


@dataclass(frozen=True)
class OilSpikeState:
    """Immutable snapshot of the current oil-spike signal."""

    active: bool = False
    # Trading days since the spike was detected (0 = today).
    days_since_spike: int = 0
    # Raw 5-day USO return that triggered the spike.
    spike_magnitude: float = 0.0
    # Current boost value after linear decay.
    boost: float = 0.0

    def __repr__(self) -> str:
        if not self.active:
            return "OilSpikeState(inactive)"
        return (
            f"OilSpikeState(active, {self.days_since_spike}d ago, "
            f"USO +{self.spike_magnitude:.1%}, boost={self.boost:+.1f})"
        )


def detect_oil_spike(
    *,
    peak_boost: float = 8.0,
    window_days: int = 20,
    spike_threshold: float = 0.10,
    lookback_trading_days: int = 5,
) -> OilSpikeState:
    """Detect whether an oil spike is currently active.

    Fetches ~40 trading days of USO prices via yfinance, computes
    5-day rolling returns, and checks if a 10%+ spike occurred
    within the last *window_days* trading days.

    Parameters
    ----------
    peak_boost:
        Max composite-score boost applied on the day the spike is
        detected.  Decays linearly to 0 over *window_days*.
    window_days:
        Number of trading days over which the boost decays to 0.
    spike_threshold:
        Minimum 5-day USO return to qualify as a spike (0.10 = 10%).
    lookback_trading_days:
        Rolling window for computing the USO return (default 5 = 1 week).

    Returns
    -------
    OilSpikeState
        Contains .active, .days_since_spike, .spike_magnitude, .boost.
        If no spike is active or data cannot be fetched, returns an
        inactive state with boost=0.
    """
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError:
        log.warning(
            "yfinance not installed — oil spike detection disabled"
        )
        return OilSpikeState()

    # Fetch enough history to cover the full decay window + rolling
    # lookback + some buffer for weekends/holidays.
    fetch_days = window_days + lookback_trading_days + 20
    try:
        import datetime as dt

        end = dt.date.today()
        start = end - dt.timedelta(days=int(fetch_days * 1.6))
        df = yf.download(
            _OIL_TICKER,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty or len(df) < lookback_trading_days + 2:
            log.warning(
                "Insufficient USO data — oil spike detection skipped"
            )
            return OilSpikeState()

        # Flatten multi-index columns if yfinance returns them
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

    except Exception as exc:
        log.warning(f"USO fetch failed — oil spike detection skipped: {exc}")
        return OilSpikeState()

    # Compute 5-day rolling returns
    close = df["Close" if "Close" in df.columns else "close"]
    rolling_ret = close.pct_change(lookback_trading_days)

    # Find the most recent spike within the decay window.
    # Walk backwards from the latest date.
    latest_idx = len(rolling_ret) - 1
    for days_ago in range(window_days + 1):
        idx = latest_idx - days_ago
        if idx < 0:
            break
        ret = rolling_ret.iloc[idx]
        if ret >= spike_threshold:
            decay_frac = 1.0 - (days_ago / window_days)
            boost = round(peak_boost * decay_frac, 2)
            state = OilSpikeState(
                active=True,
                days_since_spike=days_ago,
                spike_magnitude=float(ret),
                boost=boost,
            )
            log.info(f"Oil spike detected: {state}")
            return state

    return OilSpikeState()


def get_boost_for_ticker(
    state: OilSpikeState,
    ticker: str,
    eligible_tickers: tuple[str, ...] | list[str] = DEFAULT_OIL_SPIKE_TICKERS,
) -> float:
    """Return the composite-score boost for a specific ticker.

    Returns 0.0 if:
    - No spike is active
    - The ticker is not in the eligible set
    - The boost has fully decayed

    This is the only function the risk manager needs to call.
    """
    if not state.active:
        return 0.0
    if ticker.upper() not in {t.upper() for t in eligible_tickers}:
        return 0.0
    return state.boost
