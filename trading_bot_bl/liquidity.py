"""
Average Daily Volume (ADV) liquidity filter — blocks entries in illiquid stocks.

Illiquid stocks have wide bid-ask spreads and shallow order books, meaning:
- Entry slippage can eat a significant chunk of expected alpha
- Exit under stress (stop-loss hit, earnings surprise) may gap through
  your stop price, resulting in much larger losses than modeled
- ATR-based position sizing underestimates true risk because ATR reflects
  midpoint moves, not the spread you'd actually cross

This module fetches the 20-day average daily volume from yfinance and
checks whether the stock trades enough shares that our position size is
a small fraction of daily flow.

Data source: yfinance .info["averageDailyVolume10Day"] and
.info["averageVolume"] (free, no API key needed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

log = logging.getLogger(__name__)

# Module-level cache: {ticker: LiquidityInfo} — one yfinance call per
# ticker per execution session.
_liquidity_cache: Dict[str, "LiquidityInfo"] = {}


@dataclass(frozen=True)
class LiquidityInfo:
    """Liquidity check result for a single ticker."""

    ticker: str
    avg_daily_volume: Optional[int] = None
    avg_daily_dollar_volume: Optional[float] = None
    position_pct_of_adv: Optional[float] = None
    passes: bool = True
    rejection_reason: str = ""


def fetch_avg_daily_volume(
    ticker: str,
) -> tuple[Optional[int], Optional[float]]:
    """
    Fetch average daily volume and approximate dollar volume.

    Returns (share_volume, dollar_volume) or (None, None) on error.
    Dollar volume = share_volume × current price (approximate).
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.info or {}

        # Prefer 10-day ADV (more recent), fall back to longer-term
        adv = info.get("averageDailyVolume10Day") or info.get(
            "averageVolume"
        )

        if adv is None or adv <= 0:
            return None, None

        adv = int(adv)

        # Approximate dollar volume using current price
        price = info.get("currentPrice") or info.get(
            "regularMarketPrice"
        ) or info.get("previousClose")

        dollar_vol = None
        if price and price > 0:
            dollar_vol = float(adv) * float(price)

        return adv, dollar_vol

    except Exception as e:
        log.debug(f"ADV lookup failed for {ticker}: {e}")
        return None, None


def check_liquidity(
    ticker: str,
    position_notional: float,
    *,
    min_adv_shares: int = 500_000,
    min_dollar_volume: float = 5_000_000.0,
    max_participation_pct: float = 1.0,
    use_cache: bool = True,
) -> LiquidityInfo:
    """
    Check if a ticker has sufficient liquidity for the proposed trade.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    position_notional : float
        Dollar amount of the proposed position.
    min_adv_shares : int
        Minimum average daily volume in shares (default: 500K).
        Stocks below this are considered illiquid.
    min_dollar_volume : float
        Minimum average daily dollar volume (default: $5M).
        Catches low-priced stocks with high share volume but
        thin dollar flow.
    max_participation_pct : float
        Maximum position as % of daily dollar volume (default: 1%).
        Ensures we're a small fraction of daily flow so our
        entry/exit doesn't move the market.
    use_cache : bool
        Cache yfinance results per session (default: True).

    Returns
    -------
    LiquidityInfo with pass/fail and reason.
    """
    # Check cache
    if use_cache and ticker in _liquidity_cache:
        cached = _liquidity_cache[ticker]
        # Re-evaluate with current position size (volume stays same)
        if cached.avg_daily_volume is not None:
            return _evaluate_liquidity(
                ticker,
                cached.avg_daily_volume,
                cached.avg_daily_dollar_volume,
                position_notional,
                min_adv_shares,
                min_dollar_volume,
                max_participation_pct,
            )
        return cached

    # Fetch from yfinance
    adv_shares, adv_dollars = fetch_avg_daily_volume(ticker)

    if adv_shares is None:
        # Graceful degradation: if we can't get data, let it through
        info = LiquidityInfo(ticker=ticker)
        if use_cache:
            _liquidity_cache[ticker] = info
        return info

    # Cache raw volume data
    if use_cache:
        _liquidity_cache[ticker] = LiquidityInfo(
            ticker=ticker,
            avg_daily_volume=adv_shares,
            avg_daily_dollar_volume=adv_dollars,
        )

    return _evaluate_liquidity(
        ticker,
        adv_shares,
        adv_dollars,
        position_notional,
        min_adv_shares,
        min_dollar_volume,
        max_participation_pct,
    )


def _evaluate_liquidity(
    ticker: str,
    adv_shares: int,
    adv_dollars: Optional[float],
    position_notional: float,
    min_adv_shares: int,
    min_dollar_volume: float,
    max_participation_pct: float,
) -> LiquidityInfo:
    """Evaluate whether liquidity meets all thresholds."""
    participation_pct = None
    if adv_dollars and adv_dollars > 0:
        participation_pct = (position_notional / adv_dollars) * 100

    # Check 1: Minimum share volume
    if adv_shares < min_adv_shares:
        return LiquidityInfo(
            ticker=ticker,
            avg_daily_volume=adv_shares,
            avg_daily_dollar_volume=adv_dollars,
            position_pct_of_adv=participation_pct,
            passes=False,
            rejection_reason=(
                f"ADV {adv_shares:,} shares below minimum "
                f"{min_adv_shares:,}"
            ),
        )

    # Check 2: Minimum dollar volume
    if (
        adv_dollars is not None
        and adv_dollars < min_dollar_volume
    ):
        return LiquidityInfo(
            ticker=ticker,
            avg_daily_volume=adv_shares,
            avg_daily_dollar_volume=adv_dollars,
            position_pct_of_adv=participation_pct,
            passes=False,
            rejection_reason=(
                f"Daily dollar volume ${adv_dollars:,.0f} below "
                f"minimum ${min_dollar_volume:,.0f}"
            ),
        )

    # Check 3: Position participation rate
    if (
        participation_pct is not None
        and participation_pct > max_participation_pct
    ):
        return LiquidityInfo(
            ticker=ticker,
            avg_daily_volume=adv_shares,
            avg_daily_dollar_volume=adv_dollars,
            position_pct_of_adv=participation_pct,
            passes=False,
            rejection_reason=(
                f"Position ${position_notional:,.0f} is "
                f"{participation_pct:.2f}% of daily dollar volume "
                f"(max {max_participation_pct:.1f}%)"
            ),
        )

    return LiquidityInfo(
        ticker=ticker,
        avg_daily_volume=adv_shares,
        avg_daily_dollar_volume=adv_dollars,
        position_pct_of_adv=participation_pct,
        passes=True,
    )


def batch_check_liquidity(
    tickers_with_notional: Dict[str, float],
    *,
    min_adv_shares: int = 500_000,
    min_dollar_volume: float = 5_000_000.0,
    max_participation_pct: float = 1.0,
) -> Dict[str, LiquidityInfo]:
    """
    Check liquidity for multiple tickers.

    Parameters
    ----------
    tickers_with_notional : dict
        {ticker: proposed_notional_dollars}

    Returns {ticker: LiquidityInfo} for all tickers.
    """
    results = {}
    for ticker, notional in tickers_with_notional.items():
        results[ticker] = check_liquidity(
            ticker,
            notional,
            min_adv_shares=min_adv_shares,
            min_dollar_volume=min_dollar_volume,
            max_participation_pct=max_participation_pct,
        )
    return results


def clear_cache() -> None:
    """Clear the module-level liquidity cache."""
    _liquidity_cache.clear()
