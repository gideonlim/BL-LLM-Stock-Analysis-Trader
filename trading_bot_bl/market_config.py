"""Market configuration — carries every previously-hardcoded US constant.

Each instance of the trading bot runs with a single MarketConfig that
defines timezone, market hours, trading calendar, regime benchmarks,
cost model, and broker type.  Pre-built presets are provided for US,
LSE (London), and TSE (Tokyo).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Literal


@dataclass(frozen=True)
class MarketConfig:
    """Immutable configuration for a single market/exchange."""

    # ── Identity ─────────────────────────────────────────
    market_id: str                          # "US" | "LSE" | "TSE"

    # ── Schedule ─────────────────────────────────────────
    timezone: str                           # IANA tz name
    market_open: time                       # regular session open (local)
    market_close: time                      # regular session close (local)
    lunch_break: tuple[time, time] | None   # split-session break (TSE)

    # ── Annualization ────────────────────────────────────
    trading_days_per_year: int              # 252 US, 252 UK, 245 JP
    risk_free_rate: float                   # annualized decimal

    # ── Regime ───────────────────────────────────────────
    regime_benchmark_ticker: str            # "SPY" / "^FTSE" / "^N225"
    regime_vol_ticker: str | None           # "^VIX" / None (trend-only)

    # ── Calendar ─────────────────────────────────────────
    holiday_calendar_name: str              # pandas_market_calendars name

    # ── Currency ─────────────────────────────────────────
    currency: str                           # "USD" / "GBP" / "JPY"

    # ── Broker ───────────────────────────────────────────
    broker_type: str                        # "alpaca" | "ibkr"

    # ── Feature toggles ─────────────────────────────────
    earnings_blackout_enabled: bool = True
    news_sentiment_weight: float = 1.0

    # ── IBKR position-classification (ignored for Alpaca) ─
    ibkr_exchanges: frozenset[str] = field(
        default_factory=frozenset
    )
    ibkr_primary_exchanges: frozenset[str] = field(
        default_factory=frozenset
    )
    ibkr_currency: str = ""
    max_equity_allocation: float = 1.0


# ── Presets ───────────────────────────────────────────────


US = MarketConfig(
    market_id="US",
    timezone="America/New_York",
    market_open=time(9, 30),
    market_close=time(16, 0),
    lunch_break=None,
    trading_days_per_year=252,
    risk_free_rate=0.05,
    regime_benchmark_ticker="SPY",
    regime_vol_ticker="^VIX",
    holiday_calendar_name="XNYS",
    currency="USD",
    broker_type="alpaca",
    earnings_blackout_enabled=True,
    news_sentiment_weight=1.0,
)

LSE = MarketConfig(
    market_id="LSE",
    timezone="Europe/London",
    market_open=time(8, 0),
    market_close=time(16, 30),
    lunch_break=None,
    trading_days_per_year=252,
    risk_free_rate=0.04,
    regime_benchmark_ticker="^FTSE",
    regime_vol_ticker=None,          # trend-only by default
    holiday_calendar_name="XLON",
    currency="GBP",
    broker_type="ibkr",
    earnings_blackout_enabled=False,
    news_sentiment_weight=0.5,
    ibkr_exchanges=frozenset({"LSE", "AQSE"}),
    ibkr_primary_exchanges=frozenset({"LSE", "AQSE"}),
    ibkr_currency="GBP",
    max_equity_allocation=0.5,
)

TSE = MarketConfig(
    market_id="TSE",
    timezone="Asia/Tokyo",
    market_open=time(9, 0),
    market_close=time(15, 30),  # XTKS calendar close
    lunch_break=(time(11, 30), time(12, 30)),
    trading_days_per_year=245,
    risk_free_rate=0.005,
    regime_benchmark_ticker="^N225",
    regime_vol_ticker=None,          # trend-only by default
    holiday_calendar_name="XTKS",
    currency="JPY",
    broker_type="ibkr",
    earnings_blackout_enabled=False,
    news_sentiment_weight=0.0,
    ibkr_exchanges=frozenset({"TSEJ", "JPX"}),
    ibkr_primary_exchanges=frozenset({"TSEJ", "JPX"}),
    ibkr_currency="JPY",
    max_equity_allocation=0.5,
)

# ── Lookup ────────────────────────────────────────────────

PRESETS: dict[str, MarketConfig] = {
    "US": US,
    "LSE": LSE,
    "TSE": TSE,
}


def get_market_config(market_id: str) -> MarketConfig:
    """Return the preset MarketConfig for a known market_id."""
    try:
        return PRESETS[market_id.upper()]
    except KeyError:
        raise ValueError(
            f"Unknown market_id '{market_id}'. "
            f"Available: {sorted(PRESETS.keys())}"
        )
