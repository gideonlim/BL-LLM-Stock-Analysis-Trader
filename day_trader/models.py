"""Day-trader data models.

These are the in-flight types — signals, bars, quotes, market state,
open positions, risk verdicts. Persisted state (journal entries,
order results) reuses ``trading_bot_bl/models.py`` so the existing
journal, equity-curve, and report code can read day-trade entries
without conditional logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Market data types ─────────────────────────────────────────────


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar at the configured aggregation level (1-min default)."""

    ticker: str
    timestamp: datetime  # tz-aware UTC; convert to ET via calendar.py
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0  # session VWAP up to and including this bar
    trade_count: int = 0


@dataclass(frozen=True)
class Quote:
    """NBBO snapshot at a moment in time."""

    ticker: str
    timestamp: datetime
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread_bps(self) -> float:
        if self.mid <= 0:
            return float("inf")
        return ((self.ask_price - self.bid_price) / self.mid) * 10_000


@dataclass(frozen=True)
class Trade:
    """Tick-level trade execution print."""

    ticker: str
    timestamp: datetime
    price: float
    size: float
    exchange: str = ""


# ── Higher-level state ────────────────────────────────────────────


@dataclass
class MarketState:
    """Snapshot of broad-market conditions captured at session start.

    Strategies / filters consult this rather than re-fetching live
    data on every scan tick. Re-fetched periodically by the executor.
    """

    spy_price: float = 0.0
    spy_200_sma: float = 0.0
    spy_trend_regime: str = "BULL"  # BULL | CAUTION | BEAR | SEVERE_BEAR
    vix: float = 0.0
    breadth: float = 0.0  # % of S&P above 50-SMA, optional
    captured_at: str = ""

    @property
    def is_severe_bear(self) -> bool:
        return self.spy_trend_regime == "SEVERE_BEAR"

    @property
    def is_high_vol(self) -> bool:
        return self.vix >= 30.0


# ── Signals + intents ─────────────────────────────────────────────


@dataclass
class DayTradeSignal:
    """A scan-time signal from a day-trader strategy.

    Strategies emit these; the filter pipeline accepts/rejects;
    the risk manager sizes; the executor turns survivors into
    bracket orders.
    """

    ticker: str
    strategy: str
    side: str  # "buy" or "sell"
    signal_price: float  # decision price (last trade or quote mid)
    stop_loss_price: float  # ATR-derived hard stop
    take_profit_price: float  # 2R or VWAP target
    atr: float  # ATR(14) on the strategy's bar interval
    rvol: float  # relative volume vs trailing avg
    notes: str = ""
    catalyst_label: str = ""  # NEWS_HIGH | NEWS_LOW | NO_NEWS
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


@dataclass
class OpenDayTrade:
    """Tracks a live day-trade position the daemon is managing.

    Created on fill, updated on stop adjustments and price ticks,
    closed via close_tagged_daytrade_qty() at exit.
    """

    ticker: str
    strategy: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp_price: float
    parent_client_order_id: str  # the dt:yyyymmdd:seq:ticker tag
    seq: int  # for building the :exit child id
    bracket_legs: list = field(default_factory=list)  # broker order ids


@dataclass
class ExitIntent:
    """A request from a strategy or scheduler to close a position."""

    ticker: str
    reason: str  # "take_profit" | "stop_loss" | "time_stop" | "force_eod" | etc
    requested_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


# ── Risk verdicts ─────────────────────────────────────────────────


@dataclass
class DayRiskVerdict:
    """Result of running a signal through the day-trade risk pipeline."""

    approved: bool
    reason: str = ""
    rejected_by: str = ""  # filter or check name
    adjusted_qty: int = 0
    adjusted_notional: float = 0.0


# ── Per-ticker premarket context ─────────────────────────────────


@dataclass
class TickerContext:
    """Per-ticker data accumulated by the premarket scanner.

    The data layer's ``PremarketScanner`` populates one of these per
    ticker on the day's watchlist. Strategies read it during scan
    to access premarket RVOL / gap / catalyst signals without
    re-fetching.
    """

    ticker: str
    premkt_rvol: float = 0.0           # premarket vol / 30d avg premkt vol
    premkt_gap_pct: float = 0.0        # (premkt_open - prev_close) / prev_close
    avg_daily_volume: float = 0.0      # 30-day avg shares
    avg_dollar_volume: float = 0.0     # 30-day avg $ volume
    prev_close: float = 0.0
    catalyst_label: str = ""           # NEWS_HIGH | NEWS_LOW | NO_NEWS | ""
    captured_at: str = ""


# ── Filter pipeline types ─────────────────────────────────────────


@dataclass
class FilterContext:
    """Everything a filter needs to evaluate a signal.

    Built once per scan tick by the executor and passed to each
    filter in the pipeline. Filters read; they do not mutate.
    """

    signal: Optional[DayTradeSignal] = None
    quote: Optional[Quote] = None
    bars: list = field(default_factory=list)  # rolling history (last N bars)
    market_state: Optional[MarketState] = None
    open_positions: dict = field(default_factory=dict)  # ticker → OpenDayTrade


@dataclass
class FilterResult:
    """Outcome of running a context through the filter pipeline."""

    passed: bool
    rejected_by: str = ""
    reason: str = ""
