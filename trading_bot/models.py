"""Data models for orders, positions, and execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    """A parsed signal from the analysis bot's output."""

    ticker: str
    signal: str            # BUY / EXIT / HOLD / SELL/SHORT / ERROR
    signal_raw: int        # 1 / -1 / 0
    strategy: str
    confidence: str        # HIGH / MEDIUM / LOW
    confidence_score: int
    composite_score: float
    current_price: float
    stop_loss_price: float
    take_profit_price: float
    suggested_position_size_pct: float
    signal_expires: str
    sharpe: float
    win_rate: float
    total_trades: int
    generated_at: str = ""
    notes: str = ""


@dataclass
class OrderIntent:
    """A proposed order before risk checks."""

    ticker: str
    side: str             # "buy" or "sell"
    notional: float       # dollar amount to trade
    stop_loss_price: float
    take_profit_price: float
    signal: Signal
    reason: str = ""      # why this order was generated


@dataclass
class OrderResult:
    """Result after submitting an order to the broker."""

    ticker: str
    order_id: str = ""
    status: str = ""      # "submitted", "rejected", "skipped"
    side: str = ""
    notional: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    strategy: str = ""    # which strategy generated this order
    error: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(
            timespec="seconds"
        )
    )


@dataclass
class PositionAlert:
    """An alert raised by the position monitor."""

    ticker: str
    alert_type: str       # "orphaned", "stale_bracket", "extreme_move",
                          # "emergency_loss", "bracket_replaced"
    severity: str         # "info", "warning", "critical"
    message: str
    action_taken: str = ""   # what the monitor did about it
    current_price: float = 0.0
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    unrealized_pnl_pct: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(
            timespec="seconds"
        )
    )


@dataclass
class PortfolioSnapshot:
    """Current state of the portfolio from the broker."""

    equity: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    market_value: float = 0.0  # total value of positions
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0
    positions: dict = field(default_factory=dict)
    # positions = {ticker: {"qty": float, "market_value": float,
    #              "avg_entry": float, "unrealized_pnl": float}}
