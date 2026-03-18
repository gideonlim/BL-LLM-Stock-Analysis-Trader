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
    # Extended fields (v2) — used by BL view estimation
    sortino: float = 0.0
    profit_factor: float = 0.0
    annual_return_pct: float = 0.0
    annual_excess_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    vol_20: float = 0.0


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


# ── Trade Journal Models ──────────────────────────────────────────


@dataclass
class JournalEntry:
    """Complete lifecycle record of a single trade.

    Created when a bracket order is submitted (status='pending'),
    confirmed when the fill is detected (status='open'), and
    finalised when the position closes (status='closed').

    All journal operations are non-critical — failures must never
    disrupt the core trading pipeline.
    """

    # ── Identity ──────────────────────────────────────────
    trade_id: str              # unique: f"{ticker}_{order_id[:8]}"
    ticker: str
    strategy: str
    side: str                  # "long" or "short"

    # ── Entry ─────────────────────────────────────────────
    entry_order_id: str        # Alpaca order ID
    entry_signal_price: float  # decision price from signal
    entry_fill_price: float = 0.0
    entry_slippage: float = 0.0
    entry_slippage_pct: float = 0.0
    entry_notional: float = 0.0
    entry_qty: float = 0.0
    entry_date: str = ""
    entry_composite_score: float = 0.0
    entry_confidence: str = ""
    entry_confidence_score: int = 0

    # ── Market Context at Entry ───────────────────────────
    entry_vix: float = 0.0
    entry_market_regime: str = ""
    entry_spy_price: float = 0.0

    # ── Bracket (as placed) ───────────────────────────────
    original_sl_price: float = 0.0
    original_tp_price: float = 0.0
    initial_risk_per_share: float = 0.0
    initial_risk_dollars: float = 0.0
    initial_reward_dollars: float = 0.0
    planned_rr_ratio: float = 0.0

    # ── Exit ──────────────────────────────────────────────
    exit_price: float = 0.0
    exit_date: str = ""
    exit_reason: str = ""
    exit_order_id: str = ""
    exit_fill_price: float = 0.0
    exit_slippage: float = 0.0

    # ── Outcome (populated on close) ─────────────────────
    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    r_multiple: float = 0.0
    holding_days: int = 0

    # ── Excursion tracking ────────────────────────────────
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    mfe_date: str = ""
    mae_date: str = ""
    etd: float = 0.0
    etd_pct: float = 0.0
    edge_ratio: float = 0.0

    # ── SL modifications (audit trail) ───────────────────
    sl_modifications: list = field(default_factory=list)

    # ── Sparse price path ─────────────────────────────────
    price_samples: list = field(default_factory=list)

    # ── Status ────────────────────────────────────────────
    status: str = "pending"
    opened_at: str = ""
    closed_at: str = ""

    # ── Metadata ──────────────────────────────────────────
    tags: list = field(default_factory=list)
    notes: str = ""


@dataclass
class EquitySnapshot:
    """Point-in-time portfolio value for equity curve."""

    timestamp: str = ""
    equity: float = 0.0
    cash: float = 0.0
    market_value: float = 0.0
    num_positions: int = 0
    realized_pnl_today: float = 0.0
    unrealized_pnl: float = 0.0
    day_pnl: float = 0.0
    day_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    high_water_mark: float = 0.0
    exposure_pct: float = 0.0
