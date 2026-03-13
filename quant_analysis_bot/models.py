"""Data models for trades, backtest results, and daily signals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeRecord:
    """A single completed round-trip trade."""

    trade_num: int
    ticker: str
    strategy: str
    timeframe: str          # "3mo", "6mo", "12mo"
    direction: str          # "LONG" or "SHORT"
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    holding_days: int
    return_pct: float
    outcome: str            # "WIN" or "LOSS"


@dataclass
class BacktestResult:
    """Aggregated performance metrics for one strategy on one timeframe."""

    strategy_name: str
    ticker: str
    timeframe: str = ""
    backtest_start: str = ""
    backtest_end: str = ""
    trading_days: int = 0
    # Raw returns
    total_return_pct: float = 0.0
    buy_hold_return_pct: float = 0.0
    excess_return_pct: float = 0.0
    # Annualized returns
    annual_return_pct: float = 0.0
    annual_bh_return_pct: float = 0.0
    annual_excess_pct: float = 0.0
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    # Trade stats
    win_rate: float = 0.0
    total_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    avg_holding_days: float = 0.0


@dataclass
class DailySignal:
    """Today's actionable signal for one ticker, with execution details."""

    # ── Identity ──────────────────────────────────────────────────────
    generated_at: str        # ISO datetime e.g. "2026-03-13T09:31:00"
    date: str                # YYYY-MM-DD  (trading date)
    ticker: str

    # ── Signal ────────────────────────────────────────────────────────
    signal: str              # BUY / EXIT / HOLD / SELL/SHORT / ERROR
    signal_raw: int          # 1=BUY  -1=SHORT/EXIT  0=HOLD
    strategy: str
    confidence: str          # HIGH / MEDIUM / LOW  (human label)
    confidence_score: int    # 0-6  (raw points behind label)
    composite_score: float   # weighted multi-timeframe backtest score

    # ── Execution ─────────────────────────────────────────────────────
    current_price: float
    stop_loss_pct: float     # e.g. 5.0 = stop 5% below entry
    stop_loss_price: float   # absolute stop price
    take_profit_pct: float   # e.g. 10.0 = target 10% above entry
    take_profit_price: float  # absolute take-profit price
    suggested_position_size_pct: float  # % of portfolio (half-Kelly)
    signal_expires: str      # YYYY-MM-DD  (re-evaluate after this)

    # ── Backtest quality ──────────────────────────────────────────────
    sharpe: float
    sortino: float
    win_rate: float          # 0-100 scale
    profit_factor: float
    annual_return_pct: float
    annual_excess_pct: float
    max_drawdown_pct: float
    avg_holding_days: float
    total_trades: int
    backtest_period: str     # "2025-03-11 to 2026-03-11 (252 days)"

    # ── Market context ────────────────────────────────────────────────
    rsi: float
    vol_20: float            # annualised 20-day volatility
    sma_50: float
    sma_200: float
    trend: str               # BULLISH / BEARISH / NEUTRAL
    volatility: str          # LOW / MEDIUM / HIGH  (human label)
    notes: str
