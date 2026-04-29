"""Event-driven minute-bar backtest engine.

Replays historical bars through the real strategy + filter pipeline.
No separate "backtest" logic in strategies — the same ``scan_ticker``
and ``manage`` methods the live daemon calls.

Simulation model:

- Bars are replayed in chronological order, one per tick.
- On each tick: ``BarCache.add_bar`` → strategies scan → filter
  pipeline evaluates → risk manager reviews → simulated fill on the
  NEXT bar's open (no look-ahead).
- Slippage: configurable fraction of ATR added to entry / subtracted
  from exit. Default 10% of ATR.
- Commissions: $0 by default (Alpaca zero-commission). Configurable.
- Position management: strategies' ``manage()`` fires each tick for
  open positions → time stops etc.
- Force-flat at session close (last bar of each day).
- Tracks per-trade P&L, filter rejection counts, and aggregate
  metrics.

Thread model: purely synchronous. No async, no network. Designed
for fast iteration on a dev machine.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from day_trader.calendar import ET, session_for
from day_trader.config import DayRiskLimits
from day_trader.data.cache import BarCache
from day_trader.filters.base import Filter, FilterPipeline
from day_trader.filters.cooldown import CooldownTracker
from day_trader.models import (
    Bar,
    DayTradeSignal,
    FilterContext,
    MarketState,
    TickerContext,
)
from day_trader.strategies.base import DayTradeStrategy

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────


@dataclass
class SimulatedTrade:
    """One completed round-trip in the backtest."""

    ticker: str
    strategy: str
    side: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str
    qty: int
    pnl: float
    pnl_pct: float
    r_multiple: float
    holding_bars: int
    slippage_cost: float


@dataclass
class BacktestResult:
    """Aggregate output of a backtest run."""

    strategy_name: str
    start_date: str
    end_date: str
    total_bars: int = 0
    sessions_simulated: int = 0

    # Trade stats
    trades: list[SimulatedTrade] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_r: float = 0.0

    # Filter stats
    signals_generated: int = 0
    signals_filtered: int = 0
    filter_rejection_histogram: dict = field(default_factory=dict)

    # Slippage
    total_slippage_cost: float = 0.0

    def summary(self) -> str:
        pf_str = (
            "∞" if self.profit_factor == float("inf")
            else f"{self.profit_factor:.2f}"
        )
        return (
            f"=== Backtest: {self.strategy_name} ===\n"
            f"Period: {self.start_date} → {self.end_date} "
            f"({self.sessions_simulated} sessions, "
            f"{self.total_bars} bars)\n"
            f"Trades: {self.total_trades} "
            f"({self.wins}W / {self.losses}L)\n"
            f"Win Rate: {self.win_rate:.1%}\n"
            f"P&L: ${self.total_pnl:+,.2f} "
            f"(avg R: {self.avg_r:+.2f})\n"
            f"Profit Factor: {pf_str}\n"
            f"Sharpe: {self.sharpe_ratio:.2f}  "
            f"Sortino: {self.sortino_ratio:.2f}\n"
            f"Max DD: {self.max_drawdown_pct:.1f}%\n"
            f"Slippage: ${self.total_slippage_cost:,.2f}\n"
            f"Signals: {self.signals_generated} generated, "
            f"{self.signals_filtered} filtered\n"
            f"Filter rejections: {self.filter_rejection_histogram}\n"
        )

    @property
    def passes_plan_criteria(self) -> dict[str, bool]:
        """Check against the plan's OOS pass thresholds."""
        return {
            "sharpe_gt_1.0": self.sharpe_ratio > 1.0,
            "profit_factor_gt_1.3": self.profit_factor > 1.3,
            "max_dd_lt_8pct": self.max_drawdown_pct < 8.0,
        }


# ── Engine config ─────────────────────────────────────────────────


@dataclass
class BacktestConfig:
    """Tuning knobs for the backtest engine."""

    # Starting capital for the simulation
    starting_equity: float = 100_000.0

    # Risk limits (reuses the real config)
    risk_limits: DayRiskLimits = field(default_factory=DayRiskLimits)

    # Slippage as fraction of ATR (0.10 = 10% of ATR per side)
    slippage_atr_frac: float = 0.10

    # Commission per share (Alpaca = $0)
    commission_per_share: float = 0.0

    # Max positions per session (same as live)
    max_positions_per_session: int = 3

    # Max trades per session
    max_trades_per_session: int = 8


# ── Simulated position ───────────────────────────────────────────


@dataclass
class _SimPosition:
    ticker: str
    strategy: str
    side: str
    qty: int
    entry_price: float
    entry_bar_idx: int
    entry_date: str
    sl_price: float
    tp_price: float
    atr: float


# ── Engine ────────────────────────────────────────────────────────


class BacktestEngine:
    """Replays bars through a strategy + filter pipeline.

    Usage::

        engine = BacktestEngine(
            strategy=OrbVwapStrategy(),
            filters=[RegimeFilter(limits), SpreadFilter(limits), ...],
            config=BacktestConfig(),
        )
        result = engine.run(
            bars_by_date=grouped_bars,
            ticker_contexts=contexts,
        )
    """

    def __init__(
        self,
        strategy: DayTradeStrategy,
        filters: list[Filter],
        config: Optional[BacktestConfig] = None,
    ):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.pipeline = FilterPipeline(filters) if filters else None

    def run(
        self,
        bars_by_date: dict[date, dict[str, list[Bar]]],
        ticker_contexts: dict[str, TickerContext],
        market_state: Optional[MarketState] = None,
    ) -> BacktestResult:
        """Run the backtest over all dates.

        Args:
            bars_by_date: ``{date: {ticker: [Bar, Bar, ...]}}`` —
                bars sorted chronologically per ticker per date.
            ticker_contexts: premarket context per ticker (RVOL etc.)
            market_state: shared market state (VIX, regime). If None,
                defaults to neutral.

        Returns:
            :class:`BacktestResult` with all trades and metrics.
        """
        ms = market_state or MarketState(
            spy_price=500, vix=15.0, spy_trend_regime="BULL",
        )

        all_trades: list[SimulatedTrade] = []
        all_daily_pnls: list[float] = []
        signals_total = 0
        signals_filtered = 0
        total_bars = 0
        sessions = 0

        sorted_dates = sorted(bars_by_date.keys())
        if not sorted_dates:
            return BacktestResult(
                strategy_name=self.strategy.name,
                start_date="", end_date="",
            )

        for sim_date in sorted_dates:
            session = session_for(sim_date)
            if session is None:
                continue
            sessions += 1

            day_bars = bars_by_date[sim_date]
            day_result = self._simulate_session(
                session, day_bars, ticker_contexts, ms,
            )

            all_trades.extend(day_result["trades"])
            all_daily_pnls.append(day_result["daily_pnl"])
            total_bars += day_result["bars_processed"]
            signals_total += day_result["signals"]
            signals_filtered += day_result["filtered"]

        # Compute aggregate metrics
        result = self._compute_metrics(
            all_trades, all_daily_pnls,
            strategy_name=self.strategy.name,
            start_date=str(sorted_dates[0]),
            end_date=str(sorted_dates[-1]),
            total_bars=total_bars,
            sessions_simulated=sessions,
        )
        result.signals_generated = signals_total
        result.signals_filtered = signals_filtered
        return result

    def _simulate_session(
        self,
        session,
        day_bars: dict[str, list[Bar]],
        ticker_contexts: dict[str, TickerContext],
        ms: MarketState,
    ) -> dict:
        """Simulate one trading session."""
        cache = BarCache()
        self.strategy.reset_for_session()
        cooldowns = CooldownTracker(
            ticker_minutes=self.config.risk_limits.ticker_cooldown_minutes,
            strategy_minutes=self.config.risk_limits.strategy_cooldown_minutes,
        )

        positions: dict[str, _SimPosition] = {}
        closed_trades: list[SimulatedTrade] = []
        trades_today = 0
        daily_pnl = 0.0
        signals_count = 0
        filtered_count = 0

        # Interleave all tickers' bars chronologically
        all_bars: list[Bar] = []
        for ticker_bars in day_bars.values():
            all_bars.extend(ticker_bars)
        all_bars.sort(key=lambda b: b.timestamp)

        pending_signals: list[DayTradeSignal] = []

        for bar_idx, bar in enumerate(all_bars):
            bar = cache.add_bar(bar)
            bar_time = bar.timestamp
            if bar_time.tzinfo is None:
                bar_time = bar_time.replace(tzinfo=ET)

            # ── Fill pending signals at this bar's open ──────────
            for sig in pending_signals:
                if sig.ticker != bar.ticker:
                    continue
                if sig.ticker in positions:
                    continue
                if trades_today >= self.config.max_trades_per_session:
                    continue
                if len(positions) >= self.config.max_positions_per_session:
                    continue

                # Fill at this bar's open + slippage
                slip = self.config.slippage_atr_frac * sig.atr
                fill_price = bar.open + slip if sig.side == "buy" else bar.open - slip

                risk_per_share = abs(fill_price - sig.stop_loss_price)
                if risk_per_share <= 0:
                    continue
                max_risk = (
                    self.config.starting_equity
                    * self.config.risk_limits.per_trade_risk_pct / 100
                )
                qty = math.floor(max_risk / risk_per_share)
                if qty < 1:
                    continue

                positions[sig.ticker] = _SimPosition(
                    ticker=sig.ticker,
                    strategy=sig.strategy,
                    side=sig.side,
                    qty=qty,
                    entry_price=fill_price,
                    entry_bar_idx=bar_idx,
                    entry_date=str(bar_time.date()),
                    sl_price=sig.stop_loss_price,
                    tp_price=sig.take_profit_price,
                    atr=sig.atr,
                )
                trades_today += 1

            # Clear pending signals that were for this bar's ticker
            pending_signals = [
                s for s in pending_signals if s.ticker != bar.ticker
            ]

            # ── Check SL/TP on open positions ────────────────────
            pos = positions.get(bar.ticker)
            if pos is not None:
                exit_price = None
                exit_reason = ""

                if pos.side == "buy":
                    if bar.low <= pos.sl_price:
                        exit_price = pos.sl_price
                        exit_reason = "stop_loss"
                    elif bar.high >= pos.tp_price:
                        exit_price = pos.tp_price
                        exit_reason = "take_profit"
                else:
                    if bar.high >= pos.sl_price:
                        exit_price = pos.sl_price
                        exit_reason = "stop_loss"
                    elif bar.low <= pos.tp_price:
                        exit_price = pos.tp_price
                        exit_reason = "take_profit"

                # Force flat at last bar of session
                is_near_close = (
                    session.close_et - bar_time.astimezone(ET)
                ) < timedelta(minutes=6)
                if exit_price is None and is_near_close:
                    exit_price = bar.close
                    exit_reason = "force_eod"

                if exit_price is not None:
                    slip = self.config.slippage_atr_frac * pos.atr
                    if pos.side == "buy":
                        exit_price -= slip
                    else:
                        exit_price += slip

                    risk_per_share = abs(
                        pos.entry_price - pos.sl_price
                    )
                    if pos.side == "buy":
                        pnl_per_share = exit_price - pos.entry_price
                    else:
                        pnl_per_share = pos.entry_price - exit_price

                    pnl = pnl_per_share * pos.qty
                    r_mult = (
                        pnl_per_share / risk_per_share
                        if risk_per_share > 0 else 0.0
                    )
                    slip_cost = slip * pos.qty * 2  # entry + exit

                    closed_trades.append(SimulatedTrade(
                        ticker=pos.ticker,
                        strategy=pos.strategy,
                        side=pos.side,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=str(bar_time.date()),
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        qty=pos.qty,
                        pnl=round(pnl, 2),
                        pnl_pct=round(
                            pnl / (pos.entry_price * pos.qty) * 100, 2,
                        ),
                        r_multiple=round(r_mult, 4),
                        holding_bars=bar_idx - pos.entry_bar_idx,
                        slippage_cost=round(slip_cost, 2),
                    ))
                    daily_pnl += pnl
                    del positions[pos.ticker]

                    # Cooldown on loss
                    if pnl < 0:
                        cooldowns.record_close(
                            ticker=pos.ticker,
                            strategy=pos.strategy,
                            pnl=pnl,
                            when=bar_time if bar_time.tzinfo else None,
                        )

            # ── Strategy scan ────────────────────────────────────
            if (
                trades_today < self.config.max_trades_per_session
                and len(positions) < self.config.max_positions_per_session
                and bar_time.astimezone(ET) < session.close_minus(15)
            ):
                ctx = ticker_contexts.get(bar.ticker)
                if ctx is not None:
                    sig = self.strategy.scan_ticker(
                        bar.ticker, cache, ctx, ms,
                        bar_time.astimezone(ET) if bar_time.tzinfo else bar_time,
                        session,
                    )
                    if sig is not None:
                        signals_count += 1

                        if self.pipeline is not None:
                            fctx = FilterContext(
                                signal=sig,
                                bars=cache.get_bars(bar.ticker),
                                market_state=ms,
                            )
                            result = self.pipeline.evaluate(fctx)
                            if not result.passed:
                                filtered_count += 1
                                continue

                        # Don't fill now — queue for next bar's open
                        pending_signals.append(sig)

        # Force-flat remaining positions at session close
        for ticker, pos in list(positions.items()):
            last_bar = cache.latest(ticker)
            if last_bar is None:
                continue
            exit_price = last_bar.close
            slip = self.config.slippage_atr_frac * pos.atr
            if pos.side == "buy":
                exit_price -= slip
                pnl_per = exit_price - pos.entry_price
            else:
                exit_price += slip
                pnl_per = pos.entry_price - exit_price
            pnl = pnl_per * pos.qty
            risk_per = abs(pos.entry_price - pos.sl_price)
            closed_trades.append(SimulatedTrade(
                ticker=pos.ticker, strategy=pos.strategy,
                side=pos.side, entry_date=pos.entry_date,
                entry_price=pos.entry_price,
                exit_date=str(last_bar.timestamp.date()) if hasattr(last_bar.timestamp, "date") else "",
                exit_price=round(exit_price, 2),
                exit_reason="force_eod",
                qty=pos.qty, pnl=round(pnl, 2),
                pnl_pct=round(pnl / (pos.entry_price * pos.qty) * 100, 2) if pos.entry_price > 0 else 0,
                r_multiple=round(pnl_per / risk_per, 4) if risk_per > 0 else 0,
                holding_bars=len(all_bars) - pos.entry_bar_idx,
                slippage_cost=round(slip * pos.qty * 2, 2),
            ))
            daily_pnl += pnl

        return {
            "trades": closed_trades,
            "daily_pnl": round(daily_pnl, 2),
            "bars_processed": len(all_bars),
            "signals": signals_count,
            "filtered": filtered_count,
        }

    def _compute_metrics(
        self,
        trades: list[SimulatedTrade],
        daily_pnls: list[float],
        **kwargs,
    ) -> BacktestResult:
        """Compute aggregate metrics from all trades."""
        result = BacktestResult(**kwargs)
        result.trades = trades
        result.total_trades = len(trades)

        # Always populate filter stats — even when no trades survived,
        # the histogram tells you WHY nothing traded.
        if self.pipeline is not None:
            result.filter_rejection_histogram = dict(self.pipeline.stats)

        if not trades:
            return result

        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        result.wins = len(wins)
        result.losses = len(losses)
        result.total_pnl = round(sum(pnls), 2)
        result.win_rate = (
            result.wins / result.total_trades
            if result.total_trades > 0 else 0.0
        )
        result.avg_win = (
            round(sum(wins) / len(wins), 2) if wins else 0.0
        )
        result.avg_loss = (
            round(sum(losses) / len(losses), 2) if losses else 0.0
        )

        gross_wins = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0
        if gross_losses > 0:
            result.profit_factor = round(gross_wins / gross_losses, 2)
        elif gross_wins > 0:
            result.profit_factor = float("inf")

        r_multiples = [t.r_multiple for t in trades]
        result.avg_r = (
            round(sum(r_multiples) / len(r_multiples), 4)
            if r_multiples else 0.0
        )

        result.total_slippage_cost = round(
            sum(t.slippage_cost for t in trades), 2
        )

        # Sharpe / Sortino from daily P&L
        if len(daily_pnls) >= 2:
            mean_pnl = sum(daily_pnls) / len(daily_pnls)
            variance = sum(
                (p - mean_pnl) ** 2 for p in daily_pnls
            ) / (len(daily_pnls) - 1)
            std_pnl = math.sqrt(variance) if variance > 0 else 0
            if std_pnl > 0:
                result.sharpe_ratio = round(
                    (mean_pnl / std_pnl) * math.sqrt(252), 2,
                )
            downside = [p for p in daily_pnls if p < 0]
            if downside:
                down_var = sum(p ** 2 for p in downside) / len(daily_pnls)
                down_std = math.sqrt(down_var)
                if down_std > 0:
                    result.sortino_ratio = round(
                        (mean_pnl / down_std) * math.sqrt(252), 2,
                    )

        # Max drawdown
        if daily_pnls:
            cum = 0.0
            peak = 0.0
            max_dd = 0.0
            for pnl in daily_pnls:
                cum += pnl
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > max_dd:
                    max_dd = dd
            if self.config.starting_equity > 0:
                result.max_drawdown_pct = round(
                    max_dd / self.config.starting_equity * 100, 2,
                )

        # Filter rejection histogram
        if self.pipeline is not None:
            result.filter_rejection_histogram = dict(self.pipeline.stats)

        return result
