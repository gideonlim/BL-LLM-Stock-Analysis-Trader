"""Risk manager -- enforces portfolio constraints before execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from trading_bot_bl.config import RiskLimits
from trading_bot_bl.history import TradeHistory
from trading_bot_bl.models import OrderIntent, PortfolioSnapshot, Signal

log = logging.getLogger(__name__)


@dataclass
class RiskVerdict:
    """Result of a risk check on a proposed order."""

    approved: bool
    order: OrderIntent | None = None
    reason: str = ""
    adjusted_notional: float = 0.0


@dataclass
class RiskManager:
    """Enforces all portfolio-level risk constraints."""

    limits: RiskLimits
    history: Optional[TradeHistory] = None

    # Market-sentiment position-size multiplier (1.0 = neutral).
    # Set by the executor from the MarketSentiment module.
    sentiment_size_multiplier: float = 1.0

    _circuit_breaker_tripped: bool = field(
        default=False, init=False
    )

    def check_circuit_breaker(
        self, portfolio: PortfolioSnapshot
    ) -> bool:
        """
        Check if daily loss circuit breaker has been tripped.
        Returns True if trading should STOP.
        """
        if self._circuit_breaker_tripped:
            return True

        if portfolio.day_pnl_pct <= -self.limits.daily_loss_limit_pct:
            self._circuit_breaker_tripped = True
            log.warning(
                f"CIRCUIT BREAKER TRIPPED: "
                f"Day P&L = {portfolio.day_pnl_pct:.2f}% "
                f"(limit: -{self.limits.daily_loss_limit_pct}%)"
            )
            return True

        return False

    def check_signal_quality(self, signal: Signal) -> str | None:
        """
        Check if a signal meets minimum quality thresholds.
        Returns rejection reason or None if OK.
        """
        if signal.signal in ("HOLD", "ERROR"):
            if self.limits.skip_hold_signals:
                return f"Signal is {signal.signal} — skipped"

        if signal.composite_score < self.limits.min_composite_score:
            return (
                f"Composite score {signal.composite_score} "
                f"< min {self.limits.min_composite_score}"
            )

        if signal.confidence_score < self.limits.min_confidence_score:
            return (
                f"Confidence score {signal.confidence_score} "
                f"< min {self.limits.min_confidence_score}"
            )

        if self.limits.check_signal_expiry:
            today_str = date.today().isoformat()
            if (
                signal.signal_expires
                and signal.signal_expires < today_str
            ):
                return (
                    f"Signal expired on {signal.signal_expires}"
                )

        if signal.total_trades < self.limits.min_backtest_trades:
            return (
                f"Too few backtest trades ({signal.total_trades}) "
                f"— unreliable signal"
            )

        return None

    def check_strategy_history(
        self, intent: OrderIntent
    ) -> str | None:
        """
        Check the strategy's track record from past execution logs.

        Returns rejection reason or None if OK.
        Skips check if no history is available.
        """
        if self.history is None:
            return None

        strategy = intent.signal.strategy
        ticker = intent.ticker

        # Check if this strategy has been underperforming
        # Only considers strategy-attributable outcomes (signal
        # quality skips), not broker errors or portfolio limits
        if self.history.strategy_is_underperforming(
            strategy,
            min_orders=5,
            min_success_rate=0.3,
        ):
            rec = self.history.get_strategy_record(strategy)
            return (
                f"Strategy '{strategy}' has poor track record "
                f"({rec.submitted}/{rec.strategy_relevant_total}"
                f" quality-relevant orders filled, "
                f"{rec.success_rate:.0%} success rate)"
            )

        # Check if this ticker was traded very recently
        # (avoid churning the same stock)
        if self.history.was_recently_traded(ticker, days=2):
            th = self.history.get_ticker_history(ticker)
            return (
                f"{ticker} was already bought on "
                f"{th.last_buy_date[:10]} via "
                f"'{th.last_buy_strategy}' — "
                f"avoiding churn (2-day cooldown)"
            )

        # Check if strategy has negative P&L on current positions
        rec = self.history.get_strategy_record(strategy)
        if rec and rec.loss_count > rec.win_count >= 3:
            return (
                f"Strategy '{strategy}' currently losing "
                f"(wins={rec.win_count}, losses={rec.loss_count}, "
                f"P&L=${rec.realized_pnl:+,.0f})"
            )

        return None

    def evaluate_order(
        self,
        intent: OrderIntent,
        portfolio: PortfolioSnapshot,
    ) -> RiskVerdict:
        """
        Run all risk checks on a proposed order.

        Returns an approved/rejected verdict with reasons.
        The notional may be adjusted down to fit constraints.
        """
        ticker = intent.ticker
        equity = portfolio.equity

        if equity <= 0:
            return RiskVerdict(
                approved=False,
                reason="Account equity is zero or negative",
            )

        # ── Circuit breaker ───────────────────────────────────────
        if self.check_circuit_breaker(portfolio):
            return RiskVerdict(
                approved=False,
                reason="Daily loss circuit breaker is active",
            )

        # ── Max positions ─────────────────────────────────────────
        if (
            intent.side == "buy"
            and self.limits.max_positions > 0
            and len(portfolio.positions) >= self.limits.max_positions
        ):
            return RiskVerdict(
                approved=False,
                reason=(
                    f"Already at {len(portfolio.positions)} positions "
                    f"(max {self.limits.max_positions})"
                ),
            )

        # ── Minimum position size ──────────────────────────────────
        if (
            intent.side == "buy"
            and self.limits.min_position_pct > 0
        ):
            min_notional = equity * self.limits.min_position_pct / 100
            if intent.notional < min_notional:
                return RiskVerdict(
                    approved=False,
                    reason=(
                        f"Notional ${intent.notional:,.0f} below "
                        f"minimum {self.limits.min_position_pct}% "
                        f"of equity (${min_notional:,.0f})"
                    ),
                )

        # ── Signal quality ────────────────────────────────────────
        quality_issue = self.check_signal_quality(intent.signal)
        if quality_issue:
            return RiskVerdict(
                approved=False, reason=quality_issue
            )

        # ── Strategy history check ────────────────────────────────
        history_issue = self.check_strategy_history(intent)
        if history_issue:
            return RiskVerdict(
                approved=False, reason=history_issue
            )

        # ── Max portfolio exposure ────────────────────────────────
        current_exposure_pct = (
            portfolio.market_value / equity * 100
            if equity > 0
            else 0
        )
        max_exposure = self.limits.max_portfolio_exposure_pct
        remaining_exposure_pct = max_exposure - current_exposure_pct

        if remaining_exposure_pct <= 0:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"Portfolio exposure {current_exposure_pct:.1f}%"
                    f" >= max {max_exposure}%"
                ),
            )

        max_from_exposure = equity * remaining_exposure_pct / 100

        # ── Per-stock position cap ────────────────────────────────
        existing_value = 0.0
        if ticker in portfolio.positions:
            existing_value = abs(
                portfolio.positions[ticker]["market_value"]
            )

        max_position_value = (
            equity * self.limits.max_position_pct / 100
        )
        room_for_ticker = max_position_value - existing_value

        if room_for_ticker <= 0:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"{ticker} already at "
                    f"${existing_value:,.0f} "
                    f"(max ${max_position_value:,.0f}, "
                    f"{self.limits.max_position_pct}% of equity)"
                ),
            )

        # ── Adjust notional to fit all constraints ────────────────
        adjusted = min(
            intent.notional,
            max_from_exposure,
            room_for_ticker,
            portfolio.cash,  # can't spend more than cash
        )

        # ── Apply market-sentiment size multiplier ─────────────
        if (
            self.sentiment_size_multiplier != 1.0
            and intent.side == "buy"
        ):
            pre_sentiment = adjusted
            adjusted = round(
                adjusted * self.sentiment_size_multiplier, 2
            )
            # Never scale above the hard limits
            adjusted = min(
                adjusted,
                max_from_exposure,
                room_for_ticker,
                portfolio.cash,
            )
            if adjusted != pre_sentiment:
                log.info(
                    f"  {ticker}: sentiment multiplier "
                    f"{self.sentiment_size_multiplier:.2f} → "
                    f"${pre_sentiment:,.0f} -> ${adjusted:,.0f}"
                )

        # Don't bother with tiny orders (< $1)
        if adjusted < 1.0:
            return RiskVerdict(
                approved=False,
                reason=(
                    f"Adjusted notional ${adjusted:.2f} "
                    f"too small to execute"
                ),
            )

        # Re-check minimum position size AFTER adjustment.
        # The initial check (above) catches obviously small intents,
        # but cash/exposure limits can shrink a $12k intent to $1.8k.
        if (
            intent.side == "buy"
            and self.limits.min_position_pct > 0
        ):
            min_notional = equity * self.limits.min_position_pct / 100
            if adjusted < min_notional:
                return RiskVerdict(
                    approved=False,
                    reason=(
                        f"Adjusted notional ${adjusted:,.0f} below "
                        f"minimum {self.limits.min_position_pct}% "
                        f"of equity (${min_notional:,.0f}) after "
                        f"risk limits"
                    ),
                )

        if adjusted < intent.notional:
            log.info(
                f"  {ticker}: notional adjusted "
                f"${intent.notional:,.0f} -> ${adjusted:,.0f} "
                f"(risk limits)"
            )

        return RiskVerdict(
            approved=True,
            order=OrderIntent(
                ticker=intent.ticker,
                side=intent.side,
                notional=adjusted,
                stop_loss_price=intent.stop_loss_price,
                take_profit_price=intent.take_profit_price,
                signal=intent.signal,
                reason=intent.reason,
            ),
            adjusted_notional=adjusted,
        )
