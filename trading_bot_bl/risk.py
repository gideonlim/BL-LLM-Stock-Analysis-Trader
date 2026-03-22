"""Risk manager -- enforces portfolio constraints before execution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from trading_bot_bl.config import RiskLimits
from trading_bot_bl.earnings import check_earnings_blackout
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

    # SPY trend regime — set by executor from SpyRegime.
    # Overrides max_positions and min_composite_score when active.
    spy_trend_regime: str = "BULL"

    # Regime-adjusted limits (set by apply_spy_regime_overrides).
    # These shadow the base limits during CAUTION/BEAR/SEVERE_BEAR.
    _effective_max_positions: int = field(default=0, init=False)
    _effective_min_composite: float = field(default=0.0, init=False)

    _circuit_breaker_tripped: bool = field(
        default=False, init=False
    )

    def __post_init__(self) -> None:
        # Start with base limits; apply_spy_regime_overrides
        # will tighten them if needed.
        self._effective_max_positions = self.limits.max_positions
        self._effective_min_composite = (
            self.limits.min_composite_score
        )

    def apply_spy_regime_overrides(
        self,
        *,
        bear_max_positions: int = 4,
        bear_min_composite: float = 30.0,
        caution_max_positions: int = 6,
        caution_min_composite: float = 22.0,
    ) -> None:
        """Tighten risk limits based on the current SPY trend regime.

        Called by the executor after constructing the RiskManager.
        Does nothing if regime is BULL.
        """
        regime = self.spy_trend_regime
        if regime == "SEVERE_BEAR":
            # Hard halt — set max positions to 0
            self._effective_max_positions = 0
            self._effective_min_composite = 999.0
            log.warning(
                "SPY SEVERE_BEAR: halting all new entries "
                "(drawdown exceeds threshold)"
            )
        elif regime == "BEAR":
            self._effective_max_positions = min(
                self.limits.max_positions, bear_max_positions
            )
            self._effective_min_composite = max(
                self.limits.min_composite_score,
                bear_min_composite,
            )
            log.info(
                f"SPY BEAR regime active: max_positions="
                f"{self._effective_max_positions}, "
                f"min_composite={self._effective_min_composite}"
            )
        elif regime == "CAUTION":
            self._effective_max_positions = min(
                self.limits.max_positions, caution_max_positions
            )
            self._effective_min_composite = max(
                self.limits.min_composite_score,
                caution_min_composite,
            )
            log.info(
                f"SPY CAUTION regime active: max_positions="
                f"{self._effective_max_positions}, "
                f"min_composite={self._effective_min_composite}"
            )
        # BULL: keep base limits (already set in __post_init__)

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

        Uses regime-adjusted min_composite_score when SPY trend
        filter is active (CAUTION/BEAR raise the bar).
        """
        if signal.signal in ("HOLD", "ERROR"):
            if self.limits.skip_hold_signals:
                return f"Signal is {signal.signal} — skipped"

        effective_min = self._effective_min_composite
        if signal.composite_score < effective_min:
            regime_note = ""
            if effective_min > self.limits.min_composite_score:
                regime_note = (
                    f" [SPY {self.spy_trend_regime}: raised from "
                    f"{self.limits.min_composite_score}]"
                )
            return (
                f"Composite score {signal.composite_score} "
                f"< min {effective_min}{regime_note}"
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

        # PBO overfitting check (CSCV)
        if (
            self.limits.max_pbo < 1.0
            and hasattr(signal, "pbo")
            and signal.pbo >= 0
        ):
            if signal.pbo > self.limits.max_pbo:
                return (
                    f"PBO {signal.pbo:.0%} exceeds max "
                    f"{self.limits.max_pbo:.0%} — likely overfit"
                )

        return None

    def check_earnings_blackout(self, ticker: str) -> str | None:
        """
        Check if a ticker is in earnings blackout.
        Returns rejection reason or None if OK.

        Uses configurable pre/post day windows. Gracefully
        degrades if yfinance data is unavailable (returns None).
        """
        try:
            info = check_earnings_blackout(
                ticker,
                pre_days=self.limits.earnings_blackout_pre_days,
                post_days=self.limits.earnings_blackout_post_days,
            )
            if info.in_blackout:
                return info.blackout_reason
        except Exception as e:
            log.debug(f"Earnings check failed for {ticker}: {e}")
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

        # ── SPY regime hard halt ──────────────────────────────────
        if (
            intent.side == "buy"
            and self.spy_trend_regime == "SEVERE_BEAR"
        ):
            return RiskVerdict(
                approved=False,
                reason=(
                    "SEVERE_BEAR: SPY drawdown exceeds threshold "
                    "— all new entries halted"
                ),
            )

        # ── Max positions (regime-adjusted) ─────────────────────
        if (
            intent.side == "buy"
            and self._effective_max_positions > 0
            and len(portfolio.positions)
            >= self._effective_max_positions
        ):
            regime_note = ""
            if self._effective_max_positions < self.limits.max_positions:
                regime_note = (
                    f" [SPY {self.spy_trend_regime}: reduced from "
                    f"{self.limits.max_positions}]"
                )
            return RiskVerdict(
                approved=False,
                reason=(
                    f"Already at {len(portfolio.positions)} positions "
                    f"(max {self._effective_max_positions})"
                    f"{regime_note}"
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

        # ── Earnings blackout ────────────────────────────────────
        if (
            intent.side == "buy"
            and self.limits.earnings_blackout_enabled
        ):
            earnings_issue = self.check_earnings_blackout(
                intent.ticker
            )
            if earnings_issue:
                return RiskVerdict(
                    approved=False, reason=earnings_issue
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
