"""Day-trader risk manager.

Owns the portfolio-level constraints that don't fit inside a
single-signal filter: daily loss kill switch, max concurrent
positions, max trades per day, max position size as % of sub-budget.

Coordinates with:

- :class:`SubBudgetTracker` — the 25%-of-equity capital ceiling.
- :class:`CooldownTracker` — populated here on every closed trade
  via :meth:`record_close`; consumed by ``CooldownFilter``.
- :class:`SymbolLock` — read-only check; lock state is consulted
  by ``SymbolLockFilter``.

The risk manager is stateful across a session and reset at session
start. Recovery hands us the inherited open notional + open position
count from yesterday/crash so counters start in the right place.

Concurrency: the daemon is single-process asyncio, but the budget
tracker and cooldown tracker have their own internal locks so this
is safe across the few async tasks that touch state (scheduler,
fill handler, position monitor).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from threading import Lock
from typing import Optional

from day_trader.budget import SubBudgetTracker
from day_trader.config import DayRiskLimits
from day_trader.filters.cooldown import CooldownTracker
from day_trader.models import DayRiskVerdict, DayTradeSignal

log = logging.getLogger(__name__)


# ── Verdict reasons (string constants for stable journal/log keys) ─

REASON_KILL_SWITCH = "kill_switch_tripped"
REASON_MAX_POSITIONS = "max_positions_reached"
REASON_MAX_TRADES_TODAY = "max_trades_today_reached"
REASON_BUDGET_EXCEEDED = "budget_exceeded"
REASON_PER_TRADE_RISK = "per_trade_risk_exceeded"
REASON_POSITION_SIZE = "position_size_exceeds_max"
REASON_INVALID_SIGNAL = "invalid_signal"


@dataclass
class DayRiskManager:
    """Stateful per-session risk gate.

    Lifecycle:

    1. ``__init__`` — pass in ``limits`` and the shared
       ``budget`` / ``cooldowns`` trackers.
    2. ``start_session(equity, initial_open_notional, initial_positions)`` —
       called from the executor at session start AFTER
       ``recovery.reconcile()`` succeeds (clean state).
    3. ``review(intent_notional, signal)`` for each candidate order —
       returns ``DayRiskVerdict``.
    4. ``record_fill(notional)`` after a successful order fill —
       updates open position count + budget reservation.
    5. ``record_close(ticker, strategy, pnl, notional)`` after a
       position closes — updates daily P&L, releases budget,
       decrements open count, populates cooldowns on losses.
    6. ``reset_for_new_day()`` — only relevant if the daemon stays
       up across multiple sessions (rare; usually we restart fresh).
    """

    limits: DayRiskLimits
    budget: SubBudgetTracker
    cooldowns: CooldownTracker

    # ── Session state (set by start_session) ──────────────────────
    session_starting_equity: float = 0.0
    session_date: Optional[date] = None
    daily_realized_pnl: float = 0.0
    open_positions_count: int = 0
    trades_today: int = 0

    # ── Kill switch ───────────────────────────────────────────────
    kill_switch_tripped: bool = False
    kill_switch_reason: str = ""

    # ── External signals injected by executor ─────────────────────
    spy_severe_bear: bool = False

    # ── Lock for state mutation ───────────────────────────────────
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    # ── Lifecycle ─────────────────────────────────────────────────

    def start_session(
        self,
        *,
        equity: float,
        initial_open_notional: float = 0.0,
        initial_positions: int = 0,
        today: Optional[date] = None,
    ) -> None:
        """Reset all session counters and seed inherited state from recovery."""
        if equity < 0:
            raise ValueError(f"equity must be >= 0, got {equity!r}")
        if initial_open_notional < 0:
            raise ValueError(
                f"initial_open_notional must be >= 0, got "
                f"{initial_open_notional!r}"
            )
        if initial_positions < 0:
            raise ValueError(
                f"initial_positions must be >= 0, got "
                f"{initial_positions!r}"
            )

        self.budget.start_session(
            equity=equity,
            initial_open_notional=initial_open_notional,
        )
        self.cooldowns.reset_for_session()
        with self._lock:
            self.session_starting_equity = float(equity)
            self.session_date = today or date.today()
            self.daily_realized_pnl = 0.0
            self.open_positions_count = int(initial_positions)
            self.trades_today = 0
            self.kill_switch_tripped = False
            self.kill_switch_reason = ""

        log.info(
            "DayRiskManager session start: equity=$%.2f budget=$%.2f "
            "open_positions=%d daily_loss_limit=%.1f%% "
            "max_positions=%d max_trades=%d",
            equity, self.budget.budget,
            self.open_positions_count,
            self.limits.daily_loss_limit_pct,
            self.limits.max_positions,
            self.limits.max_trades_per_day,
        )

    # ── Pre-trade review ─────────────────────────────────────────

    def review(
        self,
        signal: DayTradeSignal,
        intent_notional: float,
        risk_dollars: float,
    ) -> DayRiskVerdict:
        """Evaluate a candidate order against portfolio-level risk.

        ``intent_notional`` = entry_price × shares the strategy wants to buy.
        ``risk_dollars`` = (entry_price − stop_loss_price) × shares — the
            amount the position would lose if the stop is hit.

        Returns a :class:`DayRiskVerdict` with ``approved`` and
        possibly ``adjusted_notional`` if we sized down (currently
        we don't auto-adjust — we either approve as-is or reject).
        """
        if signal is None:
            return DayRiskVerdict(
                approved=False, reason="signal_is_none",
                rejected_by="risk_manager",
            )
        if intent_notional <= 0:
            return DayRiskVerdict(
                approved=False, reason=REASON_INVALID_SIGNAL,
                rejected_by="risk_manager",
            )

        with self._lock:
            # 1. Kill switch (latched)
            if self.kill_switch_tripped:
                return DayRiskVerdict(
                    approved=False,
                    reason=f"{REASON_KILL_SWITCH}:{self.kill_switch_reason}",
                    rejected_by="risk_manager",
                )

            # 2. SPY SEVERE_BEAR halt — same global gate the swing bot
            # uses, propagated by the executor before each scan.
            if self.spy_severe_bear:
                return DayRiskVerdict(
                    approved=False, reason="spy_severe_bear",
                    rejected_by="risk_manager",
                )

            # 3. Max concurrent positions
            if self.open_positions_count >= self.limits.max_positions:
                return DayRiskVerdict(
                    approved=False, reason=REASON_MAX_POSITIONS,
                    rejected_by="risk_manager",
                )

            # 4. Max trades per day
            if self.trades_today >= self.limits.max_trades_per_day:
                return DayRiskVerdict(
                    approved=False, reason=REASON_MAX_TRADES_TODAY,
                    rejected_by="risk_manager",
                )

            # 5. Per-trade risk cap (0.25% of total equity by default)
            max_risk_dollars = self.session_starting_equity * (
                self.limits.per_trade_risk_pct / 100
            )
            if risk_dollars > max_risk_dollars:
                return DayRiskVerdict(
                    approved=False, reason=REASON_PER_TRADE_RISK,
                    rejected_by="risk_manager",
                )

            # 6. Per-position size cap (% of sub-budget)
            max_position_dollars = self.budget.budget * (
                self.limits.max_position_pct_of_budget / 100
            )
            if intent_notional > max_position_dollars:
                return DayRiskVerdict(
                    approved=False, reason=REASON_POSITION_SIZE,
                    rejected_by="risk_manager",
                )

        # 7. Sub-budget headroom (last because it's a reservation
        # and we want all cheaper checks to fail first)
        if not self.budget.can_reserve(intent_notional):
            return DayRiskVerdict(
                approved=False, reason=REASON_BUDGET_EXCEEDED,
                rejected_by="risk_manager",
            )

        return DayRiskVerdict(
            approved=True,
            adjusted_notional=intent_notional,
        )

    # ── State updates after broker events ─────────────────────────

    def record_fill(self, notional: float) -> bool:
        """Reserve budget + bump position/trade counters after a fill.

        Returns True on success, False if the budget reservation
        failed (which means we approved the order but lost a race —
        should be vanishingly rare in single-process daemon mode).
        """
        if notional < 0:
            raise ValueError(f"notional must be >= 0, got {notional!r}")
        if not self.budget.reserve(notional):
            log.error(
                "DayRiskManager: budget reservation lost a race for "
                "$%.2f — order WAS submitted but counters are inconsistent",
                notional,
            )
            return False
        with self._lock:
            self.open_positions_count += 1
            self.trades_today += 1
        return True

    def record_close(
        self,
        *,
        ticker: str,
        strategy: str,
        pnl: float,
        entry_notional: float,
        when: Optional[datetime] = None,
    ) -> None:
        """Update daily P&L, release budget, decrement position count,
        populate cooldowns on losses, trip kill switch if exceeded."""
        when = when or datetime.now()
        self.budget.release(entry_notional)
        with self._lock:
            self.daily_realized_pnl += float(pnl)
            self.open_positions_count = max(0, self.open_positions_count - 1)

            # Daily loss kill switch check
            if self.session_starting_equity > 0:
                loss_pct = -self.daily_realized_pnl / self.session_starting_equity * 100
                if (
                    not self.kill_switch_tripped
                    and loss_pct >= self.limits.daily_loss_limit_pct
                ):
                    self.kill_switch_tripped = True
                    self.kill_switch_reason = (
                        f"daily_loss_{loss_pct:.2f}pct_>={self.limits.daily_loss_limit_pct}pct"
                    )
                    log.warning(
                        "DayRiskManager KILL SWITCH TRIPPED: %s "
                        "(realized=$%.2f, equity=$%.2f, threshold=%.1f%%)",
                        self.kill_switch_reason,
                        self.daily_realized_pnl,
                        self.session_starting_equity,
                        self.limits.daily_loss_limit_pct,
                    )

        # Cooldown population happens OUTSIDE the lock — it has its
        # own lock and no shared state with us.
        self.cooldowns.record_close(
            ticker=ticker, strategy=strategy, pnl=pnl, when=when,
        )

    # ── External overrides ───────────────────────────────────────

    def set_spy_severe_bear(self, severe: bool) -> None:
        """Called by the executor when the swing bot's regime cache
        flips. ``True`` halts new entries (existing positions are
        managed normally)."""
        with self._lock:
            self.spy_severe_bear = bool(severe)

    def trip_kill_switch(self, reason: str) -> None:
        """Manually trip the kill switch (e.g. operator override
        via Telegram bot, or recovery incident-mode escalation)."""
        with self._lock:
            self.kill_switch_tripped = True
            self.kill_switch_reason = reason
        log.warning("DayRiskManager kill switch tripped manually: %s", reason)

    # ── Read-only views ──────────────────────────────────────────

    def is_kill_switch_tripped(self) -> bool:
        with self._lock:
            return self.kill_switch_tripped

    def daily_loss_pct(self) -> float:
        """Today's realized loss as % of session starting equity.
        Positive = profit; negative = loss."""
        with self._lock:
            if self.session_starting_equity <= 0:
                return 0.0
            return (
                self.daily_realized_pnl / self.session_starting_equity
            ) * 100

    def can_take_more_trades(self) -> bool:
        """True if at least one more trade slot is available."""
        with self._lock:
            return (
                not self.kill_switch_tripped
                and self.open_positions_count < self.limits.max_positions
                and self.trades_today < self.limits.max_trades_per_day
            )
