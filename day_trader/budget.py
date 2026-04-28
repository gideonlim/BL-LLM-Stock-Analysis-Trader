"""Sub-budget tracker — enforces the 25% day-trade ceiling.

The day-trader runs on the same Alpaca account as the swing bot.
The 25% constraint is software-enforced: at session start we snapshot
total account equity and multiply by ``budget_pct`` to get the
ceiling. The tracker maintains a running ``open_notional`` (sum of
all open day-trade entries' entry notional) and refuses new orders
that would push us over.

Recovery is the source of truth for ``open_notional`` at session
start (Alpaca positions are not tagged — only orders carry
``dt:``). After session start, the executor calls :meth:`reserve`
on order submission and :meth:`release` on close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import Lock

log = logging.getLogger(__name__)


@dataclass
class SubBudgetTracker:
    """Tracks open day-trade exposure against a fraction of total equity.

    Designed for a single-process daemon: thread-safe via an internal
    lock so the scheduler / position-monitor / fill-handler can all
    update concurrently without races.
    """

    initial_equity: float = 0.0
    budget_pct: float = 0.25
    _open_notional: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    # ── Read-only views ───────────────────────────────────────────

    @property
    def budget(self) -> float:
        """Dollar ceiling = ``initial_equity * budget_pct``."""
        return self.initial_equity * self.budget_pct

    @property
    def open_notional(self) -> float:
        """Sum of open day-trade entry notionals."""
        with self._lock:
            return self._open_notional

    @property
    def headroom(self) -> float:
        """Dollars still available to deploy."""
        with self._lock:
            return max(0.0, self.budget - self._open_notional)

    @property
    def utilization(self) -> float:
        """Fraction of budget currently deployed (0.0–1.0+)."""
        b = self.budget
        if b <= 0:
            return 0.0
        with self._lock:
            return self._open_notional / b

    # ── Lifecycle ─────────────────────────────────────────────────

    def start_session(
        self,
        *,
        equity: float,
        initial_open_notional: float = 0.0,
    ) -> None:
        """Reset for a new trading session.

        Call from the executor after :func:`recovery.reconcile` has
        confirmed which open day-trade entries the daemon inherited
        from yesterday or from a crash mid-session.

        Args:
            equity: Total account equity at session start.
            initial_open_notional: Sum of inherited open day-trade
                positions' entry notionals. From recovery, NOT from
                Alpaca positions (which aren't tagged).
        """
        if equity < 0:
            raise ValueError(f"equity must be >= 0, got {equity!r}")
        if initial_open_notional < 0:
            raise ValueError(
                f"initial_open_notional must be >= 0, got "
                f"{initial_open_notional!r}"
            )
        with self._lock:
            self.initial_equity = float(equity)
            self._open_notional = float(initial_open_notional)
        log.info(
            "SubBudgetTracker session start: equity=$%.2f "
            "budget=$%.2f open=$%.2f headroom=$%.2f",
            self.initial_equity, self.budget,
            self._open_notional, self.headroom,
        )

    # ── Reservations ──────────────────────────────────────────────

    def can_reserve(self, notional: float) -> bool:
        """True iff reserving ``notional`` would NOT exceed budget."""
        if notional < 0:
            raise ValueError(f"notional must be >= 0, got {notional!r}")
        with self._lock:
            return self._open_notional + notional <= self.budget

    def reserve(self, notional: float) -> bool:
        """Reserve ``notional`` against the budget.

        Returns True on success, False if it would exceed the ceiling.
        Atomic with the headroom check so concurrent callers can't
        over-commit.
        """
        if notional < 0:
            raise ValueError(f"notional must be >= 0, got {notional!r}")
        with self._lock:
            if self._open_notional + notional > self.budget:
                return False
            self._open_notional += notional
            return True

    def release(self, notional: float) -> None:
        """Release ``notional`` after a position closes.

        Clamped at zero — releases that exceed current open_notional
        log a warning but don't go negative.
        """
        if notional < 0:
            raise ValueError(f"notional must be >= 0, got {notional!r}")
        with self._lock:
            new_open = self._open_notional - notional
            if new_open < 0:
                log.warning(
                    "SubBudgetTracker: release of $%.2f would underflow "
                    "(open=$%.2f); clamping to 0",
                    notional, self._open_notional,
                )
                new_open = 0.0
            self._open_notional = new_open

    def __repr__(self) -> str:
        return (
            f"SubBudgetTracker(equity=${self.initial_equity:.2f}, "
            f"budget_pct={self.budget_pct:.0%}, "
            f"open=${self.open_notional:.2f}, "
            f"headroom=${self.headroom:.2f})"
        )
