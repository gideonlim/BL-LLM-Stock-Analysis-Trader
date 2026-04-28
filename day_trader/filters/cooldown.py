"""CooldownTracker — no-revenge re-entry primitive.

Tracks per-ticker and per-strategy cooldowns triggered by losing
trades. Used by both:

- :class:`day_trader.filters.cooldown_filter.CooldownFilter` to gate
  scan-time entries.
- :class:`day_trader.risk.DayRiskManager` which calls
  :meth:`record_close` on every closed trade, populating cooldowns.

Two cooldowns active at once:

- **Ticker** cooldown: after a losing trade on a ticker, that ticker
  is benched for ``ticker_minutes``.
- **Strategy** cooldown: after a losing trade by a strategy (any ticker),
  that strategy is paused entire-strategy-wide for ``strategy_minutes``.

The strategy cooldown prevents revenge cycles where a losing strategy
keeps re-firing on adjacent tickers. The user emphasised this in the
filtering-first philosophy — without it, "even a good setup turns into
expensive noise."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class CooldownTracker:
    """Thread-safe per-ticker / per-strategy cooldown bookkeeping."""

    ticker_minutes: int = 60
    strategy_minutes: int = 30

    _ticker_cooldowns: dict[str, datetime] = field(default_factory=dict)
    _strategy_cooldowns: dict[str, datetime] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    # ── Recording ────────────────────────────────────────────────

    def record_close(
        self,
        *,
        ticker: str,
        strategy: str,
        pnl: float,
        when: Optional[datetime] = None,
    ) -> None:
        """Update cooldowns based on a closed trade's P&L.

        Only losses (``pnl < 0``) trigger cooldowns. Wins and
        breakevens do not — we don't want to bench winning setups.
        """
        if pnl >= 0:
            return
        when = when or datetime.now()
        ticker = ticker.upper()
        with self._lock:
            self._ticker_cooldowns[ticker] = when + timedelta(
                minutes=self.ticker_minutes
            )
            self._strategy_cooldowns[strategy] = when + timedelta(
                minutes=self.strategy_minutes
            )
        log.info(
            "Cooldown set: %s for %d min, strategy %s for %d min "
            "(triggering loss: $%.2f)",
            ticker, self.ticker_minutes,
            strategy, self.strategy_minutes,
            pnl,
        )

    # ── Querying ─────────────────────────────────────────────────

    def is_cooled_down(
        self,
        *,
        ticker: str,
        strategy: str,
        now: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """Check both ticker and strategy cooldowns.

        Returns ``(False, "")`` if neither is active (i.e. it's OK
        to enter), ``(True, reason)`` if either is active.

        Reasons:
        - ``"ticker_cooldown"`` — ticker-level cooldown still active
        - ``"strategy_cooldown"`` — strategy-level cooldown still active
        """
        now = now or datetime.now()
        ticker = ticker.upper()
        with self._lock:
            ticker_until = self._ticker_cooldowns.get(ticker)
            strategy_until = self._strategy_cooldowns.get(strategy)
        if ticker_until and now < ticker_until:
            return True, "ticker_cooldown"
        if strategy_until and now < strategy_until:
            return True, "strategy_cooldown"
        return False, ""

    def cooldown_remaining(
        self,
        *,
        ticker: str,
        strategy: str,
        now: Optional[datetime] = None,
    ) -> Optional[timedelta]:
        """Time remaining on whichever cooldown is the longer-active.
        ``None`` if neither is active."""
        now = now or datetime.now()
        ticker = ticker.upper()
        with self._lock:
            ticker_until = self._ticker_cooldowns.get(ticker)
            strategy_until = self._strategy_cooldowns.get(strategy)
        candidates = []
        if ticker_until and now < ticker_until:
            candidates.append(ticker_until - now)
        if strategy_until and now < strategy_until:
            candidates.append(strategy_until - now)
        if not candidates:
            return None
        return max(candidates)

    # ── Maintenance ──────────────────────────────────────────────

    def prune_expired(self, *, now: Optional[datetime] = None) -> int:
        """Drop entries whose cooldown has elapsed. Returns the number
        of entries pruned. Called periodically by the executor; not
        required for correctness (queries already check expiry)."""
        now = now or datetime.now()
        pruned = 0
        with self._lock:
            for d in (self._ticker_cooldowns, self._strategy_cooldowns):
                expired = [k for k, until in d.items() if until <= now]
                for k in expired:
                    del d[k]
                pruned += len(expired)
        return pruned

    def reset_for_session(self) -> None:
        """Clear all cooldowns. Day-trader cooldowns do NOT span
        sessions — yesterday's losses don't gate today's entries.
        Called from ``DayRiskManager.start_session()``."""
        with self._lock:
            self._ticker_cooldowns.clear()
            self._strategy_cooldowns.clear()
