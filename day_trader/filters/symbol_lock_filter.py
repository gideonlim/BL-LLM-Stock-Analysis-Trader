"""SymbolLockFilter — block tickers held/pending by either bot.

Wraps :class:`day_trader.symbol_locks.SymbolLock`. First filter in
the chain: cheapest possible check (a hash lookup against a
periodically-refreshed snapshot)."""

from __future__ import annotations

from day_trader.filters.base import Filter
from day_trader.models import FilterContext
from day_trader.symbol_locks import SymbolLock


class SymbolLockFilter(Filter):
    """Reject any signal on a ticker the swing bot is in OR that the
    day-trader already has exposure to."""

    name = "symbol_lock"

    def __init__(self, symbol_lock: SymbolLock):
        self.symbol_lock = symbol_lock

    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        if ctx.signal is None:
            return False, "no_signal"
        verdict = self.symbol_lock.is_locked(ctx.signal.ticker)
        if verdict.locked:
            return False, verdict.reason
        return True, ""
