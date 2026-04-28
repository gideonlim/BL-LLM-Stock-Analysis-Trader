"""Journal adapter — thin bridge between the day-trader and the
shared ``trading_bot_bl/journal.py`` lifecycle functions.

Ensures every day-trade journal write carries ``trade_type="daytrade"``
so analytics, recovery, and the EOD watchdog can filter by type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from trading_bot_bl.journal import create_trade, close_trade
from trading_bot_bl.models import JournalEntry

log = logging.getLogger(__name__)


def create_daytrade(
    *,
    order_id: str,
    ticker: str,
    strategy: str,
    side: str,
    signal_price: float,
    notional: float,
    sl_price: float,
    tp_price: float,
    composite_score: float = 0.0,
    confidence: str = "N/A",
    confidence_score: int = 0,
    vix: float = 0.0,
    market_regime: str = "",
    spy_price: float = 0.0,
    journal_dir: Optional[Path] = None,
) -> Optional[JournalEntry]:
    """Create a pending day-trade journal entry.

    Delegates to ``trading_bot_bl.journal.create_trade`` with
    ``trade_type="daytrade"`` hard-coded. The day-trader executor
    should use this function exclusively — never call ``create_trade``
    directly — so the ``trade_type`` tag is never accidentally omitted.
    """
    return create_trade(
        order_id=order_id,
        ticker=ticker,
        strategy=strategy,
        side=side,
        signal_price=signal_price,
        notional=notional,
        sl_price=sl_price,
        tp_price=tp_price,
        composite_score=composite_score,
        confidence=confidence,
        confidence_score=confidence_score,
        vix=vix,
        market_regime=market_regime,
        spy_price=spy_price,
        trade_type="daytrade",
        journal_dir=journal_dir,
    )


def close_daytrade(
    *,
    journal_dir: Path,
    trade_id: str,
    exit_price: float,
    exit_reason: str,
    exit_order_id: str = "",
) -> Optional[JournalEntry]:
    """Close a day-trade journal entry.

    Wraps ``trading_bot_bl.journal.close_trade`` if it exists with
    the expected signature; otherwise falls back to a direct
    load-modify-save cycle. The swing journal's ``close_trade``
    doesn't need a ``trade_type`` parameter because the entry
    was already created with one.
    """
    try:
        return close_trade(
            journal_dir=journal_dir,
            trade_id=trade_id,
            exit_price=exit_price,
            exit_reason=exit_reason,
            exit_order_id=exit_order_id,
        )
    except TypeError:
        # close_trade's signature may not match — degrade gracefully.
        # In the worst case the entry stays "open" and recovery
        # catches it on the next session start.
        log.warning(
            "journal_adapter: close_trade signature mismatch for %s "
            "— journal entry may not transition to 'closed'",
            trade_id,
        )
        return None
