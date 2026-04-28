#!/usr/bin/env python3
"""Standalone EOD-flatten watchdog.

Runs INDEPENDENTLY of the main daemon — if the daemon is dead, OOM-
killed, or partitioned from network, this script still connects to
Alpaca and force-closes all tagged day-trade positions before the
session closes.

Designed to be invoked every minute by a systemd timer during a broad
ET window (12:30–16:30 weekdays). The script:

1. Consults ``day_trader.calendar`` for today's NYSE session.
2. If no session, or outside the flatten window (close − 5 min to
   close), exits immediately (no-op).
3. Inside the window: enumerates open orders/positions tagged ``dt:``,
   calls ``close_tagged_daytrade_qty`` for each.
4. Idempotent — safe to run every minute; once flat, nothing to do.

Exit codes:
- 0: success (or no-op — not in window / no session)
- 1: error (failed to close one or more positions)

Usage::

    python -m deploy.scripts.eod_flatten
    # or directly:
    /opt/day-trader/repo/.venv/bin/python deploy/scripts/eod_flatten.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add repo root to path so imports work when invoked standalone
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] eod_flatten: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("eod_flatten")


def main() -> int:
    # Load env (credentials)
    try:
        from dotenv import load_dotenv
        for p in [Path("/etc/day-trader/env"), Path(".env")]:
            if p.exists():
                load_dotenv(p)
                break
    except ImportError:
        pass  # dotenv not required if env vars are already set

    from day_trader.calendar import (
        is_within_eod_flatten_window,
        now_et,
        session_for,
    )

    current = now_et()
    session = session_for(current.date())

    if session is None:
        log.debug("No NYSE session today — no-op")
        return 0

    if not is_within_eod_flatten_window(current, minutes_before_close=5):
        log.debug(
            "Outside EOD flatten window (close=%s, now=%s) — no-op",
            session.close_et.strftime("%H:%M"),
            current.strftime("%H:%M"),
        )
        return 0

    log.info(
        "EOD flatten window active (close=%s, now=%s) — checking positions",
        session.close_et.strftime("%H:%M"),
        current.strftime("%H:%M"),
    )

    # Connect to Alpaca
    from trading_bot_bl.config import AlpacaConfig
    from trading_bot_bl.broker import AlpacaBroker

    try:
        alpaca_config = AlpacaConfig.from_env()
        broker = AlpacaBroker(alpaca_config)
    except Exception:
        log.exception("Failed to connect to Alpaca")
        return 1

    # Find tagged day-trade orders
    from day_trader.broker_helpers import (
        close_tagged_daytrade_qty,
        list_tagged_daytrade_orders,
    )
    from day_trader.order_tags import parse_order_id

    tagged = list_tagged_daytrade_orders(broker)
    if not tagged:
        log.info("No tagged day-trade orders — already flat")
        return 0

    # Group by ticker
    by_ticker: dict[str, list] = {}
    for order in tagged:
        sym = str(getattr(order, "symbol", "")).upper()
        if sym:
            by_ticker.setdefault(sym, []).append(order)

    # Also check journal for open day-trade entries to get qty
    from day_trader.config import DayTradeConfig
    config = DayTradeConfig.from_env()
    from trading_bot_bl.journal import load_open_trades

    open_entries = load_open_trades(config.journal_dir)
    daytrade_entries = {
        e.ticker: e for e in open_entries
        if (e.trade_type or "swing") == "daytrade"
    }

    errors = 0
    for ticker, orders in by_ticker.items():
        # Determine qty from journal
        entry = daytrade_entries.get(ticker)
        if entry and entry.entry_qty > 0:
            qty = entry.entry_qty
            side = entry.side
        else:
            # Fallback: can't determine qty from journal — skip.
            # We must NOT guess; a wrong qty close could over-sell
            # or leave orphaned shares.
            log.warning(
                "EOD flatten: tagged orders for %s but no journal "
                "entry with qty — skipping (requires manual cleanup)",
                ticker,
            )
            errors += 1
            continue

        # Find the parent client_order_id
        parent_id = ""
        for o in orders:
            coid = str(getattr(o, "client_order_id", "") or "")
            parsed = parse_order_id(coid)
            if parsed and not parsed.is_exit:
                parent_id = coid
                break
        if not parent_id:
            # Use the entry_order_id from journal
            parent_id = entry.entry_order_id if entry else ""
        if not parent_id:
            log.warning(
                "EOD flatten: cannot determine parent_id for %s — "
                "skipping (requires manual cleanup)",
                ticker,
            )
            errors += 1
            continue

        log.info(
            "EOD flatten: closing %s %d sh (side=%s, tag=%s)",
            ticker, qty, side, parent_id,
        )
        result = close_tagged_daytrade_qty(
            broker, ticker, qty=qty,
            side=side,
            parent_client_order_id=parent_id,
        )
        if result.succeeded:
            log.info("EOD flatten: %s closed successfully", ticker)
        else:
            log.error(
                "EOD flatten: %s failed: %s", ticker, result.error,
            )
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
