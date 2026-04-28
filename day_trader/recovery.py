"""Session-start recovery and reconciliation.

Joins three sources of truth and verifies they agree:

1. **Journal** — open ``trade_type='daytrade'`` entries on disk
   (status ``'open'`` or ``'pending'``). Lives in
   ``execution_logs/journal/``.
2. **Tagged orders** — open Alpaca orders whose ``client_order_id``
   starts with ``dt:``. The only mechanism for separating day-trade
   orders from swing orders on a same-account broker.
3. **Positions** — net Alpaca positions per ticker. Not tagged.

Failure modes we look for:

- Open journal entry has no matching tagged order *and* no matching
  position contribution → likely closed externally; or partial fill
  the daemon missed.
- Tagged order on Alpaca has no matching journal entry → orphan;
  could be a leg from a prior session or an external bracket.
- Position contribution doesn't match the journal's recorded qty.

Any mismatch → :class:`ReconcileResult.is_clean = False` and the
executor enters INCIDENT MODE: refuses all new entries for the day,
fires a Telegram alert, leaves human triage to resolve the state.
This is intentional — auto-healing under inconsistent state can
amplify a small bug into liquidating the wrong position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from trading_bot_bl.journal import load_open_trades
from trading_bot_bl.models import JournalEntry

from day_trader.broker_helpers import list_tagged_daytrade_orders
from day_trader.order_tags import is_day_trade_id, parse_order_id

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────


@dataclass
class ReconcileIncident:
    """One concrete inconsistency found during reconciliation."""

    kind: str          # "orphan_journal" | "orphan_order" | "qty_mismatch"
    ticker: str
    detail: str
    journal_entry: Optional[JournalEntry] = None
    order_id: str = ""


@dataclass
class ReconcileResult:
    """Output of :func:`reconcile`. Pass to ``SubBudgetTracker.start_session``."""

    open_journal_entries: list[JournalEntry] = field(default_factory=list)
    tagged_orders: list = field(default_factory=list)
    daytrade_position_qty: dict[str, float] = field(default_factory=dict)
    incidents: list[ReconcileIncident] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.incidents

    @property
    def open_notional(self) -> float:
        """Sum of entry notionals for all open day-trade journal entries.

        Used by ``SubBudgetTracker.start_session`` to seed the
        initial open exposure. Computed from the journal — the
        only authoritative source for trade_type tagging.
        """
        return sum(
            e.entry_notional or 0.0 for e in self.open_journal_entries
        )

    def summary(self) -> str:
        """One-line operator summary for logs / Telegram."""
        if self.is_clean:
            return (
                f"recovery clean: "
                f"{len(self.open_journal_entries)} open day-trade entries, "
                f"{len(self.tagged_orders)} tagged orders, "
                f"open=${self.open_notional:,.2f}"
            )
        return (
            f"INCIDENT MODE: {len(self.incidents)} inconsistencies "
            f"(journal={len(self.open_journal_entries)}, "
            f"tagged_orders={len(self.tagged_orders)}, "
            f"open=${self.open_notional:,.2f}). "
            f"Reasons: "
            + "; ".join(f"{i.kind}({i.ticker}: {i.detail})" for i in self.incidents[:3])
            + ("…" if len(self.incidents) > 3 else "")
        )


# ── Reconcile entry point ─────────────────────────────────────────


def reconcile(
    broker,
    journal_dir: Path,
    *,
    qty_tolerance: float = 0.001,
) -> ReconcileResult:
    """Run the three-way reconcile.

    Args:
        broker: AlpacaBroker (or duck-typed compatible).
        journal_dir: Directory containing per-trade JSON files.
        qty_tolerance: Tolerance when comparing journal qty to net
            position contribution. Float-equality on share counts
            is unsafe; ``0.001`` allows for rounding.

    Returns:
        :class:`ReconcileResult`. Always returns even when there are
        incidents — the executor is responsible for treating
        ``is_clean=False`` as an incident-mode trigger.
    """
    journal_dir = Path(journal_dir)

    # ── 1. Open day-trade journal entries ────────────────────────
    all_open = load_open_trades(journal_dir)
    open_daytrades = [
        e for e in all_open if (e.trade_type or "swing") == "daytrade"
    ]
    by_ticker_journal: dict[str, list[JournalEntry]] = {}
    for e in open_daytrades:
        by_ticker_journal.setdefault(e.ticker, []).append(e)

    # ── 2. Tagged open orders ────────────────────────────────────
    tagged_orders = list_tagged_daytrade_orders(broker)
    by_ticker_orders: dict[str, list] = {}
    for o in tagged_orders:
        sym = _order_symbol(o)
        if not sym:
            continue
        by_ticker_orders.setdefault(sym, []).append(o)

    # ── 3. Positions (net, untagged) ─────────────────────────────
    portfolio = broker.get_portfolio()
    net_positions = {
        ticker.upper(): float(info.get("qty", 0.0) or 0.0)
        for ticker, info in (portfolio.positions or {}).items()
    }

    # ── 4. Cross-check ────────────────────────────────────────────
    incidents: list[ReconcileIncident] = []
    daytrade_position_qty: dict[str, float] = {}

    # Tickers we have ANY day-trade signal for (journal or tagged orders)
    candidate_tickers = set(by_ticker_journal) | set(by_ticker_orders)

    for ticker in candidate_tickers:
        j_entries = by_ticker_journal.get(ticker, [])
        t_orders = by_ticker_orders.get(ticker, [])
        net_qty = net_positions.get(ticker, 0.0)

        journal_qty = sum(e.entry_qty or 0.0 for e in j_entries)
        tagged_open_count = len(t_orders)

        if j_entries and not t_orders and abs(net_qty) < qty_tolerance:
            # Journal says we have an open day-trade, but the broker
            # has neither a matching order nor a position. Most likely:
            # someone closed the position externally, or the journal
            # never got the close notification.
            incidents.append(ReconcileIncident(
                kind="orphan_journal",
                ticker=ticker,
                detail=(
                    f"journal has {len(j_entries)} open entry(ies) "
                    f"({journal_qty:g} sh) but no broker position or "
                    f"tagged order"
                ),
                journal_entry=j_entries[0],
            ))
            continue

        if not j_entries and t_orders:
            # Tagged orders on the broker with no journal counterpart.
            # Possible if a previous daemon crashed after submitting
            # an order but before writing the journal entry.
            for o in t_orders:
                incidents.append(ReconcileIncident(
                    kind="orphan_order",
                    ticker=ticker,
                    detail=(
                        f"tagged order {_order_client_id(o)!r} has "
                        f"no journal entry"
                    ),
                    order_id=str(getattr(o, "id", "") or ""),
                ))
            continue

        if j_entries and abs(net_qty) > qty_tolerance:
            # We have journal entries AND a net position. The position
            # is shared with swing — we can't directly verify the
            # day-trade qty in isolation, but we CAN verify that the
            # journal's recorded qty is plausible: it must not exceed
            # the broker's net position in absolute value (otherwise
            # something has been closed without our knowledge).
            if journal_qty > abs(net_qty) + qty_tolerance:
                incidents.append(ReconcileIncident(
                    kind="qty_mismatch",
                    ticker=ticker,
                    detail=(
                        f"journal qty {journal_qty:g} exceeds broker "
                        f"net position |{net_qty:g}|"
                    ),
                    journal_entry=j_entries[0],
                ))
                continue

        # Clean for this ticker. Record the journal-claimed qty as
        # the day-trade portion (recall: positions aren't tagged, so
        # the journal is authoritative on what's "ours").
        daytrade_position_qty[ticker] = journal_qty

    return ReconcileResult(
        open_journal_entries=open_daytrades,
        tagged_orders=tagged_orders,
        daytrade_position_qty=daytrade_position_qty,
        incidents=incidents,
    )


# ── Tiny helpers ──────────────────────────────────────────────────


def _order_symbol(order) -> str:
    for attr in ("symbol", "ticker"):
        t = getattr(order, attr, None)
        if t:
            return str(t).upper()
    return ""


def _order_client_id(order) -> str:
    return str(getattr(order, "client_order_id", "") or "")
