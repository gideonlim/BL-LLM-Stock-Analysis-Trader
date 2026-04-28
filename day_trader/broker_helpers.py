"""Day-trader broker helpers.

The critical safety helper here is :func:`close_tagged_daytrade_qty`:
closes EXACTLY the day-trade qty for a ticker by canceling the
tagged bracket legs first, then submitting a market order with
the opposite side and exact qty.

It NEVER calls ``broker.close_position(ticker)`` — which would
liquidate any swing position in the same symbol. Same-account safety
hinges on this distinction.

The companion :func:`list_tagged_daytrade_orders` is a thin wrapper
that filters open orders by ``client_order_id`` prefix; used by
both this module and ``recovery.reconcile()``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Iterable, Optional

from trading_bot_bl.models import OrderResult

from day_trader.order_tags import (
    DT_PREFIX,
    is_day_trade_id,
    make_exit_order_id,
    parse_order_id,
)

log = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────


@dataclass
class CloseResult:
    """Outcome of :func:`close_tagged_daytrade_qty`."""

    ticker: str
    requested_qty: float
    cancelled_order_ids: list[str] = field(default_factory=list)
    exit_order: Optional[OrderResult] = None
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return (
            not self.error
            and self.exit_order is not None
            and self.exit_order.status == "submitted"
        )


# ── Order discovery ───────────────────────────────────────────────


def list_tagged_daytrade_orders(broker, ticker: Optional[str] = None) -> list:
    """Return open orders whose ``client_order_id`` starts with ``dt:``.

    Optionally filtered by ``ticker``. This is the canonical
    "find day-trade orders" lookup used by close mechanics, recovery,
    and the EOD watchdog.
    """
    open_orders = _fetch_open_orders(broker)
    out = []
    for o in open_orders:
        coid = _order_client_id(o)
        if not is_day_trade_id(coid):
            continue
        if ticker and _order_symbol(o) != ticker.upper():
            continue
        out.append(o)
    return out


# ── The safe close ────────────────────────────────────────────────


def close_tagged_daytrade_qty(
    broker,
    ticker: str,
    qty: float,
    *,
    side: str,
    parent_client_order_id: str,
    today: Optional[_date] = None,
    cancel_timeout_seconds: float = 5.0,
    cancel_poll_interval: float = 0.2,
) -> CloseResult:
    """Close EXACTLY the day-trade qty for ``ticker`` — never the whole position.

    Steps:

    1. List open ``dt:``-tagged orders for ``ticker`` (parent + SL/TP legs).
    2. Cancel each. Poll until status transitions to ``canceled`` or
       a terminal state (``filled``, ``expired``, …) within
       ``cancel_timeout_seconds``. Bracket legs reserve shares on
       Alpaca, so a qty-close before cancel may be rejected.
    3. Submit a fresh market order with the OPPOSITE side and exact
       ``qty``. Tag it with the parent's ``:exit`` suffix so the
       journal and recovery can correlate the closing leg.
    4. Update the journal status outside this function — this helper
       is concerned only with the broker side. The caller must
       record_close() into the journal.

    Args:
        broker: AlpacaBroker (or anything duck-typed compatibly).
        ticker: Symbol to close.
        qty: Exact number of shares to close (NOT notional).
        side: The direction we are EXITING from — ``"long"`` if
            we held long and need to sell, ``"short"`` if we held
            short and need to buy. The submitted close-order's side
            is the opposite.
        parent_client_order_id: The ``dt:yyyymmdd:seq:ticker`` tag
            from the entry order. We append ``:exit`` to derive the
            close-order's tag.
        today: Override the date used for the exit tag (testing).
        cancel_timeout_seconds: How long to wait for cancels to settle.
        cancel_poll_interval: Polling cadence for cancel confirmation.
    """
    if qty <= 0:
        return CloseResult(
            ticker=ticker, requested_qty=qty,
            error=f"qty must be positive, got {qty!r}",
        )
    if side not in ("long", "short"):
        return CloseResult(
            ticker=ticker, requested_qty=qty,
            error=f"side must be 'long' or 'short', got {side!r}",
        )
    if not is_day_trade_id(parent_client_order_id):
        return CloseResult(
            ticker=ticker, requested_qty=qty,
            error=(
                f"parent_client_order_id must be a day-trade tag "
                f"(starts with {DT_PREFIX!r}), got "
                f"{parent_client_order_id!r}"
            ),
        )

    result = CloseResult(ticker=ticker, requested_qty=qty)

    # ── 1. Find the parent + its bracket children ────────────────
    # Bracket SL/TP child legs do NOT inherit our `dt:` tag — alpaca-py
    # auto-generates their client_order_id server-side. So we can't
    # find them by prefix scan; we have to fetch all open orders for
    # the ticker and link by parent_id.
    all_orders = _fetch_open_orders(broker)
    ticker_orders = [
        o for o in all_orders if _order_symbol(o) == ticker.upper()
    ]
    related = _build_parent_family(ticker_orders, parent_client_order_id)

    # ── 2. Cancel each, then wait ────────────────────────────────
    cancelled_ids: list[str] = []
    for o in related:
        oid = str(getattr(o, "id", "") or "")
        if not oid:
            continue
        try:
            _cancel_order(broker, oid)
            cancelled_ids.append(oid)
        except Exception as exc:
            # Log but keep going — a leg might already be filled
            # or canceled, which is fine; we'll still submit the
            # qty-close below.
            log.warning(
                "close_tagged_daytrade_qty: cancel(%s) on %s failed: %s",
                oid, ticker, exc,
            )
    result.cancelled_order_ids = cancelled_ids

    if cancelled_ids:
        ok = _wait_for_cancels(
            broker, cancelled_ids,
            timeout=cancel_timeout_seconds,
            poll=cancel_poll_interval,
        )
        if not ok:
            log.warning(
                "close_tagged_daytrade_qty: %d cancel(s) on %s did not settle "
                "within %.1fs — proceeding anyway",
                len(cancelled_ids), ticker, cancel_timeout_seconds,
            )

    # ── 3. Submit the close ──────────────────────────────────────
    closing_side = "sell" if side == "long" else "buy"
    exit_id = make_exit_order_id(parent_client_order_id)
    try:
        exit_order = broker.submit_market_order(
            ticker=ticker,
            side=closing_side,
            qty=qty,
            time_in_force="day",
            client_order_id=exit_id,
        )
        result.exit_order = exit_order
        if exit_order.status != "submitted":
            result.error = (
                f"close-order rejected: {exit_order.error or 'unknown'}"
            )
    except Exception as exc:
        result.error = f"submit_market_order raised: {exc}"
        log.exception(
            "close_tagged_daytrade_qty: market close on %s failed", ticker,
        )

    return result


# ── Internal helpers ──────────────────────────────────────────────


def _build_parent_family(
    ticker_orders: Iterable, parent_client_order_id: str,
) -> list:
    """From all open orders on a ticker, return the parent + its bracket legs.

    Real Alpaca behaviour:

    - The parent order has ``client_order_id == parent_client_order_id``
      (our ``dt:`` tag) and a broker-generated ``id`` (UUID).
    - The bracket SL and TP child legs carry their own server-generated
      ``client_order_id`` (no ``dt:`` prefix) and have ``parent_id``
      set to the parent's broker ``id``.

    Some alpaca-py versions also expose the relationship via the parent's
    ``legs`` attribute. We don't depend on that here.

    Matching rules (in order):

    1. The parent itself — ``client_order_id == parent_client_order_id``.
    2. Any explicitly-tagged child we created — ``client_order_id``
       starts with ``parent_client_order_id + ":"`` (e.g. ``…:exit``).
    3. Bracket children whose ``parent_id`` equals either:
       - the parent's broker ``id`` (real Alpaca), or
       - the parent's ``client_order_id`` (some test/mock setups).

    Returns the matched orders. If no parent is found in the list,
    matches against the parent's ``client_order_id`` only — never
    falls back to "all ticker orders" (that could sweep up unrelated
    swing legs on the same symbol).
    """
    ticker_orders = list(ticker_orders)
    parent_prefix = parent_client_order_id + ":"

    # Pass 1: locate the parent
    parent = None
    for o in ticker_orders:
        if _order_client_id(o) == parent_client_order_id:
            parent = o
            break
    parent_broker_id = (
        str(getattr(parent, "id", "") or "") if parent is not None else ""
    )

    # Pass 2: collect family
    family: list = []
    for o in ticker_orders:
        coid = _order_client_id(o)
        po_parent_id = _order_parent_id(o)
        if (
            coid == parent_client_order_id
            or coid.startswith(parent_prefix)
            or (parent_broker_id and po_parent_id == parent_broker_id)
            or po_parent_id == parent_client_order_id
        ):
            family.append(o)
    return family


def _cancel_order(broker, order_id: str) -> None:
    """Cancel a single order via whichever method is available."""
    if hasattr(broker, "cancel_order_by_id"):
        broker.cancel_order_by_id(order_id)
        return
    client = getattr(broker, "_client", None)
    if client is None:
        raise RuntimeError("broker has no cancel-order method")
    client.cancel_order_by_id(order_id)


def _wait_for_cancels(
    broker,
    order_ids: list[str],
    *,
    timeout: float,
    poll: float,
) -> bool:
    """Poll until every order id is in a terminal state, or timeout."""
    deadline = time.monotonic() + timeout
    pending = set(order_ids)
    while pending and time.monotonic() < deadline:
        for oid in list(pending):
            status = _order_status(broker, oid)
            if status in (
                "canceled", "filled", "expired", "rejected", "done_for_day",
            ):
                pending.discard(oid)
        if pending:
            time.sleep(poll)
    return not pending


def _order_status(broker, order_id: str) -> str:
    """Best-effort fetch of an order's current status."""
    fetcher = (
        getattr(broker, "get_order_by_id", None)
        or getattr(getattr(broker, "_client", None), "get_order_by_id", None)
    )
    if fetcher is None:
        return ""
    try:
        order = fetcher(order_id)
        return str(getattr(order, "status", "") or "").lower()
    except Exception:
        return ""


def _fetch_open_orders(broker) -> list:
    """Fetch all open orders (across all tickers, all tags)."""
    if hasattr(broker, "list_open_orders"):
        return list(broker.list_open_orders())
    client = getattr(broker, "_client", None)
    if client is None:
        return []
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        return list(client.get_orders(filter=req))
    except Exception as exc:
        log.warning("broker_helpers: open-order fetch failed: %s", exc)
        return []


def _order_symbol(order) -> str:
    for attr in ("symbol", "ticker"):
        t = getattr(order, attr, None)
        if t:
            return str(t).upper()
    return ""


def _order_client_id(order) -> str:
    return str(getattr(order, "client_order_id", "") or "")


def _order_parent_id(order) -> str:
    """Either the broker-side ``parent_id`` (alpaca-py field) or empty."""
    for attr in ("parent_id", "parent_client_order_id"):
        v = getattr(order, attr, None)
        if v:
            return str(v)
    return ""
