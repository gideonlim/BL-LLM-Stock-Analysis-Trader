"""Trade journal -- lifecycle tracking for every trade from entry to exit.

All public functions are **non-critical**: they catch and log exceptions
internally so the core trading pipeline (executor / monitor) is never
disrupted by a journal failure.  Callers can fire-and-forget.

Persistence: one JSON file per trade in ``execution_logs/journal/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from trading_bot_bl.models import JournalEntry

log = logging.getLogger(__name__)


# ── Serialisation helpers ─────────────────────────────────────────


def _entry_to_dict(entry: JournalEntry) -> dict:
    """Convert a JournalEntry to a JSON-safe dict."""
    return asdict(entry)


def _dict_to_entry(d: dict) -> JournalEntry:
    """Reconstruct a JournalEntry from a dict (loaded from JSON).

    Drops ``None`` values so the dataclass default (typically ``0.0``)
    applies instead — this prevents ``NoneType`` format errors when
    optional numeric fields were serialised as ``null``.
    """
    return JournalEntry(**{
        k: v for k, v in d.items()
        if k in JournalEntry.__dataclass_fields__ and v is not None
    })


def _save_entry(entry: JournalEntry, journal_dir: Path) -> None:
    """Write a single journal entry to disk."""
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = journal_dir / f"{entry.trade_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_entry_to_dict(entry), f, indent=2)


# ── Load helpers ──────────────────────────────────────────────────


def load_open_trades(journal_dir: Path) -> list[JournalEntry]:
    """Load all journal entries with status 'open' or 'pending'."""
    entries: list[JournalEntry] = []
    if not journal_dir.exists():
        return entries
    for path in journal_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            entry = _dict_to_entry(d)
            if entry.status in ("open", "pending"):
                entries.append(entry)
        except Exception as exc:
            log.debug(f"Skipping corrupt journal file {path}: {exc}")
    return entries


def load_all_trades(
    journal_dir: Path,
    lookback_days: int = 90,
) -> list[JournalEntry]:
    """Load all journal entries within the lookback window."""
    entries: list[JournalEntry] = []
    if not journal_dir.exists():
        return entries
    cutoff = datetime.now().isoformat()[:10]
    if lookback_days > 0:
        from datetime import timedelta
        cutoff = (
            datetime.now() - timedelta(days=lookback_days)
        ).isoformat()[:10]
    else:
        cutoff = ""

    for path in journal_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            entry = _dict_to_entry(d)
            # Include if entry date is within window or unknown
            entry_dt = entry.entry_date[:10] if entry.entry_date else ""
            if not cutoff or not entry_dt or entry_dt >= cutoff:
                entries.append(entry)
        except Exception as exc:
            log.debug(f"Skipping corrupt journal file {path}: {exc}")
    return entries


# ── Core lifecycle operations ─────────────────────────────────────


def create_trade(
    *,
    order_id: str,
    ticker: str,
    strategy: str,
    side: str,
    signal_price: float,
    notional: float,
    sl_price: float,
    tp_price: float,
    composite_score: float,
    confidence: str,
    confidence_score: int,
    vix: float = 0.0,
    market_regime: str = "",
    spy_price: float = 0.0,
    journal_dir: Path | None = None,
) -> JournalEntry | None:
    """Create a new pending trade journal entry.

    Called from executor.py after a bracket order is submitted.
    Returns the entry on success, None on failure.
    """
    try:
        trade_id = f"{ticker}_{order_id[:8]}"
        now_iso = datetime.now().isoformat(timespec="seconds")

        entry = JournalEntry(
            trade_id=trade_id,
            ticker=ticker,
            strategy=strategy,
            side="long" if side == "buy" else "short",
            entry_order_id=order_id,
            entry_signal_price=signal_price,
            entry_notional=notional,
            original_sl_price=sl_price,
            original_tp_price=tp_price,
            entry_composite_score=composite_score,
            entry_confidence=confidence,
            entry_confidence_score=confidence_score,
            entry_vix=vix,
            entry_market_regime=market_regime,
            entry_spy_price=spy_price,
            status="pending",
            opened_at=now_iso,
        )

        if journal_dir:
            _save_entry(entry, journal_dir)
            log.info(
                f"  Journal: created pending trade {trade_id}"
            )
        return entry

    except Exception as exc:
        log.warning(f"Journal: failed to create trade: {exc}")
        return None


def resolve_pending_trades(
    broker: object,
    journal_dir: Path,
) -> int:
    """Check pending entries and confirm or cancel them.

    Queries Alpaca for each pending order's fill status.
    Returns the number of entries that transitioned to 'open'.
    """
    resolved = 0
    try:
        entries = load_open_trades(journal_dir)
        pending = [e for e in entries if e.status == "pending"]
        if not pending:
            return 0

        for entry in pending:
            try:
                fill_price, fill_date, fill_qty = _resolve_fill(
                    broker, entry.entry_order_id
                )

                if fill_price == -1.0:
                    # Order cancelled / expired / rejected
                    entry.status = "closed"
                    entry.exit_reason = "order_cancelled"
                    entry.closed_at = datetime.now().isoformat(
                        timespec="seconds"
                    )
                    _save_entry(entry, journal_dir)
                    log.info(
                        f"  Journal: {entry.trade_id} order "
                        f"cancelled/expired — closed"
                    )
                elif fill_price > 0:
                    # Order filled — transition to open
                    entry.status = "open"
                    entry.entry_fill_price = fill_price
                    entry.entry_qty = fill_qty
                    entry.entry_date = fill_date
                    entry.entry_notional = round(
                        fill_price * fill_qty, 2
                    )

                    # Compute entry slippage
                    if entry.entry_signal_price > 0:
                        entry.entry_slippage = round(
                            fill_price - entry.entry_signal_price,
                            4,
                        )
                        entry.entry_slippage_pct = round(
                            entry.entry_slippage
                            / entry.entry_signal_price
                            * 100,
                            4,
                        )

                    # Compute initial risk/reward
                    if entry.original_sl_price > 0:
                        entry.initial_risk_per_share = round(
                            fill_price - entry.original_sl_price, 4
                        )
                        entry.initial_risk_dollars = round(
                            entry.initial_risk_per_share
                            * fill_qty,
                            2,
                        )
                    if entry.original_tp_price > 0:
                        entry.initial_reward_dollars = round(
                            (entry.original_tp_price - fill_price)
                            * fill_qty,
                            2,
                        )
                    if entry.initial_risk_dollars > 0:
                        entry.planned_rr_ratio = round(
                            entry.initial_reward_dollars
                            / entry.initial_risk_dollars,
                            2,
                        )

                    # Initialise excursion to fill price
                    entry.max_favorable_excursion = fill_price
                    entry.max_adverse_excursion = fill_price

                    _save_entry(entry, journal_dir)
                    resolved += 1
                    log.info(
                        f"  Journal: {entry.trade_id} filled "
                        f"@ ${fill_price:.2f} "
                        f"(slippage: {entry.entry_slippage_pct:+.2f}%)"
                    )
                # else: still pending, leave as-is

            except Exception as exc:
                log.debug(
                    f"Journal: resolve failed for "
                    f"{entry.trade_id}: {exc}"
                )

    except Exception as exc:
        log.warning(f"Journal: resolve_pending_trades failed: {exc}")

    return resolved


def update_trade(
    entry: JournalEntry,
    current_price: float,
    journal_dir: Path,
    current_date: str = "",
) -> None:
    """Update excursion tracking and price samples for an open trade.

    Called from monitor.py on every monitoring pass.
    """
    try:
        if entry.status != "open" or entry.entry_fill_price <= 0:
            return

        if not current_date:
            current_date = datetime.now().isoformat(
                timespec="seconds"
            )

        fill = entry.entry_fill_price

        # Update MFE
        if current_price > entry.max_favorable_excursion:
            entry.max_favorable_excursion = current_price
            entry.mfe_pct = round(
                (current_price - fill) / fill * 100, 2
            )
            entry.mfe_date = current_date[:10]

        # Update MAE
        if (
            current_price < entry.max_adverse_excursion
            or entry.max_adverse_excursion <= 0
        ):
            entry.max_adverse_excursion = current_price
            entry.mae_pct = round(
                (fill - current_price) / fill * 100, 2
            )
            entry.mae_date = current_date[:10]

        # Append price sample (one per call, deduplicate by date)
        sample_date = current_date[:10]
        existing_dates = {
            s.get("date", "")[:10] for s in entry.price_samples
        }
        if sample_date not in existing_dates:
            entry.price_samples.append({
                "date": sample_date,
                "price": round(current_price, 2),
            })

        _save_entry(entry, journal_dir)

    except Exception as exc:
        log.debug(
            f"Journal: update_trade failed for "
            f"{entry.trade_id}: {exc}"
        )


def record_sl_modification(
    entry: JournalEntry,
    old_sl: float,
    new_sl: float,
    reason: str,
    current_price: float,
    journal_dir: Path,
) -> None:
    """Record a stop-loss modification in the journal."""
    try:
        entry.sl_modifications.append({
            "date": datetime.now().isoformat(timespec="seconds"),
            "old_sl": round(old_sl, 2),
            "new_sl": round(new_sl, 2),
            "reason": reason,
            "price_at_modification": round(current_price, 2),
        })
        _save_entry(entry, journal_dir)
        log.debug(
            f"  Journal: SL mod recorded for {entry.trade_id} "
            f"${old_sl:.2f} -> ${new_sl:.2f} ({reason})"
        )
    except Exception as exc:
        log.debug(
            f"Journal: record_sl_modification failed: {exc}"
        )


def close_trade(
    entry: JournalEntry,
    exit_price: float,
    exit_reason: str,
    journal_dir: Path,
    exit_order_id: str = "",
    expected_exit_price: float = 0.0,
) -> None:
    """Finalise a trade with exit data and outcome metrics.

    Called when a position disappears from the portfolio or when
    the monitor takes a close action (emergency, time exit, gap).
    """
    try:
        if entry.status == "closed":
            return  # already closed, skip

        now_iso = datetime.now().isoformat(timespec="seconds")
        fill = entry.entry_fill_price or entry.entry_signal_price

        entry.exit_price = exit_price
        entry.exit_fill_price = exit_price
        entry.exit_date = now_iso
        entry.exit_reason = exit_reason
        entry.exit_order_id = exit_order_id
        entry.status = "closed"
        entry.closed_at = now_iso

        # Exit slippage (SL/TP slippage)
        if expected_exit_price > 0:
            entry.exit_slippage = round(
                expected_exit_price - exit_price, 4
            )

        # Realized P&L
        if fill > 0:
            pnl_per_share = exit_price - fill
            qty = entry.entry_qty or 1.0
            entry.realized_pnl = round(pnl_per_share * qty, 2)
            entry.realized_pnl_pct = round(
                pnl_per_share / fill * 100, 2
            )

        # R-multiple
        if entry.initial_risk_dollars > 0:
            entry.r_multiple = round(
                entry.realized_pnl / entry.initial_risk_dollars,
                2,
            )

        # Holding period
        if entry.entry_date:
            try:
                entry_d = _parse_date(entry.entry_date)
                entry.holding_days = (date.today() - entry_d).days
            except (ValueError, TypeError):
                pass

        # ETD (End Trade Drawdown): profit given back from MFE
        if entry.max_favorable_excursion > 0 and fill > 0:
            entry.etd = round(
                entry.max_favorable_excursion - exit_price, 2
            )
            entry.etd_pct = round(
                entry.etd / fill * 100, 2
            )

        # Edge ratio: MFE / MAE
        if entry.mae_pct > 0:
            entry.edge_ratio = round(
                entry.mfe_pct / entry.mae_pct, 2
            )

        _save_entry(entry, journal_dir)
        pnl = entry.realized_pnl or 0.0
        pnl_pct = entry.realized_pnl_pct or 0.0
        r_mul = entry.r_multiple or 0.0
        log.info(
            f"  Journal: closed {entry.trade_id} — "
            f"reason={exit_reason}, "
            f"P&L=${pnl:+.2f} "
            f"({pnl_pct:+.1f}%), "
            f"R={r_mul:+.2f}"
        )

    except Exception as exc:
        log.warning(
            f"Journal: close_trade failed for "
            f"{entry.trade_id}: {exc}"
        )


def _reconcile_premature_closes(
    current_positions: dict[str, dict],
    journal_dir: Path,
) -> int:
    """Revert journal entries that were prematurely marked closed.

    If a journal entry has ``status='closed'`` but the ticker is
    still held in the portfolio, the close order never actually
    filled.  Reset the entry to ``open`` so the next monitor run
    can re-attempt the close properly.

    Returns the number of entries reverted.
    """
    reverted = 0
    if not journal_dir or not journal_dir.exists():
        return reverted

    for path in journal_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("status") != "closed":
                continue
            ticker = d.get("ticker", "")
            if ticker not in current_positions:
                continue  # correctly closed — position is gone

            # Position still held but journal says closed → revert
            d["status"] = "open"
            d["closed_at"] = ""
            d["exit_price"] = 0.0
            d["exit_date"] = ""
            d["exit_reason"] = ""
            d["exit_order_id"] = ""
            d["exit_fill_price"] = 0.0
            d["exit_slippage"] = None
            d["realized_pnl"] = 0.0
            d["realized_pnl_pct"] = 0.0
            d["r_multiple"] = None
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2)

            log.warning(
                f"  Journal: reverted {d.get('trade_id')} to open "
                f"— position {ticker} still held"
            )
            reverted += 1
        except Exception:
            continue
    return reverted


def detect_closed_trades(
    current_positions: dict[str, dict],
    journal_dir: Path,
    broker: object | None = None,
) -> list[JournalEntry]:
    """Detect trades that closed since the last run.

    Compares open journal entries against current portfolio
    positions.  Any entry whose ticker no longer appears in
    positions is considered closed.

    First runs a reconciliation pass: if any journal entry is
    marked ``closed`` but the ticker is still in the portfolio
    (i.e. the close order never filled), it reverts the entry
    to ``open``.

    If a broker is provided, queries Alpaca for exit details
    (fill price, exit reason).  Otherwise uses the last known
    price from the entry's price_samples.

    Returns list of newly closed entries.
    """
    # Safety: revert prematurely closed entries
    reverted = _reconcile_premature_closes(
        current_positions, journal_dir,
    )
    if reverted:
        log.info(
            f"  Journal: reconciled {reverted} premature close(s)"
        )

    closed: list[JournalEntry] = []
    try:
        open_entries = load_open_trades(journal_dir)
        active = [e for e in open_entries if e.status == "open"]

        for entry in active:
            if entry.ticker in current_positions:
                continue  # still held

            # Position gone — determine exit details
            exit_price = 0.0
            exit_reason = "unknown"
            expected_exit = 0.0

            if broker is not None:
                ep, er, exp = _query_exit_details(
                    broker, entry
                )
                if ep > 0:
                    exit_price = ep
                    exit_reason = er
                    expected_exit = exp

            # Fallback: use last price sample
            if exit_price <= 0 and entry.price_samples:
                exit_price = entry.price_samples[-1].get(
                    "price", 0.0
                )
                if exit_reason == "unknown":
                    exit_reason = "detected_closed"

            # Fallback: use last known MFE/MAE midpoint
            if exit_price <= 0:
                exit_price = entry.entry_fill_price or 0.0

            close_trade(
                entry,
                exit_price=exit_price,
                exit_reason=exit_reason,
                journal_dir=journal_dir,
                expected_exit_price=expected_exit,
            )
            closed.append(entry)

    except Exception as exc:
        log.warning(
            f"Journal: detect_closed_trades failed: {exc}"
        )

    if closed:
        log.info(
            f"  Journal: detected {len(closed)} closed trade(s): "
            f"{[e.ticker for e in closed]}"
        )
    return closed


# ── Migration ─────────────────────────────────────────────────────


def _lookup_strategy_from_logs(
    ticker: str,
    log_dir: Path,
) -> str:
    """Search execution logs for the strategy that placed a trade.

    Scans execution_*.json files (newest first) for a submitted BUY
    order matching the ticker.  Returns the strategy name, or
    "unknown" if not found.
    """
    try:
        log_files = sorted(
            log_dir.glob("execution_*.json"), reverse=True
        )
        for f in log_files:
            data = json.loads(f.read_text())
            for order in data.get("orders", []):
                if (
                    order.get("ticker") == ticker
                    and order.get("status") == "submitted"
                    and order.get("side") == "buy"
                    and order.get("strategy")
                ):
                    return order["strategy"]
    except Exception:
        pass
    return "unknown"


def migrate_existing_positions(
    positions: dict[str, dict],
    open_orders: list,
    journal_dir: Path,
    vix: float = 0.0,
    market_regime: str = "",
    spy_price: float = 0.0,
) -> int:
    """Create journal entries for positions that pre-date the journal.

    Only creates entries for positions that don't already have a
    matching journal entry.  Attempts to look up the original
    strategy from execution logs before falling back to "unknown".
    Returns count of entries created.
    """
    created = 0
    try:
        existing = load_open_trades(journal_dir)
        existing_tickers = {e.ticker for e in existing}

        for ticker, pos in positions.items():
            if ticker in existing_tickers:
                continue

            avg_entry = pos.get("avg_entry", 0.0)
            entry_date = pos.get("entry_date", "")
            qty = pos.get("qty", 0.0)
            side = pos.get("side", "long")

            # Reconstruct SL/TP from open orders
            sl_price = 0.0
            tp_price = 0.0
            for order in open_orders:
                if order.symbol != ticker:
                    continue
                stop_px = getattr(order, "stop_price", None)
                limit_px = getattr(order, "limit_price", None)
                if stop_px:
                    sl_price = float(stop_px)
                elif limit_px:
                    tp_price = float(limit_px)

            trade_id = f"{ticker}_migrated"
            now_iso = datetime.now().isoformat(timespec="seconds")

            # Try to recover strategy name from execution logs
            log_dir = journal_dir.parent
            strategy = _lookup_strategy_from_logs(ticker, log_dir)

            entry = JournalEntry(
                trade_id=trade_id,
                ticker=ticker,
                strategy=strategy,
                side=side,
                entry_order_id="migrated",
                entry_signal_price=avg_entry,
                entry_fill_price=avg_entry,
                entry_notional=round(avg_entry * qty, 2),
                entry_qty=qty,
                entry_date=entry_date or now_iso,
                original_sl_price=sl_price,
                original_tp_price=tp_price,
                entry_vix=vix,
                entry_market_regime=market_regime,
                entry_spy_price=spy_price,
                status="open",
                opened_at=entry_date or now_iso,
                max_favorable_excursion=avg_entry,
                max_adverse_excursion=avg_entry,
                tags=["migrated"],
                notes="Migrated from pre-journal position",
            )

            # Compute initial risk if SL known
            if sl_price > 0 and avg_entry > 0:
                entry.initial_risk_per_share = round(
                    avg_entry - sl_price, 4
                )
                entry.initial_risk_dollars = round(
                    entry.initial_risk_per_share * qty, 2
                )
            if tp_price > 0 and avg_entry > 0:
                entry.initial_reward_dollars = round(
                    (tp_price - avg_entry) * qty, 2
                )
            if entry.initial_risk_dollars > 0:
                entry.planned_rr_ratio = round(
                    entry.initial_reward_dollars
                    / entry.initial_risk_dollars,
                    2,
                )

            _save_entry(entry, journal_dir)
            created += 1
            log.info(
                f"  Journal: migrated {ticker} "
                f"(entry=${avg_entry:.2f}, "
                f"SL=${sl_price:.2f}, TP=${tp_price:.2f})"
            )

    except Exception as exc:
        log.warning(
            f"Journal: migration failed: {exc}"
        )

    return created


# ── Internal helpers ──────────────────────────────────────────────


def _resolve_fill(
    broker: object,
    order_id: str,
) -> tuple[float, str, float]:
    """Get actual fill price, time, and qty from the broker.

    Uses ``broker.get_order_by_id()`` — no direct ``_client``
    access.

    Returns:
        (fill_price, fill_datetime_str, filled_qty)
        (0.0, "", 0.0) if still pending.
        (-1.0, "", 0.0) if order is dead (cancelled/expired).
    """
    from trading_bot_bl.models import OrderStatus

    try:
        broker_order = broker.get_order_by_id(order_id)  # type: ignore[attr-defined]
        if broker_order is None:
            return (0.0, "", 0.0)

        if (
            broker_order.status == OrderStatus.FILLED
            and broker_order.filled_avg_price
        ):
            return (
                broker_order.filled_avg_price,
                broker_order.filled_at,
                broker_order.filled_qty,
            )
        if broker_order.status in (
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
        ):
            return (-1.0, "", 0.0)
        return (0.0, "", 0.0)  # still pending

    except Exception as exc:
        log.debug(f"Journal: _resolve_fill error: {exc}")
        return (0.0, "", 0.0)


def _query_exit_details(
    broker: object,
    entry: JournalEntry,
) -> tuple[float, str, float]:
    """Query the broker for exit details of a closed position.

    Uses ``broker.get_filled_orders_for_ticker()`` — no direct
    ``_client`` access.

    Returns:
        (exit_fill_price, exit_reason, expected_exit_price)
    """
    try:
        orders = broker.get_filled_orders_for_ticker(  # type: ignore[attr-defined]
            entry.ticker,
            since_date=entry.entry_date or None,
        )

        for order in orders:
            if not order.filled_at:
                continue
            fill_price = order.filled_avg_price
            if fill_price <= 0:
                continue

            # Determine exit reason from order fields
            if order.stop_price:
                return (
                    fill_price,
                    "stop_loss",
                    order.stop_price,
                )
            elif order.limit_price:
                return (
                    fill_price,
                    "take_profit",
                    order.limit_price,
                )
            else:
                return (fill_price, "market_close", 0.0)

    except Exception as exc:
        log.debug(
            f"Journal: _query_exit_details error: {exc}"
        )

    return (0.0, "unknown", 0.0)


def _parse_date(date_str: str) -> date:
    """Parse an ISO date string to a date object."""
    if "T" in date_str:
        return datetime.fromisoformat(date_str).date()
    return date.fromisoformat(date_str[:10])


def get_journal_entry_for_ticker(
    ticker: str,
    journal_dir: Path,
) -> JournalEntry | None:
    """Find the open journal entry for a given ticker."""
    try:
        entries = load_open_trades(journal_dir)
        for entry in entries:
            if entry.ticker == ticker and entry.status == "open":
                return entry
    except Exception as exc:
        log.debug(
            f"Journal: lookup failed for {ticker}: {exc}"
        )
    return None
