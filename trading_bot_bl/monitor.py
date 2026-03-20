"""Position monitor -- checks health of open positions and their bracket orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from trading_bot_bl.broker import AlpacaBroker
from trading_bot_bl.config import RiskLimits
from trading_bot_bl.models import (
    OrderResult,
    PortfolioSnapshot,
    PositionAlert,
)

log = logging.getLogger(__name__)

# ── Journal imports (non-critical) ────────────────────────────────
try:
    from trading_bot_bl import journal as _journal
    _JOURNAL_AVAILABLE = True
except ImportError:
    _JOURNAL_AVAILABLE = False


@dataclass
class MonitorReport:
    """Summary of position monitoring results."""

    alerts: list[PositionAlert] = field(default_factory=list)
    actions: list[OrderResult] = field(default_factory=list)
    positions_checked: int = 0
    orphaned_count: int = 0
    stale_count: int = 0
    emergency_count: int = 0

    @property
    def has_critical(self) -> bool:
        return any(a.severity == "critical" for a in self.alerts)

    def summary(self) -> str:
        parts = [
            f"Checked {self.positions_checked} positions"
        ]
        if self.orphaned_count:
            parts.append(f"{self.orphaned_count} orphaned")
        if self.stale_count:
            parts.append(f"{self.stale_count} stale brackets")
        if self.emergency_count:
            parts.append(f"{self.emergency_count} emergency")
        if self.actions:
            parts.append(
                f"{len(self.actions)} corrective actions taken"
            )
        return ", ".join(parts)


def monitor_positions(
    broker: AlpacaBroker,
    portfolio: PortfolioSnapshot,
    limits: RiskLimits,
    dry_run: bool = False,
    journal_dir: Path | None = None,
) -> MonitorReport:
    """
    Audit every open position for health issues.

    Checks:
    1. Orphaned positions: held stock with no SL/TP orders
       protecting it. If losing and auto_close_orphaned_losers
       is enabled, closes the position.
    2. Stale brackets: SL/TP prices are far from current price
       (price moved significantly since entry but didn't trigger
       either leg). Cancels old legs and places new ones.
    3. Extreme moves: position is down more than emergency_loss_pct
       from entry. Closes immediately regardless of brackets.
    4. Price outside bracket range: current price is beyond TP
       (bracket should have triggered but didn't -- possible fill
       issue) or below SL (same).

    Args:
        broker: Connected AlpacaBroker instance.
        portfolio: Current portfolio snapshot.
        limits: Risk limits with monitoring thresholds.
        dry_run: If True, log actions but don't execute.

    Returns:
        MonitorReport with alerts and actions taken.
    """
    report = MonitorReport()
    positions = portfolio.positions

    if not positions:
        log.info("  No open positions to monitor")
        return report

    report.positions_checked = len(positions)
    tickers = list(positions.keys())

    # ── Fetch all open orders and current prices in bulk ──────
    all_orders = broker.get_open_orders()
    prices = broker.get_latest_prices(tickers)

    # Build a map: ticker -> list of protective orders (SL/TP legs)
    # Bracket legs show up as separate orders with parent_id
    orders_by_ticker: dict[str, list] = {}
    for order in all_orders:
        sym = order.symbol
        if sym not in orders_by_ticker:
            orders_by_ticker[sym] = []
        orders_by_ticker[sym].append(order)

    # ── Check each position ───────────────────────────────────
    for ticker, pos in positions.items():
        entry_price = pos.get("avg_entry", 0.0)
        current_price = prices.get(ticker)
        market_value = pos.get("market_value", 0.0)
        unrealized_pnl = pos.get("unrealized_pnl", 0.0)
        qty = pos.get("qty", 0.0)
        entry_date = pos.get("entry_date")  # ISO date str or None

        if not current_price or entry_price <= 0:
            log.debug(
                f"  {ticker}: no price data, skipping monitor"
            )
            continue

        # ── Journal: update excursion tracking ─────────────────
        _j_entry = None
        if _JOURNAL_AVAILABLE and journal_dir:
            try:
                _j_entry = _journal.get_journal_entry_for_ticker(
                    ticker, journal_dir
                )
                if _j_entry:
                    _journal.update_trade(
                        _j_entry, current_price, journal_dir
                    )
            except Exception:
                pass  # non-critical

        # Fetch ATR for volatility-aware stop management
        atr = _fetch_atr(ticker)

        pnl_pct = (
            (current_price - entry_price) / entry_price * 100
        )

        ticker_orders = orders_by_ticker.get(ticker, [])

        # Classify orders into SL and TP legs
        has_stop_loss = False
        has_take_profit = False
        sl_price = 0.0
        tp_price = 0.0
        sl_order_id = ""
        tp_order_id = ""

        for order in ticker_orders:
            raw_type = getattr(order, "order_type", "")
            order_type = str(
                getattr(raw_type, "value", raw_type) or ""
            ).lower()
            raw_side = getattr(order, "side", "")
            order_side = str(
                getattr(raw_side, "value", raw_side) or ""
            ).lower()
            stop_px = getattr(order, "stop_price", None)
            limit_px = getattr(order, "limit_price", None)
            raw_order_cls = getattr(order, "order_class", "")
            # Alpaca SDK returns an enum (e.g. OrderClass.OCO)
            # — use .value to get the plain string "oco".
            # str() on the enum may yield "OrderClass.OCO"
            # which would break the comparison.
            order_cls = str(
                getattr(raw_order_cls, "value", raw_order_cls)
                or ""
            ).lower()

            # Log raw order details for diagnostics
            log.info(
                f"  Order {order.id}: symbol={ticker}, "
                f"type={order_type}, side={order_side}, "
                f"stop_px={stop_px}, limit_px={limit_px}, "
                f"order_class={order_cls} "
                f"(raw={raw_order_cls!r}), "
                f"legs={getattr(order, 'legs', None)}"
            )

            # ── Classify order as SL or TP leg ────────────────
            # Alpaca bracket/OCO legs can appear as:
            #   - Pure stop order: stop_price only → SL
            #   - Pure limit order: limit_price only → TP
            #   - Stop-limit order: BOTH stop_price AND
            #     limit_price → SL (the limit_price caps the
            #     fill price after the stop triggers)
            #   - order_type string: "stop", "stop_limit",
            #     "limit", "trailing_stop"
            #
            # For OCO orders, the parent is a limit sell (TP)
            # and may have child legs. If the parent has legs,
            # extract SL from them rather than treating the
            # parent as TP-only.

            # ── OCO order detection ─────────────────────────────
            # An OCO order inherently has BOTH legs (TP + SL)
            # linked together. The SL child leg sits in "held"
            # status and is often invisible to the API — the
            # legs attribute may be None or empty even though
            # the child order exists.
            #
            # If order_class is "oco", trust that both legs are
            # present. Extract details from legs if available,
            # but never flag it as a partial bracket.
            if order_cls == "oco":
                legs = getattr(order, "legs", None)
                if legs:
                    for leg in legs:
                        leg_stop = getattr(
                            leg, "stop_price", None
                        )
                        leg_limit = getattr(
                            leg, "limit_price", None
                        )
                        leg_type = str(
                            getattr(leg, "order_type", "")
                        ).lower()
                        log.debug(
                            f"    OCO leg {leg.id}: "
                            f"type={leg_type}, "
                            f"stop={leg_stop}, "
                            f"limit={leg_limit}"
                        )
                        if leg_stop:
                            has_stop_loss = True
                            sl_price = float(leg_stop)
                            sl_order_id = str(leg.id)
                        elif leg_limit:
                            has_take_profit = True
                            tp_price = float(leg_limit)
                            tp_order_id = str(leg.id)

                # Parent is the TP limit sell
                if limit_px and not has_take_profit:
                    has_take_profit = True
                    tp_price = float(limit_px)
                    tp_order_id = str(order.id)

                # Trust that the SL child exists even if the
                # API didn't return it (held status).
                if not has_stop_loss:
                    has_stop_loss = True
                    log.info(
                        f"  {ticker}: OCO order {order.id} "
                        f"detected — SL leg assumed present "
                        f"(held/invisible to API)"
                    )

                continue

            # Standard (non-OCO) order classification
            if stop_px:
                # Has a stop trigger → this is a stop-loss leg
                # (covers both pure stop and stop-limit orders)
                has_stop_loss = True
                sl_price = float(stop_px)
                sl_order_id = str(order.id)
            elif limit_px:
                # Limit only, no stop → take-profit leg
                has_take_profit = True
                tp_price = float(limit_px)
                tp_order_id = str(order.id)
            elif "stop" in order_type or "trail" in order_type:
                # Fallback: classify by order_type string
                has_stop_loss = True
                sl_order_id = str(order.id)
            elif "limit" in order_type:
                has_take_profit = True
                tp_order_id = str(order.id)

        # ── Check 1: Emergency loss ──────────────────────────
        if pnl_pct <= -limits.emergency_loss_pct:
            alert = PositionAlert(
                ticker=ticker,
                alert_type="emergency_loss",
                severity="critical",
                message=(
                    f"{ticker} down {pnl_pct:.1f}% from entry "
                    f"(${entry_price:.2f} -> ${current_price:.2f})"
                    f" — exceeds emergency threshold "
                    f"({limits.emergency_loss_pct}%)"
                ),
                current_price=current_price,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                unrealized_pnl_pct=round(pnl_pct, 2),
            )

            if dry_run:
                alert.action_taken = (
                    "DRY RUN: would emergency-close position"
                )
                log.warning(
                    f"  EMERGENCY (dry): {alert.message}"
                )
            else:
                result = broker.close_position(ticker)
                alert.action_taken = (
                    f"Emergency close: {result.status}"
                )
                report.actions.append(result)
                log.warning(
                    f"  EMERGENCY CLOSE: {alert.message} "
                    f"-> {result.status}"
                )

            # Journal: record emergency close
            if _JOURNAL_AVAILABLE and _j_entry and journal_dir:
                try:
                    _journal.close_trade(
                        _j_entry,
                        exit_price=current_price,
                        exit_reason="emergency_close",
                        journal_dir=journal_dir,
                    )
                except Exception:
                    pass

            report.alerts.append(alert)
            report.emergency_count += 1
            continue  # position closed, skip other checks

        # ── Check 2: Orphaned position (no SL/TP) ────────────
        if not has_stop_loss and not has_take_profit:
            alert = PositionAlert(
                ticker=ticker,
                alert_type="orphaned",
                severity="warning" if pnl_pct >= 0 else "critical",
                message=(
                    f"{ticker} has NO protective orders "
                    f"(SL/TP legs missing). "
                    f"P&L: {pnl_pct:+.1f}%, "
                    f"entry=${entry_price:.2f}, "
                    f"now=${current_price:.2f}"
                ),
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl_pct=round(pnl_pct, 2),
            )

            # If losing beyond threshold, close it
            if (
                limits.auto_close_orphaned_losers
                and pnl_pct <= -limits.orphan_max_loss_pct
            ):
                if dry_run:
                    alert.action_taken = (
                        "DRY RUN: would close orphaned loser"
                    )
                    log.warning(
                        f"  ORPHANED (dry): {alert.message}"
                    )
                else:
                    result = broker.close_position(ticker)
                    alert.action_taken = (
                        f"Closed orphaned loser: {result.status}"
                    )
                    report.actions.append(result)
                    log.warning(
                        f"  ORPHANED CLOSE: {alert.message} "
                        f"-> {result.status}"
                    )
            else:
                # Try to re-attach bracket orders
                action = _reattach_bracket(
                    broker, ticker, qty, entry_price,
                    current_price, dry_run,
                )
                alert.action_taken = action
                log.warning(f"  ORPHANED: {alert.message}")

            report.alerts.append(alert)
            report.orphaned_count += 1
            continue

        # ── Check 3: Only one leg missing ─────────────────────
        if not has_stop_loss or not has_take_profit:
            missing = "stop loss" if not has_stop_loss else "take profit"
            alert = PositionAlert(
                ticker=ticker,
                alert_type="orphaned",
                severity="warning",
                message=(
                    f"{ticker} missing {missing} leg. "
                    f"P&L: {pnl_pct:+.1f}%"
                ),
                current_price=current_price,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                unrealized_pnl_pct=round(pnl_pct, 2),
            )

            # Remediate: cancel stale remaining leg and
            # reattach both SL + TP via a single OCO order.
            action = _reattach_bracket(
                broker, ticker, qty, entry_price,
                current_price, dry_run,
            )
            alert.action_taken = action
            log.warning(
                f"  PARTIAL BRACKET: {alert.message} "
                f"-> {action}"
            )
            report.alerts.append(alert)
            report.orphaned_count += 1

        # ── Check 4: Price outside bracket range ──────────────
        # If price is above TP or below SL, the leg should have
        # triggered. This can happen with gaps or fill issues.
        if has_stop_loss and current_price < sl_price * 0.98:
            alert = PositionAlert(
                ticker=ticker,
                alert_type="extreme_move",
                severity="critical",
                message=(
                    f"{ticker} price ${current_price:.2f} "
                    f"gapped below SL ${sl_price:.2f} "
                    f"(SL may not have filled)"
                ),
                current_price=current_price,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                unrealized_pnl_pct=round(pnl_pct, 2),
            )
            if dry_run:
                alert.action_taken = (
                    "DRY RUN: would close gapped position"
                )
            else:
                result = broker.close_position(ticker)
                alert.action_taken = (
                    f"Closed gapped position: {result.status}"
                )
                report.actions.append(result)
                log.warning(
                    f"  GAP CLOSE: {alert.message} "
                    f"-> {result.status}"
                )
            # Journal: record gap close
            if _JOURNAL_AVAILABLE and _j_entry and journal_dir:
                try:
                    _journal.close_trade(
                        _j_entry,
                        exit_price=current_price,
                        exit_reason="gap_close",
                        journal_dir=journal_dir,
                        expected_exit_price=sl_price,
                    )
                except Exception:
                    pass

            report.alerts.append(alert)
            report.emergency_count += 1
            continue

        if has_take_profit and current_price > tp_price * 1.02:
            alert = PositionAlert(
                ticker=ticker,
                alert_type="extreme_move",
                severity="warning",
                message=(
                    f"{ticker} price ${current_price:.2f} "
                    f"surged above TP ${tp_price:.2f} "
                    f"(TP may not have filled — running profit)"
                ),
                current_price=current_price,
                entry_price=entry_price,
                take_profit_price=tp_price,
                unrealized_pnl_pct=round(pnl_pct, 2),
            )
            # Don't auto-close a winner — just flag it
            log.info(f"  TP OVERSHOOT: {alert.message}")
            report.alerts.append(alert)

        # ── Check 5: Breakeven stop ──────────────────────────
        # Once the position is up enough (1× ATR or 3%), move
        # SL to breakeven to prevent a winner becoming a loser.
        if has_stop_loss and pnl_pct > 0:
            be_sl = _calculate_breakeven_stop(
                entry_price, current_price, sl_price, atr
            )
            if be_sl is not None and be_sl > sl_price:
                alert = PositionAlert(
                    ticker=ticker,
                    alert_type="breakeven_stop",
                    severity="info",
                    message=(
                        f"{ticker} up {pnl_pct:.1f}% — "
                        f"moving SL to breakeven "
                        f"${be_sl:.2f} (was ${sl_price:.2f})"
                    ),
                    current_price=current_price,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    unrealized_pnl_pct=round(pnl_pct, 2),
                )

                if dry_run:
                    alert.action_taken = (
                        "DRY RUN: would move SL to breakeven"
                    )
                else:
                    success = _replace_stop_loss(
                        broker, ticker, sl_order_id,
                        be_sl, qty,
                        oco_parent_id=tp_order_id
                        if not sl_order_id else "",
                        tp_price=tp_price,
                    )
                    if success:
                        sl_price = be_sl  # update for later checks
                        # Journal: record SL modification
                        if (
                            _JOURNAL_AVAILABLE
                            and _j_entry
                            and journal_dir
                        ):
                            try:
                                _journal.record_sl_modification(
                                    _j_entry,
                                    old_sl=sl_price,
                                    new_sl=be_sl,
                                    reason="breakeven",
                                    current_price=current_price,
                                    journal_dir=journal_dir,
                                )
                            except Exception:
                                pass
                    alert.action_taken = (
                        f"SL moved to breakeven ${be_sl:.2f}"
                        if success
                        else "Failed to move SL to breakeven"
                    )

                log.info(
                    f"  BREAKEVEN SL: {alert.message} "
                    f"-> {alert.action_taken}"
                )
                report.alerts.append(alert)

        # ── Check 6: Stale brackets / trailing stop ──────────
        # Price moved significantly from entry but hasn't triggered
        # either leg. The original SL/TP may be too far away.
        if (
            has_stop_loss
            and has_take_profit
            and abs(pnl_pct) > limits.stale_bracket_pct
        ):
            # Only flag if the price is moving away from the
            # profitable direction (i.e., getting closer to SL)
            # or if the bracket range is unreasonably wide
            bracket_width_pct = 0.0
            if entry_price > 0:
                bracket_width_pct = (
                    (tp_price - sl_price)
                    / entry_price * 100
                )

            if bracket_width_pct > 30:  # very wide bracket
                alert = PositionAlert(
                    ticker=ticker,
                    alert_type="stale_bracket",
                    severity="info",
                    message=(
                        f"{ticker} bracket is very wide "
                        f"({bracket_width_pct:.0f}% of entry). "
                        f"SL=${sl_price:.2f}, "
                        f"TP=${tp_price:.2f}, "
                        f"now=${current_price:.2f} "
                        f"({pnl_pct:+.1f}%)"
                    ),
                    current_price=current_price,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    take_profit_price=tp_price,
                    unrealized_pnl_pct=round(pnl_pct, 2),
                )
                log.info(f"  STALE: {alert.message}")
                report.alerts.append(alert)
                report.stale_count += 1

            # If profitable, consider tightening the SL to
            # lock in gains (trailing stop behavior)
            if pnl_pct > limits.stale_bracket_pct:
                new_sl = _calculate_trailing_stop(
                    entry_price, current_price, sl_price,
                    atr=atr,
                )
                if new_sl and new_sl > sl_price:
                    alert = PositionAlert(
                        ticker=ticker,
                        alert_type="stale_bracket",
                        severity="info",
                        message=(
                            f"{ticker} up {pnl_pct:.1f}% — "
                            f"tightening SL from "
                            f"${sl_price:.2f} to ${new_sl:.2f} "
                            f"(ATR={atr:.2f})"
                            if atr > 0 else
                            f"{ticker} up {pnl_pct:.1f}% — "
                            f"tightening SL from "
                            f"${sl_price:.2f} to ${new_sl:.2f} "
                            f"(%-based trail)"
                        ),
                        current_price=current_price,
                        entry_price=entry_price,
                        stop_loss_price=sl_price,
                        unrealized_pnl_pct=round(pnl_pct, 2),
                    )

                    if dry_run:
                        alert.action_taken = (
                            "DRY RUN: would tighten SL"
                        )
                    else:
                        # Cancel old SL leg and place new one.
                        # For OCO brackets the SL leg is held /
                        # invisible — pass the OCO parent ID so
                        # the function cancels the whole OCO
                        # before resubmitting a fresh one.
                        success = _replace_stop_loss(
                            broker, ticker, sl_order_id,
                            new_sl, qty,
                            oco_parent_id=tp_order_id
                            if not sl_order_id else "",
                            tp_price=tp_price,
                        )
                        if success and (
                            _JOURNAL_AVAILABLE
                            and _j_entry
                            and journal_dir
                        ):
                            try:
                                _journal.record_sl_modification(
                                    _j_entry,
                                    old_sl=sl_price,
                                    new_sl=new_sl,
                                    reason="trailing",
                                    current_price=current_price,
                                    journal_dir=journal_dir,
                                )
                            except Exception:
                                pass
                        alert.action_taken = (
                            f"SL tightened to ${new_sl:.2f}"
                            if success
                            else "Failed to replace SL"
                        )

                    log.info(
                        f"  TRAILING SL: {alert.message} "
                        f"-> {alert.action_taken}"
                    )
                    report.alerts.append(alert)
                    report.stale_count += 1

        # ── Check 7: Time-based exit ────────────────────────
        # Close positions that have exceeded max hold period.
        # Stale positions tie up capital for better opportunities.
        if _check_time_exit(
            ticker, entry_date, limits.max_hold_days
        ):
            alert = PositionAlert(
                ticker=ticker,
                alert_type="time_exit",
                severity="warning",
                message=(
                    f"{ticker} exceeded max hold period "
                    f"({limits.max_hold_days} days). "
                    f"P&L: {pnl_pct:+.1f}%"
                ),
                current_price=current_price,
                entry_price=entry_price,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                unrealized_pnl_pct=round(pnl_pct, 2),
            )

            if dry_run:
                alert.action_taken = (
                    "DRY RUN: would close stale position"
                )
                log.warning(
                    f"  TIME EXIT (dry): {alert.message}"
                )
            else:
                result = broker.close_position(ticker)
                alert.action_taken = (
                    f"Time-exit close: {result.status}"
                )
                report.actions.append(result)
                log.warning(
                    f"  TIME EXIT: {alert.message} "
                    f"-> {result.status}"
                )

            # Journal: record time exit
            if _JOURNAL_AVAILABLE and _j_entry and journal_dir:
                try:
                    _journal.close_trade(
                        _j_entry,
                        exit_price=current_price,
                        exit_reason="time_exit",
                        journal_dir=journal_dir,
                    )
                except Exception:
                    pass

            report.alerts.append(alert)

    log.info(f"  Monitor: {report.summary()}")
    return report


def _reattach_bracket(
    broker: AlpacaBroker,
    ticker: str,
    qty: float,
    entry_price: float,
    current_price: float,
    dry_run: bool,
) -> str:
    """
    Attempt to place new SL/TP orders for an orphaned position.

    Uses a default 5% SL below current price and 10% TP above.

    Submits a single OCO (One-Cancels-Other) order that links
    both the SL and TP legs together. This is critical because
    Alpaca reserves shares per order — two separate sell orders
    for the full quantity would fail since the first order
    reserves all shares, leaving none for the second.

    An OCO order's parent is a limit sell (the TP leg) and its
    child is a stop sell (the SL leg). When either leg fills,
    the other is automatically cancelled. Both legs share the
    same share reservation.

    Before placing the OCO, cancels any existing orders for
    this ticker to free up the held-for-orders quantity.
    """
    sl_price = round(current_price * 0.95, 2)
    tp_price = round(current_price * 1.10, 2)

    if dry_run:
        return (
            f"DRY RUN: would place OCO with "
            f"SL=${sl_price} and TP=${tp_price}"
        )

    try:
        from alpaca.trading.requests import (
            StopLossRequest,
            TakeProfitRequest,
            LimitOrderRequest,
        )
        from alpaca.trading.enums import (
            OrderClass,
            OrderSide,
            TimeInForce,
        )

        # Cancel any existing orders for this ticker first.
        # If the position had stale/expired bracket legs, the
        # shares may still be "held_for_orders". Cancelling
        # first frees them for the new OCO order.
        existing = broker.get_open_orders()
        for order in existing:
            if order.symbol == ticker:
                broker.cancel_order(str(order.id))
                log.info(
                    f"  Cancelled existing order {order.id} "
                    f"for {ticker} before reattach"
                )

        # Submit a single OCO order with both legs linked.
        # Alpaca OCO requires explicit take_profit AND
        # stop_loss parameters — the parent limit_price
        # alone does not satisfy the TP requirement.
        oco_request = LimitOrderRequest(
            symbol=ticker,
            qty=abs(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=tp_price,
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(
                limit_price=tp_price,
            ),
            stop_loss=StopLossRequest(
                stop_price=sl_price,
            ),
        )
        broker._client.submit_order(oco_request)

        log.info(
            f"  OCO order placed for {ticker}: "
            f"SL=${sl_price}, TP=${tp_price}, qty={abs(qty)}"
        )

        return (
            f"Reattached OCO: SL=${sl_price}, TP=${tp_price}"
        )

    except Exception as e:
        log.error(
            f"Failed to reattach bracket for {ticker}: {e}"
        )
        return f"Failed to reattach: {e}"


def _replace_stop_loss(
    broker: AlpacaBroker,
    ticker: str,
    old_sl_order_id: str,
    new_sl_price: float,
    qty: float,
    oco_parent_id: str = "",
    tp_price: float = 0.0,
) -> bool:
    """Move a stop-loss to a tighter level.

    Handles two bracket structures:

    **Standalone SL** (``old_sl_order_id`` is set, no OCO parent):
        Cancel the old stop order, then submit a new standalone
        stop order at ``new_sl_price``.

    **OCO bracket** (``oco_parent_id`` is set, SL leg is held /
    invisible to the API):
        Alpaca holds *all* shares for the entire OCO — cancelling
        only the invisible SL leg leaves the parent alive and the
        shares still reserved, so any new order is rejected with
        ``available: 0``.  The correct approach is to cancel the
        **OCO parent** (which atomically cancels both legs and
        frees the share hold), then immediately resubmit a fresh
        OCO with the original TP price and the new SL price.

    Args:
        broker:          AlpacaBroker instance.
        ticker:          Position symbol.
        old_sl_order_id: Order ID of the standalone SL leg to
                         cancel (empty string if OCO).
        new_sl_price:    New stop-loss price.
        qty:             Position quantity (positive).
        oco_parent_id:   Order ID of the OCO parent to cancel
                         (set when the bracket is an OCO).
        tp_price:        Existing take-profit price to preserve
                         in the replacement OCO (required when
                         ``oco_parent_id`` is set).
    """
    try:
        from alpaca.trading.enums import (
            OrderSide,
            TimeInForce,
        )

        if oco_parent_id:
            # ── OCO path ─────────────────────────────────────
            # 1. Cancel the OCO parent → frees held shares for
            #    both the TP limit leg and the SL stop leg.
            # 2. Resubmit a fresh OCO with the original TP and
            #    the updated SL so both legs are re-linked in
            #    a single share reservation.
            from alpaca.trading.requests import (
                LimitOrderRequest,
                TakeProfitRequest,
                StopLossRequest,
            )
            from alpaca.trading.enums import OrderClass

            broker.cancel_order(oco_parent_id)
            log.info(
                f"  OCO parent {oco_parent_id} cancelled "
                f"for {ticker} SL replacement"
            )

            oco_request = LimitOrderRequest(
                symbol=ticker,
                qty=abs(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                limit_price=round(tp_price, 2),
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(
                    limit_price=round(tp_price, 2),
                ),
                stop_loss=StopLossRequest(
                    stop_price=round(new_sl_price, 2),
                ),
            )
            broker._client.submit_order(oco_request)
            log.info(
                f"  New OCO submitted for {ticker}: "
                f"SL=${new_sl_price:.2f}, TP=${tp_price:.2f}"
            )

        else:
            # ── Standalone SL path ───────────────────────────
            # Cancel old stop order (if we have its ID), then
            # submit a plain stop order at the new price.
            from alpaca.trading.requests import StopOrderRequest

            if old_sl_order_id:
                broker.cancel_order(old_sl_order_id)

            request = StopOrderRequest(
                symbol=ticker,
                qty=abs(qty),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                stop_price=round(new_sl_price, 2),
            )
            broker._client.submit_order(request)

        return True

    except Exception as e:
        log.error(
            f"Failed to replace SL for {ticker}: {e}"
        )
        return False


def _calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    current_sl: float,
    atr: float = 0.0,
) -> float | None:
    """
    Calculate a new trailing stop loss price using a two-tier
    approach:

    1. **ATR-based trail** (preferred): Trail by 2× ATR below
       current price. This adapts to the stock's actual
       volatility — wide-ranging stocks get wider trailing
       stops, tight stocks get tighter ones.

    2. **Percentage-based fallback**: If ATR is unavailable,
       trail at entry + 50% of unrealized gain (locks in half
       the profit).

    Both tiers enforce a floor at breakeven (entry_price) to
    prevent a winning trade from becoming a loser.

    Only returns a value if the new SL is strictly higher than
    the current SL — trailing stops never widen downward.

    Args:
        entry_price: Original entry price.
        current_price: Latest market price.
        current_sl: Current stop loss price.
        atr: 14-day Average True Range. If 0, uses %-based trail.

    Returns:
        New SL price (rounded), or None if no improvement.
    """
    gain = current_price - entry_price
    if gain <= 0:
        return None

    if atr > 0:
        # ATR-based: trail 2× ATR below current price
        new_sl = current_price - 2.0 * atr
        # Floor at breakeven
        new_sl = max(new_sl, entry_price)
    else:
        # Percentage fallback: entry + 50% of gain
        new_sl = entry_price + gain * 0.5

    new_sl = round(new_sl, 2)

    if new_sl <= current_sl:
        return None  # don't widen the stop

    return new_sl


def _calculate_breakeven_stop(
    entry_price: float,
    current_price: float,
    current_sl: float,
    atr: float = 0.0,
    breakeven_threshold_atr: float = 1.0,
    breakeven_threshold_pct: float = 3.0,
) -> float | None:
    """
    Move stop loss to breakeven once price has moved enough in
    our favor.

    The "enough" threshold is defined as:
    - 1× ATR above entry (if ATR available), OR
    - 3% above entry (fallback)

    Once triggered, the SL moves to entry_price (breakeven).
    This ensures a winning trade cannot become a loser.

    Args:
        entry_price: Original entry price.
        current_price: Latest market price.
        current_sl: Current stop loss price.
        atr: 14-day ATR (0 if unavailable).
        breakeven_threshold_atr: ATRs above entry to trigger.
        breakeven_threshold_pct: Percentage threshold (fallback).

    Returns:
        entry_price as new SL, or None if conditions not met.
    """
    if current_sl >= entry_price:
        return None  # already at or above breakeven

    gain = current_price - entry_price
    if gain <= 0:
        return None

    if atr > 0:
        threshold = atr * breakeven_threshold_atr
    else:
        threshold = entry_price * breakeven_threshold_pct / 100

    if gain >= threshold:
        return round(entry_price, 2)

    return None


def _check_time_exit(
    ticker: str,
    entry_date: str | None,
    max_hold_days: int = 10,
) -> bool:
    """
    Check if a position has exceeded its maximum hold period.

    Stale positions tie up capital that could be deployed to
    better opportunities. If a stock hasn't hit TP or SL within
    max_hold_days, it's a candidate for closing.

    Args:
        ticker: Stock symbol (for logging).
        entry_date: ISO date string of when position was opened.
        max_hold_days: Maximum days to hold.

    Returns:
        True if position has exceeded max hold period.
    """
    if not entry_date:
        return False

    from datetime import datetime, date

    try:
        if "T" in entry_date:
            entry = datetime.fromisoformat(entry_date).date()
        else:
            entry = date.fromisoformat(entry_date[:10])

        days_held = (date.today() - entry).days
        if days_held > max_hold_days:
            log.info(
                f"  {ticker}: held {days_held} days "
                f"(max {max_hold_days}) — time exit candidate"
            )
            return True
    except (ValueError, TypeError):
        pass

    return False


def _fetch_atr(ticker: str) -> float:
    """
    Fetch the current 14-day ATR for a ticker via yfinance.

    Returns 0.0 if data is unavailable. Used by the trailing
    stop and breakeven stop calculations to adapt to the
    stock's actual volatility.
    """
    try:
        import yfinance as yf
        import numpy as np
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(days=30)
        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty or len(data) < 15:
            return 0.0

        high = data["High"].values
        low = data["Low"].values
        close = data["Close"].values

        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        # 14-period ATR (simple moving average of TR)
        if len(tr) >= 14:
            atr = float(np.mean(tr[-14:]))
            return round(atr, 4)
        return 0.0

    except Exception:
        return 0.0


def write_monitor_log(
    report: MonitorReport,
    log_dir: Path,
) -> Path:
    """Write monitor results to a JSON log file."""
    import json
    from datetime import datetime
    from pathlib import Path as P

    log_dir = P(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = log_dir / f"monitor_{date_str}.json"

    data = {
        "monitored_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "positions_checked": report.positions_checked,
        "orphaned": report.orphaned_count,
        "stale": report.stale_count,
        "emergency": report.emergency_count,
        "actions_taken": len(report.actions),
        "alerts": [
            {
                "ticker": a.ticker,
                "type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "action_taken": a.action_taken,
                "current_price": a.current_price,
                "entry_price": a.entry_price,
                "pnl_pct": a.unrealized_pnl_pct,
                "timestamp": a.timestamp,
            }
            for a in report.alerts
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    log.info(f"Monitor log written to {path}")
    return path
