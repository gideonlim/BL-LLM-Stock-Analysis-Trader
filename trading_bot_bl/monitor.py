"""Position monitor -- checks health of open positions and their bracket orders."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from trading_bot_bl.broker_base import BrokerInterface
from trading_bot_bl.config import RiskLimits
from trading_bot_bl.earnings import check_earnings_blackout
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
    broker: BrokerInterface,
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
        broker: Connected BrokerInterface instance.
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

    # ── Market-hours gate ────────────────────────────────────
    # Outside regular hours, Alpaca extended-hours execution is
    # unreliable (stale prices, paper fills not working).
    # Cancelling OCO brackets would strip SL/TP protection for
    # no benefit.  Force dry_run so the monitor still reports
    # issues and tracks excursions, but defers corrective
    # actions to the next market-hours run.
    market_open = broker.is_market_open()
    if not market_open and not dry_run:
        log.info(
            "  Market is closed — deferring corrective "
            "actions to next market-hours run"
        )
        dry_run = True

    # Deferred journal entries — populated during the position loop
    # and resolved after a 10s settlement wait.  Each item is:
    # (order_id, journal_entry, exit_price, exit_reason, expected_exit)
    _deferred_journal: list[tuple] = []

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
        _position_closed = False  # set by any close action

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

        # ── Earnings proximity warning ───────────────────────
        _earnings_info = None  # persisted for stop-tightening below
        if limits.earnings_blackout_enabled:
            try:
                _earnings_info = check_earnings_blackout(
                    ticker,
                    pre_days=limits.earnings_blackout_pre_days,
                    post_days=limits.earnings_blackout_post_days,
                )
                if (
                    _earnings_info.days_until_earnings is not None
                    and 0 < _earnings_info.days_until_earnings
                    <= limits.earnings_blackout_pre_days + 2
                ):
                    severity = (
                        "warning"
                        if _earnings_info.days_until_earnings
                        <= limits.earnings_blackout_pre_days
                        else "info"
                    )
                    alert = PositionAlert(
                        ticker=ticker,
                        alert_type="earnings_approaching",
                        severity=severity,
                        message=(
                            f"Earnings in "
                            f"{_earnings_info.days_until_earnings}d "
                            f"({_earnings_info.next_earnings_date})"
                            f" — consider closing to avoid gap risk"
                        ),
                        current_price=current_price,
                        entry_price=entry_price,
                        unrealized_pnl_pct=round(
                            (current_price - entry_price)
                            / entry_price * 100, 2
                        ),
                    )
                    report.alerts.append(alert)
                    log.warning(
                        f"  {ticker}: {alert.message}"
                    )
            except Exception:
                pass  # non-critical

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

        # ── Fallback: recover SL/TP from journal if API ─────
        # didn't return them (OCO held legs are invisible).
        #
        # For SL, check sl_modifications first — after a
        # breakeven or trailing stop move, the most recent
        # new_sl is the live SL price.  Falling back to
        # original_sl_price after a modification causes the
        # monitor to think the SL is still at the old value,
        # triggering an infinite cancel-replace loop.
        if _j_entry:
            if sl_price == 0.0:
                # 1. Latest modification (most recent SL move)
                if _j_entry.sl_modifications:
                    latest_mod_sl = (
                        _j_entry.sl_modifications[-1]
                        .get("new_sl", 0.0)
                    )
                    if latest_mod_sl > 0:
                        sl_price = latest_mod_sl
                        log.info(
                            f"  {ticker}: SL price recovered "
                            f"from journal modification "
                            f"→ ${sl_price:.2f}"
                        )
                # 2. Original SL (no modifications yet)
                if sl_price == 0.0 and _j_entry.original_sl_price:
                    sl_price = _j_entry.original_sl_price
                    log.info(
                        f"  {ticker}: SL price recovered from "
                        f"journal (original) → ${sl_price:.2f}"
                    )
            if tp_price == 0.0 and _j_entry.original_tp_price:
                tp_price = _j_entry.original_tp_price
                log.info(
                    f"  {ticker}: TP price recovered from "
                    f"journal → ${tp_price:.2f}"
                )

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

            result = None
            if dry_run:
                alert.action_taken = (
                    "DRY RUN: would emergency-close position"
                )
                log.warning(
                    f"  EMERGENCY (dry): {alert.message}"
                )
            else:
                result = broker.close_position(ticker)
                _position_closed = True
                alert.action_taken = (
                    f"Emergency close: {result.status}"
                )
                report.actions.append(result)
                log.warning(
                    f"  EMERGENCY CLOSE: {alert.message} "
                    f"-> {result.status}"
                )

            # Defer journal until order fill is confirmed
            if (
                not dry_run
                and result is not None
                and result.status != "rejected"
                and result.order_id
                and _JOURNAL_AVAILABLE
                and _j_entry
                and journal_dir
            ):
                _deferred_journal.append((
                    result.order_id, _j_entry,
                    current_price, "emergency_close", 0.0,
                ))

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
                    _position_closed = True
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
            result = None
            if dry_run:
                alert.action_taken = (
                    "DRY RUN: would close gapped position"
                )
            else:
                result = broker.close_position(ticker)
                _position_closed = True
                alert.action_taken = (
                    f"Closed gapped position: {result.status}"
                )
                report.actions.append(result)
                log.warning(
                    f"  GAP CLOSE: {alert.message} "
                    f"-> {result.status}"
                )
            # Defer journal until order fill is confirmed
            if (
                not dry_run
                and result is not None
                and result.status != "rejected"
                and result.order_id
                and _JOURNAL_AVAILABLE
                and _j_entry
                and journal_dir
            ):
                _deferred_journal.append((
                    result.order_id, _j_entry,
                    current_price, "gap_close", sl_price,
                ))

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
                        # Journal: record SL modification.
                        # Must record BEFORE updating sl_price,
                        # otherwise old_sl captures the new value.
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
                        sl_price = be_sl  # update for later checks
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
                # Chandelier anchor: highest price since entry,
                # tracked by the journal's MFE field.  Falls
                # back to current_price if no journal entry.
                _hh = 0.0
                if _j_entry:
                    _hh = getattr(
                        _j_entry,
                        "max_favorable_excursion",
                        0.0,
                    )
                new_sl = _calculate_trailing_stop(
                    entry_price, current_price, sl_price,
                    atr=atr,
                    highest_high=_hh,
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
                            f"(Chandelier: HH=${_hh:.2f}, "
                            f"ATR={atr:.2f})"
                            if atr > 0 and _hh > 0 else
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

            result = None
            if dry_run:
                alert.action_taken = (
                    "DRY RUN: would close stale position"
                )
                log.warning(
                    f"  TIME EXIT (dry): {alert.message}"
                )
            else:
                result = broker.close_position(ticker)
                _position_closed = True
                alert.action_taken = (
                    f"Time-exit close: {result.status}"
                )
                report.actions.append(result)
                log.warning(
                    f"  TIME EXIT: {alert.message} "
                    f"-> {result.status}"
                )

            # Defer journal until order fill is confirmed
            if (
                not dry_run
                and result is not None
                and result.status != "rejected"
                and result.order_id
                and _JOURNAL_AVAILABLE
                and _j_entry
                and journal_dir
            ):
                _deferred_journal.append((
                    result.order_id, _j_entry,
                    current_price, "time_exit", 0.0,
                ))

            report.alerts.append(alert)

        # ── Check 8: Earnings pre-close for profitable positions ─
        # When earnings are within the blackout window and the
        # position is profitable, tighten the SL to lock in gains.
        # This converts the paper gain into a protected floor
        # before the high-volatility earnings event.
        #
        # Logic:
        #   - Only for positions currently in profit (pnl_pct > 0)
        #   - Only when earnings ≤ blackout_pre_days away
        #   - New SL = max(current_sl, entry_price + 50% of gain)
        #     i.e. lock in at least half the unrealised profit
        #   - Never widens the stop (new_sl must > sl_price)
        #   - Skipped if the position was already closed by an
        #     earlier check (emergency, orphan, gap, time-exit).
        if (
            not _position_closed
            and
            _earnings_info is not None
            and _earnings_info.days_until_earnings is not None
            and 0 < _earnings_info.days_until_earnings
            <= limits.earnings_blackout_pre_days
            and has_stop_loss
            and pnl_pct > 1.0  # at least 1% profit to bother
            and sl_price > 0
        ):
            # Lock in 50% of the unrealised gain
            earn_sl = round(
                entry_price
                + 0.5 * (current_price - entry_price),
                2,
            )
            # Never widen a stop
            earn_sl = max(earn_sl, sl_price)

            if earn_sl > sl_price:
                alert = PositionAlert(
                    ticker=ticker,
                    alert_type="earnings_stop_tighten",
                    severity="warning",
                    message=(
                        f"{ticker} earnings in "
                        f"{_earnings_info.days_until_earnings}d "
                        f"({_earnings_info.next_earnings_date})"
                        f" — tightening SL from "
                        f"${sl_price:.2f} to ${earn_sl:.2f} "
                        f"to lock in gains ({pnl_pct:+.1f}%)"
                    ),
                    current_price=current_price,
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    unrealized_pnl_pct=round(pnl_pct, 2),
                )

                if dry_run:
                    alert.action_taken = (
                        f"DRY RUN: would tighten SL to "
                        f"${earn_sl:.2f}"
                    )
                else:
                    success = _replace_stop_loss(
                        broker, ticker, sl_order_id,
                        earn_sl, qty,
                        oco_parent_id=(
                            tp_order_id
                            if not sl_order_id else ""
                        ),
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
                                new_sl=earn_sl,
                                reason="earnings_protection",
                                current_price=current_price,
                                journal_dir=journal_dir,
                            )
                        except Exception:
                            pass
                    if success:
                        sl_price = earn_sl
                    alert.action_taken = (
                        f"SL tightened to ${earn_sl:.2f}"
                        if success
                        else "Failed to tighten SL"
                    )

                log.warning(
                    f"  EARNINGS SL: {alert.message} "
                    f"-> {alert.action_taken}"
                )
                report.alerts.append(alert)

    # ── Post-action settlement: verify order fills ─────────────
    # Wait 10s for orders to settle, then check fill status.
    # Only journal trades whose close orders actually filled.
    if _deferred_journal and journal_dir:
        import time

        log.info(
            f"  Waiting 10s for {len(_deferred_journal)} "
            f"close order(s) to settle..."
        )
        time.sleep(10)

        filled_count = 0
        pending_count = 0
        for (
            order_id, j_entry, exit_price,
            exit_reason, expected_exit,
        ) in _deferred_journal:
            broker_order = broker.get_order_by_id(order_id)
            if broker_order is not None:
                status = broker_order.status.value
                fill_price = broker_order.filled_avg_price
            else:
                status = "unknown"
                fill_price = 0.0

            if status == "filled":
                # Use actual fill price if available
                actual_exit = (
                    fill_price if fill_price else exit_price
                )
                try:
                    _journal.close_trade(
                        j_entry,
                        exit_price=actual_exit,
                        exit_reason=exit_reason,
                        journal_dir=journal_dir,
                        expected_exit_price=expected_exit,
                    )
                except Exception as exc:
                    log.warning(
                        f"  Journal close_trade failed for "
                        f"{j_entry.trade_id}: {exc}"
                    )
                filled_count += 1
                log.info(
                    f"  {j_entry.ticker}: order {order_id} "
                    f"filled @ ${actual_exit:.2f} — "
                    f"journal updated"
                )
            else:
                pending_count += 1
                log.info(
                    f"  {j_entry.ticker}: order {order_id} "
                    f"status={status} — journal NOT updated "
                    f"(will resolve on next monitor run)"
                )

        log.info(
            f"  Settlement: {filled_count} filled, "
            f"{pending_count} still pending"
        )

    log.info(f"  Monitor: {report.summary()}")
    return report


def _reattach_bracket(
    broker,
    ticker: str,
    qty: float,
    entry_price: float,
    current_price: float,
    dry_run: bool,
) -> str:
    """
    Attempt to place new SL/TP orders for an orphaned position.

    Uses a default 5% SL below current price and 10% TP above.
    Delegates to ``broker.submit_oco_reattach()`` which handles
    broker-specific OCO/OCA semantics internally.
    """
    sl_price = round(current_price * 0.95, 2)
    tp_price = round(current_price * 1.10, 2)

    if dry_run:
        return (
            f"DRY RUN: would place OCO with "
            f"SL=${sl_price} and TP=${tp_price}"
        )

    result = broker.submit_oco_reattach(
        ticker=ticker,
        qty=abs(qty),
        stop_loss=sl_price,
        take_profit=tp_price,
        dry_run=False,
    )

    if result.status == "submitted":
        return (
            f"Reattached OCO: SL=${sl_price}, TP=${tp_price}"
        )
    else:
        return f"Failed to reattach: {result.error}"


def _replace_stop_loss(
    broker,
    ticker: str,
    old_sl_order_id: str,
    new_sl_price: float,
    qty: float,
    oco_parent_id: str = "",
    tp_price: float = 0.0,
) -> bool:
    """Move a stop-loss to a tighter level.

    Delegates to ``broker.update_stop_loss()`` which handles
    broker-specific OCO/OCA semantics internally.

    Args:
        broker:          BrokerInterface instance.
        ticker:          Position symbol.
        old_sl_order_id: Order ID of the standalone SL leg to
                         cancel (empty string if OCO).
        new_sl_price:    New stop-loss price.
        qty:             Position quantity (positive).
        oco_parent_id:   Order ID of the OCO parent to cancel
                         (set when the bracket is an OCO).
        tp_price:        Existing take-profit price to preserve
                         in the replacement OCO.
    """
    result = broker.update_stop_loss(
        ticker=ticker,
        new_stop_loss=new_sl_price,
        qty=abs(qty),
        dry_run=False,
        old_sl_order_id=old_sl_order_id,
        oco_parent_id=oco_parent_id,
        tp_price=tp_price,
    )
    return result.status != "rejected"


def _calculate_trailing_stop(
    entry_price: float,
    current_price: float,
    current_sl: float,
    atr: float = 0.0,
    highest_high: float = 0.0,
) -> float | None:
    """
    Calculate a new trailing stop using Chandelier Exit logic.

    The stop is anchored to the **highest price since entry**
    (Chandelier style) rather than the current price.  This
    prevents pullbacks within an uptrend from dragging the stop
    down — the stop only ratchets upward as the position makes
    new highs.

    Two tiers:

    1. **ATR Chandelier** (preferred): ``highest_high - 2 × ATR``.
       Adapts to the stock's actual volatility — wide-ranging
       stocks get wider stops, tight stocks get tighter ones.

    2. **Percentage fallback**: If ATR is unavailable, trail at
       ``entry + 50% of peak gain`` (locks in half the best
       unrealised profit).

    Both tiers enforce a floor at breakeven (entry_price) to
    prevent a winning trade from becoming a loser.

    Only returns a value if the new SL is strictly higher than
    the current SL — trailing stops never widen downward.

    Args:
        entry_price: Original entry price.
        current_price: Latest market price.
        current_sl: Current stop loss price.
        atr: 14-day Average True Range.  If 0, uses %-based trail.
        highest_high: Highest price observed since entry (from
            journal MFE).  Falls back to *current_price* when 0
            or unavailable.

    Returns:
        New SL price (rounded), or None if no improvement.
    """
    gain = current_price - entry_price
    if gain <= 0:
        return None

    # Use highest high since entry (Chandelier anchor).
    # Fall back to current_price if unavailable — this matches
    # the pre-Chandelier behaviour.
    anchor = highest_high if highest_high > 0 else current_price

    if atr > 0:
        # Chandelier: trail 2× ATR below the highest high
        new_sl = anchor - 2.0 * atr
        # Floor at breakeven
        new_sl = max(new_sl, entry_price)
    else:
        # Percentage fallback: entry + 50% of *peak* gain
        peak_gain = anchor - entry_price
        new_sl = entry_price + peak_gain * 0.5

    # Round first so the clamp sees the final submitted value.
    # Without this, a raw value like 114.997 passes the "< market"
    # check but rounds to 115.00 == market, reintroducing the
    # immediate-trigger risk.
    new_sl = round(new_sl, 2)

    # Clamp: stop must stay strictly below current market price.
    # During a deep pullback the Chandelier formula can produce
    # a stop above market (e.g. HH=120, ATR=2, price=115 → 116).
    # Submitting that to the broker would either be rejected or
    # trigger an immediate fill.  A stop *at* market is equally
    # dangerous — it can fill on the next tick.
    #
    # We enforce a minimum buffer of $0.01 (standard US equity
    # tick size).  If even the buffered stop isn't viable (i.e.
    # current_price - 0.01 <= entry_price), we return None to
    # leave the existing stop untouched.
    tick = 0.01
    max_allowed = round(current_price - tick, 2)
    if new_sl >= current_price:
        new_sl = max_allowed

    # If clamped stop would be at/below entry (not an improvement)
    # or at/below current SL, treat as no-op.
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
    log_dir = Path(log_dir)
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
