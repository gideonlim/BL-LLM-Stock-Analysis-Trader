"""Position monitor -- checks health of open positions and their bracket orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from trading_bot_bl.broker import AlpacaBroker
from trading_bot_bl.config import RiskLimits
from trading_bot_bl.models import (
    OrderResult,
    PortfolioSnapshot,
    PositionAlert,
)

log = logging.getLogger(__name__)


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
            order_type = str(
                getattr(order, "order_type", "")
            ).lower()
            # Alpaca leg types: "stop" for SL, "limit" for TP
            # Also check via stop_price / limit_price presence
            stop_px = getattr(order, "stop_price", None)
            limit_px = getattr(order, "limit_price", None)

            if stop_px and not limit_px:
                # Pure stop order = stop loss leg
                has_stop_loss = True
                sl_price = float(stop_px)
                sl_order_id = str(order.id)
            elif limit_px and not stop_px:
                # Pure limit order = take profit leg
                has_take_profit = True
                tp_price = float(limit_px)
                tp_order_id = str(order.id)
            elif "stop" in order_type:
                has_stop_loss = True
                sl_price = float(stop_px) if stop_px else 0.0
                sl_order_id = str(order.id)
            elif "limit" in order_type:
                has_take_profit = True
                tp_price = float(limit_px) if limit_px else 0.0
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
            # Don't auto-close, but log prominently
            log.warning(f"  PARTIAL BRACKET: {alert.message}")
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
                    )
                    if success:
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
                        # Cancel old SL leg and place new one
                        success = _replace_stop_loss(
                            broker, ticker, sl_order_id,
                            new_sl, qty,
                        )
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
    """
    sl_price = round(current_price * 0.95, 2)
    tp_price = round(current_price * 1.10, 2)

    if dry_run:
        return (
            f"DRY RUN: would place SL=${sl_price} "
            f"and TP=${tp_price}"
        )

    try:
        from alpaca.trading.requests import (
            StopLossRequest,
            TakeProfitRequest,
            LimitOrderRequest,
            StopOrderRequest,
        )
        from alpaca.trading.enums import (
            OrderSide,
            TimeInForce,
        )

        # Place stop loss
        sl_request = StopOrderRequest(
            symbol=ticker,
            qty=abs(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=sl_price,
        )
        broker._client.submit_order(sl_request)

        # Place take profit
        tp_request = LimitOrderRequest(
            symbol=ticker,
            qty=abs(qty),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=tp_price,
        )
        broker._client.submit_order(tp_request)

        return (
            f"Reattached SL=${sl_price}, TP=${tp_price}"
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
) -> bool:
    """Cancel old SL and place a new one at a tighter level."""
    try:
        from alpaca.trading.requests import StopOrderRequest
        from alpaca.trading.enums import (
            OrderSide,
            TimeInForce,
        )

        # Cancel old SL
        if old_sl_order_id:
            broker.cancel_order(old_sl_order_id)

        # Place new SL
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
