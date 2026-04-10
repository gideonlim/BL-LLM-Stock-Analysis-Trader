"""Interactive Brokers broker implementation for EU/Asia markets.

Uses ``ib_async`` for all IBKR API calls.  The interface is
synchronous (matching ``BrokerInterface``) — async calls are
wrapped via ``self.ib.run(coro)`` or ``ib_async.util.run()``.

Connection management: the broker connects lazily on first use
and auto-reconnects on ``ConnectionError``.
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING

from trading_bot_bl.broker_base import BrokerInterface
from trading_bot_bl.models import (
    BrokerOrder,
    OrderResult,
    OrderSideNorm,
    OrderStatus,
    PortfolioSnapshot,
)

if TYPE_CHECKING:
    from trading_bot_bl.config import IBKRConfig
    from trading_bot_bl.market_config import MarketConfig

log = logging.getLogger(__name__)

try:
    from ib_async import (
        IB,
        LimitOrder,
        MarketOrder,
        Stock,
        StopOrder,
        Trade,
    )
except ImportError as exc:
    raise ImportError(
        "ib_async is required for IBKR markets. "
        "Install with: pip install ib_async"
    ) from exc


# ── IBKR status → normalized OrderStatus mapping ────────────────

_IBKR_STATUS_MAP: dict[str, OrderStatus] = {
    "submitted": OrderStatus.SUBMITTED,
    "presubmitted": OrderStatus.SUBMITTED,
    "pendingsubmit": OrderStatus.SUBMITTED,
    "pendingcancel": OrderStatus.PENDING_CANCEL,
    "filled": OrderStatus.FILLED,
    "cancelled": OrderStatus.CANCELED,
    "inactive": OrderStatus.REJECTED,
    "apicanceled": OrderStatus.CANCELED,
    "apicancelled": OrderStatus.CANCELED,
}


def _to_broker_order(trade: Trade) -> BrokerOrder:
    """Translate an ib_async Trade to a BrokerOrder."""
    order = trade.order
    fill = trade.fills[-1] if trade.fills else None

    raw_status = (trade.orderStatus.status or "").lower()
    raw_side = (order.action or "").lower()

    return BrokerOrder(
        id=str(order.orderId),
        symbol=str(
            getattr(trade.contract, "symbol", "")
        ),
        status=_IBKR_STATUS_MAP.get(
            raw_status, OrderStatus.UNKNOWN
        ),
        side=(
            OrderSideNorm.BUY if raw_side == "buy"
            else OrderSideNorm.SELL
        ),
        filled_qty=float(trade.orderStatus.filled or 0),
        filled_avg_price=float(
            trade.orderStatus.avgFillPrice or 0
        ),
        filled_at=(
            str(fill.time) if fill else ""
        ),
        stop_price=(
            float(order.auxPrice)
            if getattr(order, "auxPrice", None)
            else None
        ),
        limit_price=(
            float(order.lmtPrice)
            if getattr(order, "lmtPrice", None)
            else None
        ),
        order_type=str(order.orderType or ""),
        order_class=str(
            getattr(order, "ocaGroup", "") or ""
        ),
        commission=(
            fill.commissionReport.commission
            if fill and fill.commissionReport
            else 0.0
        ),
        commission_currency=(
            fill.commissionReport.currency
            if fill and fill.commissionReport
            else ""
        ),
    )


def _make_contract(
    ticker: str, market: "MarketConfig"
) -> Stock:
    """Translate a yfinance-style ticker to an IBKR Stock contract.

    ``VOD.L`` → ``Stock("VOD", "LSE", "GBP")``
    ``7203.T`` → ``Stock("7203", "TSEJ", "JPY")``
    ``AAPL`` → ``Stock("AAPL", "SMART", "USD")``
    """
    # Exchange suffix mapping (yfinance suffix → IBKR exchange)
    _SUFFIX_MAP = {
        ".L": ("LSE", "GBP"),
        ".T": ("TSEJ", "JPY"),
        ".DE": ("IBIS", "EUR"),
        ".HK": ("SEHK", "HKD"),
        ".PA": ("SBF", "EUR"),
        ".AS": ("AEB", "EUR"),
    }

    for suffix, (exchange, currency) in _SUFFIX_MAP.items():
        if ticker.endswith(suffix):
            symbol = ticker[: -len(suffix)]
            return Stock(symbol, exchange, currency)

    # No suffix → US stock via SMART routing
    return Stock(ticker, "SMART", market.currency)


class IBKRBroker(BrokerInterface):
    """Interactive Brokers implementation of BrokerInterface.

    Wraps ``ib_async.IB`` with synchronous methods.  Connects
    lazily on first use and reconnects automatically.
    """

    def __init__(
        self,
        ibkr_config: "IBKRConfig",
        market_config: "MarketConfig",
    ) -> None:
        self._config = ibkr_config
        self._market = market_config
        self.ib = IB()
        self._connected = False

    def _ensure_connected(self) -> None:
        """Connect to IB Gateway if not already connected."""
        if self.ib.isConnected():
            return
        try:
            self.ib.connect(
                host=self._config.host,
                port=self._config.port,
                clientId=self._config.client_id,
                readonly=False,
            )
            self._connected = True
            mode = (
                "PAPER"
                if self._config.port in (4002, 7497)
                else "LIVE"
            )
            log.info(
                f"Connected to IBKR ({mode}) — "
                f"market={self._market.market_id}, "
                f"client_id={self._config.client_id}"
            )
        except Exception as exc:
            log.error(f"IBKR connection failed: {exc}")
            raise

    def _contract(self, ticker: str) -> Stock:
        """Create an IBKR contract for the given ticker."""
        return _make_contract(ticker, self._market)

    # ── Portfolio state ──────────────────────────────────

    def get_portfolio(self) -> PortfolioSnapshot:
        self._ensure_connected()

        # Account summary
        summary = self.ib.accountSummary(
            account=self._config.account_id or ""
        )
        equity = 0.0
        cash = 0.0
        for item in summary:
            if item.tag == "NetLiquidation":
                equity = float(item.value)
            elif item.tag == "TotalCashValue":
                cash = float(item.value)

        # Apply equity allocation cap
        alloc = self._config.max_equity_allocation
        effective_equity = equity * alloc

        # Positions
        positions = {}
        market_value = 0.0
        for pos in self.ib.positions():
            if pos.contract.symbol:
                sym = pos.contract.symbol
                # Reconstruct suffixed ticker
                suffix = self._exchange_to_suffix(
                    pos.contract.exchange
                    or pos.contract.primaryExchange
                )
                full_ticker = sym + suffix
                mv = float(pos.marketValue or 0)
                positions[full_ticker] = {
                    "qty": float(pos.position),
                    "market_value": abs(mv),
                    "avg_entry": float(pos.avgCost or 0),
                    "unrealized_pnl": float(
                        pos.unrealizedPNL or 0
                    ),
                }
                market_value += abs(mv)

        return PortfolioSnapshot(
            equity=effective_equity,
            cash=cash * alloc,
            buying_power=max(0, effective_equity - market_value),
            market_value=market_value,
            day_pnl=0.0,  # IBKR doesn't provide daily P&L directly
            day_pnl_pct=0.0,
            positions=positions,
        )

    def _exchange_to_suffix(self, exchange: str) -> str:
        """Convert IBKR exchange to yfinance ticker suffix."""
        _map = {
            "LSE": ".L",
            "AQSE": ".L",
            "TSEJ": ".T",
            "JPX": ".T",
            "IBIS": ".DE",
            "SEHK": ".HK",
            "SBF": ".PA",
            "AEB": ".AS",
        }
        return _map.get(exchange or "", "")

    def is_market_open(self) -> bool:
        from shared.trading_calendar import is_session_open
        return is_session_open(self._market)

    # ── Order queries ────────────────────────────────────

    def get_open_orders(self) -> list:
        self._ensure_connected()
        return self.ib.openOrders()

    def get_orders_for_ticker(self, ticker: str) -> list:
        self._ensure_connected()
        contract = self._contract(ticker)
        return [
            o for o in self.ib.openOrders()
            if getattr(o, "symbol", "") == contract.symbol
        ]

    def get_pending_tickers(self) -> set[str]:
        self._ensure_connected()
        result = set()
        for trade in self.ib.openTrades():
            sym = getattr(trade.contract, "symbol", "")
            suffix = self._exchange_to_suffix(
                getattr(trade.contract, "exchange", "")
                or getattr(
                    trade.contract, "primaryExchange", ""
                )
            )
            if sym:
                result.add(sym + suffix)
        return result

    def get_order_by_id(
        self, order_id: str
    ) -> BrokerOrder | None:
        self._ensure_connected()
        for trade in self.ib.trades():
            if str(trade.order.orderId) == order_id:
                return _to_broker_order(trade)
        # Check completed orders
        try:
            completed = self.ib.reqCompletedOrders(
                apiOnly=True
            )
            for trade in completed:
                if str(trade.order.orderId) == order_id:
                    return _to_broker_order(trade)
        except Exception:
            pass
        return None

    def get_filled_orders_for_ticker(
        self,
        ticker: str,
        since_date: str | None = None,
    ) -> list[BrokerOrder]:
        self._ensure_connected()
        contract = self._contract(ticker)
        result = []
        try:
            completed = self.ib.reqCompletedOrders(
                apiOnly=True
            )
            for trade in completed:
                if (
                    getattr(trade.contract, "symbol", "")
                    != contract.symbol
                ):
                    continue
                bo = _to_broker_order(trade)
                if bo.status == OrderStatus.FILLED:
                    if since_date and bo.filled_at:
                        if bo.filled_at < since_date:
                            continue
                    result.append(bo)
        except Exception as exc:
            log.debug(
                f"get_filled_orders_for_ticker failed: {exc}"
            )
        # Newest first
        result.sort(key=lambda o: o.filled_at, reverse=True)
        return result

    # ── Pricing ──────────────────────────────────────────

    def get_latest_price(
        self, ticker: str
    ) -> float | None:
        self._ensure_connected()
        contract = self._contract(ticker)
        try:
            self.ib.qualifyContracts(contract)
            [ticker_data] = self.ib.reqTickers(contract)
            price = (
                ticker_data.marketPrice()
                if ticker_data.marketPrice()
                else ticker_data.close
            )
            return float(price) if price else None
        except Exception as exc:
            log.debug(
                f"get_latest_price({ticker}) failed: {exc}"
            )
            return None

    def get_latest_prices(
        self, tickers: list[str]
    ) -> dict[str, float]:
        self._ensure_connected()
        result = {}
        contracts = [self._contract(t) for t in tickers]
        try:
            self.ib.qualifyContracts(*contracts)
            ticker_data = self.ib.reqTickers(*contracts)
            for t, td in zip(tickers, ticker_data):
                price = (
                    td.marketPrice()
                    if td.marketPrice()
                    else td.close
                )
                if price:
                    result[t] = float(price)
        except Exception as exc:
            log.warning(
                f"Batch price fetch failed: {exc}"
            )
        return result

    # ── Order submission ─────────────────────────────────

    def submit_bracket_order(
        self,
        ticker: str,
        side: str,
        notional: float,
        stop_loss_price: float,
        take_profit_price: float,
        current_price: float,
        time_in_force: str = "day",
        max_entry_slippage_pct: float = 0.0,
    ) -> OrderResult:
        self._ensure_connected()
        try:
            contract = self._contract(ticker)
            self.ib.qualifyContracts(contract)

            if current_price <= 0:
                raise ValueError(
                    f"Invalid current_price={current_price}"
                )

            qty = int(notional / current_price)
            if qty < 1:
                raise ValueError(
                    f"Notional too small for {ticker}"
                )

            action = "BUY" if side == "buy" else "SELL"

            bracket = self.ib.bracketOrder(
                action=action,
                quantity=qty,
                limitPrice=round(
                    current_price
                    * (1 + max_entry_slippage_pct / 100),
                    2,
                )
                if max_entry_slippage_pct > 0
                else 0,
                takeProfitPrice=round(take_profit_price, 2),
                stopLossPrice=round(stop_loss_price, 2),
            )

            # If no slippage limit, convert parent to market
            parent = bracket[0]
            if max_entry_slippage_pct <= 0:
                parent.orderType = "MKT"
                parent.lmtPrice = 0

            # Submit all three legs
            for o in bracket:
                self.ib.placeOrder(contract, o)

            log.info(
                f"Bracket order submitted: {action} "
                f"{qty} {ticker} "
                f"(ID: {parent.orderId})"
            )

            return OrderResult(
                ticker=ticker,
                order_id=str(parent.orderId),
                status="submitted",
                side=side,
                notional=round(qty * current_price, 2),
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )

        except Exception as e:
            log.error(f"Bracket order failed for {ticker}: {e}")
            return OrderResult(
                ticker=ticker,
                status="rejected",
                side=side,
                notional=notional,
                error=str(e),
            )

    def submit_market_order(
        self,
        ticker: str,
        side: str,
        notional: float | None = None,
        qty: float | None = None,
        time_in_force: str = "day",
    ) -> OrderResult:
        self._ensure_connected()
        try:
            contract = self._contract(ticker)
            self.ib.qualifyContracts(contract)

            action = "BUY" if side == "buy" else "SELL"

            if qty is not None:
                order_qty = abs(qty)
            elif notional is not None:
                price = self.get_latest_price(ticker)
                if not price or price <= 0:
                    raise ValueError(
                        f"Cannot determine qty: "
                        f"no price for {ticker}"
                    )
                order_qty = int(notional / price)
            else:
                raise ValueError(
                    "Either notional or qty must be provided"
                )

            order = MarketOrder(action, order_qty)
            trade = self.ib.placeOrder(contract, order)

            log.info(
                f"Market order submitted: {action} "
                f"{order_qty} {ticker} "
                f"(ID: {order.orderId})"
            )

            return OrderResult(
                ticker=ticker,
                order_id=str(order.orderId),
                status="submitted",
                side=side,
                notional=notional or 0.0,
            )

        except Exception as e:
            log.error(
                f"Market order failed for {ticker}: {e}"
            )
            return OrderResult(
                ticker=ticker,
                status="rejected",
                side=side,
                error=str(e),
            )

    # ── Order management ─────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()
        for trade in self.ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self.ib.cancelOrder(trade.order)
                log.info(f"Cancelled order {order_id}")
                return True
        log.warning(f"Order {order_id} not found for cancel")
        return False

    def cancel_orphaned_orders(
        self, positions: dict
    ) -> list[tuple]:
        self._ensure_connected()
        position_tickers = set(positions.keys())
        cancelled = []
        for trade in self.ib.openTrades():
            sym = getattr(trade.contract, "symbol", "")
            suffix = self._exchange_to_suffix(
                getattr(trade.contract, "exchange", "")
            )
            full = sym + suffix
            if full not in position_tickers:
                self.ib.cancelOrder(trade.order)
                cancelled.append(
                    (full, str(trade.order.orderId))
                )
        return cancelled

    # ── Position management ──────────────────────────────

    def close_position(self, ticker: str) -> OrderResult:
        self._ensure_connected()

        # Find the position
        contract = self._contract(ticker)
        pos_qty = 0.0
        for pos in self.ib.positions():
            if (
                getattr(pos.contract, "symbol", "")
                == contract.symbol
            ):
                pos_qty = float(pos.position)
                break

        if pos_qty == 0:
            return OrderResult(
                ticker=ticker,
                status="submitted",
                side="close",
            )

        # Cancel open orders for this ticker first
        for trade in self.ib.openTrades():
            if (
                getattr(trade.contract, "symbol", "")
                == contract.symbol
            ):
                self.ib.cancelOrder(trade.order)

        _time.sleep(1)  # brief settle

        # Submit closing market order
        action = "SELL" if pos_qty > 0 else "BUY"
        return self.submit_market_order(
            ticker=ticker,
            side=action.lower(),
            qty=abs(pos_qty),
        )

    def close_all_positions(self) -> list[OrderResult]:
        self._ensure_connected()
        log.warning("CLOSING ALL IBKR POSITIONS")
        try:
            self.ib.reqGlobalCancel()
            _time.sleep(2)
        except Exception as exc:
            log.error(f"Global cancel failed: {exc}")

        results = []
        for pos in self.ib.positions():
            sym = getattr(pos.contract, "symbol", "")
            suffix = self._exchange_to_suffix(
                getattr(pos.contract, "exchange", "")
                or getattr(
                    pos.contract, "primaryExchange", ""
                )
            )
            if pos.position != 0:
                results.append(
                    self.close_position(sym + suffix)
                )
        return results or [
            OrderResult(
                ticker="ALL", status="submitted",
                side="close",
            )
        ]

    # ── Bracket management (monitor) ─────────────────────

    def submit_oco_reattach(
        self,
        ticker: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        dry_run: bool = False,
    ) -> OrderResult:
        if dry_run:
            return OrderResult(
                ticker=ticker,
                status="dry_run",
                side="sell",
            )

        self._ensure_connected()
        try:
            contract = self._contract(ticker)
            self.ib.qualifyContracts(contract)

            # Cancel existing orders for this ticker
            for trade in self.ib.openTrades():
                if (
                    getattr(trade.contract, "symbol", "")
                    == contract.symbol
                ):
                    self.ib.cancelOrder(trade.order)
            _time.sleep(1)

            # IBKR OCA: two independent orders linked by
            # an OCA group name
            import uuid
            oca_group = f"oco_{ticker}_{uuid.uuid4().hex[:8]}"

            tp_order = LimitOrder(
                "SELL", int(abs(qty)),
                round(take_profit, 2),
            )
            tp_order.ocaGroup = oca_group
            tp_order.ocaType = 1  # cancel other on fill
            tp_order.tif = "GTC"

            sl_order = StopOrder(
                "SELL", int(abs(qty)),
                round(stop_loss, 2),
            )
            sl_order.ocaGroup = oca_group
            sl_order.ocaType = 1
            sl_order.tif = "GTC"

            tp_trade = self.ib.placeOrder(contract, tp_order)
            sl_trade = self.ib.placeOrder(contract, sl_order)

            log.info(
                f"  OCA reattach for {ticker}: "
                f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}"
            )

            return OrderResult(
                ticker=ticker,
                order_id=str(tp_order.orderId),
                status="submitted",
                side="sell",
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )

        except Exception as e:
            log.error(
                f"OCA reattach failed for {ticker}: {e}"
            )
            return OrderResult(
                ticker=ticker,
                status="rejected",
                side="sell",
                error=str(e),
            )

    def update_stop_loss(
        self,
        ticker: str,
        new_stop_loss: float,
        qty: float,
        dry_run: bool = False,
        old_sl_order_id: str = "",
        oco_parent_id: str = "",
        tp_price: float = 0.0,
    ) -> OrderResult:
        if dry_run:
            return OrderResult(
                ticker=ticker,
                status="dry_run",
                side="sell",
            )

        self._ensure_connected()
        try:
            contract = self._contract(ticker)
            self.ib.qualifyContracts(contract)

            if oco_parent_id or old_sl_order_id:
                # Cancel the existing bracket/stop
                cancel_id = oco_parent_id or old_sl_order_id
                self.cancel_order(cancel_id)
                _time.sleep(1)

            if tp_price > 0:
                # Resubmit as OCA (SL + TP linked)
                return self.submit_oco_reattach(
                    ticker=ticker,
                    qty=qty,
                    stop_loss=new_stop_loss,
                    take_profit=tp_price,
                )
            else:
                # Standalone stop order
                sl_order = StopOrder(
                    "SELL", int(abs(qty)),
                    round(new_stop_loss, 2),
                )
                sl_order.tif = "GTC"
                trade = self.ib.placeOrder(
                    contract, sl_order
                )

                return OrderResult(
                    ticker=ticker,
                    order_id=str(sl_order.orderId),
                    status="submitted",
                    side="sell",
                    stop_loss_price=new_stop_loss,
                )

        except Exception as e:
            log.error(
                f"SL replacement failed for {ticker}: {e}"
            )
            return OrderResult(
                ticker=ticker,
                status="rejected",
                side="sell",
                error=str(e),
            )
