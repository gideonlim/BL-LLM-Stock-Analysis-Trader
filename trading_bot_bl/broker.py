"""Alpaca broker interface -- abstracts all API calls."""

from __future__ import annotations

import logging
from typing import Optional

from trading_bot_bl.config import AlpacaConfig
from trading_bot_bl.models import OrderResult, PortfolioSnapshot

log = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        GetAssetsRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
    )
    from alpaca.trading.enums import (
        AssetClass,
        OrderClass,
        OrderSide,
        TimeInForce,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
except ImportError as exc:
    raise ImportError(
        "alpaca-py is required. Install with: "
        "pip install alpaca-py"
    ) from exc


class AlpacaBroker:
    """Wrapper around the Alpaca Trading API."""

    def __init__(self, config: AlpacaConfig) -> None:
        config.validate()
        self._config = config
        self._client = TradingClient(
            api_key=config.api_key,
            secret_key=config.api_secret,
            paper=config.paper,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=config.api_key,
            secret_key=config.api_secret,
        )
        mode = "PAPER" if config.paper else "LIVE"
        log.info(f"Connected to Alpaca ({mode})")

    # ── Account & Portfolio ───────────────────────────────────────────

    def get_portfolio(self) -> PortfolioSnapshot:
        """Fetch current account state and all open positions."""
        account = self._client.get_account()
        positions = self._client.get_all_positions()

        # Fetch entry dates from filled orders. Alpaca's
        # Position object has no timestamp, so we look up
        # the oldest filled BUY order for each symbol.
        entry_dates = self._get_entry_dates(
            [p.symbol for p in positions]
        )

        pos_dict: dict = {}
        for p in positions:
            pos_dict[p.symbol] = {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry": float(p.avg_entry_price),
                "unrealized_pnl": float(p.unrealized_pl),
                "side": str(
                    getattr(p.side, "value", p.side)
                ),
                "entry_date": entry_dates.get(p.symbol),
            }

        equity = float(account.equity)
        last_equity = float(account.last_equity)
        day_pnl = equity - last_equity
        day_pnl_pct = (
            (day_pnl / last_equity * 100)
            if last_equity > 0
            else 0.0
        )

        return PortfolioSnapshot(
            equity=equity,
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            market_value=float(account.long_market_value)
            + float(account.short_market_value),
            day_pnl=round(day_pnl, 2),
            day_pnl_pct=round(day_pnl_pct, 2),
            positions=pos_dict,
        )

    def _get_entry_dates(
        self, symbols: list[str]
    ) -> dict[str, str]:
        """
        Look up the entry date for each position by querying
        closed (filled) orders.

        Alpaca positions have no timestamp field, so we find
        the most recent filled BUY order for each symbol.
        That order's ``filled_at`` is the position entry date.

        Returns:
            Dict of {symbol: ISO date string} for symbols
            where an entry date was found.
        """
        if not symbols:
            return {}
        try:
            from alpaca.trading.requests import (
                GetOrdersRequest,
            )
            from alpaca.trading.enums import (
                QueryOrderStatus,
                OrderSide,
            )

            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=symbols,
                side=OrderSide.BUY,
                limit=500,
            )
            orders = self._client.get_orders(filter=request)

            # For each symbol, find the most recent filled
            # buy order — that's the entry for the current
            # position.
            dates: dict[str, str] = {}
            for order in orders:
                filled_at = getattr(order, "filled_at", None)
                if not filled_at:
                    continue
                sym = order.symbol
                # filled_at is a datetime; convert to ISO date
                date_str = str(filled_at)[:10]
                # Keep the MOST recent fill date per symbol
                # (orders are returned newest-first by default)
                if sym not in dates:
                    dates[sym] = date_str

            if dates:
                log.debug(
                    f"  Entry dates resolved for "
                    f"{len(dates)}/{len(symbols)} positions"
                )
            else:
                log.warning(
                    "  Could not resolve entry dates — "
                    "no filled buy orders found"
                )
            return dates

        except Exception as e:
            log.warning(
                f"  Could not fetch entry dates: {e}"
            )
            return {}

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self._client.get_clock()
        return clock.is_open

    def get_open_orders(self) -> list:
        """
        Fetch all open/pending orders with full detail.

        Returns raw Alpaca Order objects so callers can inspect
        order class, legs, side, symbol, etc.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
            )
            return list(
                self._client.get_orders(filter=request)
            )
        except Exception as e:
            log.warning(f"Could not fetch open orders: {e}")
            return []

    def get_orders_for_ticker(
        self, ticker: str
    ) -> list:
        """Get all open orders for a specific ticker."""
        all_orders = self.get_open_orders()
        return [o for o in all_orders if o.symbol == ticker]

    def get_latest_price(self, ticker: str) -> float | None:
        """
        Get the latest trade price for a ticker.

        Falls back to last quote midpoint if trade data
        is unavailable.
        """
        try:
            request = StockLatestTradeRequest(
                symbol_or_symbols=ticker
            )
            trades = self._data_client.get_stock_latest_trade(
                request
            )
            if ticker in trades:
                return float(trades[ticker].price)
            return None
        except Exception as e:
            log.warning(
                f"Could not get price for {ticker}: {e}"
            )
            return None

    def get_latest_prices(
        self, tickers: list[str]
    ) -> dict[str, float]:
        """Get latest prices for multiple tickers at once."""
        if not tickers:
            return {}
        try:
            request = StockLatestTradeRequest(
                symbol_or_symbols=tickers
            )
            trades = self._data_client.get_stock_latest_trade(
                request
            )
            return {
                sym: float(t.price)
                for sym, t in trades.items()
            }
        except Exception as e:
            log.warning(
                f"Could not get prices: {e}"
            )
            return {}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order by ID."""
        try:
            self._client.cancel_order_by_id(order_id)
            log.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            log.warning(
                f"Could not cancel order {order_id}: {e}"
            )
            return False

    def cancel_orphaned_orders(
        self,
        positions: dict,
    ) -> list[tuple[str, str, bool]]:
        """
        Cancel open orders for tickers we no longer hold.

        If we have shares of AAPL, its bracket legs stay.
        If we have NO shares of TSLA but there are open
        orders for TSLA (e.g. from a previous run, or a
        parent fill that was reversed), cancel them so the
        bot can start fresh.

        Args:
            positions: dict of {ticker: position_info} from
                the portfolio snapshot.

        Returns:
            List of (ticker, order_id, cancelled_ok) tuples.
        """
        all_orders = self.get_open_orders()
        if not all_orders:
            return []

        held_tickers = set(positions.keys())
        results: list[tuple[str, str, bool]] = []

        for order in all_orders:
            symbol = order.symbol
            order_id = str(order.id)

            if symbol in held_tickers:
                # This order belongs to a position we hold
                # — leave it alone
                continue

            # No position for this ticker — cancel the order
            log.info(
                f"  Cancelling stale order for {symbol} "
                f"(no position held): {order_id}"
            )
            ok = self.cancel_order(order_id)
            results.append((symbol, order_id, ok))

        if results:
            cancelled = sum(1 for _, _, ok in results if ok)
            failed = len(results) - cancelled
            log.info(
                f"  Stale order cleanup: {cancelled} cancelled"
                + (f", {failed} failed" if failed else "")
            )
        else:
            log.info("  No stale orders to cancel")

        return results

    def get_pending_tickers(self) -> set[str]:
        """
        Fetch all tickers with open/pending orders.

        This prevents the bot from submitting duplicate orders
        for stocks that already have orders in flight.
        """
        try:
            from alpaca.trading.requests import (
                GetOrdersRequest,
            )
            from alpaca.trading.enums import (
                QueryOrderStatus,
            )

            request = GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
            )
            orders = self._client.get_orders(filter=request)
            tickers = {o.symbol for o in orders}
            if tickers:
                log.info(
                    f"  {len(tickers)} tickers with pending "
                    f"orders: {sorted(tickers)}"
                )
            return tickers
        except Exception as e:
            log.warning(
                f"Could not fetch pending orders: {e}. "
                f"Duplicate check will rely on positions only."
            )
            return set()

    # ── Order Submission ──────────────────────────────────────────────

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
        """
        Submit a bracket order (entry + stop loss + take profit).

        Alpaca bracket orders require whole share qty (fractional
        shares are not allowed with bracket/OCO order class).
        We convert the dollar notional into whole shares using
        the current price.

        If max_entry_slippage_pct > 0, uses a LIMIT order for the
        entry leg (current_price × (1 + slippage%)) so you don't
        fill on a massive gap-up. If 0, uses a market order.

        Args:
            ticker: Stock symbol
            side: "buy" or "sell"
            notional: Dollar amount to trade
            stop_loss_price: Stop loss trigger price
            take_profit_price: Take profit limit price
            current_price: Latest price per share (for qty calc)
            time_in_force: "day" or "gtc"
            max_entry_slippage_pct: Max % above current price
                willing to pay (0 = market order). Default: 0.

        Returns:
            OrderResult with order ID and status.
        """
        try:
            if current_price <= 0:
                raise ValueError(
                    f"Invalid current_price={current_price} "
                    f"for {ticker}"
                )

            # Bracket orders require whole shares, not notional
            qty = int(notional / current_price)
            if qty < 1:
                raise ValueError(
                    f"Notional ${notional:.2f} too small for "
                    f"{ticker} at ${current_price:.2f}/share "
                    f"(need at least ${current_price:.2f})"
                )

            actual_notional = round(qty * current_price, 2)

            order_side = (
                OrderSide.BUY
                if side == "buy"
                else OrderSide.SELL
            )
            tif = (
                TimeInForce.DAY
                if time_in_force == "day"
                else TimeInForce.GTC
            )

            use_limit = max_entry_slippage_pct > 0

            if use_limit:
                # Limit order: won't fill above this price
                limit_price = round(
                    current_price
                    * (1 + max_entry_slippage_pct / 100),
                    2,
                )
                request = LimitOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(
                        limit_price=round(take_profit_price, 2)
                    ),
                    stop_loss=StopLossRequest(
                        stop_price=round(stop_loss_price, 2)
                    ),
                )
            else:
                # Market order: fills immediately at best price
                request = MarketOrderRequest(
                    symbol=ticker,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(
                        limit_price=round(take_profit_price, 2)
                    ),
                    stop_loss=StopLossRequest(
                        stop_price=round(stop_loss_price, 2)
                    ),
                )

            order = self._client.submit_order(request)

            order_type_str = (
                f"LIMIT@${limit_price}" if use_limit else "MKT"
            )
            log.info(
                f"Order submitted: {side.upper()} {qty} shares "
                f"of {ticker} (~${actual_notional:,.2f}) "
                f"[{order_type_str}] (ID: {order.id})"
            )

            return OrderResult(
                ticker=ticker,
                order_id=str(order.id),
                status="submitted",
                side=side,
                notional=actual_notional,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )

        except Exception as e:
            log.error(
                f"Order failed for {ticker}: {e}"
            )
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
        notional: Optional[float] = None,
        qty: Optional[float] = None,
        time_in_force: str = "day",
    ) -> OrderResult:
        """Submit a simple market order (no bracket)."""
        try:
            order_side = (
                OrderSide.BUY
                if side == "buy"
                else OrderSide.SELL
            )
            tif = (
                TimeInForce.DAY
                if time_in_force == "day"
                else TimeInForce.GTC
            )

            kwargs: dict = {
                "symbol": ticker,
                "side": order_side,
                "time_in_force": tif,
            }
            if notional is not None:
                kwargs["notional"] = round(notional, 2)
            elif qty is not None:
                kwargs["qty"] = qty
            else:
                raise ValueError(
                    "Either notional or qty must be provided"
                )

            request = MarketOrderRequest(**kwargs)
            order = self._client.submit_order(request)

            log.info(
                f"Market order submitted: {side.upper()} "
                f"{ticker} (ID: {order.id})"
            )

            return OrderResult(
                ticker=ticker,
                order_id=str(order.id),
                status="submitted",
                side=side,
                notional=notional or 0.0,
            )

        except Exception as e:
            log.error(f"Market order failed for {ticker}: {e}")
            return OrderResult(
                ticker=ticker,
                status="rejected",
                side=side,
                error=str(e),
            )

    def close_position(self, ticker: str) -> OrderResult:
        """Close an existing position entirely.

        Cancels any open orders (OCO brackets, etc.) for the ticker
        first so the full quantity is available for the close.
        """
        # Cancel existing orders that lock up shares (OCO legs, etc.)
        open_orders = self.get_orders_for_ticker(ticker)
        for order in open_orders:
            oid = str(order.id)
            log.info(
                f"  {ticker}: cancelling order {oid} before close"
            )
            self.cancel_order(oid)

        try:
            self._client.close_position(ticker)
            log.info(f"Position closed: {ticker}")
            return OrderResult(
                ticker=ticker,
                status="submitted",
                side="close",
            )
        except Exception as e:
            log.error(f"Failed to close {ticker}: {e}")
            return OrderResult(
                ticker=ticker,
                status="rejected",
                side="close",
                error=str(e),
            )

    def close_all_positions(self) -> list[OrderResult]:
        """Emergency: close every open position."""
        log.warning("CLOSING ALL POSITIONS")
        try:
            self._client.close_all_positions(cancel_orders=True)
            return [
                OrderResult(
                    ticker="ALL",
                    status="submitted",
                    side="close",
                )
            ]
        except Exception as e:
            log.error(f"Failed to close all positions: {e}")
            return [
                OrderResult(
                    ticker="ALL",
                    status="rejected",
                    side="close",
                    error=str(e),
                )
            ]
