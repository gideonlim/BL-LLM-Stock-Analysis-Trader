"""Abstract broker interface for multi-market support.

Defines the contract that ``AlpacaBroker`` and ``IBKRBroker``
both implement.  All monitor/journal/executor code interacts
with this interface — never with broker internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_bot_bl.models import (
        BrokerOrder,
        OrderResult,
        PortfolioSnapshot,
    )


class BrokerInterface(ABC):
    """Abstract base class for broker implementations.

    Every public method here corresponds to a capability the
    execution pipeline requires.  Broker-specific logic (Alpaca
    OCO semantics, IBKR OCA groups, etc.) lives inside the
    concrete implementation — callers never reach into ``_client``.
    """

    # ── Portfolio state ──────────────────────────────────

    @abstractmethod
    def get_portfolio(self) -> "PortfolioSnapshot":
        """Return the current portfolio snapshot."""
        ...

    @abstractmethod
    def is_market_open(self) -> bool:
        """Check if the market is currently in regular session."""
        ...

    # ── Order queries ────────────────────────────────────

    @abstractmethod
    def get_open_orders(self) -> list:
        """Return all open/pending orders."""
        ...

    @abstractmethod
    def get_orders_for_ticker(self, ticker: str) -> list:
        """Return open orders for a specific ticker."""
        ...

    @abstractmethod
    def get_pending_tickers(self) -> set[str]:
        """Return the set of tickers with pending orders."""
        ...

    @abstractmethod
    def get_order_by_id(
        self, order_id: str
    ) -> "BrokerOrder | None":
        """Fetch a single order by its ID.

        Returns a normalized ``BrokerOrder`` or None if not found.
        Replaces direct ``_client`` access in monitor/journal.
        """
        ...

    @abstractmethod
    def get_filled_orders_for_ticker(
        self,
        ticker: str,
        since_date: str | None = None,
    ) -> "list[BrokerOrder]":
        """Return filled orders for *ticker* since *since_date*.

        Replaces inline Alpaca ``GetOrdersRequest`` in journal.py.
        """
        ...

    # ── Pricing ──────────────────────────────────────────

    @abstractmethod
    def get_latest_price(
        self, ticker: str
    ) -> float | None:
        """Return the latest trade price for *ticker*."""
        ...

    @abstractmethod
    def get_latest_prices(
        self, tickers: list[str]
    ) -> dict[str, float]:
        """Return latest prices for multiple tickers."""
        ...

    # ── Order submission ─────────────────────────────────

    @abstractmethod
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
    ) -> "OrderResult":
        """Submit a bracket (entry + SL + TP) order.

        Parameters match the existing AlpacaBroker contract exactly
        to ensure substitutability without call-site changes.

        Args:
            ticker: Stock symbol.
            side: ``"buy"`` or ``"sell"``.
            notional: Dollar (or local currency) amount to trade.
            stop_loss_price: Stop loss trigger price.
            take_profit_price: Take profit limit price.
            current_price: Latest price per share (for qty calc).
            time_in_force: ``"day"`` or ``"gtc"``.
            max_entry_slippage_pct: Max % above current price
                willing to pay (0 = market order).
        """
        ...

    @abstractmethod
    def submit_market_order(
        self,
        ticker: str,
        side: str,
        notional: float | None = None,
        qty: float | None = None,
        time_in_force: str = "day",
    ) -> "OrderResult":
        """Submit a simple market order (no bracket).

        Parameters match the existing AlpacaBroker contract exactly.
        Either *notional* or *qty* must be provided.
        """
        ...

    # ── Order management ─────────────────────────────────

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID. Returns True on success."""
        ...

    @abstractmethod
    def cancel_orphaned_orders(
        self, positions: dict
    ) -> list[tuple]:
        """Cancel orders for tickers no longer in the portfolio."""
        ...

    # ── Position management ──────────────────────────────

    @abstractmethod
    def close_position(self, ticker: str) -> "OrderResult":
        """Close the entire position for *ticker*."""
        ...

    @abstractmethod
    def close_all_positions(self) -> "list[OrderResult]":
        """Emergency: close all positions."""
        ...

    # ── Bracket management (monitor) ─────────────────────

    @abstractmethod
    def submit_oco_reattach(
        self,
        ticker: str,
        qty: float,
        stop_loss: float,
        take_profit: float,
        dry_run: bool = False,
    ) -> "OrderResult":
        """Reattach an OCO bracket (SL + TP) to an existing position.

        Replaces inline Alpaca ``OrderClass.OCO`` code in monitor.py.
        """
        ...

    @abstractmethod
    def update_stop_loss(
        self,
        ticker: str,
        new_stop_loss: float,
        qty: float,
        dry_run: bool = False,
        old_sl_order_id: str = "",
        oco_parent_id: str = "",
        tp_price: float = 0.0,
    ) -> "OrderResult":
        """Replace the stop-loss on an existing bracket.

        Parameters match the full call-site contract in monitor.py
        so future broker implementations won't raise TypeError.

        Args:
            ticker: Position symbol.
            new_stop_loss: New stop-loss price.
            qty: Position quantity (positive).
            dry_run: If True, skip actual submission.
            old_sl_order_id: Order ID of the standalone SL leg to
                cancel (empty string if OCO-managed).
            oco_parent_id: Order ID of the OCO parent to cancel
                (set when the bracket is an OCO).  Broker cancels
                the parent atomically and resubmits with the new SL.
            tp_price: Existing take-profit price to preserve in
                the replacement OCO (required when *oco_parent_id*
                is set).
        """
        ...
