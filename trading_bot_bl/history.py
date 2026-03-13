"""Trade history -- loads past execution logs and evaluates strategy performance."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Skip-reason classification ────────────────────────────────────
# These patterns indicate a rejection that is NOT the strategy's
# fault — infrastructure errors, portfolio limits, code bugs, etc.
# Only "strategy-attributable" skips should count against a
# strategy's track record.
_INFRA_PATTERNS = [
    # Broker/API errors (code bugs, connection issues)
    r"fractional orders",
    r"insufficient",
    r"connection",
    r"timeout",
    r"APIError",
    r'"code":\d+',        # JSON error responses from Alpaca
    # Portfolio-level limits (not the strategy's fault)
    r"Portfolio exposure .* >= max",
    r"Adjusted notional .* too small",
    r"Account equity is zero",
    r"Daily loss circuit breaker",
    # Churn / duplicate detection (timing, not strategy quality)
    r"was already bought on",
    r"avoiding churn",
    r"already held",
    r"pending order exists",
    # Strategy history itself (don't double-count)
    r"has poor track record",
    r"currently losing",
]

_INFRA_RE = re.compile("|".join(_INFRA_PATTERNS), re.IGNORECASE)


def _is_strategy_attributable(error: str) -> bool:
    """
    Return True if a skip/rejection reason reflects genuine
    strategy weakness (bad signal quality, too few trades, etc.).

    Returns False for infrastructure errors, portfolio limits,
    broker bugs, and duplicate-detection skips — none of which
    say anything about the strategy itself.
    """
    if not error:
        return False
    # If the error matches an infrastructure pattern, it's NOT
    # the strategy's fault
    if _INFRA_RE.search(error):
        return False
    # Everything else (composite score too low, confidence too
    # low, signal expired, too few backtest trades) IS the
    # strategy's fault
    return True


@dataclass
class StrategyRecord:
    """Aggregated performance for one strategy across all tickers."""

    strategy: str
    total_orders: int = 0
    submitted: int = 0
    rejected_by_broker: int = 0       # broker/infra errors
    skipped_by_risk: int = 0          # portfolio limits, dupes
    skipped_by_quality: int = 0       # strategy's own weakness
    total_notional: float = 0.0
    tickers: set = field(default_factory=set)
    # Populated when broker P&L data is available
    realized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0

    @property
    def strategy_relevant_total(self) -> int:
        """Orders where the outcome reflects strategy quality."""
        return self.submitted + self.skipped_by_quality

    @property
    def success_rate(self) -> float:
        """
        Fraction of strategy-relevant orders that were filled.

        Only counts submitted vs. quality-skipped.  Broker errors
        and portfolio-level limits are excluded because they don't
        reflect on the strategy itself.
        """
        total = self.strategy_relevant_total
        if total == 0:
            return 1.0  # no data = benefit of the doubt
        return self.submitted / total

    @property
    def win_rate(self) -> float:
        """Win rate from realized trades (needs broker P&L data)."""
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.0
        return self.win_count / total


@dataclass
class TickerHistory:
    """Trade history for a single ticker."""

    ticker: str
    last_buy_date: str = ""
    last_buy_strategy: str = ""
    last_buy_notional: float = 0.0
    total_buys: int = 0
    total_exits: int = 0
    strategies_used: set = field(default_factory=set)


@dataclass
class TradeHistory:
    """
    Aggregated trade history loaded from execution logs.

    Provides:
    - Per-ticker memory (what was bought, when, by which strategy)
    - Per-strategy performance stats
    - Duplicate detection across days
    """

    by_ticker: dict[str, TickerHistory] = field(default_factory=dict)
    by_strategy: dict[str, StrategyRecord] = field(default_factory=dict)
    log_count: int = 0
    date_range: tuple[str, str] = ("", "")

    def was_recently_traded(
        self, ticker: str, days: int = 3
    ) -> bool:
        """Check if a ticker was bought in the last N days."""
        if ticker not in self.by_ticker:
            return False
        th = self.by_ticker[ticker]
        if not th.last_buy_date:
            return False
        try:
            last = datetime.fromisoformat(th.last_buy_date)
            cutoff = datetime.now() - timedelta(days=days)
            return last >= cutoff
        except (ValueError, TypeError):
            return False

    def get_strategy_record(
        self, strategy: str
    ) -> Optional[StrategyRecord]:
        """Get performance stats for a strategy."""
        return self.by_strategy.get(strategy)

    def get_ticker_history(
        self, ticker: str
    ) -> Optional[TickerHistory]:
        """Get trade history for a specific ticker."""
        return self.by_ticker.get(ticker)

    def strategy_is_underperforming(
        self,
        strategy: str,
        min_orders: int = 5,
        min_success_rate: float = 0.5,
    ) -> bool:
        """
        Check if a strategy has a poor track record.

        Only flags strategies with enough *strategy-relevant*
        data (>= min_orders).  Broker errors and portfolio
        limits don't count against the strategy.
        """
        rec = self.by_strategy.get(strategy)
        if rec is None:
            return False
        if rec.strategy_relevant_total < min_orders:
            return False  # not enough data to judge
        return rec.success_rate < min_success_rate


def load_trade_history(
    log_dir: Path,
    lookback_days: int = 30,
) -> TradeHistory:
    """
    Load and aggregate execution logs from the log directory.

    Reads all execution_*.json files within the lookback window
    and builds per-ticker and per-strategy records.

    Args:
        log_dir: Directory containing execution_*.json files.
        lookback_days: Only load logs from the last N days.

    Returns:
        Aggregated TradeHistory.
    """
    history = TradeHistory()

    if not log_dir.exists():
        log.info(
            f"No execution log directory at {log_dir} — "
            f"starting with empty history"
        )
        return history

    log_files = sorted(log_dir.glob("execution_*.json"))
    if not log_files:
        log.info("No execution logs found — empty history")
        return history

    cutoff = datetime.now() - timedelta(days=lookback_days)
    loaded = 0
    earliest = ""
    latest = ""

    for path in log_files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping corrupt log {path}: {e}")
            continue

        # Parse execution timestamp
        executed_at = data.get("executed_at", "")
        try:
            exec_dt = datetime.fromisoformat(executed_at)
            if exec_dt < cutoff:
                continue  # outside lookback window
        except (ValueError, TypeError):
            pass  # keep logs with unparseable dates

        loaded += 1
        if not earliest or executed_at < earliest:
            earliest = executed_at
        if not latest or executed_at > latest:
            latest = executed_at

        # Process each order in the log
        for order in data.get("orders", []):
            ticker = order.get("ticker", "")
            status = order.get("status", "")
            side = order.get("side", "")
            notional = order.get("notional", 0.0)
            error = order.get("error", "")
            strategy = _extract_strategy(order)

            if not ticker or ticker == "ALL":
                continue

            # ── Update ticker history ───────────────────────
            if ticker not in history.by_ticker:
                history.by_ticker[ticker] = TickerHistory(
                    ticker=ticker
                )
            th = history.by_ticker[ticker]

            if side == "buy":
                th.total_buys += 1
                if status == "submitted":
                    th.last_buy_date = executed_at
                    th.last_buy_strategy = strategy
                    th.last_buy_notional = notional
                if strategy:
                    th.strategies_used.add(strategy)
            elif side in ("sell", "close"):
                th.total_exits += 1

            # ── Update strategy history ────────────────────
            if strategy:
                if strategy not in history.by_strategy:
                    history.by_strategy[strategy] = (
                        StrategyRecord(strategy=strategy)
                    )
                sr = history.by_strategy[strategy]
                sr.total_orders += 1
                sr.tickers.add(ticker)
                sr.total_notional += notional

                if status == "submitted":
                    sr.submitted += 1
                elif status == "rejected":
                    # Broker-level rejection (API error, etc.)
                    sr.rejected_by_broker += 1
                elif status == "skipped":
                    # Classify: was this the strategy's fault?
                    if _is_strategy_attributable(error):
                        sr.skipped_by_quality += 1
                    else:
                        sr.skipped_by_risk += 1

    history.log_count = loaded
    history.date_range = (earliest, latest)

    log.info(
        f"Loaded trade history: {loaded} logs, "
        f"{len(history.by_ticker)} tickers, "
        f"{len(history.by_strategy)} strategies "
        f"(range: {earliest[:10] if earliest else '?'} to "
        f"{latest[:10] if latest else '?'})"
    )

    return history


def enrich_history_with_pnl(
    history: TradeHistory,
    positions: dict,
) -> None:
    """
    Update strategy records with realized P&L from broker positions.

    For each position currently held, if we know which strategy
    placed it (from ticker history), we attribute the unrealized
    P&L to that strategy.

    Args:
        history: The trade history to enrich.
        positions: portfolio.positions dict from the broker.
    """
    for ticker, pos in positions.items():
        pnl = pos.get("unrealized_pnl", 0.0)
        th = history.by_ticker.get(ticker)
        if th and th.last_buy_strategy:
            strategy = th.last_buy_strategy
            sr = history.by_strategy.get(strategy)
            if sr:
                sr.realized_pnl += pnl
                if pnl >= 0:
                    sr.win_count += 1
                else:
                    sr.loss_count += 1


def _extract_strategy(order: dict) -> str:
    """
    Extract strategy name from an execution log order entry.

    Prefers the explicit 'strategy' field (added in v2).
    Falls back to parsing the reason/error string for older logs.
    """
    # Direct field (v2+ logs)
    strategy = order.get("strategy", "")
    if strategy:
        return strategy

    # Fallback: parse from reason/error string (v1 logs)
    for field_name in ("error", "reason"):
        text = order.get(field_name, "")
        if "signal from " in text:
            # "BUY signal from mean_reversion (score=45.2, ...)"
            after = text.split("signal from ", 1)[1]
            strategy = after.split(" (")[0].split(" —")[0].strip()
            if strategy:
                return strategy
    return ""
