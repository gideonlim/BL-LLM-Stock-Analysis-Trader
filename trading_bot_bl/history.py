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
    r"Already at \d+ positions",       # max concurrent positions cap
    r"already at \$",                  # per-ticker concentration limit
    r"Adjusted notional .* too small",
    r"Account equity is zero",
    r"Daily loss circuit breaker",
    # Market regime halts (not strategy quality)
    r"SEVERE_BEAR.*all new entries halted",
    # Earnings calendar (not strategy quality)
    r"Earnings (?:in|TODAY|was) ",
    # Churn / duplicate detection (timing, not strategy quality)
    r"was already bought on",
    r"avoiding churn",
    r"already held",
    r"pending order exists",
    # Strategy history itself (don't double-count)
    r"has poor track record",
    r"currently losing",
    # Signal-level quality filters — these reflect individual signal
    # weakness, NOT the strategy being bad.  A VWAP signal scoring 80
    # should not be penalised because another VWAP signal scored 15.
    # Counting these against the strategy creates a feedback loop in
    # bear markets where regime filters legitimately block most signals,
    # which tanks the strategy "success rate" and then blocks ALL signals
    # from that strategy — even excellent ones.
    r"Signal is (?:HOLD|ERROR)",     # signal says don't trade
    r"Composite score .* < min",
    r"Too few backtest trades",
    r"PBO .* exceeds max",
    r"Confidence score .* < min",
    r"Signal expired",
    r"Notional .* below minimum",
    r"ADV .* below minimum",
]

_INFRA_RE = re.compile("|".join(_INFRA_PATTERNS), re.IGNORECASE)


def _is_strategy_attributable(error: str) -> bool:
    """
    Return True if a skip reason reflects the strategy *itself*
    being unreliable — e.g. historically losing money on filled
    trades.

    Returns False for:
    - Infrastructure errors, portfolio limits, broker bugs
    - Signal-level quality filters (composite score, PBO, trade
      count, confidence, expiry).  These reflect individual signal
      weakness, not strategy weakness.  Counting them against the
      strategy creates a feedback loop in bear markets.
    """
    if not error:
        return False
    # If the error matches an infrastructure pattern, it's NOT
    # the strategy's fault
    if _INFRA_RE.search(error):
        return False
    # Anything not matching the exclusion patterns is genuinely
    # attributable to the strategy (e.g. a novel rejection reason
    # we haven't categorised yet).
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
    # Populated from journal closed trades
    realized_pnl: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    closed_trades: int = 0
    # Populated from broker open positions (point-in-time snapshot)
    unrealized_pnl: float = 0.0
    open_positions: int = 0

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
    last_sell_date: str = ""
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
        """Check if a ticker was bought or sold in the last N days."""
        if ticker not in self.by_ticker:
            return False
        th = self.by_ticker[ticker]
        cutoff = datetime.now() - timedelta(days=days)
        for dt_str in (th.last_buy_date, th.last_sell_date):
            if not dt_str:
                continue
            try:
                if datetime.fromisoformat(dt_str) >= cutoff:
                    return True
            except (ValueError, TypeError):
                continue
        return False

    def recent_trade_reason(
        self, ticker: str, days: int = 3
    ) -> Optional[str]:
        """Return a human-readable reason if the ticker was recently
        traded, or ``None`` if not.  Checks both buy and sell dates
        so the risk manager can log the appropriate cooldown cause.
        """
        if ticker not in self.by_ticker:
            return None
        th = self.by_ticker[ticker]
        cutoff = datetime.now() - timedelta(days=days)

        # Check sell first — selling yesterday then re-buying is
        # the more common churn pattern we want to catch.
        if th.last_sell_date:
            try:
                if datetime.fromisoformat(th.last_sell_date) >= cutoff:
                    return (
                        f"{ticker} was sold on "
                        f"{th.last_sell_date[:10]} — "
                        f"{days}-day post-exit cooldown"
                    )
            except (ValueError, TypeError):
                pass
        if th.last_buy_date:
            try:
                if datetime.fromisoformat(th.last_buy_date) >= cutoff:
                    return (
                        f"{ticker} was already bought on "
                        f"{th.last_buy_date[:10]} via "
                        f"'{th.last_buy_strategy}' — "
                        f"avoiding churn ({days}-day cooldown)"
                    )
            except (ValueError, TypeError):
                pass
        return None

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
    include_dry_runs: bool = False,
) -> TradeHistory:
    """
    Load and aggregate execution logs from the log directory.

    Reads all execution_*.json files within the lookback window
    and builds per-ticker and per-strategy records.

    Args:
        log_dir: Directory containing execution_*.json files.
        lookback_days: Only load logs from the last N days.
        include_dry_runs: When True, dry-run orders count toward
            cooldown dates (last_buy_date / last_sell_date).
            Set to True when the current run is also a dry run
            so cooldowns are simulated accurately.  Defaults to
            False so dry-run logs never block real live orders.

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

            # Statuses that count toward cooldown dates.
            # dry_run orders only count when the caller opts in
            # (i.e. the current run is also a dry run) so that
            # simulated orders never block real live trades.
            _counts_for_cooldown = (
                status == "submitted"
                or (status == "dry_run" and include_dry_runs)
            )

            if side == "buy":
                th.total_buys += 1
                if _counts_for_cooldown:
                    th.last_buy_date = executed_at
                    th.last_buy_strategy = strategy
                    th.last_buy_notional = notional
                if strategy:
                    th.strategies_used.add(strategy)
            elif side in ("sell", "close"):
                th.total_exits += 1
                if _counts_for_cooldown:
                    th.last_sell_date = executed_at

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


def reconcile_with_journal(
    history: TradeHistory,
    journal_dir: Path,
) -> None:
    """Clear churn state for orders that were never filled.

    Scans the journal directory and:

    1. **Cancelled buy orders** — if a ticker's ``last_buy_date``
       matches a journal entry closed with
       ``exit_reason == "order_cancelled"``, the buy-side churn
       fields are reset so the ticker isn't blocked by a cooldown
       that never applied.

    2. **Unfilled sell orders** — if a ticker has a
       ``last_sell_date`` but the journal shows the position is
       still open (``status != "closed"``), the sell submission
       didn't actually complete, so ``last_sell_date`` is cleared
       to avoid a false post-exit cooldown.
    """
    if not journal_dir.exists():
        return

    cancelled_tickers: dict[str, str] = {}  # ticker → opened_at
    # Track tickers with open positions (sell didn't complete)
    open_tickers: set[str] = set()

    for path in journal_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            ticker = d.get("ticker", "")
            status = d.get("status", "")

            if (
                d.get("exit_reason") == "order_cancelled"
                and status == "closed"
            ):
                opened = d.get("opened_at", "")
                if ticker and opened:
                    prev = cancelled_tickers.get(ticker, "")
                    if opened > prev:
                        cancelled_tickers[ticker] = opened

            # Position still open → any recorded sell didn't fill
            if ticker and status == "open":
                open_tickers.add(ticker)

        except (json.JSONDecodeError, OSError):
            continue

    cleared = 0

    # 1. Clear buy-side churn for cancelled entry orders
    for ticker, cancelled_at in cancelled_tickers.items():
        th = history.by_ticker.get(ticker)
        if th and th.last_buy_date:
            # Only clear if the last buy IS the cancelled order
            # (i.e. no subsequent successful buy superseded it).
            if th.last_buy_date[:10] <= cancelled_at[:10]:
                th.last_buy_date = ""
                th.last_buy_strategy = ""
                th.last_buy_notional = 0.0
                cleared += 1

    # 2. Clear sell-side churn for unfilled exit orders
    for ticker in open_tickers:
        th = history.by_ticker.get(ticker)
        if th and th.last_sell_date:
            th.last_sell_date = ""
            cleared += 1

    if cleared:
        log.info(
            f"Journal reconcile: cleared churn state for "
            f"{cleared} ticker(s) with unfilled orders"
        )


def enrich_history_with_pnl(
    history: TradeHistory,
    positions: dict,
    journal_dir: Path | None = None,
) -> None:
    """
    Update strategy records with P&L from two sources:

    1. **Realized P&L** from journal closed trades (accurate,
       reflects actual trade outcomes).
    2. **Unrealized P&L** from broker open positions (snapshot,
       stored separately so it doesn't contaminate realized stats).

    Safe to call only once per history object.  The ``_pnl_enriched``
    flag prevents accidental double-counting if called again.

    Args:
        history: The trade history to enrich.
        positions: portfolio.positions dict from the broker.
        journal_dir: Path to journal directory.  If provided,
            closed trades are loaded and their realized P&L is
            aggregated by strategy.
    """
    if getattr(history, "_pnl_enriched", False):
        log.debug("enrich_history_with_pnl already called — skipping")
        return
    history._pnl_enriched = True  # type: ignore[attr-defined]

    # ── 1. Realized P&L from journal closed trades ────────────
    if journal_dir and journal_dir.exists():
        try:
            from trading_bot_bl.journal import load_all_trades

            closed = [
                t for t in load_all_trades(
                    journal_dir,
                    lookback_days=history.lookback_days
                    if hasattr(history, "lookback_days") else 90,
                )
                if t.status == "closed"
            ]
            for t in closed:
                sr = history.by_strategy.get(t.strategy)
                if sr:
                    pnl = t.realized_pnl or 0.0
                    sr.realized_pnl += pnl
                    sr.closed_trades += 1
                    if pnl > 0:
                        sr.win_count += 1
                    elif pnl < 0:
                        sr.loss_count += 1
        except Exception as exc:
            log.warning(f"Journal P&L enrichment failed: {exc}")

    # ── 2. Unrealized P&L from open positions (separate field) ─
    for ticker, pos in positions.items():
        pnl = pos.get("unrealized_pnl", 0.0)
        th = history.by_ticker.get(ticker)
        if th and th.last_buy_strategy:
            strategy = th.last_buy_strategy
            sr = history.by_strategy.get(strategy)
            if sr:
                sr.unrealized_pnl += pnl
                sr.open_positions += 1


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
