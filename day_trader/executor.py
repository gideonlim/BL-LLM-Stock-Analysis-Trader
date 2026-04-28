"""Day-trader executor — the asyncio daemon tying it all together.

Lifecycle of one trading session:

1. ``pre_session()`` — catalyst refresh, premarket scan, recovery
   reconciliation, build watchlist, start feed, build schedule.
2. ``run_session()`` — poll the scheduler; dispatch events (scan,
   manage positions, force-flat) until session close.
3. ``post_session()`` — flush journal, equity snapshot, stop feed.

The executor is meant to be driven by ``__main__.py`` which handles
the outer ``while True: wait for next session, run session`` loop.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from day_trader.broker_helpers import close_tagged_daytrade_qty
from day_trader.budget import SubBudgetTracker
from day_trader.calendar import NyseSession, now_et, session_for
from day_trader.config import DayTradeConfig
from day_trader.data.cache import BarCache
from day_trader.data.catalyst import CatalystClassifier
from day_trader.data.feed import MarketDataFeed
from day_trader.data.premarket import PremarketScanner
from day_trader.data.universe import load_universe
from day_trader.filters.base import FilterPipeline
from day_trader.filters.cooldown import CooldownTracker
from day_trader.journal_adapter import close_daytrade, create_daytrade
from day_trader.models import (
    Bar,
    DayTradeSignal,
    FilterContext,
    MarketState,
    OpenDayTrade,
    Quote,
)
from day_trader.order_tags import SequenceCounter, make_order_id
from day_trader.position_manager import PositionManager
from day_trader.recovery import reconcile
from day_trader.risk import DayRiskManager
from day_trader.scheduler import Scheduler
from day_trader.strategies.base import DayTradeStrategy
from day_trader.symbol_locks import SymbolLock

log = logging.getLogger(__name__)


class DayTraderDaemon:
    """The main daemon orchestrating one day-trading session.

    Construction wires together all the components built in prior
    commits; the ``run()`` coroutine is the top-level entry point
    that ``__main__.py`` awaits.
    """

    def __init__(
        self,
        *,
        broker,
        config: DayTradeConfig,
        strategies: list[DayTradeStrategy],
        pipeline: FilterPipeline,
        feed: MarketDataFeed,
        premarket_scanner: PremarketScanner,
        catalyst_classifier: CatalystClassifier,
        seq_counter: SequenceCounter,
    ):
        self.broker = broker
        self.config = config
        self.strategies = {s.name: s for s in strategies}
        self.pipeline = pipeline
        self.feed = feed
        self.premarket_scanner = premarket_scanner
        self.catalyst_classifier = catalyst_classifier
        self.seq = seq_counter

        # Internal state — built at session start
        self.bar_cache = BarCache()
        self.position_mgr = PositionManager()
        self.cooldowns = CooldownTracker(
            ticker_minutes=config.risk.ticker_cooldown_minutes,
            strategy_minutes=config.risk.strategy_cooldown_minutes,
        )
        self.budget = SubBudgetTracker(budget_pct=config.risk.budget_pct)
        self.risk = DayRiskManager(
            limits=config.risk,
            budget=self.budget,
            cooldowns=self.cooldowns,
        )
        self.symbol_lock = SymbolLock(broker)
        self.scheduler: Optional[Scheduler] = None
        self.market_state = MarketState()
        self.watchlist: list[str] = []
        self.ticker_contexts: dict = {}
        self.quotes: dict[str, Quote] = {}
        self._exit_only = False
        self._session: Optional[NyseSession] = None

    # ── Main entry point ──────────────────────────────────────────

    async def run(self) -> None:
        """Run one complete trading session."""
        session = session_for()
        if session is None:
            log.info("No NYSE session today — standing down")
            return
        self._session = session
        log.info(
            "Session: %s, open=%s, close=%s, half_day=%s",
            session.date, session.open_et.strftime("%H:%M"),
            session.close_et.strftime("%H:%M"), session.is_half_day,
        )

        try:
            await self._pre_session(session)
            await self._run_session(session)
        except Exception:
            log.exception("Executor: fatal error during session")
            raise
        finally:
            await self._post_session()

    # ── Pre-session ───────────────────────────────────────────────

    async def _pre_session(self, session: NyseSession) -> None:
        """Set up everything before the first scan tick fires."""
        # Reset state
        self.bar_cache.reset_session()
        self.position_mgr.reset_for_session()
        for s in self.strategies.values():
            s.reset_for_session()
        self._exit_only = False

        # Load universe
        universe = load_universe(
            csv_path=(
                self.config.universe_path
                if self.config.universe_path
                else None
            ),
        )
        log.info("Universe: %d symbols", len(universe))

        # Catalyst classification (sync — runs once, ~30s)
        catalyst_labels = self.catalyst_classifier.classify_many(universe)

        # Premarket scan
        self.ticker_contexts = self.premarket_scanner.scan(
            universe,
            target_date=session.date,
            top_n=self.config.max_watchlist_size,
        )
        # Merge catalyst labels into contexts
        for ticker, ctx in self.ticker_contexts.items():
            if not ctx.catalyst_label:
                ctx.catalyst_label = catalyst_labels.get(ticker, "")
        self.watchlist = sorted(self.ticker_contexts.keys())
        log.info("Watchlist: %d symbols", len(self.watchlist))

        # Recovery reconciliation
        recon = reconcile(self.broker, self.config.journal_dir)
        if not recon.is_clean:
            log.error(
                "INCIDENT MODE: %s — blocking all entries for today",
                recon.summary(),
            )
            self.risk.trip_kill_switch(
                f"recovery_incident: {recon.summary()[:200]}"
            )
        else:
            log.info("Recovery: %s", recon.summary())

        # Start the risk manager
        portfolio = self.broker.get_portfolio()
        self.risk.start_session(
            equity=portfolio.equity,
            initial_open_notional=recon.open_notional,
            initial_positions=len(recon.daytrade_position_qty),
            today=session.date,
        )

        # Re-seed positions from recovery
        for entry in recon.open_journal_entries:
            self.position_mgr.open_position(OpenDayTrade(
                ticker=entry.ticker,
                strategy=entry.strategy,
                side=entry.side,
                qty=entry.entry_qty,
                entry_price=entry.entry_fill_price,
                entry_time=now_et(),
                sl_price=entry.original_sl_price,
                tp_price=entry.original_tp_price,
                parent_client_order_id=entry.entry_order_id,
                seq=0,
            ))

        # Refresh symbol lock
        self.symbol_lock.refresh()

        # Build schedule
        self.scheduler = Scheduler.for_session(
            session,
            catalyst_refresh_min_before_open=(
                self.config.catalyst_refresh_min_before_open
            ),
            premarket_scan_min_before_open=(
                self.config.premarket_scan_min_before_open
            ),
            regime_snapshot_min_before_open=(
                self.config.regime_snapshot_min_before_open
            ),
            first_scan_min_after_open=(
                self.config.first_scan_min_after_open
            ),
            exit_only_min_before_close=(
                self.config.exit_only_min_before_close
            ),
            force_flat_min_before_close=(
                self.config.force_flat_min_before_close
            ),
        )

        # Start the live data feed
        async def on_bar(bar: Bar) -> None:
            bar = self.bar_cache.add_bar(bar)

        async def on_quote(q: Quote) -> None:
            self.quotes[q.ticker] = q

        self.feed.on_bar(on_bar)
        self.feed.on_quote(on_quote)
        await self.feed.start(self.watchlist)

    # ── Session main loop ─────────────────────────────────────────

    async def _run_session(self, session: NyseSession) -> None:
        """Poll scheduler and dispatch events until session closes."""
        while True:
            current = now_et()
            if current >= session.close_et:
                break

            events = self.scheduler.due_events(current)
            for event in events:
                await self._handle_event(event.name, session)

            # Adaptive sleep: if next event is far away, sleep longer
            # to save CPU. Never sleep more than 5s so the daemon
            # stays responsive to stop signals.
            next_at = self.scheduler.next_event_at(current)
            if next_at is not None:
                delay = min(
                    (next_at - now_et()).total_seconds(), 5.0,
                )
                delay = max(delay, 0.2)
            else:
                delay = 1.0
            await asyncio.sleep(delay)

    async def _handle_event(
        self, event_name: str, session: NyseSession,
    ) -> None:
        """Dispatch a single scheduled event."""
        log.debug("Event: %s", event_name)

        if event_name == "regime_snapshot":
            await self._snapshot_regime()

        elif event_name in ("first_scan", "scan_tick"):
            if not self._exit_only:
                await self._run_scan(session)
            # Also manage open positions every tick
            await self._manage_positions(session)

        elif event_name == "exit_only":
            self._exit_only = True
            log.info("Exit-only mode — no new entries")

        elif event_name == "force_flat":
            await self._force_flat_all()

        elif event_name == "session_close":
            log.info("Session close")

        # Other events (market_open, catalyst_refresh, etc.) are
        # informational — the pre_session already handled them.

    # ── Scan + entry ──────────────────────────────────────────────

    async def _run_scan(self, session: NyseSession) -> None:
        """Iterate strategies, emit signals, filter, risk-check, submit."""
        if not self.risk.can_take_more_trades():
            return

        current = now_et()
        for strat in self.strategies.values():
            candidates = strat.scan(
                self.watchlist,
                self.bar_cache,
                self.ticker_contexts,
                self.market_state,
                current,
                session,
            )
            for sig in candidates:
                await self._process_signal(sig, session)

    async def _process_signal(
        self, sig: DayTradeSignal, session: NyseSession,
    ) -> None:
        """Filter → risk-check → size → submit one signal."""
        if not self.risk.can_take_more_trades():
            return

        # Build filter context
        ctx = FilterContext(
            signal=sig,
            quote=self.quotes.get(sig.ticker),
            bars=self.bar_cache.get_bars(sig.ticker),
            market_state=self.market_state,
            open_positions={
                t: p for t, p in zip(
                    self.position_mgr.tickers(),
                    self.position_mgr.all_positions(),
                )
            },
        )
        result = self.pipeline.evaluate(ctx)
        if not result.passed:
            return

        # Size the order
        risk_per_share = abs(sig.signal_price - sig.stop_loss_price)
        if risk_per_share <= 0:
            return
        max_risk = self.risk.session_starting_equity * (
            self.config.risk.per_trade_risk_pct / 100
        )
        qty = math.floor(max_risk / risk_per_share)
        if qty < self.config.risk.min_qty:
            return
        notional = qty * sig.signal_price

        # Risk manager review
        verdict = self.risk.review(sig, notional, qty * risk_per_share)
        if not verdict.approved:
            return

        # Submit order
        seq = self.seq.next()
        tag = make_order_id(seq, sig.ticker)

        if self.config.dry_run:
            log.info(
                "[DRY-RUN] Would submit: %s %s %d sh @ %.2f (tag=%s)",
                sig.side, sig.ticker, qty, sig.signal_price, tag,
            )
            return

        order_result = self.broker.submit_bracket_order(
            ticker=sig.ticker,
            side=sig.side,
            notional=notional,
            stop_loss_price=sig.stop_loss_price,
            take_profit_price=sig.take_profit_price,
            current_price=sig.signal_price,
            time_in_force="day",
            max_entry_slippage_pct=(
                self.config.risk.per_trade_risk_pct  # reuse as slippage cap
            ),
            client_order_id=tag,
        )

        if order_result.status != "submitted":
            log.warning(
                "Order rejected by broker: %s %s — %s",
                sig.ticker, order_result.error, tag,
            )
            return

        # Record the fill (optimistic — we'll reconcile at next session)
        self.risk.record_fill(notional)
        self.position_mgr.open_position(OpenDayTrade(
            ticker=sig.ticker,
            strategy=sig.strategy,
            side="long" if sig.side == "buy" else "short",
            qty=qty,
            entry_price=sig.signal_price,
            entry_time=now_et(),
            sl_price=sig.stop_loss_price,
            tp_price=sig.take_profit_price,
            parent_client_order_id=tag,
            seq=seq,
        ))

        # Write journal entry
        create_daytrade(
            order_id=order_result.order_id,
            ticker=sig.ticker,
            strategy=sig.strategy,
            side=sig.side,
            signal_price=sig.signal_price,
            notional=notional,
            sl_price=sig.stop_loss_price,
            tp_price=sig.take_profit_price,
            vix=self.market_state.vix,
            market_regime=self.market_state.spy_trend_regime,
            spy_price=self.market_state.spy_price,
            journal_dir=self.config.journal_dir,
        )
        log.info(
            "Submitted: %s %s %d sh @ %.2f SL=%.2f TP=%.2f (tag=%s)",
            sig.side, sig.ticker, qty, sig.signal_price,
            sig.stop_loss_price, sig.take_profit_price, tag,
        )

    # ── Position management ───────────────────────────────────────

    async def _manage_positions(self, session: NyseSession) -> None:
        """Check each open position for strategy-driven exits."""
        exits = self.position_mgr.check_all(
            self.strategies, self.bar_cache, now_et(), session,
        )
        for intent in exits:
            await self._close_position(intent.ticker, intent.reason)

    async def _force_flat_all(self) -> None:
        """Force-close ALL open day-trade positions."""
        positions = self.position_mgr.all_for_force_close()
        if not positions:
            log.info("Force-flat: no open positions")
            return
        log.info("Force-flat: closing %d position(s)", len(positions))
        for pos in positions:
            await self._close_position(pos.ticker, "force_eod")

    async def _close_position(self, ticker: str, reason: str) -> None:
        """Close one day-trade position via the safe tagged helper."""
        pos = self.position_mgr.get(ticker)
        if pos is None:
            return

        if self.config.dry_run:
            log.info(
                "[DRY-RUN] Would close: %s %d sh (reason=%s)",
                ticker, pos.qty, reason,
            )
            self.position_mgr.close_position(ticker)
            return

        result = close_tagged_daytrade_qty(
            self.broker,
            ticker,
            qty=pos.qty,
            side=pos.side,
            parent_client_order_id=pos.parent_client_order_id,
        )
        if result.succeeded:
            self.position_mgr.close_position(ticker)
            # Estimate P&L from latest bar / signal price
            latest = self.bar_cache.latest(ticker)
            exit_price = latest.close if latest else pos.entry_price
            pnl = (exit_price - pos.entry_price) * pos.qty
            if pos.side == "short":
                pnl = -pnl

            self.risk.record_close(
                ticker=ticker,
                strategy=pos.strategy,
                pnl=pnl,
                entry_notional=pos.entry_price * pos.qty,
            )
            log.info(
                "Closed %s: reason=%s P&L=$%.2f", ticker, reason, pnl,
            )
        else:
            log.error(
                "Failed to close %s: %s", ticker, result.error,
            )

    # ── Regime snapshot ───────────────────────────────────────────

    async def _snapshot_regime(self) -> None:
        """Fetch SPY/VIX for the market-state snapshot."""
        try:
            spy_bar = self.bar_cache.latest("SPY")
            spy_price = spy_bar.close if spy_bar else 0.0
            # VIX would need a separate fetch — for v1, use 0 as
            # placeholder (the filter will not trip since
            # halt_above_vix defaults to 35, and 0 < 35)
            self.market_state = MarketState(
                spy_price=spy_price,
                vix=0.0,
                spy_trend_regime="BULL",
                captured_at=now_et().isoformat(timespec="seconds"),
            )
        except Exception:
            log.exception("Regime snapshot failed")

    # ── Post-session ──────────────────────────────────────────────

    async def _post_session(self) -> None:
        """Cleanup after session ends."""
        try:
            await self.feed.stop()
        except Exception:
            log.exception("Feed stop failed")

        log.info(
            "Session complete: trades=%d P&L=$%.2f kill_switch=%s "
            "filter_stats=%s",
            self.risk.trades_today,
            self.risk.daily_realized_pnl,
            self.risk.kill_switch_tripped,
            self.pipeline.stats,
        )
        self.pipeline.reset_stats()
