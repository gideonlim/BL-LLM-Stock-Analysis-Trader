"""Execution engine -- reads signals, builds orders, routes through risk."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from trading_bot_bl.broker import AlpacaBroker
from trading_bot_bl.config import TradingConfig
from trading_bot_bl.cppi import (
    CppiState,
    load_cppi_state,
    save_cppi_state,
    update_cppi,
)
from trading_bot_bl.history import (
    TradeHistory,
    enrich_history_with_pnl,
    load_trade_history,
    reconcile_with_journal,
)
from trading_bot_bl.models import (
    OrderIntent,
    OrderResult,
    PortfolioSnapshot,
    Signal,
)
from trading_bot_bl.market_sentiment import (
    MarketSentiment,
    fetch_market_sentiment,
)
from trading_bot_bl.monitor import (
    MonitorReport,
    monitor_positions,
    write_monitor_log,
)
from trading_bot_bl.oil_spike import (
    OilSpikeState,
    OilSpikeTier,
    detect_oil_spike,
    get_boost_for_ticker,
)
from trading_bot_bl.portfolio_optimizer import optimize_intents
from trading_bot_bl.risk import RiskManager

# ── Journal imports (non-critical) ────────────────────────────────
try:
    from trading_bot_bl import journal as _journal
    from trading_bot_bl import equity_curve as _equity
    _JOURNAL_AVAILABLE = True
except ImportError:
    _JOURNAL_AVAILABLE = False

# ── Strategy monitor imports (non-critical) ──────────────────────
try:
    from trading_bot_bl.strategy_monitor import StrategyMonitor
    _STRATEGY_MONITOR_AVAILABLE = True
except ImportError:
    _STRATEGY_MONITOR_AVAILABLE = False

log = logging.getLogger(__name__)


def load_latest_signals(signals_dir: Path) -> list[Signal]:
    """
    Find the most recent signals JSON file and parse it.

    Looks for files matching signals_YYYY-MM-DD.json
    and picks the most recent by filename date.
    """
    json_files = sorted(
        signals_dir.glob("signals_*.json"), reverse=True
    )

    if not json_files:
        log.warning(f"No signal files found in {signals_dir}")
        return []

    latest = json_files[0]
    log.info(f"Loading signals from {latest}")

    with open(latest, encoding="utf-8") as f:
        data = json.load(f)

    raw_signals = data.get("signals", [])
    signals: list[Signal] = []

    for s in raw_signals:
        try:
            signals.append(
                Signal(
                    ticker=s["ticker"],
                    signal=s["signal"],
                    signal_raw=s["signal_raw"],
                    strategy=s["strategy"],
                    confidence=s["confidence"],
                    confidence_score=s["confidence_score"],
                    composite_score=s["composite_score"],
                    current_price=s["current_price"],
                    stop_loss_price=s["stop_loss_price"],
                    take_profit_price=s["take_profit_price"],
                    suggested_position_size_pct=s[
                        "suggested_position_size_pct"
                    ],
                    signal_expires=s.get("signal_expires", ""),
                    sharpe=s.get("sharpe", 0),
                    win_rate=s.get("win_rate", 0),
                    total_trades=s.get("total_trades", 0),
                    generated_at=s.get("generated_at", ""),
                    notes=s.get("notes", ""),
                    sortino=s.get("sortino", 0.0),
                    profit_factor=s.get("profit_factor", 0.0),
                    annual_return_pct=s.get(
                        "annual_return_pct", 0.0
                    ),
                    annual_excess_pct=s.get(
                        "annual_excess_pct", 0.0
                    ),
                    max_drawdown_pct=s.get(
                        "max_drawdown_pct", 0.0
                    ),
                    vol_20=s.get("vol_20", 0.0),
                    pbo=s.get("pbo", -1.0),
                    meta_label_prob=s.get("meta_label_prob", -1.0),
                )
            )
        except (KeyError, TypeError) as e:
            log.warning(f"Skipping malformed signal: {e}")

    log.info(
        f"Loaded {len(signals)} signals "
        f"(generated: {data.get('generated_at', 'unknown')})"
    )
    return signals


def build_order_intents(
    signals: list[Signal],
    portfolio: PortfolioSnapshot,
    config: TradingConfig,
    pending_tickers: set[str] | None = None,
) -> list[OrderIntent]:
    """
    Convert signals into order intents.

    For BUY signals: calculate notional from suggested position
    size and account equity.  Skips tickers we already hold or
    have pending orders for.
    For EXIT signals: close the existing position.

    Args:
        signals: Parsed signals from the analysis bot.
        portfolio: Current broker portfolio snapshot.
        config: Trading configuration.
        pending_tickers: Set of tickers with open/pending
            orders (fetched from broker before calling this).
    """
    equity = portfolio.equity
    intents: list[OrderIntent] = []
    seen_tickers: set[str] = set()  # dedup within this batch
    if pending_tickers is None:
        pending_tickers = set()

    for sig in signals:
        # Skip non-actionable signals
        if sig.signal_raw == 0:
            continue

        if sig.signal == "ERROR":
            continue

        if sig.signal_raw == 1:  # BUY
            # ── Duplicate checks ────────────────────────────────
            # 1) Already hold this stock
            if sig.ticker in portfolio.positions:
                log.info(
                    f"  {sig.ticker}: already held — "
                    f"skipping BUY signal"
                )
                continue

            # 2) Pending/open order for this stock
            if sig.ticker in pending_tickers:
                log.info(
                    f"  {sig.ticker}: pending order exists — "
                    f"skipping BUY signal"
                )
                continue

            # 3) Already generated an intent for this ticker
            #    in this batch (signals may contain duplicates)
            if sig.ticker in seen_tickers:
                log.debug(
                    f"  {sig.ticker}: duplicate signal in "
                    f"this batch — skipping"
                )
                continue
            seen_tickers.add(sig.ticker)

            # Use the analysis bot's suggested size, as a % of equity
            notional = equity * sig.suggested_position_size_pct / 100

            if notional < 1.0:
                log.debug(
                    f"  {sig.ticker}: suggested size too small "
                    f"(${notional:.2f}), skipping"
                )
                continue

            intents.append(
                OrderIntent(
                    ticker=sig.ticker,
                    side="buy",
                    notional=round(notional, 2),
                    stop_loss_price=sig.stop_loss_price,
                    take_profit_price=sig.take_profit_price,
                    signal=sig,
                    reason=(
                        f"BUY signal from {sig.strategy} "
                        f"(score={sig.composite_score}, "
                        f"confidence={sig.confidence})"
                    ),
                )
            )

        elif sig.signal_raw == -1:  # EXIT or SELL/SHORT
            # If we hold this stock, close the position
            if sig.ticker in portfolio.positions:
                pos = portfolio.positions[sig.ticker]
                intents.append(
                    OrderIntent(
                        ticker=sig.ticker,
                        side="sell",
                        notional=abs(pos["market_value"]),
                        stop_loss_price=0.0,
                        take_profit_price=0.0,
                        signal=sig,
                        reason=(
                            f"EXIT signal from {sig.strategy} "
                            f"— closing position"
                        ),
                    )
                )
            else:
                log.debug(
                    f"  {sig.ticker}: EXIT signal but "
                    f"no position held, skipping"
                )

    return intents


def execute(
    config: TradingConfig,
    log_dir: Path | None = None,
) -> list[OrderResult]:
    """
    Main execution pipeline:
     1. Load latest signals
     2. Load trade history (memory of past runs)
     3. Connect to Alpaca
     4. Check market hours
     5. Get portfolio snapshot + pending orders
     6. Monitor existing positions (orphaned brackets,
        emergency losses, stale SL/TP, gap detection)
     7. Enrich history with live P&L
     8. Build order intents (with duplicate detection)
     9. Risk check each intent (with strategy eval)
    10. Submit approved orders

    Returns list of OrderResult for every signal processed.
    """
    results: list[OrderResult] = []

    # ── 1. Load signals ───────────────────────────────────────────
    signals = load_latest_signals(config.signals_dir)
    if not signals:
        log.warning("No signals to process. Exiting.")
        return results

    actionable = [s for s in signals if s.signal_raw != 0]
    log.info(
        f"  {len(actionable)} actionable signals "
        f"out of {len(signals)} total"
    )

    if not actionable:
        log.info("No actionable signals today. Nothing to do.")
        return results

    # ── 2. Load trade history ──────────────────────────────────────
    history_dir = log_dir or Path("execution_logs")
    history = load_trade_history(
        history_dir,
        lookback_days=config.history_lookback_days,
        include_dry_runs=False,
    )

    # Cross-reference with journal to clear churn cooldowns for
    # orders that were submitted but never filled (expired/cancelled).
    journal_dir = history_dir / "journal"
    reconcile_with_journal(history, journal_dir)

    # ── 3. Connect to broker ──────────────────────────────────────
    broker = AlpacaBroker(config.alpaca)

    # ── 4. Check market hours ─────────────────────────────────────
    market_open = broker.is_market_open()
    if not market_open:
        log.warning(
            "Market is closed. Orders will queue for next open."
        )

    # ── 5. Get portfolio snapshot + pending orders ─────────────────
    portfolio = broker.get_portfolio()
    pending_tickers = broker.get_pending_tickers()
    log.info(
        f"  Account equity: ${portfolio.equity:,.2f}  "
        f"Cash: ${portfolio.cash:,.2f}  "
        f"Positions: {len(portfolio.positions)}  "
        f"Pending: {len(pending_tickers)}  "
        f"Day P&L: {portfolio.day_pnl_pct:+.2f}%"
    )

    # ── 5b. Clean up stale orders (market-open only) ──────────────
    #    Cancel open orders for tickers we don't hold a position in.
    #    This prevents leftover orders from previous runs from
    #    interfering with today's fresh signals.
    if market_open and pending_tickers:
        stale_cleanup = broker.cancel_orphaned_orders(
            portfolio.positions,
        )
        if stale_cleanup:
            # Refresh pending tickers after cancellations
            pending_tickers = broker.get_pending_tickers()

    # ── 5c. Fetch market-wide sentiment ───────────────────────
    #    VIX + put/call → regime → position-size multiplier.
    #    Disabled via MARKET_SENTIMENT_ENABLED=false.
    sentiment = MarketSentiment()  # neutral default
    if config.market_sentiment_enabled:
        sentiment = fetch_market_sentiment(
            fear_vix=config.sentiment_fear_vix,
            greed_vix=config.sentiment_greed_vix,
            fear_pc=config.sentiment_fear_pc,
            greed_pc=config.sentiment_greed_pc,
            fear_size_mult=config.sentiment_fear_size_mult,
            greed_size_mult=config.sentiment_greed_size_mult,
            spy_bear_confirmation_days=(
                config.spy_bear_confirmation_days
            ),
            spy_severe_drawdown_pct=(
                config.spy_severe_drawdown_pct
            ),
        )
    else:
        log.info("  Market sentiment: DISABLED")

    # ── 5d. Journal: equity snapshot + lifecycle checks ────────────
    #    Non-critical — all wrapped in try/except so failures
    #    never block the trading pipeline.
    #    Skipped entirely in dry-run mode so simulated runs
    #    don't pollute real journal/equity data.
    journal_dir = history_dir / "journal"
    if _JOURNAL_AVAILABLE and not config.dry_run:
        try:
            _equity.record_snapshot(portfolio, history_dir)
        except Exception as exc:
            log.debug(f"Equity snapshot failed: {exc}")
        try:
            _journal.resolve_pending_trades(broker, journal_dir)
        except Exception as exc:
            log.debug(f"Journal resolve failed: {exc}")
        try:
            _journal.detect_closed_trades(
                portfolio.positions, journal_dir, broker,
                max_hold_days=config.risk.max_hold_days,
            )
        except Exception as exc:
            log.debug(f"Journal detect_closed failed: {exc}")
        try:
            _journal.migrate_existing_positions(
                portfolio.positions,
                broker.get_open_orders(),
                journal_dir,
                vix=sentiment.vix,
                market_regime=sentiment.regime,
            )
        except Exception as exc:
            log.debug(f"Journal migration failed: {exc}")

    # ── 6. Monitor existing positions ─────────────────────────────
    #    Run BEFORE new orders: check for orphaned brackets,
    #    emergency losses, stale SL/TP, and gapped prices.
    if portfolio.positions:
        log.info(
            f"  Monitoring {len(portfolio.positions)} "
            f"existing positions..."
        )
        monitor_report = monitor_positions(
            broker=broker,
            portfolio=portfolio,
            limits=config.risk,
            dry_run=config.dry_run,
            journal_dir=(
                journal_dir
                if _JOURNAL_AVAILABLE and not config.dry_run
                else None
            ),
        )
        # Write monitor log alongside execution logs
        write_monitor_log(monitor_report, history_dir)

        # If emergency actions were taken, refresh portfolio
        # so order intents use the updated state
        if monitor_report.actions:
            log.info(
                "  Refreshing portfolio after monitor actions..."
            )
            portfolio = broker.get_portfolio()
            pending_tickers = broker.get_pending_tickers()

        # If circuit-breaker-level damage found, stop trading
        if monitor_report.emergency_count >= 3:
            log.warning(
                f"  {monitor_report.emergency_count} emergency "
                f"alerts — halting new order placement"
            )
            return results

    # ── 7. Enrich history with live P&L ────────────────────────────
    enrich_history_with_pnl(history, portfolio.positions)

    # Log strategy performance summary
    if history.by_strategy:
        log.info("  Strategy performance (last "
                 f"{config.history_lookback_days}d):")
        for name, sr in sorted(
            history.by_strategy.items(),
            key=lambda x: -x[1].submitted,
        )[:10]:
            pnl_str = (
                f"  P&L=${sr.realized_pnl:+,.0f}"
                if sr.realized_pnl != 0 else ""
            )
            infra_skips = (
                sr.rejected_by_broker + sr.skipped_by_risk
            )
            log.info(
                f"    {name:<25} "
                f"filled={sr.submitted:>3} "
                f"quality_skip={sr.skipped_by_quality:>3} "
                f"infra_skip={infra_skips:>3} "
                f"rate={sr.success_rate:.0%}"
                f"{pnl_str}"
            )

    # ── 7b. FinBERT news sentiment ──────────────────────────────────
    #    Score recent headlines for BUY candidates and adjust their
    #    composite scores BEFORE building order intents, so the
    #    optimizer ranks sentiment-adjusted signals correctly.
    #    Feature-flagged via FINBERT_ENABLED (default: false).
    if config.finbert_enabled:
        try:
            from trading_bot_bl.news_sentiment import (
                aggregate_ticker_sentiment,
                adjust_composite_scores,
                is_available as finbert_available,
            )
            from trading_bot_bl.news_fetcher import (
                fetch_news_batch,
            )

            if finbert_available():
                buy_tickers_for_sent = [
                    s.ticker
                    for s in actionable
                    if s.signal_raw == 1
                ]
                if buy_tickers_for_sent:
                    news_map = fetch_news_batch(
                        buy_tickers_for_sent,
                        max_headlines=config.finbert_max_headlines,
                    )
                    ticker_sent = aggregate_ticker_sentiment(
                        news_map
                    )
                    if ticker_sent:
                        buy_signals = [
                            s
                            for s in actionable
                            if s.signal_raw == 1
                        ]
                        adjustments = adjust_composite_scores(
                            buy_signals,
                            ticker_sent,
                            weight=config.finbert_score_weight,
                        )
                        for tkr, delta in adjustments:
                            log.info(
                                f"  FinBERT: {tkr} "
                                f"composite {delta:+.1f}"
                            )
                        log.info(
                            f"  FinBERT: scored "
                            f"{len(ticker_sent)} tickers, "
                            f"adjusted {len(adjustments)}"
                        )
            else:
                log.info(
                    "  FinBERT enabled but dependencies missing "
                    "(install transformers + torch)"
                )
        except Exception as exc:
            log.warning(f"  FinBERT scoring failed: {exc}")

    # ── 7c. Strategy health monitor (observation only) ────────────
    _ACTION_LABEL = {"would_reduce": "reduced", "would_block": "blocked"}
    if _STRATEGY_MONITOR_AVAILABLE and _JOURNAL_AVAILABLE:
        try:
            sm = StrategyMonitor(history_dir)
            closed_trades = [
                t for t in _journal.load_all_trades(
                    history_dir / "journal"
                )
                if t.status == "closed"
            ]
            buy_signals = [
                s for s in actionable if s.signal_raw == 1
            ]
            sm_result = sm.evaluate(closed_trades, buy_signals)
            for v in sm_result.verdicts:
                label = _ACTION_LABEL.get(v.action)
                if label:
                    log.info(
                        f"  OBSERVATION: would have {label} "
                        f"{v.ticker} via {v.strategy}, "
                        f"reason: {v.reason}"
                    )
        except Exception as exc:
            log.warning(f"  Strategy monitor failed: {exc}")

    # ── 8. Build order intents ────────────────────────────────────
    intents = build_order_intents(
        signals, portfolio, config,
        pending_tickers=pending_tickers,
    )
    log.info(f"  {len(intents)} order intents generated")

    # ── 8b. Portfolio-optimize intent ordering ─────────────────────
    #    Use Black-Litterman (if enabled) or marginal Sharpe
    #    to rank BUY intents by portfolio-level benefit.
    buy_intents = [i for i in intents if i.side == "buy"]
    sell_intents = [i for i in intents if i.side != "buy"]

    if buy_intents:
        ranked = optimize_intents(
            buy_intents,
            held_positions=portfolio.positions,
            portfolio_equity=portfolio.equity,
            config=config,
        )
        # Reorder: sells first, then buys by optimization rank
        intents = sell_intents + [r.intent for r in ranked]
    else:
        log.info("  No BUY intents to optimize")

    # ── 8b. Oil spike detection (disabled by default) ──────────────
    oil_state = OilSpikeState()  # inactive unless enabled
    oil_tickers: tuple[str, ...] = ()
    oil_spike_tiers: list[OilSpikeTier] = []
    if config.oil_spike_enabled:
        # Tier 1 — fertilizer/ag (immediate boost, 20-day decay)
        tier1_tickers = tuple(
            t.strip().upper()
            for t in config.oil_spike_tickers.split(",")
            if t.strip()
        )
        oil_tickers = tier1_tickers  # backward-compat field
        oil_spike_tiers.append(
            OilSpikeTier(
                tickers=tier1_tickers,
                peak_boost=config.oil_spike_boost,
                delay_days=0,
                decay_days=config.oil_spike_window_days,
            )
        )
        # Tier 2 — airlines (delayed entry, shorter decay)
        tier2_tickers = tuple(
            t.strip().upper()
            for t in config.oil_spike_airline_tickers.split(",")
            if t.strip()
        )
        if tier2_tickers:
            oil_spike_tiers.append(
                OilSpikeTier(
                    tickers=tier2_tickers,
                    peak_boost=config.oil_spike_airline_boost,
                    delay_days=config.oil_spike_airline_delay_days,
                    decay_days=config.oil_spike_airline_decay_days,
                )
            )

        oil_state = detect_oil_spike(
            peak_boost=config.oil_spike_boost,
            window_days=max(
                config.oil_spike_window_days,
                config.oil_spike_airline_delay_days
                + config.oil_spike_airline_decay_days,
            ),
            spike_threshold=config.oil_spike_threshold,
        )
        if oil_state.active:
            all_tickers = tier1_tickers + tier2_tickers
            log.info(
                f"  Oil spike active: USO "
                f"+{oil_state.spike_magnitude:.1%} "
                f"({oil_state.days_since_spike}d ago), "
                f"boost={oil_state.boost:+.1f} for "
                f"{', '.join(all_tickers)}"
            )

    # ── 8c. Oil spike ranking promotion ─────────────────────────────
    #    If an oil spike is active, promote eligible BUY intents
    #    upward in the ranking.  BL/marginal-Sharpe doesn't know about
    #    short-term event signals, so this is a post-optimization
    #    overlay that ensures spike tickers get executed before the
    #    position cap is exhausted.
    if oil_state.active and oil_spike_tiers:
        # Partition intents into sells (keep first) and buys
        _sells = [i for i in intents if i.side != "buy"]
        _buys = [i for i in intents if i.side == "buy"]
        if _buys:
            # Score each buy: boosted tickers get their tier boost
            # added so they sort ahead of non-boosted peers.
            _boosted: list[tuple[float, int, object]] = []
            for idx, intent in enumerate(_buys):
                boost = get_boost_for_ticker(
                    oil_state,
                    intent.ticker,
                    oil_tickers,
                    tiers=oil_spike_tiers,
                )
                # Negate for descending sort (higher boost = earlier)
                # Use original index as tiebreaker to preserve BL rank
                _boosted.append((-boost, idx, intent))
            _boosted.sort()
            reordered_buys = [b[2] for b in _boosted]

            # Log any promotions
            promoted = [
                (b[2].ticker, -b[0])
                for b in _boosted
                if b[0] < 0  # negative means boost > 0
            ]
            if promoted:
                for tkr, bst in promoted:
                    old_rank = next(
                        i + 1
                        for i, x in enumerate(_buys)
                        if x.ticker == tkr
                    )
                    new_rank = next(
                        i + 1
                        for i, x in enumerate(reordered_buys)
                        if x.ticker == tkr
                    )
                    if new_rank < old_rank:
                        log.info(
                            f"  Oil spike: {tkr} promoted "
                            f"rank {old_rank} → {new_rank} "
                            f"(boost={bst:+.1f})"
                        )
                intents = _sells + reordered_buys

    # ── 8d. CPPI drawdown control ───────────────────────────────────
    #    Load or initialize CPPI state, update with current equity,
    #    and pass to the risk manager for notional scaling.
    cppi_state = CppiState()  # neutral (multiplier=1.0) when disabled
    if config.risk.cppi_enabled:
        cppi_path = history_dir / "cppi_state.json"
        saved = load_cppi_state(cppi_path)
        if saved is not None:
            # Overlay current config so env/config changes take
            # effect immediately (peak/floor/cushion are preserved).
            cppi_state = saved
            cppi_state.max_drawdown_pct = config.risk.cppi_max_drawdown_pct
            cppi_state.multiplier = config.risk.cppi_multiplier
            cppi_state.min_exposure_pct = config.risk.cppi_min_exposure_pct
        else:
            cppi_state = CppiState.from_portfolio(
                equity=portfolio.equity,
                max_drawdown_pct=config.risk.cppi_max_drawdown_pct,
                multiplier=config.risk.cppi_multiplier,
                min_exposure_pct=config.risk.cppi_min_exposure_pct,
            )

        # Determine if we have *actual* SPY regime data.
        # spy_price > 0 proves the fetch succeeded; the default
        # SpyRegime() has spy_price=0.  Without real data (regime
        # disabled, sentiment disabled, or fetch failure) we pass
        # UNKNOWN so the CPPI floor does NOT auto-reset — the
        # portfolio must recover above the floor organically.
        _has_regime_data = (
            config.spy_regime_enabled
            and sentiment.spy_regime.spy_price > 0
        )
        cppi_state = update_cppi(
            cppi_state,
            current_equity=portfolio.equity,
            spy_trend_regime=(
                sentiment.spy_regime.trend_regime
                if _has_regime_data
                else "UNKNOWN"
            ),
        )
        save_cppi_state(cppi_state, cppi_path)

        log.info(
            f"  CPPI: floor=${cppi_state.floor:,.0f}, "
            f"peak=${cppi_state.peak_equity:,.0f}, "
            f"cushion={cppi_state.cushion_pct:.1f}%, "
            f"exposure={cppi_state.exposure_multiplier:.0%}"
        )

    # ── 9. Risk check each intent ─────────────────────────────────
    risk_mgr = RiskManager(
        limits=config.risk,
        history=history,
        sentiment_size_multiplier=sentiment.size_multiplier,
        spy_trend_regime=sentiment.spy_regime.trend_regime,
        oil_spike_state=oil_state,
        oil_spike_tickers=oil_tickers,
        oil_spike_tiers=oil_spike_tiers,
        cppi_state=cppi_state,
    )
    # Apply regime-adjusted limits (CAUTION/BEAR/SEVERE_BEAR)
    if config.spy_regime_enabled:
        risk_mgr.apply_spy_regime_overrides(
            bear_max_positions=config.spy_bear_max_positions,
            bear_min_composite=config.spy_bear_min_composite_score,
            caution_max_positions=config.spy_caution_max_positions,
            caution_min_composite=(
                config.spy_caution_min_composite_score
            ),
        )

    for intent in intents:
        verdict = risk_mgr.evaluate_order(intent, portfolio)

        strategy_name = intent.signal.strategy

        if not verdict.approved:
            log.info(
                f"  SKIP {intent.ticker}: {verdict.reason}"
            )
            results.append(
                OrderResult(
                    ticker=intent.ticker,
                    status="skipped",
                    side=intent.side,
                    notional=intent.notional,
                    strategy=strategy_name,
                    error=verdict.reason,
                )
            )
            continue

        approved_order = verdict.order
        assert approved_order is not None

        # ── 10. Submit order ──────────────────────────────────────
        if config.dry_run:
            bracket_info = (
                f" (SL=${approved_order.stop_loss_price}, "
                f"TP=${approved_order.take_profit_price})"
                if approved_order.side == "buy"
                else ""
            )
            log.info(
                f"  DRY RUN: would {approved_order.side.upper()} "
                f"${approved_order.notional:,.2f} of "
                f"{approved_order.ticker}{bracket_info}"
            )
            results.append(
                OrderResult(
                    ticker=approved_order.ticker,
                    status="dry_run",
                    side=approved_order.side,
                    notional=approved_order.notional,
                    stop_loss_price=approved_order.stop_loss_price,
                    take_profit_price=(
                        approved_order.take_profit_price
                    ),
                    strategy=strategy_name,
                )
            )
            # Update portfolio snapshot even in dry-run so that
            # subsequent risk checks (max positions, exposure cap)
            # account for orders already "placed" in this run.
            if approved_order.side == "buy":
                portfolio.cash -= approved_order.notional
                portfolio.market_value += approved_order.notional
                portfolio.positions[approved_order.ticker] = {
                    "qty": 0,
                    "market_value": approved_order.notional,
                    "avg_entry": 0,
                    "unrealized_pnl": 0,
                    "side": "long",
                }
            elif approved_order.side == "sell":
                sold_pos = portfolio.positions.pop(
                    approved_order.ticker, None
                )
                if sold_pos:
                    freed = float(
                        sold_pos.get("market_value", 0)
                    )
                    portfolio.cash += freed
                    portfolio.market_value -= freed
            continue

        # EXIT signals: close position directly (no bracket)
        if intent.side == "sell" and intent.signal.signal_raw == -1:
            result = broker.close_position(approved_order.ticker)
        else:
            # BUY signals: bracket order with SL + TP
            # Fetch live price so qty calc and SL/TP are based
            # on actual market price, not yesterday's close.
            signal_price = intent.signal.current_price
            live_price = broker.get_latest_price(
                approved_order.ticker
            )

            if live_price and live_price > 0:
                price_for_order = live_price
                drift_pct = (
                    (live_price - signal_price)
                    / signal_price * 100
                )

                # Reject if live price has drifted too far
                # from signal price. Signals are generated
                # pre-market; a large gap-up means the entry
                # thesis may no longer hold and we'd be
                # chasing.
                max_drift = config.max_signal_drift_pct
                if max_drift > 0 and abs(drift_pct) > max_drift:
                    log.warning(
                        f"  {approved_order.ticker}: SKIPPED — "
                        f"live ${live_price:.2f} vs signal "
                        f"${signal_price:.2f} "
                        f"({drift_pct:+.1f}% drift exceeds "
                        f"max {max_drift}%)"
                    )
                    result = OrderResult(
                        ticker=approved_order.ticker,
                        status="rejected",
                        order_id="",
                        side=intent.side,
                        notional=0.0,
                        error=(
                            f"Signal drift {drift_pct:+.1f}% "
                            f"exceeds max {max_drift}%"
                        ),
                    )
                    result.strategy = strategy_name
                    results.append(result)
                    continue

                # Recalculate SL/TP relative to live price
                # so the dollar distance stays proportional
                if abs(drift_pct) > 0.1 and signal_price > 0:
                    sl_pct = abs(
                        signal_price
                        - approved_order.stop_loss_price
                    ) / signal_price
                    tp_pct = abs(
                        approved_order.take_profit_price
                        - signal_price
                    ) / signal_price

                    new_sl = round(
                        live_price * (1 - sl_pct), 2
                    )
                    new_tp = round(
                        live_price * (1 + tp_pct), 2
                    )

                    log.info(
                        f"  {approved_order.ticker}: live price "
                        f"${live_price:.2f} vs signal "
                        f"${signal_price:.2f} "
                        f"({drift_pct:+.1f}%) — "
                        f"SL ${approved_order.stop_loss_price}"
                        f" -> ${new_sl}, "
                        f"TP ${approved_order.take_profit_price}"
                        f" -> ${new_tp}"
                    )
                    approved_order = OrderIntent(
                        ticker=approved_order.ticker,
                        side=approved_order.side,
                        notional=approved_order.notional,
                        stop_loss_price=new_sl,
                        take_profit_price=new_tp,
                        signal=approved_order.signal,
                        reason=approved_order.reason,
                    )
            else:
                price_for_order = signal_price
                log.warning(
                    f"  {approved_order.ticker}: could not fetch "
                    f"live price — using signal price "
                    f"${signal_price:.2f}"
                )

            result = broker.submit_bracket_order(
                ticker=approved_order.ticker,
                side=approved_order.side,
                notional=approved_order.notional,
                stop_loss_price=approved_order.stop_loss_price,
                take_profit_price=(
                    approved_order.take_profit_price
                ),
                current_price=price_for_order,
                time_in_force=config.time_in_force,
                max_entry_slippage_pct=(
                    config.max_entry_slippage_pct
                ),
            )

        # Stamp the strategy onto broker results
        result.strategy = strategy_name
        results.append(result)

        # ── Journal: record new trade ──────────────────────────
        if (
            _JOURNAL_AVAILABLE
            and result.status == "submitted"
            and intent.side == "buy"
            and result.order_id
        ):
            try:
                _journal.create_trade(
                    order_id=result.order_id,
                    ticker=result.ticker,
                    strategy=strategy_name,
                    side=intent.side,
                    signal_price=intent.signal.current_price,
                    notional=result.notional,
                    sl_price=approved_order.stop_loss_price,
                    tp_price=approved_order.take_profit_price,
                    composite_score=intent.signal.composite_score,
                    confidence=intent.signal.confidence,
                    confidence_score=intent.signal.confidence_score,
                    vix=sentiment.vix,
                    market_regime=sentiment.regime,
                    journal_dir=journal_dir,
                )
            except Exception as exc:
                log.debug(
                    f"Journal: create_trade failed: {exc}"
                )

        # Update portfolio snapshot to reflect the new order
        # (approximate, so subsequent risk checks account for it)
        if result.status == "submitted" and intent.side == "buy":
            portfolio.cash -= approved_order.notional
            portfolio.market_value += approved_order.notional
            portfolio.positions[approved_order.ticker] = {
                "qty": 0,
                "market_value": approved_order.notional,
                "avg_entry": 0,
                "unrealized_pnl": 0,
                "side": "long",
            }

    return results


def write_execution_log(
    results: list[OrderResult],
    log_dir: Path,
) -> Path:
    """Write execution results to a JSON log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = log_dir / f"execution_{date_str}.json"

    data = {
        "executed_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "total_orders": len(results),
        "submitted": sum(
            1 for r in results if r.status == "submitted"
        ),
        "skipped": sum(
            1 for r in results if r.status == "skipped"
        ),
        "rejected": sum(
            1 for r in results if r.status == "rejected"
        ),
        "orders": [
            {
                "ticker": r.ticker,
                "status": r.status,
                "side": r.side,
                "notional": r.notional,
                "stop_loss_price": r.stop_loss_price,
                "take_profit_price": r.take_profit_price,
                "strategy": r.strategy,
                "order_id": r.order_id,
                "error": r.error,
                "timestamp": r.timestamp,
            }
            for r in results
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    log.info(f"Execution log written to {path}")
    return path
