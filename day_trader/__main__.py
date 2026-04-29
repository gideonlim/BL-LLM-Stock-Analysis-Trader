"""Day-trader CLI entry point.

Usage::

    python -m day_trader live       # live Alpaca (paper or real per env)
    python -m day_trader paper      # explicit paper mode
    python -m day_trader dry-run    # full pipeline, no orders submitted
    python -m day_trader scan-only  # premarket scan + log watchlist, exit

The daemon runs one session per invocation. For continuous operation
the systemd unit restarts it each morning; it waits for the next
NYSE session, runs, then exits cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

log = logging.getLogger("day_trader")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="day_trader",
        description="Day-trader daemon — 25%% sub-portfolio intraday trading",
    )
    parser.add_argument(
        "mode",
        choices=["live", "paper", "dry-run", "scan-only"],
        help="Execution mode",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config override file",
    )
    return parser.parse_args()


async def _run_daemon(args: argparse.Namespace) -> int:
    """Build components + run one session."""
    from dotenv import load_dotenv

    # Load /etc/day-trader/env or local .env
    env_paths = [
        Path("/etc/day-trader/env"),
        Path("trading_bot_bl/.env"),
        Path(".env"),
    ]
    for p in env_paths:
        if p.exists():
            load_dotenv(p)
            log.info("Loaded env from %s", p)
            break

    from day_trader.calendar import next_session, now_et, session_for
    from day_trader.config import DayTradeConfig
    from day_trader.data.catalyst import CatalystClassifier
    from day_trader.data.feed import MarketDataFeed
    from day_trader.data.premarket import (
        AlpacaPremarketFetcher,
        PremarketScanner,
    )
    from day_trader.executor import DayTraderDaemon
    from day_trader.filters.base import FilterPipeline
    from day_trader.filters.catalyst_filter import CatalystFilter
    from day_trader.filters.cooldown import CooldownTracker
    from day_trader.filters.cooldown_filter import CooldownFilter
    from day_trader.filters.regime_filter import RegimeFilter
    from day_trader.filters.rvol_filter import RvolFilter
    from day_trader.filters.spread_filter import SpreadFilter
    from day_trader.filters.symbol_lock_filter import SymbolLockFilter
    from day_trader.filters.whole_share_sizing_filter import (
        WholeShareSizingFilter,
    )
    from day_trader.order_tags import SequenceCounter
    from day_trader.strategies.orb_vwap import OrbVwapStrategy
    from day_trader.strategies.vwap_pullback import VwapPullbackStrategy
    from day_trader.symbol_locks import SymbolLock
    from trading_bot_bl.broker import AlpacaBroker
    from trading_bot_bl.config import AlpacaConfig

    # Config
    config = DayTradeConfig.from_env()
    if args.mode == "paper":
        import os
        os.environ["ALPACA_PAPER"] = "true"
    if args.mode == "dry-run":
        config.dry_run = True

    log.info("Config: %r", config)

    # Broker
    alpaca_config = AlpacaConfig.from_env()
    broker = AlpacaBroker(alpaca_config)

    # Wait for session if needed
    session = session_for()
    if session is None:
        nxt = next_session()
        if nxt is None:
            log.info("No upcoming sessions in the next 14 days")
            return 0
        log.info(
            "No session today — next session is %s. "
            "systemd will restart tomorrow.",
            nxt.date,
        )
        return 0

    # Scan-only mode: just run premarket scan and exit
    if args.mode == "scan-only":
        from day_trader.data.universe import load_universe
        universe = load_universe()
        hd_client = getattr(broker, "_data_client", None)
        fetcher = AlpacaPremarketFetcher(hd_client) if hd_client else None
        if fetcher is None:
            log.error("No historical data client — scan-only needs Alpaca SIP")
            return 1
        scanner = PremarketScanner(fetcher)
        results = scanner.scan(universe, target_date=session.date)
        for ticker, ctx in sorted(
            results.items(),
            key=lambda kv: kv[1].premkt_rvol * abs(kv[1].premkt_gap_pct),
            reverse=True,
        ):
            log.info(
                "  %s: RVOL=%.1f gap=%.1f%% catalyst=%s",
                ticker, ctx.premkt_rvol, ctx.premkt_gap_pct,
                ctx.catalyst_label,
            )
        return 0

    # Build components
    seq = SequenceCounter(config.state_dir / "seq.json")
    symbol_lock = SymbolLock(broker)
    cooldowns = CooldownTracker(
        ticker_minutes=config.risk.ticker_cooldown_minutes,
        strategy_minutes=config.risk.strategy_cooldown_minutes,
    )

    # Initial equity for WholeShareSizingFilter — we'll refresh at
    # session start, but need a bootstrap value for construction.
    portfolio = broker.get_portfolio()

    pipeline = FilterPipeline([
        SymbolLockFilter(symbol_lock),
        RegimeFilter(config.risk),
        CooldownFilter(cooldowns),
        SpreadFilter(config.risk),
        RvolFilter(config.risk),
        WholeShareSizingFilter(
            config.risk,
            equity_at_session_start=portfolio.equity,
        ),
        CatalystFilter(),
    ])

    strategies = [OrbVwapStrategy(), VwapPullbackStrategy()]

    hd_client = getattr(broker, "_data_client", None)
    premarket_fetcher = (
        AlpacaPremarketFetcher(hd_client) if hd_client else None
    )
    premarket_scanner = PremarketScanner(
        premarket_fetcher,
        catalyst_classifier=CatalystClassifier(),
    ) if premarket_fetcher else PremarketScanner(
        # Stub for paper testing without SIP
        type("NullFetcher", (), {
            k: lambda self, *a, **kw: 0.0
            for k in [
                "fetch_premarket_volume",
                "fetch_premarket_dollar_volume",
            ]
        } | {
            k: lambda self, *a, **kw: None
            for k in [
                "fetch_first_premarket_price",
                "fetch_prev_close",
                "fetch_avg_premarket_volume",
            ]
        })(),
    )

    # Data feed is tied to the API account, NOT the paper/live mode.
    # Default "sip" assumes Algo Trader Plus or Unlimited subscription
    # (per the plan's locked decision). Override via DAYTRADE_DATA_FEED
    # env var to "iex" only if downgrading the subscription — note
    # IEX-only data renders the Stocks-in-Play RVOL gating useless
    # (~3-5% of consolidated volume), so most v1 strategies will fail
    # to find tradeable setups.
    feed = MarketDataFeed(
        alpaca_config.api_key,
        alpaca_config.api_secret,
        feed=config.data_feed,
    )

    daemon = DayTraderDaemon(
        broker=broker,
        config=config,
        strategies=strategies,
        pipeline=pipeline,
        feed=feed,
        premarket_scanner=premarket_scanner,
        catalyst_classifier=CatalystClassifier(),
        seq_counter=seq,
    )

    log.info("Starting session for %s", session.date)
    await daemon.run()
    log.info("Session complete")
    return 0


def main() -> None:
    args = _parse_args()
    _setup_logging(verbose=args.verbose)
    try:
        rc = asyncio.run(_run_daemon(args))
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
        rc = 130
    except Exception:
        log.exception("Fatal error")
        rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
