"""CLI entry point -- designed for cron / GitHub Actions."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from trading_bot.config import TradingConfig
from trading_bot.executor import execute, write_execution_log

log = logging.getLogger(__name__)


def _archive_logs(log_dir: Path) -> None:
    """Move existing execution logs to an archive subdirectory."""
    import shutil

    if not log_dir.exists():
        log.info("No execution logs to archive.")
        return

    log_files = list(log_dir.glob("execution_*.json"))
    monitor_files = list(log_dir.glob("monitor_*.json"))
    all_files = log_files + monitor_files

    if not all_files:
        log.info("No execution logs to archive.")
        return

    archive_dir = log_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    for f in all_files:
        dest = archive_dir / f.name
        shutil.move(str(f), str(dest))

    log.info(
        f"Archived {len(all_files)} log files to "
        f"{archive_dir}/"
    )
    log.info(
        "Strategy history has been reset. "
        "The bot will start fresh."
    )


def _run_monitor_only(
    config: TradingConfig, log_dir: Path
) -> None:
    """Run the position monitor without placing new orders."""
    from trading_bot.broker import AlpacaBroker
    from trading_bot.monitor import (
        monitor_positions,
        write_monitor_log,
    )

    broker = AlpacaBroker(config.alpaca)
    portfolio = broker.get_portfolio()

    log.info(
        f"  Account equity: ${portfolio.equity:,.2f}  "
        f"Cash: ${portfolio.cash:,.2f}  "
        f"Positions: {len(portfolio.positions)}  "
        f"Day P&L: {portfolio.day_pnl_pct:+.2f}%"
    )

    # Clean up stale orders if market is open
    if broker.is_market_open():
        stale = broker.cancel_orphaned_orders(
            portfolio.positions,
        )
        if stale:
            log.info(
                f"  Cleaned up {len(stale)} stale orders"
            )

    if not portfolio.positions:
        log.info("  No open positions to monitor.")
        return

    report = monitor_positions(
        broker=broker,
        portfolio=portfolio,
        limits=config.risk,
        dry_run=config.dry_run,
    )
    write_monitor_log(report, log_dir)

    log.info(f"\n{'=' * 60}")
    log.info(f"  MONITOR SUMMARY")
    log.info(f"{'=' * 60}")
    log.info(f"  {report.summary()}")

    if report.alerts:
        for a in report.alerts:
            icon = (
                "!!!" if a.severity == "critical"
                else " ! " if a.severity == "warning"
                else "   "
            )
            log.info(
                f"  {icon} [{a.alert_type}] {a.ticker}: "
                f"{a.message}"
            )
            if a.action_taken:
                log.info(f"       -> {a.action_taken}")
    else:
        log.info("  All positions healthy.")

    log.info(f"{'=' * 60}\n")

    if report.has_critical:
        sys.exit(1)


def main() -> None:
    """Entry point for the trading bot."""
    # Load .env file before anything reads environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; rely on real env vars

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Trading Bot -- execute signals via Alpaca"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--signals-dir",
        type=str,
        help="Override signals directory path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log orders without submitting them",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading (default: paper)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="execution_logs",
        help="Directory for execution logs (default: execution_logs)",
    )
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help=(
            "Only check existing positions for health "
            "(orphaned brackets, emergency losses, stale SL/TP). "
            "No new orders will be placed."
        ),
    )
    parser.add_argument(
        "--reset-history",
        action="store_true",
        help=(
            "Archive old execution logs so the bot starts "
            "with a clean strategy history. Existing logs "
            "are moved to execution_logs/archive/."
        ),
    )
    args = parser.parse_args()

    # ── Build config ──────────────────────────────────────────────
    if args.config:
        config = TradingConfig.from_file(Path(args.config))
    else:
        config = TradingConfig.from_env()

    if args.signals_dir:
        config.signals_dir = Path(args.signals_dir)
    if args.dry_run:
        config.dry_run = True
    if args.live:
        config.alpaca.paper = False

    # ── Validate ──────────────────────────────────────────────────
    mode = "PAPER" if config.alpaca.paper else "*** LIVE ***"
    dry = " (DRY RUN)" if config.dry_run else ""
    monitor = " [MONITOR ONLY]" if args.monitor_only else ""
    log.info(f"{'=' * 60}")
    log.info(f"  Trading Bot -- {mode}{dry}{monitor}")
    log.info(f"  Signals dir: {config.signals_dir}")
    log.info(
        f"  Risk limits: "
        f"exposure={config.risk.max_portfolio_exposure_pct}%, "
        f"per-stock={config.risk.max_position_pct}%, "
        f"daily-loss={config.risk.daily_loss_limit_pct}%"
    )
    log.info(f"{'=' * 60}")

    if not config.dry_run:
        try:
            config.alpaca.validate()
        except ValueError as e:
            log.error(str(e))
            sys.exit(1)

    execution_log_dir = Path(args.log_dir)

    # ── Reset history ─────────────────────────────────────────────
    if args.reset_history:
        _archive_logs(execution_log_dir)

    # ── Monitor-only mode ─────────────────────────────────────────
    if args.monitor_only:
        _run_monitor_only(config, execution_log_dir)
        return

    # ── Execute ───────────────────────────────────────────────────
    results = execute(config, log_dir=execution_log_dir)

    # ── Log results ───────────────────────────────────────────────
    log_path = write_execution_log(
        results, execution_log_dir
    )

    # ── Summary ───────────────────────────────────────────────────
    submitted = [r for r in results if r.status == "submitted"]
    skipped = [r for r in results if r.status == "skipped"]
    rejected = [r for r in results if r.status == "rejected"]
    dry_runs = [r for r in results if r.status == "dry_run"]

    log.info(f"\n{'=' * 60}")
    log.info(f"  EXECUTION SUMMARY")
    log.info(f"{'=' * 60}")

    if submitted:
        total_deployed = sum(r.notional for r in submitted)
        log.info(
            f"  Submitted: {len(submitted)} orders "
            f"(${total_deployed:,.2f} deployed)"
        )
        for r in submitted:
            log.info(
                f"    {r.side.upper():>5} {r.ticker:<6} "
                f"${r.notional:>10,.2f}  "
                f"SL=${r.stop_loss_price}  "
                f"TP=${r.take_profit_price}  "
                f"ID={r.order_id}"
            )

    if dry_runs:
        total_would = sum(r.notional for r in dry_runs)
        log.info(
            f"  Dry run: {len(dry_runs)} orders "
            f"(would deploy ${total_would:,.2f})"
        )
        for r in dry_runs:
            log.info(
                f"    {r.side.upper():>5} {r.ticker:<6} "
                f"${r.notional:>10,.2f}  "
                f"SL=${r.stop_loss_price}  "
                f"TP=${r.take_profit_price}"
            )

    if skipped:
        log.info(f"  Skipped: {len(skipped)} signals")
        for r in skipped:
            log.info(f"    {r.ticker:<6} {r.error}")

    if rejected:
        log.info(f"  Rejected: {len(rejected)} orders")
        for r in rejected:
            log.info(f"    {r.ticker:<6} {r.error}")

    log.info(f"  Execution log: {log_path}")
    log.info(f"{'=' * 60}\n")

    # Exit with error code if any orders were rejected
    if rejected and not config.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
