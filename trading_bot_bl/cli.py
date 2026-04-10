"""CLI entry point -- designed for cron / GitHub Actions."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from trading_bot_bl.config import TradingConfig

log = logging.getLogger(__name__)


def _archive_logs(log_dir: Path) -> None:
    """Move existing execution logs to an archive subdirectory."""
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


def _run_report(log_dir: Path, as_json: bool = False) -> None:
    """Load journal data and print performance analytics."""
    import json as _json
    from dataclasses import fields as dc_fields

    from trading_bot_bl.models import JournalEntry, EquitySnapshot
    from trading_bot_bl.journal_analytics import (
        compute_journal_metrics,
        format_metrics_text,
    )

    journal_dir = log_dir / "journal"
    equity_file = log_dir / "equity_curve.jsonl"

    # Build set of valid field names from the dataclass definition
    # (hasattr misses required fields that have no default)
    _journal_fields = {f.name for f in dc_fields(JournalEntry)}

    # ── Load closed trades ──────────────────────────────────────
    trades: list[JournalEntry] = []
    pending_count = 0
    open_count = 0

    if journal_dir.exists():
        for f in sorted(journal_dir.glob("*.json")):
            try:
                data = _json.loads(f.read_text())
                entry = JournalEntry(
                    **{
                        k: v
                        for k, v in data.items()
                        if k in _journal_fields
                    }
                )
                if entry.status == "closed":
                    trades.append(entry)
                elif entry.status == "open":
                    open_count += 1
                elif entry.status == "pending":
                    pending_count += 1
            except Exception as exc:
                log.debug(f"Skipping {f.name}: {exc}")

    # ── Load equity snapshots ───────────────────────────────────
    snapshots: list[EquitySnapshot] = []
    if equity_file.exists():
        for line in equity_file.read_text().strip().splitlines():
            if line.strip():
                try:
                    snapshots.append(
                        EquitySnapshot(**_json.loads(line))
                    )
                except Exception:
                    pass

    # ── Print summary header ────────────────────────────────────
    log.info(f"{'=' * 60}")
    log.info("  TRADE JOURNAL REPORT")
    log.info(f"{'=' * 60}")
    log.info(
        f"  Closed trades: {len(trades)}  |  "
        f"Open: {open_count}  |  "
        f"Pending: {pending_count}"
    )
    log.info(f"  Equity snapshots: {len(snapshots)}")

    if snapshots:
        first = snapshots[0]
        last = snapshots[-1]
        log.info(
            f"  Tracking period: {first.timestamp[:10]} "
            f"to {last.timestamp[:10]}"
        )
        log.info(
            f"  Starting equity: ${first.equity:,.2f}  |  "
            f"Current: ${last.equity:,.2f}  |  "
            f"Drawdown: {last.drawdown_pct:.2f}%  |  "
            f"HWM: ${last.high_water_mark:,.2f}"
        )
    log.info(f"{'=' * 60}")

    if not trades:
        log.info(
            "  No closed trades yet. Run the bot and let some "
            "trades complete before generating a report."
        )
        log.info(f"{'=' * 60}\n")
        return

    # ── Compute and display ─────────────────────────────────────
    metrics = compute_journal_metrics(trades, snapshots)

    if as_json:
        from dataclasses import asdict
        print(_json.dumps(asdict(metrics), indent=2, default=str))
    else:
        log.info(format_metrics_text(metrics))
        log.info(f"{'=' * 60}\n")


def _prune_old_logs(log_dir: Path) -> None:
    """Remove old execution and monitor logs.

    Execution logs are kept for 1 year, monitor logs for 30 days.
    Journal files and equity_curve.jsonl are never touched.
    """
    if not log_dir.exists():
        return

    now = datetime.now()
    removed = 0

    for pattern, max_age in [
        ("execution_*.json", timedelta(days=365)),
        ("monitor_*.json", timedelta(days=30)),
    ]:
        for f in log_dir.glob(pattern):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if now - mtime > max_age:
                    f.unlink()
                    removed += 1
            except OSError:
                pass

    if removed:
        log.info(f"  Pruned {removed} old log files")


def _run_monitor_only(
    config: TradingConfig, log_dir: Path
) -> None:
    """Run the position monitor without placing new orders."""
    from trading_bot_bl.config import get_broker
    from trading_bot_bl.monitor import (
        monitor_positions,
        write_monitor_log,
    )

    broker = get_broker(config)
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

    # ── Journal lifecycle hooks ─────────────────────────────────
    #    Same hooks the full executor runs (step 5d).  Without
    #    these, trades that fill or exit between executor runs
    #    aren't recorded until the next morning.
    journal_dir: Path | None = None
    if not config.dry_run:
        try:
            from trading_bot_bl import journal as _journal
            from trading_bot_bl import equity_curve as _equity

            journal_dir = log_dir / "journal"

            try:
                _equity.record_snapshot(portfolio, log_dir)
            except Exception as exc:
                log.debug(f"Equity snapshot failed: {exc}")
            try:
                _journal.resolve_pending_trades(
                    broker, journal_dir
                )
            except Exception as exc:
                log.debug(f"Journal resolve failed: {exc}")
            try:
                _journal.detect_closed_trades(
                    portfolio.positions, journal_dir, broker
                )
            except Exception as exc:
                log.debug(
                    f"Journal detect_closed failed: {exc}"
                )
        except ImportError:
            pass  # journal/equity modules not available

    if not portfolio.positions:
        log.info("  No open positions to monitor.")
        _prune_old_logs(log_dir)
        return

    report = monitor_positions(
        broker=broker,
        portfolio=portfolio,
        limits=config.risk,
        dry_run=config.dry_run,
        journal_dir=journal_dir,
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

    # Prune old log files (execution: 1 year, monitor: 30 days)
    _prune_old_logs(log_dir)

    # Exit with error only if there are critical alerts that
    # were NOT resolved by the monitor (i.e. no action was taken
    # or the action failed).  Resolved criticals (reattached
    # brackets, successful closes) should not fail the workflow.
    unresolved = [
        a for a in report.alerts
        if a.severity == "critical"
        and (
            not a.action_taken
            or "failed" in a.action_taken.lower()
            or "DRY RUN" in a.action_taken
        )
    ]
    if unresolved:
        sys.exit(1)


def main() -> None:
    """Entry point for the trading bot."""
    # Load .env from inside the trading_bot_bl folder
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(env_path)
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
    parser.add_argument(
        "--report",
        action="store_true",
        help=(
            "Show trade journal performance report. "
            "Loads all closed trades and equity snapshots, "
            "computes analytics (Sharpe, win rate, P&L, etc.), "
            "and prints a summary. No orders are placed."
        ),
    )
    parser.add_argument(
        "--report-json",
        action="store_true",
        help=(
            "Same as --report but outputs machine-readable JSON "
            "instead of formatted text."
        ),
    )
    parser.add_argument(
        "--report-pdf",
        type=str,
        nargs="?",
        const="",
        default=None,
        help=(
            "Generate a PDF performance report with charts. "
            "Optionally provide an output path "
            "(default: reports/performance/report_YYYY-MM-DD.pdf)."
        ),
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        nargs="?",
        const="",
        default=None,
        help=(
            "Export all closed trades as CSV. "
            "Optionally provide an output path "
            "(default: reports/trades_YYYY-MM-DD.csv)."
        ),
    )
    parser.add_argument(
        "--no-bl",
        action="store_true",
        help=(
            "Disable Black-Litterman optimization. "
            "Falls back to marginal Sharpe ranking."
        ),
    )
    parser.add_argument(
        "--llm-views",
        action="store_true",
        help=(
            "Enable LLM-enhanced views for Black-Litterman. "
            "Requires ANTHROPIC_API_KEY or OPENAI_API_KEY."
        ),
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        help=(
            "Override LLM model "
            "(e.g. claude-sonnet-4-5-20250514)"
        ),
    )
    parser.add_argument(
        "--llm-samples",
        type=int,
        help="Number of LLM samples per ticker (default: 10)",
    )
    parser.add_argument(
        "--llm-max-tickers",
        type=int,
        help=(
            "Max tickers to query LLM for — "
            "only top-N by confidence (default: 10)"
        ),
    )
    parser.add_argument(
        "--bl-tau",
        type=float,
        help="Black-Litterman tau parameter (default: 0.05)",
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

    # Black-Litterman / LLM overrides from CLI
    if args.no_bl:
        config.use_black_litterman = False
    if args.llm_views:
        config.llm_views_enabled = True
    if args.llm_model:
        config.llm_model = args.llm_model
    if args.llm_samples:
        config.llm_num_samples = args.llm_samples
    if args.llm_max_tickers:
        config.llm_max_tickers = args.llm_max_tickers
    if args.bl_tau:
        config.bl_tau = args.bl_tau

    # ── Validate ──────────────────────────────────────────────────
    mode = "PAPER" if config.alpaca.paper else "*** LIVE ***"
    dry = " (DRY RUN)" if config.dry_run else ""
    monitor = " [MONITOR ONLY]" if args.monitor_only else ""
    bl_mode = (
        "Black-Litterman"
        + (" + LLM" if config.llm_views_enabled else "")
        if config.use_black_litterman
        else "Marginal Sharpe"
    )
    log.info(f"{'=' * 60}")
    log.info(f"  Trading Bot -- {mode}{dry}{monitor}")
    log.info(f"  Optimization: {bl_mode}")
    log.info(f"  Signals dir: {config.signals_dir}")
    log.info(
        f"  Risk limits: "
        f"exposure={config.risk.max_portfolio_exposure_pct}%, "
        f"per-stock={config.risk.max_position_pct}%, "
        f"max-positions={config.risk.max_positions}, "
        f"daily-loss={config.risk.daily_loss_limit_pct}%"
    )
    if config.use_black_litterman:
        log.info(
            f"  BL params: tau={config.bl_tau}, "
            f"lookback={config.bl_lookback_days}d"
        )
    if config.llm_views_enabled:
        log.info(
            f"  LLM: {config.llm_provider}/{config.llm_model}, "
            f"samples={config.llm_num_samples}, "
            f"max_tickers={config.llm_max_tickers}, "
            f"weight={config.llm_weight}"
        )
    log.info(f"{'=' * 60}")

    # Use config-scoped path unless operator explicitly overrides
    if args.log_dir != "execution_logs":
        # Explicit --log-dir override — respect it
        execution_log_dir = Path(args.log_dir)
    else:
        # Default: use market-scoped path
        execution_log_dir = config.path_for("execution_logs")

    # ── Reset history ─────────────────────────────────────────────
    if args.reset_history:
        _archive_logs(execution_log_dir)

    # ── Report modes (no Alpaca credentials needed) ───────────────
    if args.report or args.report_json:
        _run_report(
            execution_log_dir, as_json=args.report_json
        )
        return

    if args.report_pdf is not None:
        from trading_bot_bl.journal_report import generate_pdf_report
        market = config.get_market()
        date_str = datetime.now().strftime("%Y-%m-%d")
        if args.report_pdf:
            pdf_path = Path(args.report_pdf)
        else:
            reports_dir = (
                config.path_for("reports") / "performance"
            )
            reports_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = reports_dir / f"report_{date_str}.pdf"
        generate_pdf_report(
            execution_log_dir,
            pdf_path,
            benchmark_ticker=market.regime_benchmark_ticker,
            trading_days_per_year=market.trading_days_per_year,
        )
        log.info(f"  PDF report: {pdf_path}")
        return

    if args.report_csv is not None:
        from trading_bot_bl.journal_report import generate_csv_export
        date_str = datetime.now().strftime("%Y-%m-%d")
        if args.report_csv:
            csv_path = Path(args.report_csv)
        else:
            reports_dir = config.path_for("reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            csv_path = reports_dir / f"trades_{date_str}.csv"
        generate_csv_export(execution_log_dir, csv_path)
        log.info(f"  CSV export: {csv_path}")
        return

    # ── Validate broker credentials (only needed for trading/monitoring) ──
    if not config.dry_run:
        broker_type = config.get_market().broker_type
        if broker_type == "alpaca":
            try:
                config.alpaca.validate()
            except ValueError as e:
                log.error(str(e))
                sys.exit(1)
        elif broker_type == "ibkr":
            log.info(
                f"  IBKR broker — credentials managed by "
                f"IB Gateway ({config.ibkr})"
            )

    # ── Monitor-only mode ─────────────────────────────────────────
    if args.monitor_only:
        _run_monitor_only(config, execution_log_dir)
        return

    # ── Execute ───────────────────────────────────────────────────
    from trading_bot_bl.executor import execute, write_execution_log

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
