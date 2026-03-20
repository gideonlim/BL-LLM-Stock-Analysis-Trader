"""CLI entry point -- designed for cron / GitHub Actions."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
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


def _run_monitor_only(
    config: TradingConfig, log_dir: Path
) -> None:
    """Run the position monitor without placing new orders."""
    from trading_bot_bl.broker import AlpacaBroker
    from trading_bot_bl.monitor import (
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
            "(default: execution_logs/report_YYYY-MM-DD.pdf)."
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
            "(default: execution_logs/trades_YYYY-MM-DD.csv)."
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

    # ── Report mode ────────────────────────────────────────────────
    if args.report or args.report_json:
        _run_report(
            execution_log_dir, as_json=args.report_json
        )
        return

    if args.report_pdf is not None:
        from trading_bot_bl.journal_report import generate_pdf_report
        date_str = datetime.now().strftime("%Y-%m-%d")
        pdf_path = (
            Path(args.report_pdf)
            if args.report_pdf
            else execution_log_dir / f"report_{date_str}.pdf"
        )
        generate_pdf_report(execution_log_dir, pdf_path)
        log.info(f"  PDF report: {pdf_path}")
        return

    if args.report_csv is not None:
        from trading_bot_bl.journal_report import generate_csv_export
        date_str = datetime.now().strftime("%Y-%m-%d")
        csv_path = (
            Path(args.report_csv)
            if args.report_csv
            else execution_log_dir / f"trades_{date_str}.csv"
        )
        generate_csv_export(execution_log_dir, csv_path)
        log.info(f"  CSV export: {csv_path}")
        return

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
