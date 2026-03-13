"""Command-line interface and main execution pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import nullcontext
from datetime import datetime

from quant_analysis_bot.backtest import select_best_strategy
from quant_analysis_bot.config import load_config
from quant_analysis_bot.data import enrich_dataframe, fetch_data
from quant_analysis_bot.models import DailySignal
from quant_analysis_bot.output import (
    write_backtest_report,
    write_signals,
    write_trade_logs,
)
from quant_analysis_bot.progress import ProgressBar
from quant_analysis_bot.signals import generate_daily_signal
from quant_analysis_bot.universe import fetch_top_us_stocks

log = logging.getLogger(__name__)


def run(config: dict) -> None:
    """Main execution pipeline."""
    today = datetime.now().strftime("%Y-%m-%d")
    mode = (
        "LONG-ONLY"
        if config.get("long_only", True)
        else "LONG + SHORT"
    )
    log.info(f"{'=' * 60}")
    log.info(f"  Quant Analysis Bot -- {today}")
    tickers = config["tickers"]
    if len(tickers) <= 15:
        log.info(f"  Tickers: {', '.join(tickers)}")
    else:
        log.info(
            f"  Tickers: {len(tickers)} stocks "
            f"(top by market cap)"
        )
    log.info(
        f"  Risk Profile: {config['risk_profile']}  |  Mode: {mode}"
    )
    log.info(
        f"  Timeframes: "
        f"{', '.join(config['backtest_windows'].keys())}"
    )
    log.info(f"{'=' * 60}")

    all_signals: list[DailySignal] = []
    all_window_results: dict = {}
    all_composite_scores: dict = {}
    best_strategies: dict = {}
    all_trade_logs: dict = {}

    tickers = config["tickers"]
    n_tickers = len(tickers)
    use_progress = n_tickers > 10
    failed_tickers: list[str] = []

    ctx = (
        ProgressBar(
            total=n_tickers,
            desc="Analyzing stocks",
            unit="stocks",
        )
        if use_progress
        else nullcontext()
    )

    with ctx as pbar:
        for _i, ticker in enumerate(tickers):
            if not use_progress:
                log.info(f"\n>>> Analyzing {ticker}...")

            try:
                df = fetch_data(
                    ticker,
                    config["lookback_days"],
                    config["data_cache_dir"],
                )
                df = enrich_dataframe(df)

                (
                    best_strat,
                    best_result,
                    per_window,
                    comp_scores,
                    trade_logs,
                ) = select_best_strategy(df, ticker, config)

                if not use_progress:
                    log.info(
                        f"  Best strategy: {best_strat.name} "
                        f"(Composite="
                        f"{best_result.composite_score:.1f}, "
                        f"Sharpe={best_result.sharpe_ratio}, "
                        f"Ann.Return="
                        f"{best_result.annual_return_pct}%, "
                        f"WinRate={best_result.win_rate:.1%})"
                    )

                all_window_results[ticker] = per_window
                all_composite_scores[ticker] = comp_scores
                best_strategies[ticker] = best_strat.name
                all_trade_logs[ticker] = trade_logs

                signal = generate_daily_signal(
                    df, ticker, best_strat, best_result, config
                )
                all_signals.append(signal)

                if not use_progress:
                    log.info(
                        f"  Signal: {signal.signal} "
                        f"(Confidence: {signal.confidence})"
                    )
                    log.info(
                        f"  Total trades logged: "
                        f"{len(trade_logs)}"
                    )

            except Exception as e:
                if not use_progress:
                    log.error(f"  FAILED for {ticker}: {e}")
                failed_tickers.append(ticker)
                all_signals.append(
                    DailySignal(
                        generated_at=datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        date=today,
                        ticker=ticker,
                        signal="ERROR",
                        signal_raw=0,
                        strategy="N/A",
                        confidence="N/A",
                        confidence_score=0,
                        composite_score=0.0,
                        current_price=0.0,
                        stop_loss_pct=0.0,
                        stop_loss_price=0.0,
                        take_profit_pct=0.0,
                        take_profit_price=0.0,
                        suggested_position_size_pct=0.0,
                        signal_expires=today,
                        sharpe=0.0,
                        sortino=0.0,
                        win_rate=0.0,
                        profit_factor=0.0,
                        annual_return_pct=0.0,
                        annual_excess_pct=0.0,
                        max_drawdown_pct=0.0,
                        avg_holding_days=0.0,
                        total_trades=0,
                        backtest_period="N/A",
                        rsi=0.0,
                        vol_20=0.0,
                        sma_50=0.0,
                        sma_200=0.0,
                        trend="N/A",
                        volatility="N/A",
                        notes=str(e),
                    )
                )

            if use_progress and pbar:
                pbar.update(1, suffix=ticker)

    # Report failures summary for large runs
    if use_progress and failed_tickers:
        log.warning(
            f"\n  {len(failed_tickers)}/{n_tickers} tickers "
            f"failed: {', '.join(failed_tickers[:20])}"
            f"{'...' if len(failed_tickers) > 20 else ''}"
        )

    # ── Write outputs ─────────────────────────────────────────────────
    csv_path, json_path = "", ""
    if all_signals:
        csv_path, json_path = write_signals(all_signals, config)

    if all_trade_logs:
        write_trade_logs(all_trade_logs, config)

    if all_window_results:
        _report_path, report_text = write_backtest_report(
            all_window_results,
            all_composite_scores,
            best_strategies,
            config,
        )
        print(f"\n{report_text}")

    # ── Signal summary ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  TODAY'S SIGNALS  ({today})")
    print(f"{'=' * 70}")
    for s in all_signals:
        icon = {
            "BUY": "+",
            "SELL/SHORT": "-",
            "EXIT": "x",
            "HOLD": "=",
            "ERROR": "!",
        }
        marker = icon.get(s.signal, "?")
        print(f"  [{marker}] {s.ticker:<6} {s.signal:<5} | {s.strategy}")
        print(
            f"       Price: ${s.current_price}  RSI: {s.rsi}  "
            f"Trend: {s.trend}  Confidence: {s.confidence}"
        )
        print(
            f"       Ann.Return: {s.annual_return_pct}%  "
            f"Ann.Excess: {s.annual_excess_pct}%  "
            f"MaxDD: {s.max_drawdown_pct}%"
        )
        print(f"       Backtest: {s.backtest_period}")
        if s.notes and s.notes != "No special conditions":
            print(f"       Notes: {s.notes}")
    print(f"{'=' * 70}")
    if csv_path:
        print(f"  Signals:    {csv_path}  |  {json_path}")
    if all_trade_logs:
        total_trades = sum(
            len(t) for t in all_trade_logs.values()
        )
        print(
            f"  Trade logs: {config['trade_log_dir']}/ "
            f"({total_trades} trades total)"
        )
    print(f"{'=' * 70}\n")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Quant Analysis Bot"
    )
    parser.add_argument(
        "--config", type=str, help="Path to JSON config file"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        help="Override ticker list",
    )
    parser.add_argument(
        "--risk",
        type=str,
        choices=["conservative", "moderate", "aggressive"],
        help="Override risk profile",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        default=None,
        help="Long-only mode (default)",
    )
    parser.add_argument(
        "--long-short",
        action="store_true",
        default=None,
        help="Long+Short mode",
    )
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="Analyze top US stocks by market cap",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1000,
        help="Number of top stocks with --all-stocks (default: 1000)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.tickers:
        config["tickers"] = args.tickers
    if args.risk:
        config["risk_profile"] = args.risk
    if args.long_only:
        config["long_only"] = True
    elif args.long_short:
        config["long_only"] = False

    if args.all_stocks:
        log.info(
            f"Fetching top {args.top_n} US stocks by market cap..."
        )
        config["tickers"] = fetch_top_us_stocks(
            n=args.top_n,
            cache_dir=config.get("data_cache_dir", "cache"),
        )
        if not config["tickers"]:
            log.error(
                "Failed to fetch stock universe. Exiting."
            )
            sys.exit(1)
        log.info(
            f"Will analyze {len(config['tickers'])} stocks"
        )
        if len(config["tickers"]) > 20:
            logging.getLogger("quant_analysis_bot").setLevel(
                logging.WARNING
            )

    run(config)


if __name__ == "__main__":
    main()
