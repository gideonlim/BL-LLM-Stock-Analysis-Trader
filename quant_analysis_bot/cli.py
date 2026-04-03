"""Command-line interface and main execution pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from quant_analysis_bot.config import load_config
from quant_analysis_bot.models import DailySignal, TradeRecord
from quant_analysis_bot.output import (
    write_backtest_report,
    write_signals,
    write_trade_logs,
)
from quant_analysis_bot.progress import ProgressBar
from quant_analysis_bot.universe import fetch_top_us_stocks

log = logging.getLogger(__name__)


# ── Per-ticker result container ──────────────────────────────────────

@dataclass
class _TickerResult:
    """Everything produced by analysing a single ticker."""

    ticker: str
    signal: DailySignal
    per_window: dict = field(default_factory=dict)
    composite_scores: dict = field(default_factory=dict)
    best_strategy_name: str = ""
    trade_logs: list[TradeRecord] = field(default_factory=list)
    cscv_result: object = None          # CSCVResult | None
    error: Optional[str] = None


# ── Worker function (runs in child process) ──────────────────────────

def _analyze_ticker(
    ticker: str,
    price_df_or_none: Optional[object],
    regime_df: object,
    config: dict,
    today: str,
) -> _TickerResult:
    """Analyse one ticker end-to-end.

    This is a **module-level** function so it can be pickled by
    ``ProcessPoolExecutor``.  It imports everything it needs inside
    the function body to avoid serialising heavy modules.
    """
    from quant_analysis_bot.backtest import select_best_strategy
    from quant_analysis_bot.cscv import run_cscv_for_ticker
    from quant_analysis_bot.data import enrich_dataframe, fetch_data
    from quant_analysis_bot.pead import (
        build_earnings_context,
        enrich_with_pead,
    )
    from quant_analysis_bot.regime import enrich_with_regime
    from quant_analysis_bot.signals import generate_daily_signal

    skip_pead = config.get("skip_pead", False)
    run_cscv_flag = config.get("run_cscv", False)

    try:
        # 1. Price data — from pre-fetched cache or individual download
        if price_df_or_none is not None:
            df = price_df_or_none
        else:
            df = fetch_data(
                ticker,
                config["lookback_days"],
                config["data_cache_dir"],
            )

        # 2. Feature enrichment
        df = enrich_dataframe(df)
        df = enrich_with_regime(df, regime_df)
        if not skip_pead:
            df = enrich_with_pead(df, ticker)

        # 2b. Earnings context (forward-looking date + last surprise)
        #     Uses the PEAD-enriched df (for backward surprise) and
        #     yfinance calendar (for next earnings date).
        #     Skipped when --skip-pead is set (yfinance unavailable).
        earnings_ctx = None
        if not skip_pead:
            try:
                earnings_ctx = build_earnings_context(df, ticker)
            except Exception:
                earnings_ctx = None

        # 3. Strategy selection (14 strategies × 3 timeframes)
        (
            best_strat,
            best_result,
            per_window,
            comp_scores,
            trade_logs,
        ) = select_best_strategy(df, ticker, config)

        # 4. Signal generation (with earnings awareness)
        signal = generate_daily_signal(
            df, ticker, best_strat, best_result, config,
            earnings_ctx=earnings_ctx,
        )

        # 5. CSCV overfitting check (for BUY signals or --validate)
        cscv_res = None
        should_cscv = run_cscv_flag or signal.signal == "BUY"
        if should_cscv:
            try:
                cscv_res = run_cscv_for_ticker(df, ticker, config)
                if cscv_res.is_valid:
                    best_result.pbo = cscv_res.pbo
                    signal = generate_daily_signal(
                        df, ticker, best_strat,
                        best_result, config,
                        earnings_ctx=earnings_ctx,
                    )
                else:
                    cscv_res = None
            except Exception:
                cscv_res = None

        return _TickerResult(
            ticker=ticker,
            signal=signal,
            per_window=per_window,
            composite_scores=comp_scores,
            best_strategy_name=best_strat.name,
            trade_logs=trade_logs,
            cscv_result=cscv_res,
        )

    except Exception as e:
        return _TickerResult(
            ticker=ticker,
            signal=DailySignal(
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
                pbo=-1.0,
                rsi=0.0,
                vol_20=0.0,
                sma_50=0.0,
                sma_200=0.0,
                trend="N/A",
                volatility="N/A",
                notes=str(e),
            ),
            error=str(e),
        )


# ── Main pipeline ────────────────────────────────────────────────────

def run(config: dict) -> None:
    """Main execution pipeline."""
    from quant_analysis_bot.cscv import format_cscv_report
    from quant_analysis_bot.data import batch_fetch_data
    from quant_analysis_bot.regime import fetch_regime_data

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
    n_workers = config.get("workers", 1)
    if n_workers > 1:
        log.info(f"  Workers: {n_workers}")
    log.info(f"{'=' * 60}")

    all_signals: list[DailySignal] = []
    all_window_results: dict = {}
    all_composite_scores: dict = {}
    best_strategies: dict = {}
    all_trade_logs: dict = {}
    cscv_results: dict = {}

    tickers = config["tickers"]
    n_tickers = len(tickers)
    use_progress = n_tickers > 10
    failed_tickers: list[str] = []

    # ── Fetch market regime data FIRST (only 2 tickers: VIX + SPY)
    regime_df = fetch_regime_data(
        lookback_days=config["lookback_days"],
        vix_fear_threshold=config.get("vix_fear_threshold", 25.0),
    )

    # ── Batch download price data (all tickers at once) ───────────
    if n_tickers > 1:
        log.info("Batch downloading price data...")
        price_cache = batch_fetch_data(
            tickers,
            config["lookback_days"],
            config["data_cache_dir"],
        )
    else:
        price_cache = {}

    # ── Analyze all tickers ──────────────────────────────────────

    ctx = (
        ProgressBar(
            total=n_tickers,
            desc="Analyzing stocks",
            unit="stocks",
        )
        if use_progress
        else nullcontext()
    )

    if n_workers > 1 and n_tickers > 1:
        # ── Parallel execution ───────────────────────────────────
        results: list[_TickerResult] = []

        with ctx as pbar:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _analyze_ticker,
                        ticker,
                        price_cache.get(ticker),
                        regime_df,
                        config,
                        today,
                    ): ticker
                    for ticker in tickers
                }

                try:
                    for future in as_completed(futures):
                        ticker = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            # Should not happen — _analyze_ticker
                            # catches internally — but just in case.
                            result = _TickerResult(
                                ticker=ticker,
                                signal=_make_error_signal(
                                    ticker, today, str(e)
                                ),
                                error=str(e),
                            )
                        results.append(result)

                        if use_progress and pbar:
                            pbar.update(1, suffix=ticker)

                except KeyboardInterrupt:
                    log.warning(
                        "\nInterrupted — cancelling remaining "
                        "workers and outputting partial results..."
                    )
                    pool.shutdown(
                        wait=False, cancel_futures=True
                    )

        # Sort results back into original ticker order
        order = {t: i for i, t in enumerate(tickers)}
        results.sort(key=lambda r: order.get(r.ticker, 999999))

    else:
        # ── Sequential execution (original path) ─────────────────
        results = []

        with ctx as pbar:
            for _i, ticker in enumerate(tickers):
                if not use_progress:
                    log.info(f"\n>>> Analyzing {ticker}...")

                result = _analyze_ticker(
                    ticker,
                    price_cache.get(ticker),
                    regime_df,
                    config,
                    today,
                )
                results.append(result)

                if not use_progress and result.error is None:
                    log.info(
                        f"  Best strategy: "
                        f"{result.best_strategy_name} "
                        f"(Signal: {result.signal.signal}, "
                        f"Confidence: "
                        f"{result.signal.confidence})"
                    )
                elif not use_progress and result.error:
                    log.error(
                        f"  FAILED for {ticker}: {result.error}"
                    )

                if use_progress and pbar:
                    pbar.update(1, suffix=ticker)

    # ── Collect results ──────────────────────────────────────────
    for result in results:
        all_signals.append(result.signal)
        if result.error:
            failed_tickers.append(result.ticker)
            continue
        all_window_results[result.ticker] = result.per_window
        all_composite_scores[result.ticker] = (
            result.composite_scores
        )
        best_strategies[result.ticker] = (
            result.best_strategy_name
        )
        all_trade_logs[result.ticker] = result.trade_logs
        if result.cscv_result is not None:
            cscv_results[result.ticker] = result.cscv_result

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

    # ── CSCV overfitting report ────────────────────────────────────────
    if cscv_results:
        print(f"\n{format_cscv_report(cscv_results)}")

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
        pbo_str = (
            f"  PBO: {s.pbo:.0%}" if s.pbo >= 0 else ""
        )
        print(
            f"       Ann.Return: {s.annual_return_pct}%  "
            f"Ann.Excess: {s.annual_excess_pct}%  "
            f"MaxDD: {s.max_drawdown_pct}%{pbo_str}"
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


def _make_error_signal(
    ticker: str, today: str, error: str
) -> DailySignal:
    """Build an ERROR signal for a failed ticker."""
    return DailySignal(
        generated_at=datetime.now().isoformat(timespec="seconds"),
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
        pbo=-1.0,
        rsi=0.0,
        vol_20=0.0,
        sma_50=0.0,
        sma_200=0.0,
        trend="N/A",
        volatility="N/A",
        notes=error,
    )


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
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Run CSCV overfitting analysis (PBO) on each ticker. "
            "Adds a validation report showing probability of "
            "backtest overfitting."
        ),
    )
    parser.add_argument(
        "--skip-pead",
        action="store_true",
        help=(
            "Skip PEAD earnings data fetching for faster runs. "
            "The PEAD Earnings Drift strategy will return HOLD "
            "for all tickers."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes for analysis. "
            "0 (default) = auto-detect CPU cores (capped at 8). "
            "1 = sequential (no multiprocessing)."
        ),
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

    config["run_cscv"] = args.validate
    config["skip_pead"] = args.skip_pead

    # Worker count: 0 = auto, 1 = sequential, N = explicit
    if args.workers == 0:
        config["workers"] = min(os.cpu_count() or 1, 4)
    else:
        config["workers"] = max(1, args.workers)

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
