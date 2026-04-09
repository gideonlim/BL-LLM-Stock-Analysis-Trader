"""Output writers for signals, trade logs, and backtest reports."""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List

from quant_analysis_bot.models import DailySignal, TradeRecord

log = logging.getLogger(__name__)


def write_signals(
    signals: List[DailySignal], config: dict
) -> tuple[str, str]:
    """Write signals to CSV and JSON, sorted by composite score."""
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    sorted_signals = sorted(
        signals, key=lambda s: -s.composite_score
    )

    # CSV
    csv_path = os.path.join(out_dir, f"signals_{date_str}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(sorted_signals[0]).keys()),
        )
        writer.writeheader()
        for s in sorted_signals:
            writer.writerow(asdict(s))
    log.info(f"Signals written to {csv_path}")

    # JSON
    json_path = os.path.join(out_dir, f"signals_{date_str}.json")
    output = {
        "generated_at": datetime.now().isoformat(),
        "risk_profile": config["risk_profile"],
        "signals": [asdict(s) for s in sorted_signals],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log.info(f"Signals written to {json_path}")

    return csv_path, json_path


def write_trade_logs(
    all_trade_logs: Dict[str, List[TradeRecord]], config: dict
) -> List[str]:
    """Write per-ticker trade log CSVs."""
    trade_dir = config["trade_log_dir"]
    os.makedirs(trade_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    paths: List[str] = []

    for ticker, trades in all_trade_logs.items():
        if not trades:
            continue
        path = os.path.join(
            trade_dir, f"trades_{ticker}_{date_str}.csv"
        )
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(asdict(trades[0]).keys())
            )
            writer.writeheader()
            for t in trades:
                writer.writerow(asdict(t))
        paths.append(path)
        log.info(f"Trade log written: {path} ({len(trades)} trades)")

    return paths


def write_backtest_report(
    all_window_results: Dict[str, Dict[str, list]],
    composite_scores: Dict[str, Dict[str, float]],
    best_strategies: Dict[str, str],
    config: dict,
) -> tuple[str, str]:
    """Write backtest report as CSV, sorted by composite score."""
    report_dir = config["report_dir"]
    os.makedirs(report_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(
        report_dir, f"backtest_report_{date_str}.csv"
    )
    weights = config["window_weights"]
    mode = (
        "LONG-ONLY"
        if config.get("long_only", True)
        else "LONG + SHORT"
    )

    CSV_COLUMNS = [
        "ticker",
        "strategy",
        "composite_score",
        "is_best",
        "timeframe",
        "annual_return_pct",
        "annual_excess_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "win_rate",
        "max_drawdown_pct",
        "total_trades",
        "profit_factor",
        "avg_holding_days",
        "calmar_ratio",
        "total_return_pct",
        "buy_hold_return_pct",
        "excess_return_pct",
        "window_score",
        "backtest_start",
        "backtest_end",
        "trading_days",
        # ── Triple-barrier metrics + tp_mode knobs ──────────────
        # Populated only when config["triple_barrier_enabled"]
        # is True; otherwise the fields default to zero.
        # tp_mode / tp_cap_multiplier / tp_cap_holding_days make
        # rows self-describing so future comparisons don't need
        # a config-to-row join.
        "tp_mode",
        "tp_cap_multiplier",
        "tp_cap_holding_days",
        "tb_total_trades",
        "tb_win_rate",
        "tb_sl_rate",
        "tb_timeout_rate",
        "tb_profit_factor",
        "tb_edge_ratio",
        "tb_avg_winner_pct",
        "tb_avg_loser_pct",
    ]

    rows: list[dict] = []
    for ticker in all_window_results:
        best_name = best_strategies[ticker]
        comp = composite_scores[ticker]

        for window_name, window_results in all_window_results[
            ticker
        ].items():
            for strat, r, sc in window_results:
                rows.append(
                    {
                        "ticker": ticker,
                        "strategy": r.strategy_name,
                        "composite_score": round(
                            comp.get(r.strategy_name, 0), 2
                        ),
                        "is_best": (
                            "Y" if strat.name == best_name else ""
                        ),
                        "timeframe": window_name,
                        "annual_return_pct": round(
                            r.annual_return_pct, 2
                        ),
                        "annual_excess_pct": round(
                            r.annual_excess_pct, 2
                        ),
                        "sharpe_ratio": round(r.sharpe_ratio, 3),
                        "sortino_ratio": round(r.sortino_ratio, 3),
                        "win_rate": round(r.win_rate, 4),
                        "max_drawdown_pct": round(
                            r.max_drawdown_pct, 2
                        ),
                        "total_trades": r.total_trades,
                        "profit_factor": round(r.profit_factor, 3),
                        "avg_holding_days": round(
                            r.avg_holding_days, 1
                        ),
                        "calmar_ratio": round(r.calmar_ratio, 3),
                        "total_return_pct": round(
                            r.total_return_pct, 2
                        ),
                        "buy_hold_return_pct": round(
                            r.buy_hold_return_pct, 2
                        ),
                        "excess_return_pct": round(
                            r.excess_return_pct, 2
                        ),
                        "window_score": round(sc, 2),
                        "backtest_start": r.backtest_start,
                        "backtest_end": r.backtest_end,
                        "trading_days": r.trading_days,
                        # ── TP mode knobs (self-describing rows) ─
                        "tp_mode": config.get("tp_mode", "current"),
                        "tp_cap_multiplier": config.get(
                            "tp_cap_multiplier", 1.5
                        ),
                        "tp_cap_holding_days": config.get(
                            "tp_cap_holding_days", 20.0
                        ),
                        # ── Triple-barrier metrics ───────────────
                        "tb_total_trades": r.tb_total_trades,
                        "tb_win_rate": round(r.tb_win_rate, 4),
                        "tb_sl_rate": round(r.tb_sl_rate, 4),
                        "tb_timeout_rate": round(
                            r.tb_timeout_rate, 4
                        ),
                        "tb_profit_factor": round(
                            r.tb_profit_factor, 3
                        ),
                        "tb_edge_ratio": round(r.tb_edge_ratio, 3),
                        "tb_avg_winner_pct": round(
                            r.tb_avg_winner_pct, 2
                        ),
                        "tb_avg_loser_pct": round(
                            r.tb_avg_loser_pct, 2
                        ),
                    }
                )

    rows.sort(
        key=lambda r: (
            -r["composite_score"],
            r["ticker"],
            r["timeframe"],
        )
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Backtest report written to {csv_path}")

    # ── Console summary ───────────────────────────────────────────────
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(
        f"  QUANT ANALYSIS BOT -- BACKTEST SUMMARY  ({date_str})"
    )
    lines.append(
        f"  Risk Profile: {config['risk_profile'].upper()}"
        f"  |  Mode: {mode}"
    )
    lines.append(
        "  Timeframe Weights: "
        + ", ".join(
            f"{k}={v:.0%}" for k, v in weights.items()
        )
    )
    lines.append("=" * 100)

    ticker_best = []
    for ticker in all_window_results:
        best_name = best_strategies[ticker]
        best_score = composite_scores[ticker].get(best_name, 0)
        ticker_best.append((ticker, best_name, best_score))
    ticker_best.sort(key=lambda x: -x[2])

    lines.append(
        f"  {'Rank':>4}  {'Ticker':<7} {'Best Strategy':<32} "
        f"{'Score':>7} {'Ann.Ret%':>9} {'Sharpe':>7} "
        f"{'WinRate':>8} {'MaxDD%':>8}"
    )
    lines.append(f"  {'-' * 92}")

    for rank, (ticker, best_name, best_score) in enumerate(
        ticker_best, 1
    ):
        best_row = None
        for row in rows:
            if row["ticker"] == ticker and row["is_best"] == "Y":
                best_row = row
                break
        if best_row:
            lines.append(
                f"  {rank:>4}  {ticker:<7} {best_name:<32} "
                f"{best_score:>7.1f} "
                f"{best_row['annual_return_pct']:>8.1f}% "
                f"{best_row['sharpe_ratio']:>7.2f} "
                f"{best_row['win_rate']:>7.1%} "
                f"{best_row['max_drawdown_pct']:>8.1f}"
            )

    lines.append("=" * 100)
    lines.append(f"  Full report: {csv_path}")
    lines.append("=" * 100)

    report_text = "\n".join(lines)
    return csv_path, report_text
