"""Backtesting engine and multi-timeframe strategy selector."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quant_analysis_bot.config import RISK_PROFILES
from quant_analysis_bot.models import BacktestResult, TradeRecord
from quant_analysis_bot.strategies import ALL_STRATEGIES, Strategy

log = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    ticker: str,
    strategy_name: str,
    timeframe: str,
    cost_bps: float = 10,
    long_only: bool = True,
) -> Tuple[BacktestResult, List[TradeRecord]]:
    """
    Run a full backtest on a signal series.

    Returns performance metrics AND a list of every individual trade.
    """
    result = BacktestResult(
        strategy_name=strategy_name,
        ticker=ticker,
        timeframe=timeframe,
    )
    trade_log: List[TradeRecord] = []

    df = df.copy()
    df["Signal"] = signals
    df = df.dropna(subset=["Close"])

    if len(df) < 30:
        return result, trade_log

    dates = df.index
    close = df["Close"].values
    sig = df["Signal"].values
    cost = cost_bps / 10000

    result.backtest_start = str(dates[0].date())
    result.backtest_end = str(dates[-1].date())
    result.trading_days = len(df)

    # Track positions and returns
    position = 0        # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_idx = 0
    trades_raw: list = []
    daily_returns: list = []
    equity = [1.0]
    trade_count = 0

    for i in range(1, len(close)):
        daily_ret = 0.0

        if position != 0:
            daily_ret = position * (close[i] / close[i - 1] - 1)

        # ── Process signals ───────────────────────────────────────────
        if sig[i] == 1 and position <= 0:
            if position == -1 and not long_only:
                # Close short
                trade_ret = entry_price / close[i] - 1 - cost
                holding = (dates[i] - dates[entry_idx]).days
                trade_count += 1
                trade_log.append(
                    TradeRecord(
                        trade_num=trade_count,
                        ticker=ticker,
                        strategy=strategy_name,
                        timeframe=timeframe,
                        direction="SHORT",
                        entry_date=str(dates[entry_idx].date()),
                        entry_price=round(close[entry_idx], 2),
                        exit_date=str(dates[i].date()),
                        exit_price=round(close[i], 2),
                        holding_days=holding,
                        return_pct=round(trade_ret * 100, 2),
                        outcome="WIN" if trade_ret > 0 else "LOSS",
                    )
                )
                trades_raw.append(
                    {"return": trade_ret, "holding_days": holding}
                )
            # Open long
            position = 1
            entry_price = close[i]
            entry_idx = i
            daily_ret -= cost

        elif sig[i] == -1 and position >= 0:
            if position == 1:
                # Close long
                trade_ret = close[i] / entry_price - 1 - cost
                holding = (dates[i] - dates[entry_idx]).days
                trade_count += 1
                trade_log.append(
                    TradeRecord(
                        trade_num=trade_count,
                        ticker=ticker,
                        strategy=strategy_name,
                        timeframe=timeframe,
                        direction="LONG",
                        entry_date=str(dates[entry_idx].date()),
                        entry_price=round(close[entry_idx], 2),
                        exit_date=str(dates[i].date()),
                        exit_price=round(close[i], 2),
                        holding_days=holding,
                        return_pct=round(trade_ret * 100, 2),
                        outcome="WIN" if trade_ret > 0 else "LOSS",
                    )
                )
                trades_raw.append(
                    {"return": trade_ret, "holding_days": holding}
                )

            if long_only:
                position = 0
                entry_price = 0.0
                daily_ret -= cost if position == 1 else 0
            else:
                position = -1
                entry_price = close[i]
                entry_idx = i
                daily_ret -= cost

        daily_returns.append(daily_ret)
        equity.append(equity[-1] * (1 + daily_ret))

    equity_arr = np.array(equity)
    returns_arr = np.array(daily_returns)

    if len(returns_arr) == 0:
        return result, trade_log

    n_days = len(returns_arr)

    # ── Core metrics ──────────────────────────────────────────────────
    result.total_return_pct = round(
        (equity_arr[-1] / equity_arr[0] - 1) * 100, 2
    )
    result.buy_hold_return_pct = round(
        (close[-1] / close[0] - 1) * 100, 2
    )
    result.excess_return_pct = round(
        result.total_return_pct - result.buy_hold_return_pct, 2
    )

    # ── Annualized returns ────────────────────────────────────────────
    if n_days > 0:
        ann_factor = 252 / n_days
        total_r = equity_arr[-1] / equity_arr[0]
        bh_r = close[-1] / close[0]
        result.annual_return_pct = round(
            (total_r**ann_factor - 1) * 100, 2
        )
        result.annual_bh_return_pct = round(
            (bh_r**ann_factor - 1) * 100, 2
        )
        result.annual_excess_pct = round(
            result.annual_return_pct - result.annual_bh_return_pct, 2
        )

    # ── Sharpe Ratio ──────────────────────────────────────────────────
    if returns_arr.std() > 0:
        result.sharpe_ratio = round(
            np.sqrt(252) * returns_arr.mean() / returns_arr.std(), 2
        )

    # ── Sortino Ratio ─────────────────────────────────────────────────
    downside = returns_arr[returns_arr < 0]
    if len(downside) > 0 and downside.std() > 0:
        result.sortino_ratio = round(
            np.sqrt(252) * returns_arr.mean() / downside.std(), 2
        )

    # ── Max Drawdown ──────────────────────────────────────────────────
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - peak) / peak
    result.max_drawdown_pct = round(drawdown.min() * 100, 2)

    # ── Calmar Ratio ──────────────────────────────────────────────────
    if result.max_drawdown_pct != 0:
        result.calmar_ratio = round(
            result.annual_return_pct / abs(result.max_drawdown_pct), 2
        )

    # ── Trade stats ───────────────────────────────────────────────────
    result.total_trades = len(trades_raw)
    if trades_raw:
        wins = [t["return"] for t in trades_raw if t["return"] > 0]
        losses = [t["return"] for t in trades_raw if t["return"] <= 0]
        result.win_rate = round(len(wins) / len(trades_raw), 3)
        result.avg_win_pct = (
            round(np.mean(wins) * 100, 2) if wins else 0.0
        )
        result.avg_loss_pct = (
            round(np.mean(losses) * 100, 2) if losses else 0.0
        )
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0.001
        result.profit_factor = round(total_wins / total_losses, 2)
        result.avg_holding_days = round(
            np.mean([t["holding_days"] for t in trades_raw]), 1
        )

    return result, trade_log


# ── Scoring ───────────────────────────────────────────────────────────


def score_single_window(
    result: BacktestResult, risk_config: dict
) -> float:
    """Score a strategy on a single timeframe window."""
    score = 0.0

    # Sharpe ratio (35%)
    score += min(result.sharpe_ratio, 3.0) * 35
    # Annualized excess return (20%)
    score += np.clip(result.annual_excess_pct, -100, 100) * 0.2
    # Win rate (15%)
    score += (result.win_rate - 0.5) * 150
    # Profit factor (15%)
    score += min(result.profit_factor, 3.0) * 15
    # Drawdown penalty (15%)
    score += result.max_drawdown_pct * 0.5

    # Penalize too few trades
    if result.total_trades < 3:
        score *= 0.4
    elif result.total_trades < 5:
        score *= 0.7

    if result.win_rate < risk_config["min_win_rate"]:
        score *= 0.7

    return round(score, 2)


# ── Multi-timeframe strategy selector ─────────────────────────────────


def select_best_strategy(
    df: pd.DataFrame, ticker: str, config: dict
) -> Tuple[
    Strategy,
    BacktestResult,
    Dict[str, list],
    Dict[str, float],
    List[TradeRecord],
]:
    """
    Test all strategies across multiple timeframes.

    Returns
    -------
    best_strategy, best_result, per_window_results,
    composite_scores, all_trade_logs
    """
    risk_config = RISK_PROFILES[config["risk_profile"]]
    windows = config["backtest_windows"]
    weights = config["window_weights"]

    strategy_windows: Dict[str, Dict[str, Tuple]] = {}
    per_window_results: Dict[str, list] = {}
    all_trade_logs: List[TradeRecord] = []

    for window_name, n_days in windows.items():
        window_df = df.tail(n_days).copy()
        if len(window_df) < 30:
            log.warning(
                f"  Skipping {window_name} for {ticker}: "
                f"only {len(window_df)} days"
            )
            continue

        window_results = []

        for strategy in ALL_STRATEGIES:
            try:
                signals = strategy.generate_signals(window_df)
                result, trade_log = run_backtest(
                    window_df,
                    signals,
                    ticker,
                    strategy.name,
                    window_name,
                    config["transaction_cost_bps"],
                    long_only=config.get("long_only", True),
                )
                sc = score_single_window(result, risk_config)
                result.score = sc
                window_results.append((strategy, result, sc))
                all_trade_logs.extend(trade_log)

                if strategy.name not in strategy_windows:
                    strategy_windows[strategy.name] = {}
                strategy_windows[strategy.name][window_name] = (
                    strategy,
                    result,
                    sc,
                )

            except Exception as e:
                log.warning(
                    f"  {strategy.name} failed on {window_name} "
                    f"for {ticker}: {e}"
                )

        per_window_results[window_name] = sorted(
            window_results, key=lambda x: x[2], reverse=True
        )

    # ── Composite score across timeframes ─────────────────────────────
    composite_scores: Dict[str, float] = {}
    strategy_objs: Dict[str, Strategy] = {}

    for strat_name, window_data in strategy_windows.items():
        weighted_score = 0.0
        total_weight = 0.0
        for window_name, weight in weights.items():
            if window_name in window_data:
                _, _, sc = window_data[window_name]
                weighted_score += sc * weight
                total_weight += weight
        if total_weight > 0:
            composite_scores[strat_name] = round(
                weighted_score / total_weight, 2
            )
            strategy_objs[strat_name] = list(
                window_data.values()
            )[0][0]

    if not composite_scores:
        raise ValueError(f"All strategies failed for {ticker}")

    best_name = max(composite_scores, key=composite_scores.get)
    best_strategy = strategy_objs[best_name]

    longest_window = max(windows.keys(), key=lambda w: windows[w])
    if longest_window in strategy_windows.get(best_name, {}):
        best_result = strategy_windows[best_name][longest_window][1]
    else:
        best_result = list(
            strategy_windows[best_name].values()
        )[0][1]

    best_result.composite_score = composite_scores[best_name]

    return (
        best_strategy,
        best_result,
        per_window_results,
        composite_scores,
        all_trade_logs,
    )
