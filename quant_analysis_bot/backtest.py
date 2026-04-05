"""Backtesting engine and multi-timeframe strategy selector."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from quant_analysis_bot.config import RISK_PROFILES
from quant_analysis_bot.models import BacktestResult, TradeRecord
from quant_analysis_bot.signals import compute_atr_stop_loss_pct
from quant_analysis_bot.strategies import ALL_STRATEGIES, Strategy
from quant_analysis_bot.triple_barrier import (
    BarrierTrade,
    apply_triple_barrier,
    cusum_filter,
    score_barrier_trades,
)

log = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    ticker: str,
    strategy_name: str,
    timeframe: str,
    cost_bps: float = 10,
    long_only: bool = True,
    next_bar_execution: bool = True,
) -> Tuple[BacktestResult, List[TradeRecord]]:
    """
    Run a full backtest on a signal series.

    If next_bar_execution is True (default), signals generated on bar[i]
    execute at close[i+1]. This eliminates look-ahead bias -- you can't
    trade at the close of the bar that generated the signal in practice.
    Typically reduces reported Sharpe by 10-20% for a more honest estimate.

    Returns performance metrics AND a list of every individual trade.
    """
    result = BacktestResult(
        strategy_name=strategy_name,
        ticker=ticker,
        timeframe=timeframe,
    )
    trade_log: List[TradeRecord] = []

    df = df.copy()
    if next_bar_execution:
        # Shift signals forward by 1 bar: a signal on bar[i] executes
        # on bar[i+1]. This prevents look-ahead bias.
        df["Signal"] = signals.shift(1, fill_value=0)
    else:
        df["Signal"] = signals
    df = df.dropna(subset=["Close"])

    if len(df) < 30:
        return result, trade_log, np.array([])

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
                daily_ret -= cost  # exit transaction cost
                position = 0
                entry_price = 0.0
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
        return result, trade_log, np.array([])

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
    # Target Downside Deviation (TDD) per Sortino & Price (1994):
    #   TDD = sqrt( (1/N) × Σ min(Rᵢ - T, 0)² )
    # where T = target return (0 for excess-of-zero), N = ALL
    # observations.  Common mistake: filtering to only negative
    # returns and taking std() — that divides by n_negative instead
    # of n_total, systematically overstating downside deviation.
    downside_diff = np.minimum(returns_arr, 0.0)  # clamp positives to 0
    tdd = np.sqrt(np.mean(downside_diff ** 2))
    if tdd > 0:
        result.sortino_ratio = round(
            np.sqrt(252) * returns_arr.mean() / tdd, 2
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

    return result, trade_log, returns_arr


# ── Normal distribution helpers (pure numpy, no scipy) ────────────────


def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    Standard normal inverse CDF (percent-point function).

    Uses the rational approximation from Abramowitz & Stegun (26.2.23)
    which is accurate to ~4.5e-4 for 0 < p < 1.
    """
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    if p == 0.5:
        return 0.0

    # Work in the upper tail
    if p > 0.5:
        return -_norm_ppf(1.0 - p)

    t = math.sqrt(-2.0 * math.log(p))
    # Rational approximation coefficients
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    return -(t - num / den)


def _skewness(arr: np.ndarray) -> float:
    """Sample skewness (Fisher definition)."""
    n = len(arr)
    if n < 3:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3) * n / ((n - 1) * (n - 2) / n))


def _kurtosis_excess(arr: np.ndarray) -> float:
    """Sample excess kurtosis (Fisher definition, normal = 0)."""
    n = len(arr)
    if n < 4:
        return 0.0
    m = arr.mean()
    s = arr.std(ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_strategies_tested: int,
    skewness: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR) from Bailey & de Prado (2014).

    Adjusts the observed Sharpe for multiple testing bias. When you test
    N strategies and pick the best, the expected max Sharpe under the null
    (all strategies are noise) grows with N. DSR gives the probability
    that the observed Sharpe exceeds this expected max.

    Returns a value between 0 and 1 -- think of it as a p-value-like
    measure. Higher = more likely the Sharpe is genuine.

    Args:
        observed_sharpe: The best Sharpe ratio found.
        n_observations: Number of return observations (trading days).
        n_strategies_tested: Number of strategies that were evaluated.
        skewness: Skewness of the return distribution.
        kurtosis_excess: Excess kurtosis of the return distribution.
    """
    if n_observations < 10 or n_strategies_tested < 2:
        return 1.0  # not enough data to penalize

    T = float(n_observations)
    N = float(n_strategies_tested)

    # Expected maximum Sharpe under the null (Euler-Mascheroni approx)
    euler_mascheroni = 0.5772156649
    z_N = _norm_ppf(1.0 - 1.0 / N)
    expected_max_sharpe = (
        z_N * (1.0 - euler_mascheroni)
        + euler_mascheroni * _norm_ppf(1.0 - 1.0 / (N * math.e))
    ) if N > 1 else 0.0

    # Variance of the Sharpe estimator accounting for non-normality
    # (Lo, 2002): Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt/4)*SR^2) / T
    sr = observed_sharpe
    var_sr = (
        1.0
        + 0.5 * sr * sr
        - skewness * sr
        + (kurtosis_excess / 4.0) * sr * sr
    ) / T

    if var_sr <= 0:
        return 1.0

    # DSR = P(SR* > E[max(SR)]) under the null
    # = Φ((SR* - E[max]) / σ(SR*))
    z_score = (sr - expected_max_sharpe) / math.sqrt(var_sr)
    dsr = _norm_cdf(z_score)

    return max(0.0, min(1.0, dsr))


# ── Scoring ───────────────────────────────────────────────────────────


def score_single_window(
    result: BacktestResult,
    risk_config: dict,
    n_strategies_tested: int = 14,
    returns_arr: np.ndarray | None = None,
) -> float:
    """
    Score a strategy on a single timeframe window.

    Uses calibrated weights with normalized inputs so all components
    contribute meaningfully to the final score. Also applies a
    Deflated Sharpe Ratio penalty to guard against multiple testing bias.

    Components (on a 0-100 target scale):
      - Sharpe ratio:           30%  (capped at 3.0, scaled to ~30 pts)
      - Raw excess return:      20%  (non-annualized, ~20 pts)
      - Win rate:               15%  (centered on 50%, ~15 pts)
      - Profit factor:          15%  (capped at 3.0, ~15 pts)
      - Drawdown penalty:       20%  (meaningful penalty for high DD)

    Multiplied by DSR confidence (0-1) if enough data exists.
    """
    score = 0.0

    # ── Sharpe ratio (30%) ─────────────────────────────────────────
    # Sharpe range ~0-3, mapped to 0-30 points
    score += min(max(result.sharpe_ratio, 0.0), 3.0) * 10.0

    # ── Excess return (20%) ─────────────────────────────────────────
    # Use RAW (non-annualized) excess return to avoid distortion from
    # short backtest windows.  Annualizing with 252/n_days amplifies
    # small differences into extreme values when n_days < 100 (e.g.
    # -15% raw → -73% annualized at 75 days).  Raw excess treats each
    # window on its own merit regardless of length.
    # Clip at ±25% and scale so 25% raw excess → 20 points max.
    excess_clipped = np.clip(result.excess_return_pct, -25, 25)
    score += excess_clipped * 0.8

    # ── Win rate (15%) ─────────────────────────────────────────────
    # Center on 50%: a 60% WR → +15, a 40% WR → -15
    score += (result.win_rate - 0.5) * 150

    # ── Profit factor (15%) ────────────────────────────────────────
    # PF range ~0-3, mapped to 0-15 points
    score += min(max(result.profit_factor, 0.0), 3.0) * 5.0

    # ── Drawdown penalty (20%) ─────────────────────────────────────
    # max_drawdown_pct is negative (e.g., -15.0)
    # -15% DD → -22.5 points; -40% DD → -60 points
    score += result.max_drawdown_pct * 1.5

    # ── Trade count reliability penalty ────────────────────────────
    # Smooth curve: fewer trades → heavier penalty to reflect
    # lower statistical confidence in the backtest metrics.
    # Note: DSR (applied below at ≥5 trades) already penalizes
    # the 5+ range, so no additional trade-count penalty is needed
    # above 5. Below 5, DSR is statistically unreliable so we use
    # these fixed multipliers instead.
    if result.total_trades < 2:
        score *= 0.3
    elif result.total_trades < 3:
        score *= 0.5
    elif result.total_trades < 5:
        score *= 0.7

    # ── Win rate below risk profile minimum ────────────────────────
    if result.win_rate < risk_config["min_win_rate"]:
        score *= 0.7

    # ── Deflated Sharpe Ratio penalty ──────────────────────────────
    # Penalizes strategies that may have high Sharpe by chance due
    # to testing 14 strategies × 3 timeframes = 42 hypotheses
    if result.sharpe_ratio > 0 and result.total_trades >= 5:
        skew = 0.0
        kurt_excess = 0.0
        if returns_arr is not None and len(returns_arr) > 10:
            skew = _skewness(returns_arr)
            kurt_excess = _kurtosis_excess(returns_arr)

        dsr = deflated_sharpe_ratio(
            observed_sharpe=result.sharpe_ratio,
            n_observations=result.trading_days,
            n_strategies_tested=n_strategies_tested,
            skewness=skew,
            kurtosis_excess=kurt_excess,
        )
        # Blend: if DSR is high (>0.8), barely penalize.
        # If DSR is low (<0.3), shrink score significantly.
        # Penalty range: score × [0.5, 1.0]
        dsr_multiplier = 0.5 + 0.5 * dsr
        score *= dsr_multiplier

    return round(score, 2)


# ── Triple barrier helper ─────────────────────────────────────────────


@dataclass
class _TBResult:
    """Internal result from triple-barrier run, including arrays for training."""
    trades: list
    sl_pct_arr: np.ndarray
    tp_pct_arr: np.ndarray
    val_df: pd.DataFrame


def _compute_dynamic_rr(val_df: pd.DataFrame) -> np.ndarray:
    """Compute per-bar reward/risk ratio matching live signal logic.

    Uses trend (SMA50 vs SMA200), ADX, and a default confidence
    level to mirror the RR computation in signals.py.
    """
    n = len(val_df)
    rr = np.full(n, 2.0)

    sma50 = val_df.get("SMA_50", pd.Series(np.zeros(n), index=val_df.index)).values
    sma200 = val_df.get("SMA_200", pd.Series(np.zeros(n), index=val_df.index)).values
    adx = val_df.get("ADX_14", pd.Series(np.zeros(n), index=val_df.index)).values

    for i in range(n):
        is_bullish = sma50[i] > sma200[i] if (sma50[i] > 0 and sma200[i] > 0) else False
        is_bearish = sma50[i] < sma200[i] if (sma50[i] > 0 and sma200[i] > 0) else False

        if is_bullish:
            rr[i] = 2.5
        elif is_bearish:
            rr[i] = 1.5

        # ADX boost for strong bullish trend
        if adx[i] > 30 and is_bullish:
            rr[i] = min(rr[i] + 0.5, 3.5)

    return rr


def _run_triple_barrier(
    val_df: pd.DataFrame,
    val_signals: pd.Series,
    config: dict,
) -> _TBResult | None:
    """Run triple-barrier labeling on a validation window.

    Computes per-bar ATR-based SL/TP with dynamic RR, applies
    CUSUM filter intersected with strategy signals, then runs the
    barrier engine.

    Returns a _TBResult with trades AND the SL/TP arrays used,
    so callers can pass them to meta-label training without leakage.
    """
    close = val_df["Close"].values
    high = val_df["High"].values
    low = val_df["Low"].values
    dates = val_df.index

    n = len(close)
    if n < 5:
        return None

    # Build per-bar SL% array
    atr_arr = val_df.get("ATR_14", pd.Series(np.zeros(n), index=dates)).values
    vol_arr = val_df.get("Volatility_20", pd.Series(np.full(n, 0.25), index=dates)).values
    sl_pct_arr = np.array([
        compute_atr_stop_loss_pct(
            float(atr_arr[i]), float(vol_arr[i]), float(close[i]),
        )
        for i in range(n)
    ])

    # Build per-bar TP% using dynamic RR (matches live signal logic)
    rr_arr = _compute_dynamic_rr(val_df)
    tp_pct_arr = sl_pct_arr * rr_arr

    # Entry mask: strategy BUY signals
    # Apply same next-bar execution shift as the main backtest:
    # signal on bar[i] enters at bar[i+1].
    use_next_bar = config.get("next_bar_execution", True)
    if use_next_bar:
        shifted = pd.Series(val_signals.values).shift(1, fill_value=0)
        sig_entries = np.asarray(shifted == 1, dtype=bool)
    else:
        sig_entries = np.asarray(val_signals == 1, dtype=bool)

    # CUSUM filter: intersect with strategy signals
    cusum_mult = config.get("cusum_mult", 0.5)
    # Compute adaptive threshold per-bar (use median ATR/price)
    median_atr = np.nanmedian(atr_arr[atr_arr > 0]) if np.any(atr_arr > 0) else 0
    median_price = np.nanmedian(close[close > 0]) if np.any(close > 0) else 1
    cusum_threshold = cusum_mult * median_atr / median_price
    if cusum_threshold > 0:
        cusum_events = cusum_filter(close, cusum_threshold)
        entries = sig_entries & cusum_events
    else:
        entries = sig_entries

    if not np.any(entries):
        return None

    # Max holding bars
    tb_mult = config.get("tb_max_holding_mult", 1.5)
    entry_positions = np.where(entries)[0]
    if len(entry_positions) > 1:
        avg_gap = np.mean(np.diff(entry_positions))
        max_holding = max(int(avg_gap * tb_mult), 5)
    else:
        max_holding = 20

    trades = apply_triple_barrier(
        close=close,
        high=high,
        low=low,
        dates=dates,
        entries=entries,
        sl_pct=sl_pct_arr,
        tp_pct=tp_pct_arr,
        max_holding_bars=max_holding,
        cost_bps=config.get("transaction_cost_bps", 10),
    )

    if not trades:
        return None

    return _TBResult(
        trades=trades,
        sl_pct_arr=sl_pct_arr,
        tp_pct_arr=tp_pct_arr,
        val_df=val_df,
    )


# ── Multi-timeframe strategy selector ─────────────────────────────────


def select_best_strategy(
    df: pd.DataFrame, ticker: str, config: dict
) -> Tuple[
    Strategy,
    BacktestResult,
    Dict[str, list],
    Dict[str, float],
    List[TradeRecord],
    list,
]:
    """
    Test all strategies across multiple timeframes using walk-forward
    validation.

    Walk-forward approach:
      Each window is split into a training portion (first 70%) and a
      validation portion (last 30%). Signals are generated on the full
      window (so indicators have warm-up data), but the BacktestResult
      metrics used for scoring come only from the validation portion.
      This prevents in-sample overfitting.

    Next-bar execution:
      Signals on bar[i] execute at close[i+1]. This eliminates the
      subtle look-ahead bias where you'd need to know bar[i]'s close
      to generate the signal AND trade at that same close.

    Returns
    -------
    best_strategy, best_result, per_window_results,
    composite_scores, all_trade_logs
    """
    risk_config = RISK_PROFILES[config["risk_profile"]]
    windows = config["backtest_windows"]
    weights = config["window_weights"]
    walk_forward_pct = config.get("walk_forward_validation_pct", 0.30)
    use_next_bar = config.get("next_bar_execution", True)

    n_strategies = len(ALL_STRATEGIES)

    strategy_windows: Dict[str, Dict[str, Tuple]] = {}
    per_window_results: Dict[str, list] = {}
    all_trade_logs: List[TradeRecord] = []
    # Collect barrier trades + context for meta-label training
    _tb_training_data: list[tuple[_TBResult, str, str]] = []

    for window_name, n_days in windows.items():
        window_df = df.tail(n_days).copy()
        if len(window_df) < 30:
            log.warning(
                f"  Skipping {window_name} for {ticker}: "
                f"only {len(window_df)} days"
            )
            continue

        # ── Walk-forward split ─────────────────────────────────────
        # Generate signals on the FULL window (so indicators like
        # SMA_200 have warm-up), but score only on the validation
        # (out-of-sample) portion.
        n_total = len(window_df)
        n_val = max(int(n_total * walk_forward_pct), 35)
        val_start_idx = n_total - n_val

        window_results = []

        for strategy in ALL_STRATEGIES:
            try:
                # Generate signals on full window
                signals = strategy.generate_signals(window_df)

                # Run backtest on VALIDATION portion only
                val_df = window_df.iloc[val_start_idx:].copy()
                val_signals = signals.iloc[val_start_idx:]

                result, trade_log, strat_returns = run_backtest(
                    val_df,
                    val_signals,
                    ticker,
                    strategy.name,
                    window_name,
                    config["transaction_cost_bps"],
                    long_only=config.get("long_only", True),
                    next_bar_execution=use_next_bar,
                )

                # Strategy return stream for DSR skew/kurtosis
                returns_for_dsr = (
                    strat_returns
                    if len(strat_returns) > 10
                    else None
                )

                sc = score_single_window(
                    result,
                    risk_config,
                    n_strategies_tested=(
                        n_strategies * len(windows)
                    ),
                    returns_arr=returns_for_dsr,
                )
                result.score = sc
                window_results.append((strategy, result, sc))
                all_trade_logs.extend(trade_log)

                # ── Triple barrier parallel path ──────────────
                if config.get("triple_barrier_enabled", False):
                    try:
                        tb_result = _run_triple_barrier(
                            val_df, val_signals, config,
                        )
                        if tb_result is not None:
                            metrics = score_barrier_trades(
                                tb_result.trades,
                            )
                            result.tb_win_rate = metrics.win_rate
                            result.tb_sl_rate = metrics.sl_rate
                            result.tb_timeout_rate = metrics.timeout_rate
                            result.tb_avg_winner_pct = metrics.avg_winner_pct
                            result.tb_avg_loser_pct = metrics.avg_loser_pct
                            result.tb_profit_factor = metrics.profit_factor
                            result.tb_total_trades = metrics.total_trades
                            result.tb_edge_ratio = metrics.edge_ratio
                            # Collect for meta-label training
                            _tb_training_data.append((
                                tb_result, ticker, strategy.name,
                            ))
                    except Exception as tb_err:
                        log.debug(
                            "  TB failed for %s/%s: %s",
                            strategy.name, window_name, tb_err,
                        )

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
        _tb_training_data,
    )


def train_meta_model_from_tb(
    tb_data: list[tuple],
    config: dict,
    ticker: str,
) -> None:
    """Train meta-label model from accumulated barrier trades.

    Should be called AFTER signal generation so that today's signal
    uses the model from the previous training cycle, preserving
    out-of-sample integrity.

    Uses per-ticker model paths to avoid races when multiple
    tickers are analysed in parallel via ProcessPoolExecutor.

    Parameters
    ----------
    tb_data : list of (_TBResult, ticker, strategy_name) tuples
        Collected during ``select_best_strategy()``.
    config : dict
        Must contain ``meta_label_model_dir``.
    ticker : str
        Ticker being analysed (used for per-ticker model path).
    """
    from quant_analysis_bot.meta_label import (
        build_training_data,
        load_meta_model,
        save_meta_model,
        should_promote,
        should_retrain,
        train_meta_model,
    )

    model_dir = config.get("meta_label_model_dir", "models")

    # Check retrain cadence (per-ticker)
    retrain_days = config.get("meta_label_retrain_days", 7)
    if not should_retrain(
        base_dir=model_dir, retrain_days=retrain_days, ticker=ticker,
    ):
        log.debug("Meta-label: skipping retrain for %s (within cadence)", ticker)
        return

    # Map strategy names to integer IDs
    strat_ids = {s.name: i for i, s in enumerate(ALL_STRATEGIES)}

    # Aggregate training data across all strategies/windows
    all_X = []
    all_y = []
    all_events = []

    for tb_result, tb_ticker, strat_name in tb_data:
        strategy_id = strat_ids.get(strat_name, 0)
        X, y, events = build_training_data(
            barrier_trades=tb_result.trades,
            df=tb_result.val_df,
            ticker=tb_ticker,
            strategy_id=strategy_id,
            sl_pct_arr=tb_result.sl_pct_arr,
            tp_pct_arr=tb_result.tp_pct_arr,
        )
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_events.append(events)

    if not all_X:
        return

    X_combined = np.concatenate(all_X)
    y_combined = np.concatenate(all_y)
    events_combined = pd.concat(all_events, ignore_index=True)

    log.info(
        "Meta-label: training on %d barrier trades for %s",
        len(y_combined), ticker,
    )

    trained = train_meta_model(X_combined, y_combined, events_combined)
    if trained is None:
        return

    # Promotion check against existing model
    old_model = load_meta_model(base_dir=model_dir, ticker=ticker)
    if old_model is not None:
        if not should_promote(trained, old_model, X_combined, y_combined):
            return

    save_meta_model(trained, base_dir=model_dir, ticker=ticker)
