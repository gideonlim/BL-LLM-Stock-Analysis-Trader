"""Triple Barrier labeling engine with CUSUM event filter.

Implements the triple-barrier method from López de Prado (2018):
  - Upper barrier (TP): price hits take-profit → label = +1
  - Lower barrier (SL): price hits stop-loss  → label = -1
  - Vertical barrier (time): max holding period expires → label = sign(return)

The CUSUM filter reduces entry events to structurally meaningful
price moves, cutting label overlap by ~60-80%.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class BarrierTrade:
    """One completed triple-barrier trade."""

    entry_idx: int
    entry_date: str
    entry_price: float
    exit_idx: int
    exit_date: str
    exit_price: float
    holding_days: int
    return_pct: float
    label: int              # +1 (TP hit), -1 (SL hit), 0 (vertical, negligible)
    exit_barrier: str       # "upper", "lower", "vertical"
    # Excursion data (for meta-label features)
    mfe_pct: float          # max favorable excursion %
    mae_pct: float          # max adverse excursion %
    mfe_bar: int            # bars to MFE
    mae_bar: int            # bars to MAE


def cusum_filter(
    close: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return boolean mask of bars where symmetric CUSUM triggers.

    Tracks cumulative positive and negative deviations of log-returns.
    Fires when either exceeds ``threshold``, then resets.

    Parameters
    ----------
    close : np.ndarray
        Close prices.
    threshold : float
        CUSUM trigger level (e.g. ATR * 0.5 / price).

    Returns
    -------
    np.ndarray
        Boolean mask, True on bars where CUSUM fires.
    """
    if len(close) < 2 or threshold <= 0:
        return np.zeros(len(close), dtype=bool)

    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    s_pos = 0.0
    s_neg = 0.0
    events = np.zeros(len(close), dtype=bool)

    for i in range(1, len(close)):
        s_pos = max(0.0, s_pos + log_returns[i])
        s_neg = min(0.0, s_neg + log_returns[i])
        if s_pos > threshold:
            events[i] = True
            s_pos = 0.0
        elif s_neg < -threshold:
            events[i] = True
            s_neg = 0.0

    return events


def apply_triple_barrier(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    dates: pd.DatetimeIndex,
    entries: np.ndarray,
    sl_pct: np.ndarray | float,
    tp_pct: np.ndarray | float,
    max_holding_bars: int = 20,
    cost_bps: float = 10,
) -> list[BarrierTrade]:
    """Apply triple-barrier labeling to a set of entry signals.

    For each entry bar, walks forward checking High/Low against
    the take-profit (upper) and stop-loss (lower) barriers.  When
    both barriers are breached on the same bar, the stop-loss wins
    (conservative assumption matching real slippage behaviour).

    Parameters
    ----------
    close, high, low : np.ndarray
        OHLC price arrays (same length).
    dates : pd.DatetimeIndex
        Corresponding dates.
    entries : np.ndarray
        Boolean mask — True on bars with an entry signal.
    sl_pct : float or np.ndarray
        Stop-loss as percentage of entry price (e.g. 5.0 = 5%).
        Scalar for uniform, or per-bar array.
    tp_pct : float or np.ndarray
        Take-profit as percentage of entry price.
    max_holding_bars : int
        Vertical barrier — maximum bars to hold before forced exit.
    cost_bps : float
        Round-trip transaction cost in basis points.

    Returns
    -------
    list[BarrierTrade]
        One record per completed trade.
    """
    n = len(close)
    if n == 0:
        return []

    cost = cost_bps / 10000
    sl_arr = np.full(n, sl_pct) if np.isscalar(sl_pct) else np.asarray(sl_pct)
    tp_arr = np.full(n, tp_pct) if np.isscalar(tp_pct) else np.asarray(tp_pct)
    entries = np.asarray(entries, dtype=bool)

    trades: list[BarrierTrade] = []

    entry_indices = np.where(entries)[0]

    for entry_idx in entry_indices:
        entry_price = close[entry_idx]
        if entry_price <= 0:
            continue

        sl_frac = sl_arr[entry_idx] / 100.0
        tp_frac = tp_arr[entry_idx] / 100.0

        sl_price = entry_price * (1 - sl_frac)
        tp_price = entry_price * (1 + tp_frac)

        end_idx = min(entry_idx + max_holding_bars, n - 1)

        # Walk forward checking barriers
        exit_idx = end_idx
        exit_barrier = "vertical"
        exit_price = close[end_idx]

        # Track excursion
        mfe = 0.0  # max favorable excursion (price above entry)
        mae = 0.0  # max adverse excursion (price below entry)
        mfe_bar = 0
        mae_bar = 0

        for j in range(entry_idx + 1, end_idx + 1):
            # Update excursion tracking using High/Low
            fav = (high[j] - entry_price) / entry_price
            adv = (entry_price - low[j]) / entry_price
            if fav > mfe:
                mfe = fav
                mfe_bar = j - entry_idx
            if adv > mae:
                mae = adv
                mae_bar = j - entry_idx

            # Check barriers using High/Low (intrabar)
            hit_tp = high[j] >= tp_price
            hit_sl = low[j] <= sl_price

            if hit_sl and hit_tp:
                # Both hit same bar — SL wins (conservative)
                exit_idx = j
                exit_barrier = "lower"
                exit_price = sl_price
                break
            elif hit_sl:
                exit_idx = j
                exit_barrier = "lower"
                exit_price = sl_price
                break
            elif hit_tp:
                exit_idx = j
                exit_barrier = "upper"
                exit_price = tp_price
                break

        # Compute return
        raw_return = (exit_price / entry_price) - 1 - cost
        return_pct = round(raw_return * 100, 4)

        # Compute holding days
        holding_days = (dates[exit_idx] - dates[entry_idx]).days

        # Assign label
        if exit_barrier == "upper":
            label = 1
        elif exit_barrier == "lower":
            label = -1
        else:
            # Vertical: sign of return, 0 if negligible (<= cost)
            if abs(raw_return) <= cost:
                label = 0
            else:
                label = 1 if raw_return > 0 else -1

        trades.append(
            BarrierTrade(
                entry_idx=entry_idx,
                entry_date=str(dates[entry_idx].date()),
                entry_price=round(entry_price, 4),
                exit_idx=exit_idx,
                exit_date=str(dates[exit_idx].date()),
                exit_price=round(exit_price, 4),
                holding_days=holding_days,
                return_pct=return_pct,
                label=label,
                exit_barrier=exit_barrier,
                mfe_pct=round(mfe * 100, 4),
                mae_pct=round(mae * 100, 4),
                mfe_bar=mfe_bar,
                mae_bar=mae_bar,
            )
        )

    return trades


@dataclass
class BarrierMetrics:
    """Aggregated metrics from triple-barrier trades."""

    total_trades: int = 0
    win_rate: float = 0.0       # % hitting TP
    sl_rate: float = 0.0        # % hitting SL
    timeout_rate: float = 0.0   # % hitting vertical barrier
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    profit_factor: float = 0.0
    edge_ratio: float = 0.0     # avg MFE / avg MAE


def score_barrier_trades(trades: list[BarrierTrade]) -> BarrierMetrics:
    """Compute aggregate metrics from a list of barrier trades.

    Parameters
    ----------
    trades : list[BarrierTrade]
        Completed barrier trades from ``apply_triple_barrier()``.

    Returns
    -------
    BarrierMetrics
        Aggregate statistics.
    """
    if not trades:
        return BarrierMetrics()

    n = len(trades)
    tp_trades = [t for t in trades if t.exit_barrier == "upper"]
    sl_trades = [t for t in trades if t.exit_barrier == "lower"]
    vb_trades = [t for t in trades if t.exit_barrier == "vertical"]

    winners = [t.return_pct for t in trades if t.return_pct > 0]
    losers = [t.return_pct for t in trades if t.return_pct <= 0]

    total_wins = sum(winners) if winners else 0.0
    total_losses = abs(sum(losers)) if losers else 0.001

    mfe_vals = [t.mfe_pct for t in trades if t.mfe_pct > 0]
    mae_vals = [t.mae_pct for t in trades if t.mae_pct > 0]
    avg_mfe = np.mean(mfe_vals) if mfe_vals else 0.0
    avg_mae = np.mean(mae_vals) if mae_vals else 0.001

    return BarrierMetrics(
        total_trades=n,
        win_rate=round(len(tp_trades) / n, 4),
        sl_rate=round(len(sl_trades) / n, 4),
        timeout_rate=round(len(vb_trades) / n, 4),
        avg_winner_pct=round(np.mean(winners), 4) if winners else 0.0,
        avg_loser_pct=round(np.mean(losers), 4) if losers else 0.0,
        profit_factor=round(total_wins / total_losses, 4),
        edge_ratio=round(avg_mfe / avg_mae, 4),
    )
