"""Shared take-profit / reward-risk / reachability-cap logic.

Single source of truth for TP computation used by both ``signals.py``
(live daily signal generation) and ``backtest.py`` (triple-barrier
simulation).  Before this module existed, the two call sites had
subtly different RR ladders that drifted over time.  Both now call
``compute_rr_ratio`` which selects the right ladder based on whether
the caller supplies a ``confidence_score`` (live) or ``None`` (backtest).

See ``research/tp_experiment.py`` for the experiment that motivated
the reachability cap and the ``tp_mode`` parameter.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Optional

log = logging.getLogger(__name__)

TpMode = Literal["current", "capped", "capped+strategy"]
StrategyFamily = Literal["mean_reversion", "trend_following"]


# ── Strategy family classification ──────────────────────────────────────
#
# Names must match the ``Strategy.name`` class attributes in
# ``strategies.py``.  The drift-guard test in
# ``tests/test_tp_logic.py`` asserts every entry is present in
# ``strategies.ALL_STRATEGIES`` so a rename fails loudly.
_MEAN_REVERSION_STRATEGIES: frozenset[str] = frozenset({
    "RSI Mean Reversion",
    "Bollinger Band Mean Reversion",
    "Z-Score Mean Reversion",
    # Stochastic Oscillator is mechanically a zone-reversal strategy
    # (buy in oversold, sell in overbought) so it's classified as
    # mean reversion for TP-cap purposes even though its name
    # doesn't include "Mean Reversion".
    "Stochastic Oscillator",
})


def classify_strategy_family(strategy_name: str) -> StrategyFamily:
    """Classify a registered strategy into a family for TP-cap logic.

    Unknown names default to ``trend_following`` (conservative:
    we don't shrink RR on strategies we haven't thought about).
    """
    if strategy_name in _MEAN_REVERSION_STRATEGIES:
        return "mean_reversion"
    return "trend_following"


# ── Reachability (expected max move) ────────────────────────────────────


def compute_expected_max_move_pct(
    vol_20: float,
    holding_days: float,
) -> float:
    """Expected 1-sigma price move over ``holding_days``, in percent.

    Uses the Brownian-motion assumption from quantitative finance:
    ``daily_vol = vol_20 / sqrt(252)``, then scale by sqrt(T).
    The result is a theoretical 1-sigma band (~68% CI); callers
    multiply it by ``cap_multiplier`` to get the actual TP cap.

    Parameters
    ----------
    vol_20 : float
        Annualised 20-day realised volatility (e.g. 0.25 = 25%).
        Values ≤ 0 or NaN return ``inf`` (pass-through, no cap).
    holding_days : float
        Horizon to project the move over, in trading days.  Values
        ≤ 0 return ``inf`` (pass-through, no cap).
    """
    if vol_20 is None:
        return float("inf")
    try:
        vol_f = float(vol_20)
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(vol_f) or vol_f <= 0.0:
        return float("inf")

    if holding_days is None:
        return float("inf")
    try:
        horizon = float(holding_days)
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(horizon) or horizon <= 0.0:
        return float("inf")

    daily_vol_pct = (vol_f / math.sqrt(252.0)) * 100.0
    return daily_vol_pct * math.sqrt(horizon)


# ── Reward-risk ratio ───────────────────────────────────────────────────


def compute_rr_ratio(
    trend: str,
    adx: float,
    confidence_score: Optional[int],
    family: StrategyFamily,
    tp_mode: TpMode,
) -> float:
    """Compute the dynamic reward/risk ratio.

    Two schedules are preserved so both historical call sites are
    bit-identical under ``tp_mode="current"``:

    * ``confidence_score is None`` → **backtest schedule**
      (matches today's ``_compute_dynamic_rr`` in ``backtest.py``
      which has no confidence signal available).  Ladder:
      bullish→2.5, bearish→1.5, else 2.0; bullish+ADX>30 adds 0.5
      capped at 3.5.
    * ``confidence_score is int`` → **live schedule**
      (matches today's TP block in ``signals.py``).  Ladder:
      bullish+conf≥4→3.0, bearish OR conf≤1→1.5, else 2.0;
      bullish+ADX>30 adds 0.5 capped at 3.5.

    When ``tp_mode == "capped+strategy"`` and the strategy is in
    the mean-reversion family, the final ratio is clamped at 1.5
    (mean reversion doesn't benefit from letting winners run, so
    a tighter RR captures closer targets before snap-back reverses).

    Parameters
    ----------
    trend : str
        ``"BULLISH"`` | ``"BEARISH"`` | ``"NEUTRAL"`` (or anything
        else, which is treated as NEUTRAL).
    adx : float
        Current ADX value.  Values > 30 with bullish trend add the
        ADX boost.
    confidence_score : int | None
        ``None`` selects the backtest ladder; any int selects the
        live ladder.
    family : StrategyFamily
        ``"mean_reversion"`` | ``"trend_following"``.  Only used
        when ``tp_mode == "capped+strategy"``.
    tp_mode : TpMode
        ``"current"`` | ``"capped"`` | ``"capped+strategy"``.
    """
    is_bullish = trend == "BULLISH"
    is_bearish = trend == "BEARISH"

    if confidence_score is None:
        # Backtest schedule — no confidence signal.
        if is_bullish:
            rr = 2.5
        elif is_bearish:
            rr = 1.5
        else:
            rr = 2.0
    else:
        # Live schedule — uses confidence_score.
        if is_bullish and confidence_score >= 4:
            rr = 3.0
        elif is_bearish or confidence_score <= 1:
            rr = 1.5
        else:
            rr = 2.0

    # ADX boost: very strong bullish trend bumps ratio by 0.5.
    # Same in both schedules.
    if adx is not None and adx > 30 and is_bullish:
        rr = min(rr + 0.5, 3.5)

    # Strategy-family clamp (only in capped+strategy mode).
    if tp_mode == "capped+strategy" and family == "mean_reversion":
        rr = min(rr, 1.5)

    return rr


# ── TP cap ──────────────────────────────────────────────────────────────


def apply_tp_cap(
    tp_pct: float,
    expected_max_move_pct: float,
    cap_multiplier: float,
    tp_mode: TpMode,
) -> float:
    """Cap a raw TP% at ``cap_multiplier × expected_max_move_pct``.

    Pass-through when:
    * ``tp_mode == "current"``
    * ``expected_max_move_pct`` is infinite (vol/horizon unavailable)
    * ``cap_multiplier`` is non-finite or non-positive (misconfiguration
      — a zero/negative cap would produce a TP ≤ 0, which would fire
      immediately at entry and corrupt live trading and backtests)

    Otherwise returns ``min(tp_pct, cap_multiplier * expected_max_move_pct)``.
    Never raises a TP — caps only shrink.
    """
    if tp_mode == "current":
        return tp_pct
    if not math.isfinite(expected_max_move_pct):
        return tp_pct
    # Guard against misconfigured cap_multiplier. A value ≤ 0 would
    # produce cap ≤ 0 and collapse TPs to zero/negative, which would
    # make every trade fire the TP at entry. Fall back to pass-through
    # (equivalent to "current" mode for this call) and warn loudly so
    # the misconfiguration is visible in logs.
    if not math.isfinite(cap_multiplier) or cap_multiplier <= 0.0:
        log.warning(
            "apply_tp_cap: invalid cap_multiplier=%r "
            "(must be positive and finite). Falling back to "
            "uncapped TP for this bar.",
            cap_multiplier,
        )
        return tp_pct
    cap = cap_multiplier * expected_max_move_pct
    if tp_pct <= cap:
        return tp_pct
    return cap
