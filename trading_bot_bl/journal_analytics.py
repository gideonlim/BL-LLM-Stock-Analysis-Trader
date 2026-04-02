"""Journal analytics -- compute performance metrics from closed trades.

Pure computation, no side effects, no I/O beyond what callers pass in.
All functions operate on lists of JournalEntry and EquitySnapshot.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from trading_bot_bl.models import EquitySnapshot, JournalEntry

log = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────


@dataclass
class OverallMetrics:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0


@dataclass
class RiskAdjustedMetrics:
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    probabilistic_sharpe: float = 0.0
    min_track_record_length: int = 0
    daily_return_count: int = 0


@dataclass
class DrawdownMetrics:
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    time_in_drawdown_pct: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class ExecutionMetrics:
    avg_entry_slippage_pct: float = 0.0
    avg_entry_slippage_dollars: float = 0.0
    avg_exit_slippage_dollars: float = 0.0
    total_implementation_shortfall: float = 0.0
    slippage_pct_of_pnl: float = 0.0


@dataclass
class RDistribution:
    mean_r: float = 0.0
    median_r: float = 0.0
    std_r: float = 0.0
    skewness_r: float = 0.0
    pct_above_2r: float = 0.0
    pct_above_3r: float = 0.0
    pct_below_neg1r: float = 0.0


@dataclass
class ExcursionMetrics:
    avg_mae_winners: float = 0.0
    avg_mae_losers: float = 0.0
    avg_mfe_winners: float = 0.0
    avg_mfe_losers: float = 0.0
    avg_etd_winners: float = 0.0
    avg_etd_losers: float = 0.0
    avg_edge_ratio: float = 0.0
    avg_edge_ratio_winners: float = 0.0
    avg_time_to_mfe_days: float = 0.0
    avg_time_to_mae_days: float = 0.0


@dataclass
class HoldingMetrics:
    avg_hold_all: float = 0.0
    avg_hold_winners: float = 0.0
    avg_hold_losers: float = 0.0
    hold_return_correlation: float = 0.0
    time_exit_rate: float = 0.0
    exit_reason_distribution: dict = field(default_factory=dict)


@dataclass
class StreakMetrics:
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    current_streak_type: str = ""  # "win" or "loss"
    expected_max_losing_streak: int = 0


@dataclass
class ConfidenceMetrics:
    probabilistic_sharpe: float = 0.0
    min_track_record_length: int = 0
    trades_until_significant: int = 0
    is_statistically_significant: bool = False


@dataclass
class StrategyBreakdown:
    strategy: str = ""
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_edge_ratio: float = 0.0
    avg_holding_days: float = 0.0
    avg_etd_pct: float = 0.0
    total_pnl: float = 0.0


@dataclass
class RegimeBreakdown:
    regime: str = ""
    trade_count: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0


@dataclass
class JournalMetrics:
    """Complete metrics report."""

    overall: OverallMetrics = field(
        default_factory=OverallMetrics
    )
    risk_adjusted: RiskAdjustedMetrics = field(
        default_factory=RiskAdjustedMetrics
    )
    drawdown: DrawdownMetrics = field(
        default_factory=DrawdownMetrics
    )
    execution: ExecutionMetrics = field(
        default_factory=ExecutionMetrics
    )
    r_distribution: RDistribution = field(
        default_factory=RDistribution
    )
    excursion: ExcursionMetrics = field(
        default_factory=ExcursionMetrics
    )
    holding: HoldingMetrics = field(
        default_factory=HoldingMetrics
    )
    streaks: StreakMetrics = field(
        default_factory=StreakMetrics
    )
    confidence: ConfidenceMetrics = field(
        default_factory=ConfidenceMetrics
    )
    by_strategy: dict[str, StrategyBreakdown] = field(
        default_factory=dict
    )
    by_regime: dict[str, RegimeBreakdown] = field(
        default_factory=dict
    )
    computed_at: str = ""


# ── Main entry point ──────────────────────────────────────────────


def compute_journal_metrics(
    trades: list[JournalEntry],
    equity_snapshots: list[EquitySnapshot] | None = None,
    group_by_strategy: bool = True,
    group_by_regime: bool = True,
) -> JournalMetrics:
    """Compute all performance metrics from closed trades.

    Args:
        trades: All journal entries (open + closed).  Only closed
            entries are used for trade-level metrics.
        equity_snapshots: Portfolio snapshots for Sharpe/drawdown.
        group_by_strategy: Compute per-strategy breakdowns.
        group_by_regime: Compute per-regime breakdowns.

    Returns:
        JournalMetrics with all computed fields.
    """
    closed = [t for t in trades if t.status == "closed"]
    snapshots = equity_snapshots or []

    metrics = JournalMetrics(
        computed_at=datetime.now().isoformat(timespec="seconds"),
    )

    if not closed:
        return metrics

    metrics.overall = _compute_overall(closed)
    metrics.r_distribution = _compute_r_distribution(closed)
    metrics.excursion = _compute_excursion(closed)
    metrics.execution = _compute_execution(closed)
    metrics.holding = _compute_holding(closed)
    metrics.streaks = _compute_streaks(closed)

    if snapshots:
        metrics.risk_adjusted = _compute_risk_adjusted(snapshots)
        metrics.drawdown = _compute_drawdown(snapshots)
        metrics.confidence = _compute_confidence(
            snapshots, len(closed)
        )

    if group_by_strategy:
        metrics.by_strategy = _compute_by_strategy(closed)

    if group_by_regime:
        metrics.by_regime = _compute_by_regime(closed)

    return metrics


# ── Sub-computations ──────────────────────────────────────────────


def _compute_overall(closed: list[JournalEntry]) -> OverallMetrics:
    m = OverallMetrics()
    m.total_trades = len(closed)

    pnls = [t.realized_pnl or 0.0 for t in closed]
    r_multiples = [t.r_multiple for t in closed]

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    breakeven = [p for p in pnls if p == 0]

    m.wins = len(wins)
    m.losses = len(losses)
    m.breakeven = len(breakeven)
    m.total_pnl = round(sum(pnls), 2)

    if m.total_trades > 0:
        m.win_rate = round(m.wins / m.total_trades, 4)

    if wins:
        m.avg_win = round(sum(wins) / len(wins), 2)
        m.largest_win = round(max(wins), 2)
    if losses:
        m.avg_loss = round(sum(losses) / len(losses), 2)
        m.largest_loss = round(min(losses), 2)

    if m.avg_loss != 0:
        m.win_loss_ratio = round(
            abs(m.avg_win / m.avg_loss), 2
        )

    # Expectancy
    loss_rate = 1 - m.win_rate
    m.expectancy = round(
        m.win_rate * m.avg_win + loss_rate * m.avg_loss, 2
    )

    # Expectancy in R
    valid_r = [r for r in r_multiples if r is not None]
    if valid_r:
        m.expectancy_r = round(
            sum(valid_r) / len(valid_r), 4
        )

    # Profit factor
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses)) if losses else 0
    if gross_losses > 0:
        m.profit_factor = round(gross_wins / gross_losses, 2)
    elif gross_wins > 0:
        m.profit_factor = float("inf")

    return m


def _compute_r_distribution(
    closed: list[JournalEntry],
) -> RDistribution:
    r = RDistribution()
    r_vals = [t.r_multiple for t in closed if t.r_multiple is not None]
    if not r_vals:
        return r

    n = len(r_vals)
    r.mean_r = round(sum(r_vals) / n, 4)
    sorted_r = sorted(r_vals)
    r.median_r = round(
        sorted_r[n // 2]
        if n % 2 == 1
        else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2,
        4,
    )

    if n > 1:
        mean = r.mean_r
        variance = sum((x - mean) ** 2 for x in r_vals) / (n - 1)
        r.std_r = round(math.sqrt(variance), 4)

        # Skewness
        if r.std_r > 0:
            r.skewness_r = round(
                sum((x - mean) ** 3 for x in r_vals)
                / (n * r.std_r ** 3),
                4,
            )

    r.pct_above_2r = round(
        sum(1 for x in r_vals if x > 2) / n, 4
    )
    r.pct_above_3r = round(
        sum(1 for x in r_vals if x > 3) / n, 4
    )
    r.pct_below_neg1r = round(
        sum(1 for x in r_vals if x < -1) / n, 4
    )

    return r


def _compute_excursion(
    closed: list[JournalEntry],
) -> ExcursionMetrics:
    e = ExcursionMetrics()
    winners = [t for t in closed if (t.realized_pnl or 0) > 0]
    losers = [t for t in closed if (t.realized_pnl or 0) < 0]

    if winners:
        e.avg_mae_winners = round(
            _safe_mean([t.mae_pct for t in winners]), 2
        )
        e.avg_mfe_winners = round(
            _safe_mean([t.mfe_pct for t in winners]), 2
        )
        e.avg_etd_winners = round(
            _safe_mean([t.etd_pct for t in winners]), 2
        )
        e.avg_edge_ratio_winners = round(
            _safe_mean([
                t.edge_ratio for t in winners
                if (t.edge_ratio or 0) > 0
            ]),
            2,
        )
    if losers:
        e.avg_mae_losers = round(
            _safe_mean([t.mae_pct for t in losers]), 2
        )
        e.avg_mfe_losers = round(
            _safe_mean([t.mfe_pct for t in losers]), 2
        )
        e.avg_etd_losers = round(
            _safe_mean([t.etd_pct for t in losers]), 2
        )

    all_er = [
        t.edge_ratio for t in closed
        if (t.edge_ratio or 0) > 0
    ]
    if all_er:
        e.avg_edge_ratio = round(_safe_mean(all_er), 2)

    # Time to MFE/MAE
    mfe_days = []
    mae_days = []
    for t in closed:
        if t.mfe_date and t.entry_date:
            try:
                d = _days_between(t.entry_date, t.mfe_date)
                if d >= 0:
                    mfe_days.append(d)
            except Exception:
                pass
        if t.mae_date and t.entry_date:
            try:
                d = _days_between(t.entry_date, t.mae_date)
                if d >= 0:
                    mae_days.append(d)
            except Exception:
                pass

    if mfe_days:
        e.avg_time_to_mfe_days = round(_safe_mean(mfe_days), 1)
    if mae_days:
        e.avg_time_to_mae_days = round(_safe_mean(mae_days), 1)

    return e


def _compute_execution(
    closed: list[JournalEntry],
) -> ExecutionMetrics:
    e = ExecutionMetrics()
    n = len(closed)
    if not n:
        return e

    entry_slips_pct = [t.entry_slippage_pct for t in closed]
    entry_slips_dollar = [t.entry_slippage for t in closed]
    exit_slips = [t.exit_slippage for t in closed]

    e.avg_entry_slippage_pct = round(
        _safe_mean(entry_slips_pct), 4
    )
    e.avg_entry_slippage_dollars = round(
        _safe_mean(entry_slips_dollar), 4
    )
    e.avg_exit_slippage_dollars = round(
        _safe_mean(exit_slips), 4
    )

    total_slip = sum(
        abs(t.entry_slippage or 0) + abs(t.exit_slippage or 0)
        for t in closed
    )
    e.total_implementation_shortfall = round(total_slip, 2)

    total_abs_pnl = sum(abs(t.realized_pnl or 0) for t in closed)
    if total_abs_pnl > 0:
        e.slippage_pct_of_pnl = round(
            total_slip / total_abs_pnl * 100, 2
        )

    return e


def _compute_holding(
    closed: list[JournalEntry],
) -> HoldingMetrics:
    h = HoldingMetrics()
    winners = [t for t in closed if (t.realized_pnl or 0) > 0]
    losers = [t for t in closed if (t.realized_pnl or 0) < 0]

    all_days = [t.holding_days for t in closed if (t.holding_days or 0) > 0]
    win_days = [t.holding_days for t in winners if (t.holding_days or 0) > 0]
    loss_days = [t.holding_days for t in losers if (t.holding_days or 0) > 0]

    if all_days:
        h.avg_hold_all = round(_safe_mean(all_days), 1)
    if win_days:
        h.avg_hold_winners = round(_safe_mean(win_days), 1)
    if loss_days:
        h.avg_hold_losers = round(_safe_mean(loss_days), 1)

    # Correlation between hold time and return
    if len(closed) > 2:
        days_list = [float(t.holding_days or 0) for t in closed]
        pnl_list = [t.realized_pnl_pct or 0 for t in closed]
        h.hold_return_correlation = round(
            _pearson_corr(days_list, pnl_list), 4
        )

    # Time exit rate
    time_exits = sum(
        1 for t in closed if t.exit_reason == "time_exit"
    )
    if closed:
        h.time_exit_rate = round(time_exits / len(closed), 4)

    # Exit reason distribution
    reasons: dict[str, int] = defaultdict(int)
    for t in closed:
        reasons[t.exit_reason or "unknown"] += 1
    h.exit_reason_distribution = dict(reasons)

    return h


def _compute_streaks(
    closed: list[JournalEntry],
) -> StreakMetrics:
    s = StreakMetrics()
    if not closed:
        return s

    # Sort by exit date
    sorted_trades = sorted(
        closed,
        key=lambda t: t.closed_at or t.exit_date or "",
    )

    outcomes = [
        "win" if (t.realized_pnl or 0) > 0 else "loss"
        for t in sorted_trades
    ]

    max_wins = 0
    max_losses = 0
    current_run = 1

    for i in range(1, len(outcomes)):
        if outcomes[i] == outcomes[i - 1]:
            current_run += 1
        else:
            if outcomes[i - 1] == "win":
                max_wins = max(max_wins, current_run)
            else:
                max_losses = max(max_losses, current_run)
            current_run = 1

    # Final run
    if outcomes:
        if outcomes[-1] == "win":
            max_wins = max(max_wins, current_run)
        else:
            max_losses = max(max_losses, current_run)

    s.max_consecutive_wins = max_wins
    s.max_consecutive_losses = max_losses

    # Current streak
    if outcomes:
        streak = 1
        for i in range(len(outcomes) - 2, -1, -1):
            if outcomes[i] == outcomes[-1]:
                streak += 1
            else:
                break
        s.current_streak = streak
        s.current_streak_type = outcomes[-1]

    # Expected max losing streak
    n = len(closed)
    loss_rate = 1 - (
        sum(1 for t in closed if (t.realized_pnl or 0) > 0) / n
    )
    if 0 < loss_rate < 1 and n > 0:
        s.expected_max_losing_streak = max(
            1,
            round(math.log(n) / math.log(1 / loss_rate)),
        )

    return s


def _compute_risk_adjusted(
    snapshots: list[EquitySnapshot],
) -> RiskAdjustedMetrics:
    m = RiskAdjustedMetrics()
    if len(snapshots) < 2:
        return m

    # Daily returns from equity curve
    equities = [s.equity for s in snapshots if s.equity > 0]
    if len(equities) < 2:
        return m

    daily_returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
    ]
    m.daily_return_count = len(daily_returns)

    if not daily_returns:
        return m

    mean_ret = sum(daily_returns) / len(daily_returns)
    std_ret = _std(daily_returns)

    # Sharpe (annualised)
    if std_ret > 1e-10:
        m.sharpe_ratio = round(
            mean_ret / std_ret * math.sqrt(252), 4
        )

    # Sortino (downside deviation)
    downside = [r for r in daily_returns if r < 0]
    if downside:
        downside_std = _std(downside)
        if downside_std > 1e-10:
            m.sortino_ratio = round(
                mean_ret / downside_std * math.sqrt(252), 4
            )

    # Calmar
    max_dd_pct = max(
        (s.drawdown_pct for s in snapshots), default=0
    )
    if max_dd_pct > 0 and len(equities) > 1:
        total_return = (equities[-1] - equities[0]) / equities[0]
        # Rough annualise
        days = len(equities)
        annual_return = total_return * (252 / max(days, 1))
        m.calmar_ratio = round(
            annual_return / (max_dd_pct / 100), 4
        )

    # Probabilistic Sharpe Ratio
    if len(daily_returns) > 3 and std_ret > 1e-10:
        n = len(daily_returns)
        skew = _skewness(daily_returns)
        kurt = _kurtosis(daily_returns)
        psr = _probabilistic_sharpe(
            m.sharpe_ratio / math.sqrt(252),  # de-annualise
            0.0,
            n,
            skew,
            kurt,
        )
        m.probabilistic_sharpe = round(psr, 4)

        mtrl = _min_track_record_length(
            m.sharpe_ratio / math.sqrt(252),
            0.0,
            skew,
            kurt,
        )
        m.min_track_record_length = mtrl

    return m


def _compute_drawdown(
    snapshots: list[EquitySnapshot],
) -> DrawdownMetrics:
    d = DrawdownMetrics()
    if not snapshots:
        return d

    dds = [s.drawdown_pct for s in snapshots]
    d.max_drawdown_pct = round(max(dds), 2) if dds else 0
    d.current_drawdown_pct = round(dds[-1], 2) if dds else 0

    dd_values = [s for s in snapshots if s.drawdown_pct > 0]
    if dd_values:
        d.avg_drawdown_pct = round(
            _safe_mean([s.drawdown_pct for s in dd_values]), 2
        )

    if snapshots:
        d.time_in_drawdown_pct = round(
            len(dd_values) / len(snapshots) * 100, 2
        )

    # Max drawdown in dollars
    d.max_drawdown_dollars = round(
        max(
            (s.high_water_mark - s.equity for s in snapshots),
            default=0,
        ),
        2,
    )

    # Max drawdown duration (consecutive snapshots in DD)
    max_dur = 0
    cur_dur = 0
    for s in snapshots:
        if s.drawdown_pct > 0:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    d.max_drawdown_duration_days = max_dur

    # Recovery factor
    total_pnl = 0.0
    if len(snapshots) > 1:
        total_pnl = snapshots[-1].equity - snapshots[0].equity
    if d.max_drawdown_dollars > 0:
        d.recovery_factor = round(
            total_pnl / d.max_drawdown_dollars, 2
        )

    return d


def _compute_confidence(
    snapshots: list[EquitySnapshot],
    trade_count: int,
) -> ConfidenceMetrics:
    c = ConfidenceMetrics()
    equities = [s.equity for s in snapshots if s.equity > 0]
    if len(equities) < 5:
        return c

    daily_returns = [
        (equities[i] - equities[i - 1]) / equities[i - 1]
        for i in range(1, len(equities))
    ]
    n = len(daily_returns)
    if n < 3:
        return c

    std_ret = _std(daily_returns)
    if std_ret < 1e-10:
        return c

    mean_ret = sum(daily_returns) / n
    sr = mean_ret / std_ret  # daily Sharpe
    skew = _skewness(daily_returns)
    kurt = _kurtosis(daily_returns)

    c.probabilistic_sharpe = round(
        _probabilistic_sharpe(sr, 0.0, n, skew, kurt), 4
    )
    c.min_track_record_length = _min_track_record_length(
        sr, 0.0, skew, kurt
    )
    c.trades_until_significant = max(
        0, c.min_track_record_length - n
    )
    c.is_statistically_significant = n >= c.min_track_record_length

    return c


def _compute_by_strategy(
    closed: list[JournalEntry],
) -> dict[str, StrategyBreakdown]:
    groups: dict[str, list[JournalEntry]] = defaultdict(list)
    for t in closed:
        groups[t.strategy or "unknown"].append(t)

    result: dict[str, StrategyBreakdown] = {}
    for name, trades in groups.items():
        b = StrategyBreakdown(strategy=name)
        b.trade_count = len(trades)
        wins = [t for t in trades if (t.realized_pnl or 0) > 0]
        losses = [t for t in trades if (t.realized_pnl or 0) < 0]
        b.win_rate = round(len(wins) / len(trades), 4)
        b.total_pnl = round(
            sum(t.realized_pnl or 0 for t in trades), 2
        )

        r_vals = [t.r_multiple for t in trades]
        if r_vals:
            b.avg_r = round(_safe_mean(r_vals), 4)

        gross_w = sum(t.realized_pnl or 0 for t in wins)
        gross_l = abs(sum(t.realized_pnl or 0 for t in losses))
        if gross_l > 0:
            b.profit_factor = round(gross_w / gross_l, 2)
        elif gross_w > 0:
            b.profit_factor = float("inf")

        b.expectancy = round(
            _safe_mean([t.realized_pnl for t in trades]), 2
        )

        edge_ratios = [
            t.edge_ratio for t in trades
            if (t.edge_ratio or 0) > 0
        ]
        if edge_ratios:
            b.avg_edge_ratio = round(_safe_mean(edge_ratios), 2)

        hold_days = [
            t.holding_days for t in trades
            if (t.holding_days or 0) > 0
        ]
        if hold_days:
            b.avg_holding_days = round(_safe_mean(hold_days), 1)

        etds = [t.etd_pct for t in trades if (t.etd_pct or 0) != 0]
        if etds:
            b.avg_etd_pct = round(_safe_mean(etds), 2)

        result[name] = b

    return result


def _compute_by_regime(
    closed: list[JournalEntry],
) -> dict[str, RegimeBreakdown]:
    groups: dict[str, list[JournalEntry]] = defaultdict(list)
    for t in closed:
        regime = t.entry_market_regime or "unknown"
        groups[regime].append(t)

    result: dict[str, RegimeBreakdown] = {}
    for name, trades in groups.items():
        b = RegimeBreakdown(regime=name)
        b.trade_count = len(trades)
        wins = [t for t in trades if (t.realized_pnl or 0) > 0]
        losses = [t for t in trades if (t.realized_pnl or 0) < 0]
        b.win_rate = round(len(wins) / len(trades), 4)
        b.total_pnl = round(
            sum(t.realized_pnl or 0 for t in trades), 2
        )
        r_vals = [t.r_multiple for t in trades]
        if r_vals:
            b.avg_r = round(_safe_mean(r_vals), 4)
        gross_w = sum(t.realized_pnl or 0 for t in wins)
        gross_l = abs(sum(t.realized_pnl or 0 for t in losses))
        if gross_l > 0:
            b.profit_factor = round(gross_w / gross_l, 2)
        elif gross_w > 0:
            b.profit_factor = float("inf")
        result[name] = b

    return result


# ── Output helpers ────────────────────────────────────────────────


def format_metrics_text(metrics: JournalMetrics) -> str:
    """Human-readable summary for logging."""
    o = metrics.overall
    r = metrics.risk_adjusted
    d = metrics.drawdown
    c = metrics.confidence

    lines = [
        f"=== Trade Journal Report ({metrics.computed_at}) ===",
        "",
        f"Trades: {o.total_trades} "
        f"({o.wins}W / {o.losses}L / {o.breakeven}BE)",
        f"Win Rate: {o.win_rate:.1%}",
        f"Avg Win: ${o.avg_win:+,.2f}  "
        f"Avg Loss: ${o.avg_loss:+,.2f}  "
        f"Ratio: {o.win_loss_ratio:.2f}",
        f"Expectancy: ${o.expectancy:+,.2f}/trade  "
        f"({o.expectancy_r:+.2f}R/trade)",
        f"Profit Factor: {o.profit_factor:.2f}",
        f"Total P&L: ${o.total_pnl:+,.2f}",
        "",
        f"R-Distribution: "
        f"mean={metrics.r_distribution.mean_r:+.2f}, "
        f"median={metrics.r_distribution.median_r:+.2f}, "
        f"skew={metrics.r_distribution.skewness_r:+.2f}",
        f"  >+2R: {metrics.r_distribution.pct_above_2r:.0%}  "
        f"<-1R: {metrics.r_distribution.pct_below_neg1r:.0%}",
        "",
    ]

    if r.daily_return_count > 0:
        lines.extend([
            f"Sharpe: {r.sharpe_ratio:.2f}  "
            f"Sortino: {r.sortino_ratio:.2f}  "
            f"Calmar: {r.calmar_ratio:.2f}",
            f"Max DD: {d.max_drawdown_pct:.1f}% "
            f"(${d.max_drawdown_dollars:,.0f})  "
            f"Duration: {d.max_drawdown_duration_days}d  "
            f"Recovery: {d.recovery_factor:.2f}x",
            "",
        ])

    if c.min_track_record_length > 0:
        sig = "YES" if c.is_statistically_significant else "NO"
        lines.extend([
            f"PSR: {c.probabilistic_sharpe:.1%} "
            f"(MinTRL={c.min_track_record_length}, "
            f"significant={sig})",
        ])
        if not c.is_statistically_significant:
            lines.append(
                f"  Need {c.trades_until_significant} more "
                f"observations for 95% confidence"
            )
        lines.append("")

    e = metrics.excursion
    lines.extend([
        f"Edge Ratio: {e.avg_edge_ratio:.2f} "
        f"(winners: {e.avg_edge_ratio_winners:.2f})",
        f"Avg ETD: winners={e.avg_etd_winners:.1f}%  "
        f"losers={e.avg_etd_losers:.1f}%",
        f"Time to MFE: {e.avg_time_to_mfe_days:.1f}d  "
        f"Time to MAE: {e.avg_time_to_mae_days:.1f}d",
        "",
    ])

    h = metrics.holding
    lines.extend([
        f"Avg Hold: {h.avg_hold_all:.1f}d "
        f"(W={h.avg_hold_winners:.1f}d, "
        f"L={h.avg_hold_losers:.1f}d)",
        f"Time Exit Rate: {h.time_exit_rate:.1%}",
    ])
    if h.exit_reason_distribution:
        lines.append(
            f"Exit Reasons: {h.exit_reason_distribution}"
        )

    s = metrics.streaks
    lines.extend([
        "",
        f"Streaks: max win={s.max_consecutive_wins}, "
        f"max loss={s.max_consecutive_losses} "
        f"(expected max loss={s.expected_max_losing_streak})",
        f"Current: {s.current_streak} {s.current_streak_type}(s)",
    ])

    # Strategy breakdown
    if metrics.by_strategy:
        lines.extend(["", "--- Strategy Breakdown ---"])
        for name, b in sorted(
            metrics.by_strategy.items(),
            key=lambda x: -x[1].total_pnl,
        ):
            lines.append(
                f"  {name}: {b.trade_count} trades, "
                f"WR={b.win_rate:.0%}, "
                f"PF={b.profit_factor:.2f}, "
                f"avgR={b.avg_r:+.2f}, "
                f"P&L=${b.total_pnl:+,.0f}"
            )

    return "\n".join(lines)


def export_metrics_json(
    metrics: JournalMetrics,
    path: Path,
) -> None:
    """Write metrics to a JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(metrics)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"  Metrics exported to {path}")
    except Exception as exc:
        log.warning(f"Metrics export failed: {exc}")


# ── Math helpers ──────────────────────────────────────────────────


def _safe_mean(values: list) -> float:
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (
        len(values) - 1
    )
    return math.sqrt(variance)


def _skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    std = _std(values)
    if std < 1e-10:
        return 0.0
    return sum((x - mean) ** 3 for x in values) / (n * std ** 3)


def _kurtosis(values: list[float]) -> float:
    """Excess kurtosis (normal = 0)."""
    n = len(values)
    if n < 4:
        return 0.0
    mean = sum(values) / n
    std = _std(values)
    if std < 1e-10:
        return 0.0
    return (
        sum((x - mean) ** 4 for x in values) / (n * std ** 4)
    ) - 3.0


def _pearson_corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2 or len(y) < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum(
        (x[i] - mean_x) * (y[i] - mean_y) for i in range(n)
    )
    std_x = math.sqrt(sum((v - mean_x) ** 2 for v in x))
    std_y = math.sqrt(sum((v - mean_y) ** 2 for v in y))
    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0
    return cov / (std_x * std_y)


def _probabilistic_sharpe(
    observed_sr: float,
    benchmark_sr: float,
    n: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Probability that true SR exceeds benchmark (López de Prado)."""
    if n < 2:
        return 0.5
    sr_std_sq = (
        1
        - skew * observed_sr
        + (kurtosis - 1) / 4 * observed_sr ** 2
    ) / (n - 1)
    if sr_std_sq <= 0:
        return 0.5
    sr_std = math.sqrt(sr_std_sq)
    if sr_std < 1e-10:
        return 0.5
    z = (observed_sr - benchmark_sr) / sr_std
    # Approximate normal CDF
    return _norm_cdf(z)


def _min_track_record_length(
    observed_sr: float,
    benchmark_sr: float,
    skew: float,
    kurtosis: float,
    confidence: float = 0.95,
) -> int:
    """Minimum observations needed for SR significance."""
    if abs(observed_sr - benchmark_sr) < 1e-10:
        return 9999
    z_c = _norm_ppf(confidence)
    factor = (
        1
        - skew * observed_sr
        + (kurtosis - 1) / 4 * observed_sr ** 2
    )
    min_n = 1 + factor * (
        z_c / (observed_sr - benchmark_sr)
    ) ** 2
    return max(2, math.ceil(min_n))


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (rational approximation)."""
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0
    # Beasley-Springer-Moro algorithm (simplified)
    if p < 0.5:
        return -_norm_ppf(1 - p)
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (
        1 + d1 * t + d2 * t * t + d3 * t * t * t
    )


def _days_between(date_a: str, date_b: str) -> int:
    """Days between two ISO date strings."""
    a = date_a[:10]
    b = date_b[:10]
    da = date.fromisoformat(a)
    db = date.fromisoformat(b)
    return (db - da).days
