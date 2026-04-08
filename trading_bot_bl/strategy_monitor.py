"""Strategy health monitor -- observation-mode detection of strategy degradation.

Runs every execution cycle between history enrichment and order building.
Computes rolling metrics per strategy from closed journal trades, maintains
a per-strategy state machine (ACTIVE / CAUTION / SUSPENDED), and logs what
it *would* have done — but never blocks trades.

After 50-100 closed trades accumulate, the paper trail lets us measure
whether the monitor's flags predict future underperformance.  Only then
do we turn on enforcement.

All operations are **non-critical**: failures are logged and swallowed
so the core trading pipeline is never disrupted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from statistics import mean, stdev

from trading_bot_bl.models import JournalEntry, Signal

log = logging.getLogger(__name__)

_STATE_FILE = "strategy_monitor_state.json"
_LOG_FILE = "strategy_monitor.jsonl"


# ── Enums & config ──────────────────────────────────────────────


class StrategyState(str, Enum):
    ACTIVE = "ACTIVE"
    CAUTION = "CAUTION"
    SUSPENDED = "SUSPENDED"


@dataclass
class MonitorThresholds:
    """Configurable thresholds for state transitions.

    Kept inside this module (not in TradingConfig) because the right
    values are unknown until the observation period produces data.
    """

    # Minimum closed trades before a strategy is evaluated.
    min_trades_for_eval: int = 5

    # Rolling window size (uses min(window, available) when < window).
    rolling_window: int = 20

    # ACTIVE → CAUTION triggers (any one fires).
    caution_sharpe: float = 0.5
    caution_wr_drop_pp: float = 15.0   # percentage-point drop from baseline
    caution_consec_losses: int = 3

    # CAUTION → SUSPENDED triggers (any one fires).
    suspend_sharpe: float = 0.0
    suspend_wr_drop_pp: float = 25.0
    suspend_consec_losses: int = 5

    # Recovery: SUSPENDED → CAUTION (all must hold).
    recover_to_caution_sharpe: float = 0.3
    recover_to_caution_max_losses: int = 2  # consecutive losses <

    # Recovery: CAUTION → ACTIVE (all must hold).
    recover_to_active_sharpe: float = 0.8
    recover_to_active_wr_pp: float = 10.0   # within this many pp of baseline
    recover_to_active_max_losses: int = 1   # consecutive losses <

    # Size multiplier when CAUTION verdict fires.
    caution_size_mult: float = 0.5


# ── Data classes ────────────────────────────────────────────────


@dataclass
class StrategyHealth:
    """Rolling metrics and current state for one strategy."""

    strategy: str
    state: StrategyState = StrategyState.ACTIVE
    prev_state: StrategyState = StrategyState.ACTIVE

    # Rolling metrics (over last N trades).
    rolling_sharpe: float = 0.0
    rolling_win_rate: float = 0.0
    baseline_win_rate: float = -1.0   # frozen at first eval; -1 = not yet set
    consec_losses: int = 0
    mean_r: float = 0.0
    strategy_drawdown_pct: float = 0.0

    closed_trades: int = 0
    last_transition: str = ""  # ISO date of last state change

    note: str = ""  # human-readable reason for current state


@dataclass
class MonitorVerdict:
    """Per-signal recommendation from the monitor."""

    ticker: str
    strategy: str
    action: str          # "pass" | "would_reduce" | "would_block"
    reason: str = ""
    size_mult: float = 1.0


@dataclass
class MonitorResult:
    """Full output of one monitor cycle."""

    timestamp: str
    strategy_states: dict[str, StrategyHealth] = field(
        default_factory=dict,
    )
    verdicts: list[MonitorVerdict] = field(default_factory=list)
    total_closed_trades: int = 0


# ── Rolling metric helpers ──────────────────────────────────────


def _rolling_sharpe(pnl_pcts: list[float]) -> float:
    """Trade-level Sharpe: mean(pnl%) / std(pnl%).

    Returns 0.0 when std is zero (all identical returns).
    """
    if len(pnl_pcts) < 2:
        return 0.0
    s = stdev(pnl_pcts)
    if s == 0:
        return 0.0
    return mean(pnl_pcts) / s


def _win_rate(pnl_pcts: list[float]) -> float:
    """Fraction of trades with positive P&L (0.0 – 1.0)."""
    if not pnl_pcts:
        return 0.0
    return sum(1 for p in pnl_pcts if p > 0) / len(pnl_pcts)


def _consec_losses(trades: list[JournalEntry]) -> int:
    """Count consecutive losses from most recent trade backward."""
    count = 0
    for t in reversed(trades):
        if t.realized_pnl_pct <= 0:
            count += 1
        else:
            break
    return count


def _strategy_drawdown(pnl_pcts: list[float]) -> float:
    """Peak-to-trough drawdown of cumulative trade P&L (percentage).

    Returns 0.0 when no drawdown.
    """
    if not pnl_pcts:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnl_pcts:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


# ── State machine ───────────────────────────────────────────────


def _evaluate_transition(
    health: StrategyHealth,
    thresholds: MonitorThresholds,
) -> StrategyState:
    """Determine the new state for a strategy based on its metrics."""
    s = health.state
    sharpe = health.rolling_sharpe
    wr_drop = (health.baseline_win_rate - health.rolling_win_rate) * 100
    losses = health.consec_losses

    if s == StrategyState.ACTIVE:
        if (
            sharpe < thresholds.caution_sharpe
            or wr_drop > thresholds.caution_wr_drop_pp
            or losses >= thresholds.caution_consec_losses
        ):
            return StrategyState.CAUTION
        return StrategyState.ACTIVE

    if s == StrategyState.CAUTION:
        # Can degrade further.
        if (
            sharpe < thresholds.suspend_sharpe
            or wr_drop > thresholds.suspend_wr_drop_pp
            or losses >= thresholds.suspend_consec_losses
        ):
            return StrategyState.SUSPENDED
        # Can recover.
        if (
            sharpe >= thresholds.recover_to_active_sharpe
            and wr_drop <= thresholds.recover_to_active_wr_pp
            and losses < thresholds.recover_to_active_max_losses
        ):
            return StrategyState.ACTIVE
        return StrategyState.CAUTION

    # SUSPENDED
    if (
        sharpe >= thresholds.recover_to_caution_sharpe
        and losses < thresholds.recover_to_caution_max_losses
    ):
        return StrategyState.CAUTION
    return StrategyState.SUSPENDED


# ── Persistence ─────────────────────────────────────────────────


def _save_state(
    states: dict[str, StrategyHealth],
    path: Path,
) -> None:
    """Atomic write of monitor state (CPPI pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        name: asdict(h) for name, h in states.items()
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _load_state(path: Path) -> dict[str, StrategyHealth]:
    """Load persisted state.  Returns empty dict if missing/corrupt."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        result: dict[str, StrategyHealth] = {}
        for name, d in raw.items():
            d["state"] = StrategyState(d["state"])
            d["prev_state"] = StrategyState(d["prev_state"])
            result[name] = StrategyHealth(**d)
        return result
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
        log.warning(
            f"  Strategy monitor: corrupt state file, "
            f"starting fresh: {exc}"
        )
        return {}


# ── JSONL log ───────────────────────────────────────────────────


def _append_log(result: MonitorResult, path: Path) -> None:
    """Append one JSON line per cycle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": result.timestamp,
        "strategy_states": {
            name: {
                "state": h.state.value,
                "prev_state": h.prev_state.value,
                "rolling_sharpe": h.rolling_sharpe,
                "rolling_win_rate": h.rolling_win_rate,
                "baseline_win_rate": h.baseline_win_rate,
                "consec_losses": h.consec_losses,
                "closed_trades": h.closed_trades,
                "mean_r": h.mean_r,
                "strategy_drawdown_pct": h.strategy_drawdown_pct,
                "note": h.note,
            }
            for name, h in result.strategy_states.items()
        },
        "verdicts": [
            {
                "ticker": v.ticker,
                "strategy": v.strategy,
                "action": v.action,
                "size_mult": v.size_mult,
                "reason": v.reason,
            }
            for v in result.verdicts
        ],
        "total_closed_trades": result.total_closed_trades,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Main class ──────────────────────────────────────────────────


class StrategyMonitor:
    """Observation-mode strategy health monitor.

    Constructor takes ``history_dir`` (resolved ``execution_logs/``).
    Call ``evaluate(closed_trades, buy_signals)`` once per cycle.
    """

    def __init__(
        self,
        history_dir: Path,
        thresholds: MonitorThresholds | None = None,
    ) -> None:
        self._dir = history_dir
        self._thresholds = thresholds or MonitorThresholds()
        self._state_path = history_dir / _STATE_FILE
        self._log_path = history_dir / _LOG_FILE

    def evaluate(
        self,
        closed_trades: list[JournalEntry],
        buy_signals: list[Signal],
    ) -> MonitorResult:
        """Run one monitor cycle.

        1. Group closed trades by strategy.
        2. Compute rolling metrics per strategy.
        3. Transition state machine.
        4. Produce verdicts for each BUY signal.
        5. Persist state + append JSONL log.
        """
        now = datetime.now().isoformat(timespec="seconds")
        th = self._thresholds

        # Load persisted state from previous run.
        prev_states = _load_state(self._state_path)

        # Group closed trades by strategy, sorted by exit date.
        by_strategy: dict[str, list[JournalEntry]] = {}
        for t in closed_trades:
            by_strategy.setdefault(t.strategy, []).append(t)
        for trades in by_strategy.values():
            trades.sort(key=lambda t: t.exit_date or t.entry_date)

        # Compute health per strategy.
        health_map: dict[str, StrategyHealth] = {}

        for strat_name, trades in by_strategy.items():
            n = len(trades)
            prev = prev_states.get(strat_name)

            # Rolling window: last N trades.
            window = trades[-th.rolling_window:]
            pnl_pcts = [t.realized_pnl_pct for t in window]
            r_mults = [
                t.r_multiple for t in window if t.r_multiple != 0
            ]
            all_pnl_pcts = [t.realized_pnl_pct for t in trades]

            # Frozen baseline: use persisted value if it exists,
            # otherwise compute now and it will be saved.
            # Sentinel -1.0 means "not yet initialized".
            if prev and prev.baseline_win_rate >= 0:
                baseline_wr = prev.baseline_win_rate
            else:
                baseline_wr = _win_rate(all_pnl_pcts)

            health = StrategyHealth(
                strategy=strat_name,
                state=(
                    prev.state if prev else StrategyState.ACTIVE
                ),
                prev_state=(
                    prev.state if prev else StrategyState.ACTIVE
                ),
                rolling_sharpe=round(_rolling_sharpe(pnl_pcts), 4),
                rolling_win_rate=round(_win_rate(pnl_pcts), 4),
                baseline_win_rate=round(baseline_wr, 4),
                consec_losses=_consec_losses(trades),
                mean_r=(
                    round(mean(r_mults), 4) if r_mults else 0.0
                ),
                strategy_drawdown_pct=_strategy_drawdown(
                    all_pnl_pcts,
                ),
                closed_trades=n,
                last_transition=(
                    prev.last_transition if prev else ""
                ),
            )

            # Evaluate state machine only with enough data.
            if n < th.min_trades_for_eval:
                health.state = StrategyState.ACTIVE
                health.prev_state = StrategyState.ACTIVE
                health.note = (
                    f"Insufficient trade history "
                    f"({n}/{th.min_trades_for_eval})"
                )
            else:
                new_state = _evaluate_transition(health, th)
                if new_state != health.state:
                    health.prev_state = health.state
                    health.state = new_state
                    health.last_transition = now
                health.note = _build_note(health, th)

            health_map[strat_name] = health

        # Produce verdicts for each BUY signal.
        verdicts: list[MonitorVerdict] = []
        for sig in buy_signals:
            if sig.signal_raw != 1:
                continue
            h = health_map.get(sig.strategy)
            if h is None or h.state == StrategyState.ACTIVE:
                verdicts.append(MonitorVerdict(
                    ticker=sig.ticker,
                    strategy=sig.strategy,
                    action="pass",
                ))
            elif h.state == StrategyState.CAUTION:
                verdicts.append(MonitorVerdict(
                    ticker=sig.ticker,
                    strategy=sig.strategy,
                    action="would_reduce",
                    size_mult=th.caution_size_mult,
                    reason=h.note,
                ))
            else:  # SUSPENDED
                verdicts.append(MonitorVerdict(
                    ticker=sig.ticker,
                    strategy=sig.strategy,
                    action="would_block",
                    reason=h.note,
                ))

        result = MonitorResult(
            timestamp=now,
            strategy_states=health_map,
            verdicts=verdicts,
            total_closed_trades=len(closed_trades),
        )

        # Persist state + JSONL log.
        try:
            _save_state(health_map, self._state_path)
        except Exception as exc:
            log.warning(f"  Strategy monitor: state save failed: {exc}")
        try:
            _append_log(result, self._log_path)
        except Exception as exc:
            log.warning(f"  Strategy monitor: log write failed: {exc}")

        return result


# ── Helpers ─────────────────────────────────────────────────────


def _build_note(
    health: StrategyHealth,
    th: MonitorThresholds,
) -> str:
    """Human-readable explanation of the current state."""
    parts: list[str] = []
    sharpe = health.rolling_sharpe
    wr_drop = (
        health.baseline_win_rate - health.rolling_win_rate
    ) * 100
    losses = health.consec_losses

    if health.state == StrategyState.CAUTION:
        if sharpe < th.caution_sharpe:
            parts.append(
                f"rolling Sharpe {sharpe:.2f} "
                f"< {th.caution_sharpe}"
            )
        if wr_drop > th.caution_wr_drop_pp:
            parts.append(
                f"win rate dropped {wr_drop:.1f}pp "
                f"from baseline"
            )
        if losses >= th.caution_consec_losses:
            parts.append(f"{losses} consecutive losses")
    elif health.state == StrategyState.SUSPENDED:
        if sharpe < th.suspend_sharpe:
            parts.append(
                f"rolling Sharpe {sharpe:.2f} "
                f"< {th.suspend_sharpe}"
            )
        if wr_drop > th.suspend_wr_drop_pp:
            parts.append(
                f"win rate dropped {wr_drop:.1f}pp "
                f"from baseline"
            )
        if losses >= th.suspend_consec_losses:
            parts.append(f"{losses} consecutive losses")

    if not parts:
        return health.state.value
    return "; ".join(parts)
