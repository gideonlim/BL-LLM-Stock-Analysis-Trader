"""CPPI (Constant Proportion Portfolio Insurance) drawdown control.

Implements a continuous exposure scaling function that replaces the
binary circuit breaker.  As the portfolio draws down toward a
configurable floor, exposure shrinks smoothly.  When the portfolio
recovers past its peak, the floor ratchets up (TIPP variant).

The floor can also be reset when the SPY trend regime returns to
BULL or CAUTION after a drawdown, preventing permanent "cash lock"
where the cushion stays at zero forever.

Usage:
    state = CppiState.from_portfolio(equity=100_000)
    state = update_cppi(state, current_equity=97_000)
    multiplier = state.exposure_multiplier  # 0.0 - 1.0
    # Apply: adjusted_notional *= multiplier

The module is feature-flagged via CPPI_ENABLED (default: false).
When disabled, ``compute_exposure_multiplier`` always returns 1.0.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Default CPPI parameters
_DEFAULT_MAX_DRAWDOWN_PCT = 10.0  # floor = peak * (1 - 10%)
_DEFAULT_MULTIPLIER = 5           # exposure = m * cushion
_DEFAULT_MIN_EXPOSURE_PCT = 10.0  # never go fully to cash


@dataclass
class CppiState:
    """Snapshot of CPPI drawdown control state.

    Attributes:
        peak_equity: High-water mark of the portfolio.
        floor: Minimum acceptable portfolio value.
        max_drawdown_pct: Maximum drawdown before exposure → 0.
        multiplier: Risk multiplier (m).  Higher = more aggressive.
        min_exposure_pct: Minimum exposure even at the floor (prevents
            permanent cash lock).
        exposure_multiplier: Current scaling factor for notional (0-1).
        cushion: Current cushion = equity - floor.
        cushion_pct: Cushion as % of equity.
        floor_was_reset: True if the floor was reset this cycle
            (for logging).
    """

    peak_equity: float = 0.0
    floor: float = 0.0
    max_drawdown_pct: float = _DEFAULT_MAX_DRAWDOWN_PCT
    multiplier: int = _DEFAULT_MULTIPLIER
    min_exposure_pct: float = _DEFAULT_MIN_EXPOSURE_PCT
    exposure_multiplier: float = 1.0
    cushion: float = 0.0
    cushion_pct: float = 0.0
    floor_was_reset: bool = False

    @classmethod
    def from_portfolio(
        cls,
        equity: float,
        max_drawdown_pct: float = _DEFAULT_MAX_DRAWDOWN_PCT,
        multiplier: int = _DEFAULT_MULTIPLIER,
        min_exposure_pct: float = _DEFAULT_MIN_EXPOSURE_PCT,
    ) -> CppiState:
        """Initialize CPPI state from current portfolio equity."""
        floor = equity * (1.0 - max_drawdown_pct / 100.0)
        cushion = equity - floor
        cushion_pct = (cushion / equity * 100) if equity > 0 else 0.0
        return cls(
            peak_equity=equity,
            floor=floor,
            max_drawdown_pct=max_drawdown_pct,
            multiplier=multiplier,
            min_exposure_pct=min_exposure_pct,
            exposure_multiplier=1.0,
            cushion=cushion,
            cushion_pct=cushion_pct,
        )


def update_cppi(
    state: CppiState,
    current_equity: float,
    spy_trend_regime: str = "BULL",
) -> CppiState:
    """Recompute CPPI state given current equity.

    The floor ratchets up when equity makes a new high (TIPP variant).
    If SPY regime returns to BULL or CAUTION after the portfolio was
    at or below the floor, the floor resets to the current equity to
    prevent permanent cash lock.

    Args:
        state: Previous CPPI state.
        current_equity: Current portfolio equity from broker.
        spy_trend_regime: Current SPY regime (BULL, CAUTION, BEAR,
            SEVERE_BEAR).

    Returns:
        Updated CppiState with new exposure_multiplier.
    """
    if current_equity <= 0:
        return CppiState(
            peak_equity=state.peak_equity,
            floor=state.floor,
            max_drawdown_pct=state.max_drawdown_pct,
            multiplier=state.multiplier,
            min_exposure_pct=state.min_exposure_pct,
            exposure_multiplier=0.0,
            cushion=0.0,
            cushion_pct=0.0,
        )

    peak = state.peak_equity
    floor = state.floor
    dd_pct = state.max_drawdown_pct
    floor_was_reset = False

    # ── Ratchet floor up on new equity highs (TIPP) ──────────
    if current_equity > peak:
        peak = current_equity
        floor = peak * (1.0 - dd_pct / 100.0)

    # ── Reset floor on regime recovery ───────────────────────
    # If the portfolio hit or breached the floor (cushion ≤ 0)
    # and SPY has recovered to BULL or CAUTION, reset the floor
    # to the current level so trading can resume.
    old_cushion = current_equity - floor
    if old_cushion <= 0 and spy_trend_regime in ("BULL", "CAUTION"):
        peak = current_equity
        floor = peak * (1.0 - dd_pct / 100.0)
        floor_was_reset = True
        log.info(
            f"  CPPI: floor reset on {spy_trend_regime} regime — "
            f"new floor=${floor:,.0f} "
            f"(peak=${peak:,.0f})"
        )

    # ── Compute cushion and exposure ─────────────────────────
    cushion = current_equity - floor
    cushion_pct = (cushion / current_equity * 100) if current_equity > 0 else 0.0

    if cushion <= 0:
        # At or below floor — use minimum exposure
        raw_exposure = state.min_exposure_pct / 100.0
    else:
        # Modified CPPI: normalize by max possible cushion so
        # exposure = 1.0 at peak (no drawdown).  The multiplier
        # controls how long exposure stays at 1.0 as drawdown
        # deepens: m=5 means exposure only drops below 1.0 when
        # 80%+ of the cushion is consumed.
        max_cushion = peak * dd_pct / 100.0
        raw_exposure = (
            state.multiplier * cushion / max_cushion
        ) if max_cushion > 0 else 0.0

    # Clamp to [min_exposure, 1.0]
    exposure = max(
        state.min_exposure_pct / 100.0,
        min(1.0, raw_exposure),
    )

    return CppiState(
        peak_equity=peak,
        floor=floor,
        max_drawdown_pct=dd_pct,
        multiplier=state.multiplier,
        min_exposure_pct=state.min_exposure_pct,
        exposure_multiplier=round(exposure, 4),
        cushion=round(cushion, 2),
        cushion_pct=round(cushion_pct, 2),
        floor_was_reset=floor_was_reset,
    )


# ── State persistence ────────────────────────────────────────────


def save_cppi_state(state: CppiState, path: Path) -> None:
    """Persist CPPI state to a JSON file for cross-run continuity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(state)
    # Don't persist transient flags
    data.pop("floor_was_reset", None)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_cppi_state(path: Path) -> CppiState | None:
    """Load persisted CPPI state.  Returns None if file is missing
    or corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CppiState(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        log.warning(f"  CPPI: corrupt state file, starting fresh: {exc}")
        return None
