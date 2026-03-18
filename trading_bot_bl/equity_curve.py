"""Equity curve -- append-only portfolio snapshots for drawdown and return analysis.

Writes to ``execution_logs/equity_curve.jsonl`` (one JSON object per line).
All operations are **non-critical**: failures are logged and swallowed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from trading_bot_bl.models import EquitySnapshot, PortfolioSnapshot

log = logging.getLogger(__name__)

_DEFAULT_FILE = "equity_curve.jsonl"


def record_snapshot(
    portfolio: PortfolioSnapshot,
    log_dir: Path,
    filename: str = _DEFAULT_FILE,
) -> EquitySnapshot | None:
    """Append an equity snapshot from the current portfolio state.

    Reads the last high-water mark from the file to compute
    drawdown.  If the file doesn't exist yet, the current equity
    becomes the initial HWM.

    Returns the snapshot on success, None on failure.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / filename

        # Load previous HWM
        hwm = _read_last_hwm(path)
        equity = portfolio.equity
        if equity > hwm:
            hwm = equity

        drawdown_pct = 0.0
        if hwm > 0:
            drawdown_pct = round(
                (hwm - equity) / hwm * 100, 4
            )

        exposure_pct = 0.0
        if equity > 0:
            exposure_pct = round(
                portfolio.market_value / equity * 100, 2
            )

        # Sum unrealized P&L from positions
        unrealized = sum(
            pos.get("unrealized_pnl", 0.0)
            for pos in portfolio.positions.values()
        )

        snap = EquitySnapshot(
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
            equity=round(equity, 2),
            cash=round(portfolio.cash, 2),
            market_value=round(portfolio.market_value, 2),
            num_positions=len(portfolio.positions),
            realized_pnl_today=0.0,  # enriched later if needed
            unrealized_pnl=round(unrealized, 2),
            day_pnl=round(portfolio.day_pnl, 2),
            day_pnl_pct=round(portfolio.day_pnl_pct, 2),
            drawdown_pct=drawdown_pct,
            high_water_mark=round(hwm, 2),
            exposure_pct=exposure_pct,
        )

        # Append as single JSONL line
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(snap)) + "\n")

        log.debug(
            f"  Equity snapshot: ${equity:,.2f} "
            f"(DD={drawdown_pct:.2f}%, "
            f"HWM=${hwm:,.2f}, "
            f"exposure={exposure_pct:.1f}%)"
        )
        return snap

    except Exception as exc:
        log.warning(f"Equity curve: snapshot failed: {exc}")
        return None


def load_snapshots(
    log_dir: Path,
    filename: str = _DEFAULT_FILE,
) -> list[EquitySnapshot]:
    """Load all equity snapshots from the JSONL file."""
    path = log_dir / filename
    snapshots: list[EquitySnapshot] = []
    if not path.exists():
        return snapshots
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    snapshots.append(EquitySnapshot(**{
                        k: v for k, v in d.items()
                        if k in EquitySnapshot.__dataclass_fields__
                    }))
                except Exception:
                    continue  # skip corrupt lines
    except Exception as exc:
        log.warning(f"Equity curve: load failed: {exc}")
    return snapshots


def _read_last_hwm(path: Path) -> float:
    """Read the high-water mark from the last line of the JSONL."""
    if not path.exists():
        return 0.0
    try:
        last_line = ""
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if last_line:
            d = json.loads(last_line)
            return float(d.get("high_water_mark", 0.0))
    except Exception:
        pass
    return 0.0
