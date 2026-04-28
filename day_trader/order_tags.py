"""Compact day-trade order tagging.

Every day-trade order carries a ``client_order_id`` of the form::

    dt:yyyymmdd:seq:ticker          # parent / entry order
    dt:yyyymmdd:seq:ticker:exit     # closing leg (optional suffix)

Examples: ``dt:20260428:0007:AAPL``, ``dt:20260428:0007:AAPL:exit``.

Compact, prefix-scannable, well under Alpaca's 128-char limit.
This is the ONLY mechanism for separating day-trade orders from
swing orders on a same-account broker — Alpaca *positions* are
not tagged; only orders carry our identifier. See plan §
"Position-netting, symbol locks, and tagged-close mechanics".

The :class:`SequenceCounter` persists the daily counter to disk
so a daemon crash + restart mid-session keeps numbering consistent.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Public constants
DT_PREFIX = "dt:"
EXIT_SUFFIX = "exit"

# Strict format:
#   dt:<8 digits date>:<4 digits seq>:<ticker>[:exit]
# Ticker rules: 1+ uppercase letter, then [A-Z0-9.] (e.g. BRK.B).
_DT_PATTERN = re.compile(
    r"^dt:(?P<date>\d{8}):(?P<seq>\d{4}):(?P<ticker>[A-Z][A-Z0-9.]*)(?::(?P<suffix>exit))?$"
)


# ── Building ids ──────────────────────────────────────────────────


def make_order_id(
    seq: int,
    ticker: str,
    *,
    today: Optional[date] = None,
) -> str:
    """Return a compact day-trade ``client_order_id``.

    Args:
        seq: Today's sequence number (0..9999). Get from a SequenceCounter.
        ticker: Uppercase symbol (e.g. ``"AAPL"``, ``"BRK.B"``).
        today: Override the date (testing); defaults to today.
    """
    if not isinstance(seq, int) or seq < 0 or seq > 9999:
        raise ValueError(f"seq must be int in 0..9999, got {seq!r}")
    if not ticker or not isinstance(ticker, str):
        raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")
    if ticker != ticker.upper():
        raise ValueError(f"ticker must be uppercase, got {ticker!r}")
    if not re.match(r"^[A-Z][A-Z0-9.]*$", ticker):
        raise ValueError(f"ticker has invalid chars: {ticker!r}")
    d = today or date.today()
    return f"{DT_PREFIX}{d.strftime('%Y%m%d')}:{seq:04d}:{ticker}"


def make_exit_order_id(parent: str) -> str:
    """Append the ``:exit`` suffix to a parent day-trade id."""
    if not is_day_trade_id(parent):
        raise ValueError(f"parent must be a day-trade id: {parent!r}")
    if parent.endswith(f":{EXIT_SUFFIX}"):
        raise ValueError(f"parent is already an exit id: {parent!r}")
    return f"{parent}:{EXIT_SUFFIX}"


# ── Inspecting ids ────────────────────────────────────────────────


def is_day_trade_id(client_order_id: Optional[str]) -> bool:
    """True iff ``client_order_id`` starts with ``dt:``.

    Cheap prefix check — does NOT validate the full structure.
    Use :func:`parse_order_id` for that.
    """
    if not client_order_id or not isinstance(client_order_id, str):
        return False
    return client_order_id.startswith(DT_PREFIX)


@dataclass(frozen=True)
class ParsedOrderId:
    """Structured form of a day-trade client_order_id."""

    raw: str
    date_str: str  # "YYYYMMDD"
    seq: int
    ticker: str
    is_exit: bool

    @property
    def parent_id(self) -> str:
        """Return the parent (non-exit) form of this id."""
        return f"{DT_PREFIX}{self.date_str}:{self.seq:04d}:{self.ticker}"


def parse_order_id(client_order_id: str) -> Optional[ParsedOrderId]:
    """Parse a day-trade id into its components.

    Returns ``None`` if ``client_order_id`` is not a well-formed
    day-trade id. Use this rather than ``is_day_trade_id`` when
    you need the seq/ticker/date/exit fields.
    """
    if not is_day_trade_id(client_order_id):
        return None
    m = _DT_PATTERN.match(client_order_id)
    if not m:
        return None
    return ParsedOrderId(
        raw=client_order_id,
        date_str=m.group("date"),
        seq=int(m.group("seq")),
        ticker=m.group("ticker"),
        is_exit=m.group("suffix") == EXIT_SUFFIX,
    )


# ── Daily sequence counter (persisted) ────────────────────────────


class SequenceCounter:
    """Per-day monotonic sequence counter, persisted to disk.

    The counter resets to 0 on each new trading date. The persisted
    state lets us survive a daemon crash mid-session without
    re-using a seq (which would collide on Alpaca's
    ``client_order_id`` uniqueness constraint).

    State file format (atomic write via ``rename``)::

        {"date": "20260428", "next": 8}
    """

    def __init__(self, state_path: Path):
        self.state_path = Path(state_path)
        self._lock = threading.Lock()

    def next(self, *, today: Optional[date] = None) -> int:
        """Allocate the next sequence number for today and persist.

        Thread-safe; safe across crashes (atomic rename of state file).
        Raises ``RuntimeError`` if the daily counter is exhausted (>9999).
        """
        d = today or date.today()
        date_str = d.strftime("%Y%m%d")
        with self._lock:
            state = self._load()
            if state.get("date") != date_str:
                state = {"date": date_str, "next": 0}
            seq = int(state["next"])
            if seq > 9999:
                raise RuntimeError(
                    f"day-trade sequence exhausted for {date_str} "
                    f"(>9999 orders); something is very wrong"
                )
            state["next"] = seq + 1
            self._save(state)
            return seq

    def peek(self, *, today: Optional[date] = None) -> int:
        """Return the NEXT seq that would be allocated, without allocating."""
        d = today or date.today()
        date_str = d.strftime("%Y%m%d")
        with self._lock:
            state = self._load()
            if state.get("date") != date_str:
                return 0
            return int(state["next"])

    def reset_for_testing(self) -> None:
        """Wipe the on-disk state. Tests only — never call in production."""
        with self._lock:
            if self.state_path.exists():
                self.state_path.unlink()

    def _load(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            # Corrupt state — log loudly but don't crash. Returning
            # an empty dict triggers a fresh-day reset which is the
            # safer failure mode (re-allocates from 0; only risk is
            # a duplicate id, which Alpaca will reject with a clear
            # error rather than silently mis-tag).
            log.warning(
                "Sequence counter state corrupt at %s: %s — "
                "resetting to fresh-day baseline",
                self.state_path, exc,
            )
            return {}

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        # Atomic replace — survives crash mid-write
        os.replace(tmp, self.state_path)
