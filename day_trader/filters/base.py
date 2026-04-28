"""Filter ABC + FilterPipeline.

Each Filter is a small object with a single ``passes()`` method that
returns ``(passed, reason)``. The pipeline runs them in sequence,
short-circuits on first reject, and records per-filter rejection
counts in a histogram for the plan-required validation step.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable

from day_trader.models import FilterContext, FilterResult

log = logging.getLogger(__name__)


class Filter(ABC):
    """A single check on a candidate signal.

    Subclasses set the ``name`` class-attr (used for rejection-stats
    keys and log lines) and implement ``passes()``.
    """

    name: str = "filter"

    @abstractmethod
    def passes(self, ctx: FilterContext) -> tuple[bool, str]:
        """Return ``(True, "")`` to accept the signal, or
        ``(False, reason)`` where ``reason`` is a short tag describing
        the rejection (e.g. ``"swing_position"``, ``"vix_too_high"``).

        Reasons are used in the rejection histogram and journal entries,
        so prefer short snake_case strings over full sentences.
        """

    def __repr__(self) -> str:  # pragma: no cover — debug only
        return f"<{self.__class__.__name__} name={self.name}>"


class FilterPipeline:
    """Chain of responsibility over the filter stack.

    Tracks two counters per filter for the daily filter-rejection
    histogram:

    - ``rejected_by_<filter_name>``           — total rejections
    - ``rejected_by_<filter_name>:<reason>``  — per-reason breakdown

    Plus a top-level ``passed`` counter. The histogram lets us spot
    a filter that rejects 0% (trivially passing — broken or
    miscalibrated) or 100% (trivially blocking — wrong threshold).
    """

    def __init__(self, filters: Iterable[Filter]):
        self._filters: list[Filter] = list(filters)
        if not self._filters:
            raise ValueError("FilterPipeline requires at least one filter")
        # Validate names are unique — duplicate names break the histogram
        seen: set[str] = set()
        for f in self._filters:
            if f.name in seen:
                raise ValueError(f"duplicate filter name: {f.name!r}")
            seen.add(f.name)
        self._stats: dict[str, int] = defaultdict(int)

    @property
    def filters(self) -> list[Filter]:
        """Read-only view of the filter chain (for inspection / logging)."""
        return list(self._filters)

    def evaluate(self, ctx: FilterContext) -> FilterResult:
        """Run the chain. Short-circuits on first reject."""
        for f in self._filters:
            try:
                ok, reason = f.passes(ctx)
            except Exception as exc:
                # A filter exception is a bug, not a rejection. Surface
                # loudly but treat as reject (defensive — we'd rather
                # skip a trade than silently let a broken filter pass).
                log.exception(
                    "filter %s raised on signal %r — treating as reject",
                    f.name, getattr(ctx.signal, "ticker", "?"),
                )
                self._record_reject(f.name, "filter_error")
                return FilterResult(
                    passed=False,
                    rejected_by=f.name,
                    reason="filter_error",
                )

            if not ok:
                # Defensive: empty reason → coerce to a stable token so
                # histogram keys stay consistent.
                if not reason:
                    reason = "unspecified"
                self._record_reject(f.name, reason)
                return FilterResult(
                    passed=False,
                    rejected_by=f.name,
                    reason=reason,
                )

        self._stats["passed"] += 1
        return FilterResult(passed=True)

    def _record_reject(self, filter_name: str, reason: str) -> None:
        self._stats[f"rejected_by_{filter_name}"] += 1
        self._stats[f"rejected_by_{filter_name}:{reason}"] += 1

    @property
    def stats(self) -> dict[str, int]:
        """Snapshot of the rejection histogram + pass count."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Clear all counters. Call at session end after writing the
        daily histogram log."""
        self._stats.clear()

    def total_evaluated(self) -> int:
        """Sum of ``passed + every rejected_by_<name>`` (no per-reason
        keys). Useful for the histogram denominator."""
        passed = self._stats.get("passed", 0)
        rejected = sum(
            v for k, v in self._stats.items()
            if k.startswith("rejected_by_") and ":" not in k
        )
        return passed + rejected
