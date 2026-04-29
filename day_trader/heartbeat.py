"""Heartbeat — daemon writes a timestamp every loop tick.

Provides a way for an external watchdog (systemd timer →
``deploy/scripts/healthcheck.py``) to detect a stalled or crashed
daemon. The daemon's executor calls :meth:`Heartbeat.beat` every
scan tick; if the watchdog finds the heartbeat file is older than
the freshness threshold during market hours, it pages Telegram.

File format (atomic write via rename)::

    {"timestamp": "2026-04-29T14:30:00-04:00", "session_active": true}

The session_active flag lets the watchdog know whether to alarm:
out-of-session staleness is normal (daemon idle waiting for next
session). In-session staleness for >2 minutes means trouble.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

log = logging.getLogger(__name__)


DEFAULT_HEARTBEAT_PATH = Path("/var/run/day-trader/heartbeat.json")


@dataclass
class HeartbeatStatus:
    """One read of the heartbeat file."""

    exists: bool
    timestamp: Optional[datetime] = None
    session_active: bool = False
    age_seconds: float = 0.0
    is_stale: bool = False


class Heartbeat:
    """Atomic-write heartbeat file owned by the daemon."""

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        stale_after_seconds: int = 120,
    ):
        self.path = Path(path) if path else DEFAULT_HEARTBEAT_PATH
        self.stale_after = timedelta(seconds=stale_after_seconds)
        self._lock = Lock()

    def beat(self, *, session_active: bool = True) -> None:
        """Write a fresh heartbeat. Idempotent; cheap to call often."""
        now = datetime.now(tz=timezone.utc)
        payload = {
            "timestamp": now.isoformat(timespec="seconds"),
            "session_active": session_active,
        }
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                os.replace(tmp, self.path)
            except Exception as exc:
                log.warning(
                    "Heartbeat: write failed: %s — continuing", exc,
                )


def read_status(
    path: Optional[Path] = None,
    *,
    stale_after_seconds: int = 120,
) -> HeartbeatStatus:
    """Read the heartbeat file and return a status snapshot.

    Used by the watchdog (``deploy/scripts/healthcheck.py``) — the
    daemon itself never reads its own heartbeat."""
    path = Path(path) if path else DEFAULT_HEARTBEAT_PATH
    stale_after = timedelta(seconds=stale_after_seconds)

    if not path.exists():
        return HeartbeatStatus(exists=False)

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log.warning("Heartbeat: read failed: %s", exc)
        return HeartbeatStatus(exists=False)

    ts_str = data.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return HeartbeatStatus(exists=False)

    age = (datetime.now(tz=timezone.utc) - ts).total_seconds()
    return HeartbeatStatus(
        exists=True,
        timestamp=ts,
        session_active=bool(data.get("session_active", False)),
        age_seconds=age,
        is_stale=timedelta(seconds=age) > stale_after,
    )
