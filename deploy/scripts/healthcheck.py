#!/usr/bin/env python3
"""Heartbeat watchdog — alerts when the daemon goes stale.

Runs every minute during market hours via systemd timer
(``day-trader-watchdog.timer``). Reads the heartbeat file the
daemon writes each loop tick and compares its age against the
stale threshold (default 120 s).

Behaviour:

- File missing during market hours → CRIT alert ("daemon hasn't
  started or crashed before first beat")
- File present but >120 s old AND session_active=true → CRIT alert
  ("daemon stalled mid-session")
- File present and fresh → silent success
- File present, session_active=false → silent (daemon idle between
  sessions; not stale per se)

Exits 0 on success, 1 on alert sent. systemd captures the exit
code; the alert itself is sent inline via Telegram.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add repo root to path
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] healthcheck: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("healthcheck")


def main() -> int:
    try:
        from dotenv import load_dotenv
        for p in [Path("/etc/day-trader/env"), Path(".env")]:
            if p.exists():
                load_dotenv(p)
                break
    except ImportError:
        pass

    from day_trader.alerts import get_default_alerter
    from day_trader.calendar import is_market_open
    from day_trader.heartbeat import read_status

    alerter = get_default_alerter()

    if not is_market_open():
        log.debug("Market not open — watchdog no-op")
        return 0

    status = read_status()

    if not status.exists:
        log.error("Heartbeat file missing during market hours")
        alerter.crit(
            "Day-trader heartbeat MISSING during market hours",
            context={
                "expected_path": "/var/run/day-trader/heartbeat.json",
                "action": "check `systemctl status day-trader`",
            },
        )
        return 1

    if status.is_stale and status.session_active:
        log.error(
            "Heartbeat stale: age=%.0fs (last=%s)",
            status.age_seconds, status.timestamp,
        )
        alerter.crit(
            "Day-trader STALLED — heartbeat is stale",
            context={
                "age_seconds": int(status.age_seconds),
                "last_beat": str(status.timestamp),
                "action": "check `systemctl status day-trader` and journalctl",
            },
        )
        return 1

    log.info(
        "Heartbeat fresh: age=%.0fs session_active=%s",
        status.age_seconds, status.session_active,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
