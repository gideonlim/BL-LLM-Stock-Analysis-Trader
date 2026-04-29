"""Operator alerts — Telegram with graceful no-op fallback.

Used by the daemon's crash handler, EOD watchdog, recovery
incident-mode trigger, kill switch, and heartbeat-stalled detector.
Each call is a fire-and-forget POST to Telegram's bot API.

Design principles:

- **No-op when unconfigured.** If ``TELEGRAM_BOT_TOKEN`` is empty,
  every send call logs at INFO level and returns silently. The
  daemon must work without Telegram set up.
- **Best-effort.** Network errors logged at WARNING but never raised.
  An alert that fails to deliver should never crash the daemon.
- **Severity-prefixed.** Messages start with ``[INFO]`` / ``[WARN]``
  / ``[CRIT]`` so phone notifications are scannable.
- **Rate-limited.** Repeating the same alert within
  ``rate_limit_seconds`` (default 60s) is dropped to avoid
  notification storms when something flaps.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Optional
from urllib import request as urlrequest
from urllib.error import URLError

log = logging.getLogger(__name__)


SEVERITY_INFO = "INFO"
SEVERITY_WARN = "WARN"
SEVERITY_CRIT = "CRIT"


class Alerter:
    """Telegram-backed alert sender with no-op fallback."""

    def __init__(
        self,
        *,
        bot_token: str = "",
        chat_id: str = "",
        rate_limit_seconds: int = 60,
        timeout_seconds: float = 5.0,
    ):
        self._bot_token = bot_token.strip()
        self._chat_id = chat_id.strip()
        self._rate_limit = timedelta(seconds=rate_limit_seconds)
        self._timeout = timeout_seconds
        # Per-message rate limiting: key = (severity, hash of body)
        self._last_sent: dict[tuple[str, int], datetime] = {}
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        """True iff both bot token and chat id are set."""
        return bool(self._bot_token and self._chat_id)

    # ── Public send API ──────────────────────────────────────────

    def info(self, message: str, *, context: Optional[Mapping] = None) -> bool:
        return self._send(SEVERITY_INFO, message, context)

    def warn(self, message: str, *, context: Optional[Mapping] = None) -> bool:
        return self._send(SEVERITY_WARN, message, context)

    def crit(self, message: str, *, context: Optional[Mapping] = None) -> bool:
        return self._send(SEVERITY_CRIT, message, context)

    # ── Internal ─────────────────────────────────────────────────

    def _send(
        self,
        severity: str,
        message: str,
        context: Optional[Mapping],
    ) -> bool:
        """Build the formatted message and dispatch.

        Returns True on successful send (or rate-limit / no-op skip),
        False on actual delivery failure."""
        body = self._format(severity, message, context)

        # Rate limit
        key = (severity, hash(message))
        now = datetime.now()
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < self._rate_limit:
                log.debug(
                    "Alerter: rate-limited [%s] %s", severity, message[:80],
                )
                return True
            self._last_sent[key] = now

        # No-op when unconfigured — log so operators know it would have fired
        if not self.is_configured:
            log.info("Alert [%s] (telegram unset): %s", severity, body)
            return True

        return self._post_to_telegram(body)

    @staticmethod
    def _format(
        severity: str,
        message: str,
        context: Optional[Mapping],
    ) -> str:
        ts = datetime.now().strftime("%H:%M:%S ET")
        out = f"[{severity}] {ts}\n{message}"
        if context:
            try:
                ctx_str = "\n".join(
                    f"  {k}: {v}" for k, v in context.items()
                )
                out += f"\n{ctx_str}"
            except Exception:
                # Fall through if context isn't iterable as expected
                pass
        return out

    def _post_to_telegram(self, body: str) -> bool:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": body,
        }).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self._timeout) as resp:
                if 200 <= resp.status < 300:
                    return True
                log.warning(
                    "Alerter: Telegram returned status %d", resp.status,
                )
                return False
        except URLError as exc:
            log.warning("Alerter: Telegram POST failed: %s", exc)
            return False
        except Exception:
            log.exception("Alerter: unexpected error sending to Telegram")
            return False


# ── Module-level convenience ─────────────────────────────────────


_default_alerter: Optional[Alerter] = None


def get_default_alerter() -> Alerter:
    """Lazy module-level alerter — reads env on first call.

    Usage::

        from day_trader.alerts import get_default_alerter
        get_default_alerter().crit("Kill switch tripped", context={...})
    """
    global _default_alerter
    if _default_alerter is None:
        import os
        _default_alerter = Alerter(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
    return _default_alerter


def reset_default_alerter() -> None:
    """Test helper — drop the cached alerter so a fresh one is built
    next call."""
    global _default_alerter
    _default_alerter = None
