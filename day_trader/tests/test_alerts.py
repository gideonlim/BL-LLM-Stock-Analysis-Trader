"""Tests for the Alerter."""

from __future__ import annotations

import unittest
import unittest.mock as _um
from datetime import datetime, timedelta

from day_trader.alerts import Alerter, SEVERITY_CRIT, SEVERITY_INFO


class TestAlerterUnconfigured(unittest.TestCase):
    def test_unconfigured_no_op(self):
        a = Alerter()
        self.assertFalse(a.is_configured)
        # Send call returns True (no-op success), doesn't raise
        self.assertTrue(a.info("hello"))

    def test_partial_config_is_unconfigured(self):
        # Token without chat_id, or vice versa
        self.assertFalse(Alerter(bot_token="t", chat_id="").is_configured)
        self.assertFalse(Alerter(bot_token="", chat_id="c").is_configured)
        self.assertTrue(Alerter(bot_token="t", chat_id="c").is_configured)


class TestAlerterFormatting(unittest.TestCase):
    def test_format_includes_severity_and_message(self):
        formatted = Alerter._format(SEVERITY_CRIT, "test message", None)
        self.assertIn("[CRIT]", formatted)
        self.assertIn("test message", formatted)

    def test_format_with_context(self):
        formatted = Alerter._format(
            SEVERITY_INFO, "test", {"key1": "val1", "key2": 42},
        )
        self.assertIn("key1: val1", formatted)
        self.assertIn("key2: 42", formatted)

    def test_format_handles_bad_context(self):
        # A non-mapping context shouldn't crash
        formatted = Alerter._format(SEVERITY_INFO, "test", "not_a_mapping")
        self.assertIn("test", formatted)


class TestAlerterRateLimit(unittest.TestCase):
    def test_same_message_rate_limited(self):
        a = Alerter(rate_limit_seconds=60)
        # First call succeeds (no-op since unconfigured)
        self.assertTrue(a.info("dupe"))
        # Second identical call within window — rate limited but
        # still returns True (caller doesn't need to know)
        self.assertTrue(a.info("dupe"))
        # The internal state should show only one entry per
        # (severity, message) key
        self.assertEqual(len(a._last_sent), 1)

    def test_different_messages_not_rate_limited(self):
        a = Alerter(rate_limit_seconds=60)
        a.info("msg A")
        a.info("msg B")
        a.info("msg C")
        self.assertEqual(len(a._last_sent), 3)

    def test_different_severities_not_rate_limited(self):
        a = Alerter(rate_limit_seconds=60)
        a.info("same")
        a.warn("same")
        a.crit("same")
        self.assertEqual(len(a._last_sent), 3)

    def test_rate_limit_expires(self):
        a = Alerter(rate_limit_seconds=60)
        a.info("expires")
        # Manually rewind the timestamp past the window
        for key in a._last_sent:
            a._last_sent[key] -= timedelta(seconds=61)
        # Now the same message goes through again
        # (still no-op since unconfigured, but not rate-limited)
        a.info("expires")
        # The single key was just replaced with a fresh timestamp
        self.assertEqual(len(a._last_sent), 1)


class TestAlerterConfigured(unittest.TestCase):
    def test_configured_send_attempts_post(self):
        a = Alerter(bot_token="bot", chat_id="chat", rate_limit_seconds=0)
        with _um.patch(
            "day_trader.alerts.urlrequest.urlopen"
        ) as mock_open:
            mock_resp = _um.MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__.return_value = mock_resp
            mock_open.return_value = mock_resp
            self.assertTrue(a.crit("test"))
            mock_open.assert_called_once()

    def test_failed_send_returns_false(self):
        a = Alerter(bot_token="bot", chat_id="chat", rate_limit_seconds=0)
        from urllib.error import URLError
        with _um.patch(
            "day_trader.alerts.urlrequest.urlopen",
            side_effect=URLError("network down"),
        ):
            # Even on network failure, alerter doesn't raise
            self.assertFalse(a.crit("test"))


class TestModuleLevel(unittest.TestCase):
    def test_get_default_alerter_caches(self):
        from day_trader.alerts import (
            get_default_alerter,
            reset_default_alerter,
        )
        reset_default_alerter()
        a1 = get_default_alerter()
        a2 = get_default_alerter()
        self.assertIs(a1, a2)
        reset_default_alerter()


if __name__ == "__main__":
    unittest.main()
