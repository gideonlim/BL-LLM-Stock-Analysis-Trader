"""Tests for Heartbeat + read_status."""

from __future__ import annotations

import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from day_trader.heartbeat import Heartbeat, read_status


class TestHeartbeatWrite(unittest.TestCase):
    def test_beat_creates_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hb.json"
            hb = Heartbeat(path)
            hb.beat(session_active=True)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text())
            self.assertIn("timestamp", data)
            self.assertEqual(data["session_active"], True)

    def test_beat_atomic_no_tmp_leak(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hb.json"
            hb = Heartbeat(path)
            hb.beat()
            tmp_files = list(Path(tmp).glob("hb.json.tmp"))
            self.assertEqual(tmp_files, [])

    def test_beat_creates_parent_dirs(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "deeply" / "hb.json"
            hb = Heartbeat(path)
            hb.beat()
            self.assertTrue(path.exists())

    def test_beat_session_inactive(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hb.json"
            hb = Heartbeat(path)
            hb.beat(session_active=False)
            data = json.loads(path.read_text())
            self.assertEqual(data["session_active"], False)


class TestReadStatus(unittest.TestCase):
    def test_missing_file(self):
        with TemporaryDirectory() as tmp:
            status = read_status(Path(tmp) / "missing.json")
        self.assertFalse(status.exists)

    def test_fresh_heartbeat_not_stale(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hb.json"
            Heartbeat(path).beat(session_active=True)
            status = read_status(path, stale_after_seconds=120)
        self.assertTrue(status.exists)
        self.assertTrue(status.session_active)
        self.assertFalse(status.is_stale)
        self.assertLess(status.age_seconds, 5)

    def test_stale_heartbeat_detected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hb.json"
            # Manually write a heartbeat with an old timestamp
            old_ts = datetime.now(tz=timezone.utc).replace(
                year=2020,
            )
            path.write_text(json.dumps({
                "timestamp": old_ts.isoformat(timespec="seconds"),
                "session_active": True,
            }))
            status = read_status(path, stale_after_seconds=60)
        self.assertTrue(status.exists)
        self.assertTrue(status.is_stale)

    def test_corrupt_file_returns_not_exists(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hb.json"
            path.write_text("{not json")
            status = read_status(path)
        self.assertFalse(status.exists)

    def test_iso_timestamp_with_z_suffix(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hb.json"
            path.write_text(json.dumps({
                "timestamp": "2026-04-29T14:00:00Z",
                "session_active": True,
            }))
            status = read_status(path, stale_after_seconds=60)
        self.assertTrue(status.exists)


if __name__ == "__main__":
    unittest.main()
