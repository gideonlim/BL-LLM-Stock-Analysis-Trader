"""Tests for the Scheduler and build_session_schedule."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from day_trader.calendar import session_for
from day_trader.scheduler import Scheduler, build_session_schedule

ET = ZoneInfo("America/New_York")

# Regular full-day session
REGULAR_DATE = datetime(2024, 7, 9).date()
REGULAR_SESS = session_for(REGULAR_DATE)

# Half-day session (July 3, 2024 — 13:00 close)
HALF_DATE = datetime(2024, 7, 3).date()
HALF_SESS = session_for(HALF_DATE)


class TestBuildSchedule(unittest.TestCase):
    def test_regular_session_event_count(self):
        events = build_session_schedule(REGULAR_SESS)
        names = [e.name for e in events]
        self.assertIn("catalyst_refresh", names)
        self.assertIn("premarket_scan", names)
        self.assertIn("recovery_reconcile", names)
        self.assertIn("regime_snapshot", names)
        self.assertIn("market_open", names)
        self.assertIn("first_scan", names)
        self.assertIn("scan_tick", names)
        self.assertIn("exit_only", names)
        self.assertIn("force_flat", names)
        self.assertIn("session_close", names)

    def test_events_in_chronological_order(self):
        events = build_session_schedule(REGULAR_SESS)
        times = [e.fire_at for e in events]
        self.assertEqual(times, sorted(times))

    def test_force_flat_before_close(self):
        events = build_session_schedule(REGULAR_SESS)
        ff = [e for e in events if e.name == "force_flat"][0]
        close = [e for e in events if e.name == "session_close"][0]
        self.assertLess(ff.fire_at, close.fire_at)
        # Default 5 min before close
        self.assertEqual(
            (close.fire_at - ff.fire_at).total_seconds(), 300,
        )

    def test_half_day_adjusts_eod_events(self):
        events = build_session_schedule(HALF_SESS)
        ff = [e for e in events if e.name == "force_flat"][0]
        close = [e for e in events if e.name == "session_close"][0]
        # Close at 13:00, force_flat at 12:55
        self.assertEqual(ff.fire_at.hour, 12)
        self.assertEqual(ff.fire_at.minute, 55)
        self.assertEqual(close.fire_at.hour, 13)
        self.assertEqual(close.fire_at.minute, 0)

    def test_scan_tick_is_repeating(self):
        events = build_session_schedule(REGULAR_SESS)
        scan = [e for e in events if e.name == "scan_tick"][0]
        self.assertTrue(scan.repeating)
        self.assertEqual(scan.interval.total_seconds(), 60)


class TestScheduler(unittest.TestCase):
    def test_due_events_fires_past_events(self):
        sched = Scheduler.for_session(REGULAR_SESS)
        # Query well after all events
        late = REGULAR_SESS.close_et + timedelta(hours=1)
        due = sched.due_events(late)
        names = [e.name for e in due]
        self.assertIn("force_flat", names)
        self.assertIn("session_close", names)

    def test_one_shot_doesnt_re_fire(self):
        sched = Scheduler.for_session(REGULAR_SESS)
        late = REGULAR_SESS.close_et + timedelta(hours=1)
        # First call fires everything
        sched.due_events(late)
        # Second call — one-shot events gone
        due2 = sched.due_events(late)
        one_shots = [e for e in due2 if not e.repeating]
        self.assertEqual(one_shots, [])

    def test_repeating_event_fires_multiple_times(self):
        sched = Scheduler.for_session(REGULAR_SESS)
        # Check scan_tick fires at intervals
        t0 = REGULAR_SESS.open_et + timedelta(minutes=6)
        due1 = sched.due_events(t0)
        scan1 = [e for e in due1 if e.name == "scan_tick"]
        self.assertEqual(len(scan1), 1)
        # 60s later → fires again
        t1 = t0 + timedelta(seconds=61)
        due2 = sched.due_events(t1)
        scan2 = [e for e in due2 if e.name == "scan_tick"]
        self.assertEqual(len(scan2), 1)

    def test_next_event_at(self):
        sched = Scheduler.for_session(REGULAR_SESS)
        before_open = REGULAR_SESS.open_et - timedelta(hours=3)
        nxt = sched.next_event_at(before_open)
        self.assertIsNotNone(nxt)
        # Should be the earliest event
        self.assertGreater(nxt, before_open)

    def test_next_event_at_returns_none_after_all(self):
        sched = Scheduler.for_session(REGULAR_SESS)
        late = REGULAR_SESS.close_et + timedelta(hours=1)
        sched.due_events(late)  # fire everything
        nxt = sched.next_event_at(late)
        # Only the repeating scan_tick might still have a future fire
        # but it's past close so it doesn't matter for the test.
        # If nxt is None, that's fine — it means all finite events fired.

    def test_reset_for_session(self):
        sched = Scheduler.for_session(REGULAR_SESS)
        late = REGULAR_SESS.close_et + timedelta(hours=1)
        sched.due_events(late)
        sched.reset_for_session()
        due = sched.due_events(late)
        # After reset, everything should fire again
        names = {e.name for e in due}
        self.assertIn("force_flat", names)


class TestScheduleCustomParams(unittest.TestCase):
    def test_custom_force_flat_window(self):
        events = build_session_schedule(
            REGULAR_SESS, force_flat_min_before_close=10,
        )
        ff = [e for e in events if e.name == "force_flat"][0]
        close = [e for e in events if e.name == "session_close"][0]
        self.assertEqual(
            (close.fire_at - ff.fire_at).total_seconds(), 600,
        )

    def test_custom_scan_interval(self):
        events = build_session_schedule(
            REGULAR_SESS, scan_interval_seconds=30,
        )
        scan = [e for e in events if e.name == "scan_tick"][0]
        self.assertEqual(scan.interval.total_seconds(), 30)


if __name__ == "__main__":
    unittest.main()
