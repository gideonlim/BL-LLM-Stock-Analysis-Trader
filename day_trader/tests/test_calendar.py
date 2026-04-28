"""Tests for the NYSE calendar wrapper.

Uses real ``pandas_market_calendars`` data (no network). Picks
specific historical dates with known properties: a regular session,
a half-day, and an observed holiday.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from day_trader.calendar import (
    ET,
    NyseSession,
    is_market_open,
    is_within_eod_flatten_window,
    next_session,
    session_for,
    time_since_open,
    time_until_close,
)


# Reference dates with known NYSE schedule properties
REGULAR_SESSION = date(2024, 7, 9)        # Tue, regular 9:30–16:00 ET
HALF_DAY = date(2024, 7, 3)               # Wed before July 4: 13:00 close
HOLIDAY = date(2024, 7, 4)                # Independence Day (closed)
SATURDAY = date(2024, 7, 6)               # weekend
DST_TRANSITION_DAY = date(2024, 11, 4)    # Mon after fall-back


class TestSessionFor(unittest.TestCase):
    def test_regular_session_open_close(self):
        sess = session_for(REGULAR_SESSION)
        self.assertIsNotNone(sess)
        self.assertEqual(sess.date, REGULAR_SESSION)
        self.assertEqual(sess.open_et.hour, 9)
        self.assertEqual(sess.open_et.minute, 30)
        self.assertEqual(sess.close_et.hour, 16)
        self.assertEqual(sess.close_et.minute, 0)
        self.assertFalse(sess.is_half_day)

    def test_half_day_detected(self):
        sess = session_for(HALF_DAY)
        self.assertIsNotNone(sess)
        self.assertTrue(sess.is_half_day)
        self.assertEqual(sess.close_et.hour, 13)
        # Open is still 9:30
        self.assertEqual(sess.open_et.hour, 9)

    def test_holiday_returns_none(self):
        self.assertIsNone(session_for(HOLIDAY))

    def test_weekend_returns_none(self):
        self.assertIsNone(session_for(SATURDAY))


class TestSessionMath(unittest.TestCase):
    def test_open_minus_and_plus(self):
        sess = session_for(REGULAR_SESSION)
        self.assertEqual(
            sess.open_minus(60),
            sess.open_et - timedelta(minutes=60),
        )
        self.assertEqual(
            sess.open_plus(5),
            sess.open_et + timedelta(minutes=5),
        )

    def test_close_minus_handles_half_day(self):
        # On a half-day, close_minus(5) should land at 12:55 ET, not 15:55.
        sess = session_for(HALF_DAY)
        target = sess.close_minus(5)
        self.assertEqual(target.hour, 12)
        self.assertEqual(target.minute, 55)

    def test_contains_inside_session(self):
        sess = session_for(REGULAR_SESSION)
        noon = datetime(2024, 7, 9, 12, 0, tzinfo=ET)
        self.assertTrue(sess.contains(noon))

    def test_contains_outside_session(self):
        sess = session_for(REGULAR_SESSION)
        early = datetime(2024, 7, 9, 8, 0, tzinfo=ET)
        late = datetime(2024, 7, 9, 17, 0, tzinfo=ET)
        self.assertFalse(sess.contains(early))
        self.assertFalse(sess.contains(late))


class TestIsMarketOpen(unittest.TestCase):
    def test_during_regular_session(self):
        at = datetime(2024, 7, 9, 12, 0, tzinfo=ET)
        self.assertTrue(is_market_open(at))

    def test_before_open(self):
        at = datetime(2024, 7, 9, 8, 0, tzinfo=ET)
        self.assertFalse(is_market_open(at))

    def test_after_close(self):
        at = datetime(2024, 7, 9, 16, 30, tzinfo=ET)
        self.assertFalse(is_market_open(at))

    def test_holiday_closed(self):
        at = datetime(2024, 7, 4, 12, 0, tzinfo=ET)
        self.assertFalse(is_market_open(at))

    def test_naive_datetime_rejected(self):
        with self.assertRaises(ValueError):
            is_market_open(datetime(2024, 7, 9, 12, 0))

    def test_accepts_utc_input(self):
        # 16:00 UTC is 12:00 ET in summer (EDT, UTC-4)
        at = datetime(2024, 7, 9, 16, 0, tzinfo=ZoneInfo("UTC"))
        self.assertTrue(is_market_open(at))


class TestTimeMath(unittest.TestCase):
    def test_time_until_close_during_session(self):
        at = datetime(2024, 7, 9, 12, 0, tzinfo=ET)
        delta = time_until_close(at)
        self.assertEqual(delta, timedelta(hours=4))

    def test_time_until_close_after_session(self):
        at = datetime(2024, 7, 9, 17, 0, tzinfo=ET)
        self.assertIsNone(time_until_close(at))

    def test_time_since_open_during_session(self):
        at = datetime(2024, 7, 9, 11, 30, tzinfo=ET)
        delta = time_since_open(at)
        self.assertEqual(delta, timedelta(hours=2))

    def test_time_since_open_before_session(self):
        at = datetime(2024, 7, 9, 8, 0, tzinfo=ET)
        self.assertIsNone(time_since_open(at))


class TestNextSession(unittest.TestCase):
    def test_skips_holiday(self):
        # July 3 (half-day) → July 5 should be next (July 4 is holiday).
        nxt = next_session(after=HALF_DAY)
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.date, date(2024, 7, 5))

    def test_skips_weekend(self):
        # Friday Jul 5 → Mon Jul 8
        nxt = next_session(after=date(2024, 7, 5))
        self.assertIsNotNone(nxt)
        self.assertEqual(nxt.date, date(2024, 7, 8))


class TestEodFlattenWindow(unittest.TestCase):
    def test_inside_window_regular_day(self):
        # 15:56 ET on a regular day with 16:00 close, default 5-min window
        at = datetime(2024, 7, 9, 15, 56, tzinfo=ET)
        self.assertTrue(is_within_eod_flatten_window(at))

    def test_outside_window_regular_day(self):
        # 15:50 ET — too early, default window is close-5
        at = datetime(2024, 7, 9, 15, 50, tzinfo=ET)
        self.assertFalse(is_within_eod_flatten_window(at))

    def test_inside_window_half_day(self):
        # On the July 3 half-day, close is 13:00. 12:56 ET should
        # trigger the window — NOT 15:55, which would be wrong.
        at = datetime(2024, 7, 3, 12, 56, tzinfo=ET)
        self.assertTrue(is_within_eod_flatten_window(at))

    def test_half_day_normal_close_time_outside_window(self):
        # 15:56 on a half-day is post-close — outside the window.
        at = datetime(2024, 7, 3, 15, 56, tzinfo=ET)
        self.assertFalse(is_within_eod_flatten_window(at))

    def test_holiday_never_in_window(self):
        at = datetime(2024, 7, 4, 15, 56, tzinfo=ET)
        self.assertFalse(is_within_eod_flatten_window(at))

    def test_weekend_never_in_window(self):
        at = datetime(2024, 7, 6, 15, 56, tzinfo=ET)
        self.assertFalse(is_within_eod_flatten_window(at))

    def test_custom_window_minutes(self):
        # 15-min flatten window on a regular day
        at = datetime(2024, 7, 9, 15, 50, tzinfo=ET)
        self.assertTrue(
            is_within_eod_flatten_window(at, minutes_before_close=15)
        )


if __name__ == "__main__":
    unittest.main()
