"""Tests for shared.trading_calendar helpers."""

import unittest
from datetime import datetime, time
from zoneinfo import ZoneInfo

from trading_bot_bl.market_config import US, LSE, TSE
from shared.trading_calendar import is_session_open, is_trading_day


class TestUSMarketSession(unittest.TestCase):
    """US (XNYS) session checks."""

    def _make_dt(self, year, month, day, hour, minute):
        """Create a timezone-aware datetime in US Eastern."""
        return datetime(
            year, month, day, hour, minute,
            tzinfo=ZoneInfo("America/New_York"),
        )

    def test_weekday_during_session(self):
        # Wednesday 2026-01-07 at 10:30 AM ET
        dt = self._make_dt(2026, 1, 7, 10, 30)
        self.assertTrue(is_session_open(US, dt))

    def test_weekday_before_open(self):
        # Wednesday 2026-01-07 at 8:00 AM ET
        dt = self._make_dt(2026, 1, 7, 8, 0)
        self.assertFalse(is_session_open(US, dt))

    def test_weekday_after_close(self):
        # Wednesday 2026-01-07 at 16:30 ET
        dt = self._make_dt(2026, 1, 7, 16, 30)
        self.assertFalse(is_session_open(US, dt))

    def test_weekend_saturday(self):
        # Saturday 2026-01-10
        dt = self._make_dt(2026, 1, 10, 11, 0)
        self.assertFalse(is_session_open(US, dt))

    def test_us_holiday_christmas(self):
        # Christmas 2025 is Thursday Dec 25
        dt = self._make_dt(2025, 12, 25, 11, 0)
        self.assertFalse(is_session_open(US, dt))

    def test_is_trading_day_weekday(self):
        from datetime import date
        # Wednesday 2026-01-07
        self.assertTrue(is_trading_day(US, date(2026, 1, 7)))

    def test_is_trading_day_weekend(self):
        from datetime import date
        # Saturday 2026-01-10
        self.assertFalse(is_trading_day(US, date(2026, 1, 10)))


class TestTSELunchBreak(unittest.TestCase):
    """TSE (XTKS) lunch break: 11:30-12:30 JST."""

    def _make_dt(self, hour, minute, day=7):
        """Create a datetime in Asia/Tokyo on Wed 2026-01-07."""
        return datetime(
            2026, 1, day, hour, minute,
            tzinfo=ZoneInfo("Asia/Tokyo"),
        )

    def test_morning_session_open(self):
        # 10:00 JST → open
        dt = self._make_dt(10, 0)
        self.assertTrue(is_session_open(TSE, dt))

    def test_just_before_lunch(self):
        # 11:29 JST → open
        dt = self._make_dt(11, 29)
        self.assertTrue(is_session_open(TSE, dt))

    def test_lunch_break_start(self):
        # 11:30 JST → closed (lunch)
        dt = self._make_dt(11, 30)
        self.assertFalse(is_session_open(TSE, dt))

    def test_during_lunch(self):
        # 12:00 JST → closed (lunch)
        dt = self._make_dt(12, 0)
        self.assertFalse(is_session_open(TSE, dt))

    def test_lunch_end(self):
        # 12:30 JST → open (afternoon session)
        dt = self._make_dt(12, 30)
        self.assertTrue(is_session_open(TSE, dt))

    def test_afternoon_session(self):
        # 14:00 JST → open
        dt = self._make_dt(14, 0)
        self.assertTrue(is_session_open(TSE, dt))

    def test_after_close(self):
        # XTKS calendar reports close at 15:30 JST
        dt = self._make_dt(15, 30)
        self.assertFalse(is_session_open(TSE, dt))


class TestLSESession(unittest.TestCase):
    """LSE (XLON) session checks."""

    def _make_dt(self, hour, minute, month=1, day=7):
        return datetime(
            2026, month, day, hour, minute,
            tzinfo=ZoneInfo("Europe/London"),
        )

    def test_during_session(self):
        # 10:00 GMT → open
        dt = self._make_dt(10, 0)
        self.assertTrue(is_session_open(LSE, dt))

    def test_before_open(self):
        # 07:30 GMT → closed
        dt = self._make_dt(7, 30)
        self.assertFalse(is_session_open(LSE, dt))


if __name__ == "__main__":
    unittest.main()
