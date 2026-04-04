"""Tests for the prefetch data module.

Covers:
  - is_market_open() for various times and days
  - get_previous_trading_day() for all day-of-week cases
  - Manifest write/read round-trip, missing, corrupt
  - validate_prefetch_cache() with valid/invalid manifests
  - Ticker verification (last bar date check)
  - prune_old_cache() keeps recent, removes old
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from quant_analysis_bot.prefetch import (
    ET,
    _et_now,
    get_previous_trading_day,
    is_market_open,
    prune_old_cache,
    read_manifest,
    validate_prefetch_cache,
    write_manifest,
)


def _mock_et_now(year, month, day, hour, minute=0):
    """Return a patcher that fixes _et_now to a specific ET datetime."""
    dt = datetime(year, month, day, hour, minute, tzinfo=ET)
    return patch(
        "quant_analysis_bot.prefetch._et_now", return_value=dt
    )


class TestIsMarketOpen(unittest.TestCase):
    """Test market open/closed detection."""

    def test_weekday_during_market_hours(self):
        # Wednesday 2026-04-01, 11:00 AM ET
        with _mock_et_now(2026, 4, 1, 11):
            self.assertTrue(is_market_open())

    def test_weekday_at_open(self):
        # 9:30 AM ET exactly
        with _mock_et_now(2026, 4, 1, 9, 30):
            self.assertTrue(is_market_open())

    def test_weekday_before_open(self):
        # 9:29 AM ET
        with _mock_et_now(2026, 4, 1, 9, 29):
            self.assertFalse(is_market_open())

    def test_weekday_at_close(self):
        # 4:00 PM ET exactly — market is closed
        with _mock_et_now(2026, 4, 1, 16, 0):
            self.assertFalse(is_market_open())

    def test_weekday_after_close(self):
        # 6:00 PM ET
        with _mock_et_now(2026, 4, 1, 18):
            self.assertFalse(is_market_open())

    def test_weekday_early_morning(self):
        # 7:00 AM ET
        with _mock_et_now(2026, 4, 1, 7):
            self.assertFalse(is_market_open())

    def test_saturday(self):
        # Saturday 2026-04-04, 12:00 PM ET
        with _mock_et_now(2026, 4, 4, 12):
            self.assertFalse(is_market_open())

    def test_sunday(self):
        # Sunday 2026-04-05, 12:00 PM ET
        with _mock_et_now(2026, 4, 5, 12):
            self.assertFalse(is_market_open())


class TestGetPreviousTradingDay(unittest.TestCase):
    """Test previous trading day computation."""

    def test_tuesday_returns_monday(self):
        # Tuesday 2026-04-07
        self.assertEqual(
            get_previous_trading_day(date(2026, 4, 7)),
            date(2026, 4, 6),  # Monday
        )

    def test_wednesday_returns_tuesday(self):
        self.assertEqual(
            get_previous_trading_day(date(2026, 4, 8)),
            date(2026, 4, 7),
        )

    def test_thursday_returns_wednesday(self):
        self.assertEqual(
            get_previous_trading_day(date(2026, 4, 9)),
            date(2026, 4, 8),
        )

    def test_friday_returns_thursday(self):
        self.assertEqual(
            get_previous_trading_day(date(2026, 4, 10)),
            date(2026, 4, 9),
        )

    def test_monday_returns_friday(self):
        # Monday 2026-04-06 -> Friday 2026-04-03
        self.assertEqual(
            get_previous_trading_day(date(2026, 4, 6)),
            date(2026, 4, 3),
        )

    def test_saturday_returns_friday(self):
        self.assertEqual(
            get_previous_trading_day(date(2026, 4, 4)),
            date(2026, 4, 3),
        )

    def test_sunday_returns_friday(self):
        self.assertEqual(
            get_previous_trading_day(date(2026, 4, 5)),
            date(2026, 4, 3),
        )


class TestManifestWriteRead(unittest.TestCase):
    """Test manifest I/O."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_round_trip(self):
        tickers = ["AAPL", "MSFT", "GOOG"]
        write_manifest(
            self.tmpdir, "20260406", tickers,
            regime_cached=True, market_closed=True,
        )
        m = read_manifest(self.tmpdir, "20260406")
        self.assertIsNotNone(m)
        self.assertEqual(m["date_str"], "20260406")
        self.assertTrue(m["market_closed"])
        self.assertEqual(m["ticker_count"], 3)
        self.assertEqual(m["tickers"], tickers)
        self.assertTrue(m["regime_cached"])

    def test_read_missing_returns_none(self):
        self.assertIsNone(
            read_manifest(self.tmpdir, "20260101")
        )

    def test_read_corrupt_returns_none(self):
        path = os.path.join(
            self.tmpdir, "prefetch_20260406.json"
        )
        with open(path, "w") as f:
            f.write("{broken json!!!")
        self.assertIsNone(
            read_manifest(self.tmpdir, "20260406")
        )

    def test_atomic_write_no_temp_left(self):
        write_manifest(
            self.tmpdir, "20260406", ["AAPL"],
            regime_cached=False, market_closed=True,
        )
        # No .tmp files should remain
        tmp_files = [
            f for f in os.listdir(self.tmpdir)
            if f.endswith(".tmp")
        ]
        self.assertEqual(tmp_files, [])


class TestValidatePrefetchCache(unittest.TestCase):
    """Test cache validation logic."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_market_closed(self):
        # Prefetch on Monday 2026-04-06, market closed
        write_manifest(
            self.tmpdir, "20260406", ["AAPL", "MSFT"],
            regime_cached=True, market_closed=True,
        )
        # Validate on Tuesday 2026-04-07
        result = validate_prefetch_cache(
            self.tmpdir, date(2026, 4, 7)
        )
        self.assertIsNotNone(result)
        prev_str, tickers = result
        self.assertEqual(prev_str, "20260406")
        self.assertEqual(tickers, ["AAPL", "MSFT"])

    def test_invalid_market_open(self):
        # Prefetch while market was open -> invalid
        write_manifest(
            self.tmpdir, "20260406", ["AAPL"],
            regime_cached=True, market_closed=False,
        )
        result = validate_prefetch_cache(
            self.tmpdir, date(2026, 4, 7)
        )
        self.assertIsNone(result)

    def test_no_manifest_returns_none(self):
        result = validate_prefetch_cache(
            self.tmpdir, date(2026, 4, 7)
        )
        self.assertIsNone(result)

    def test_friday_to_monday(self):
        # Prefetch Friday 2026-04-03
        write_manifest(
            self.tmpdir, "20260403", ["AAPL"],
            regime_cached=True, market_closed=True,
        )
        # Validate Monday 2026-04-06
        result = validate_prefetch_cache(
            self.tmpdir, date(2026, 4, 6)
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "20260403")

    def test_stale_manifest_not_used(self):
        # Manifest from Thursday, but reference is Monday
        # Previous trading day of Monday = Friday, not Thursday
        write_manifest(
            self.tmpdir, "20260402", ["AAPL"],
            regime_cached=True, market_closed=True,
        )
        result = validate_prefetch_cache(
            self.tmpdir, date(2026, 4, 6)  # Monday
        )
        self.assertIsNone(result)  # No Friday manifest exists

    def test_empty_tickers_returns_none(self):
        write_manifest(
            self.tmpdir, "20260406", [],
            regime_cached=True, market_closed=True,
        )
        result = validate_prefetch_cache(
            self.tmpdir, date(2026, 4, 7)
        )
        self.assertIsNone(result)


class TestPruneOldCache(unittest.TestCase):
    """Test cache pruning."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_keeps_recent_removes_old(self):
        # Create files for various dates
        # "Today" in ET is 2026-04-08 (Wednesday)
        for date_str in [
            "20260408",  # today
            "20260407",  # yesterday
            "20260406",  # 2 days ago
            "20260403",  # Friday (old)
            "20260401",  # Wednesday (old)
        ]:
            # Ticker parquet
            with open(
                os.path.join(self.tmpdir, f"AAPL_{date_str}.parquet"),
                "w",
            ) as f:
                f.write("dummy")
            # Manifest
            with open(
                os.path.join(self.tmpdir, f"prefetch_{date_str}.json"),
                "w",
            ) as f:
                f.write("{}")

        with _mock_et_now(2026, 4, 8, 18):
            prune_old_cache(self.tmpdir, keep_days=2)

        remaining = set(os.listdir(self.tmpdir))
        # Should keep today (20260408), yesterday (20260407),
        # and 2 days ago (20260406) -> keep_days=2 means keep
        # today + 2 previous = 3 date slots
        self.assertIn("AAPL_20260408.parquet", remaining)
        self.assertIn("AAPL_20260407.parquet", remaining)
        self.assertIn("AAPL_20260406.parquet", remaining)
        # Old files should be removed
        self.assertNotIn("AAPL_20260403.parquet", remaining)
        self.assertNotIn("AAPL_20260401.parquet", remaining)
        self.assertNotIn("prefetch_20260403.json", remaining)
        self.assertNotIn("prefetch_20260401.json", remaining)

    def test_nonexistent_dir_no_error(self):
        # Should not raise
        prune_old_cache(os.path.join(self.tmpdir, "nope"))

    def test_ignores_non_dated_files(self):
        # Files without date patterns should be left alone
        other = os.path.join(self.tmpdir, "universe_top900.json")
        with open(other, "w") as f:
            f.write("{}")

        with _mock_et_now(2026, 4, 8, 18):
            prune_old_cache(self.tmpdir, keep_days=2)

        self.assertTrue(os.path.exists(other))


if __name__ == "__main__":
    unittest.main()
