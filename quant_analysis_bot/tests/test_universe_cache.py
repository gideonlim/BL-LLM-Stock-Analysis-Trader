"""Tests for universe ticker cache freshness logic.

Covers:
  - _cache_is_fresh() with various ages and weekday boundaries
  - Monday-reset behaviour (cache always expires on Monday)
  - 14-day hard expiry
  - force_refresh parameter on fetch_top_us_stocks
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

# Stub yfinance so universe.py can import without the real package.
if "yfinance" not in sys.modules:
    _yf = types.ModuleType("yfinance")
    _yf.Ticker = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["yfinance"] = _yf

from quant_analysis_bot.universe import (  # noqa: E402
    _CACHE_MAX_AGE_DAYS,
    _cache_is_fresh,
    fetch_top_us_stocks,
)


def _touch_cache(path: str, mtime_dt: datetime) -> None:
    """Create a cache file and set its mtime to *mtime_dt*."""
    with open(path, "w") as f:
        json.dump(["AAPL", "MSFT", "GOOG"], f)
    ts = mtime_dt.timestamp()
    os.utime(path, (ts, ts))


class TestCacheIsFresh(unittest.TestCase):
    """Unit tests for _cache_is_fresh()."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.tmpdir, "universe.json")

    # -- Basic cases --------------------------------------------------

    def test_missing_file_is_not_fresh(self) -> None:
        self.assertFalse(_cache_is_fresh(self.cache_path))

    def test_just_created_is_fresh(self) -> None:
        """A file written moments ago should be fresh."""
        now = datetime(2026, 4, 1, 10, 0)  # Wednesday
        _touch_cache(self.cache_path, now)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = now + timedelta(minutes=5)
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertTrue(_cache_is_fresh(self.cache_path))

    # -- 14-day hard expiry -------------------------------------------

    def test_expired_after_14_days(self) -> None:
        mtime = datetime(2026, 3, 16, 10, 0)  # Monday
        now = mtime + timedelta(days=15)
        _touch_cache(self.cache_path, mtime)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertFalse(_cache_is_fresh(self.cache_path))

    def test_fresh_at_13_days_same_week(self) -> None:
        """13 days old but still within same week-span → depends on
        Monday crossing. In practice 13 days always crosses a Monday,
        so this should be stale."""
        mtime = datetime(2026, 4, 1, 10, 0)  # Wednesday
        now = mtime + timedelta(days=13)  # Tuesday next-next week
        _touch_cache(self.cache_path, mtime)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.fromtimestamp = datetime.fromtimestamp
            # 13 days from Wednesday crosses at least one Monday
            self.assertFalse(_cache_is_fresh(self.cache_path))

    # -- Monday reset -------------------------------------------------

    def test_stale_across_monday_boundary(self) -> None:
        """Cache written on Friday should expire on the next Monday."""
        friday = datetime(2026, 3, 27, 14, 0)  # Friday
        monday = datetime(2026, 3, 30, 9, 0)  # Monday morning
        _touch_cache(self.cache_path, friday)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = monday
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertFalse(_cache_is_fresh(self.cache_path))

    def test_fresh_within_same_week(self) -> None:
        """Cache written on Tuesday, checked on Thursday → fresh."""
        tuesday = datetime(2026, 3, 31, 10, 0)  # Tuesday
        thursday = datetime(2026, 4, 2, 10, 0)  # Thursday
        _touch_cache(self.cache_path, tuesday)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = thursday
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertTrue(_cache_is_fresh(self.cache_path))

    def test_fresh_saturday_to_sunday(self) -> None:
        """Weekend cache should survive until Monday."""
        saturday = datetime(2026, 3, 28, 12, 0)
        sunday = datetime(2026, 3, 29, 12, 0)
        _touch_cache(self.cache_path, saturday)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = sunday
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertTrue(_cache_is_fresh(self.cache_path))

    def test_stale_saturday_to_monday(self) -> None:
        """Weekend cache should expire once Monday arrives."""
        saturday = datetime(2026, 3, 28, 12, 0)
        monday = datetime(2026, 3, 30, 0, 1)
        _touch_cache(self.cache_path, saturday)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = monday
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertFalse(_cache_is_fresh(self.cache_path))

    def test_monday_cache_survives_until_next_monday(self) -> None:
        """Cache written on Monday should survive until the following
        Monday."""
        monday = datetime(2026, 3, 30, 10, 0)
        next_sunday = datetime(2026, 4, 5, 23, 59)
        _touch_cache(self.cache_path, monday)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = next_sunday
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertTrue(_cache_is_fresh(self.cache_path))

    def test_monday_cache_stale_on_next_monday(self) -> None:
        monday = datetime(2026, 3, 30, 10, 0)
        next_monday = datetime(2026, 4, 6, 0, 0)
        _touch_cache(self.cache_path, monday)
        with patch(
            "quant_analysis_bot.universe.datetime"
        ) as mock_dt:
            mock_dt.now.return_value = next_monday
            mock_dt.fromtimestamp = datetime.fromtimestamp
            self.assertFalse(_cache_is_fresh(self.cache_path))

    # -- Config constant ----------------------------------------------

    def test_max_age_is_14(self) -> None:
        self.assertEqual(_CACHE_MAX_AGE_DAYS, 14)


class TestForceRefresh(unittest.TestCase):
    """Ensure force_refresh=True bypasses the cache."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    @patch("quant_analysis_bot.universe._fetch_wiki_tickers")
    @patch("quant_analysis_bot.universe._get_market_caps")
    def test_force_refresh_ignores_cache(
        self, mock_caps, mock_wiki
    ) -> None:
        # Write a valid, fresh cache
        cache_file = os.path.join(self.tmpdir, "universe_top5.json")
        with open(cache_file, "w") as f:
            json.dump(["OLD1", "OLD2", "OLD3", "OLD4", "OLD5"], f)

        # Mock wiki to return new tickers
        mock_wiki.return_value = ["NEW1", "NEW2", "NEW3", "NEW4", "NEW5"]
        mock_caps.return_value = {}

        result = fetch_top_us_stocks(
            n=5, cache_dir=self.tmpdir, force_refresh=True
        )
        # Should NOT return the cached ["OLD1",...] — wiki was called
        mock_wiki.assert_called()
        self.assertNotIn("OLD1", result)

    @patch("quant_analysis_bot.universe.load_extra_tickers", return_value=[])
    @patch("quant_analysis_bot.universe._fetch_wiki_tickers")
    @patch("quant_analysis_bot.universe._get_market_caps")
    def test_normal_uses_fresh_cache(
        self, mock_caps, mock_wiki, _mock_extra
    ) -> None:
        cache_file = os.path.join(self.tmpdir, "universe_top5.json")
        with open(cache_file, "w") as f:
            json.dump(["AAPL", "MSFT", "GOOG", "AMZN", "META"], f)

        result = fetch_top_us_stocks(
            n=5, cache_dir=self.tmpdir, force_refresh=False
        )
        # Should return cached data — wiki never called
        mock_wiki.assert_not_called()
        self.assertEqual(result, ["AAPL", "MSFT", "GOOG", "AMZN", "META"])


class TestCorruptCache(unittest.TestCase):
    """Corrupt cache should fall through to rebuild, not crash."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    @patch("quant_analysis_bot.universe._fetch_wiki_tickers")
    @patch("quant_analysis_bot.universe._get_market_caps")
    def test_corrupt_json_triggers_rebuild(
        self, mock_caps, mock_wiki
    ) -> None:
        cache_file = os.path.join(self.tmpdir, "universe_top5.json")
        # Write truncated / corrupt JSON
        with open(cache_file, "w") as f:
            f.write('["AAPL", "MS')

        mock_wiki.return_value = ["X1", "X2", "X3", "X4", "X5"]
        mock_caps.return_value = {}

        # Should NOT raise — should log warning and rebuild
        result = fetch_top_us_stocks(
            n=5, cache_dir=self.tmpdir, force_refresh=False
        )
        mock_wiki.assert_called()
        self.assertNotIn("AAPL", result)

    @patch("quant_analysis_bot.universe._fetch_wiki_tickers")
    @patch("quant_analysis_bot.universe._get_market_caps")
    def test_empty_file_triggers_rebuild(
        self, mock_caps, mock_wiki
    ) -> None:
        cache_file = os.path.join(self.tmpdir, "universe_top5.json")
        with open(cache_file, "w") as f:
            pass  # empty file

        mock_wiki.return_value = ["Y1", "Y2", "Y3"]
        mock_caps.return_value = {}

        result = fetch_top_us_stocks(
            n=3, cache_dir=self.tmpdir, force_refresh=False
        )
        mock_wiki.assert_called()


if __name__ == "__main__":
    unittest.main()
