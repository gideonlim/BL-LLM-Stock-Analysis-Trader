"""Tests for the CatalystClassifier."""

from __future__ import annotations

import unittest
from datetime import date
from types import SimpleNamespace
from unittest import mock

from day_trader.data.catalyst import (
    NEWS_HIGH,
    NEWS_LOW,
    NO_NEWS,
    CatalystClassifier,
)


def _info(days: int | None, has_date: bool = True):
    """Build a stub EarningsInfo-like object."""
    return SimpleNamespace(
        ticker="AAPL",
        days_until_earnings=days,
        next_earnings_date=date.today() if has_date else None,
        in_blackout=False,
        blackout_reason="",
    )


class TestCatalystClassifier(unittest.TestCase):
    def test_news_high_for_earnings_today(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            return_value=_info(days=0),
        ):
            c = CatalystClassifier()
            self.assertEqual(c.classify("AAPL"), NEWS_HIGH)

    def test_news_high_for_earnings_yesterday(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            return_value=_info(days=-1),
        ):
            c = CatalystClassifier()
            self.assertEqual(c.classify("AAPL"), NEWS_HIGH)

    def test_news_high_for_earnings_tomorrow(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            return_value=_info(days=1),
        ):
            c = CatalystClassifier()
            self.assertEqual(c.classify("AAPL"), NEWS_HIGH)

    def test_news_low_within_3_days(self):
        for days in (-3, -2, 2, 3):
            with mock.patch(
                "day_trader.data.catalyst.check_earnings_blackout",
                return_value=_info(days=days),
            ):
                c = CatalystClassifier()
                self.assertEqual(
                    c.classify("AAPL"), NEWS_LOW,
                    f"expected NEWS_LOW for days={days}",
                )

    def test_no_news_far_away(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            return_value=_info(days=10),
        ):
            c = CatalystClassifier()
            self.assertEqual(c.classify("AAPL"), NO_NEWS)

    def test_no_news_when_no_earnings_date(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            return_value=_info(days=None, has_date=False),
        ):
            c = CatalystClassifier()
            self.assertEqual(c.classify("AAPL"), NO_NEWS)

    def test_no_news_on_lookup_error(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            side_effect=RuntimeError("yfinance hiccup"),
        ):
            c = CatalystClassifier()
            # Fail-open — must not crash, defaults to NO_NEWS
            self.assertEqual(c.classify("AAPL"), NO_NEWS)

    def test_classify_many(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            side_effect=lambda t, **kw: _info(
                days=0 if t == "AAPL" else 5
            ),
        ):
            c = CatalystClassifier()
            result = c.classify_many(["AAPL", "MSFT"])
        self.assertEqual(result["AAPL"], NEWS_HIGH)
        self.assertEqual(result["MSFT"], NO_NEWS)

    def test_classify_many_normalizes_keys(self):
        with mock.patch(
            "day_trader.data.catalyst.check_earnings_blackout",
            return_value=_info(days=10),
        ):
            c = CatalystClassifier()
            result = c.classify_many(["aapl"])
        self.assertIn("AAPL", result)
        self.assertEqual(result["AAPL"], NO_NEWS)

    def test_invalid_window_config_rejected(self):
        with self.assertRaises(ValueError):
            CatalystClassifier(high_window_days=5, low_window_days=2)


if __name__ == "__main__":
    unittest.main()
