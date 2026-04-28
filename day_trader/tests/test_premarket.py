"""Tests for the PremarketScanner.

Uses a stub fetcher to feed deterministic data; we don't hit the
Alpaca SDK in unit tests.
"""

from __future__ import annotations

import unittest
from datetime import date
from typing import Optional

from day_trader.data.catalyst import NO_NEWS, CatalystClassifier
from day_trader.data.premarket import (
    PremarketDataFetcher,
    PremarketRanking,
    PremarketScanner,
)


class _StubClassifier(CatalystClassifier):
    """Bypass the real earnings lookup."""

    def __init__(self, label: str = NO_NEWS):
        super().__init__()
        self._label = label

    def classify(self, ticker: str) -> str:
        return self._label


class _StubFetcher(PremarketDataFetcher):
    """Per-ticker dict-driven fetcher for tests."""

    def __init__(self, data: dict[str, dict]):
        self.data = data

    def fetch_premarket_volume(self, ticker, target_date) -> float:
        return self.data.get(ticker, {}).get("share_vol", 0.0)

    def fetch_premarket_dollar_volume(self, ticker, target_date) -> float:
        return self.data.get(ticker, {}).get("dollar_vol", 0.0)

    def fetch_first_premarket_price(
        self, ticker, target_date,
    ) -> Optional[float]:
        return self.data.get(ticker, {}).get("first_price")

    def fetch_prev_close(
        self, ticker, target_date,
    ) -> Optional[float]:
        return self.data.get(ticker, {}).get("prev_close")

    def fetch_avg_premarket_volume(
        self, ticker, target_date, lookback_days=30,
    ) -> Optional[float]:
        return self.data.get(ticker, {}).get("avg_vol")


class TestPremarketScanner(unittest.TestCase):
    def test_basic_scan(self):
        fetcher = _StubFetcher({
            "AAPL": {
                "share_vol": 1_000_000, "dollar_vol": 150_000_000,
                "first_price": 102, "prev_close": 100, "avg_vol": 200_000,
            },
        })
        scanner = PremarketScanner(
            fetcher, catalyst_classifier=_StubClassifier(),
        )
        result = scanner.scan(
            ["AAPL"], target_date=date(2026, 4, 28),
        )
        self.assertIn("AAPL", result)
        ctx = result["AAPL"]
        # RVOL = 1M / 200K = 5.0
        self.assertAlmostEqual(ctx.premkt_rvol, 5.0)
        # Gap = (102 - 100)/100 * 100 = 2.0%
        self.assertAlmostEqual(ctx.premkt_gap_pct, 2.0)
        self.assertEqual(ctx.prev_close, 100)
        self.assertEqual(ctx.catalyst_label, NO_NEWS)

    def test_skips_low_dollar_volume(self):
        fetcher = _StubFetcher({
            "ILLIQ": {
                "share_vol": 5_000, "dollar_vol": 50_000,  # below 100k default
                "first_price": 5, "prev_close": 5, "avg_vol": 1_000,
            },
        })
        scanner = PremarketScanner(fetcher)
        result = scanner.scan(["ILLIQ"], target_date=date.today())
        self.assertNotIn("ILLIQ", result)

    def test_skips_no_prev_close(self):
        fetcher = _StubFetcher({
            "NEW": {
                "share_vol": 100_000, "dollar_vol": 10_000_000,
                "first_price": 100, "prev_close": None, "avg_vol": 50_000,
            },
        })
        scanner = PremarketScanner(fetcher)
        result = scanner.scan(["NEW"], target_date=date.today())
        self.assertNotIn("NEW", result)

    def test_skips_no_first_price(self):
        fetcher = _StubFetcher({
            "QUIET": {
                "share_vol": 100_000, "dollar_vol": 10_000_000,
                "first_price": None, "prev_close": 100, "avg_vol": 50_000,
            },
        })
        scanner = PremarketScanner(fetcher)
        result = scanner.scan(["QUIET"], target_date=date.today())
        self.assertNotIn("QUIET", result)

    def test_handles_missing_avg_volume(self):
        # Symbol qualifies for everything but avg_vol is None
        # → premkt_rvol should be 0.0, but ticker still included
        fetcher = _StubFetcher({
            "NEW": {
                "share_vol": 100_000, "dollar_vol": 15_000_000,
                "first_price": 102, "prev_close": 100, "avg_vol": None,
            },
        })
        scanner = PremarketScanner(fetcher)
        result = scanner.scan(["NEW"], target_date=date.today())
        self.assertIn("NEW", result)
        self.assertEqual(result["NEW"].premkt_rvol, 0.0)

    def test_top_n_selection_by_composite(self):
        # 3 symbols with different composite scores → top 2 only
        fetcher = _StubFetcher({
            "HIGH": {  # rvol=10, gap=5% → 50
                "share_vol": 1_000_000, "dollar_vol": 150_000_000,
                "first_price": 105, "prev_close": 100, "avg_vol": 100_000,
            },
            "MID": {   # rvol=5, gap=2% → 10
                "share_vol": 500_000, "dollar_vol": 75_000_000,
                "first_price": 102, "prev_close": 100, "avg_vol": 100_000,
            },
            "LOW": {   # rvol=2, gap=1% → 2
                "share_vol": 200_000, "dollar_vol": 30_000_000,
                "first_price": 101, "prev_close": 100, "avg_vol": 100_000,
            },
        })
        scanner = PremarketScanner(fetcher)
        result = scanner.scan(
            ["HIGH", "MID", "LOW"], target_date=date.today(), top_n=2,
        )
        self.assertEqual(set(result.keys()), {"HIGH", "MID"})
        self.assertNotIn("LOW", result)

    def test_negative_gap_handled(self):
        # Gap-down stock — composite uses |gap|, so ranks correctly
        fetcher = _StubFetcher({
            "GAP_DOWN": {
                "share_vol": 1_000_000, "dollar_vol": 150_000_000,
                "first_price": 95, "prev_close": 100, "avg_vol": 100_000,
            },
        })
        scanner = PremarketScanner(fetcher)
        result = scanner.scan(
            ["GAP_DOWN"], target_date=date.today(),
        )
        self.assertIn("GAP_DOWN", result)
        self.assertAlmostEqual(result["GAP_DOWN"].premkt_gap_pct, -5.0)
        # Composite uses absolute gap
        score = PremarketRanking.composite(result["GAP_DOWN"])
        self.assertGreater(score, 0)

    def test_per_ticker_failures_isolated(self):
        # If one ticker's fetch raises, others still scan.
        class FlakyFetcher(_StubFetcher):
            def fetch_premarket_dollar_volume(self, ticker, target_date):
                if ticker == "BROKEN":
                    raise RuntimeError("oops")
                return super().fetch_premarket_dollar_volume(
                    ticker, target_date,
                )

        fetcher = FlakyFetcher({
            "AAPL": {
                "share_vol": 1_000_000, "dollar_vol": 150_000_000,
                "first_price": 102, "prev_close": 100, "avg_vol": 200_000,
            },
            "BROKEN": {
                "share_vol": 100, "dollar_vol": 1000,
                "first_price": 1, "prev_close": 1, "avg_vol": 100,
            },
        })
        scanner = PremarketScanner(fetcher)
        result = scanner.scan(
            ["AAPL", "BROKEN"], target_date=date.today(),
        )
        self.assertIn("AAPL", result)
        self.assertNotIn("BROKEN", result)

    def test_classifier_label_propagated(self):
        from day_trader.data.catalyst import NEWS_HIGH
        fetcher = _StubFetcher({
            "AAPL": {
                "share_vol": 1_000_000, "dollar_vol": 150_000_000,
                "first_price": 102, "prev_close": 100, "avg_vol": 100_000,
            },
        })
        scanner = PremarketScanner(
            fetcher, catalyst_classifier=_StubClassifier(NEWS_HIGH),
        )
        result = scanner.scan(["AAPL"], target_date=date.today())
        self.assertEqual(result["AAPL"].catalyst_label, NEWS_HIGH)


if __name__ == "__main__":
    unittest.main()
