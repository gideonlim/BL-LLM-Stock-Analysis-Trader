"""Tests for BarCache — VWAP, ATR, ring buffer, session reset."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from day_trader.data.cache import BarCache, DEFAULT_RING_SIZE
from day_trader.models import Bar


ET = ZoneInfo("America/New_York")
T0 = datetime(2026, 4, 28, 9, 30, tzinfo=ET)


def _bar(
    ticker: str = "AAPL",
    minute_offset: int = 0,
    o: float = 100,
    h: float = 100.5,
    l: float = 99.5,
    c: float = 100,
    v: float = 100_000,
):
    return Bar(
        ticker=ticker,
        timestamp=T0 + timedelta(minutes=minute_offset),
        open=o, high=h, low=l, close=c, volume=v,
    )


class TestBasicCache(unittest.TestCase):
    def test_empty_cache(self):
        c = BarCache()
        self.assertFalse(c.has_bars("AAPL"))
        self.assertEqual(c.bar_count("AAPL"), 0)
        self.assertIsNone(c.latest("AAPL"))
        self.assertEqual(c.vwap("AAPL"), 0.0)

    def test_add_one_bar(self):
        c = BarCache()
        b = c.add_bar(_bar(c=100, v=1000))
        self.assertEqual(b.ticker, "AAPL")
        self.assertGreater(b.vwap, 0)
        self.assertEqual(c.bar_count("AAPL"), 1)

    def test_ring_buffer_limit(self):
        c = BarCache(ring_size=5)
        for i in range(10):
            c.add_bar(_bar(minute_offset=i, c=100 + i))
        self.assertEqual(c.bar_count("AAPL"), 5)
        # Oldest 5 dropped
        bars = c.get_bars("AAPL")
        self.assertEqual([b.close for b in bars], [105, 106, 107, 108, 109])

    def test_ring_size_must_be_at_least_2(self):
        with self.assertRaises(ValueError):
            BarCache(ring_size=1)

    def test_ticker_normalized_to_upper(self):
        c = BarCache()
        c.add_bar(_bar(ticker="aapl"))
        # Lookups should work with either case
        self.assertEqual(c.bar_count("AAPL"), 1)
        self.assertEqual(c.bar_count("aapl"), 1)


class TestVwap(unittest.TestCase):
    def test_vwap_single_bar_equals_typical_price(self):
        # typical = (h + l + c) / 3 = (101 + 99 + 100) / 3 = 100
        c = BarCache()
        c.add_bar(_bar(h=101, l=99, c=100, v=100))
        self.assertAlmostEqual(c.vwap("AAPL"), 100.0)

    def test_vwap_volume_weighted(self):
        c = BarCache()
        # Bar 1: tp=100, vol=100 → cum = 100*100=10_000, vwap=100
        c.add_bar(_bar(h=101, l=99, c=100, v=100, minute_offset=0))
        # Bar 2: tp=110, vol=300 → cum = 10k + 110*300=43_000,
        # cum_v=400 → vwap = 43000/400 = 107.5
        c.add_bar(_bar(h=111, l=109, c=110, v=300, minute_offset=1))
        self.assertAlmostEqual(c.vwap("AAPL"), 107.5)

    def test_vwap_resets_on_session_reset(self):
        c = BarCache()
        c.add_bar(_bar(h=101, l=99, c=100, v=100, minute_offset=0))
        c.reset_session("AAPL")
        c.add_bar(_bar(h=121, l=119, c=120, v=100, minute_offset=0))
        self.assertAlmostEqual(c.vwap("AAPL"), 120.0)

    def test_zero_volume_bar_does_not_break_vwap(self):
        c = BarCache()
        c.add_bar(_bar(h=101, l=99, c=100, v=0))
        self.assertEqual(c.vwap("AAPL"), 0.0)
        # Then a real bar
        c.add_bar(_bar(h=101, l=99, c=100, v=100, minute_offset=1))
        self.assertAlmostEqual(c.vwap("AAPL"), 100.0)

    def test_pre_supplied_vwap_preserved(self):
        c = BarCache()
        bar = Bar(
            ticker="AAPL", timestamp=T0,
            open=100, high=101, low=99, close=100, volume=100,
            vwap=99.99,
        )
        out = c.add_bar(bar)
        self.assertAlmostEqual(out.vwap, 99.99)


class TestAtr(unittest.TestCase):
    def test_atr_zero_until_period_plus_one_bars(self):
        c = BarCache()
        for i in range(5):
            c.add_bar(_bar(minute_offset=i))
        # period=14 default needs 15 bars
        self.assertEqual(c.atr("AAPL", period=14), 0.0)

    def test_atr_simple_case(self):
        c = BarCache()
        # Each bar: H=101, L=99, C=100 → TR = max(2, |101-100|, |99-100|) = 2
        for i in range(20):
            c.add_bar(_bar(minute_offset=i, h=101, l=99, c=100))
        self.assertAlmostEqual(c.atr("AAPL", period=14), 2.0)

    def test_atr_handles_gaps(self):
        c = BarCache()
        # Bar 1: H=100, L=98, C=99
        c.add_bar(_bar(minute_offset=0, h=100, l=98, c=99))
        # Bar 2: gaps up - H=110, L=108, C=109. TR = max(2, |110-99|, |108-99|) = 11
        c.add_bar(_bar(minute_offset=1, h=110, l=108, c=109))
        # Need more bars for ATR(2) to compute
        c.add_bar(_bar(minute_offset=2, h=111, l=108, c=110))
        atr = c.atr("AAPL", period=2)
        # TRs computed: bar 2 = max(2, 11, 9) = 11; bar 3 = max(3, 2, 1) = 3
        # ATR(2) = (11 + 3) / 2 = 7
        self.assertAlmostEqual(atr, 7.0)


class TestSessionHighLow(unittest.TestCase):
    def test_returns_zero_zero_when_empty(self):
        c = BarCache()
        h, l = c.session_high_low("AAPL")
        self.assertEqual((h, l), (0.0, 0.0))

    def test_full_session(self):
        c = BarCache()
        for i in range(5):
            c.add_bar(_bar(minute_offset=i, h=100 + i, l=99 - i))
        h, l = c.session_high_low("AAPL")
        self.assertEqual((h, l), (104, 95))

    def test_last_n_bars(self):
        c = BarCache()
        for i in range(10):
            c.add_bar(_bar(minute_offset=i, h=100 + i, l=99 - i))
        # Last 3 bars: highs are 107, 108, 109; lows are 92, 91, 90
        h, l = c.session_high_low("AAPL", n=3)
        self.assertEqual((h, l), (109, 90))


class TestSessionVolume(unittest.TestCase):
    def test_sums_volume(self):
        c = BarCache()
        for i in range(3):
            c.add_bar(_bar(minute_offset=i, v=100_000))
        self.assertEqual(c.session_volume("AAPL"), 300_000)


class TestMultiTicker(unittest.TestCase):
    def test_independent_state_per_ticker(self):
        c = BarCache()
        c.add_bar(_bar(ticker="AAPL", minute_offset=0, c=100, v=100))
        c.add_bar(_bar(ticker="MSFT", minute_offset=0, c=200, v=200))
        self.assertEqual(c.bar_count("AAPL"), 1)
        self.assertEqual(c.bar_count("MSFT"), 1)
        self.assertNotEqual(c.vwap("AAPL"), c.vwap("MSFT"))

    def test_reset_one_ticker_keeps_others(self):
        c = BarCache()
        c.add_bar(_bar(ticker="AAPL"))
        c.add_bar(_bar(ticker="MSFT"))
        c.reset_session("AAPL")
        self.assertFalse(c.has_bars("AAPL"))
        self.assertTrue(c.has_bars("MSFT"))

    def test_tickers_lists_all(self):
        c = BarCache()
        c.add_bar(_bar(ticker="AAPL"))
        c.add_bar(_bar(ticker="MSFT"))
        c.add_bar(_bar(ticker="TSLA"))
        self.assertEqual(set(c.tickers()), {"AAPL", "MSFT", "TSLA"})


class TestGetBars(unittest.TestCase):
    def test_empty_returns_empty(self):
        c = BarCache()
        self.assertEqual(c.get_bars("AAPL"), [])

    def test_n_zero_or_negative_returns_empty(self):
        c = BarCache()
        c.add_bar(_bar())
        self.assertEqual(c.get_bars("AAPL", n=0), [])
        self.assertEqual(c.get_bars("AAPL", n=-1), [])

    def test_n_larger_than_buffer_returns_all(self):
        c = BarCache()
        c.add_bar(_bar())
        bars = c.get_bars("AAPL", n=100)
        self.assertEqual(len(bars), 1)

    def test_returns_copy_not_reference(self):
        c = BarCache()
        c.add_bar(_bar())
        bars = c.get_bars("AAPL")
        bars.append("not a bar")
        self.assertEqual(c.bar_count("AAPL"), 1)


if __name__ == "__main__":
    unittest.main()
