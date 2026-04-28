"""Tests for OrbVwapStrategy and VwapPullbackStrategy.

Uses the real BarCache and a synthesized NYSE session at a known
date so the time-gating logic is exercised under realistic conditions.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from day_trader.calendar import session_for
from day_trader.data.cache import BarCache
from day_trader.models import Bar, MarketState, TickerContext
from day_trader.strategies.orb_vwap import OrbVwapStrategy
from day_trader.strategies.vwap_pullback import VwapPullbackStrategy


ET = ZoneInfo("America/New_York")
# Use a known regular session
SESSION_DATE = datetime(2024, 7, 9).date()  # Tue, regular full day
SESS = session_for(SESSION_DATE)
assert SESS is not None and not SESS.is_half_day


def _bar(
    ticker: str,
    minute_offset: int,
    o: float = 100,
    h: float = 100.5,
    l: float = 99.5,
    c: float = 100,
    v: float = 100_000,
) -> Bar:
    return Bar(
        ticker=ticker,
        timestamp=SESS.open_et + timedelta(minutes=minute_offset),
        open=o, high=h, low=l, close=c, volume=v,
    )


def _ticker_ctx(
    ticker: str = "AAPL",
    rvol: float = 3.0,
    catalyst: str = "",
) -> TickerContext:
    return TickerContext(
        ticker=ticker, premkt_rvol=rvol, catalyst_label=catalyst,
        avg_daily_volume=10_000_000, avg_dollar_volume=2e9,
        prev_close=100.0,
    )


def _market_state() -> MarketState:
    return MarketState(
        spy_price=550, vix=15.0, spy_trend_regime="BULL",
    )


# ── OrbVwapStrategy ──────────────────────────────────────────────


class TestOrbVwap(unittest.TestCase):
    def setUp(self):
        self.strat = OrbVwapStrategy(
            or_minutes=5, atr_period=5, rr_target=2.0,
            min_premkt_rvol=2.0, time_stop_minutes=90,
        )
        self.cache = BarCache()
        self.ctx = _ticker_ctx()
        self.ms = _market_state()

    def _populate_or_window(self, or_high: float = 101.0, or_low: float = 99.0):
        """5 1-min bars from 09:30 to 09:34 inclusive."""
        for i in range(5):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=100, h=or_high - 0.5, l=or_low + 0.5, c=100,
                v=200_000,
            ))
        # The actual or_high/or_low across the OR — use one bar that
        # touches the extremes
        self.cache.add_bar(_bar(
            "AAPL", minute_offset=4,
            o=100, h=or_high, l=or_low, c=100, v=200_000,
        ))

    def _at(self, minute_offset: int) -> datetime:
        return SESS.open_et + timedelta(minutes=minute_offset)

    def test_no_signal_before_or_window_closes(self):
        self._populate_or_window()
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms,
            self._at(3), SESS,
        )
        self.assertIsNone(sig)

    def test_no_signal_when_no_breakout(self):
        self._populate_or_window(or_high=101, or_low=99)
        # Post-OR bar that does NOT break out (close stays at 100)
        for i in range(5, 15):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=100, h=100.5, l=99.5, c=100, v=200_000,
            ))
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms,
            self._at(15), SESS,
        )
        self.assertIsNone(sig)

    def test_signal_on_clean_breakout(self):
        self._populate_or_window(or_high=101, or_low=99)
        # Post-OR bars: enough to get ATR(5), all rising
        for i in range(5, 15):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=100 + (i - 5) * 0.1,
                h=100.5 + (i - 5) * 0.1,
                l=99.5 + (i - 5) * 0.1,
                c=100 + (i - 5) * 0.1,
                v=200_000,
            ))
        # Final breakout bar — close above OR high (101) AND above VWAP
        self.cache.add_bar(_bar(
            "AAPL", minute_offset=15,
            o=101, h=102, l=100.9, c=101.5, v=300_000,
        ))
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms,
            self._at(15), SESS,
        )
        self.assertIsNotNone(sig)
        self.assertEqual(sig.ticker, "AAPL")
        self.assertEqual(sig.side, "buy")
        self.assertEqual(sig.strategy, "orb_vwap")
        # Stop must be ABOVE OR low (since ATR-stop is closer)
        self.assertLess(sig.stop_loss_price, sig.signal_price)
        # TP at 2R from entry
        risk = sig.signal_price - sig.stop_loss_price
        expected_tp = sig.signal_price + 2.0 * risk
        self.assertAlmostEqual(sig.take_profit_price, expected_tp, places=2)

    def test_one_shot_per_ticker(self):
        # Build a clean breakout, fire once, then re-scan — must NOT fire.
        self._populate_or_window(or_high=101, or_low=99)
        for i in range(5, 16):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=101, h=102, l=100.9, c=101.5, v=300_000,
            ))
        sig1 = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms,
            self._at(15), SESS,
        )
        self.assertIsNotNone(sig1)
        # Same data, different time — no new signal
        sig2 = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms,
            self._at(20), SESS,
        )
        self.assertIsNone(sig2)

    def test_reset_clears_fired_state(self):
        self._populate_or_window(or_high=101, or_low=99)
        for i in range(5, 16):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=101, h=102, l=100.9, c=101.5, v=300_000,
            ))
        self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms,
            self._at(15), SESS,
        )
        self.assertTrue(self.strat.already_fired("AAPL"))
        self.strat.reset_for_session()
        self.assertFalse(self.strat.already_fired("AAPL"))

    def test_low_rvol_blocks_signal(self):
        self._populate_or_window(or_high=101, or_low=99)
        for i in range(5, 16):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=101, h=102, l=100.9, c=101.5, v=300_000,
            ))
        low_rvol_ctx = _ticker_ctx(rvol=1.0)  # below the 2.0 threshold
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, low_rvol_ctx, self.ms,
            self._at(15), SESS,
        )
        self.assertIsNone(sig)

    def test_manage_force_flat_at_time_stop(self):
        from day_trader.models import OpenDayTrade
        position = OpenDayTrade(
            ticker="AAPL", strategy="orb_vwap", side="long",
            qty=10, entry_price=100, entry_time=self._at(10),
            sl_price=99, tp_price=102,
            parent_client_order_id="dt:20240709:0001:AAPL", seq=1,
        )
        # Before time_stop_minutes (90)
        out = self.strat.manage(position, self.cache, self._at(60), SESS)
        self.assertIsNone(out)
        # At/after time_stop_minutes
        out = self.strat.manage(position, self.cache, self._at(91), SESS)
        self.assertIsNotNone(out)
        self.assertEqual(out.reason, "time_stop")


# ── VwapPullbackStrategy ─────────────────────────────────────────


class TestVwapPullback(unittest.TestCase):
    def setUp(self):
        self.strat = VwapPullbackStrategy(
            atr_period=5, trend_lookback=10, min_rvol=1.5,
            warmup_minutes=30,
        )
        self.cache = BarCache()
        self.ctx = _ticker_ctx(rvol=2.0)
        self.ms = _market_state()

    def _at(self, minute_offset: int) -> datetime:
        return SESS.open_et + timedelta(minutes=minute_offset)

    def _populate_uptrend_then_pullback(self):
        """30 mins of higher-highs, then a pullback to VWAP, then a
        bullish reversal candle."""
        # Bars 0..15: low-volume warm-up at 100
        for i in range(15):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=100, h=100.2, l=99.8, c=100, v=100_000,
            ))
        # Bars 15..35: rising — higher highs
        for i in range(15, 35):
            base = 100 + (i - 15) * 0.5
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=base, h=base + 0.5, l=base - 0.2, c=base + 0.4,
                v=200_000,
            ))
        # Pullback bar — touches near VWAP (around 105)
        # but closes below the high, then a reversal candle
        # Insert pullback at minute 35
        self.cache.add_bar(_bar(
            "AAPL", minute_offset=35,
            o=109, h=109, l=106, c=106.5, v=200_000,  # pullback
        ))
        # Reversal — close above open, low touches near VWAP
        # We need this close > VWAP and close > open
        latest_vwap = self.cache.vwap("AAPL")
        # Bar 36: open below current price, close above
        self.cache.add_bar(_bar(
            "AAPL", minute_offset=36,
            o=106.5,
            h=107.5,
            l=latest_vwap + 0.05,  # touches near VWAP
            c=107.0,
            v=300_000,
        ))

    def test_no_signal_during_warmup(self):
        # Add some bars; query at minute 5 (before warmup of 30)
        for i in range(5):
            self.cache.add_bar(_bar("AAPL", minute_offset=i))
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms, self._at(5), SESS,
        )
        self.assertIsNone(sig)

    def test_no_signal_without_uptrend(self):
        # Flat sideways — no higher-highs
        for i in range(40):
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=100, h=100.5, l=99.5, c=100, v=200_000,
            ))
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms, self._at(40), SESS,
        )
        self.assertIsNone(sig)

    def test_no_signal_when_below_vwap(self):
        # Build downtrend that's below VWAP — pullback strategy
        # is long-only; should not fire when price is below VWAP.
        for i in range(40):
            base = 100 - i * 0.1  # falling
            self.cache.add_bar(_bar(
                "AAPL", minute_offset=i,
                o=base, h=base + 0.1, l=base - 0.5, c=base - 0.2,
                v=200_000,
            ))
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, self.ctx, self.ms, self._at(40), SESS,
        )
        self.assertIsNone(sig)

    def test_low_rvol_blocks(self):
        for i in range(40):
            self.cache.add_bar(_bar("AAPL", minute_offset=i))
        low_rvol_ctx = _ticker_ctx(rvol=1.0)
        sig = self.strat.scan_ticker(
            "AAPL", self.cache, low_rvol_ctx, self.ms, self._at(40), SESS,
        )
        self.assertIsNone(sig)


if __name__ == "__main__":
    unittest.main()
