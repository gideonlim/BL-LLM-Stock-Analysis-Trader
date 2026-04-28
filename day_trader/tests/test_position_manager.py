"""Tests for PositionManager."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from day_trader.calendar import session_for
from day_trader.data.cache import BarCache
from day_trader.models import ExitIntent, OpenDayTrade
from day_trader.position_manager import PositionManager

ET = ZoneInfo("America/New_York")
SESSION_DATE = datetime(2024, 7, 9).date()
SESS = session_for(SESSION_DATE)


def _pos(ticker: str = "AAPL", strategy: str = "orb_vwap") -> OpenDayTrade:
    return OpenDayTrade(
        ticker=ticker, strategy=strategy, side="long",
        qty=10, entry_price=100.0,
        entry_time=SESS.open_et + timedelta(minutes=10),
        sl_price=95.0, tp_price=110.0,
        parent_client_order_id=f"dt:20240709:0001:{ticker}",
        seq=1,
    )


class TestPositionManager(unittest.TestCase):
    def test_open_and_get(self):
        pm = PositionManager()
        pm.open_position(_pos())
        self.assertEqual(pm.count(), 1)
        self.assertTrue(pm.has("AAPL"))
        self.assertIsNotNone(pm.get("AAPL"))

    def test_close_returns_position(self):
        pm = PositionManager()
        pm.open_position(_pos())
        closed = pm.close_position("AAPL")
        self.assertIsNotNone(closed)
        self.assertEqual(closed.ticker, "AAPL")
        self.assertEqual(pm.count(), 0)

    def test_close_nonexistent_returns_none(self):
        pm = PositionManager()
        self.assertIsNone(pm.close_position("AAPL"))

    def test_close_idempotent(self):
        pm = PositionManager()
        pm.open_position(_pos())
        pm.close_position("AAPL")
        self.assertIsNone(pm.close_position("AAPL"))

    def test_tickers_sorted(self):
        pm = PositionManager()
        pm.open_position(_pos("TSLA"))
        pm.open_position(_pos("AAPL"))
        pm.open_position(_pos("MSFT"))
        self.assertEqual(pm.tickers(), ["AAPL", "MSFT", "TSLA"])

    def test_reset_clears_all(self):
        pm = PositionManager()
        pm.open_position(_pos("AAPL"))
        pm.open_position(_pos("MSFT"))
        pm.reset_for_session()
        self.assertEqual(pm.count(), 0)

    def test_all_for_force_close_returns_all(self):
        pm = PositionManager()
        pm.open_position(_pos("AAPL"))
        pm.open_position(_pos("MSFT"))
        all_pos = pm.all_for_force_close()
        self.assertEqual(len(all_pos), 2)
        # Doesn't remove them
        self.assertEqual(pm.count(), 2)

    def test_check_all_calls_strategy_manage(self):
        from day_trader.strategies.orb_vwap import OrbVwapStrategy
        pm = PositionManager()
        pm.open_position(_pos("AAPL", strategy="orb_vwap"))
        strat = OrbVwapStrategy(time_stop_minutes=90)
        # At minute 91 → should trigger time stop
        exits = pm.check_all(
            strategies={"orb_vwap": strat},
            bar_cache=BarCache(),
            now_et=SESS.open_et + timedelta(minutes=91),
            session=SESS,
        )
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].ticker, "AAPL")
        self.assertEqual(exits[0].reason, "time_stop")

    def test_check_all_missing_strategy_skips(self):
        pm = PositionManager()
        pm.open_position(_pos("AAPL", strategy="unknown"))
        exits = pm.check_all(
            strategies={},
            bar_cache=BarCache(),
            now_et=SESS.open_et + timedelta(minutes=91),
            session=SESS,
        )
        self.assertEqual(exits, [])

    def test_ticker_normalized_to_upper(self):
        pm = PositionManager()
        pm.open_position(_pos("AAPL"))
        self.assertTrue(pm.has("aapl"))
        self.assertIsNotNone(pm.get("aapl"))


if __name__ == "__main__":
    unittest.main()
