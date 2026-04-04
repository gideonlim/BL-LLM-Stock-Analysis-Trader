"""Tests for trade history — churn/cooldown logic.

Covers:
  - was_recently_traded() with buy-only, sell-only, and both
  - recent_trade_reason() message formatting
  - Sell cooldown: re-buying after a recent sell is blocked
  - Reconcile: cancelled buys clear churn, but not sell dates
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from trading_bot_bl.history import (
    TickerHistory,
    TradeHistory,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _days_ago(n: int) -> str:
    return _iso(datetime.now() - timedelta(days=n))


class TestWasRecentlyTraded(unittest.TestCase):
    """was_recently_traded() should check both buy and sell dates."""

    def _history_with(self, **kwargs) -> TradeHistory:
        th = TradeHistory()
        th.by_ticker["VRSN"] = TickerHistory(
            ticker="VRSN", **kwargs
        )
        return th

    def test_no_history(self) -> None:
        h = TradeHistory()
        self.assertFalse(h.was_recently_traded("VRSN", days=2))

    def test_recent_buy_triggers(self) -> None:
        h = self._history_with(
            last_buy_date=_days_ago(1),
            last_buy_strategy="sma_crossover",
        )
        self.assertTrue(h.was_recently_traded("VRSN", days=2))

    def test_old_buy_does_not_trigger(self) -> None:
        h = self._history_with(
            last_buy_date=_days_ago(3),
            last_buy_strategy="sma_crossover",
        )
        self.assertFalse(h.was_recently_traded("VRSN", days=2))

    def test_recent_sell_triggers(self) -> None:
        h = self._history_with(last_sell_date=_days_ago(1))
        self.assertTrue(h.was_recently_traded("VRSN", days=2))

    def test_old_sell_does_not_trigger(self) -> None:
        h = self._history_with(last_sell_date=_days_ago(3))
        self.assertFalse(h.was_recently_traded("VRSN", days=2))

    def test_old_buy_recent_sell_triggers(self) -> None:
        """Buy was 5 days ago (stale) but sell was yesterday → blocked."""
        h = self._history_with(
            last_buy_date=_days_ago(5),
            last_buy_strategy="sma_crossover",
            last_sell_date=_days_ago(1),
        )
        self.assertTrue(h.was_recently_traded("VRSN", days=2))

    def test_recent_buy_old_sell_triggers(self) -> None:
        """Buy was yesterday, sell was 5 days ago → blocked (buy)."""
        h = self._history_with(
            last_buy_date=_days_ago(1),
            last_buy_strategy="sma_crossover",
            last_sell_date=_days_ago(5),
        )
        self.assertTrue(h.was_recently_traded("VRSN", days=2))

    def test_both_old_does_not_trigger(self) -> None:
        h = self._history_with(
            last_buy_date=_days_ago(5),
            last_buy_strategy="sma_crossover",
            last_sell_date=_days_ago(4),
        )
        self.assertFalse(h.was_recently_traded("VRSN", days=2))


class TestRecentTradeReason(unittest.TestCase):
    """recent_trade_reason() returns descriptive messages."""

    def _history_with(self, **kwargs) -> TradeHistory:
        th = TradeHistory()
        th.by_ticker["VRSN"] = TickerHistory(
            ticker="VRSN", **kwargs
        )
        return th

    def test_no_history_returns_none(self) -> None:
        h = TradeHistory()
        self.assertIsNone(h.recent_trade_reason("VRSN", days=2))

    def test_recent_sell_reason(self) -> None:
        h = self._history_with(last_sell_date=_days_ago(1))
        reason = h.recent_trade_reason("VRSN", days=2)
        self.assertIsNotNone(reason)
        self.assertIn("sold", reason)
        self.assertIn("post-exit cooldown", reason)

    def test_recent_buy_reason(self) -> None:
        h = self._history_with(
            last_buy_date=_days_ago(1),
            last_buy_strategy="sma_crossover",
        )
        reason = h.recent_trade_reason("VRSN", days=2)
        self.assertIsNotNone(reason)
        self.assertIn("bought", reason)
        self.assertIn("sma_crossover", reason)

    def test_sell_takes_priority_when_both_recent(self) -> None:
        """When both buy and sell are recent, sell reason appears
        first since it's the more common churn pattern."""
        h = self._history_with(
            last_buy_date=_days_ago(1),
            last_buy_strategy="sma_crossover",
            last_sell_date=_days_ago(0),
        )
        reason = h.recent_trade_reason("VRSN", days=2)
        self.assertIn("sold", reason)

    def test_old_dates_return_none(self) -> None:
        h = self._history_with(
            last_buy_date=_days_ago(5),
            last_buy_strategy="sma_crossover",
            last_sell_date=_days_ago(4),
        )
        self.assertIsNone(h.recent_trade_reason("VRSN", days=2))

    def test_invalid_date_returns_none(self) -> None:
        h = self._history_with(
            last_buy_date="not-a-date",
            last_sell_date="also-bad",
        )
        self.assertIsNone(h.recent_trade_reason("VRSN", days=2))


class TestLoadTradeHistoryDryRunFlag(unittest.TestCase):
    """load_trade_history() should only count dry-run orders toward
    cooldown dates when include_dry_runs=True."""

    def _write_log(self, tmpdir, orders, executed_at=None):
        import json, os
        from pathlib import Path

        if executed_at is None:
            executed_at = _iso(datetime.now())
        data = {"executed_at": executed_at, "orders": orders}
        # Use timestamp in filename to avoid collisions
        fname = f"execution_{executed_at.replace(':', '')}.json"
        p = Path(tmpdir) / fname
        p.write_text(json.dumps(data))

    def test_dry_run_sell_ignored_by_default(self) -> None:
        """Without include_dry_runs, dry-run sells don't set
        last_sell_date — so they can't block live buys."""
        import tempfile
        from pathlib import Path
        from trading_bot_bl.history import load_trade_history

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_log(tmpdir, [{
                "ticker": "VRSN", "status": "dry_run",
                "side": "sell", "notional": 5000,
                "error": "", "strategy": "test",
            }])
            h = load_trade_history(Path(tmpdir), lookback_days=30)
            th = h.by_ticker.get("VRSN")
            self.assertIsNotNone(th)
            self.assertEqual(th.last_sell_date, "")
            self.assertIsNone(
                h.recent_trade_reason("VRSN", days=2)
            )

    def test_dry_run_sell_counted_when_opted_in(self) -> None:
        """With include_dry_runs=True, dry-run sells DO set
        last_sell_date — accurate for dry-run simulation."""
        import tempfile
        from pathlib import Path
        from trading_bot_bl.history import load_trade_history

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_log(tmpdir, [{
                "ticker": "VRSN", "status": "dry_run",
                "side": "sell", "notional": 5000,
                "error": "", "strategy": "test",
            }])
            h = load_trade_history(
                Path(tmpdir), lookback_days=30,
                include_dry_runs=True,
            )
            th = h.by_ticker.get("VRSN")
            self.assertIsNotNone(th)
            self.assertNotEqual(th.last_sell_date, "")
            reason = h.recent_trade_reason("VRSN", days=2)
            self.assertIn("sold", reason)

    def test_submitted_always_counted(self) -> None:
        """Submitted orders always count, regardless of flag."""
        import tempfile
        from pathlib import Path
        from trading_bot_bl.history import load_trade_history

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_log(tmpdir, [{
                "ticker": "VRSN", "status": "submitted",
                "side": "buy", "notional": 5000,
                "error": "", "strategy": "test",
            }])
            # Without flag
            h = load_trade_history(Path(tmpdir), lookback_days=30)
            self.assertNotEqual(
                h.by_ticker["VRSN"].last_buy_date, ""
            )

    def test_dry_run_buy_ignored_by_default(self) -> None:
        """Dry-run buys don't set last_buy_date by default."""
        import tempfile
        from pathlib import Path
        from trading_bot_bl.history import load_trade_history

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_log(tmpdir, [{
                "ticker": "VRSN", "status": "dry_run",
                "side": "buy", "notional": 5000,
                "error": "", "strategy": "test",
            }])
            h = load_trade_history(Path(tmpdir), lookback_days=30)
            th = h.by_ticker.get("VRSN")
            self.assertIsNotNone(th)
            self.assertEqual(th.last_buy_date, "")


if __name__ == "__main__":
    unittest.main()
