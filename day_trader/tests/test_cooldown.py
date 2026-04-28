"""Tests for CooldownTracker."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from day_trader.filters.cooldown import CooldownTracker


T0 = datetime(2026, 4, 28, 10, 0, 0)


class TestCooldownTracker(unittest.TestCase):
    def test_no_cooldown_when_no_losses(self):
        c = CooldownTracker()
        cooled, reason = c.is_cooled_down(
            ticker="AAPL", strategy="ORB", now=T0,
        )
        self.assertFalse(cooled)
        self.assertEqual(reason, "")

    def test_loss_creates_ticker_cooldown(self):
        c = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        # Same ticker, just after the loss → cooldown active
        cooled, reason = c.is_cooled_down(
            ticker="AAPL", strategy="ORB", now=T0 + timedelta(minutes=10),
        )
        self.assertTrue(cooled)
        self.assertIn(reason, ("ticker_cooldown", "strategy_cooldown"))

    def test_strategy_cooldown_blocks_other_tickers(self):
        c = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        # Loss on AAPL ORB → strategy cooldown for ORB regardless of ticker
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        # Now try TSLA on the same strategy — should be blocked by
        # strategy cooldown
        cooled, reason = c.is_cooled_down(
            ticker="TSLA", strategy="ORB", now=T0 + timedelta(minutes=15),
        )
        self.assertTrue(cooled)
        self.assertEqual(reason, "strategy_cooldown")

    def test_strategy_cooldown_does_not_block_other_strategy(self):
        c = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        # Different strategy on different ticker — no cooldown
        cooled, _ = c.is_cooled_down(
            ticker="MSFT", strategy="VWAP_PULLBACK",
            now=T0 + timedelta(minutes=15),
        )
        self.assertFalse(cooled)

    def test_ticker_cooldown_blocks_other_strategies(self):
        c = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        # Different strategy, SAME ticker — blocked by ticker cooldown
        cooled, reason = c.is_cooled_down(
            ticker="AAPL", strategy="VWAP_PULLBACK",
            now=T0 + timedelta(minutes=45),  # past strategy cd, in ticker cd
        )
        self.assertTrue(cooled)
        self.assertEqual(reason, "ticker_cooldown")

    def test_win_does_not_create_cooldown(self):
        c = CooldownTracker()
        c.record_close(ticker="AAPL", strategy="ORB", pnl=50.0, when=T0)
        cooled, _ = c.is_cooled_down(
            ticker="AAPL", strategy="ORB", now=T0 + timedelta(minutes=5),
        )
        self.assertFalse(cooled)

    def test_breakeven_does_not_create_cooldown(self):
        c = CooldownTracker()
        c.record_close(ticker="AAPL", strategy="ORB", pnl=0.0, when=T0)
        cooled, _ = c.is_cooled_down(
            ticker="AAPL", strategy="ORB", now=T0 + timedelta(minutes=5),
        )
        self.assertFalse(cooled)

    def test_cooldown_expires(self):
        c = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        # 61 minutes later, both cooldowns expired
        cooled, _ = c.is_cooled_down(
            ticker="AAPL", strategy="ORB", now=T0 + timedelta(minutes=61),
        )
        self.assertFalse(cooled)

    def test_cooldown_remaining(self):
        c = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        # Right after: longest cooldown is ticker (60 min)
        rem = c.cooldown_remaining(
            ticker="AAPL", strategy="ORB", now=T0 + timedelta(minutes=5),
        )
        self.assertIsNotNone(rem)
        self.assertAlmostEqual(rem.total_seconds(), 55 * 60, delta=1)

    def test_reset_for_session_clears_all(self):
        c = CooldownTracker()
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        c.record_close(ticker="MSFT", strategy="VWAP", pnl=-25.0, when=T0)
        c.reset_for_session()
        cooled, _ = c.is_cooled_down(
            ticker="AAPL", strategy="ORB", now=T0 + timedelta(minutes=5),
        )
        self.assertFalse(cooled)

    def test_prune_expired(self):
        c = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        c.record_close(ticker="AAPL", strategy="ORB", pnl=-50.0, when=T0)
        n = c.prune_expired(now=T0 + timedelta(minutes=70))
        # Both ticker and strategy cooldowns expired → 2 entries pruned
        self.assertEqual(n, 2)

    def test_ticker_normalization(self):
        # Lowercase ticker in record / query should normalize to upper
        c = CooldownTracker()
        c.record_close(ticker="aapl", strategy="ORB", pnl=-50.0, when=T0)
        cooled, _ = c.is_cooled_down(
            ticker="AAPL", strategy="ORB", now=T0 + timedelta(minutes=5),
        )
        self.assertTrue(cooled)


if __name__ == "__main__":
    unittest.main()
