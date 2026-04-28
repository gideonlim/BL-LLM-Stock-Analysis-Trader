"""Tests for DayRiskManager."""

from __future__ import annotations

import unittest
from datetime import datetime

from day_trader.budget import SubBudgetTracker
from day_trader.config import DayRiskLimits
from day_trader.filters.cooldown import CooldownTracker
from day_trader.models import DayTradeSignal
from day_trader.risk import (
    DayRiskManager,
    REASON_BUDGET_EXCEEDED,
    REASON_KILL_SWITCH,
    REASON_MAX_POSITIONS,
    REASON_MAX_TRADES_TODAY,
    REASON_PER_TRADE_RISK,
    REASON_POSITION_SIZE,
)


def _signal(ticker="AAPL", strategy="ORB", price=100.0, sl=95.0):
    return DayTradeSignal(
        ticker=ticker, strategy=strategy, side="buy",
        signal_price=price, stop_loss_price=sl,
        take_profit_price=price + (price - sl) * 2,
        atr=(price - sl), rvol=2.5,
    )


def _make_rm(equity=100_000.0, **limit_overrides):
    limits = DayRiskLimits(**limit_overrides)
    rm = DayRiskManager(
        limits=limits,
        budget=SubBudgetTracker(budget_pct=limits.budget_pct),
        cooldowns=CooldownTracker(
            ticker_minutes=limits.ticker_cooldown_minutes,
            strategy_minutes=limits.strategy_cooldown_minutes,
        ),
    )
    rm.start_session(equity=equity)
    return rm


class TestStartSession(unittest.TestCase):
    def test_initializes_counters(self):
        rm = _make_rm(equity=100_000.0)
        self.assertEqual(rm.session_starting_equity, 100_000.0)
        self.assertEqual(rm.daily_realized_pnl, 0.0)
        self.assertEqual(rm.open_positions_count, 0)
        self.assertEqual(rm.trades_today, 0)
        self.assertFalse(rm.kill_switch_tripped)
        self.assertAlmostEqual(rm.budget.budget, 25_000.0)

    def test_seeds_open_positions_from_recovery(self):
        rm = DayRiskManager(
            limits=DayRiskLimits(),
            budget=SubBudgetTracker(),
            cooldowns=CooldownTracker(),
        )
        rm.start_session(
            equity=100_000.0,
            initial_open_notional=8_000.0,
            initial_positions=2,
        )
        self.assertEqual(rm.open_positions_count, 2)
        self.assertAlmostEqual(rm.budget.open_notional, 8_000.0)

    def test_negative_equity_rejected(self):
        rm = DayRiskManager(
            limits=DayRiskLimits(),
            budget=SubBudgetTracker(),
            cooldowns=CooldownTracker(),
        )
        with self.assertRaises(ValueError):
            rm.start_session(equity=-1.0)


class TestReview(unittest.TestCase):
    def test_approves_normal_signal(self):
        rm = _make_rm(equity=100_000.0)
        verdict = rm.review(
            signal=_signal(),
            intent_notional=1000.0,
            risk_dollars=50.0,
        )
        self.assertTrue(verdict.approved)

    def test_rejects_when_kill_switch_tripped(self):
        rm = _make_rm()
        rm.trip_kill_switch("manual_halt")
        verdict = rm.review(
            signal=_signal(),
            intent_notional=1000.0,
            risk_dollars=50.0,
        )
        self.assertFalse(verdict.approved)
        self.assertIn(REASON_KILL_SWITCH, verdict.reason)

    def test_rejects_at_max_positions(self):
        rm = _make_rm(max_positions=3)
        rm.open_positions_count = 3
        verdict = rm.review(
            signal=_signal(), intent_notional=1000.0, risk_dollars=50.0,
        )
        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.reason, REASON_MAX_POSITIONS)

    def test_rejects_at_max_trades_today(self):
        rm = _make_rm(max_trades_per_day=8)
        rm.trades_today = 8
        verdict = rm.review(
            signal=_signal(), intent_notional=1000.0, risk_dollars=50.0,
        )
        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.reason, REASON_MAX_TRADES_TODAY)

    def test_rejects_per_trade_risk_exceeded(self):
        rm = _make_rm(equity=100_000.0, per_trade_risk_pct=0.25)
        # max risk = 100k * 0.25% = $250
        verdict = rm.review(
            signal=_signal(), intent_notional=1000.0, risk_dollars=300.0,
        )
        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.reason, REASON_PER_TRADE_RISK)

    def test_rejects_position_size_exceeds_max(self):
        rm = _make_rm(
            equity=100_000.0,
            max_position_pct_of_budget=50.0,
        )
        # budget = 25k, max position = 12.5k
        verdict = rm.review(
            signal=_signal(),
            intent_notional=15_000.0,
            risk_dollars=50.0,
        )
        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.reason, REASON_POSITION_SIZE)

    def test_rejects_budget_exceeded(self):
        rm = _make_rm(equity=100_000.0)
        # Reserve almost the whole budget
        rm.budget.reserve(24_000)
        verdict = rm.review(
            signal=_signal(),
            intent_notional=2_000.0,  # would push us over 25k
            risk_dollars=50.0,
        )
        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.reason, REASON_BUDGET_EXCEEDED)

    def test_rejects_severe_bear(self):
        rm = _make_rm()
        rm.set_spy_severe_bear(True)
        verdict = rm.review(
            signal=_signal(), intent_notional=1000.0, risk_dollars=50.0,
        )
        self.assertFalse(verdict.approved)
        self.assertIn("spy_severe_bear", verdict.reason)


class TestRecordFill(unittest.TestCase):
    def test_increments_counters(self):
        rm = _make_rm()
        ok = rm.record_fill(notional=1000.0)
        self.assertTrue(ok)
        self.assertEqual(rm.open_positions_count, 1)
        self.assertEqual(rm.trades_today, 1)
        self.assertAlmostEqual(rm.budget.open_notional, 1000.0)


class TestRecordClose(unittest.TestCase):
    def test_release_decrements_position_count(self):
        rm = _make_rm()
        rm.record_fill(notional=1000.0)
        rm.record_close(
            ticker="AAPL", strategy="ORB",
            pnl=50.0, entry_notional=1000.0,
        )
        self.assertEqual(rm.open_positions_count, 0)
        self.assertAlmostEqual(rm.daily_realized_pnl, 50.0)
        self.assertAlmostEqual(rm.budget.open_notional, 0.0)

    def test_loss_populates_cooldown(self):
        rm = _make_rm()
        rm.record_fill(notional=1000.0)
        rm.record_close(
            ticker="AAPL", strategy="ORB",
            pnl=-50.0, entry_notional=1000.0,
        )
        cooled, _ = rm.cooldowns.is_cooled_down(
            ticker="AAPL", strategy="ORB",
        )
        self.assertTrue(cooled)

    def test_kill_switch_trips_at_daily_loss_threshold(self):
        rm = _make_rm(equity=100_000.0, daily_loss_limit_pct=1.5)
        # Need to lose >$1500 (1.5% of $100k)
        rm.record_fill(notional=10_000.0)
        rm.record_close(
            ticker="AAPL", strategy="ORB",
            pnl=-1600.0, entry_notional=10_000.0,
        )
        self.assertTrue(rm.kill_switch_tripped)
        self.assertIn("daily_loss", rm.kill_switch_reason)

    def test_kill_switch_doesnt_trip_below_threshold(self):
        rm = _make_rm(equity=100_000.0, daily_loss_limit_pct=1.5)
        rm.record_fill(notional=10_000.0)
        rm.record_close(
            ticker="AAPL", strategy="ORB",
            pnl=-1000.0, entry_notional=10_000.0,
        )
        self.assertFalse(rm.kill_switch_tripped)

    def test_subsequent_review_rejected_after_kill(self):
        rm = _make_rm(equity=100_000.0, daily_loss_limit_pct=1.5)
        rm.record_fill(notional=10_000.0)
        rm.record_close(
            ticker="AAPL", strategy="ORB",
            pnl=-2000.0, entry_notional=10_000.0,
        )
        verdict = rm.review(
            signal=_signal("MSFT", "VWAP_PULLBACK"),
            intent_notional=1000.0, risk_dollars=50.0,
        )
        self.assertFalse(verdict.approved)


class TestCanTakeMoreTrades(unittest.TestCase):
    def test_true_when_room(self):
        rm = _make_rm()
        self.assertTrue(rm.can_take_more_trades())

    def test_false_at_max_positions(self):
        rm = _make_rm(max_positions=3)
        rm.open_positions_count = 3
        self.assertFalse(rm.can_take_more_trades())

    def test_false_after_kill(self):
        rm = _make_rm()
        rm.trip_kill_switch("test")
        self.assertFalse(rm.can_take_more_trades())


if __name__ == "__main__":
    unittest.main()
