"""Tests for SubBudgetTracker."""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor

from day_trader.budget import SubBudgetTracker


class TestSubBudgetTracker(unittest.TestCase):
    def test_budget_is_pct_of_equity(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        # Default 25%
        self.assertAlmostEqual(t.budget, 25_000.0)
        self.assertEqual(t.open_notional, 0.0)
        self.assertAlmostEqual(t.headroom, 25_000.0)

    def test_custom_pct(self):
        t = SubBudgetTracker(budget_pct=0.10)
        t.start_session(equity=50_000.0)
        self.assertAlmostEqual(t.budget, 5_000.0)

    def test_reserve_decrements_headroom(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        ok = t.reserve(10_000)
        self.assertTrue(ok)
        self.assertAlmostEqual(t.open_notional, 10_000)
        self.assertAlmostEqual(t.headroom, 15_000)

    def test_reserve_at_ceiling_succeeds(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        # Exactly the budget — should succeed (boundary condition)
        self.assertTrue(t.reserve(25_000))
        self.assertAlmostEqual(t.headroom, 0.0)

    def test_reserve_above_ceiling_fails(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        self.assertTrue(t.reserve(20_000))
        # Adding 6_000 more would total 26_000 > budget(25_000)
        self.assertFalse(t.reserve(6_000))
        # Open didn't change
        self.assertAlmostEqual(t.open_notional, 20_000)

    def test_release_increments_headroom(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        t.reserve(15_000)
        t.release(5_000)
        self.assertAlmostEqual(t.open_notional, 10_000)
        self.assertAlmostEqual(t.headroom, 15_000)

    def test_release_clamps_to_zero(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        t.reserve(5_000)
        # Release more than open — must clamp, not go negative
        t.release(10_000)
        self.assertEqual(t.open_notional, 0.0)

    def test_initial_open_notional_seeded_from_recovery(self):
        # Simulates the executor calling start_session() with the
        # open notional that recovery.reconcile() determined.
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0, initial_open_notional=8_000)
        self.assertAlmostEqual(t.open_notional, 8_000)
        self.assertAlmostEqual(t.headroom, 17_000)

    def test_negative_equity_rejected(self):
        t = SubBudgetTracker()
        with self.assertRaises(ValueError):
            t.start_session(equity=-1_000.0)

    def test_negative_initial_notional_rejected(self):
        t = SubBudgetTracker()
        with self.assertRaises(ValueError):
            t.start_session(equity=100_000.0, initial_open_notional=-500.0)

    def test_negative_reserve_rejected(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        with self.assertRaises(ValueError):
            t.reserve(-100)

    def test_negative_release_rejected(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        with self.assertRaises(ValueError):
            t.release(-50)

    def test_can_reserve_does_not_mutate(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        # Multiple checks shouldn't actually reserve
        self.assertTrue(t.can_reserve(10_000))
        self.assertTrue(t.can_reserve(10_000))
        self.assertEqual(t.open_notional, 0.0)

    def test_thread_safety_no_overcommit(self):
        """Concurrent reservations must never exceed budget."""
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        # Budget is 25_000. We try 50 concurrent reserves of $1000 each.
        # Only 25 should succeed (25 * 1000 = 25_000 ceiling).
        successes: list[bool] = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(t.reserve, 1_000) for _ in range(50)]
            for f in futures:
                successes.append(f.result())
        self.assertEqual(sum(successes), 25)
        self.assertAlmostEqual(t.open_notional, 25_000)

    def test_utilization_pct(self):
        t = SubBudgetTracker()
        t.start_session(equity=100_000.0)
        t.reserve(12_500)  # half the 25k budget
        self.assertAlmostEqual(t.utilization, 0.5)

    def test_utilization_zero_equity(self):
        t = SubBudgetTracker()
        # Don't call start_session — equity stays 0
        self.assertEqual(t.utilization, 0.0)


if __name__ == "__main__":
    unittest.main()
