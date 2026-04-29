"""Tests for in-sample / out-of-sample backtest split."""

from __future__ import annotations

import unittest
from datetime import date, timedelta

from day_trader.backtest.runners import (
    IsOosResult,
    split_dates_is_oos,
)
from day_trader.backtest.engine import BacktestResult


class TestSplitDatesIsOos(unittest.TestCase):
    def test_default_75_25_split(self):
        bars_by_date = {
            date(2024, 1, 1) + timedelta(days=i): {"AAPL": []}
            for i in range(100)
        }
        is_bars, oos_bars = split_dates_is_oos(bars_by_date)
        self.assertEqual(len(is_bars), 75)
        self.assertEqual(len(oos_bars), 25)

    def test_oos_dates_after_is_dates(self):
        bars_by_date = {
            date(2024, 1, 1) + timedelta(days=i): {"AAPL": []}
            for i in range(100)
        }
        is_bars, oos_bars = split_dates_is_oos(bars_by_date)
        max_is = max(is_bars)
        min_oos = min(oos_bars)
        self.assertLess(max_is, min_oos)

    def test_custom_oos_pct(self):
        bars_by_date = {
            date(2024, 1, 1) + timedelta(days=i): {"AAPL": []}
            for i in range(100)
        }
        is_bars, oos_bars = split_dates_is_oos(
            bars_by_date, oos_pct=0.10,
        )
        self.assertEqual(len(is_bars), 90)
        self.assertEqual(len(oos_bars), 10)

    def test_invalid_oos_pct_rejected(self):
        with self.assertRaises(ValueError):
            split_dates_is_oos({}, oos_pct=0.0)
        with self.assertRaises(ValueError):
            split_dates_is_oos({}, oos_pct=1.0)
        with self.assertRaises(ValueError):
            split_dates_is_oos({}, oos_pct=-0.1)

    def test_single_date_returns_full_in_sample(self):
        bars = {date(2024, 1, 1): {"AAPL": []}}
        is_bars, oos_bars = split_dates_is_oos(bars)
        self.assertEqual(is_bars, bars)
        self.assertEqual(oos_bars, {})


class TestIsOosResult(unittest.TestCase):
    def _result(
        self,
        sharpe: float = 0.0,
        pf: float = 0.0,
        dd: float = 100.0,
    ) -> BacktestResult:
        return BacktestResult(
            strategy_name="x", start_date="", end_date="",
            sharpe_ratio=sharpe,
            profit_factor=pf,
            max_drawdown_pct=dd,
        )

    def test_passes_when_all_oos_thresholds_met(self):
        result = IsOosResult(
            strategy_name="orb",
            in_sample=self._result(sharpe=2.0, pf=2.0, dd=3.0),
            out_of_sample=self._result(sharpe=1.5, pf=1.5, dd=5.0),
        )
        self.assertTrue(result.oos_passes_plan)

    def test_fails_on_low_sharpe(self):
        result = IsOosResult(
            strategy_name="orb",
            in_sample=self._result(sharpe=2.0, pf=2.0, dd=3.0),
            out_of_sample=self._result(sharpe=0.5, pf=1.5, dd=5.0),
        )
        self.assertFalse(result.oos_passes_plan)

    def test_fails_on_low_profit_factor(self):
        result = IsOosResult(
            strategy_name="orb",
            in_sample=self._result(sharpe=2.0, pf=2.0, dd=3.0),
            out_of_sample=self._result(sharpe=1.5, pf=1.0, dd=5.0),
        )
        self.assertFalse(result.oos_passes_plan)

    def test_fails_on_high_drawdown(self):
        result = IsOosResult(
            strategy_name="orb",
            in_sample=self._result(sharpe=2.0, pf=2.0, dd=3.0),
            out_of_sample=self._result(sharpe=1.5, pf=1.5, dd=10.0),
        )
        self.assertFalse(result.oos_passes_plan)

    def test_summary_includes_pass_fail(self):
        passing = IsOosResult(
            strategy_name="orb",
            in_sample=self._result(sharpe=2.0, pf=2.0, dd=3.0),
            out_of_sample=self._result(sharpe=1.5, pf=1.5, dd=5.0),
        )
        self.assertIn("PASS", passing.summary())

        failing = IsOosResult(
            strategy_name="orb",
            in_sample=self._result(),
            out_of_sample=self._result(),
        )
        self.assertIn("FAIL", failing.summary())


if __name__ == "__main__":
    unittest.main()
