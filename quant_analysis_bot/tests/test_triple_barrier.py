"""Tests for triple barrier labeling engine and CUSUM filter."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.triple_barrier import (
    BarrierTrade,
    apply_triple_barrier,
    cusum_filter,
    score_barrier_trades,
)


def _make_price_data(
    n: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create synthetic OHLC data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    return close, high, low, dates


class TestCusumFilter(unittest.TestCase):
    """Tests for the symmetric CUSUM event filter."""

    def test_returns_boolean_mask(self):
        close, _, _, _ = _make_price_data(200)
        result = cusum_filter(close, threshold=0.02)
        self.assertEqual(result.dtype, bool)
        self.assertEqual(len(result), 200)

    def test_no_events_on_flat_price(self):
        close = np.full(100, 50.0)
        result = cusum_filter(close, threshold=0.01)
        self.assertFalse(result.any())

    def test_lower_threshold_fires_more(self):
        close, _, _, _ = _make_price_data(500, seed=7)
        events_tight = cusum_filter(close, threshold=0.005)
        events_loose = cusum_filter(close, threshold=0.05)
        self.assertGreaterEqual(events_tight.sum(), events_loose.sum())

    def test_first_bar_never_fires(self):
        close, _, _, _ = _make_price_data()
        result = cusum_filter(close, threshold=0.001)
        self.assertFalse(result[0])

    def test_empty_input(self):
        result = cusum_filter(np.array([]), threshold=0.01)
        self.assertEqual(len(result), 0)

    def test_single_bar(self):
        result = cusum_filter(np.array([100.0]), threshold=0.01)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0])

    def test_zero_threshold_returns_no_events(self):
        close, _, _, _ = _make_price_data()
        result = cusum_filter(close, threshold=0.0)
        self.assertFalse(result.any())


class TestApplyTripleBarrier(unittest.TestCase):
    """Tests for the triple barrier labeling engine."""

    def test_tp_hit(self):
        """Price rises 10% — should hit 5% TP barrier."""
        n = 20
        close = np.linspace(100, 110, n)
        high = close * 1.005
        low = close * 0.995
        dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
        entries = np.zeros(n, dtype=bool)
        entries[0] = True

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=5.0, tp_pct=5.0, max_holding_bars=50,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_barrier, "upper")
        self.assertEqual(trades[0].label, 1)

    def test_sl_hit(self):
        """Price drops 10% — should hit 5% SL barrier."""
        n = 20
        close = np.linspace(100, 90, n)
        high = close * 1.005
        low = close * 0.995
        dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
        entries = np.zeros(n, dtype=bool)
        entries[0] = True

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=5.0, tp_pct=5.0, max_holding_bars=50,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_barrier, "lower")
        self.assertEqual(trades[0].label, -1)

    def test_vertical_barrier(self):
        """Flat price — should hit time barrier."""
        n = 30
        close = np.full(n, 100.0)
        high = np.full(n, 100.5)
        low = np.full(n, 99.5)
        dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
        entries = np.zeros(n, dtype=bool)
        entries[0] = True

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=5.0, tp_pct=5.0, max_holding_bars=10,
            cost_bps=0,  # zero costs so flat price → negligible return
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_barrier, "vertical")
        # Negligible return with zero costs → label 0
        self.assertEqual(trades[0].label, 0)

    def test_both_barriers_same_bar_sl_wins(self):
        """When both SL and TP are hit on same bar, SL wins."""
        n = 5
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        # Bar 1: huge range hits both barriers
        high = np.array([100.0, 120.0, 100.0, 100.0, 100.0])
        low = np.array([100.0, 80.0, 100.0, 100.0, 100.0])
        dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
        entries = np.zeros(n, dtype=bool)
        entries[0] = True

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=5.0, tp_pct=5.0, max_holding_bars=50,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_barrier, "lower")

    def test_no_entries_returns_empty(self):
        close, high, low, dates = _make_price_data(50)
        entries = np.zeros(50, dtype=bool)
        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=5.0, tp_pct=5.0,
        )
        self.assertEqual(trades, [])

    def test_multiple_entries(self):
        close, high, low, dates = _make_price_data(200)
        entries = np.zeros(200, dtype=bool)
        entries[10] = True
        entries[50] = True
        entries[100] = True

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=3.0, tp_pct=6.0, max_holding_bars=30,
        )
        self.assertEqual(len(trades), 3)

    def test_per_trade_barrier_widths(self):
        """Each entry can have different SL/TP."""
        n = 50
        close, high, low, dates = _make_price_data(n)
        entries = np.zeros(n, dtype=bool)
        entries[5] = True
        entries[25] = True

        sl_arr = np.full(n, 3.0)
        sl_arr[25] = 8.0  # wider stop for second trade
        tp_arr = np.full(n, 6.0)

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=sl_arr, tp_pct=tp_arr, max_holding_bars=30,
        )
        self.assertEqual(len(trades), 2)

    def test_excursion_data_populated(self):
        close, high, low, dates = _make_price_data(100)
        entries = np.zeros(100, dtype=bool)
        entries[5] = True

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=5.0, tp_pct=10.0, max_holding_bars=50,
        )
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertGreaterEqual(trade.mfe_pct, 0)
        self.assertGreaterEqual(trade.mae_pct, 0)
        self.assertGreaterEqual(trade.mfe_bar, 0)
        self.assertGreaterEqual(trade.mae_bar, 0)

    def test_empty_input(self):
        trades = apply_triple_barrier(
            np.array([]), np.array([]), np.array([]),
            pd.DatetimeIndex([]),
            np.array([], dtype=bool),
            sl_pct=5.0, tp_pct=5.0,
        )
        self.assertEqual(trades, [])

    def test_transaction_costs_applied(self):
        """Returns should reflect transaction costs."""
        n = 5
        close = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        high = np.array([100.0, 106.0, 100.0, 100.0, 100.0])
        low = np.array([100.0, 99.0, 100.0, 100.0, 100.0])
        dates = pd.bdate_range("2025-01-01", periods=n, freq="B")
        entries = np.zeros(n, dtype=bool)
        entries[0] = True

        trades = apply_triple_barrier(
            close, high, low, dates, entries,
            sl_pct=5.0, tp_pct=5.0, max_holding_bars=50,
            cost_bps=100,  # 1% cost
        )
        self.assertEqual(len(trades), 1)
        # TP hit at 105, return = 5% - 1% cost = ~4%
        self.assertLess(trades[0].return_pct, 5.0)


class TestScoreBarrierTrades(unittest.TestCase):
    """Tests for barrier trade scoring."""

    def test_empty_trades(self):
        metrics = score_barrier_trades([])
        self.assertEqual(metrics.total_trades, 0)

    def test_all_winners(self):
        trades = [
            BarrierTrade(
                entry_idx=0, entry_date="2025-01-01", entry_price=100,
                exit_idx=5, exit_date="2025-01-08", exit_price=105,
                holding_days=5, return_pct=4.9, label=1,
                exit_barrier="upper", mfe_pct=5.0, mae_pct=1.0,
                mfe_bar=3, mae_bar=1,
            )
            for _ in range(5)
        ]
        metrics = score_barrier_trades(trades)
        self.assertEqual(metrics.total_trades, 5)
        self.assertAlmostEqual(metrics.win_rate, 1.0)
        self.assertAlmostEqual(metrics.sl_rate, 0.0)

    def test_mixed_outcomes(self):
        tp_trade = BarrierTrade(
            entry_idx=0, entry_date="2025-01-01", entry_price=100,
            exit_idx=5, exit_date="2025-01-08", exit_price=105,
            holding_days=5, return_pct=4.9, label=1,
            exit_barrier="upper", mfe_pct=5.0, mae_pct=1.0,
            mfe_bar=3, mae_bar=1,
        )
        sl_trade = BarrierTrade(
            entry_idx=10, entry_date="2025-01-15", entry_price=100,
            exit_idx=15, exit_date="2025-01-22", exit_price=95,
            holding_days=5, return_pct=-5.1, label=-1,
            exit_barrier="lower", mfe_pct=1.0, mae_pct=5.0,
            mfe_bar=1, mae_bar=4,
        )
        vb_trade = BarrierTrade(
            entry_idx=20, entry_date="2025-02-01", entry_price=100,
            exit_idx=30, exit_date="2025-02-15", exit_price=100.5,
            holding_days=10, return_pct=0.4, label=0,
            exit_barrier="vertical", mfe_pct=2.0, mae_pct=1.5,
            mfe_bar=5, mae_bar=7,
        )
        metrics = score_barrier_trades([tp_trade, sl_trade, vb_trade])
        self.assertEqual(metrics.total_trades, 3)
        self.assertAlmostEqual(metrics.win_rate, 1 / 3, places=3)
        self.assertAlmostEqual(metrics.sl_rate, 1 / 3, places=3)
        self.assertAlmostEqual(metrics.timeout_rate, 1 / 3, places=3)
        self.assertGreater(metrics.edge_ratio, 0)


if __name__ == "__main__":
    unittest.main()
