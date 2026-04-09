"""Tests for run_backtest trade lifecycle.

These tests target the walk-forward validation path's handling of
open positions — specifically, whether trades that remain open at
the window boundary are correctly recorded.  Before the fix, the
for-loop in run_backtest exited without closing open positions,
silently discarding them and leaving ``trades_raw`` empty for
sparse-signal strategies like fear-gated mean reversion.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.backtest import run_backtest


def _daily_price_df(n: int, start: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Make a synthetic daily-bar DataFrame with a Close column."""
    rng = np.random.RandomState(seed)
    # Log-normal drift with small noise so trades can plausibly be
    # wins or losses depending on direction and entry bar.
    returns = rng.normal(0.001, 0.01, n)
    close = start * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


class TestRunBacktestOpenPositionClose(unittest.TestCase):
    """Open positions at the window boundary must be closed and recorded."""

    def test_long_position_never_exits_is_still_recorded(self):
        """A strategy that enters long and never generates an exit
        signal should still produce one recorded trade at window close,
        not silently drop the trade.

        Before the fix, this test fails because the for-loop exits
        with the position open and no entry is appended to trades_raw.
        """
        n = 60
        df = _daily_price_df(n)

        # Signal: buy at bar 5, hold forever (no -1 ever).
        signals = pd.Series(0, index=df.index)
        signals.iloc[5] = 1

        result, trade_log, _ = run_backtest(
            df=df,
            signals=signals,
            ticker="TEST",
            strategy_name="HoldForever",
            timeframe="3mo",
            cost_bps=10,
            long_only=True,
            next_bar_execution=True,
        )

        # There should be exactly one trade: the entry at bar 6
        # (next-bar execution from signal at bar 5) closing at
        # the final bar.
        self.assertEqual(
            result.total_trades, 1,
            msg=(
                "Open long position at window boundary was not closed: "
                "trades_raw is empty so no trade was recorded."
            ),
        )
        self.assertEqual(len(trade_log), 1)
        trade = trade_log[0]
        self.assertEqual(trade.direction, "LONG")
        self.assertGreater(trade.holding_days, 0)
        self.assertGreater(result.avg_holding_days, 0)

    def test_short_position_never_exits_is_still_recorded(self):
        """Same end-of-window guarantee for short positions."""
        n = 60
        df = _daily_price_df(n)

        signals = pd.Series(0, index=df.index)
        signals.iloc[5] = -1  # open short, never exits

        result, trade_log, _ = run_backtest(
            df=df,
            signals=signals,
            ticker="TEST",
            strategy_name="ShortForever",
            timeframe="3mo",
            cost_bps=10,
            long_only=False,
            next_bar_execution=True,
        )

        self.assertEqual(
            result.total_trades, 1,
            msg="Open short position at window boundary was not closed",
        )
        self.assertEqual(len(trade_log), 1)
        self.assertEqual(trade_log[0].direction, "SHORT")
        self.assertGreater(trade_log[0].holding_days, 0)

    def test_long_only_short_signal_does_not_create_phantom_trade(self):
        """In long-only mode, a short signal with no prior long should
        not create an open short that then gets closed at the window
        boundary.  This guards against the fix over-recording trades.
        """
        n = 60
        df = _daily_price_df(n)

        signals = pd.Series(0, index=df.index)
        signals.iloc[5] = -1  # short signal, but long_only

        result, trade_log, _ = run_backtest(
            df=df,
            signals=signals,
            ticker="TEST",
            strategy_name="OnlyShortSignals",
            timeframe="3mo",
            cost_bps=10,
            long_only=True,
            next_bar_execution=True,
        )

        # In long-only mode, a -1 signal when flat is a no-op.
        # No position is ever opened, so no trade should be recorded.
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(len(trade_log), 0)

    def test_closed_trade_is_unaffected_by_fix(self):
        """If a trade cleanly exits mid-window via a -1 signal, the
        end-of-window close logic should not touch it.  Behavior
        must be identical before and after the fix for this case.
        """
        n = 60
        df = _daily_price_df(n)

        signals = pd.Series(0, index=df.index)
        signals.iloc[5] = 1   # enter long
        signals.iloc[20] = -1  # exit long

        result, trade_log, _ = run_backtest(
            df=df,
            signals=signals,
            ticker="TEST",
            strategy_name="EntersAndExits",
            timeframe="3mo",
            cost_bps=10,
            long_only=True,
            next_bar_execution=True,
        )

        # Exactly one trade, closed mid-window (not at end).
        self.assertEqual(result.total_trades, 1)
        self.assertEqual(len(trade_log), 1)
        trade = trade_log[0]
        # Exit should be at/around bar 21 (next-bar execution
        # from signal at 20), well before the last bar (59).
        self.assertEqual(trade.direction, "LONG")
        self.assertLess(
            pd.to_datetime(trade.exit_date),
            df.index[-1],
            msg="Trade exit should be before the last bar",
        )

    def test_flat_strategy_has_zero_trades(self):
        """A strategy that never generates any signal should have
        zero trades, confirming the fix only fires when position != 0.
        """
        n = 60
        df = _daily_price_df(n)
        signals = pd.Series(0, index=df.index)

        result, trade_log, _ = run_backtest(
            df=df,
            signals=signals,
            ticker="TEST",
            strategy_name="Flat",
            timeframe="3mo",
            cost_bps=10,
            long_only=True,
            next_bar_execution=True,
        )

        self.assertEqual(result.total_trades, 0)
        self.assertEqual(len(trade_log), 0)
        self.assertEqual(result.avg_holding_days, 0.0)


class TestRunBacktestHoldingDaysForRsiMrPattern(unittest.TestCase):
    """Realistic regression: sparse-signal MR strategy entering
    once within a short validation window and never seeing an exit
    condition.  Mirrors the RSI Mean Reversion pattern that the
    TP experiment revealed to have 0/339 legacy trades.
    """

    def test_rsi_mr_style_single_entry_no_exit(self):
        n = 30  # short validation window (≈ 3mo × 30%)
        df = _daily_price_df(n, seed=7)

        # Enter at bar 8, never see an exit condition.
        signals = pd.Series(0, index=df.index)
        signals.iloc[8] = 1

        result, _, _ = run_backtest(
            df=df,
            signals=signals,
            ticker="TEST",
            strategy_name="RSI_MR_Style",
            timeframe="3mo",
        )

        # Before the fix: result.total_trades == 0 and
        # result.avg_holding_days == 0.0.
        # After the fix: exactly 1 trade with ~20 holding days.
        self.assertEqual(result.total_trades, 1)
        self.assertGreater(result.avg_holding_days, 0)


if __name__ == "__main__":
    unittest.main()
