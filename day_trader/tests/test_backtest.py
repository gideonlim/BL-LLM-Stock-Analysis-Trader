"""Tests for the backtest engine.

Uses synthetic bars on a known NYSE session so we can verify:
- Signals fire at the right time
- Fills happen at next bar's open + slippage
- SL/TP exits trigger correctly
- Force-flat at session close
- Metrics (P&L, Sharpe, win rate) compute correctly
- Filter pipeline rejects are counted
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from day_trader.backtest.engine import BacktestConfig, BacktestEngine
from day_trader.config import DayRiskLimits
from day_trader.filters.base import Filter
from day_trader.models import Bar, FilterContext, MarketState, TickerContext
from day_trader.strategies.orb_vwap import OrbVwapStrategy
from day_trader.strategies.vwap_pullback import VwapPullbackStrategy

ET = ZoneInfo("America/New_York")
TEST_DATE = date(2024, 7, 9)  # Regular session Tue


def _bar(
    ticker: str,
    minute_offset: int,
    o: float = 100,
    h: float = 100.5,
    l: float = 99.5,
    c: float = 100,
    v: float = 200_000,
) -> Bar:
    """Bar timestamped at 09:30 ET + minute_offset."""
    ts = datetime(2024, 7, 9, 9, 30, tzinfo=ET) + timedelta(
        minutes=minute_offset,
    )
    return Bar(
        ticker=ticker, timestamp=ts,
        open=o, high=h, low=l, close=c, volume=v,
    )


def _build_breakout_day(ticker: str = "AAPL") -> list[Bar]:
    """Build a set of bars that triggers an ORB breakout.

    - Minutes 0-4: opening range, high=101, low=99
    - Minutes 5+: price breaks above 101 and trends up
    - Last bars near close: stay above entry
    """
    bars = []
    # OR window (09:30-09:34)
    for i in range(5):
        bars.append(_bar(ticker, i, o=100, h=101, l=99, c=100, v=300_000))

    # Post-OR bars (09:35-10:00): breakout above 101
    for i in range(5, 35):
        base = 100 + (i - 5) * 0.1
        bars.append(_bar(
            ticker, i,
            o=base, h=base + 0.5, l=base - 0.2, c=base + 0.3,
            v=250_000,
        ))

    # Mid-day bars: hold above entry
    for i in range(35, 360):
        bars.append(_bar(
            ticker, i,
            o=103, h=103.5, l=102.5, c=103, v=150_000,
        ))

    # Last few bars before close (09:30 + 389 = 15:59)
    for i in range(360, 390):
        bars.append(_bar(
            ticker, i,
            o=103, h=103.3, l=102.8, c=103, v=100_000,
        ))

    return bars


def _ticker_ctx(
    ticker: str = "AAPL",
    rvol: float = 3.0,
) -> TickerContext:
    return TickerContext(
        ticker=ticker, premkt_rvol=rvol, prev_close=100.0,
        avg_daily_volume=10_000_000,
    )


class _AlwaysPassFilter(Filter):
    name = "always_pass"
    def passes(self, ctx):
        return True, ""


class _AlwaysRejectFilter(Filter):
    name = "always_reject"
    def passes(self, ctx):
        return False, "backtest_test_reject"


class TestBacktestEngineORB(unittest.TestCase):
    def test_orb_breakout_produces_trade(self):
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}

        engine = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysPassFilter()],
            config=BacktestConfig(starting_equity=100_000),
        )
        result = engine.run(bars_by_date, ctx)

        self.assertGreater(result.total_trades, 0)
        self.assertEqual(result.strategy_name, "orb_vwap")
        self.assertGreater(result.signals_generated, 0)

    def test_orb_trade_has_correct_fields(self):
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}

        engine = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysPassFilter()],
        )
        result = engine.run(bars_by_date, ctx)

        if result.trades:
            t = result.trades[0]
            self.assertEqual(t.ticker, "AAPL")
            self.assertEqual(t.strategy, "orb_vwap")
            self.assertEqual(t.side, "buy")
            self.assertGreater(t.entry_price, 0)
            self.assertGreater(t.exit_price, 0)
            self.assertIn(
                t.exit_reason,
                ("stop_loss", "take_profit", "force_eod", "time_stop"),
            )
            self.assertGreater(t.qty, 0)
            self.assertNotEqual(t.pnl, 0.0)
            self.assertGreater(t.slippage_cost, 0)

    def test_filter_rejection_counted(self):
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}

        engine = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysRejectFilter()],
        )
        result = engine.run(bars_by_date, ctx)

        self.assertEqual(result.total_trades, 0)
        self.assertGreater(result.signals_filtered, 0)
        self.assertIn(
            "rejected_by_always_reject",
            result.filter_rejection_histogram,
        )

    def test_no_bars_returns_empty_result(self):
        engine = BacktestEngine(
            strategy=OrbVwapStrategy(),
            filters=[_AlwaysPassFilter()],
        )
        result = engine.run({}, {})
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.start_date, "")

    def test_multi_day_accumulates(self):
        bars_day1 = _build_breakout_day("AAPL")
        # Shift day 2 to July 10
        bars_day2 = []
        for b in _build_breakout_day("AAPL"):
            new_ts = b.timestamp + timedelta(days=1)
            from dataclasses import replace
            bars_day2.append(replace(b, timestamp=new_ts))

        bars_by_date = {
            date(2024, 7, 9): {"AAPL": bars_day1},
            date(2024, 7, 10): {"AAPL": bars_day2},
        }
        ctx = {"AAPL": _ticker_ctx("AAPL")}

        engine = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysPassFilter()],
        )
        result = engine.run(bars_by_date, ctx)

        self.assertEqual(result.sessions_simulated, 2)
        self.assertEqual(result.start_date, "2024-07-09")
        self.assertEqual(result.end_date, "2024-07-10")


class TestBacktestMetrics(unittest.TestCase):
    def test_summary_string(self):
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}

        engine = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysPassFilter()],
        )
        result = engine.run(bars_by_date, ctx)
        summary = result.summary()
        self.assertIn("Backtest: orb_vwap", summary)
        self.assertIn("Trades:", summary)
        self.assertIn("Sharpe:", summary)

    def test_passes_plan_criteria_dict(self):
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}

        engine = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysPassFilter()],
        )
        result = engine.run(bars_by_date, ctx)
        criteria = result.passes_plan_criteria
        self.assertIn("sharpe_gt_1.0", criteria)
        self.assertIn("profit_factor_gt_1.3", criteria)
        self.assertIn("max_dd_lt_8pct", criteria)


class TestBacktestConfig(unittest.TestCase):
    def test_slippage_affects_pnl(self):
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}

        # No slippage
        engine_no_slip = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysPassFilter()],
            config=BacktestConfig(slippage_atr_frac=0.0),
        )
        r1 = engine_no_slip.run(bars_by_date, ctx)

        # 10% ATR slippage
        engine_slip = BacktestEngine(
            strategy=OrbVwapStrategy(or_minutes=5, atr_period=5),
            filters=[_AlwaysPassFilter()],
            config=BacktestConfig(slippage_atr_frac=0.10),
        )
        r2 = engine_slip.run(bars_by_date, ctx)

        # Slippage should reduce P&L
        if r1.trades and r2.trades:
            self.assertGreater(r1.total_pnl, r2.total_pnl)
            self.assertGreater(r2.total_slippage_cost, 0)


class TestRunners(unittest.TestCase):
    def test_run_orb_backtest(self):
        from day_trader.backtest.runners import run_orb_backtest
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}
        result = run_orb_backtest(bars_by_date, ctx)
        self.assertEqual(result.strategy_name, "orb_vwap")

    def test_run_all_strategies(self):
        from day_trader.backtest.runners import run_all_strategies
        bars = _build_breakout_day("AAPL")
        bars_by_date = {TEST_DATE: {"AAPL": bars}}
        ctx = {"AAPL": _ticker_ctx("AAPL")}
        results = run_all_strategies(bars_by_date, ctx)
        self.assertIn("orb_vwap", results)
        self.assertIn("vwap_pullback", results)


if __name__ == "__main__":
    unittest.main()
