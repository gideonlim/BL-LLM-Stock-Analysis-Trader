"""Tests for strategy_monitor -- state machine, metrics, persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from trading_bot_bl.models import JournalEntry, Signal
from trading_bot_bl.strategy_monitor import (
    MonitorThresholds,
    StrategyHealth,
    StrategyMonitor,
    StrategyState,
    _consec_losses,
    _evaluate_transition,
    _load_state,
    _rolling_sharpe,
    _save_state,
    _strategy_drawdown,
    _win_rate,
)


# ── Helpers ─────────────────────────────────────────────────────


def _make_trade(
    strategy: str = "VWAP Trend",
    ticker: str = "AAPL",
    pnl_pct: float = 1.0,
    r_multiple: float = 0.5,
    exit_date: str = "2026-04-01",
) -> JournalEntry:
    return JournalEntry(
        trade_id=f"{ticker}_test",
        ticker=ticker,
        strategy=strategy,
        side="long",
        entry_order_id="ord123",
        entry_signal_price=100.0,
        entry_fill_price=100.0,
        realized_pnl_pct=pnl_pct,
        r_multiple=r_multiple,
        exit_date=exit_date,
        status="closed",
    )


def _make_signal(
    ticker: str = "AAPL",
    strategy: str = "VWAP Trend",
) -> Signal:
    return Signal(
        ticker=ticker, signal="BUY", signal_raw=1,
        strategy=strategy, confidence="MEDIUM",
        confidence_score=3, composite_score=25.0,
        current_price=150.0, stop_loss_price=140.0,
        take_profit_price=170.0, suggested_position_size_pct=5.0,
        signal_expires="2099-12-31", sharpe=1.5, win_rate=60.0,
        total_trades=20,
    )


def _make_trades(pnl_pcts: list[float], strategy: str = "VWAP Trend"):
    """Create a series of closed trades with given P&L percentages."""
    trades = []
    for i, pnl in enumerate(pnl_pcts):
        trades.append(_make_trade(
            strategy=strategy,
            pnl_pct=pnl,
            r_multiple=pnl / 2,
            exit_date=f"2026-04-{i + 1:02d}",
        ))
    return trades


# ── Metric tests ────────────────────────────────────────────────


class TestRollingMetrics(unittest.TestCase):

    def test_rolling_sharpe_positive(self):
        pnls = [2.0, 1.5, 3.0, 1.0, 2.5]
        s = _rolling_sharpe(pnls)
        self.assertGreater(s, 0)

    def test_rolling_sharpe_negative(self):
        pnls = [-2.0, -1.5, -3.0, -1.0, -2.5]
        s = _rolling_sharpe(pnls)
        self.assertLess(s, 0)

    def test_rolling_sharpe_single_trade(self):
        self.assertEqual(_rolling_sharpe([1.0]), 0.0)

    def test_rolling_sharpe_identical_returns(self):
        self.assertEqual(_rolling_sharpe([1.0, 1.0, 1.0]), 0.0)

    def test_win_rate_all_winners(self):
        self.assertEqual(_win_rate([1.0, 2.0, 0.5]), 1.0)

    def test_win_rate_all_losers(self):
        self.assertEqual(_win_rate([-1.0, -2.0, -0.5]), 0.0)

    def test_win_rate_mixed(self):
        self.assertAlmostEqual(
            _win_rate([1.0, -1.0, 2.0, -0.5]), 0.5,
        )

    def test_win_rate_empty(self):
        self.assertEqual(_win_rate([]), 0.0)

    def test_consec_losses(self):
        trades = _make_trades([1.0, 2.0, -1.0, -0.5, -2.0])
        self.assertEqual(_consec_losses(trades), 3)

    def test_consec_losses_none(self):
        trades = _make_trades([1.0, 2.0, 3.0])
        self.assertEqual(_consec_losses(trades), 0)

    def test_consec_losses_broken_by_win(self):
        trades = _make_trades([-1.0, -2.0, 1.0, -0.5])
        self.assertEqual(_consec_losses(trades), 1)

    def test_strategy_drawdown(self):
        # Cumulative: 2, 4, 2, 1, 3 → peak=4, trough=1, dd=3
        dd = _strategy_drawdown([2.0, 2.0, -2.0, -1.0, 2.0])
        self.assertAlmostEqual(dd, 3.0)

    def test_strategy_drawdown_no_loss(self):
        self.assertEqual(_strategy_drawdown([1.0, 2.0, 3.0]), 0.0)

    def test_strategy_drawdown_empty(self):
        self.assertEqual(_strategy_drawdown([]), 0.0)


# ── State machine tests ────────────────────────────────────────


class TestStateTransitions(unittest.TestCase):

    def setUp(self):
        self.th = MonitorThresholds()

    def test_active_stays_active_with_good_metrics(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.ACTIVE,
            rolling_sharpe=1.0, rolling_win_rate=0.6,
            baseline_win_rate=0.6, consec_losses=0,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.ACTIVE,
        )

    def test_active_to_caution_on_low_sharpe(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.ACTIVE,
            rolling_sharpe=0.3, rolling_win_rate=0.6,
            baseline_win_rate=0.6, consec_losses=0,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.CAUTION,
        )

    def test_active_to_caution_on_win_rate_drop(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.ACTIVE,
            rolling_sharpe=0.8, rolling_win_rate=0.40,
            baseline_win_rate=0.60, consec_losses=0,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.CAUTION,
        )

    def test_active_to_caution_on_consec_losses(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.ACTIVE,
            rolling_sharpe=0.8, rolling_win_rate=0.6,
            baseline_win_rate=0.6, consec_losses=3,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.CAUTION,
        )

    def test_caution_to_suspended_on_negative_sharpe(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.CAUTION,
            rolling_sharpe=-0.5, rolling_win_rate=0.4,
            baseline_win_rate=0.6, consec_losses=2,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.SUSPENDED,
        )

    def test_caution_to_suspended_on_consec_losses(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.CAUTION,
            rolling_sharpe=0.3, rolling_win_rate=0.5,
            baseline_win_rate=0.6, consec_losses=5,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.SUSPENDED,
        )

    def test_caution_recovers_to_active(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.CAUTION,
            rolling_sharpe=1.0, rolling_win_rate=0.58,
            baseline_win_rate=0.60, consec_losses=0,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.ACTIVE,
        )

    def test_caution_stays_caution_with_moderate_metrics(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.CAUTION,
            rolling_sharpe=0.4, rolling_win_rate=0.50,
            baseline_win_rate=0.60, consec_losses=1,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.CAUTION,
        )

    def test_suspended_recovers_to_caution(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.SUSPENDED,
            rolling_sharpe=0.5, rolling_win_rate=0.5,
            baseline_win_rate=0.6, consec_losses=1,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.CAUTION,
        )

    def test_suspended_stays_suspended(self):
        h = StrategyHealth(
            strategy="test", state=StrategyState.SUSPENDED,
            rolling_sharpe=-0.2, rolling_win_rate=0.3,
            baseline_win_rate=0.6, consec_losses=4,
        )
        self.assertEqual(
            _evaluate_transition(h, self.th), StrategyState.SUSPENDED,
        )


# ── Persistence tests ──────────────────────────────────────────


class TestPersistence(unittest.TestCase):

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            states = {
                "VWAP Trend": StrategyHealth(
                    strategy="VWAP Trend",
                    state=StrategyState.CAUTION,
                    prev_state=StrategyState.ACTIVE,
                    rolling_sharpe=0.3,
                    baseline_win_rate=0.62,
                    last_transition="2026-04-07T09:31:00",
                ),
            }
            _save_state(states, path)
            loaded = _load_state(path)

            self.assertIn("VWAP Trend", loaded)
            h = loaded["VWAP Trend"]
            self.assertEqual(h.state, StrategyState.CAUTION)
            self.assertEqual(h.prev_state, StrategyState.ACTIVE)
            self.assertAlmostEqual(h.rolling_sharpe, 0.3)
            self.assertAlmostEqual(h.baseline_win_rate, 0.62)

    def test_load_missing_file(self):
        result = _load_state(Path("/nonexistent/state.json"))
        self.assertEqual(result, {})

    def test_load_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("not json{{{")
            result = _load_state(path)
            self.assertEqual(result, {})


# ── Integration tests ───────────────────────────────────────────


class TestStrategyMonitor(unittest.TestCase):

    def test_insufficient_trades_stays_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sm = StrategyMonitor(d)
            trades = _make_trades([1.0, -0.5, 2.0])  # Only 3
            signals = [_make_signal()]
            result = sm.evaluate(trades, signals)

            h = result.strategy_states["VWAP Trend"]
            self.assertEqual(h.state, StrategyState.ACTIVE)
            self.assertIn("Insufficient", h.note)
            self.assertEqual(result.verdicts[0].action, "pass")

    def test_degradation_produces_would_reduce(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sm = StrategyMonitor(d)
            # 5 trades: all losses → low Sharpe, 5 consec losses
            trades = _make_trades(
                [-1.0, -2.0, -0.5, -1.5, -3.0],
            )
            signals = [_make_signal()]
            result = sm.evaluate(trades, signals)

            h = result.strategy_states["VWAP Trend"]
            # With 5 consecutive losses, should be at least CAUTION.
            self.assertIn(
                h.state, (StrategyState.CAUTION, StrategyState.SUSPENDED),
            )
            v = result.verdicts[0]
            self.assertIn(v.action, ("would_reduce", "would_block"))

    def test_suspended_produces_would_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Pre-seed state as CAUTION so next eval can degrade to SUSPENDED.
            state_path = d / "strategy_monitor_state.json"
            d.mkdir(parents=True, exist_ok=True)
            seed = {
                "VWAP Trend": {
                    "strategy": "VWAP Trend",
                    "state": "CAUTION",
                    "prev_state": "ACTIVE",
                    "rolling_sharpe": 0.3,
                    "rolling_win_rate": 0.4,
                    "baseline_win_rate": 0.6,
                    "consec_losses": 2,
                    "mean_r": -0.5,
                    "strategy_drawdown_pct": 5.0,
                    "closed_trades": 10,
                    "last_transition": "2026-04-01",
                    "note": "",
                },
            }
            state_path.write_text(json.dumps(seed))

            sm = StrategyMonitor(d)
            # 6 trades: all losses → triggers CAUTION → SUSPENDED.
            trades = _make_trades(
                [-1.0, -2.0, -1.5, -3.0, -0.5, -2.0],
            )
            signals = [_make_signal()]
            result = sm.evaluate(trades, signals)

            h = result.strategy_states["VWAP Trend"]
            self.assertEqual(h.state, StrategyState.SUSPENDED)
            self.assertEqual(result.verdicts[0].action, "would_block")

    def test_jsonl_log_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sm = StrategyMonitor(d)
            trades = _make_trades([1.0, 2.0, -0.5, 1.5, 3.0])
            signals = [_make_signal()]
            sm.evaluate(trades, signals)

            log_path = d / "strategy_monitor.jsonl"
            self.assertTrue(log_path.exists())
            line = json.loads(log_path.read_text().strip())
            self.assertIn("strategy_states", line)
            self.assertIn("verdicts", line)
            self.assertIn("total_closed_trades", line)

    def test_state_persisted_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Run 1: all losses → degrades from ACTIVE to CAUTION.
            sm1 = StrategyMonitor(d)
            trades = _make_trades(
                [-1.0, -2.0, -0.5, -1.5, -3.0],
            )
            r1 = sm1.evaluate(trades, [_make_signal()])
            state1 = r1.strategy_states["VWAP Trend"].state
            self.assertEqual(state1, StrategyState.CAUTION)

            # Run 2: same bad trades, but now starts from persisted
            # CAUTION → degrades further to SUSPENDED.
            sm2 = StrategyMonitor(d)
            r2 = sm2.evaluate(trades, [_make_signal()])
            state2 = r2.strategy_states["VWAP Trend"].state
            self.assertEqual(state2, StrategyState.SUSPENDED)

            # Confirms state was loaded from disk (not reset to ACTIVE).
            self.assertNotEqual(state2, StrategyState.ACTIVE)

    def test_frozen_baseline_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Run 1: establish baseline from 5 mixed trades.
            sm1 = StrategyMonitor(d)
            trades1 = _make_trades([1.0, -0.5, 2.0, 1.5, -1.0])
            sm1.evaluate(trades1, [])
            baseline1 = (
                _load_state(d / "strategy_monitor_state.json")
                ["VWAP Trend"].baseline_win_rate
            )

            # Run 2: add more losing trades — baseline shouldn't change.
            sm2 = StrategyMonitor(d)
            trades2 = trades1 + _make_trades(
                [-2.0, -3.0], strategy="VWAP Trend",
            )
            sm2.evaluate(trades2, [])
            baseline2 = (
                _load_state(d / "strategy_monitor_state.json")
                ["VWAP Trend"].baseline_win_rate
            )

            self.assertEqual(baseline1, baseline2)

    def test_multiple_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sm = StrategyMonitor(d)
            trades = (
                _make_trades([1.0, 2.0, 3.0, 1.5, 2.0], strategy="VWAP Trend")
                + _make_trades([-1.0, -2.0, -0.5, -1.5, -3.0], strategy="RSI Mean Reversion")
            )
            signals = [
                _make_signal(strategy="VWAP Trend"),
                _make_signal(strategy="RSI Mean Reversion", ticker="SBUX"),
            ]
            result = sm.evaluate(trades, signals)

            self.assertEqual(
                result.strategy_states["VWAP Trend"].state,
                StrategyState.ACTIVE,
            )
            self.assertNotEqual(
                result.strategy_states["RSI Mean Reversion"].state,
                StrategyState.ACTIVE,
            )

    def test_no_signals_no_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sm = StrategyMonitor(d)
            trades = _make_trades([1.0, 2.0, -0.5, 1.5, 3.0])
            result = sm.evaluate(trades, [])
            self.assertEqual(len(result.verdicts), 0)

    def test_signal_for_unknown_strategy_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sm = StrategyMonitor(d)
            trades = _make_trades([1.0, 2.0, 3.0])
            signals = [_make_signal(strategy="Never Seen Before")]
            result = sm.evaluate(trades, signals)
            self.assertEqual(result.verdicts[0].action, "pass")

    def test_insufficient_data_forces_active_over_persisted_state(self):
        """P1: Even if state file says SUSPENDED, insufficient data
        must reset to ACTIVE and produce 'pass' verdicts."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            d.mkdir(parents=True, exist_ok=True)
            # Pre-seed as SUSPENDED.
            state_path = d / "strategy_monitor_state.json"
            seed = {
                "VWAP Trend": {
                    "strategy": "VWAP Trend",
                    "state": "SUSPENDED",
                    "prev_state": "CAUTION",
                    "rolling_sharpe": -0.5,
                    "rolling_win_rate": 0.2,
                    "baseline_win_rate": 0.6,
                    "consec_losses": 6,
                    "mean_r": -1.0,
                    "strategy_drawdown_pct": 10.0,
                    "closed_trades": 20,
                    "last_transition": "2026-04-01",
                    "note": "",
                },
            }
            state_path.write_text(json.dumps(seed))

            sm = StrategyMonitor(d)
            # Only 3 trades → below min_trades_for_eval.
            trades = _make_trades([-1.0, -2.0, -0.5])
            signals = [_make_signal()]
            result = sm.evaluate(trades, signals)

            h = result.strategy_states["VWAP Trend"]
            self.assertEqual(h.state, StrategyState.ACTIVE)
            self.assertIn("Insufficient", h.note)
            self.assertEqual(result.verdicts[0].action, "pass")

    def test_frozen_baseline_zero_preserved(self):
        """P2: A frozen baseline of 0.0 (all losses) must not be
        recomputed on subsequent runs."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Run 1: all losses → baseline = 0.0.
            sm1 = StrategyMonitor(d)
            trades = _make_trades([-1.0, -2.0, -0.5, -1.5, -3.0])
            sm1.evaluate(trades, [])
            state = _load_state(d / "strategy_monitor_state.json")
            self.assertEqual(
                state["VWAP Trend"].baseline_win_rate, 0.0,
            )

            # Run 2: add a winner → baseline must stay 0.0.
            sm2 = StrategyMonitor(d)
            trades2 = trades + _make_trades(
                [5.0, 4.0], strategy="VWAP Trend",
            )
            sm2.evaluate(trades2, [])
            state2 = _load_state(d / "strategy_monitor_state.json")
            self.assertEqual(
                state2["VWAP Trend"].baseline_win_rate, 0.0,
            )

    def test_non_buy_signals_filtered_internally(self):
        """P2: Even if caller passes SELL signals, evaluate()
        must skip them and produce no verdicts for them."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            sm = StrategyMonitor(d)
            trades = _make_trades([-1.0, -2.0, -0.5, -1.5, -3.0])
            sell_signal = Signal(
                ticker="AAPL", signal="EXIT", signal_raw=-1,
                strategy="VWAP Trend", confidence="MEDIUM",
                confidence_score=3, composite_score=25.0,
                current_price=150.0, stop_loss_price=140.0,
                take_profit_price=170.0,
                suggested_position_size_pct=5.0,
                signal_expires="2099-12-31", sharpe=1.5,
                win_rate=60.0, total_trades=20,
            )
            buy_signal = _make_signal(ticker="SBUX")
            result = sm.evaluate(
                trades, [sell_signal, buy_signal],
            )
            # Only the BUY signal should produce a verdict.
            self.assertEqual(len(result.verdicts), 1)
            self.assertEqual(result.verdicts[0].ticker, "SBUX")


if __name__ == "__main__":
    unittest.main()
