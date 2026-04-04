"""Tests for CPPI drawdown control module."""

import json
import tempfile
import unittest
from pathlib import Path

from trading_bot_bl.config import RiskLimits
from trading_bot_bl.cppi import (
    CppiState,
    load_cppi_state,
    save_cppi_state,
    update_cppi,
)
from trading_bot_bl.models import OrderIntent, PortfolioSnapshot, Signal
from trading_bot_bl.risk import RiskManager


# ── Helpers ──────────────────────────────────────────────────────


def _make_signal(ticker: str = "AAPL", **kwargs) -> Signal:
    defaults = dict(
        ticker=ticker, signal="BUY", signal_raw=1,
        strategy="test_strategy", confidence="MEDIUM",
        confidence_score=3, composite_score=25.0,
        current_price=150.0, stop_loss_price=140.0,
        take_profit_price=170.0, suggested_position_size_pct=5.0,
        signal_expires="2099-12-31", sharpe=1.5, win_rate=60.0,
        total_trades=20,
    )
    defaults.update(kwargs)
    return Signal(**defaults)


def _make_intent(side: str = "buy", ticker: str = "AAPL",
                 notional: float = 5000.0) -> OrderIntent:
    return OrderIntent(
        ticker=ticker, side=side, notional=notional,
        stop_loss_price=140.0, take_profit_price=170.0,
        signal=_make_signal(ticker=ticker),
    )


def _make_portfolio(equity: float = 100_000.0, cash: float = 50_000.0,
                    day_pnl_pct: float = 0.0,
                    n_positions: int = 0) -> PortfolioSnapshot:
    positions = {}
    for i in range(n_positions):
        t = f"POS{i}"
        positions[t] = {
            "market_value": 5000.0, "qty": 10,
            "avg_entry_price": 500.0, "unrealized_pnl": 0.0,
        }
    return PortfolioSnapshot(
        equity=equity, cash=cash,
        market_value=n_positions * 5000.0,
        day_pnl=equity * day_pnl_pct / 100,
        day_pnl_pct=day_pnl_pct,
        positions=positions,
    )


class TestCppiStateInit(unittest.TestCase):
    """Test CppiState.from_portfolio initialization."""

    def test_default_initialization(self) -> None:
        state = CppiState.from_portfolio(equity=100_000)
        self.assertEqual(state.peak_equity, 100_000)
        self.assertEqual(state.floor, 90_000)  # 10% drawdown
        self.assertEqual(state.cushion, 10_000)
        self.assertAlmostEqual(state.cushion_pct, 10.0)
        self.assertEqual(state.exposure_multiplier, 1.0)

    def test_custom_drawdown(self) -> None:
        state = CppiState.from_portfolio(
            equity=100_000, max_drawdown_pct=5.0
        )
        self.assertEqual(state.floor, 95_000)
        self.assertEqual(state.cushion, 5_000)

    def test_zero_equity(self) -> None:
        state = CppiState.from_portfolio(equity=0)
        self.assertEqual(state.floor, 0)
        self.assertEqual(state.cushion_pct, 0)


class TestUpdateCppi(unittest.TestCase):
    """Test the CPPI update logic."""

    def setUp(self) -> None:
        self.state = CppiState.from_portfolio(
            equity=100_000,
            max_drawdown_pct=10.0,
            multiplier=5,
            min_exposure_pct=10.0,
        )

    def test_no_drawdown_full_exposure(self) -> None:
        """At peak equity, exposure should be 100%."""
        result = update_cppi(self.state, current_equity=100_000)
        self.assertEqual(result.exposure_multiplier, 1.0)
        self.assertEqual(result.peak_equity, 100_000)
        self.assertEqual(result.floor, 90_000)

    def test_small_drawdown_reduces_exposure(self) -> None:
        """9% drawdown → only 10% of cushion left → exposure drops."""
        result = update_cppi(self.state, current_equity=91_000)
        # cushion = 91000 - 90000 = 1000
        # max_cushion = 100000 * 0.10 = 10000
        # raw = 5 * 1000 / 10000 = 0.5
        self.assertAlmostEqual(
            result.exposure_multiplier, 0.5, places=3
        )
        self.assertAlmostEqual(result.cushion, 1_000, places=0)

    def test_at_floor_minimum_exposure(self) -> None:
        """At the floor, exposure = min_exposure_pct."""
        result = update_cppi(
            self.state, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        self.assertAlmostEqual(
            result.exposure_multiplier, 0.10
        )
        self.assertEqual(result.cushion, 0)

    def test_below_floor_minimum_exposure(self) -> None:
        """Below the floor, exposure stays at minimum."""
        result = update_cppi(
            self.state, current_equity=85_000,
            spy_trend_regime="BEAR",
        )
        self.assertAlmostEqual(
            result.exposure_multiplier, 0.10
        )
        self.assertLess(result.cushion, 0)

    def test_new_high_ratchets_floor(self) -> None:
        """New equity high ratchets the floor up (TIPP)."""
        result = update_cppi(self.state, current_equity=110_000)
        self.assertEqual(result.peak_equity, 110_000)
        self.assertEqual(result.floor, 99_000)  # 110k * 0.9
        self.assertEqual(result.exposure_multiplier, 1.0)

    def test_floor_ratchet_one_way(self) -> None:
        """Floor should not decrease when equity drops."""
        # First go up
        high = update_cppi(self.state, current_equity=110_000)
        self.assertEqual(high.floor, 99_000)
        # Then drop — floor stays at 99k, not recalculated
        low = update_cppi(high, current_equity=100_000)
        self.assertEqual(low.floor, 99_000)
        self.assertEqual(low.peak_equity, 110_000)

    def test_regime_reset_at_floor(self) -> None:
        """When at floor and SPY goes BULL, floor resets."""
        # Drive to floor
        at_floor = update_cppi(
            self.state, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        self.assertAlmostEqual(at_floor.exposure_multiplier, 0.10)
        self.assertFalse(at_floor.floor_was_reset)

        # SPY recovers → floor resets
        recovered = update_cppi(
            at_floor, current_equity=90_000,
            spy_trend_regime="BULL",
        )
        self.assertTrue(recovered.floor_was_reset)
        self.assertEqual(recovered.peak_equity, 90_000)
        self.assertEqual(recovered.floor, 81_000)  # 90k * 0.9
        # Should have full exposure again relative to new floor
        self.assertEqual(recovered.exposure_multiplier, 1.0)

    def test_regime_reset_caution(self) -> None:
        """CAUTION also triggers floor reset when at floor."""
        at_floor = update_cppi(
            self.state, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        recovered = update_cppi(
            at_floor, current_equity=90_000,
            spy_trend_regime="CAUTION",
        )
        self.assertTrue(recovered.floor_was_reset)

    def test_no_reset_during_bear(self) -> None:
        """BEAR regime does NOT reset the floor."""
        at_floor = update_cppi(
            self.state, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        still_bear = update_cppi(
            at_floor, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        self.assertFalse(still_bear.floor_was_reset)
        self.assertAlmostEqual(
            still_bear.exposure_multiplier, 0.10
        )

    def test_no_reset_when_above_floor(self) -> None:
        """Floor reset only triggers when cushion <= 0."""
        # 5% drawdown, still above floor
        above = update_cppi(
            self.state, current_equity=95_000,
            spy_trend_regime="BULL",
        )
        self.assertFalse(above.floor_was_reset)

    def test_zero_equity_returns_zero_exposure(self) -> None:
        result = update_cppi(self.state, current_equity=0)
        self.assertEqual(result.exposure_multiplier, 0.0)

    def test_multiplier_effect(self) -> None:
        """Higher multiplier → more aggressive scaling."""
        conservative = CppiState.from_portfolio(
            equity=100_000, multiplier=3
        )
        aggressive = CppiState.from_portfolio(
            equity=100_000, multiplier=5
        )
        # Both at 9% drawdown (deep enough for both to be < 1.0)
        # cushion = 1000, max_cushion = 10000
        # m=3: 3*1000/10000 = 0.3
        # m=5: 5*1000/10000 = 0.5
        c_result = update_cppi(conservative, current_equity=91_000)
        a_result = update_cppi(aggressive, current_equity=91_000)
        self.assertLess(
            c_result.exposure_multiplier,
            a_result.exposure_multiplier,
        )

    def test_min_exposure_clamping(self) -> None:
        """Custom min_exposure_pct is respected."""
        state = CppiState.from_portfolio(
            equity=100_000, min_exposure_pct=20.0
        )
        result = update_cppi(
            state, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        self.assertAlmostEqual(
            result.exposure_multiplier, 0.20
        )

    def test_exposure_capped_at_one(self) -> None:
        """Exposure multiplier should never exceed 1.0."""
        # With high multiplier and small drawdown, raw could exceed 1.0
        state = CppiState.from_portfolio(
            equity=100_000, multiplier=20
        )
        result = update_cppi(state, current_equity=99_000)
        self.assertLessEqual(result.exposure_multiplier, 1.0)


class TestCppiPersistence(unittest.TestCase):
    """Test CPPI state save/load."""

    def test_round_trip(self) -> None:
        state = CppiState.from_portfolio(equity=100_000)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cppi.json"
            save_cppi_state(state, path)
            loaded = load_cppi_state(path)
            assert loaded is not None
            self.assertEqual(loaded.peak_equity, state.peak_equity)
            self.assertEqual(loaded.floor, state.floor)
            self.assertEqual(loaded.multiplier, state.multiplier)

    def test_missing_file_returns_none(self) -> None:
        result = load_cppi_state(Path("/nonexistent/cppi.json"))
        self.assertIsNone(result)

    def test_corrupt_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cppi.json"
            path.write_text("NOT JSON {{{")
            result = load_cppi_state(path)
            self.assertIsNone(result)

    def test_floor_was_reset_not_persisted(self) -> None:
        """Transient flag should not be saved."""
        state = CppiState.from_portfolio(equity=100_000)
        state = CppiState(
            **{**state.__dict__, "floor_was_reset": True}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cppi.json"
            save_cppi_state(state, path)
            data = json.loads(path.read_text())
            self.assertNotIn("floor_was_reset", data)


class TestCppiUnknownRegime(unittest.TestCase):
    """Test that UNKNOWN regime does not trigger floor reset."""

    def test_unknown_regime_does_not_reset_floor(self) -> None:
        """When SPY regime is disabled (UNKNOWN), floor should not
        auto-reset — the portfolio must recover organically."""
        state = CppiState.from_portfolio(
            equity=100_000, max_drawdown_pct=10.0,
            multiplier=5, min_exposure_pct=10.0,
        )
        # Drive to floor
        at_floor = update_cppi(
            state, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        self.assertAlmostEqual(at_floor.exposure_multiplier, 0.10)

        # With UNKNOWN regime, floor should NOT reset
        still_locked = update_cppi(
            at_floor, current_equity=90_000,
            spy_trend_regime="UNKNOWN",
        )
        self.assertFalse(still_locked.floor_was_reset)
        self.assertAlmostEqual(still_locked.exposure_multiplier, 0.10)

    def test_organic_recovery_without_regime(self) -> None:
        """Without regime data, equity rising above floor restores
        exposure naturally (via new peak ratchet)."""
        state = CppiState.from_portfolio(
            equity=100_000, max_drawdown_pct=10.0,
            multiplier=5, min_exposure_pct=10.0,
        )
        # Drawdown to floor
        at_floor = update_cppi(
            state, current_equity=90_000,
            spy_trend_regime="BEAR",
        )
        # Equity recovers above floor (no regime help)
        recovered = update_cppi(
            at_floor, current_equity=95_000,
            spy_trend_regime="UNKNOWN",
        )
        self.assertFalse(recovered.floor_was_reset)
        # cushion = 95000 - 90000 = 5000, max = 100000*0.1 = 10000
        # raw = 5*5000/10000 = 2.5, capped at 1.0
        self.assertEqual(recovered.exposure_multiplier, 1.0)


class TestCppiConfigOverlay(unittest.TestCase):
    """Test that loading persisted state overlays current config."""

    def test_config_overlay_on_load(self) -> None:
        """Config changes (multiplier, drawdown, min_exposure) should
        take effect even when state is loaded from disk."""
        # Save state with original config
        state = CppiState.from_portfolio(
            equity=100_000, max_drawdown_pct=10.0,
            multiplier=5, min_exposure_pct=10.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cppi.json"
            save_cppi_state(state, path)

            # Load and overlay different config
            loaded = load_cppi_state(path)
            assert loaded is not None
            self.assertEqual(loaded.multiplier, 5)  # as saved

            # Simulate what executor does: overlay config
            loaded.max_drawdown_pct = 5.0
            loaded.multiplier = 3
            loaded.min_exposure_pct = 20.0

            # Verify the overlaid values are used
            result = update_cppi(loaded, current_equity=100_000)
            self.assertEqual(result.multiplier, 3)
            self.assertEqual(result.max_drawdown_pct, 5.0)
            self.assertEqual(result.min_exposure_pct, 20.0)


class TestCppiCircuitBreakerInteraction(unittest.TestCase):
    """Test that sells always pass and circuit breaker is bypassed
    when CPPI is enabled."""

    def test_sell_allowed_during_circuit_breaker(self) -> None:
        """Sell orders must never be blocked by the circuit breaker.
        They are risk-reducing."""
        rm = RiskManager(
            limits=RiskLimits(daily_loss_limit_pct=3.0),
        )
        sell = _make_intent(side="sell")
        # Day P&L at -5% → breaker would trigger
        portfolio = _make_portfolio(day_pnl_pct=-5.0)
        verdict = rm.evaluate_order(sell, portfolio)
        self.assertTrue(verdict.approved)

    def test_sell_allowed_with_zero_equity(self) -> None:
        """Sell orders must go through even when equity is zero.
        Blocking exits during a crisis worsens the situation."""
        rm = RiskManager(limits=RiskLimits())
        sell = _make_intent(side="sell")
        portfolio = _make_portfolio(equity=0.0, cash=0.0)
        verdict = rm.evaluate_order(sell, portfolio)
        self.assertTrue(verdict.approved)

    def test_sell_allowed_with_negative_equity(self) -> None:
        """Sell orders must go through even with negative equity
        (margin call scenario)."""
        rm = RiskManager(limits=RiskLimits())
        sell = _make_intent(side="sell")
        portfolio = _make_portfolio(equity=-5000.0, cash=0.0)
        verdict = rm.evaluate_order(sell, portfolio)
        self.assertTrue(verdict.approved)

    def test_buy_rejected_with_zero_equity(self) -> None:
        """Buy orders are still rejected when equity is zero."""
        rm = RiskManager(limits=RiskLimits())
        buy = _make_intent(side="buy")
        portfolio = _make_portfolio(equity=0.0, cash=0.0)
        verdict = rm.evaluate_order(buy, portfolio)
        self.assertFalse(verdict.approved)
        self.assertIn("equity", verdict.reason.lower())

    def test_buy_blocked_by_breaker_without_cppi(self) -> None:
        """Without CPPI, the circuit breaker still blocks buys."""
        rm = RiskManager(
            limits=RiskLimits(
                daily_loss_limit_pct=3.0,
                cppi_enabled=False,
            ),
        )
        buy = _make_intent(side="buy")
        portfolio = _make_portfolio(day_pnl_pct=-5.0)
        verdict = rm.evaluate_order(buy, portfolio)
        self.assertFalse(verdict.approved)
        self.assertIn("circuit breaker", verdict.reason.lower())

    def test_buy_not_blocked_by_breaker_with_cppi(self) -> None:
        """With CPPI enabled, the circuit breaker does not block buys.
        CPPI handles drawdown scaling continuously instead."""
        cppi = CppiState.from_portfolio(equity=100_000)
        rm = RiskManager(
            limits=RiskLimits(
                daily_loss_limit_pct=3.0,
                cppi_enabled=True,
            ),
            cppi_state=cppi,
        )
        buy = _make_intent(side="buy")
        # Day P&L at -5% → breaker would normally trigger
        portfolio = _make_portfolio(day_pnl_pct=-5.0)
        verdict = rm.evaluate_order(buy, portfolio)
        # Should NOT be rejected by the circuit breaker
        # (may still be rejected by other checks, but not breaker)
        if not verdict.approved:
            self.assertNotIn(
                "circuit breaker", verdict.reason.lower()
            )


if __name__ == "__main__":
    unittest.main()
