"""Tests for the 9 individual filter modules.

Each filter is small; we test the pass/reject contract end-to-end
with synthetic FilterContexts and minimal mocking.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime
from types import SimpleNamespace
from unittest import mock

from day_trader.config import DayRiskLimits
from day_trader.filters.catalyst_filter import CatalystFilter
from day_trader.filters.cooldown import CooldownTracker
from day_trader.filters.cooldown_filter import CooldownFilter
from day_trader.filters.earnings_filter import EarningsFilter
from day_trader.filters.liquidity_filter import LiquidityFilter
from day_trader.filters.regime_filter import RegimeFilter
from day_trader.filters.rvol_filter import RvolFilter
from day_trader.filters.spread_filter import SpreadFilter
from day_trader.filters.symbol_lock_filter import SymbolLockFilter
from day_trader.filters.whole_share_sizing_filter import WholeShareSizingFilter
from day_trader.models import (
    Bar,
    DayTradeSignal,
    FilterContext,
    MarketState,
    Quote,
)
from day_trader.symbol_locks import SymbolLock


def _signal(
    ticker="AAPL",
    strategy="ORB",
    price=100.0,
    sl=95.0,
    rvol=3.0,
    catalyst="",
):
    return DayTradeSignal(
        ticker=ticker, strategy=strategy, side="buy",
        signal_price=price, stop_loss_price=sl,
        take_profit_price=price + (price - sl) * 2,
        atr=(price - sl), rvol=rvol, catalyst_label=catalyst,
    )


def _quote(ticker="AAPL", bid=99.99, ask=100.01):
    return Quote(
        ticker=ticker, timestamp=datetime.now(),
        bid_price=bid, bid_size=10, ask_price=ask, ask_size=10,
    )


def _bars(n: int, volume_each: float = 100_000) -> list[Bar]:
    return [
        Bar(
            ticker="AAPL", timestamp=datetime.now(),
            open=100, high=101, low=99.5, close=100.0,
            volume=volume_each,
        ) for _ in range(n)
    ]


# ── SymbolLockFilter ─────────────────────────────────────────────


def _portfolio(positions: dict):
    return SimpleNamespace(
        equity=100_000.0, cash=50_000.0, positions=positions,
    )


def _broker(positions=None, open_orders=None):
    b = SimpleNamespace()
    b.get_portfolio = lambda: _portfolio(positions or {})
    b.list_open_orders = lambda: list(open_orders or [])
    return b


def _order(symbol, client_order_id="", order_id="x"):
    return SimpleNamespace(
        symbol=symbol, client_order_id=client_order_id, id=order_id,
    )


class TestSymbolLockFilter(unittest.TestCase):
    def test_allowed_when_unlocked(self):
        f = SymbolLockFilter(SymbolLock(_broker()))
        ok, reason = f.passes(FilterContext(signal=_signal()))
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_rejected_when_swing_holds(self):
        f = SymbolLockFilter(
            SymbolLock(_broker(positions={"AAPL": {"qty": 10}}))
        )
        ok, reason = f.passes(FilterContext(signal=_signal()))
        self.assertFalse(ok)
        self.assertEqual(reason, "swing_position")

    def test_no_signal_rejected(self):
        f = SymbolLockFilter(SymbolLock(_broker()))
        ok, reason = f.passes(FilterContext(signal=None))
        self.assertFalse(ok)


# ── RegimeFilter ─────────────────────────────────────────────────


class TestRegimeFilter(unittest.TestCase):
    def test_passes_in_normal_regime(self):
        ms = MarketState(vix=18.0, spy_trend_regime="BULL")
        ctx = FilterContext(signal=_signal(), market_state=ms)
        f = RegimeFilter(DayRiskLimits())
        ok, _ = f.passes(ctx)
        self.assertTrue(ok)

    def test_rejects_high_vix(self):
        ms = MarketState(vix=40.0)
        ctx = FilterContext(signal=_signal(), market_state=ms)
        f = RegimeFilter(DayRiskLimits())
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "vix_too_high")

    def test_rejects_severe_bear(self):
        ms = MarketState(vix=15.0, spy_trend_regime="SEVERE_BEAR")
        ctx = FilterContext(signal=_signal(), market_state=ms)
        f = RegimeFilter(DayRiskLimits())
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "spy_severe_bear")

    def test_rejects_when_market_state_missing(self):
        ctx = FilterContext(signal=_signal(), market_state=None)
        f = RegimeFilter(DayRiskLimits())
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_market_state")


# ── CooldownFilter ───────────────────────────────────────────────


class TestCooldownFilter(unittest.TestCase):
    def test_passes_when_no_cooldown(self):
        cd = CooldownTracker()
        f = CooldownFilter(cd)
        ok, _ = f.passes(FilterContext(signal=_signal()))
        self.assertTrue(ok)

    def test_rejects_on_active_cooldown(self):
        cd = CooldownTracker(ticker_minutes=60, strategy_minutes=30)
        T = datetime(2026, 4, 28, 10, 0)
        cd.record_close(ticker="AAPL", strategy="ORB", pnl=-50, when=T)
        # Inject a clock that returns a time within the cooldown
        f = CooldownFilter(cd, clock=lambda: T.replace(minute=10))
        ok, reason = f.passes(FilterContext(signal=_signal()))
        self.assertFalse(ok)
        self.assertIn("cooldown", reason)


# ── EarningsFilter ───────────────────────────────────────────────


class TestEarningsFilter(unittest.TestCase):
    def test_catalyst_strategy_bypasses(self):
        f = EarningsFilter()
        ok, _ = f.passes(
            FilterContext(signal=_signal(strategy="catalyst_momentum"))
        )
        self.assertTrue(ok)

    def test_passes_when_no_blackout(self):
        # Mock check_earnings_blackout to return clean info
        with mock.patch(
            "day_trader.filters.earnings_filter.check_earnings_blackout"
        ) as m:
            m.return_value = SimpleNamespace(
                ticker="AAPL", in_blackout=False, blackout_reason="",
                next_earnings_date=None, days_until_earnings=None,
            )
            f = EarningsFilter()
            ok, _ = f.passes(FilterContext(signal=_signal()))
        self.assertTrue(ok)

    def test_rejects_when_in_blackout(self):
        with mock.patch(
            "day_trader.filters.earnings_filter.check_earnings_blackout"
        ) as m:
            m.return_value = SimpleNamespace(
                ticker="AAPL", in_blackout=True,
                blackout_reason="earnings_pre_window",
                next_earnings_date=None, days_until_earnings=2,
            )
            f = EarningsFilter()
            ok, reason = f.passes(FilterContext(signal=_signal()))
        self.assertFalse(ok)
        self.assertEqual(reason, "earnings_pre_window")

    def test_fail_open_on_lookup_error(self):
        with mock.patch(
            "day_trader.filters.earnings_filter.check_earnings_blackout",
            side_effect=RuntimeError("yfinance down"),
        ):
            f = EarningsFilter()
            ok, _ = f.passes(FilterContext(signal=_signal()))
        self.assertTrue(ok)


# ── LiquidityFilter ──────────────────────────────────────────────


class TestLiquidityFilter(unittest.TestCase):
    def test_passes_for_liquid_stock(self):
        with mock.patch(
            "day_trader.filters.liquidity_filter.check_liquidity"
        ) as m:
            m.return_value = SimpleNamespace(
                ticker="AAPL", passes=True, rejection_reason="",
                avg_daily_volume=10_000_000, avg_daily_dollar_volume=1.5e9,
                position_pct_of_adv=0.01,
            )
            f = LiquidityFilter(DayRiskLimits())
            ok, _ = f.passes(FilterContext(signal=_signal()))
        self.assertTrue(ok)

    def test_rejects_thinly_traded(self):
        with mock.patch(
            "day_trader.filters.liquidity_filter.check_liquidity"
        ) as m:
            m.return_value = SimpleNamespace(
                ticker="ILLIQ", passes=False,
                rejection_reason="below_min_adv",
                avg_daily_volume=10_000, avg_daily_dollar_volume=200_000,
                position_pct_of_adv=0.5,
            )
            f = LiquidityFilter(DayRiskLimits())
            ok, reason = f.passes(FilterContext(signal=_signal()))
        self.assertFalse(ok)
        self.assertEqual(reason, "below_min_adv")


# ── SpreadFilter ─────────────────────────────────────────────────


class TestSpreadFilter(unittest.TestCase):
    def test_tight_spread_passes(self):
        # 1 bp spread on a $100 mid stock
        ctx = FilterContext(
            signal=_signal(), quote=_quote(bid=99.99, ask=100.01),
        )
        f = SpreadFilter(DayRiskLimits())
        ok, _ = f.passes(ctx)
        self.assertTrue(ok)

    def test_wide_spread_rejected_above_10(self):
        # 30 bp spread on $100 stock — over the 15 bp threshold
        ctx = FilterContext(
            signal=_signal(), quote=_quote(bid=99.85, ask=100.15),
        )
        f = SpreadFilter(DayRiskLimits())
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "spread_too_wide")

    def test_lower_priced_stocks_have_higher_threshold(self):
        # 25 bp spread on a $5 stock — under the 30 bp threshold
        # for ≤$10 stocks, so should pass
        ctx = FilterContext(
            signal=_signal(price=5.0),
            quote=_quote(bid=4.99375, ask=5.00625),  # ~25 bps
        )
        f = SpreadFilter(DayRiskLimits())
        ok, _ = f.passes(ctx)
        self.assertTrue(ok)

    def test_no_quote_rejected(self):
        ctx = FilterContext(signal=_signal(), quote=None)
        f = SpreadFilter(DayRiskLimits())
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_quote")


# ── RvolFilter ───────────────────────────────────────────────────


class TestRvolFilter(unittest.TestCase):
    def test_passes_with_high_rvol(self):
        f = RvolFilter(DayRiskLimits(min_premkt_rvol=2.0))
        ctx = FilterContext(signal=_signal(rvol=3.0))
        ok, _ = f.passes(ctx)
        self.assertTrue(ok)

    def test_rejects_low_rvol(self):
        f = RvolFilter(DayRiskLimits(min_premkt_rvol=2.0))
        ctx = FilterContext(signal=_signal(rvol=1.5))
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "premkt_rvol_too_low")

    def test_rejects_unknown_rvol(self):
        f = RvolFilter(DayRiskLimits())
        ctx = FilterContext(signal=_signal(rvol=0.0))
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "rvol_unknown")


# ── WholeShareSizingFilter ───────────────────────────────────────


class TestWholeShareSizingFilter(unittest.TestCase):
    def test_passes_normal_sizing(self):
        # $100k equity, 0.25% risk = $250 per trade. ATR stop $5/share
        # at $100 entry → up to 50 shares. Plenty of room.
        f = WholeShareSizingFilter(
            DayRiskLimits(per_trade_risk_pct=0.25),
            equity_at_session_start=100_000.0,
        )
        ctx = FilterContext(signal=_signal(price=100, sl=95))  # $5 risk/sh
        ok, _ = f.passes(ctx)
        self.assertTrue(ok)

    def test_rejects_when_stop_too_wide_for_budget(self):
        # $1k equity, 0.25% = $2.50 per trade. ATR stop $5 → 0 shares fit.
        f = WholeShareSizingFilter(
            DayRiskLimits(per_trade_risk_pct=0.25),
            equity_at_session_start=1_000.0,
        )
        ctx = FilterContext(signal=_signal(price=100, sl=95))
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "stop_too_wide_for_risk_budget")

    def test_rejects_zero_stop_distance(self):
        f = WholeShareSizingFilter(
            DayRiskLimits(),
            equity_at_session_start=100_000.0,
        )
        ctx = FilterContext(signal=_signal(price=100, sl=100))
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "no_stop_distance")


# ── CatalystFilter ───────────────────────────────────────────────


class TestCatalystFilter(unittest.TestCase):
    def test_passes_when_no_requirement(self):
        # ORB has no catalyst requirement (None)
        f = CatalystFilter()
        ctx = FilterContext(signal=_signal(strategy="orb_vwap"))
        ok, _ = f.passes(ctx)
        self.assertTrue(ok)

    def test_passes_when_catalyst_matches(self):
        f = CatalystFilter()
        ctx = FilterContext(signal=_signal(
            strategy="vwap_reversion", catalyst="NO_NEWS",
        ))
        ok, _ = f.passes(ctx)
        self.assertTrue(ok)

    def test_rejects_when_catalyst_wrong(self):
        f = CatalystFilter()
        ctx = FilterContext(signal=_signal(
            strategy="vwap_reversion", catalyst="NEWS_HIGH",
        ))
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertIn("want_NO_NEWS_got_NEWS_HIGH", reason)

    def test_rejects_when_catalyst_missing_for_strict_strategy(self):
        f = CatalystFilter()
        ctx = FilterContext(signal=_signal(
            strategy="catalyst_momentum", catalyst="",
        ))
        ok, reason = f.passes(ctx)
        self.assertFalse(ok)
        self.assertEqual(reason, "catalyst_label_unknown")


if __name__ == "__main__":
    unittest.main()
