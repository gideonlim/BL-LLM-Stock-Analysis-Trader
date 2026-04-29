"""Tests for fetch_market_state.

Mocks yfinance — we don't hit the network in unit tests."""

from __future__ import annotations

import unittest
import unittest.mock as _um

import pandas as pd

from day_trader.data.market_state import fetch_market_state


def _make_spy_history(closes: list[float]):
    """Build a fake yfinance DataFrame for SPY."""
    return pd.DataFrame({"Close": closes})


def _make_vix_history(close: float):
    return pd.DataFrame({"Close": [close]})


class TestFetchMarketState(unittest.TestCase):
    def test_bull_regime(self):
        # SPY trending up, latest > 200-SMA
        closes = list(range(400, 600))  # 200 sessions, rising
        with _um.patch("yfinance.Ticker") as mock_ticker:
            spy_mock = _um.MagicMock()
            spy_mock.history.return_value = _make_spy_history(closes)
            vix_mock = _um.MagicMock()
            vix_mock.history.return_value = _make_vix_history(15.0)
            mock_ticker.side_effect = lambda symbol: (
                spy_mock if symbol == "SPY" else vix_mock
            )
            state = fetch_market_state()
        self.assertEqual(state.spy_trend_regime, "BULL")
        self.assertGreater(state.spy_price, state.spy_200_sma)
        self.assertEqual(state.vix, 15.0)

    def test_caution_regime(self):
        # SPY just dipped below 200-SMA on the latest bar only
        # (single-day dip — not confirmed bear, not severe drawdown)
        closes = list(range(500, 700))
        # Latest close drops just below SMA but only by ~5%
        closes[-1] = 595
        with _um.patch("yfinance.Ticker") as mock_ticker:
            spy_mock = _um.MagicMock()
            spy_mock.history.return_value = _make_spy_history(closes)
            vix_mock = _um.MagicMock()
            vix_mock.history.return_value = _make_vix_history(20.0)
            mock_ticker.side_effect = lambda symbol: (
                spy_mock if symbol == "SPY" else vix_mock
            )
            state = fetch_market_state(bear_confirmation_days=3)
        # CAUTION requires < 200-SMA but not confirmed bear
        self.assertEqual(state.spy_trend_regime, "CAUTION")

    def test_bear_regime_confirmed(self):
        # SPY closed below 200-SMA for 3 consecutive days
        closes = [600] * 200  # SMA stays high
        # Last 3 closes: well below SMA
        closes[-3] = 500
        closes[-2] = 490
        closes[-1] = 480
        with _um.patch("yfinance.Ticker") as mock_ticker:
            spy_mock = _um.MagicMock()
            spy_mock.history.return_value = _make_spy_history(closes)
            vix_mock = _um.MagicMock()
            vix_mock.history.return_value = _make_vix_history(28.0)
            mock_ticker.side_effect = lambda symbol: (
                spy_mock if symbol == "SPY" else vix_mock
            )
            state = fetch_market_state(
                bear_confirmation_days=3,
                severe_drawdown_pct=50.0,  # high so we don't hit SEVERE
            )
        self.assertEqual(state.spy_trend_regime, "BEAR")

    def test_severe_bear_on_deep_drawdown(self):
        # SPY -20% from 52-wk high
        closes = [600] * 199 + [480]  # 20% drop
        with _um.patch("yfinance.Ticker") as mock_ticker:
            spy_mock = _um.MagicMock()
            spy_mock.history.return_value = _make_spy_history(closes)
            vix_mock = _um.MagicMock()
            vix_mock.history.return_value = _make_vix_history(40.0)
            mock_ticker.side_effect = lambda symbol: (
                spy_mock if symbol == "SPY" else vix_mock
            )
            state = fetch_market_state(severe_drawdown_pct=15.0)
        self.assertEqual(state.spy_trend_regime, "SEVERE_BEAR")
        self.assertTrue(state.is_severe_bear)

    def test_yfinance_failure_falls_back(self):
        with _um.patch(
            "yfinance.Ticker",
            side_effect=RuntimeError("yfinance down"),
        ):
            state = fetch_market_state()
        # Defaults to BULL (conservative — keeps the daemon running)
        self.assertEqual(state.spy_trend_regime, "BULL")
        self.assertEqual(state.vix, 0.0)

    def test_vix_fetch_failure_returns_zero(self):
        # SPY succeeds, VIX fails
        closes = list(range(400, 600))
        call_count = [0]

        def ticker_side_effect(symbol):
            call_count[0] += 1
            if symbol == "^VIX":
                raise RuntimeError("vix unavailable")
            mock = _um.MagicMock()
            mock.history.return_value = _make_spy_history(closes)
            return mock

        with _um.patch("yfinance.Ticker", side_effect=ticker_side_effect):
            state = fetch_market_state()
        self.assertEqual(state.vix, 0.0)
        self.assertEqual(state.spy_trend_regime, "BULL")


if __name__ == "__main__":
    unittest.main()
