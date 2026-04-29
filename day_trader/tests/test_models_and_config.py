"""Quick coverage on the data models and config dataclasses.

These are mostly typed containers — tests focus on derived
properties and env-driven config loading."""

from __future__ import annotations

import os
import unittest
from datetime import datetime
from unittest import mock

from day_trader.config import DayRiskLimits, DayTradeConfig
from day_trader.models import (
    Bar,
    DayTradeSignal,
    MarketState,
    Quote,
    Trade,
)


class TestQuoteDerived(unittest.TestCase):
    def test_mid_average(self):
        q = Quote("AAPL", datetime.now(), bid_price=100.0,
                  bid_size=10, ask_price=100.10, ask_size=10)
        self.assertAlmostEqual(q.mid, 100.05)

    def test_spread_bps(self):
        q = Quote("AAPL", datetime.now(), bid_price=100.0,
                  bid_size=10, ask_price=100.10, ask_size=10)
        # 10 cent spread on $100.05 mid = ~9.99 bps
        self.assertAlmostEqual(q.spread_bps, 9.995, places=2)

    def test_spread_bps_zero_mid(self):
        q = Quote("AAPL", datetime.now(), bid_price=0.0,
                  bid_size=0, ask_price=0.0, ask_size=0)
        self.assertEqual(q.spread_bps, float("inf"))


class TestMarketStateDerived(unittest.TestCase):
    def test_severe_bear(self):
        s = MarketState(spy_trend_regime="SEVERE_BEAR")
        self.assertTrue(s.is_severe_bear)
        s.spy_trend_regime = "BULL"
        self.assertFalse(s.is_severe_bear)

    def test_high_vol(self):
        s = MarketState(vix=30.0)
        self.assertTrue(s.is_high_vol)
        s.vix = 15.0
        self.assertFalse(s.is_high_vol)


class TestSignalDefaults(unittest.TestCase):
    def test_generated_at_auto_set(self):
        sig = DayTradeSignal(
            ticker="AAPL", strategy="ORB", side="buy",
            signal_price=100.0, stop_loss_price=95.0,
            take_profit_price=110.0, atr=2.0, rvol=2.5,
        )
        # Should be a non-empty ISO-format string set automatically
        self.assertTrue(sig.generated_at)
        # Round-trip parse-able
        datetime.fromisoformat(sig.generated_at)


class TestBar(unittest.TestCase):
    def test_basic_construction(self):
        b = Bar(
            ticker="AAPL", timestamp=datetime.now(),
            open=100, high=101, low=99.5, close=100.5,
            volume=100_000, vwap=100.2, trade_count=400,
        )
        self.assertEqual(b.ticker, "AAPL")


class TestDayRiskLimitsDefaults(unittest.TestCase):
    def test_locked_to_plan_values(self):
        limits = DayRiskLimits()
        # Plan-locked values
        self.assertEqual(limits.budget_pct, 0.25)
        self.assertEqual(limits.max_positions, 3)
        self.assertEqual(limits.daily_loss_limit_pct, 1.5)
        self.assertEqual(limits.per_trade_risk_pct, 0.25)
        self.assertEqual(limits.max_trades_per_day, 8)
        # Cooldowns (no-revenge, per the user's filtering-first philosophy)
        self.assertEqual(limits.ticker_cooldown_minutes, 60)
        self.assertEqual(limits.strategy_cooldown_minutes, 30)


class TestDayTradeConfigFromEnv(unittest.TestCase):
    def test_defaults_match_dataclass(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in [
                "DAYTRADE_BUDGET_PCT",
                "DAYTRADE_MAX_POSITIONS",
                "DAYTRADE_DAILY_LOSS_LIMIT_PCT",
            ]:
                os.environ.pop(k, None)
            cfg = DayTradeConfig.from_env()
        self.assertEqual(cfg.risk.budget_pct, 0.25)
        self.assertEqual(cfg.risk.max_positions, 3)

    def test_env_overrides(self):
        with mock.patch.dict(os.environ, {
            "DAYTRADE_BUDGET_PCT": "20.0",
            "DAYTRADE_MAX_POSITIONS": "5",
            "DAYTRADE_DAILY_LOSS_LIMIT_PCT": "2.0",
            "DAYTRADE_PER_TRADE_RISK_PCT": "0.5",
            "DAYTRADE_DRY_RUN": "true",
        }, clear=False):
            cfg = DayTradeConfig.from_env()
        self.assertEqual(cfg.risk.budget_pct, 0.20)
        self.assertEqual(cfg.risk.max_positions, 5)
        self.assertEqual(cfg.risk.daily_loss_limit_pct, 2.0)
        self.assertEqual(cfg.risk.per_trade_risk_pct, 0.5)
        self.assertTrue(cfg.dry_run)

    def test_data_feed_defaults_to_sip(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DAYTRADE_DATA_FEED", None)
            cfg = DayTradeConfig.from_env()
        self.assertEqual(cfg.data_feed, "sip")

    def test_data_feed_override(self):
        with mock.patch.dict(os.environ, {
            "DAYTRADE_DATA_FEED": "iex",
        }, clear=False):
            cfg = DayTradeConfig.from_env()
        self.assertEqual(cfg.data_feed, "iex")

    def test_data_feed_normalized_to_lowercase(self):
        with mock.patch.dict(os.environ, {
            "DAYTRADE_DATA_FEED": "SIP",
        }, clear=False):
            cfg = DayTradeConfig.from_env()
        self.assertEqual(cfg.data_feed, "sip")

    def test_repr_does_not_leak_secrets(self):
        with mock.patch.dict(os.environ, {
            "TELEGRAM_BOT_TOKEN": "real-secret-token-xyz",
            "TELEGRAM_CHAT_ID": "12345",
        }, clear=False):
            cfg = DayTradeConfig.from_env()
        s = repr(cfg)
        self.assertNotIn("real-secret-token-xyz", s)
        self.assertNotIn("12345", s)
        # Should indicate that telegram is configured though
        self.assertIn("telegram=set", s)


if __name__ == "__main__":
    unittest.main()
