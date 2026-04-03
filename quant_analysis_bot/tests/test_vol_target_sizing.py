"""Tests for volatility-targeted position sizing.

Covers:
  - compute_vol_target_size() pure function
  - blend_position_sizes() blending logic
  - generate_daily_signal() integration with vol-target sizing
  - Edge cases (zero vol, NaN vol, extreme values)
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from quant_analysis_bot.signals import (
    blend_position_sizes,
    compute_vol_target_size,
    generate_daily_signal,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_minimal_df(
    n: int = 300,
    volatility_20: float = 0.25,
) -> pd.DataFrame:
    """Create minimal enriched DataFrame for signal generation."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": close,
            "Volume": rng.randint(1_000_000, 5_000_000, n),
        },
        index=dates,
    )
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["RSI_14"] = 50 + rng.normal(0, 10, n)
    df["Volatility_20"] = volatility_20
    df["Vol_Ratio"] = 1.0
    df["ZScore_20"] = 0.0
    df["ADX_14"] = 20.0
    df["ATR_14"] = close * 0.02
    return df


def _make_backtest_result():
    from quant_analysis_bot.models import BacktestResult

    result = BacktestResult(
        strategy_name="SMA_Crossover",
        ticker="TEST",
        timeframe="12mo",
        backtest_start="2024-01-01",
        backtest_end="2025-01-01",
        trading_days=252,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        win_rate=0.60,
        total_trades=20,
        profit_factor=2.0,
        avg_holding_days=5.0,
        max_drawdown_pct=-10.0,
        annual_return_pct=15.0,
        annual_excess_pct=5.0,
    )
    result.composite_score = 55.0
    return result


def _make_strategy():
    class StubStrategy:
        name = "StubBuy"

        def generate_signals(self, df):
            s = pd.Series(0, index=df.index)
            s.iloc[-1] = 1  # BUY on last bar
            return s

    return StubStrategy()


def _default_config(**overrides):
    cfg = {
        "risk_profile": "moderate",
        "min_sharpe": 0.5,
        "long_only": True,
    }
    cfg.update(overrides)
    return cfg


# ── compute_vol_target_size Tests ────────────────────────────────────


class TestComputeVolTargetSize(unittest.TestCase):
    """Test the pure vol-target sizing function."""

    def test_typical_stock(self):
        """AAPL-like vol (25%) → 7.5% with default params."""
        size = compute_vol_target_size(0.25)
        # 0.15 / (8 × 0.25) = 0.075 = 7.5%
        self.assertAlmostEqual(size, 7.5)

    def test_high_vol_stock(self):
        """ASTS-like vol (60%) → smaller position."""
        size = compute_vol_target_size(0.60)
        # 0.15 / (8 × 0.60) = 0.03125 = 3.12%
        self.assertAlmostEqual(size, 3.12, places=1)

    def test_low_vol_stock(self):
        """GLD-like vol (15%) → larger position."""
        size = compute_vol_target_size(0.15)
        # 0.15 / (8 × 0.15) = 0.125 = 12.5%
        self.assertAlmostEqual(size, 12.5)

    def test_custom_target_vol(self):
        """Custom target vol should scale output."""
        size = compute_vol_target_size(
            0.30, target_annual_vol=0.20
        )
        # 0.20 / (8 × 0.30) = 0.0833 = 8.33%
        self.assertAlmostEqual(size, 8.33, places=1)

    def test_custom_max_positions(self):
        """Fewer positions → larger per-stock allocation."""
        size = compute_vol_target_size(
            0.30, max_positions=4
        )
        # 0.15 / (4 × 0.30) = 0.125 = 12.5%
        self.assertAlmostEqual(size, 12.5)

    def test_string_params_coerced(self):
        """String config values should be coerced to numeric."""
        size = compute_vol_target_size(
            0.30,
            target_annual_vol="0.15",
            max_positions="8",
        )
        # Same as default: 0.15 / (8 × 0.30) = 6.25%
        self.assertAlmostEqual(size, 6.25)

    def test_invalid_string_params_return_sentinel(self):
        """Non-numeric strings should return -1, not crash."""
        self.assertEqual(
            compute_vol_target_size(
                0.30, target_annual_vol="abc"
            ),
            -1.0,
        )

    def test_zero_vol_returns_sentinel(self):
        """Zero volatility → can't compute, return -1."""
        self.assertEqual(compute_vol_target_size(0.0), -1.0)

    def test_negative_vol_returns_sentinel(self):
        """Negative volatility → can't compute, return -1."""
        self.assertEqual(compute_vol_target_size(-0.1), -1.0)

    def test_nan_vol_returns_sentinel(self):
        """NaN volatility → can't compute, return -1."""
        self.assertEqual(
            compute_vol_target_size(float("nan")), -1.0
        )

    def test_none_vol_returns_sentinel(self):
        """None volatility → can't compute, return -1."""
        self.assertEqual(compute_vol_target_size(None), -1.0)

    def test_max_positions_clamped_to_one(self):
        """max_positions < 1 should be clamped to 1."""
        size = compute_vol_target_size(
            0.30, max_positions=0
        )
        # 0.15 / (1 × 0.30) = 0.5 = 50.0%
        self.assertAlmostEqual(size, 50.0)

    def test_very_high_vol_gives_tiny_size(self):
        """Extremely volatile stock → very small position."""
        size = compute_vol_target_size(2.0)
        # 0.15 / (8 × 2.0) = 0.009375 = 0.94%
        self.assertAlmostEqual(size, 0.94, places=1)


# ── blend_position_sizes Tests ───────────────────────────────────────


class TestBlendPositionSizes(unittest.TestCase):
    """Test the blending function."""

    def test_equal_blend(self):
        """50/50 blend of 8% and 4% → 6%."""
        result = blend_position_sizes(8.0, 4.0, blend=0.5)
        self.assertAlmostEqual(result, 6.0)

    def test_pure_kelly(self):
        """blend=0 → pure Kelly."""
        result = blend_position_sizes(8.0, 4.0, blend=0.0)
        self.assertAlmostEqual(result, 8.0)

    def test_pure_vol_target(self):
        """blend=1 → pure vol-target."""
        result = blend_position_sizes(8.0, 4.0, blend=1.0)
        self.assertAlmostEqual(result, 4.0)

    def test_vol_target_unavailable_falls_back(self):
        """When vol-target is -1 (unavailable), return Kelly."""
        result = blend_position_sizes(8.0, -1.0, blend=0.5)
        self.assertAlmostEqual(result, 8.0)

    def test_blend_clamped_above_one(self):
        """blend > 1 should clamp to 1."""
        result = blend_position_sizes(8.0, 4.0, blend=1.5)
        self.assertAlmostEqual(result, 4.0)

    def test_blend_clamped_below_zero(self):
        """blend < 0 should clamp to 0."""
        result = blend_position_sizes(8.0, 4.0, blend=-0.5)
        self.assertAlmostEqual(result, 8.0)

    def test_weighted_blend(self):
        """30/70 blend (0.3 → 30% vol-target)."""
        result = blend_position_sizes(10.0, 5.0, blend=0.3)
        # (1-0.3)×10 + 0.3×5 = 7 + 1.5 = 8.5
        self.assertAlmostEqual(result, 8.5)

    def test_string_blend_coerced(self):
        """String blend value from JSON config should be coerced."""
        result = blend_position_sizes(8.0, 4.0, blend="0.5")
        self.assertAlmostEqual(result, 6.0)

    def test_invalid_string_blend_uses_default(self):
        """Non-numeric blend string should fall back to 0.5."""
        result = blend_position_sizes(8.0, 4.0, blend="abc")
        self.assertAlmostEqual(result, 6.0)


# ── Signal Generation Integration Tests ──────────────────────────────


class TestSignalGenerationWithVolTarget(unittest.TestCase):
    """Test that vol-target sizing flows through to DailySignal."""

    def setUp(self):
        self.result = _make_backtest_result()
        self.strategy = _make_strategy()

    def test_default_config_blends_sizing(self):
        """With default config (blend=0.5), signal should blend."""
        df = _make_minimal_df(volatility_20=0.30)
        config = _default_config(
            vol_target_annual=0.15,
            vol_target_max_positions=8,
            vol_sizing_blend=0.5,
        )
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        # vol_target: 0.15/(8×0.30) = 6.25%
        self.assertAlmostEqual(sig.vol_target_size_pct, 6.25)

        # Kelly: win_rate=0.60, pf=2.0
        # kelly_f = (0.60×2.0 - 0.40)/2.0 = (1.2-0.4)/2.0 = 0.4
        # half_kelly = 0.2 → 20%, capped at 10%
        # Blend: (0.5×10.0 + 0.5×6.25) = 8.125
        expected_blend = round(0.5 * 10.0 + 0.5 * 6.25, 2)
        self.assertAlmostEqual(
            sig.suggested_position_size_pct,
            expected_blend,
            places=1,
        )

    def test_pure_kelly_when_blend_zero(self):
        """blend=0 → vol target computed but not used in final size."""
        df = _make_minimal_df(volatility_20=0.30)
        config = _default_config(vol_sizing_blend=0.0)
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        # Kelly capped at 10%, blend=0 → pure Kelly
        self.assertAlmostEqual(
            sig.suggested_position_size_pct, 10.0
        )
        # vol_target should still be computed for transparency
        self.assertGreater(sig.vol_target_size_pct, 0)

    def test_pure_vol_target_when_blend_one(self):
        """blend=1 → pure vol-target sizing."""
        df = _make_minimal_df(volatility_20=0.30)
        config = _default_config(vol_sizing_blend=1.0)
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        # 0.15 / (8 × 0.30) = 6.25%
        self.assertAlmostEqual(
            sig.suggested_position_size_pct, 6.25
        )

    def test_zero_vol_falls_back_to_kelly(self):
        """When vol_20 is 0, vol target is unavailable → pure Kelly."""
        df = _make_minimal_df(volatility_20=0.0)
        config = _default_config(vol_sizing_blend=0.5)
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        self.assertEqual(sig.vol_target_size_pct, -1.0)
        # Falls back to pure Kelly (capped at 10%)
        self.assertAlmostEqual(
            sig.suggested_position_size_pct, 10.0
        )

    def test_high_vol_reduces_position(self):
        """High-vol stock should get a smaller final size."""
        df_low = _make_minimal_df(volatility_20=0.15)
        df_high = _make_minimal_df(volatility_20=0.60)
        config = _default_config(vol_sizing_blend=0.5)

        sig_low = generate_daily_signal(
            df_low, "LOW_VOL", self.strategy,
            self.result, config,
        )
        sig_high = generate_daily_signal(
            df_high, "HIGH_VOL", self.strategy,
            self.result, config,
        )
        self.assertGreater(
            sig_low.suggested_position_size_pct,
            sig_high.suggested_position_size_pct,
        )

    def test_vol_target_capped_at_confidence_max(self):
        """Vol-target shouldn't exceed confidence-based ceiling."""
        # Very low vol → huge vol-target before cap
        df = _make_minimal_df(volatility_20=0.05)
        config = _default_config(vol_sizing_blend=1.0)
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        # 0.15 / (8 × 0.05) = 37.5% before cap → 10% after cap
        self.assertAlmostEqual(sig.vol_target_size_pct, 10.0)
        self.assertLessEqual(
            sig.suggested_position_size_pct, 10.0
        )

    def test_hold_signal_zero_size_regardless(self):
        """HOLD signal → 0 size even with vol-target active."""

        class HoldStrategy:
            name = "StubHold"

            def generate_signals(self, df):
                return pd.Series(0, index=df.index)

        df = _make_minimal_df(volatility_20=0.25)
        config = _default_config(vol_sizing_blend=0.5)
        sig = generate_daily_signal(
            df, "TEST", HoldStrategy(),
            self.result, config,
        )
        self.assertEqual(sig.suggested_position_size_pct, 0.0)

    def test_sizing_notes_present(self):
        """Signal notes should show Kelly/VolTgt/Blend breakdown."""
        df = _make_minimal_df(volatility_20=0.30)
        config = _default_config(vol_sizing_blend=0.5)
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        self.assertIn("Size:", sig.notes)
        self.assertIn("Kelly", sig.notes)
        self.assertIn("VolTgt", sig.notes)
        self.assertIn("Blend", sig.notes)

    def test_no_sizing_notes_when_vol_unavailable(self):
        """No sizing breakdown when vol-target can't be computed."""
        df = _make_minimal_df(volatility_20=0.0)
        config = _default_config(vol_sizing_blend=0.5)
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        self.assertNotIn("VolTgt", sig.notes)

    def test_vol_target_in_json_serialization(self):
        """vol_target_size_pct should serialize to JSON cleanly."""
        import json
        from dataclasses import asdict

        df = _make_minimal_df(volatility_20=0.30)
        config = _default_config(vol_sizing_blend=0.5)
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        d = asdict(sig)
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        self.assertAlmostEqual(
            parsed["vol_target_size_pct"], 6.25
        )

    def test_config_defaults_used_when_keys_missing(self):
        """Missing config keys should fall back to sensible defaults."""
        df = _make_minimal_df(volatility_20=0.30)
        config = _default_config()  # No vol-target keys
        sig = generate_daily_signal(
            df, "TEST", self.strategy,
            self.result, config,
        )
        # Should still compute with defaults (0.15 / 8 / 0.30 = 6.25)
        self.assertAlmostEqual(sig.vol_target_size_pct, 6.25)


if __name__ == "__main__":
    unittest.main()
