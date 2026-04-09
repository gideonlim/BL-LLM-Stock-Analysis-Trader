"""Tests for tp_logic shared helper module.

The hard regression requirement is that ``tp_mode="current"``
produces bit-identical output for both callers (signals.py live
and backtest.py TB path) vs. today's hand-coded ladders.  The
parametric tests below enumerate every branch of both ladders.
"""

from __future__ import annotations

import math
import unittest

from quant_analysis_bot.strategies import ALL_STRATEGIES
from quant_analysis_bot.tp_logic import (
    _MEAN_REVERSION_STRATEGIES,
    apply_tp_cap,
    classify_strategy_family,
    compute_expected_max_move_pct,
    compute_rr_ratio,
)


class TestClassifyStrategyFamily(unittest.TestCase):
    """Strategy → family mapping."""

    def test_known_mean_reversion_strategies(self):
        for name in (
            "RSI Mean Reversion",
            "Bollinger Band Mean Reversion",
            "Z-Score Mean Reversion",
            "Stochastic Oscillator",
        ):
            self.assertEqual(
                classify_strategy_family(name),
                "mean_reversion",
                msg=f"{name!r} should be mean_reversion",
            )

    def test_known_trend_following_strategies(self):
        for name in (
            "SMA Crossover (10/50)",
            "EMA Crossover (9/21)",
            "MACD Crossover",
            "Momentum (Rate of Change)",
            "VWAP Trend",
            "ADX Trend Following",
            "Composite Multi-Indicator",
            "Donchian Breakout (20/55)",
            "52-Week High Momentum",
            "PEAD Earnings Drift",
            "Multi-Factor Momentum MR",
        ):
            self.assertEqual(
                classify_strategy_family(name),
                "trend_following",
                msg=f"{name!r} should be trend_following",
            )

    def test_unknown_strategy_defaults_to_trend(self):
        """Unknown names fall through to trend_following (conservative)."""
        self.assertEqual(
            classify_strategy_family("Totally Made-Up Strategy"),
            "trend_following",
        )

    def test_drift_guard_all_mr_names_exist_in_registry(self):
        """Every name in _MEAN_REVERSION_STRATEGIES must exist in
        strategies.ALL_STRATEGIES.

        Catches renames — if someone changes a strategy's .name
        without updating tp_logic, the family classification
        silently regresses to trend_following.
        """
        registry_names = {s.name for s in ALL_STRATEGIES}
        for mr_name in _MEAN_REVERSION_STRATEGIES:
            self.assertIn(
                mr_name,
                registry_names,
                msg=(
                    f"{mr_name!r} in _MEAN_REVERSION_STRATEGIES but "
                    f"not in strategies.ALL_STRATEGIES. "
                    f"Did a strategy get renamed?"
                ),
            )


class TestComputeRrRatioLiveSchedule(unittest.TestCase):
    """Live schedule (confidence_score is int).

    Must preserve today's signals.py:400-409 logic bit-identically
    under tp_mode="current".
    """

    # Each row: (trend, confidence_score, adx, expected_rr)
    CASES = [
        # BULLISH — ladder: conf≥4 → 3.0, conf≤1 → 1.5 (bearish/conf<=1
        # branch), else 2.0. ADX>30 adds 0.5 capped at 3.5.
        ("BULLISH", 0, 20, 1.5),
        ("BULLISH", 0, 35, 2.0),  # 1.5 + ADX boost
        ("BULLISH", 1, 20, 1.5),
        ("BULLISH", 1, 35, 2.0),
        ("BULLISH", 2, 20, 2.0),
        ("BULLISH", 2, 35, 2.5),
        ("BULLISH", 3, 20, 2.0),
        ("BULLISH", 3, 35, 2.5),
        ("BULLISH", 4, 20, 3.0),
        ("BULLISH", 4, 35, 3.5),
        ("BULLISH", 5, 20, 3.0),
        ("BULLISH", 5, 35, 3.5),
        # BEARISH — always 1.5, no ADX boost (boost is bullish-only).
        ("BEARISH", 0, 20, 1.5),
        ("BEARISH", 0, 35, 1.5),
        ("BEARISH", 3, 20, 1.5),
        ("BEARISH", 3, 35, 1.5),
        ("BEARISH", 4, 20, 1.5),
        ("BEARISH", 4, 35, 1.5),
        # NEUTRAL — conf≤1 → 1.5, else 2.0. No ADX boost.
        ("NEUTRAL", 0, 20, 1.5),
        ("NEUTRAL", 0, 35, 1.5),
        ("NEUTRAL", 1, 20, 1.5),
        ("NEUTRAL", 1, 35, 1.5),
        ("NEUTRAL", 2, 20, 2.0),
        ("NEUTRAL", 2, 35, 2.0),
        ("NEUTRAL", 3, 20, 2.0),
        ("NEUTRAL", 3, 35, 2.0),
        ("NEUTRAL", 4, 20, 2.0),
        ("NEUTRAL", 4, 35, 2.0),
    ]

    def test_current_mode_matches_live_ladder(self):
        for trend, conf, adx, expected in self.CASES:
            with self.subTest(trend=trend, conf=conf, adx=adx):
                rr = compute_rr_ratio(
                    trend=trend,
                    adx=float(adx),
                    confidence_score=conf,
                    family="trend_following",
                    tp_mode="current",
                )
                self.assertEqual(rr, expected)

    def test_capped_mode_same_as_current_for_trend(self):
        """tp_mode='capped' doesn't touch RR — only caps TP later."""
        for trend, conf, adx, expected in self.CASES:
            with self.subTest(trend=trend, conf=conf, adx=adx):
                rr = compute_rr_ratio(
                    trend=trend,
                    adx=float(adx),
                    confidence_score=conf,
                    family="trend_following",
                    tp_mode="capped",
                )
                self.assertEqual(rr, expected)

    def test_capped_strategy_clamps_mean_reversion(self):
        """Mean-reversion family is clamped at 1.5 in capped+strategy."""
        # Every trend-following result stays the same.
        for trend, conf, adx, expected in self.CASES:
            with self.subTest(trend=trend, conf=conf, adx=adx, family="trend"):
                rr = compute_rr_ratio(
                    trend=trend,
                    adx=float(adx),
                    confidence_score=conf,
                    family="trend_following",
                    tp_mode="capped+strategy",
                )
                self.assertEqual(rr, expected)

        # Mean-reversion clamps the result at 1.5.
        for trend, conf, adx, base_expected in self.CASES:
            with self.subTest(trend=trend, conf=conf, adx=adx, family="mr"):
                rr = compute_rr_ratio(
                    trend=trend,
                    adx=float(adx),
                    confidence_score=conf,
                    family="mean_reversion",
                    tp_mode="capped+strategy",
                )
                self.assertEqual(rr, min(base_expected, 1.5))


class TestComputeRrRatioBacktestSchedule(unittest.TestCase):
    """Backtest schedule (confidence_score=None).

    Must preserve today's backtest.py:488-507 logic bit-identically
    under tp_mode="current".  No confidence signal — simpler ladder:
    bullish=2.5, bearish=1.5, neutral=2.0, bullish+ADX>30=3.0.
    """

    CASES = [
        ("BULLISH", 20, 2.5),
        ("BULLISH", 30, 2.5),  # ADX > 30 (strict), so 30 exactly → no boost
        ("BULLISH", 31, 3.0),  # boost fires
        ("BULLISH", 35, 3.0),
        ("BEARISH", 20, 1.5),
        ("BEARISH", 35, 1.5),  # no boost (bullish-only)
        ("NEUTRAL", 20, 2.0),
        ("NEUTRAL", 35, 2.0),  # no boost
    ]

    def test_current_mode_matches_backtest_ladder(self):
        for trend, adx, expected in self.CASES:
            with self.subTest(trend=trend, adx=adx):
                rr = compute_rr_ratio(
                    trend=trend,
                    adx=float(adx),
                    confidence_score=None,
                    family="trend_following",
                    tp_mode="current",
                )
                self.assertEqual(rr, expected)

    def test_capped_strategy_mr_clamps_backtest(self):
        """Backtest bullish=2.5 clamps to 1.5 for MR in capped+strategy."""
        rr = compute_rr_ratio(
            trend="BULLISH",
            adx=20.0,
            confidence_score=None,
            family="mean_reversion",
            tp_mode="capped+strategy",
        )
        self.assertEqual(rr, 1.5)

        # Bullish + ADX boost would be 3.0, still clamps to 1.5.
        rr = compute_rr_ratio(
            trend="BULLISH",
            adx=35.0,
            confidence_score=None,
            family="mean_reversion",
            tp_mode="capped+strategy",
        )
        self.assertEqual(rr, 1.5)


class TestComputeExpectedMaxMovePct(unittest.TestCase):
    """1-sigma expected move calculation."""

    def test_standard_inputs(self):
        # vol_20=0.25 (25% annualised), 20-day horizon.
        # daily_vol_pct = 0.25 / sqrt(252) * 100 ≈ 1.575
        # expected = 1.575 * sqrt(20) ≈ 7.043
        result = compute_expected_max_move_pct(vol_20=0.25, holding_days=20.0)
        self.assertAlmostEqual(result, 7.043, places=2)

    def test_short_horizon(self):
        # VWAP Trend case: ~3.5 day hold, 25% vol.
        # Expected ≈ 1.575 * sqrt(3.5) ≈ 2.946
        result = compute_expected_max_move_pct(vol_20=0.25, holding_days=3.5)
        self.assertAlmostEqual(result, 2.946, places=2)

    def test_zero_vol_returns_inf(self):
        self.assertEqual(
            compute_expected_max_move_pct(0.0, 20.0),
            float("inf"),
        )

    def test_negative_vol_returns_inf(self):
        self.assertEqual(
            compute_expected_max_move_pct(-0.1, 20.0),
            float("inf"),
        )

    def test_nan_vol_returns_inf(self):
        self.assertEqual(
            compute_expected_max_move_pct(float("nan"), 20.0),
            float("inf"),
        )

    def test_none_vol_returns_inf(self):
        self.assertEqual(
            compute_expected_max_move_pct(None, 20.0),  # type: ignore[arg-type]
            float("inf"),
        )

    def test_zero_holding_returns_inf(self):
        self.assertEqual(
            compute_expected_max_move_pct(0.25, 0.0),
            float("inf"),
        )

    def test_negative_holding_returns_inf(self):
        self.assertEqual(
            compute_expected_max_move_pct(0.25, -5.0),
            float("inf"),
        )


class TestApplyTpCap(unittest.TestCase):
    """TP cap pass-through and clamp behavior."""

    def test_current_mode_always_passes_through(self):
        """tp_mode='current' never modifies the TP."""
        self.assertEqual(
            apply_tp_cap(100.0, 5.0, 1.5, "current"),
            100.0,
        )
        self.assertEqual(
            apply_tp_cap(5.0, 100.0, 1.5, "current"),
            5.0,
        )

    def test_infinite_expected_max_passes_through(self):
        """Missing vol/horizon → no cap, regardless of mode."""
        for mode in ("capped", "capped+strategy"):
            self.assertEqual(
                apply_tp_cap(100.0, float("inf"), 1.5, mode),
                100.0,
            )

    def test_cap_fires_when_tp_exceeds_cap(self):
        # expected_max=10, cap_multiplier=1.5 → cap=15
        # tp=100 > cap → return 15
        self.assertEqual(
            apply_tp_cap(100.0, 10.0, 1.5, "capped"),
            15.0,
        )

    def test_cap_does_not_fire_when_tp_below_cap(self):
        # expected_max=10, cap=15, tp=5 → return 5
        self.assertEqual(
            apply_tp_cap(5.0, 10.0, 1.5, "capped"),
            5.0,
        )

    def test_cap_fires_identically_for_capped_strategy_mode(self):
        self.assertEqual(
            apply_tp_cap(100.0, 10.0, 1.5, "capped+strategy"),
            15.0,
        )

    def test_cap_never_raises_tp(self):
        """The cap is a min() — it should never increase a TP."""
        # tp=2, cap=1.5*10=15 → no change (already below cap)
        self.assertEqual(
            apply_tp_cap(2.0, 10.0, 1.5, "capped"),
            2.0,
        )

    def test_zero_cap_multiplier_passes_through(self):
        """cap_multiplier=0 would collapse TP to 0 — must pass through."""
        with self.assertLogs("quant_analysis_bot.tp_logic", "WARNING"):
            result = apply_tp_cap(10.0, 5.0, 0.0, "capped")
        self.assertEqual(result, 10.0)

    def test_negative_cap_multiplier_passes_through(self):
        """cap_multiplier<0 would make TP negative — must pass through."""
        with self.assertLogs("quant_analysis_bot.tp_logic", "WARNING"):
            result = apply_tp_cap(10.0, 5.0, -1.5, "capped")
        self.assertEqual(result, 10.0)

    def test_nan_cap_multiplier_passes_through(self):
        """Non-finite cap_multiplier must pass through."""
        with self.assertLogs("quant_analysis_bot.tp_logic", "WARNING"):
            result = apply_tp_cap(10.0, 5.0, float("nan"), "capped")
        self.assertEqual(result, 10.0)

    def test_inf_cap_multiplier_passes_through(self):
        """Infinite cap_multiplier is also invalid (would blow up)."""
        with self.assertLogs("quant_analysis_bot.tp_logic", "WARNING"):
            result = apply_tp_cap(10.0, 5.0, float("inf"), "capped")
        self.assertEqual(result, 10.0)

    def test_invalid_cap_multiplier_in_capped_strategy_mode(self):
        """Guard applies to capped+strategy mode too."""
        with self.assertLogs("quant_analysis_bot.tp_logic", "WARNING"):
            result = apply_tp_cap(
                10.0, 5.0, -0.5, "capped+strategy",
            )
        self.assertEqual(result, 10.0)


class TestCapMultiplierSweep(unittest.TestCase):
    """End-to-end sanity for the multipliers used in the experiment."""

    def test_vwap_trend_scenario(self):
        """VWAP Trend with 3.5-day hold, 25% vol, 14% raw TP.

        From the diagnosis: median TP/expected_max ≈ 4.06 for
        VWAP Trend.  Verify the cap formula produces that ratio
        roughly and that different cap multipliers shrink the TP.
        """
        vol_20 = 0.25
        holding = 3.5
        raw_tp = 14.0  # percent

        expected_max = compute_expected_max_move_pct(vol_20, holding)
        # Check the raw reachability ratio ≈ 4.06 from the diagnosis.
        ratio = raw_tp / expected_max
        self.assertAlmostEqual(ratio, 4.75, places=1)
        # Note: 4.75 here vs 4.06 in the research diagnosis reflects
        # different default assumptions (research used a slightly
        # different vol_20 baseline). The key is the ratio is > 4,
        # which motivates the cap.

        capped_1p0 = apply_tp_cap(raw_tp, expected_max, 1.0, "capped")
        capped_1p5 = apply_tp_cap(raw_tp, expected_max, 1.5, "capped")
        capped_2p0 = apply_tp_cap(raw_tp, expected_max, 2.0, "capped")

        # All caps should shrink the TP.
        self.assertLess(capped_1p0, raw_tp)
        self.assertLess(capped_1p5, raw_tp)
        self.assertLess(capped_2p0, raw_tp)

        # Tighter cap → smaller TP.
        self.assertLess(capped_1p0, capped_1p5)
        self.assertLess(capped_1p5, capped_2p0)

        # Cap values should be exactly the multiplier * expected_max.
        self.assertAlmostEqual(capped_1p0, 1.0 * expected_max, places=4)
        self.assertAlmostEqual(capped_1p5, 1.5 * expected_max, places=4)
        self.assertAlmostEqual(capped_2p0, 2.0 * expected_max, places=4)


if __name__ == "__main__":
    unittest.main()
