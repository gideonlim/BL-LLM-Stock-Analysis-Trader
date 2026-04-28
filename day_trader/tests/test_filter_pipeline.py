"""Tests for the FilterPipeline + Filter ABC."""

from __future__ import annotations

import unittest

from day_trader.filters.base import Filter, FilterPipeline
from day_trader.models import DayTradeSignal, FilterContext


class _PassFilter(Filter):
    name = "pass_all"
    def passes(self, ctx):
        return True, ""


class _RejectFilter(Filter):
    def __init__(self, name: str, reason: str):
        # Override class-level name on the instance
        self.name = name
        self.reason = reason
    def passes(self, ctx):
        return False, self.reason


class _RaisingFilter(Filter):
    name = "exploder"
    def passes(self, ctx):
        raise RuntimeError("boom")


def _ctx_with_signal(ticker="AAPL", strategy="ORB"):
    sig = DayTradeSignal(
        ticker=ticker, strategy=strategy, side="buy",
        signal_price=100.0, stop_loss_price=95.0,
        take_profit_price=110.0, atr=2.0, rvol=2.5,
    )
    return FilterContext(signal=sig)


class TestFilterPipeline(unittest.TestCase):
    def test_empty_filters_rejected(self):
        with self.assertRaises(ValueError):
            FilterPipeline(filters=[])

    def test_duplicate_names_rejected(self):
        a = _RejectFilter("dup", "x")
        b = _RejectFilter("dup", "y")
        with self.assertRaises(ValueError):
            FilterPipeline([a, b])

    def test_all_pass(self):
        pipe = FilterPipeline([_PassFilter()])
        result = pipe.evaluate(_ctx_with_signal())
        self.assertTrue(result.passed)
        self.assertEqual(result.rejected_by, "")

    def test_short_circuits_on_first_reject(self):
        first_reject = _RejectFilter("first", "alpha")
        second_reject = _RejectFilter("second", "beta")
        pipe = FilterPipeline([first_reject, second_reject])
        result = pipe.evaluate(_ctx_with_signal())
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "first")
        self.assertEqual(result.reason, "alpha")
        # Second filter never ran
        self.assertEqual(pipe.stats.get("rejected_by_first"), 1)
        self.assertNotIn("rejected_by_second", pipe.stats)

    def test_filter_exception_treated_as_reject(self):
        pipe = FilterPipeline([_RaisingFilter(), _PassFilter()])
        result = pipe.evaluate(_ctx_with_signal())
        self.assertFalse(result.passed)
        self.assertEqual(result.rejected_by, "exploder")
        self.assertEqual(result.reason, "filter_error")

    def test_histogram_tracks_per_reason(self):
        rej = _RejectFilter("regime", "vix_too_high")
        pipe = FilterPipeline([rej])
        for _ in range(3):
            pipe.evaluate(_ctx_with_signal())
        self.assertEqual(pipe.stats["rejected_by_regime"], 3)
        self.assertEqual(
            pipe.stats["rejected_by_regime:vix_too_high"], 3,
        )

    def test_passed_counter(self):
        pipe = FilterPipeline([_PassFilter()])
        for _ in range(5):
            pipe.evaluate(_ctx_with_signal())
        self.assertEqual(pipe.stats["passed"], 5)

    def test_total_evaluated_excludes_per_reason_keys(self):
        # Mix of pass and reject — total_evaluated counts passes +
        # rejected_by_X (no per-reason :X duplicates).
        rej = _RejectFilter("regime", "vix")
        pipe = FilterPipeline([rej])
        for _ in range(2):
            pipe.evaluate(_ctx_with_signal())
        # 2 rejections → rejected_by_regime=2 + rejected_by_regime:vix=2
        self.assertEqual(pipe.total_evaluated(), 2)

    def test_reset_stats(self):
        pipe = FilterPipeline([_PassFilter()])
        pipe.evaluate(_ctx_with_signal())
        self.assertEqual(pipe.stats["passed"], 1)
        pipe.reset_stats()
        self.assertEqual(pipe.stats, {})

    def test_empty_reason_coerced_to_unspecified(self):
        rej = _RejectFilter("rejector", "")
        pipe = FilterPipeline([rej])
        result = pipe.evaluate(_ctx_with_signal())
        self.assertEqual(result.reason, "unspecified")


if __name__ == "__main__":
    unittest.main()
