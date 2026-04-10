"""Tests for shared.annualization helpers."""

import math
import unittest

from shared.annualization import (
    ann_factor,
    ann_return,
    ann_volatility,
    daily_risk_free,
)
from trading_bot_bl.market_config import US, TSE


class TestAnnFactor(unittest.TestCase):
    def test_us_252(self):
        self.assertAlmostEqual(ann_factor(US), math.sqrt(252))

    def test_tse_245(self):
        self.assertAlmostEqual(ann_factor(TSE), math.sqrt(245))


class TestDailyRiskFree(unittest.TestCase):
    def test_us_5pct(self):
        expected = 0.05 / 252
        self.assertAlmostEqual(daily_risk_free(US), expected)

    def test_tse_half_pct(self):
        expected = 0.005 / 245
        self.assertAlmostEqual(daily_risk_free(TSE), expected)


class TestAnnReturn(unittest.TestCase):
    def test_zero_days(self):
        self.assertEqual(ann_return(0.10, 0, US), 0.0)

    def test_negative_days(self):
        self.assertEqual(ann_return(0.10, -5, US), 0.0)

    def test_full_year_us(self):
        # 10% over 252 days = 10% annualized
        result = ann_return(0.10, 252, US)
        self.assertAlmostEqual(result, 0.10, places=6)

    def test_half_year_us(self):
        # 5% over 126 days → annualized > 10%
        result = ann_return(0.05, 126, US)
        # (1.05)^2 - 1 = 0.1025
        self.assertAlmostEqual(result, 0.1025, places=4)


class TestAnnVolatility(unittest.TestCase):
    def test_us(self):
        daily_std = 0.01
        expected = 0.01 * math.sqrt(252)
        self.assertAlmostEqual(
            ann_volatility(daily_std, US), expected
        )

    def test_tse(self):
        daily_std = 0.012
        expected = 0.012 * math.sqrt(245)
        self.assertAlmostEqual(
            ann_volatility(daily_std, TSE), expected
        )


if __name__ == "__main__":
    unittest.main()
