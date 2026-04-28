"""Tests for universe loading."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from day_trader.data.universe import DEFAULT_UNIVERSE, load_universe


class TestLoadUniverse(unittest.TestCase):
    def test_default_universe(self):
        u = load_universe()
        # Sanity: at least 50 symbols, includes core ETFs
        self.assertGreater(len(u), 50)
        self.assertIn("SPY", u)
        self.assertIn("QQQ", u)
        self.assertIn("AAPL", u)

    def test_default_is_sorted(self):
        u = load_universe()
        self.assertEqual(u, sorted(u))

    def test_default_no_duplicates(self):
        u = load_universe()
        self.assertEqual(len(u), len(set(u)))

    def test_default_all_uppercase(self):
        u = load_universe()
        for s in u:
            self.assertEqual(s, s.upper())

    def test_extra_symbols_added(self):
        u = load_universe(extra_symbols=["NEW1", "NEW2"])
        self.assertIn("NEW1", u)
        self.assertIn("NEW2", u)

    def test_excluded_symbols_removed(self):
        u = load_universe(excluded_symbols=["SPY", "QQQ"])
        self.assertNotIn("SPY", u)
        self.assertNotIn("QQQ", u)
        # But other defaults still present
        self.assertIn("AAPL", u)

    def test_excluded_overrides_extra(self):
        # If a symbol appears in both extra and excluded, it should
        # NOT appear in the result.
        u = load_universe(
            extra_symbols=["FOO"],
            excluded_symbols=["FOO"],
        )
        self.assertNotIn("FOO", u)

    def test_normalizes_to_uppercase(self):
        u = load_universe(extra_symbols=["aapl", "msft"])
        # AAPL/MSFT already in default; deduplicated
        self.assertEqual(u.count("AAPL"), 1)
        self.assertEqual(u.count("MSFT"), 1)

    def test_csv_override(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "u.csv"
            csv_path.write_text(
                "ticker\n"
                "AAPL\n"
                "msft\n"
                "TSLA\n"
                "\n"
                "# a comment\n"
                "NVDA\n"
            )
            u = load_universe(csv_path=csv_path)
        self.assertEqual(u, ["AAPL", "MSFT", "NVDA", "TSLA"])
        # SPY etc. NOT in result — CSV overrode default
        self.assertNotIn("SPY", u)

    def test_csv_skips_blanks_and_comments(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "u.csv"
            csv_path.write_text(
                "\n"
                "# header comment\n"
                "AAPL\n"
                "  \n"
                "# another comment\n"
                "MSFT\n"
            )
            u = load_universe(csv_path=csv_path)
        self.assertEqual(u, ["AAPL", "MSFT"])

    def test_csv_with_only_header(self):
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "u.csv"
            csv_path.write_text("ticker\n")
            u = load_universe(csv_path=csv_path)
        self.assertEqual(u, [])

    def test_missing_csv_falls_back_to_default(self):
        u = load_universe(csv_path=Path("/nonexistent/path.csv"))
        self.assertGreater(len(u), 50)

    def test_csv_with_extra_columns(self):
        # Multi-column CSV — only first column wins
        with TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "u.csv"
            csv_path.write_text(
                "AAPL,1,2,3\n"
                "MSFT,a,b,c\n"
            )
            u = load_universe(csv_path=csv_path)
        self.assertEqual(u, ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
