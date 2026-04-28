"""Smoke tests for the daemon assembly.

These don't start a real daemon or connect to Alpaca. They verify
that the module graph imports cleanly and the CLI parse works.
Integration testing happens during paper trading.
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock as _um

# Mock alpaca SDK modules so imports resolve without the real SDK
_alpaca_mock = _um.MagicMock()
for mod in [
    "alpaca", "alpaca.trading", "alpaca.trading.client",
    "alpaca.trading.requests", "alpaca.trading.enums",
    "alpaca.data", "alpaca.data.historical", "alpaca.data.requests",
    "alpaca.data.live", "alpaca.data.timeframe",
    "alpaca.data.models",
]:
    sys.modules.setdefault(mod, _alpaca_mock)


class TestImportSmoke(unittest.TestCase):
    """All day_trader modules import without error."""

    def test_import_executor(self):
        import day_trader.executor
        self.assertTrue(hasattr(day_trader.executor, "DayTraderDaemon"))

    def test_import_main(self):
        import day_trader.__main__
        self.assertTrue(hasattr(day_trader.__main__, "main"))

    def test_import_position_manager(self):
        import day_trader.position_manager
        self.assertTrue(
            hasattr(day_trader.position_manager, "PositionManager"),
        )

    def test_import_scheduler(self):
        import day_trader.scheduler
        self.assertTrue(hasattr(day_trader.scheduler, "Scheduler"))

    def test_import_journal_adapter(self):
        import day_trader.journal_adapter
        self.assertTrue(
            hasattr(day_trader.journal_adapter, "create_daytrade"),
        )

    def test_import_eod_flatten(self):
        import deploy.scripts.eod_flatten
        self.assertTrue(hasattr(deploy.scripts.eod_flatten, "main"))


class TestCLIParse(unittest.TestCase):
    """CLI argument parsing produces the right modes."""

    def test_parse_live(self):
        from day_trader.__main__ import _parse_args
        with _um.patch("sys.argv", ["day_trader", "live"]):
            args = _parse_args()
        self.assertEqual(args.mode, "live")

    def test_parse_dry_run(self):
        from day_trader.__main__ import _parse_args
        with _um.patch("sys.argv", ["day_trader", "dry-run"]):
            args = _parse_args()
        self.assertEqual(args.mode, "dry-run")

    def test_parse_scan_only(self):
        from day_trader.__main__ import _parse_args
        with _um.patch("sys.argv", ["day_trader", "scan-only"]):
            args = _parse_args()
        self.assertEqual(args.mode, "scan-only")


if __name__ == "__main__":
    unittest.main()
