"""Default configuration and risk profiles."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_CONFIG: dict = {
    "tickers": [
        "AAPL", "GOOG", "NFLX", "GLD", "TEM", "ASTS", "WTI",
    ],
    "lookback_days": 500,
    "long_only": True,
    "risk_profile": "moderate",
    "min_sharpe": 0.5,
    "position_size": 1.0,
    "transaction_cost_bps": 10,
    "output_dir": "signals",
    "report_dir": "reports",
    "trade_log_dir": "trade_logs",
    "data_cache_dir": "cache",
    "backtest_windows": {
        "3mo": 63,
        "6mo": 126,
        "12mo": 252,
    },
    "window_weights": {
        "3mo": 0.50,
        "6mo": 0.30,
        "12mo": 0.20,
    },
}

RISK_PROFILES: dict = {
    "conservative": {
        "min_win_rate": 0.55,
        "max_trades_per_month": 4,
        "signal_threshold": 0.7,
    },
    "moderate": {
        "min_win_rate": 0.48,
        "max_trades_per_month": 8,
        "signal_threshold": 0.5,
    },
    "aggressive": {
        "min_win_rate": 0.40,
        "max_trades_per_month": 15,
        "signal_threshold": 0.3,
    },
}


def load_config(config_path: Optional[str] = None) -> dict:
    """Load config from JSON file or use defaults."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            user_config = json.load(f)
        config.update(user_config)
        log.info(f"Loaded config from {config_path}")
    return config
