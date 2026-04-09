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
        "3mo": 0.25,
        "6mo": 0.35,
        "12mo": 0.40,
    },
    # Walk-forward validation: score on last N% of each window (OOS)
    "walk_forward_validation_pct": 0.30,
    # Next-bar execution: signals on bar[i] execute at close[i+1]
    "next_bar_execution": True,
    # Regime-dependent mean-reversion gate:
    # VIX threshold above which the regime is classified as "fear"
    "vix_fear_threshold": 25.0,
    # ── Volatility-targeted position sizing ────────────────────────
    # Target annualised portfolio volatility.  Each stock's weight is
    # sized so that its contribution to portfolio vol ≈ σ_target / N.
    # Formula: vol_target_pct = σ_target / (N × σ_i) × 100
    "vol_target_annual": 0.15,          # 15% annualised target
    # Number of positions to size for (denominator N).  Should match
    # the max_positions limit on the trading bot side.
    "vol_target_max_positions": 8,
    # Blend factor: 0.0 = pure Half-Kelly, 1.0 = pure vol-target.
    # 0.5 weights them equally.
    "vol_sizing_blend": 0.5,
    # ── Triple barrier + meta-labeling ─────────────────────────────
    # Feature flags (both default off for backward compat)
    "triple_barrier_enabled": False,
    "meta_label_enabled": False,
    # Meta-label: minimum P(success) to keep a BUY signal
    "meta_label_min_prob": 0.35,
    # Meta-label: minimum training trades before trusting model
    "meta_label_min_training_trades": 50,
    # Triple barrier: max holding = avg_holding_days × this mult
    "tb_max_holding_mult": 1.5,
    # CUSUM filter: threshold = cusum_mult × ATR_14 / price
    "cusum_mult": 0.5,
    # Meta-label: retrain every N days (load cached on other days)
    "meta_label_retrain_days": 7,
    # Directory for meta-label model files (relative to working dir)
    "meta_label_model_dir": "models",
    # ── Take-profit mode (experimental, defaults preserve today's behavior) ─
    # "current"         : existing SL × dynamic RR, no cap
    # "capped"          : RR unchanged, TP capped at tp_cap_multiplier ×
    #                     expected_max_move (1σ move over holding_days)
    # "capped+strategy" : capped + mean-reversion strategy family clamped
    #                     to max RR 1.5
    # See quant_analysis_bot/tp_logic.py and research/tp_experiment.py.
    "tp_mode": "current",
    # Multiplier on the 1σ expected max move for the reachability cap.
    # 1.0 = strict (only ~16% of moves exceed it),
    # 1.5 = moderate (default),
    # 2.0 = loose (only caps clearly unreachable TPs).
    "tp_cap_multiplier": 1.5,
    # Fallback holding window for the cap formula when a strategy
    # has no measured avg_holding_days (e.g. zero trades in a window).
    # The normal path uses result.avg_holding_days per strategy/window.
    "tp_cap_holding_days": 20.0,
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
