"""Trading bot configuration -- credentials from env, risk limits."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class AlpacaConfig:
    """
    Alpaca API credentials and endpoint selection.

    Credentials are NEVER stored as instance attributes.
    They are read from environment variables on demand and
    cannot leak through serialization, logging, or repr.
    """

    _ENV_KEY = "ALPACA_API_KEY"
    _ENV_SECRET = "ALPACA_API_SECRET"

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper

    @property
    def api_key(self) -> str:
        """Read API key from env every time — never cached."""
        return os.environ.get(self._ENV_KEY, "")

    @property
    def api_secret(self) -> str:
        """Read API secret from env every time — never cached."""
        return os.environ.get(self._ENV_SECRET, "")

    @property
    def base_url(self) -> str:
        if self.paper:
            return "https://paper-api.alpaca.markets"
        return "https://api.alpaca.markets"

    @classmethod
    def from_env(cls) -> AlpacaConfig:
        """Build config from environment variables."""
        paper = os.getenv(
            "ALPACA_PAPER", "true"
        ).lower() in ("true", "1", "yes")
        return cls(paper=paper)

    def validate(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError(
                f"{self._ENV_KEY} and {self._ENV_SECRET} "
                f"environment variables must be set. "
                f"Check your .env file or GitHub Actions secrets."
            )

    def __repr__(self) -> str:
        mode = "paper" if self.paper else "live"
        has_key = "set" if self.api_key else "MISSING"
        has_secret = "set" if self.api_secret else "MISSING"
        return (
            f"AlpacaConfig(mode={mode}, "
            f"key={has_key}, secret={has_secret})"
        )


@dataclass
class RiskLimits:
    """Portfolio-level risk constraints."""

    # Max % of account equity that can be deployed in positions
    max_portfolio_exposure_pct: float = 80.0

    # Max % of account equity in any single stock
    max_position_pct: float = 15.0

    # Daily loss circuit breaker: stop trading if portfolio
    # drops more than this % from day's starting equity
    daily_loss_limit_pct: float = 3.0

    # Minimum composite score to consider executing a signal
    min_composite_score: float = 20.0

    # Minimum confidence score (0-6) to execute
    min_confidence_score: int = 2

    # Only execute signals that are BUY or SELL/SHORT (skip HOLD)
    skip_hold_signals: bool = True

    # Only execute signals that haven't expired
    check_signal_expiry: bool = True

    # ── Position monitoring thresholds ──────────────────────────
    # Auto-close positions that lost more than this % from entry
    emergency_loss_pct: float = 10.0

    # Replace stale brackets if price moved > this % from entry
    # without triggering SL or TP
    stale_bracket_pct: float = 5.0

    # Auto-close orphaned positions (no SL/TP legs) that are
    # losing money. If False, just log a warning.
    auto_close_orphaned_losers: bool = True

    # Max % loss on an orphaned position before force-closing
    orphan_max_loss_pct: float = 5.0


@dataclass
class TradingConfig:
    """Full trading bot configuration."""

    alpaca: AlpacaConfig = field(  # type: ignore[type-var]
        default_factory=AlpacaConfig.from_env
    )
    risk: RiskLimits = field(default_factory=RiskLimits)

    # Path to the signals directory produced by quant_analysis_bot
    signals_dir: Path = Path("signals")

    # Order defaults
    time_in_force: str = "day"  # "day" or "gtc"
    order_type: str = "market"  # "market" or "limit"

    # Dry run mode: log what would be done without submitting
    dry_run: bool = False

    # How many days of execution logs to load for strategy memory
    history_lookback_days: int = 30

    @classmethod
    def from_env(cls) -> TradingConfig:
        """Build config from environment variables."""
        alpaca = AlpacaConfig.from_env()
        risk = RiskLimits(
            max_portfolio_exposure_pct=float(
                os.getenv("MAX_EXPOSURE_PCT", "80.0")
            ),
            max_position_pct=float(
                os.getenv("MAX_POSITION_PCT", "15.0")
            ),
            daily_loss_limit_pct=float(
                os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0")
            ),
            min_composite_score=float(
                os.getenv("MIN_COMPOSITE_SCORE", "20.0")
            ),
            min_confidence_score=int(
                os.getenv("MIN_CONFIDENCE_SCORE", "2")
            ),
        )
        return cls(
            alpaca=alpaca,
            risk=risk,
            signals_dir=Path(
                os.getenv("SIGNALS_DIR", "signals")
            ),
            dry_run=os.getenv("DRY_RUN", "false").lower()
            in ("true", "1", "yes"),
            history_lookback_days=int(
                os.getenv("HISTORY_LOOKBACK_DAYS", "30")
            ),
        )

    @classmethod
    def from_file(cls, path: Path) -> TradingConfig:
        """
        Load config from a JSON file, with env fallback.

        IMPORTANT: Credentials are NEVER read from the config
        file. They must come from environment variables only.
        Any 'api_key'/'api_secret' keys in the JSON are ignored.
        """
        config = cls.from_env()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                overrides = json.load(f)

            # Reject any attempt to put credentials in a file
            _forbidden = {"api_key", "api_secret", "alpaca_api_key",
                          "alpaca_api_secret"}
            found = _forbidden & set(overrides.keys())
            if found:
                log.warning(
                    f"Ignoring credential keys in config file: "
                    f"{found}. Use environment variables instead."
                )

            if "risk" in overrides:
                for k, v in overrides["risk"].items():
                    if hasattr(config.risk, k):
                        setattr(config.risk, k, v)
            if "signals_dir" in overrides:
                config.signals_dir = Path(
                    overrides["signals_dir"]
                )
            if "dry_run" in overrides:
                config.dry_run = overrides["dry_run"]
            if "time_in_force" in overrides:
                config.time_in_force = overrides["time_in_force"]
            if "order_type" in overrides:
                config.order_type = overrides["order_type"]
            if "paper" in overrides:
                config.alpaca.paper = overrides["paper"]
            if "history_lookback_days" in overrides:
                config.history_lookback_days = int(
                    overrides["history_lookback_days"]
                )
            log.info(f"Loaded trading config from {path}")
        return config
