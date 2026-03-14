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
    max_position_pct: float = 12.0

    # Daily loss circuit breaker: stop trading if portfolio
    # drops more than this % from day's starting equity
    daily_loss_limit_pct: float = 3.0

    # Minimum composite score to consider executing a signal
    min_composite_score: float = 20.0

    # Minimum confidence score (0-6) to execute
    min_confidence_score: int = 2

    # Max number of concurrent positions (existing + new)
    max_positions: int = 8

    # Minimum position size as % of equity.  Orders below this
    # threshold are skipped — too small to be worth the bracket
    # order overhead (commissions, monitoring, slippage).
    min_position_pct: float = 2.0

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

    # Max days to hold a position before time-based exit
    max_hold_days: int = 10


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

    # Max % above signal price willing to pay on entry.
    # 0 = market order (fills at any price).
    # 1.5 = limit order at signal_price × 1.015 (won't fill if
    # stock gaps up more than 1.5% from the signal price).
    max_entry_slippage_pct: float = 1.5

    # Dry run mode: log what would be done without submitting
    dry_run: bool = False

    # How many days of execution logs to load for strategy memory
    history_lookback_days: int = 30

    # ── Black-Litterman configuration ──────────────────────────
    # Enable Black-Litterman portfolio optimization
    use_black_litterman: bool = True

    # BL prior uncertainty scalar (0.01-0.1, lower = trust prior)
    bl_tau: float = 0.05

    # Risk aversion coefficient (None = estimate from market)
    bl_risk_aversion: float | None = None

    # Days of return history for BL covariance estimation
    bl_lookback_days: int = 60

    # Regime-sensitive covariance: blend short-term EWMA and
    # long-term Ledoit-Wolf based on volatility regime
    bl_regime_sensitive: bool = True

    # Max portfolio weight per GICS sector (0-1). Prevents
    # concentration in a single sector. 0.40 = max 40% in tech
    bl_max_sector_pct: float = 0.40

    # ── LLM view generation ────────────────────────────────────
    # Enable LLM-enhanced views (requires API key)
    llm_views_enabled: bool = False

    # LLM provider: "anthropic" or "openai"
    llm_provider: str = "anthropic"

    # LLM model to use
    llm_model: str = "claude-haiku-4-5-20251001"

    # Number of repeated samples for uncertainty estimation
    llm_num_samples: int = 10

    # Max tickers to query LLM for (only top-N by confidence)
    llm_max_tickers: int = 10

    # Temperature for LLM sampling (needs > 0 for variance)
    llm_temperature: float = 0.7

    # Weight for LLM views when blending with quant views (0-1)
    llm_weight: float = 0.3

    @classmethod
    def from_env(cls) -> TradingConfig:
        """Build config from environment variables."""
        alpaca = AlpacaConfig.from_env()
        risk = RiskLimits(
            max_portfolio_exposure_pct=float(
                os.getenv("MAX_EXPOSURE_PCT", "80.0")
            ),
            max_position_pct=float(
                os.getenv("MAX_POSITION_PCT", "12.0")
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
            max_positions=int(
                os.getenv("MAX_POSITIONS", "8")
            ),
            min_position_pct=float(
                os.getenv("MIN_POSITION_PCT", "2.0")
            ),
        )
        return cls(
            alpaca=alpaca,
            risk=risk,
            signals_dir=Path(
                os.getenv("SIGNALS_DIR", "signals")
            ),
            max_entry_slippage_pct=float(
                os.getenv("MAX_ENTRY_SLIPPAGE_PCT", "1.5")
            ),
            dry_run=os.getenv("DRY_RUN", "false").lower()
            in ("true", "1", "yes"),
            history_lookback_days=int(
                os.getenv("HISTORY_LOOKBACK_DAYS", "30")
            ),
            use_black_litterman=os.getenv(
                "USE_BLACK_LITTERMAN", "true"
            ).lower() in ("true", "1", "yes"),
            bl_tau=float(os.getenv("BL_TAU", "0.05")),
            bl_lookback_days=int(
                os.getenv("BL_LOOKBACK_DAYS", "60")
            ),
            bl_regime_sensitive=os.getenv(
                "BL_REGIME_SENSITIVE", "true"
            ).lower() in ("true", "1", "yes"),
            bl_max_sector_pct=float(
                os.getenv("BL_MAX_SECTOR_PCT", "0.40")
            ),
            llm_views_enabled=os.getenv(
                "LLM_VIEWS_ENABLED", "false"
            ).lower() in ("true", "1", "yes"),
            llm_provider=os.getenv(
                "LLM_PROVIDER", "anthropic"
            ),
            llm_model=os.getenv(
                "LLM_MODEL", "claude-haiku-4-5-20251001"
            ),
            llm_num_samples=int(
                os.getenv("LLM_NUM_SAMPLES", "10")
            ),
            llm_max_tickers=int(
                os.getenv("LLM_MAX_TICKERS", "10")
            ),
            llm_temperature=float(
                os.getenv("LLM_TEMPERATURE", "0.7")
            ),
            llm_weight=float(
                os.getenv("LLM_WEIGHT", "0.3")
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
            if "max_entry_slippage_pct" in overrides:
                config.max_entry_slippage_pct = float(
                    overrides["max_entry_slippage_pct"]
                )
            if "history_lookback_days" in overrides:
                config.history_lookback_days = int(
                    overrides["history_lookback_days"]
                )

            # Black-Litterman overrides
            if "use_black_litterman" in overrides:
                config.use_black_litterman = bool(
                    overrides["use_black_litterman"]
                )
            if "bl_tau" in overrides:
                config.bl_tau = float(overrides["bl_tau"])
            if "bl_risk_aversion" in overrides:
                config.bl_risk_aversion = float(
                    overrides["bl_risk_aversion"]
                )
            if "bl_lookback_days" in overrides:
                config.bl_lookback_days = int(
                    overrides["bl_lookback_days"]
                )
            if "bl_regime_sensitive" in overrides:
                config.bl_regime_sensitive = bool(
                    overrides["bl_regime_sensitive"]
                )
            if "bl_max_sector_pct" in overrides:
                config.bl_max_sector_pct = float(
                    overrides["bl_max_sector_pct"]
                )

            # LLM view overrides
            if "llm_views_enabled" in overrides:
                config.llm_views_enabled = bool(
                    overrides["llm_views_enabled"]
                )
            if "llm_provider" in overrides:
                config.llm_provider = overrides["llm_provider"]
            if "llm_model" in overrides:
                config.llm_model = overrides["llm_model"]
            if "llm_num_samples" in overrides:
                config.llm_num_samples = int(
                    overrides["llm_num_samples"]
                )
            if "llm_max_tickers" in overrides:
                config.llm_max_tickers = int(
                    overrides["llm_max_tickers"]
                )
            if "llm_temperature" in overrides:
                config.llm_temperature = float(
                    overrides["llm_temperature"]
                )
            if "llm_weight" in overrides:
                config.llm_weight = float(
                    overrides["llm_weight"]
                )

            log.info(f"Loaded trading config from {path}")
        return config
