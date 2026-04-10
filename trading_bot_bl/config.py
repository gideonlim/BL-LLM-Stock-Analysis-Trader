"""Trading bot configuration -- credentials from env, risk limits."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trading_bot_bl.broker_base import BrokerInterface

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
class IBKRConfig:
    """Interactive Brokers connection configuration.

    Credentials are managed by IB Gateway / TWS — the bot only
    needs the host, port, and client ID to connect.
    """

    host: str = "127.0.0.1"
    port: int = 7497          # 7497 = paper, 7496 = live
    client_id: int = 1
    account_id: str = ""      # optional — auto-detected if blank
    max_equity_allocation: float = 1.0  # fraction of total equity

    @classmethod
    def from_env(cls) -> IBKRConfig:
        return cls(
            host=os.getenv("IBKR_HOST", "127.0.0.1"),
            port=int(os.getenv("IBKR_PORT", "7497")),
            client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
            account_id=os.getenv("IBKR_ACCOUNT_ID", ""),
            max_equity_allocation=float(
                os.getenv("IBKR_MAX_EQUITY_ALLOCATION", "1.0")
            ),
        )

    def __repr__(self) -> str:
        mode = "paper" if self.port == 7497 else "live"
        return (
            f"IBKRConfig(mode={mode}, "
            f"host={self.host}:{self.port}, "
            f"client_id={self.client_id})"
        )


@dataclass
class RiskLimits:
    """Portfolio-level risk constraints."""

    # Max % of account equity that can be deployed in positions
    max_portfolio_exposure_pct: float = 80.0

    # Max % of account equity in any single stock
    max_position_pct: float = 10.0

    # Daily loss circuit breaker: stop trading if portfolio
    # drops more than this % from day's starting equity
    daily_loss_limit_pct: float = 3.0

    # Minimum composite score to consider executing a signal
    min_composite_score: float = 15.0

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
    emergency_loss_pct: float = 8.0

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

    # Minimum backtest trades for a signal to be considered
    # reliable enough to execute
    min_backtest_trades: int = 3

    # Maximum PBO (probability of backtest overfitting) allowed.
    # Signals with PBO above this are rejected as likely overfit.
    # Set to 1.0 to disable PBO gating.
    # PBO of -1 (not computed) always passes.
    max_pbo: float = 0.50

    # ── Churn / cooldown ──────────────────────────────────────
    # Minimum days between trades on the same ticker.  Applies to
    # both re-buying after a recent buy *and* re-buying after a
    # recent sell (post-exit cooldown).
    ticker_cooldown_days: int = 2

    # ── Earnings blackout ─────────────────────────────────────
    # Block new entries near earnings announcements to avoid
    # overnight gap risk that ATR-based stops can't protect against.
    earnings_blackout_enabled: bool = True
    earnings_blackout_pre_days: int = 3   # days before earnings
    earnings_blackout_post_days: int = 1  # days after earnings

    # ── ADV liquidity filter ────────────────────────────────
    # Block entries in illiquid stocks where slippage and exit
    # risk are high.  Uses yfinance average daily volume.
    adv_liquidity_enabled: bool = True
    min_adv_shares: int = 500_000       # min avg daily shares
    min_adv_dollar_volume: float = 5_000_000.0  # min avg daily $
    max_adv_participation_pct: float = 1.0  # max position as % of ADV$

    # ── CPPI drawdown control ──────────────────────────────
    # Continuous exposure scaling that replaces the binary circuit
    # breaker.  As drawdown deepens, position sizes shrink smoothly.
    cppi_enabled: bool = False
    cppi_max_drawdown_pct: float = 10.0  # floor = peak * (1 - this%)
    cppi_multiplier: int = 5             # risk multiplier (3-5 typical)
    cppi_min_exposure_pct: float = 10.0  # minimum exposure % at floor


@dataclass
class TradingConfig:
    """Full trading bot configuration."""

    alpaca: AlpacaConfig = field(  # type: ignore[type-var]
        default_factory=AlpacaConfig.from_env
    )
    ibkr: IBKRConfig = field(default_factory=IBKRConfig.from_env)
    risk: RiskLimits = field(default_factory=RiskLimits)

    # Market configuration (defaults to US)
    market: "MarketConfig | None" = None  # set in from_file or lazily

    # Path to the signals directory produced by quant_analysis_bot
    signals_dir: Path = Path("signals")

    # Order defaults
    time_in_force: str = "day"  # "day" or "gtc"
    order_type: str = "market"  # "market" or "limit"

    # Max % above signal price willing to pay on entry.
    # 0 = market order (fills at any price).
    # 1.5 = limit order at signal_price × 1.015 (won't fill if
    # stock gaps up more than 1.5% from the signal price).
    max_entry_slippage_pct: float = 1.0

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

    # ── Market sentiment ───────────────────────────────────────
    # Enable market-wide sentiment indicators (VIX + put/call)
    # to modify position sizing.  Set to False to disable.
    market_sentiment_enabled: bool = True

    # VIX thresholds for fear / greed regime classification
    sentiment_fear_vix: float = 25.0
    sentiment_greed_vix: float = 15.0

    # Put/call ratio thresholds
    sentiment_fear_pc: float = 0.95
    sentiment_greed_pc: float = 0.65

    # Position-size multiplier during extreme fear (contrarian)
    sentiment_fear_size_mult: float = 1.10

    # Position-size multiplier during extreme greed (defensive)
    sentiment_greed_size_mult: float = 0.90

    # ── FinBERT news sentiment ─────────────────────────────────
    # Score headlines with FinBERT and adjust composite scores.
    # Requires: pip install transformers torch
    # Disabled by default — enable once dependencies are installed.
    finbert_enabled: bool = False

    # Max composite-score adjustment from FinBERT sentiment.
    # A fully-positive headline set (+1.0) adds this many points;
    # fully-negative (-1.0) subtracts them.
    finbert_score_weight: float = 5.0

    # Max headlines to score per ticker (controls API + inference cost)
    finbert_max_headlines: int = 5

    # ── SPY trend regime (bear market filter) ─────────────────
    # Enable SPY 200-SMA bear market detection.  When enabled,
    # the risk manager restricts exposure during sustained
    # downtrends (separate from VIX contrarian sizing).
    spy_regime_enabled: bool = True

    # Consecutive days SPY must close below 200-SMA to confirm
    # BEAR regime (prevents whipsaws from single-day dips).
    spy_bear_confirmation_days: int = 3

    # In BEAR regime: reduce max concurrent positions to this
    spy_bear_max_positions: int = 5

    # In BEAR regime: raise min composite score to this
    spy_bear_min_composite_score: float = 30.0

    # In CAUTION regime: reduce max positions to this
    spy_caution_max_positions: int = 6

    # In CAUTION regime: raise min composite score to this
    spy_caution_min_composite_score: float = 22.0

    # SPY drawdown from 52-week high (%) that triggers
    # SEVERE_BEAR — halts all new entries entirely
    spy_severe_drawdown_pct: float = 15.0

    # ── Oil spike → fertilizer boost ──────────────────────────
    # When USO posts a 10%+ weekly gain, temporarily boost
    # composite scores for fertilizer/ag-chemical tickers.
    # Backed by 10-year event study: MOS +6.3% / 79% WR at 20d,
    # survives beta-adjustment and 2022 regime removal.
    # Disabled by default — flip to True after dry-run testing.
    oil_spike_enabled: bool = False

    # Peak composite-score boost on spike day (decays linearly
    # to 0 over oil_spike_window_days).  +8 nudges a score-25
    # signal past the bear-regime threshold of 30.
    oil_spike_boost: float = 8.0

    # Trading days over which the boost decays to zero.
    oil_spike_window_days: int = 20

    # Minimum 5-day USO return to qualify as a spike.
    oil_spike_threshold: float = 0.10

    # Tier 1 tickers that receive the boost (comma-separated in .env).
    oil_spike_tickers: str = "MOS,CF"

    # ── Oil spike → airline mean-reversion (Tier 2) ──────────
    # Airlines dip 1-3 days after a crude spike then snap back.
    # Entry delayed to day 3; smaller boost; shorter decay.
    # Backed by 21-event study: +1.87% excess vs SPY, 71% WR,
    # t=2.21.  UAL strongest (t=3.21, 86% WR).
    oil_spike_airline_tickers: str = "UAL,DAL"
    oil_spike_airline_boost: float = 5.0
    oil_spike_airline_delay_days: int = 3
    oil_spike_airline_decay_days: int = 10

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

    def get_market(self) -> "MarketConfig":
        """Return the market config, defaulting to US if unset."""
        if self.market is not None:
            return self.market
        from trading_bot_bl.market_config import US
        return US

    def path_for(self, kind: str) -> Path:
        """Return the output directory for *kind*, scoped by market.

        US (market_id="US") returns the flat base path (unchanged).
        Non-US returns ``base_path / market_id``.

        ``kind`` is one of: "signals", "execution_logs", "journal",
        "reports", "equity_curve".
        """
        _base_map = {
            "signals": self.signals_dir,
            "execution_logs": Path("execution_logs"),
            "journal": Path("journal"),
            "reports": Path("reports"),
            "equity_curve": Path("execution_logs"),
        }
        base = _base_map.get(kind, Path(kind))
        mkt = self.get_market()
        if mkt.market_id == "US":
            return base
        result = base / mkt.market_id.lower()
        result.mkdir(parents=True, exist_ok=True)
        return result

    @classmethod
    def from_env(cls) -> TradingConfig:
        """Build config from environment variables."""
        alpaca = AlpacaConfig.from_env()
        risk = RiskLimits(
            max_portfolio_exposure_pct=float(
                os.getenv("MAX_EXPOSURE_PCT", "80.0")
            ),
            max_position_pct=float(
                os.getenv("MAX_POSITION_PCT", "10.0")
            ),
            daily_loss_limit_pct=float(
                os.getenv("DAILY_LOSS_LIMIT_PCT", "3.0")
            ),
            min_composite_score=float(
                os.getenv("MIN_COMPOSITE_SCORE", "15.0")
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
            min_backtest_trades=int(
                os.getenv("MIN_BACKTEST_TRADES", "3")
            ),
            max_pbo=float(
                os.getenv("MAX_PBO", "0.50")
            ),
            earnings_blackout_enabled=(
                os.getenv(
                    "EARNINGS_BLACKOUT_ENABLED", "true"
                ).lower() == "true"
            ),
            earnings_blackout_pre_days=int(
                os.getenv("EARNINGS_BLACKOUT_PRE_DAYS", "3")
            ),
            earnings_blackout_post_days=int(
                os.getenv("EARNINGS_BLACKOUT_POST_DAYS", "1")
            ),
            adv_liquidity_enabled=(
                os.getenv(
                    "ADV_LIQUIDITY_ENABLED", "true"
                ).lower() == "true"
            ),
            min_adv_shares=int(
                os.getenv("MIN_ADV_SHARES", "500000")
            ),
            min_adv_dollar_volume=float(
                os.getenv("MIN_ADV_DOLLAR_VOLUME", "5000000")
            ),
            max_adv_participation_pct=float(
                os.getenv("MAX_ADV_PARTICIPATION_PCT", "1.0")
            ),
            ticker_cooldown_days=int(
                os.getenv("TICKER_COOLDOWN_DAYS", "2")
            ),
            cppi_enabled=os.getenv(
                "CPPI_ENABLED", "false"
            ).lower() in ("true", "1", "yes"),
            cppi_max_drawdown_pct=float(
                os.getenv("CPPI_MAX_DRAWDOWN_PCT", "10.0")
            ),
            cppi_multiplier=int(
                os.getenv("CPPI_MULTIPLIER", "5")
            ),
            cppi_min_exposure_pct=float(
                os.getenv("CPPI_MIN_EXPOSURE_PCT", "10.0")
            ),
        )
        return cls(
            alpaca=alpaca,
            risk=risk,
            signals_dir=Path(
                os.getenv("SIGNALS_DIR", "signals")
            ),
            max_entry_slippage_pct=float(
                os.getenv("MAX_ENTRY_SLIPPAGE_PCT", "1.0")
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
            market_sentiment_enabled=os.getenv(
                "MARKET_SENTIMENT_ENABLED", "true"
            ).lower() in ("true", "1", "yes"),
            sentiment_fear_vix=float(
                os.getenv("SENTIMENT_FEAR_VIX", "25.0")
            ),
            sentiment_greed_vix=float(
                os.getenv("SENTIMENT_GREED_VIX", "15.0")
            ),
            sentiment_fear_pc=float(
                os.getenv("SENTIMENT_FEAR_PC", "0.95")
            ),
            sentiment_greed_pc=float(
                os.getenv("SENTIMENT_GREED_PC", "0.65")
            ),
            sentiment_fear_size_mult=float(
                os.getenv("SENTIMENT_FEAR_SIZE_MULT", "1.10")
            ),
            sentiment_greed_size_mult=float(
                os.getenv("SENTIMENT_GREED_SIZE_MULT", "0.90")
            ),
            finbert_enabled=os.getenv(
                "FINBERT_ENABLED", "false"
            ).lower() in ("true", "1", "yes"),
            finbert_score_weight=float(
                os.getenv("FINBERT_SCORE_WEIGHT", "5.0")
            ),
            finbert_max_headlines=int(
                os.getenv("FINBERT_MAX_HEADLINES", "5")
            ),
            spy_regime_enabled=os.getenv(
                "SPY_REGIME_ENABLED", "true"
            ).lower() in ("true", "1", "yes"),
            spy_bear_confirmation_days=int(
                os.getenv("SPY_BEAR_CONFIRMATION_DAYS", "3")
            ),
            spy_bear_max_positions=int(
                os.getenv("SPY_BEAR_MAX_POSITIONS", "5")
            ),
            spy_bear_min_composite_score=float(
                os.getenv("SPY_BEAR_MIN_COMPOSITE_SCORE", "30.0")
            ),
            spy_caution_max_positions=int(
                os.getenv("SPY_CAUTION_MAX_POSITIONS", "6")
            ),
            spy_caution_min_composite_score=float(
                os.getenv("SPY_CAUTION_MIN_COMPOSITE_SCORE", "22.0")
            ),
            spy_severe_drawdown_pct=float(
                os.getenv("SPY_SEVERE_DRAWDOWN_PCT", "15.0")
            ),
            oil_spike_enabled=os.getenv(
                "OIL_SPIKE_ENABLED", "false"
            ).lower()
            in ("true", "1", "yes"),
            oil_spike_boost=float(
                os.getenv("OIL_SPIKE_BOOST", "8.0")
            ),
            oil_spike_window_days=int(
                os.getenv("OIL_SPIKE_WINDOW_DAYS", "20")
            ),
            oil_spike_threshold=float(
                os.getenv("OIL_SPIKE_THRESHOLD", "0.10")
            ),
            oil_spike_tickers=os.getenv(
                "OIL_SPIKE_TICKERS", "MOS,CF"
            ),
            oil_spike_airline_tickers=os.getenv(
                "OIL_SPIKE_AIRLINE_TICKERS", "UAL,DAL"
            ),
            oil_spike_airline_boost=float(
                os.getenv("OIL_SPIKE_AIRLINE_BOOST", "5.0")
            ),
            oil_spike_airline_delay_days=int(
                os.getenv("OIL_SPIKE_AIRLINE_DELAY_DAYS", "3")
            ),
            oil_spike_airline_decay_days=int(
                os.getenv("OIL_SPIKE_AIRLINE_DECAY_DAYS", "10")
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

            # Market sentiment overrides
            if "market_sentiment_enabled" in overrides:
                config.market_sentiment_enabled = bool(
                    overrides["market_sentiment_enabled"]
                )
            for k in (
                "sentiment_fear_vix",
                "sentiment_greed_vix",
                "sentiment_fear_pc",
                "sentiment_greed_pc",
                "sentiment_fear_size_mult",
                "sentiment_greed_size_mult",
            ):
                if k in overrides:
                    setattr(config, k, float(overrides[k]))

            # FinBERT overrides
            if "finbert_enabled" in overrides:
                config.finbert_enabled = bool(
                    overrides["finbert_enabled"]
                )
            if "finbert_score_weight" in overrides:
                config.finbert_score_weight = float(
                    overrides["finbert_score_weight"]
                )
            if "finbert_max_headlines" in overrides:
                config.finbert_max_headlines = int(
                    overrides["finbert_max_headlines"]
                )

            # SPY regime overrides
            if "spy_regime_enabled" in overrides:
                config.spy_regime_enabled = bool(
                    overrides["spy_regime_enabled"]
                )
            for k in (
                "spy_bear_confirmation_days",
                "spy_bear_max_positions",
                "spy_caution_max_positions",
            ):
                if k in overrides:
                    setattr(config, k, int(overrides[k]))
            for k in (
                "spy_bear_min_composite_score",
                "spy_caution_min_composite_score",
                "spy_severe_drawdown_pct",
            ):
                if k in overrides:
                    setattr(config, k, float(overrides[k]))

            # Oil spike overrides
            if "oil_spike_enabled" in overrides:
                config.oil_spike_enabled = bool(
                    overrides["oil_spike_enabled"]
                )
            if "oil_spike_boost" in overrides:
                config.oil_spike_boost = float(
                    overrides["oil_spike_boost"]
                )
            if "oil_spike_window_days" in overrides:
                config.oil_spike_window_days = int(
                    overrides["oil_spike_window_days"]
                )
            if "oil_spike_threshold" in overrides:
                config.oil_spike_threshold = float(
                    overrides["oil_spike_threshold"]
                )
            if "oil_spike_tickers" in overrides:
                config.oil_spike_tickers = str(
                    overrides["oil_spike_tickers"]
                )
            if "oil_spike_airline_tickers" in overrides:
                config.oil_spike_airline_tickers = str(
                    overrides["oil_spike_airline_tickers"]
                )
            if "oil_spike_airline_boost" in overrides:
                config.oil_spike_airline_boost = float(
                    overrides["oil_spike_airline_boost"]
                )
            if "oil_spike_airline_delay_days" in overrides:
                config.oil_spike_airline_delay_days = int(
                    overrides["oil_spike_airline_delay_days"]
                )
            if "oil_spike_airline_decay_days" in overrides:
                config.oil_spike_airline_decay_days = int(
                    overrides["oil_spike_airline_decay_days"]
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

            # ── Market config (required in all config files) ────
            if "market" not in overrides:
                raise ValueError(
                    f"Config file {path} is missing required "
                    f"'market' block. Add "
                    f'\"market\": {{\"market_id\": \"US\"}} '
                    f"(or LSE, TSE) to specify the target market."
                )
            market_block = overrides["market"]
            market_id = market_block.get("market_id", "")
            if not market_id:
                raise ValueError(
                    f"Config file {path}: 'market' block must "
                    f"contain 'market_id' (e.g. \"US\", \"LSE\", "
                    f"\"TSE\")."
                )
            from trading_bot_bl.market_config import (
                get_market_config,
            )
            config.market = get_market_config(market_id)
            log.info(
                f"  Market: {config.market.market_id} "
                f"({config.market.currency})"
            )

            # Wire IBKR allocation from market config
            if config.market.broker_type == "ibkr":
                config.ibkr.max_equity_allocation = (
                    config.market.max_equity_allocation
                )

            # ── IBKR overrides ───────────────────────────────
            if "ibkr" in overrides:
                ib = overrides["ibkr"]
                if "host" in ib:
                    config.ibkr.host = ib["host"]
                if "port" in ib:
                    config.ibkr.port = int(ib["port"])
                if "client_id" in ib:
                    config.ibkr.client_id = int(
                        ib["client_id"]
                    )
                if "account_id" in ib:
                    config.ibkr.account_id = ib["account_id"]
                if "max_equity_allocation" in ib:
                    config.ibkr.max_equity_allocation = float(
                        ib["max_equity_allocation"]
                    )

            log.info(f"Loaded trading config from {path}")
        return config


def get_broker(config: TradingConfig) -> "BrokerInterface":
    """Factory: return the correct broker for the configured market.

    - ``broker_type == "alpaca"`` → ``AlpacaBroker``
    - ``broker_type == "ibkr"`` → ``IBKRBroker`` (Phase 2)
    """
    market = config.get_market()
    if market.broker_type == "alpaca":
        from trading_bot_bl.broker import AlpacaBroker
        return AlpacaBroker(config.alpaca)
    elif market.broker_type == "ibkr":
        raise NotImplementedError(
            "IBKRBroker is not yet implemented (Phase 2). "
            f"Market '{market.market_id}' requires broker_type='ibkr'."
        )
    else:
        raise ValueError(
            f"Unknown broker_type '{market.broker_type}' "
            f"for market '{market.market_id}'"
        )
