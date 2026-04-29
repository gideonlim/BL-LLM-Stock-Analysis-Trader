"""Day-trader configuration.

Mirrors the env-var-driven pattern in ``trading_bot_bl/config.py``.
Defaults track the approved plan
(~/.claude/plans/i-want-to-allocate-bubbly-knuth.md).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DayRiskLimits:
    """Day-trader-specific risk constraints.

    Tighter than the swing limits because intraday horizons,
    smaller capital base (25% sub-budget), and no overnight risk.
    """

    # ── Capital & exposure ─────────────────────────────────────
    # Fraction of TOTAL account equity allocated to day trading.
    # 0.25 = 25% — locked by the approved plan.
    budget_pct: float = 0.25

    # Max notional in any single day-trade position, as % of the
    # day-trade sub-budget. 100% / max_positions gives even sizing;
    # using 50% lets a high-conviction trade take a double slot.
    max_position_pct_of_budget: float = 50.0

    # Max concurrent day-trade positions.
    max_positions: int = 3

    # ── Daily stops ────────────────────────────────────────────
    # Daily loss kill switch — % of TOTAL equity (not sub-budget).
    # Trips → no new entries for the rest of the day.
    daily_loss_limit_pct: float = 1.5

    # Max trades placed per day (entries + scaling). Prevents
    # over-trading when many setups fire.
    max_trades_per_day: int = 8

    # ── Per-trade ──────────────────────────────────────────────
    # Risk per trade as % of TOTAL equity. ATR-sized stops scale
    # qty so each trade caps at this dollar loss.
    per_trade_risk_pct: float = 0.25

    # Hard floor on share count after ATR sizing. Below this we
    # reject (whole-share rounding makes risk inaccurate).
    min_qty: int = 1

    # ── Liquidity ──────────────────────────────────────────────
    min_adv_shares: int = 500_000
    min_adv_dollar_volume: float = 5_000_000.0
    max_spread_bps_above_10: float = 15.0
    max_spread_bps_at_or_under_10: float = 30.0
    min_premkt_rvol: float = 2.0
    min_intraday_rvol: float = 1.5

    # ── Market regime cutoffs ──────────────────────────────────
    halt_above_vix: float = 35.0
    halt_below_spy_sma200: bool = True
    halt_in_severe_bear: bool = True

    # ── Cooldowns (no-revenge) ─────────────────────────────────
    # After a losing trade on a ticker, bench the ticker for N min.
    ticker_cooldown_minutes: int = 60
    # After any losing trade (any ticker), pause that strategy
    # for N min to avoid revenge cycles.
    strategy_cooldown_minutes: int = 30


@dataclass
class DayTradeConfig:
    """Top-level day-trader runtime configuration."""

    risk: DayRiskLimits = field(default_factory=DayRiskLimits)

    # ── Paths ──────────────────────────────────────────────────
    # Shared with trading_bot_bl so analytics see all trades together.
    journal_dir: Path = Path("execution_logs/journal")
    log_dir: Path = Path("execution_logs")
    # Day-trade-specific scratch (sequence counter, BarCache snaps,
    # incident files).
    state_dir: Path = Path("execution_logs/day_trader")

    # ── Modes ──────────────────────────────────────────────────
    dry_run: bool = False

    # ── Universe ───────────────────────────────────────────────
    # Russell 1000 by default; override via env.
    universe_source: str = "russell_1000"
    universe_path: Path | None = None  # CSV override
    max_watchlist_size: int = 150

    # ── Market data feed ──────────────────────────────────────
    # Alpaca data plan tied to the API account (NOT paper/live):
    #   "sip"  — Algo Trader Plus or Unlimited subscription.
    #            Full consolidated tape (~100% of volume). Required
    #            for Stocks-in-Play RVOL calculations.
    #   "iex"  — Free Basic tier. IEX-only (~3-5% of volume) —
    #            insufficient for the day-trader's RVOL gating.
    # Default "sip" matches the plan's locked decision (user has
    # Algo Trader Plus). Override only if downgrading subscription.
    data_feed: str = "sip"

    # ── Schedule offsets (minutes) ─────────────────────────────
    # Resolved via day_trader/calendar.py against the live NYSE
    # session — never hard-coded against wall-clock 09:30/16:00.
    catalyst_refresh_min_before_open: int = 90
    premarket_scan_min_before_open: int = 60
    regime_snapshot_min_before_open: int = 5
    first_scan_min_after_open: int = 5
    exit_only_min_before_close: int = 15
    force_flat_min_before_close: int = 5

    # ── Telegram alerts (optional) ─────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls) -> "DayTradeConfig":
        risk = DayRiskLimits(
            budget_pct=float(os.getenv("DAYTRADE_BUDGET_PCT", "25.0")) / 100,
            max_positions=int(os.getenv("DAYTRADE_MAX_POSITIONS", "3")),
            daily_loss_limit_pct=float(
                os.getenv("DAYTRADE_DAILY_LOSS_LIMIT_PCT", "1.5")
            ),
            max_trades_per_day=int(
                os.getenv("DAYTRADE_MAX_TRADES_PER_DAY", "8")
            ),
            per_trade_risk_pct=float(
                os.getenv("DAYTRADE_PER_TRADE_RISK_PCT", "0.25")
            ),
            min_premkt_rvol=float(
                os.getenv("DAYTRADE_MIN_PREMKT_RVOL", "2.0")
            ),
            min_intraday_rvol=float(
                os.getenv("DAYTRADE_MIN_INTRADAY_RVOL", "1.5")
            ),
            halt_above_vix=float(
                os.getenv("DAYTRADE_HALT_ABOVE_VIX", "35.0")
            ),
            ticker_cooldown_minutes=int(
                os.getenv("DAYTRADE_TICKER_COOLDOWN_MIN", "60")
            ),
            strategy_cooldown_minutes=int(
                os.getenv("DAYTRADE_STRATEGY_COOLDOWN_MIN", "30")
            ),
        )

        return cls(
            risk=risk,
            journal_dir=Path(
                os.getenv(
                    "DAYTRADE_JOURNAL_DIR", "execution_logs/journal"
                )
            ),
            log_dir=Path(
                os.getenv("DAYTRADE_LOG_DIR", "execution_logs")
            ),
            state_dir=Path(
                os.getenv(
                    "DAYTRADE_STATE_DIR", "execution_logs/day_trader"
                )
            ),
            dry_run=os.getenv("DAYTRADE_DRY_RUN", "false").lower()
            in ("true", "1", "yes"),
            universe_source=os.getenv(
                "DAYTRADE_UNIVERSE_SOURCE", "russell_1000"
            ),
            max_watchlist_size=int(
                os.getenv("DAYTRADE_MAX_WATCHLIST", "150")
            ),
            data_feed=os.getenv("DAYTRADE_DATA_FEED", "sip").lower(),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )

    def __repr__(self) -> str:
        # Avoid leaking secrets in logs
        bot = "set" if self.telegram_bot_token else "unset"
        return (
            f"DayTradeConfig(budget={self.risk.budget_pct:.0%}, "
            f"max_positions={self.risk.max_positions}, "
            f"daily_loss={self.risk.daily_loss_limit_pct}%, "
            f"per_trade_risk={self.risk.per_trade_risk_pct}%, "
            f"dry_run={self.dry_run}, telegram={bot})"
        )
