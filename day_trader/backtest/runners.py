"""Per-strategy backtest runners.

Convenience functions that wire up a strategy + its appropriate
filters + the engine for a one-call backtest. Used for:

- Development iteration: ``run_orb_backtest(bars, ...)`` during
  strategy tuning
- Validation: automated pass/fail against the plan's thresholds
  before paper-trading
- Comparison: run all v1 strategies side-by-side on the same data

Each runner returns a :class:`BacktestResult`. The caller decides
whether the result passes (``result.passes_plan_criteria``).

Usage::

    from day_trader.backtest.data_loader import load_bars
    from day_trader.backtest.runners import run_orb_backtest

    bars = load_bars(["AAPL", "MSFT"], date(2024, 1, 1), date(2025, 12, 31))
    result = run_orb_backtest(bars, ticker_contexts={...})
    print(result.summary())
    print(result.passes_plan_criteria)
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from day_trader.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from day_trader.config import DayRiskLimits
from day_trader.filters.base import Filter
from day_trader.filters.cooldown import CooldownTracker
from day_trader.filters.cooldown_filter import CooldownFilter
from day_trader.filters.regime_filter import RegimeFilter
from day_trader.filters.rvol_filter import RvolFilter
from day_trader.filters.spread_filter import SpreadFilter
from day_trader.models import Bar, MarketState, TickerContext
from day_trader.strategies.orb_vwap import OrbVwapStrategy
from day_trader.strategies.vwap_pullback import VwapPullbackStrategy


def _default_filters(
    limits: DayRiskLimits,
    cooldowns: Optional[CooldownTracker] = None,
) -> list[Filter]:
    """Filters suitable for backtesting (no broker-dependent filters
    like SymbolLock, Earnings, Liquidity, WholeShareSizing — those
    need live data or broker state not available in backtest)."""
    cd = cooldowns or CooldownTracker(
        ticker_minutes=limits.ticker_cooldown_minutes,
        strategy_minutes=limits.strategy_cooldown_minutes,
    )
    return [
        RegimeFilter(limits),
        CooldownFilter(cd),
        SpreadFilter(limits),
        RvolFilter(limits),
    ]


def run_orb_backtest(
    bars_by_date: dict[date, dict[str, list[Bar]]],
    ticker_contexts: dict[str, TickerContext],
    *,
    config: Optional[BacktestConfig] = None,
    market_state: Optional[MarketState] = None,
    or_minutes: int = 5,
    atr_period: int = 14,
    rr_target: float = 2.0,
    min_premkt_rvol: float = 2.0,
    time_stop_minutes: int = 90,
) -> BacktestResult:
    """Run the ORB+VWAP strategy backtest."""
    cfg = config or BacktestConfig()
    strategy = OrbVwapStrategy(
        or_minutes=or_minutes,
        atr_period=atr_period,
        rr_target=rr_target,
        min_premkt_rvol=min_premkt_rvol,
        time_stop_minutes=time_stop_minutes,
    )
    filters = _default_filters(cfg.risk_limits)
    engine = BacktestEngine(strategy, filters, cfg)
    return engine.run(bars_by_date, ticker_contexts, market_state)


def run_vwap_pullback_backtest(
    bars_by_date: dict[date, dict[str, list[Bar]]],
    ticker_contexts: dict[str, TickerContext],
    *,
    config: Optional[BacktestConfig] = None,
    market_state: Optional[MarketState] = None,
    atr_period: int = 14,
    trend_lookback: int = 30,
    min_rvol: float = 1.5,
    warmup_minutes: int = 30,
) -> BacktestResult:
    """Run the VWAP pullback strategy backtest."""
    cfg = config or BacktestConfig()
    strategy = VwapPullbackStrategy(
        atr_period=atr_period,
        trend_lookback=trend_lookback,
        min_rvol=min_rvol,
        warmup_minutes=warmup_minutes,
    )
    filters = _default_filters(cfg.risk_limits)
    engine = BacktestEngine(strategy, filters, cfg)
    return engine.run(bars_by_date, ticker_contexts, market_state)


def run_all_strategies(
    bars_by_date: dict[date, dict[str, list[Bar]]],
    ticker_contexts: dict[str, TickerContext],
    *,
    config: Optional[BacktestConfig] = None,
    market_state: Optional[MarketState] = None,
) -> dict[str, BacktestResult]:
    """Run all v1 strategies on the same data and return a comparison.

    Returns ``{strategy_name: BacktestResult}``."""
    return {
        "orb_vwap": run_orb_backtest(
            bars_by_date, ticker_contexts,
            config=config, market_state=market_state,
        ),
        "vwap_pullback": run_vwap_pullback_backtest(
            bars_by_date, ticker_contexts,
            config=config, market_state=market_state,
        ),
    }
