"""Day-trader backtest engine.

Event-driven minute-bar backtester that replays historical data
through the same strategy + filter pipeline the live daemon uses.
This ensures the backtest sees the same signal/reject/fill logic
as production — no "backtester says yes but live says no" drift.

Key design decisions:

- Reuses the real ``BarCache``, ``FilterPipeline``, ``DayRiskManager``,
  and strategy ``scan_ticker`` / ``manage`` methods. No separate
  "backtest mode" code paths in the strategies.
- Simulates fills at the next bar's open (conservative: you can't
  trade at the bar you scanned on — that bar is already closed).
- Models slippage as a configurable fraction of ATR (default 10%).
- Models commission as $0 (Alpaca is zero-commission; override for
  other brokers).
- Tracks per-trade P&L, filter rejection histogram, and aggregate
  metrics (Sharpe, profit factor, max drawdown, win rate).
- Outputs a ``BacktestResult`` that can be compared against the
  plan's pass criteria: OOS Sharpe > 1.0, PF > 1.3, max DD < 8%.

Usage::

    from day_trader.backtest.runners import run_orb_backtest
    result = run_orb_backtest(
        bars_by_ticker={...},  # dict[str, list[Bar]]
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
    )
    print(result.summary())
"""
