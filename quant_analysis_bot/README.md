# Quant Analysis Bot

A multi-timeframe backtesting engine that automatically discovers the best trading strategy for each stock, then generates daily actionable signals with execution details (stop loss, take profit, position sizing).

## Quick Start

```bash
# Install dependencies
pip install yfinance numpy pandas
pip install tqdm  # optional, for nicer progress bars

# Run with default tickers
python -m quant_analysis_bot

# Run with custom tickers
python -m quant_analysis_bot --tickers AAPL MSFT NVDA TSLA

# Run on top 500 US stocks by market cap
python -m quant_analysis_bot --all-stocks --top-n 500

# Use a custom config file
python -m quant_analysis_bot --config my_config.json
```

## CLI Flags

| Flag | Description |
|---|---|
| `--config PATH` | Path to a JSON config file (overrides defaults) |
| `--tickers AAPL MSFT ...` | Override the ticker list |
| `--risk conservative\|moderate\|aggressive` | Set risk profile |
| `--long-only` | Long-only mode — SELL exits to cash (default) |
| `--long-short` | Long+short mode — SELL opens a short position |
| `--all-stocks` | Fetch top US stocks by market cap instead of using the default list |
| `--top-n N` | How many top stocks to analyze with `--all-stocks` (default: 1000) |

## Output Files

Each run produces three types of output in dated files:

```
signals/
  signals_2026-03-13.csv      # Today's signals sorted by composite score
  signals_2026-03-13.json     # Same data in JSON (for automated consumers)

reports/
  backtest_report_2026-03-13.csv   # Full per-strategy, per-timeframe results

trade_logs/
  trades_AAPL_2026-03-13.csv  # Every individual trade for each ticker
```

### Signal Fields

The signal CSV/JSON includes everything a trading bot needs to execute:

- **Signal**: `signal_raw` (1/0/-1), `confidence_score` (0–6), `composite_score`
- **Execution**: `stop_loss_price`, `take_profit_price`, `suggested_position_size_pct`, `signal_expires`
- **Backtest quality**: `sharpe`, `sortino`, `win_rate`, `profit_factor`, `total_trades`, `avg_holding_days`
- **Market context**: `rsi`, `vol_20`, `sma_50`, `sma_200`, `trend`, `volatility`

## Package Structure

```
quant_analysis_bot/
├── config.py       # Default config, risk profiles, load_config()
├── models.py       # TradeRecord, BacktestResult, DailySignal dataclasses
├── indicators.py   # 11 technical indicators (SMA, EMA, RSI, MACD, ATR, etc.)
├── data.py         # Yahoo Finance data fetching with parquet caching
├── strategies.py   # 11 trading strategies + Strategy base class
├── backtest.py     # Backtesting engine + multi-timeframe strategy selector
├── signals.py      # Daily signal generation with stop/profit/sizing
├── universe.py     # Top US stocks by market cap (Wikipedia + bundled fallback)
├── output.py       # CSV/JSON writers for signals, trade logs, reports
├── progress.py     # Terminal progress bar (tqdm fallback)
└── cli.py          # CLI entry point, main pipeline
```

## Strategies

The bot tests 11 strategies per stock across 3 timeframes (3-month, 6-month, 12-month):

1. SMA Crossover (10/50)
2. EMA Crossover (9/21)
3. RSI Mean Reversion
4. MACD Crossover
5. Bollinger Band Mean Reversion
6. Momentum (Rate of Change)
7. Z-Score Mean Reversion
8. Stochastic Oscillator
9. VWAP Trend
10. ADX Trend Following
11. Composite Multi-Indicator

The best strategy is selected via a weighted composite score across all timeframes, with recent performance weighted higher (3mo: 50%, 6mo: 30%, 12mo: 20%).

## Custom Config

Create a JSON file with any of these keys to override defaults:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOG"],
  "lookback_days": 500,
  "long_only": true,
  "risk_profile": "moderate",
  "min_sharpe": 0.5,
  "transaction_cost_bps": 10,
  "backtest_windows": {
    "3mo": 63,
    "6mo": 126,
    "12mo": 252
  },
  "window_weights": {
    "3mo": 0.50,
    "6mo": 0.30,
    "12mo": 0.20
  }
}
```

## Data Caching

Downloaded OHLCV data and stock universe results are cached daily in the `cache/` directory as parquet and JSON files. Delete the cache directory to force fresh downloads.
