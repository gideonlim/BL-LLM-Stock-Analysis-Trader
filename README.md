# Quant Analysis Bot + Automated Trading

A two-stage system: (1) a quant analysis bot that backtests 14 trading strategies across multiple timeframes and generates daily signals, and (2) a Black-Litterman portfolio-optimized trading bot that executes those signals via Alpaca with bracket orders, limit entry, and position monitoring.

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: Generate signals
python -m quant_analysis_bot --all-stocks --top-n 200

# Step 2: Execute trades (configure trading_bot_bl/.env first)
python -m trading_bot_bl --dry-run    # preview orders
python -m trading_bot_bl              # paper trading (default)
```

See [trading_bot_bl/README.md](trading_bot_bl/README.md) for full setup (Alpaca API keys, .env configuration, GitHub Actions automation).

## Signal Generation Only

The quant bot works standalone. It will analyze stocks, backtest every strategy, and output today's signals:

## Scan the Entire Market

To analyze the top US stocks by market cap instead of a fixed list:

```bash
# Scan top 1000 US stocks (default)
python -m quant_analysis_bot --all-stocks

# Scan top 100 only (faster)
python -m quant_analysis_bot --all-stocks --top-n 100

# Scan top 500
python -m quant_analysis_bot --all-stocks --top-n 500
```

The `--all-stocks` flag pulls S&P 500, S&P MidCap 400, and S&P SmallCap 600 tickers from Wikipedia, fetches their market caps via Yahoo Finance, sorts them, and takes the top N. Results are cached for the day so the first run takes a few minutes but subsequent runs are instant. A progress bar with ETA is shown throughout.

## What It Does

For each stock, the bot:

1. Downloads historical price data from Yahoo Finance
2. Computes 30 technical indicators (RSI, MACD, Bollinger Bands, ADX, etc.)
3. Runs 11 different trading strategies against the data
4. Backtests each strategy across 3 timeframes (3-month, 6-month, 12-month)
5. Ranks strategies using a weighted composite score (recent performance weighted higher)
6. Selects the best-performing strategy for that stock
7. Generates today's signal: **BUY**, **EXIT**, or **HOLD** (long-only mode, default)

## Trading Modes

The bot defaults to **long-only mode**, which means SELL signals exit your position to cash rather than opening a short. This is realistic for most retail traders.

```bash
# Long-only (default) -- SELL means "exit to cash"
python -m quant_analysis_bot
python -m quant_analysis_bot --long-only

# Long+Short mode -- SELL means "open a short position"
python -m quant_analysis_bot --long-short
```

In long-only mode, signals are **BUY / EXIT / HOLD**. In long+short mode, signals are **BUY / SELL/SHORT / HOLD**. Backtest results will differ between modes since long-only skips short trades entirely, giving you honest numbers for how the strategy performs without shorting.

You can also set this in a config file: `"long_only": true` or `"long_only": false`.

## Strategies Tested

- SMA Crossover (10/50)
- EMA Crossover (9/21)
- RSI Mean Reversion
- MACD Crossover
- Bollinger Band Mean Reversion
- Momentum (Rate of Change)
- Z-Score Mean Reversion
- Stochastic Oscillator
- VWAP Trend
- ADX Trend Following
- Composite Multi-Indicator

## Output Files

Each run creates three types of output:

### Signals (`signals/`)

- `signals_YYYY-MM-DD.csv` and `.json` -- Today's signal for each stock with confidence level, strategy used, annualized returns, and market context.

### Backtest Report (`reports/`)

- `backtest_report_YYYY-MM-DD.txt` -- Full leaderboard of all 14 strategies per stock, broken out by timeframe (3mo/6mo/12mo), with annualized returns, Sharpe ratio, win rate, max drawdown, and composite ranking.

### Trade Logs (`trade_logs/`)

- `trades_TICKER_YYYY-MM-DD.csv` -- Every individual trade the backtester executed: entry/exit dates, prices, holding period, return, and win/loss outcome. One file per stock.

## Configuration

### Command Line Options

```bash
# Use a custom config file
python -m quant_analysis_bot --config my_config.json

# Override which stocks to analyze
python -m quant_analysis_bot --tickers TSLA AMZN MSFT

# Change risk profile
python -m quant_analysis_bot --risk aggressive

# Scan top 200 US stocks in long-short mode
python -m quant_analysis_bot --all-stocks --top-n 200 --long-short
```

### Config File

Create a JSON file to override any defaults:

```json
{
    "tickers": ["AAPL", "GOOG", "NFLX", "GLD", "TEM"],
    "long_only": true,
    "risk_profile": "moderate",
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

### Risk Profiles

| Profile | Min Win Rate | Max Trades/Month | Signal Threshold |
|---|---|---|---|
| conservative | 55% | 4 | 0.7 |
| moderate | 48% | 8 | 0.5 |
| aggressive | 40% | 15 | 0.3 |

## How Scoring Works

Each strategy is scored on a combination of Sharpe ratio (35%), annualized excess return over buy-and-hold (20%), win rate (15%), profit factor (15%), and max drawdown penalty (15%). Strategies with fewer than 5 trades are penalized for unreliability.

Scores are computed independently for each timeframe, then combined using recency weights: the 3-month window counts for 50%, 6-month for 30%, and 12-month for 20%. This means a strategy that worked well recently but poorly a year ago can still rank high, while a strategy that stopped working gets penalized.

## Reading the Signal Output

Each signal includes:

- **signal** -- BUY, EXIT, or HOLD (long-only) / BUY, SELL/SHORT, or HOLD (long+short)
- **confidence** -- HIGH, MEDIUM, or LOW (based on Sharpe, win rate, profit factor, and trade count)
- **strategy** -- Which of the 14 strategies was selected for this stock
- **annual_return_pct** -- Annualized return of the strategy during backtesting
- **annual_excess_pct** -- How much the strategy beat buy-and-hold (annualized)
- **backtest_period** -- Exact date range and number of trading days tested
- **trend** -- BULLISH/BEARISH/NEUTRAL (based on SMA 50 vs SMA 200)
- **volatility** -- LOW/MEDIUM/HIGH (based on 20-day realized volatility)

A LOW confidence signal with a Sharpe below the minimum threshold is automatically downgraded to HOLD.

## Daily Usage

Run the script once per day, ideally after markets close. The bot caches downloaded data for the day, so re-running won't re-download. To force fresh data, delete the `cache/` folder.

## Adding Stocks

Edit the `tickers` list in `DEFAULT_CONFIG` at the top of the script, or pass `--tickers` on the command line. Any ticker supported by Yahoo Finance will work (stocks, ETFs, crypto pairs like BTC-USD, etc.).

## Estimated Run Times

Analysis runs in parallel by default (`--workers 0` auto-detects CPU cores, capped at 8).

| Stocks | 1 worker (sequential) | 2 workers (GH Actions) | 8 workers (local) |
|---|---|---|---|
| 5-10 | ~30 seconds | ~30 seconds | ~30 seconds |
| 100 | ~5 minutes | ~3 minutes | ~1.5 minutes |
| 500 | ~25 minutes | ~14 minutes | ~5 minutes |
| 1000 | ~50 minutes | ~28 minutes | ~10 minutes |

Times depend on your internet speed and Yahoo Finance rate limits. The stock universe fetch (with `--all-stocks`) adds a few extra minutes on the first run but is cached afterward. Use `--workers 1` to disable multiprocessing.

## Automated Trading (trading_bot_bl)

The trading bot reads signals from the `signals/` directory and executes them via Alpaca:

- **Black-Litterman portfolio optimization** ranks and sizes positions
- **Limit orders** with 1.5% slippage cap (no market orders)
- **Live price fetching** with SL/TP recalculation
- **Position monitoring** (orphaned brackets, emergency losses, trailing stops)
- **Risk management** (circuit breaker, exposure caps, min/max position sizes)
- **Optional LLM-enhanced views** (ICLR 2025 repeated sampling method)

Default config targets ~8 positions at 10-12% each on a $100k account.

See [trading_bot_bl/README.md](trading_bot_bl/README.md) for details.

## GitHub Actions

Three workflows automate the full pipeline on weekday market hours:

1. **generate_signals.yml** — 9:00 AM ET: generate signals, commit to repo
2. **execute_trades.yml** — 10:15 AM ET: execute trades via Alpaca
3. **monitor_positions.yml** — every 2h during market: position health checks

Requires GitHub Secrets: `ALPACA_API_KEY`, `ALPACA_API_SECRET`. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for full architecture documentation.

## Requirements

- Python 3.8+
- pandas, numpy, yfinance, pyarrow, lxml (see `requirements.txt`)
