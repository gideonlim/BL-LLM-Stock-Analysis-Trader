# Trading Bot

Automated order execution that reads signals from the quant analysis bot, ranks them using Modern Portfolio Theory, and submits bracket orders to Alpaca with comprehensive risk management and position monitoring.

## Setup

1. Install dependencies:
```bash
pip install alpaca-py python-dotenv yfinance numpy pandas
```

2. Get Alpaca API keys from [https://app.alpaca.markets](https://app.alpaca.markets) (start with paper trading).

3. Copy the env template and fill in your keys:
```bash
cp trading_bot/.env.example .env
# Edit .env with your ALPACA_API_KEY and ALPACA_API_SECRET
```

## Usage

```bash
# Dry run -- see what would be traded without submitting orders
python -m trading_bot --dry-run

# Paper trading (default)
python -m trading_bot

# Live trading (be careful)
python -m trading_bot --live

# Monitor only -- check position health, no new orders
python -m trading_bot --monitor-only

# Reset strategy history -- archive old execution logs
python -m trading_bot --reset-history

# With a config file
python -m trading_bot --config trading_config.json
```

## Full Pipeline

```bash
# Step 1: Generate today's signals
python -m quant_analysis_bot --all-stocks --top-n 500

# Step 2: Execute signals
python -m trading_bot
```

## Execution Pipeline (10 Steps)

1. **Load signals** — reads latest `signals_YYYY-MM-DD.json`
2. **Load trade history** — aggregates past execution logs (30 days)
3. **Connect to broker** — initializes Alpaca API
4. **Check market hours** — warns if market closed
5. **Get portfolio + pending orders** — fetches account state
5b. **Cancel stale orders** — cancels orders for tickers with no position (market open only)
6. **Monitor positions** — health checks on held positions (emergency loss, orphans, stale brackets, gaps)
7. **Enrich history** — attributes unrealized P&L to strategies
8. **Build order intents** — converts signals to orders with 3-layer duplicate detection
8b. **Portfolio optimize** — ranks BUY intents by marginal Sharpe contribution (MPT)
9. **Risk check** — validates each intent against exposure, quality, and history constraints
10. **Submit orders** — approved orders go to Alpaca as bracket orders

## Risk Management

The risk manager enforces multiple layers of protection:

- **Circuit breaker** — halts all trading if day P&L drops below −3%
- **Signal quality** — minimum composite score (20), confidence score (2), signal expiry check
- **Strategy history** — blocks strategies with poor track records (only counts strategy-attributable failures, not infrastructure errors)
- **Ticker churn** — 2-day cooldown between trades on the same ticker
- **Portfolio exposure** — won't exceed 80% total exposure
- **Per-stock cap** — no single position above 15% of equity
- **Cash check** — can't spend more than available cash

## Position Monitoring

The monitor checks each held position for:

- **Emergency loss** — down > 10% from entry → close immediately
- **Orphaned positions** — missing SL/TP legs → reattach bracket or close if losing
- **Partial brackets** — one leg missing → warning
- **Price gaps** — price gapped beyond SL or TP → close position
- **Stale brackets** — significant price move without triggering legs → tighten stop loss via trailing stop

Run with `--monitor-only` to check positions without placing new orders.

## Portfolio Optimization

BUY intents are ranked by marginal Sharpe contribution using Modern Portfolio Theory:

- Fetches 60-day daily returns for candidates + held positions
- For each candidate, simulates adding it to the portfolio and measures the change in portfolio Sharpe ratio
- Stocks that improve diversification (low correlation to existing holdings) rank higher
- When capital is limited (can only buy 5-7 of 15 candidates), the top-ranked subset is the best-diversifying group

## CLI Flags

| Flag | Description |
|---|---|
| `--config PATH` | JSON config file (overrides env vars) |
| `--signals-dir PATH` | Override signals directory |
| `--dry-run` | Log orders without submitting |
| `--live` | Use live trading instead of paper |
| `--log-dir PATH` | Where to write execution logs (default: `execution_logs`) |
| `--monitor-only` | Only check existing positions, no new orders |
| `--reset-history` | Archive old execution logs for clean history |

## Risk Limits (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `MAX_EXPOSURE_PCT` | 80.0 | Max total portfolio exposure (%) |
| `MAX_POSITION_PCT` | 15.0 | Max single-stock position (%) |
| `DAILY_LOSS_LIMIT_PCT` | 3.0 | Circuit breaker threshold (%) |
| `MIN_COMPOSITE_SCORE` | 20.0 | Min backtest score to trade |
| `MIN_CONFIDENCE_SCORE` | 2 | Min confidence score (0-6) |
| `HISTORY_LOOKBACK_DAYS` | 30 | Days of execution history to consider |

## Package Structure

```
trading_bot/
├── config.py                # Credentials, risk limits, trading params
├── models.py                # Signal, OrderIntent, OrderResult, PositionAlert
├── broker.py                # Alpaca API wrapper (bracket orders, position mgmt)
├── executor.py              # 10-step execution pipeline
├── risk.py                  # Risk manager (exposure, quality, history checks)
├── history.py               # Trade history aggregation + infrastructure vs strategy classification
├── monitor.py               # Position health monitoring (orphans, emergencies, trailing stops)
├── portfolio_optimizer.py   # MPT-based marginal Sharpe ranking
└── cli.py                   # CLI entry point with monitor-only and reset-history modes
```

## Position Sizing

Position sizes are calculated by the quant analysis bot using the half-Kelly criterion, capped by confidence level (HIGH=20%, MEDIUM=10%, LOW=5% of equity). The trading bot's risk manager can reduce these sizes further to stay within exposure and per-stock limits, but never increases them.

## GitHub Actions

```yaml
name: Daily Trading
on:
  schedule:
    - cron: '0 14 * * 1-5'  # 10:00 AM ET (UTC-4), weekdays only

jobs:
  trade:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install yfinance numpy pandas alpaca-py python-dotenv
      - name: Run analysis bot
        run: python -m quant_analysis_bot --all-stocks
      - name: Execute trades
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_API_SECRET: ${{ secrets.ALPACA_API_SECRET }}
          ALPACA_PAPER: 'true'
        run: python -m trading_bot
      - name: Upload execution logs
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: execution-logs
          path: execution_logs/
```
