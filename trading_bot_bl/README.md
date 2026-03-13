# Trading Bot (Black-Litterman + LLM)

Automated order execution powered by Black-Litterman portfolio optimization. Reads signals from the quant analysis bot, computes optimal portfolio weights using Bayesian return estimation, and submits bracket orders to Alpaca. Optionally enriches views with LLM-generated predictions (ICLR 2025 method).

This is the upgraded version of `trading_bot/`. The original is preserved as a backup.

## What's New vs. Original Trading Bot

- **Black-Litterman optimization** replaces marginal Sharpe ranking as the default portfolio optimizer
- **Ledoit-Wolf covariance shrinkage** for robust estimation (pure numpy, no sklearn needed)
- **LLM-enhanced views** (optional) using repeated sampling for uncertainty estimation
- **Market equilibrium prior** — starts from what the market "believes" rather than nothing
- **Confidence-to-uncertainty mapping** — our 0-6 confidence score directly controls how much each view shifts the posterior
- **Limit orders with slippage control** — protects against gap-up fills at market open
- **Live price fetching + SL/TP recalculation** — ensures bracket orders reflect current market conditions
- **Min/max position limits** — enforces 2% floor and 12% cap per position, 8 max concurrent positions
- News context for LLM prompts via Yahoo Finance / Finnhub
- Falls back gracefully to marginal Sharpe if BL data requirements aren't met

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get Alpaca API keys from [https://app.alpaca.markets](https://app.alpaca.markets) (start with paper trading).

3. Copy the env template and fill in your keys:
```bash
cp .env.example .env
# Edit .env with your ALPACA_API_KEY and ALPACA_API_SECRET
```

**Note:** The `.env` file lives inside this directory (`trading_bot_bl/.env`), not the project root.

4. (Optional) For LLM views, add your API key:
```bash
# In .env:
ANTHROPIC_API_KEY=sk-ant-...
LLM_VIEWS_ENABLED=true
```

## Usage

```bash
# Dry run with Black-Litterman (default)
python -m trading_bot_bl --dry-run

# Paper trading
python -m trading_bot_bl

# Live trading (be careful)
python -m trading_bot_bl --live

# BL + LLM-enhanced views
python -m trading_bot_bl --dry-run --llm-views

# Disable BL, fall back to marginal Sharpe ranking
python -m trading_bot_bl --dry-run --no-bl

# Custom BL parameters
python -m trading_bot_bl --dry-run --bl-tau 0.025

# LLM with custom model and more samples
python -m trading_bot_bl --dry-run --llm-views --llm-model claude-sonnet-4-5-20250514 --llm-samples 30

# Limit LLM to top 5 tickers only
python -m trading_bot_bl --dry-run --llm-views --llm-max-tickers 5

# Monitor only -- check position health, no new orders
python -m trading_bot_bl --monitor-only

# Reset strategy history
python -m trading_bot_bl --reset-history

# With a config file
python -m trading_bot_bl --config trading_config.json
```

## Full Pipeline

```bash
# Step 1: Generate today's signals
python quant_bot.py --all-stocks --top-n 200

# Step 2: Execute with Black-Litterman optimization
python -m trading_bot_bl
```

**Important:** These are two separate steps. If you change sizing/config in quant_bot.py, regenerate signals before running the trading bot.

## How Black-Litterman Works

The classic Markowitz approach requires expected return estimates, which are notoriously unreliable. Black-Litterman solves this by starting from a stable prior (market equilibrium) and blending in our signal-generated views:

1. **Market equilibrium (prior)** — Reverse-optimize from market-cap weights to find the returns the market "believes" in: `pi = delta * Sigma * w`
2. **Signal views** — Each BUY signal becomes a view with an expected return and uncertainty derived from our confidence score (0-6)
3. **LLM views (optional)** — Prompt an LLM N times, use mean as the view and variance as uncertainty (ICLR 2025 method)
4. **Posterior returns** — Bayesian blend of prior + views
5. **Optimal weights** — Analytical solution from posterior returns, subject to long-only and per-stock caps

High-confidence signals shift the posterior more; low-confidence signals defer to the market prior.

### BL Weights vs. Kelly Sizing

BL weights are used for **ranking** (which stocks to buy first). Position sizing uses `max(BL weight * equity, Kelly notional)`. This prevents BL from shrinking well-sized Kelly positions while still allowing BL to scale up when it sees a better allocation.

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
8b. **Portfolio optimize** — Black-Litterman (default) or marginal Sharpe (fallback)
9. **Risk check** — validates each intent against exposure, quality, position size, and history constraints
10. **Submit orders** — fetch live price, recalculate SL/TP, submit as limit bracket order

## Risk Management

The risk manager enforces multiple layers of protection (in order):

- **Circuit breaker** — halts all trading if day P&L drops below -3%
- **Max positions** — won't exceed 8 concurrent positions (configurable)
- **Min position size** — skips orders below 2% of equity (configurable)
- **Signal quality** — minimum composite score (20), confidence score (2), signal expiry check, min backtest trades (5)
- **Strategy history** — blocks strategies with poor track records (only counts strategy-attributable failures, not infrastructure errors)
- **Ticker churn** — 2-day cooldown between trades on the same ticker
- **Portfolio exposure** — won't exceed 80% total exposure
- **Per-stock cap** — no single position above 12% of equity
- **Cash check** — can't spend more than available cash

## Order Entry Protection

Three-layer gap protection prevents bad fills at market open:

1. **Delayed execution** — 10:15 AM ET (45 min after open) lets opening volatility settle
2. **Live price fetch** — gets current price; recalculates SL/TP proportionally if drifted from signal
3. **Limit orders** — entry at `live_price * (1 + 1.5%)` — won't fill if stock gaps up more than 1.5%

## Position Monitoring

The monitor checks each held position for:

- **Emergency loss** — down > 10% from entry, close immediately
- **Orphaned positions** — missing SL/TP legs, reattach bracket or close if losing
- **Partial brackets** — one leg missing, warning
- **Price gaps** — price gapped beyond SL or TP, close position
- **Stale brackets** — significant price move without triggering legs, tighten stop loss via trailing stop

Run with `--monitor-only` to check positions without placing new orders.

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
| `--no-bl` | Disable Black-Litterman, use marginal Sharpe fallback |
| `--llm-views` | Enable LLM-enhanced views (requires API key) |
| `--llm-model MODEL` | Override LLM model (default: claude-haiku-4-5-20251001) |
| `--llm-samples N` | Number of LLM samples per ticker (default: 10) |
| `--llm-max-tickers N` | Max tickers to query LLM for (default: 10) |
| `--bl-tau FLOAT` | Black-Litterman tau parameter (default: 0.05) |

## Environment Variables

### Alpaca (required)

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Alpaca API key |
| `ALPACA_API_SECRET` | — | Alpaca API secret |
| `ALPACA_PAPER` | true | Use paper trading |

### Risk Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_EXPOSURE_PCT` | 80.0 | Max total portfolio exposure (%) |
| `MAX_POSITION_PCT` | 12.0 | Max single-stock position (%) |
| `MAX_POSITIONS` | 8 | Max concurrent positions |
| `MIN_POSITION_PCT` | 2.0 | Skip orders below this % of equity |
| `DAILY_LOSS_LIMIT_PCT` | 3.0 | Circuit breaker threshold (%) |
| `MIN_COMPOSITE_SCORE` | 20.0 | Min backtest score to trade |
| `MIN_CONFIDENCE_SCORE` | 2 | Min confidence score (0-6) |
| `MAX_ENTRY_SLIPPAGE_PCT` | 1.5 | Limit order slippage cap (0 = market order) |
| `HISTORY_LOOKBACK_DAYS` | 30 | Days of execution history to consider |

### Black-Litterman

| Variable | Default | Description |
|---|---|---|
| `USE_BLACK_LITTERMAN` | true | Enable BL optimization |
| `BL_TAU` | 0.05 | Prior uncertainty scalar (0.01-0.1) |
| `BL_LOOKBACK_DAYS` | 60 | Days of returns for covariance estimation |

### LLM Views (optional)

| Variable | Default | Description |
|---|---|---|
| `LLM_VIEWS_ENABLED` | false | Enable LLM view generation |
| `LLM_PROVIDER` | anthropic | LLM provider (anthropic or openai) |
| `LLM_MODEL` | claude-haiku-4-5-20251001 | Model to use |
| `LLM_NUM_SAMPLES` | 10 | Repeated samples per ticker |
| `LLM_MAX_TICKERS` | 10 | Only query top-N tickers by confidence |
| `LLM_TEMPERATURE` | 0.7 | Sampling temperature (needs >0) |
| `LLM_WEIGHT` | 0.3 | Blending weight for LLM views (0-1) |
| `ANTHROPIC_API_KEY` | — | Required if provider is anthropic |
| `OPENAI_API_KEY` | — | Required if provider is openai |

## Package Structure

```
trading_bot_bl/
├── .env.example             # Template — copy to .env and fill in keys
├── config.py                # Credentials, risk limits, BL/LLM params
├── models.py                # Signal, OrderIntent, OrderResult, PositionAlert
├── broker.py                # Alpaca API wrapper (limit + bracket orders)
├── executor.py              # 10-step execution pipeline
├── risk.py                  # Risk manager (exposure, quality, sizing, history)
├── history.py               # Trade history aggregation + infra vs strategy classification
├── monitor.py               # Position health monitoring (orphans, emergencies, trailing stops)
├── black_litterman.py       # BL model: equilibrium, views, posterior, Ledoit-Wolf shrinkage
├── llm_views.py             # LLM-enhanced view generation (ICLR 2025 repeated sampling)
├── news_fetcher.py          # News headline fetcher for LLM context
├── portfolio_optimizer.py   # Unified optimizer: BL (primary) → marginal Sharpe (fallback)
├── cli.py                   # CLI entry point with BL/LLM flags
├── __main__.py              # Enables `python -m trading_bot_bl`
└── test_black_litterman.py  # Tests for BL math (8 tests)
```

## GitHub Actions

Three workflows automate the full pipeline on weekday market hours:

1. **generate_signals.yml** — 9:00 AM ET: runs `quant_bot.py --all-stocks --top-n 200`, commits signals
2. **execute_trades.yml** — 10:15 AM ET: pulls latest signals, runs `python -m trading_bot_bl`, commits logs
3. **monitor_positions.yml** — 10 AM, 12 PM, 2 PM ET: runs `--monitor-only`, uploads logs

### Required Secrets
- `ALPACA_API_KEY`, `ALPACA_API_SECRET`
- `ANTHROPIC_API_KEY` (only if LLM views enabled)

### Optional Variables
Override config via repository variables (Settings → Variables → Actions). All have defaults matching `.env.example`.

## Running Tests

```bash
cd trading_bot_bl
python test_black_litterman.py
```

Tests verify: Ledoit-Wolf shrinkage, equilibrium returns, posterior shifts, confidence mapping, weight optimization, no-views fallback, full pipeline, and LLM view blending.
