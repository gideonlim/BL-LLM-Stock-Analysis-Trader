# TP Experiment — Artifact Analysis

**Source CSV**: `research_output\tp_experiment_raw_2026-04-09.csv`
**Rows**: 25425  **Aggregated cells**: 70  **Modes**: current, capped@1.0, capped@1.5, capped@2.0, capped+strategy@1.5

**Thresholds**: min_trades=20, min_losses=5, max_pf=50.0

## Artifact hall of fame

Top 10 (strategy, mode) cells with the most inflated PF values.  These are the ones pulling the portfolio-level PF ranking away from reality.

| Strategy | Mode | Trades | Est Losses | PF | Edge | Reason |
|----------|------|-------:|-----------:|---:|-----:|--------|
| RSI Mean Reversion | capped@1.0 | 127 | 31.0 | 4160.25 | 3.21 | PF=4160.2 implausibly high (denominator effect) |
| RSI Mean Reversion | current | 127 | 31.0 | 3720.30 | 3.21 | PF=3720.3 implausibly high (denominator effect) |
| RSI Mean Reversion | capped@1.5 | 127 | 31.0 | 3720.30 | 3.21 | PF=3720.3 implausibly high (denominator effect) |
| RSI Mean Reversion | capped@2.0 | 127 | 31.0 | 3720.30 | 3.21 | PF=3720.3 implausibly high (denominator effect) |
| Bollinger Band Mean Reversion | capped@1.0 | 229 | 59.0 | 3680.00 | 2.81 | PF=3680.0 implausibly high (denominator effect) |
| Bollinger Band Mean Reversion | capped@1.5 | 229 | 59.0 | 3680.00 | 2.81 | PF=3680.0 implausibly high (denominator effect) |
| RSI Mean Reversion | capped+strategy@1.5 | 127 | 31.0 | 3620.00 | 3.21 | PF=3620.0 implausibly high (denominator effect) |
| Bollinger Band Mean Reversion | capped+strategy@1.5 | 229 | 59.0 | 3185.00 | 2.81 | PF=3185.0 implausibly high (denominator effect) |
| Bollinger Band Mean Reversion | current | 229 | 62.0 | 1638.60 | 2.65 | PF=1638.6 implausibly high (denominator effect) |
| Bollinger Band Mean Reversion | capped@2.0 | 229 | 62.0 | 1638.60 | 2.65 | PF=1638.6 implausibly high (denominator effect) |

Total artifact cells: **10**

## Portfolio-level recommendation

### Ranked by `tb_profit_factor` (with stricter filter)

| Mode | Weight | % |
|------|-------:|---:|
| capped@1.0 | 3890 | 62.2% |
| current | 1916 | 30.6% |
| capped@2.0 | 290 | 4.6% |
| capped@1.5 | 156 | 2.5% |

**PF winner**: `capped@1.0`

### Ranked by `tb_edge_ratio` (artifact-resistant)

| Mode | Weight | % |
|------|-------:|---:|
| current | 4159 | 66.5% |
| capped@1.0 | 2093 | 33.5% |

**Edge winner**: `current`

### Per-family winners (by edge_ratio)

**mean_reversion**

| Mode | Weight | % |
|------|-------:|---:|
| capped@1.0 | 718 | 54.1% |
| current | 610 | 45.9% |

→ `capped@1.0` preferred for mean_reversion

**trend_following**

| Mode | Weight | % |
|------|-------:|---:|
| current | 3549 | 72.1% |
| capped@1.0 | 1375 | 27.9% |

→ `current` preferred for trend_following

## Per-strategy analysis

Each row is one (strategy, mode) cell.  `est_losses` is the trade-count-weighted estimate of actual losing trades in the sample.  Cells with `est_losses < 5` or `PF > 50.0` are flagged as denominator artifacts because the PF ratio becomes unreliable when the loss denominator is tiny.

### 52-Week High Momentum (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 1953 | 755.0 | 0.0 | 47.1 | 0.35 | 0.77 | -10.7 |  |
| capped@1.0 | 1953 | 754.0 | 0.0 | 47.1 | 0.38 | 0.77 | -10.7 |  |
| capped@1.5 | 1953 | 755.0 | 0.0 | 47.1 | 0.36 | 0.77 | -10.7 |  |
| capped@2.0 | 1953 | 755.0 | 0.0 | 47.1 | 0.35 | 0.77 | -10.7 |  |
| capped+strategy@1.5 | 1953 | 755.0 | 0.0 | 47.1 | 0.36 | 0.77 | -10.7 |  |

- **Best by PF (artifact-free)**: `capped@1.0` — PF 0.38 vs base 0.35 (Δ +0.02)
- **Best by edge_ratio**: `current` — edge 0.77 vs base 0.77 (Δ +0.00)
- **Agreement**: ⚠️ metrics disagree — UNCERTAIN winner

### ADX Trend Following (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 227 | 119.0 | 0.0 | 50.0 | 0.00 | 0.79 | -7.5 |  |
| capped@1.0 | 227 | 113.0 | 0.0 | 50.0 | 0.00 | 0.72 | -7.5 |  |
| capped@1.5 | 227 | 116.0 | 0.0 | 50.0 | 0.00 | 0.72 | -7.5 |  |
| capped@2.0 | 227 | 119.0 | 0.0 | 50.0 | 0.00 | 0.79 | -7.5 |  |
| capped+strategy@1.5 | 227 | 116.0 | 0.0 | 50.0 | 0.00 | 0.72 | -7.5 |  |

- **Best by PF (artifact-free)**: `current` — PF 0.00 vs base 0.00 (Δ +0.00)
- **Best by edge_ratio**: `current` — edge 0.79 vs base 0.79 (Δ +0.00)
- **Agreement**: ✅ both metrics agree

### Bollinger Band Mean Reversion (_mean_reversion_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 229 | 62.0 | 0.0 | 0.0 | 1638.60 | 2.65 | -5.7 | ARTIFACT: PF=1638.6 implausibly high (denominator effect) |
| capped@1.0 | 229 | 59.0 | 0.0 | 0.0 | 3680.00 | 2.81 | -5.7 | ARTIFACT: PF=3680.0 implausibly high (denominator effect) |
| capped@1.5 | 229 | 59.0 | 0.0 | 0.0 | 3680.00 | 2.81 | -5.7 | ARTIFACT: PF=3680.0 implausibly high (denominator effect) |
| capped@2.0 | 229 | 62.0 | 0.0 | 0.0 | 1638.60 | 2.65 | -5.7 | ARTIFACT: PF=1638.6 implausibly high (denominator effect) |
| capped+strategy@1.5 | 229 | 59.0 | 0.0 | 0.0 | 3185.00 | 2.81 | -5.7 | ARTIFACT: PF=3185.0 implausibly high (denominator effect) |

- **Best by PF (artifact-free)**: _(none qualified)_
- **Best by edge_ratio**: _(none qualified)_

### Composite Multi-Indicator (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 156 | 47.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |  |
| capped@1.0 | 156 | 44.0 | 0.0 | 0.0 | 1.74 | 2.00 | -9.1 |  |
| capped@1.5 | 156 | 46.0 | 0.0 | 0.0 | 1.98 | 2.00 | -9.1 |  |
| capped@2.0 | 156 | 47.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |  |
| capped+strategy@1.5 | 156 | 46.0 | 0.0 | 0.0 | 1.98 | 2.00 | -9.1 |  |

- **Best by PF (artifact-free)**: `capped@1.5` — PF 1.98 vs base 1.74 (Δ +0.24)
- **Best by edge_ratio**: `capped@1.0` — edge 2.00 vs base 1.88 (Δ +0.12)
- **Agreement**: ⚠️ metrics disagree — UNCERTAIN winner

### Donchian Breakout (20/55) (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 14 | 11.0 | 0.0 | 100.0 | 0.00 | 0.11 | -14.2 |  |
| capped@1.0 | 14 | 11.0 | 0.0 | 100.0 | 0.00 | 0.11 | -14.2 |  |
| capped@1.5 | 14 | 11.0 | 0.0 | 100.0 | 0.00 | 0.11 | -14.2 |  |
| capped@2.0 | 14 | 11.0 | 0.0 | 100.0 | 0.00 | 0.11 | -14.2 |  |
| capped+strategy@1.5 | 14 | 11.0 | 0.0 | 100.0 | 0.00 | 0.11 | -14.2 |  |

- **Best by PF (artifact-free)**: _(none qualified)_
- **Best by edge_ratio**: _(none qualified)_

### EMA Crossover (9/21) (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 111 | 48.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |  |
| capped@1.0 | 111 | 46.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |  |
| capped@1.5 | 111 | 48.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |  |
| capped@2.0 | 111 | 48.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |  |
| capped+strategy@1.5 | 111 | 48.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |  |

- **Best by PF (artifact-free)**: `current` — PF 0.00 vs base 0.00 (Δ +0.00)
- **Best by edge_ratio**: `current` — edge 0.32 vs base 0.32 (Δ +0.00)
- **Agreement**: ✅ both metrics agree

### MACD Crossover (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 290 | 117.0 | 0.0 | 0.0 | 1.87 | 1.20 | -7.7 |  |
| capped@1.0 | 290 | 112.0 | 0.0 | 0.0 | 1.62 | 1.20 | -7.7 |  |
| capped@1.5 | 290 | 117.0 | 0.0 | 0.0 | 1.79 | 1.20 | -7.7 |  |
| capped@2.0 | 290 | 117.0 | 0.0 | 0.0 | 1.91 | 1.20 | -7.7 |  |
| capped+strategy@1.5 | 290 | 117.0 | 0.0 | 0.0 | 1.79 | 1.20 | -7.7 |  |

- **Best by PF (artifact-free)**: `capped@2.0` — PF 1.91 vs base 1.87 (Δ +0.03)
- **Best by edge_ratio**: `current` — edge 1.20 vs base 1.20 (Δ +0.00)
- **Agreement**: ⚠️ metrics disagree — UNCERTAIN winner

### Momentum (Rate of Change) (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 1219 | 427.0 | 0.0 | 37.5 | 0.48 | 0.97 | -8.8 |  |
| capped@1.0 | 1219 | 423.0 | 0.0 | 37.2 | 0.55 | 1.00 | -8.8 |  |
| capped@1.5 | 1219 | 426.0 | 0.0 | 37.5 | 0.49 | 0.97 | -8.8 |  |
| capped@2.0 | 1219 | 426.0 | 0.0 | 37.5 | 0.51 | 0.97 | -8.8 |  |
| capped+strategy@1.5 | 1219 | 426.0 | 0.0 | 37.5 | 0.49 | 0.97 | -8.8 |  |

- **Best by PF (artifact-free)**: `capped@1.0` — PF 0.55 vs base 0.48 (Δ +0.07)
- **Best by edge_ratio**: `capped@1.0` — edge 1.00 vs base 0.97 (Δ +0.03)
- **Agreement**: ✅ both metrics agree

### Multi-Factor Momentum MR (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 49 | 26.0 | 0.0 | 100.0 | 0.00 | 0.83 | -8.9 |  |
| capped@1.0 | 49 | 25.0 | 0.0 | 66.7 | 0.00 | 0.83 | -8.9 |  |
| capped@1.5 | 49 | 25.0 | 0.0 | 66.7 | 0.00 | 0.83 | -8.9 |  |
| capped@2.0 | 49 | 25.0 | 0.0 | 66.7 | 0.00 | 0.83 | -8.9 |  |
| capped+strategy@1.5 | 49 | 25.0 | 0.0 | 66.7 | 0.00 | 0.83 | -8.9 |  |

- **Best by PF (artifact-free)**: `current` — PF 0.00 vs base 0.00 (Δ +0.00)
- **Best by edge_ratio**: `current` — edge 0.83 vs base 0.83 (Δ +0.00)
- **Agreement**: ✅ both metrics agree

### RSI Mean Reversion (_mean_reversion_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 127 | 31.0 | 0.0 | 0.0 | 3720.30 | 3.21 | -3.8 | ARTIFACT: PF=3720.3 implausibly high (denominator effect) |
| capped@1.0 | 127 | 31.0 | 0.0 | 0.0 | 4160.25 | 3.21 | -3.8 | ARTIFACT: PF=4160.2 implausibly high (denominator effect) |
| capped@1.5 | 127 | 31.0 | 0.0 | 0.0 | 3720.30 | 3.21 | -3.8 | ARTIFACT: PF=3720.3 implausibly high (denominator effect) |
| capped@2.0 | 127 | 31.0 | 0.0 | 0.0 | 3720.30 | 3.21 | -3.8 | ARTIFACT: PF=3720.3 implausibly high (denominator effect) |
| capped+strategy@1.5 | 127 | 31.0 | 33.3 | 0.0 | 3620.00 | 3.21 | -3.8 | ARTIFACT: PF=3620.0 implausibly high (denominator effect) |

- **Best by PF (artifact-free)**: _(none qualified)_
- **Best by edge_ratio**: _(none qualified)_

### SMA Crossover (10/50) (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 46 | 24.0 | 0.0 | 75.0 | 0.00 | 0.69 | -10.8 |  |
| capped@1.0 | 46 | 24.0 | 0.0 | 75.0 | 0.00 | 0.69 | -10.8 |  |
| capped@1.5 | 46 | 24.0 | 0.0 | 75.0 | 0.00 | 0.69 | -10.8 |  |
| capped@2.0 | 46 | 24.0 | 0.0 | 75.0 | 0.00 | 0.69 | -10.8 |  |
| capped+strategy@1.5 | 46 | 24.0 | 0.0 | 75.0 | 0.00 | 0.69 | -10.8 |  |

- **Best by PF (artifact-free)**: `current` — PF 0.00 vs base 0.00 (Δ +0.00)
- **Best by edge_ratio**: `current` — edge 0.69 vs base 0.69 (Δ +0.00)
- **Agreement**: ✅ both metrics agree

### Stochastic Oscillator (_mean_reversion_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 718 | 249.0 | 0.0 | 28.6 | 1.26 | 1.21 | -6.6 |  |
| capped@1.0 | 718 | 242.0 | 0.0 | 25.0 | 1.41 | 1.22 | -6.6 |  |
| capped@1.5 | 718 | 249.0 | 0.0 | 28.6 | 1.26 | 1.21 | -6.6 |  |
| capped@2.0 | 718 | 249.0 | 0.0 | 28.6 | 1.26 | 1.21 | -6.6 |  |
| capped+strategy@1.5 | 718 | 243.0 | 0.0 | 25.0 | 1.20 | 1.18 | -6.6 |  |

- **Best by PF (artifact-free)**: `capped@1.0` — PF 1.41 vs base 1.26 (Δ +0.15)
- **Best by edge_ratio**: `capped@1.0` — edge 1.22 vs base 1.21 (Δ +0.01)
- **Agreement**: ✅ both metrics agree

### VWAP Trend (_trend_following_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 873 | 414.0 | 0.0 | 50.0 | 1.04 | 1.25 | -5.7 |  |
| capped@1.0 | 873 | 336.0 | 50.0 | 33.3 | 0.99 | 1.00 | -5.7 |  |
| capped@1.5 | 873 | 374.0 | 40.0 | 50.0 | 0.90 | 1.08 | -5.7 |  |
| capped@2.0 | 873 | 394.0 | 25.0 | 50.0 | 0.99 | 1.14 | -5.7 |  |
| capped+strategy@1.5 | 873 | 374.0 | 40.0 | 50.0 | 0.90 | 1.08 | -5.7 |  |

- **Best by PF (artifact-free)**: `current` — PF 1.04 vs base 1.04 (Δ +0.00)
- **Best by edge_ratio**: `current` — edge 1.25 vs base 1.25 (Δ +0.00)
- **Agreement**: ✅ both metrics agree

### Z-Score Mean Reversion (_mean_reversion_)

| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |
|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|
| current | 610 | 173.0 | 0.0 | 0.0 | 2.28 | 1.48 | -6.1 |  |
| capped@1.0 | 610 | 173.0 | 0.0 | 0.0 | 2.28 | 1.48 | -6.1 |  |
| capped@1.5 | 610 | 173.0 | 0.0 | 0.0 | 2.28 | 1.48 | -6.1 |  |
| capped@2.0 | 610 | 173.0 | 0.0 | 0.0 | 2.28 | 1.48 | -6.1 |  |
| capped+strategy@1.5 | 610 | 173.0 | 0.0 | 0.0 | 2.09 | 1.46 | -6.1 |  |

- **Best by PF (artifact-free)**: `current` — PF 2.28 vs base 2.28 (Δ +0.00)
- **Best by edge_ratio**: `current` — edge 1.48 vs base 1.48 (Δ +0.00)
- **Agreement**: ✅ both metrics agree
