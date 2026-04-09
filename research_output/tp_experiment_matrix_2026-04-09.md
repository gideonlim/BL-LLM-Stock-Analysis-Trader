# TP Mode Experiment — Decision Matrix (2026-04-09)

**Run ID**: `c178ecee`  **Timestamp**: 2026-04-09T20:51:19

**Tickers**: 113  **Modes**: 5  **Wall-clock**: 140s

All medians are across (ticker × window) cells.  Only cells with TB trades > 0 are aggregated.  Winner rule: highest `tb_pf` among modes with ≥20 total TB trades and `max_dd >= base_dd - 5pp`, tie-break prefers simpler modes.

## 52-Week High Momentum (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 1953 | 0.0 | 47.1 | 50.0 | 0.35 | 0.77 | -10.7 |
| capped@1.0 | 1953 | 0.0 | 47.1 | 50.0 | 0.38 | 0.77 | -10.7 |
| capped@1.5 | 1953 | 0.0 | 47.1 | 50.0 | 0.36 | 0.77 | -10.7 |
| capped@2.0 | 1953 | 0.0 | 47.1 | 50.0 | 0.35 | 0.77 | -10.7 |
| capped+strategy@1.5 | 1953 | 0.0 | 47.1 | 50.0 | 0.36 | 0.77 | -10.7 |

**Winner**: `capped@1.0` (PF 0.38 vs base 0.35, Δ=+0.02)

## ADX Trend Following (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.79 | -7.5 |
| capped@1.0 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.72 | -7.5 |
| capped@1.5 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.72 | -7.5 |
| capped@2.0 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.79 | -7.5 |
| capped+strategy@1.5 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.72 | -7.5 |

**Winner**: `current` (PF 0.00 vs base 0.00, Δ=+0.00)

## Bollinger Band Mean Reversion (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 229 | 0.0 | 0.0 | 100.0 | 1638.60 | 2.65 | -5.7 |
| capped@1.0 | 229 | 0.0 | 0.0 | 33.3 | 3680.00 | 2.81 | -5.7 |
| capped@1.5 | 229 | 0.0 | 0.0 | 50.0 | 3680.00 | 2.81 | -5.7 |
| capped@2.0 | 229 | 0.0 | 0.0 | 100.0 | 1638.60 | 2.65 | -5.7 |
| capped+strategy@1.5 | 229 | 0.0 | 0.0 | 0.0 | 3185.00 | 2.81 | -5.7 |

**Winner**: `capped@1.0` (PF 3680.00 vs base 1638.60, Δ=+2041.40)

## Composite Multi-Indicator (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |
| capped@1.0 | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 2.00 | -9.1 |
| capped@1.5 | 156 | 0.0 | 0.0 | 0.0 | 1.98 | 2.00 | -9.1 |
| capped@2.0 | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |
| capped+strategy@1.5 | 156 | 0.0 | 0.0 | 0.0 | 1.98 | 2.00 | -9.1 |

**Winner**: `capped@1.5` (PF 1.98 vs base 1.74, Δ=+0.24)

## Donchian Breakout (20/55) (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 14 | 0.0 | 100.0 | 0.0 | 0.00 | 0.11 | -14.2 |
| capped@1.0 | 14 | 0.0 | 100.0 | 0.0 | 0.00 | 0.11 | -14.2 |
| capped@1.5 | 14 | 0.0 | 100.0 | 0.0 | 0.00 | 0.11 | -14.2 |
| capped@2.0 | 14 | 0.0 | 100.0 | 0.0 | 0.00 | 0.11 | -14.2 |
| capped+strategy@1.5 | 14 | 0.0 | 100.0 | 0.0 | 0.00 | 0.11 | -14.2 |

**Winner**: _(no mode passed filters)_

## EMA Crossover (9/21) (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 111 | 0.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |
| capped@1.0 | 111 | 0.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |
| capped@1.5 | 111 | 0.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |
| capped@2.0 | 111 | 0.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |
| capped+strategy@1.5 | 111 | 0.0 | 0.0 | 0.0 | 0.00 | 0.32 | -6.4 |

**Winner**: `current` (PF 0.00 vs base 0.00, Δ=+0.00)

## MACD Crossover (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 290 | 0.0 | 0.0 | 50.0 | 1.87 | 1.20 | -7.7 |
| capped@1.0 | 290 | 0.0 | 0.0 | 0.0 | 1.62 | 1.20 | -7.7 |
| capped@1.5 | 290 | 0.0 | 0.0 | 33.3 | 1.79 | 1.20 | -7.7 |
| capped@2.0 | 290 | 0.0 | 0.0 | 50.0 | 1.91 | 1.20 | -7.7 |
| capped+strategy@1.5 | 290 | 0.0 | 0.0 | 33.3 | 1.79 | 1.20 | -7.7 |

**Winner**: `capped@2.0` (PF 1.91 vs base 1.87, Δ=+0.03)

## Momentum (Rate of Change) (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 1219 | 0.0 | 37.5 | 50.0 | 0.48 | 0.97 | -8.8 |
| capped@1.0 | 1219 | 0.0 | 37.2 | 50.0 | 0.55 | 1.00 | -8.8 |
| capped@1.5 | 1219 | 0.0 | 37.5 | 50.0 | 0.49 | 0.97 | -8.8 |
| capped@2.0 | 1219 | 0.0 | 37.5 | 50.0 | 0.51 | 0.97 | -8.8 |
| capped+strategy@1.5 | 1219 | 0.0 | 37.5 | 50.0 | 0.49 | 0.97 | -8.8 |

**Winner**: `capped@1.0` (PF 0.55 vs base 0.48, Δ=+0.07)

## Multi-Factor Momentum MR (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 49 | 0.0 | 100.0 | 0.0 | 0.00 | 0.83 | -8.9 |
| capped@1.0 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -8.9 |
| capped@1.5 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -8.9 |
| capped@2.0 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -8.9 |
| capped+strategy@1.5 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -8.9 |

**Winner**: `current` (PF 0.00 vs base 0.00, Δ=+0.00)

## RSI Mean Reversion (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 127 | 0.0 | 0.0 | 100.0 | 3720.30 | 3.21 | -3.8 |
| capped@1.0 | 127 | 0.0 | 0.0 | 33.3 | 4160.25 | 3.21 | -3.8 |
| capped@1.5 | 127 | 0.0 | 0.0 | 100.0 | 3720.30 | 3.21 | -3.8 |
| capped@2.0 | 127 | 0.0 | 0.0 | 100.0 | 3720.30 | 3.21 | -3.8 |
| capped+strategy@1.5 | 127 | 33.3 | 0.0 | 0.0 | 3620.00 | 3.21 | -3.8 |

**Winner**: `capped@1.0` (PF 4160.25 vs base 3720.30, Δ=+439.95)

## SMA Crossover (10/50) (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped@1.0 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped@1.5 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped@2.0 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped+strategy@1.5 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |

**Winner**: `current` (PF 0.00 vs base 0.00, Δ=+0.00)

## Stochastic Oscillator (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 718 | 0.0 | 28.6 | 50.0 | 1.26 | 1.21 | -6.6 |
| capped@1.0 | 718 | 0.0 | 25.0 | 50.0 | 1.41 | 1.22 | -6.6 |
| capped@1.5 | 718 | 0.0 | 28.6 | 50.0 | 1.26 | 1.21 | -6.6 |
| capped@2.0 | 718 | 0.0 | 28.6 | 50.0 | 1.26 | 1.21 | -6.6 |
| capped+strategy@1.5 | 718 | 0.0 | 25.0 | 50.0 | 1.20 | 1.18 | -6.6 |

**Winner**: `capped@1.0` (PF 1.41 vs base 1.26, Δ=+0.15)

## VWAP Trend (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 873 | 0.0 | 50.0 | 40.0 | 1.04 | 1.25 | -5.7 |
| capped@1.0 | 873 | 50.0 | 33.3 | 0.0 | 0.99 | 1.00 | -5.7 |
| capped@1.5 | 873 | 40.0 | 50.0 | 0.0 | 0.90 | 1.08 | -5.7 |
| capped@2.0 | 873 | 25.0 | 50.0 | 0.0 | 0.99 | 1.14 | -5.7 |
| capped+strategy@1.5 | 873 | 40.0 | 50.0 | 0.0 | 0.90 | 1.08 | -5.7 |

**Winner**: `current` (PF 1.04 vs base 1.04, Δ=+0.00)

## Z-Score Mean Reversion (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 610 | 0.0 | 0.0 | 66.7 | 2.28 | 1.48 | -6.1 |
| capped@1.0 | 610 | 0.0 | 0.0 | 50.0 | 2.28 | 1.48 | -6.1 |
| capped@1.5 | 610 | 0.0 | 0.0 | 66.7 | 2.28 | 1.48 | -6.1 |
| capped@2.0 | 610 | 0.0 | 0.0 | 66.7 | 2.28 | 1.48 | -6.1 |
| capped+strategy@1.5 | 610 | 0.0 | 0.0 | 50.0 | 2.09 | 1.46 | -6.1 |

**Winner**: `current` (PF 2.28 vs base 2.28, Δ=+0.00)
