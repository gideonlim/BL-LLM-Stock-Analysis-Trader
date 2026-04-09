# TP Mode Experiment — Decision Matrix (2026-04-09)

**Run ID**: `6336161c`  **Timestamp**: 2026-04-09T22:42:41

**Tickers**: 113  **Modes**: 5  **Wall-clock**: 89s

All medians are across (ticker × window) cells.  Only cells with TB trades > 0 are aggregated.  Winner rule: highest `tb_pf` among modes with ≥20 total TB trades and `max_dd >= base_dd - 5pp`, tie-break prefers simpler modes.

## 52-Week High Momentum (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 1953 | 0.0 | 47.1 | 50.0 | 0.35 | 0.77 | -10.7 |
| capped@1.0 | 1953 | 0.0 | 47.1 | 50.0 | 0.38 | 0.77 | -10.7 |
| capped@1.5 | 1953 | 0.0 | 47.1 | 50.0 | 0.36 | 0.77 | -10.7 |
| capped@2.0 | 1953 | 0.0 | 47.1 | 50.0 | 0.35 | 0.77 | -10.7 |
| capped+strategy@1.5 | 1953 | 0.0 | 47.1 | 50.0 | 0.36 | 0.77 | -10.7 |

**Winner**: `current` (tb_edge_ratio 0.77 vs base 0.77, Δ=+0.00; PF 0.35, losses≈755)

## ADX Trend Following (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.79 | -7.5 |
| capped@1.0 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.69 | -7.5 |
| capped@1.5 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.72 | -7.5 |
| capped@2.0 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.79 | -7.5 |
| capped+strategy@1.5 | 227 | 0.0 | 50.0 | 0.0 | 0.00 | 0.72 | -7.5 |

**Winner**: `current` (tb_edge_ratio 0.79 vs base 0.79, Δ=+0.00; PF 0.00, losses≈119)

## Bollinger Band Mean Reversion (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 229 | 0.0 | 0.0 | 100.0 | 1638.60 | 2.65 | -5.7 |
| capped@1.0 | 229 | 0.0 | 0.0 | 0.0 | 3680.00 | 2.81 | -5.7 |
| capped@1.5 | 229 | 0.0 | 0.0 | 50.0 | 3680.00 | 2.81 | -5.7 |
| capped@2.0 | 229 | 0.0 | 0.0 | 100.0 | 1638.60 | 2.65 | -5.7 |
| capped+strategy@1.5 | 229 | 0.0 | 0.0 | 0.0 | 3185.00 | 2.81 | -5.7 |

**Winner**: `capped@1.0` (tb_edge_ratio 2.81 vs base 2.65, Δ=+0.15; PF 3680.00, losses≈59)

## Composite Multi-Indicator (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |
| capped@1.0 | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |
| capped@1.5 | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |
| capped@2.0 | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |
| capped+strategy@1.5 | 156 | 0.0 | 0.0 | 0.0 | 1.74 | 1.88 | -9.1 |

**Winner**: `current` (tb_edge_ratio 1.88 vs base 1.88, Δ=+0.00; PF 1.74, losses≈47)

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

**Winner**: `current` (tb_edge_ratio 0.32 vs base 0.32, Δ=+0.00; PF 0.00, losses≈48)

## MACD Crossover (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 290 | 0.0 | 0.0 | 50.0 | 1.87 | 1.20 | -7.8 |
| capped@1.0 | 290 | 0.0 | 0.0 | 0.0 | 1.91 | 1.20 | -7.8 |
| capped@1.5 | 290 | 0.0 | 0.0 | 33.3 | 1.91 | 1.20 | -7.8 |
| capped@2.0 | 290 | 0.0 | 0.0 | 50.0 | 1.87 | 1.20 | -7.8 |
| capped+strategy@1.5 | 290 | 0.0 | 0.0 | 33.3 | 1.91 | 1.20 | -7.8 |

**Winner**: `current` (tb_edge_ratio 1.20 vs base 1.20, Δ=+0.00; PF 1.87, losses≈117)

## Momentum (Rate of Change) (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 1219 | 0.0 | 37.5 | 50.0 | 0.48 | 0.97 | -8.8 |
| capped@1.0 | 1219 | 0.0 | 37.2 | 50.0 | 0.56 | 1.02 | -8.8 |
| capped@1.5 | 1219 | 0.0 | 37.5 | 50.0 | 0.48 | 0.97 | -8.8 |
| capped@2.0 | 1219 | 0.0 | 37.5 | 50.0 | 0.48 | 0.97 | -8.8 |
| capped+strategy@1.5 | 1219 | 0.0 | 37.5 | 50.0 | 0.48 | 0.97 | -8.8 |

**Winner**: `capped@1.0` (tb_edge_ratio 1.02 vs base 0.97, Δ=+0.05; PF 0.56, losses≈421)

## Multi-Factor Momentum MR (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 49 | 0.0 | 100.0 | 0.0 | 0.00 | 0.83 | -9.0 |
| capped@1.0 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -9.0 |
| capped@1.5 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -9.0 |
| capped@2.0 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -9.0 |
| capped+strategy@1.5 | 49 | 0.0 | 66.7 | 0.0 | 0.00 | 0.83 | -9.0 |

**Winner**: `current` (tb_edge_ratio 0.83 vs base 0.83, Δ=+0.00; PF 0.00, losses≈26)

## RSI Mean Reversion (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 127 | 0.0 | 0.0 | 100.0 | 3720.30 | 3.21 | -3.8 |
| capped@1.0 | 127 | 33.3 | 0.0 | 0.0 | 3447.00 | 3.21 | -3.8 |
| capped@1.5 | 127 | 0.0 | 0.0 | 50.0 | 3720.30 | 3.21 | -3.8 |
| capped@2.0 | 127 | 0.0 | 0.0 | 100.0 | 3720.30 | 3.21 | -3.8 |
| capped+strategy@1.5 | 127 | 33.3 | 0.0 | 0.0 | 3620.00 | 3.21 | -3.8 |

**Winner**: `current` (tb_edge_ratio 3.21 vs base 3.21, Δ=+0.00; PF 3720.30, losses≈31)

## SMA Crossover (10/50) (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped@1.0 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped@1.5 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped@2.0 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |
| capped+strategy@1.5 | 46 | 0.0 | 75.0 | 0.0 | 0.00 | 0.69 | -10.8 |

**Winner**: `current` (tb_edge_ratio 0.69 vs base 0.69, Δ=+0.00; PF 0.00, losses≈24)

## Stochastic Oscillator (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 718 | 0.0 | 28.6 | 50.0 | 1.26 | 1.21 | -6.6 |
| capped@1.0 | 718 | 0.0 | 25.0 | 50.0 | 1.41 | 1.22 | -6.6 |
| capped@1.5 | 718 | 0.0 | 28.6 | 50.0 | 1.26 | 1.21 | -6.6 |
| capped@2.0 | 718 | 0.0 | 28.6 | 50.0 | 1.26 | 1.21 | -6.6 |
| capped+strategy@1.5 | 718 | 0.0 | 25.0 | 45.0 | 1.20 | 1.18 | -6.6 |

**Winner**: `capped@1.0` (tb_edge_ratio 1.22 vs base 1.21, Δ=+0.01; PF 1.41, losses≈244)

## VWAP Trend (_trend_following_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 873 | 0.0 | 50.0 | 40.0 | 1.04 | 1.25 | -5.7 |
| capped@1.0 | 873 | 50.0 | 33.3 | 0.0 | 1.01 | 1.02 | -5.7 |
| capped@1.5 | 873 | 40.0 | 50.0 | 0.0 | 0.88 | 1.08 | -5.7 |
| capped@2.0 | 873 | 25.0 | 50.0 | 0.0 | 0.98 | 1.14 | -5.7 |
| capped+strategy@1.5 | 873 | 40.0 | 50.0 | 0.0 | 0.88 | 1.08 | -5.7 |

**Winner**: `current` (tb_edge_ratio 1.25 vs base 1.25, Δ=+0.00; PF 1.04, losses≈414)

## Z-Score Mean Reversion (_mean_reversion_)

| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |
|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|
| current | 610 | 0.0 | 0.0 | 66.7 | 2.28 | 1.48 | -6.1 |
| capped@1.0 | 610 | 0.0 | 0.0 | 50.0 | 2.26 | 1.48 | -6.1 |
| capped@1.5 | 610 | 0.0 | 0.0 | 66.7 | 2.28 | 1.48 | -6.1 |
| capped@2.0 | 610 | 0.0 | 0.0 | 66.7 | 2.28 | 1.48 | -6.1 |
| capped+strategy@1.5 | 610 | 0.0 | 0.0 | 50.0 | 2.09 | 1.46 | -6.1 |

**Winner**: `current` (tb_edge_ratio 1.48 vs base 1.48, Δ=+0.00; PF 2.28, losses≈173)
