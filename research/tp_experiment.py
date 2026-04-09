"""TP calculation mode experiment driver.

Sweeps ``tp_mode`` across 5 variants on the same historical data
and emits a decision matrix comparing per-strategy TB metrics.

Modes swept:
    current             — existing SL × dynamic RR, no cap
    capped@1.0          — cap TP at 1.0σ expected max move
    capped@1.5          — cap TP at 1.5σ (moderate, default)
    capped@2.0          — cap TP at 2.0σ (loose, only catches outliers)
    capped+strategy@1.5 — capped + mean-reversion family clamped to 1.5 RR

Usage
-----
    python research/tp_experiment.py                  # default: top 100 US stocks
    python research/tp_experiment.py --tickers AAPL GOOG NFLX  # smoke test
    python research/tp_experiment.py --top-n 30       # smaller sweep

All other config comes from DEFAULT_CONFIG; the driver forces
``triple_barrier_enabled=True`` (required for TP hit rate
measurement) and ``skip_pead=True`` (for experiment speed).

Outputs (under research_output/):
    tp_experiment_raw_{date}.csv      — full row-level grid with metadata
    tp_experiment_matrix_{date}.md    — per-strategy decision table
    tp_experiment_summary_{date}.txt  — winner recommendation + top-line params
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is on sys.path so `quant_analysis_bot` imports
# work whether the script is invoked as `python research/tp_experiment.py`
# from the project root or from elsewhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from quant_analysis_bot.backtest import select_best_strategy
from quant_analysis_bot.config import load_config
from quant_analysis_bot.data import batch_fetch_data, enrich_dataframe
from quant_analysis_bot.regime import enrich_with_regime, fetch_regime_data
from quant_analysis_bot.tp_logic import classify_strategy_family
from quant_analysis_bot.universe import fetch_top_us_stocks

log = logging.getLogger("tp_experiment")


# ── Mode definitions ────────────────────────────────────────────────────


# Each mode is a (label, tp_mode, tp_cap_multiplier) tuple.  The label
# is what shows up in CSV rows and the markdown matrix.  cap_multiplier
# is ignored for "current" mode.
TP_MODES: List[Tuple[str, str, float]] = [
    ("current", "current", 1.5),
    ("capped@1.0", "capped", 1.0),
    ("capped@1.5", "capped", 1.5),
    ("capped@2.0", "capped", 2.0),
    ("capped+strategy@1.5", "capped+strategy", 1.5),
]


# ── CLI ─────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tickers",
        nargs="+",
        help=(
            "Ticker list for the experiment. If omitted, uses "
            "--top-n top US stocks by market cap."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top-cap stocks when --tickers is not given (default: 100).",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional JSON config path (overrides DEFAULT_CONFIG).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research_output",
        help="Directory for experiment outputs (default: research_output/).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[m[0] for m in TP_MODES],
        help="Subset of modes to run (default: all 5).",
    )
    return parser.parse_args()


# ── Data preparation ────────────────────────────────────────────────────


def _prepare_enriched_data(
    config: Dict[str, Any],
    tickers: List[str],
) -> Dict[str, pd.DataFrame]:
    """Fetch and enrich price data once for reuse across all modes."""
    cache_dir = config.get("data_cache_dir", "cache")
    lookback = int(config.get("lookback_days", 500))

    log.info("Fetching regime data (VIX / SPY / etc.)...")
    regime_df = fetch_regime_data(
        lookback_days=lookback,
        vix_fear_threshold=float(config.get("vix_fear_threshold", 25.0)),
        cache_dir=cache_dir,
    )

    log.info("Fetching price data for %d tickers...", len(tickers))
    price_cache = batch_fetch_data(tickers, lookback, cache_dir)

    enriched: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = price_cache.get(t)
        if df is None or len(df) < 100:
            log.debug("%s: insufficient data (%s rows)", t, len(df) if df is not None else 0)
            continue
        try:
            df = enrich_dataframe(df)
            df = enrich_with_regime(df, regime_df)
            enriched[t] = df
        except Exception as e:
            log.warning("%s: enrichment failed: %s", t, e)

    log.info(
        "Enriched %d/%d tickers successfully", len(enriched), len(tickers)
    )
    return enriched


# ── Per-mode backtest run ───────────────────────────────────────────────


def _run_mode(
    mode_label: str,
    tp_mode: str,
    cap_multiplier: float,
    base_config: Dict[str, Any],
    enriched: Dict[str, pd.DataFrame],
    experiment_run_id: str,
    experiment_timestamp: str,
) -> List[Dict[str, Any]]:
    """Run the backtest for one mode, return a list of row dicts."""
    cfg = dict(base_config)
    cfg["tp_mode"] = tp_mode
    cfg["tp_cap_multiplier"] = cap_multiplier
    cfg["triple_barrier_enabled"] = True
    cfg["skip_pead"] = True
    cfg.setdefault("run_cscv", False)

    rows: List[Dict[str, Any]] = []
    tickers = sorted(enriched.keys())

    start = time.time()
    log.info(
        "[%s] Running %d tickers with tp_mode=%s cap_mult=%.2f",
        mode_label, len(tickers), tp_mode, cap_multiplier,
    )

    for idx, ticker in enumerate(tickers, 1):
        df = enriched[ticker]
        try:
            (
                best_strat,
                best_result,
                per_window,
                composite_scores,
                _trade_logs,
                _tb_data,
            ) = select_best_strategy(df, ticker, cfg)
        except Exception as e:
            log.warning(
                "[%s] %s (%d/%d) failed: %s",
                mode_label, ticker, idx, len(tickers), e,
            )
            continue

        # Emit one row per (ticker, strategy, window) in this mode.
        for window_name, window_results in per_window.items():
            for strat, r, sc in window_results:
                family = classify_strategy_family(r.strategy_name)
                avg_hold = float(r.avg_holding_days or 0.0)
                if avg_hold > 0:
                    holding_source = "avg_holding_days"
                    holding_used = avg_hold
                else:
                    holding_source = "config_fallback"
                    holding_used = float(cfg.get("tp_cap_holding_days", 20.0))

                rows.append({
                    "experiment_run_id": experiment_run_id,
                    "experiment_timestamp": experiment_timestamp,
                    "mode_label": mode_label,
                    "tp_mode": tp_mode,
                    "tp_cap_multiplier": cap_multiplier,
                    "triple_barrier_enabled": True,
                    "tp_cap_holding_days_source": holding_source,
                    "cap_holding_days_used": round(holding_used, 2),
                    "ticker": ticker,
                    "strategy": r.strategy_name,
                    "family": family,
                    "timeframe": window_name,
                    "window_score": round(float(sc), 2),
                    "composite_score": round(
                        float(composite_scores.get(r.strategy_name, 0)), 2
                    ),
                    "tb_total_trades": int(r.tb_total_trades),
                    "tb_win_rate": round(float(r.tb_win_rate), 4),
                    "tb_sl_rate": round(float(r.tb_sl_rate), 4),
                    "tb_timeout_rate": round(float(r.tb_timeout_rate), 4),
                    "tb_profit_factor": round(float(r.tb_profit_factor), 3),
                    "tb_edge_ratio": round(float(r.tb_edge_ratio), 3),
                    "tb_avg_winner_pct": round(float(r.tb_avg_winner_pct), 3),
                    "tb_avg_loser_pct": round(float(r.tb_avg_loser_pct), 3),
                    "legacy_total_trades": int(r.total_trades),
                    "legacy_win_rate": round(float(r.win_rate), 4),
                    "legacy_profit_factor": round(float(r.profit_factor), 3),
                    "legacy_max_drawdown_pct": round(
                        float(r.max_drawdown_pct), 2
                    ),
                    "legacy_sharpe_ratio": round(float(r.sharpe_ratio), 3),
                    "avg_holding_days": round(avg_hold, 2),
                })

        if idx % 10 == 0 or idx == len(tickers):
            elapsed = time.time() - start
            rate = idx / max(elapsed, 0.01)
            log.info(
                "[%s] %d/%d tickers done (%.1fs, %.2f/s)",
                mode_label, idx, len(tickers), elapsed, rate,
            )

    elapsed = time.time() - start
    log.info("[%s] complete in %.1fs (%d rows)", mode_label, elapsed, len(rows))
    return rows


# ── Aggregation & output ────────────────────────────────────────────────


_ROW_COLUMNS = [
    "experiment_run_id",
    "experiment_timestamp",
    "mode_label",
    "tp_mode",
    "tp_cap_multiplier",
    "triple_barrier_enabled",
    "tp_cap_holding_days_source",
    "cap_holding_days_used",
    "ticker",
    "strategy",
    "family",
    "timeframe",
    "window_score",
    "composite_score",
    "tb_total_trades",
    "tb_win_rate",
    "tb_sl_rate",
    "tb_timeout_rate",
    "tb_profit_factor",
    "tb_edge_ratio",
    "tb_avg_winner_pct",
    "tb_avg_loser_pct",
    "legacy_total_trades",
    "legacy_win_rate",
    "legacy_profit_factor",
    "legacy_max_drawdown_pct",
    "legacy_sharpe_ratio",
    "avg_holding_days",
]


def _write_raw_csv(rows: List[Dict[str, Any]], out_dir: str, date_str: str) -> str:
    path = os.path.join(out_dir, f"tp_experiment_raw_{date_str}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Raw CSV written: %s (%d rows)", path, len(rows))
    return path


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _aggregate_by_strategy_mode(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Collapse rows to one entry per (strategy, mode_label).

    Takes medians across (ticker × window) cells for robustness
    and weights trades by sum.  Also computes
    ``estimated_losing_trades``, a trade-count-weighted estimate of
    actual losing trades — used by the winner picker's liquidity
    filter to reject samples with too few losses for a reliable
    profit factor (PF explodes on tiny denominators).
    """
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        # Only aggregate cells that actually produced TB trades so
        # zero-trade rows don't dilute the medians.
        if int(r["tb_total_trades"]) <= 0:
            continue
        key = (r["strategy"], r["mode_label"])
        buckets[key].append(r)

    agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, bucket in buckets.items():
        strat, mode = key
        family = bucket[0]["family"]
        total_trades = sum(int(r["tb_total_trades"]) for r in bucket)
        # Weighted-by-trade estimate of losing trades is more robust
        # than median(sl_rate) * total_trades because cells vary in
        # size, and it matches the actual number of losses feeding
        # the profit factor denominator.
        estimated_losing_trades = sum(
            float(r["tb_sl_rate"]) * int(r["tb_total_trades"])
            for r in bucket
        )
        agg[key] = {
            "strategy": strat,
            "family": family,
            "mode_label": mode,
            "n_cells": len(bucket),
            "tb_total_trades_sum": total_trades,
            "estimated_losing_trades": estimated_losing_trades,
            "tb_win_rate_median": _median([r["tb_win_rate"] for r in bucket]),
            "tb_sl_rate_median": _median([r["tb_sl_rate"] for r in bucket]),
            "tb_timeout_rate_median": _median(
                [r["tb_timeout_rate"] for r in bucket]
            ),
            "tb_profit_factor_median": _median(
                [r["tb_profit_factor"] for r in bucket]
            ),
            "tb_edge_ratio_median": _median(
                [r["tb_edge_ratio"] for r in bucket]
            ),
            "legacy_max_drawdown_pct_median": _median(
                [r["legacy_max_drawdown_pct"] for r in bucket]
            ),
            "legacy_sharpe_ratio_median": _median(
                [r["legacy_sharpe_ratio"] for r in bucket]
            ),
        }
    return agg


# Minimum estimated losing trades per (strategy, mode) cell for the
# liquidity filter.  PF = sum(wins) / abs(sum(losses)) explodes on
# tiny denominators, so we require enough losses for the ratio to be
# meaningful.  Also applied to edge_ratio ranking for consistency.
_MIN_LOSING_TRADES = 5


def _pick_winner_per_strategy(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
    mode_labels: List[str],
    *,
    metric: str = "tb_edge_ratio_median",
    min_losing_trades: float = _MIN_LOSING_TRADES,
) -> Dict[str, Dict[str, Any]]:
    """Apply the per-strategy decision rule.

    Winner: highest ``metric`` among modes satisfying
      (a) tb_total_trades_sum >= 20 (liquidity: total sample)
      (b) estimated_losing_trades >= min_losing_trades (liquidity:
          actual losses — rejects tiny-denominator PF artifacts)
      (c) max_drawdown >= base_dd - 5pp (no catastrophic DD regression)

    Default metric is ``tb_edge_ratio_median`` (avg MFE / avg MAE),
    which is artifact-resistant because it's averaged per-trade and
    bounded.  The earlier default ``tb_profit_factor_median`` was
    shown to produce false-positive winners on samples with zero
    losing trades (PF = sum(wins)/~0 = thousands).  See
    research/tp_experiment_analyze.py for the analysis.

    base_dd comes from the "current" mode for each strategy.
    """
    strategies = sorted({key[0] for key in agg.keys()})
    winners: Dict[str, Dict[str, Any]] = {}

    for strat in strategies:
        base_key = (strat, "current")
        if base_key not in agg:
            continue
        base = agg[base_key]
        base_dd = base["legacy_max_drawdown_pct_median"]
        base_trades = base["tb_total_trades_sum"]

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for mode_label in mode_labels:
            key = (strat, mode_label)
            if key not in agg:
                continue
            entry = agg[key]
            if entry["tb_total_trades_sum"] < 20:
                continue
            if entry["estimated_losing_trades"] < min_losing_trades:
                continue
            if entry["legacy_max_drawdown_pct_median"] < base_dd - 5.0:
                continue
            candidates.append((mode_label, entry))

        if not candidates:
            continue

        # Tie-break: prefer simpler modes (current > capped > capped+strategy)
        # and higher metric value.
        def _rank(
            item: Tuple[str, Dict[str, Any]],
        ) -> Tuple[float, int]:
            mode_label, entry = item
            simpler_order = {
                "current": 0,
                "capped@1.0": 1,
                "capped@1.5": 2,
                "capped@2.0": 3,
                "capped+strategy@1.5": 4,
            }
            value = entry[metric]
            return (-value, simpler_order.get(mode_label, 99))

        candidates.sort(key=_rank)
        best_label, best_entry = candidates[0]
        winners[strat] = {
            "mode_label": best_label,
            "family": best_entry["family"],
            "ranking_metric": metric,
            "metric_value": best_entry[metric],
            "base_metric_value": base[metric],
            "metric_delta": best_entry[metric] - base[metric],
            # Keep PF on the winner record even when it wasn't the
            # ranking metric — useful for the summary output.
            "tb_profit_factor": best_entry["tb_profit_factor_median"],
            "base_tb_profit_factor": base["tb_profit_factor_median"],
            "tb_total_trades": best_entry["tb_total_trades_sum"],
            "estimated_losing_trades": best_entry["estimated_losing_trades"],
            "base_trades": base_trades,
        }
    return winners


def _write_matrix_md(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
    winners: Dict[str, Dict[str, Any]],
    mode_labels: List[str],
    out_dir: str,
    date_str: str,
    meta: Dict[str, Any],
) -> str:
    path = os.path.join(out_dir, f"tp_experiment_matrix_{date_str}.md")
    strategies = sorted({key[0] for key in agg.keys()})

    lines: List[str] = []
    lines.append(f"# TP Mode Experiment — Decision Matrix ({date_str})")
    lines.append("")
    lines.append(
        f"**Run ID**: `{meta['experiment_run_id']}`  "
        f"**Timestamp**: {meta['experiment_timestamp']}"
    )
    lines.append("")
    lines.append(
        f"**Tickers**: {meta['n_tickers']}  "
        f"**Modes**: {len(mode_labels)}  "
        f"**Wall-clock**: {meta['wall_clock_sec']:.0f}s"
    )
    lines.append("")
    lines.append(
        "All medians are across (ticker × window) cells.  "
        "Only cells with TB trades > 0 are aggregated.  "
        "Winner rule: highest `tb_pf` among modes with "
        "≥20 total TB trades and `max_dd >= base_dd - 5pp`, "
        "tie-break prefers simpler modes."
    )
    lines.append("")

    # One table per strategy.
    for strat in strategies:
        base = agg.get((strat, "current"))
        if not base:
            continue
        family = base["family"]
        lines.append(f"## {strat} (_{family}_)")
        lines.append("")
        lines.append(
            "| Mode | Trades | Win% | SL% | Timeout% | PF | Edge | Legacy MaxDD% |"
        )
        lines.append(
            "|------|-------:|-----:|----:|---------:|---:|-----:|--------------:|"
        )
        for mode_label in mode_labels:
            entry = agg.get((strat, mode_label))
            if not entry:
                lines.append(f"| {mode_label} | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {mode_label} "
                f"| {entry['tb_total_trades_sum']} "
                f"| {entry['tb_win_rate_median']*100:.1f} "
                f"| {entry['tb_sl_rate_median']*100:.1f} "
                f"| {entry['tb_timeout_rate_median']*100:.1f} "
                f"| {entry['tb_profit_factor_median']:.2f} "
                f"| {entry['tb_edge_ratio_median']:.2f} "
                f"| {entry['legacy_max_drawdown_pct_median']:.1f} |"
            )
        winner = winners.get(strat)
        if winner:
            metric_name = winner["ranking_metric"].replace("_median", "")
            lines.append("")
            lines.append(
                f"**Winner**: `{winner['mode_label']}` "
                f"({metric_name} {winner['metric_value']:.2f} vs base "
                f"{winner['base_metric_value']:.2f}, "
                f"Δ={winner['metric_delta']:+.2f}; "
                f"PF {winner['tb_profit_factor']:.2f}, "
                f"losses≈{winner['estimated_losing_trades']:.0f})"
            )
        else:
            lines.append("")
            lines.append("**Winner**: _(no mode passed filters)_")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("Matrix MD written: %s", path)
    return path


def _write_summary(
    winners: Dict[str, Dict[str, Any]],
    out_dir: str,
    date_str: str,
    meta: Dict[str, Any],
    mode_labels: List[str],
) -> str:
    """Portfolio-level winner + rationale."""
    path = os.path.join(out_dir, f"tp_experiment_summary_{date_str}.txt")

    # Weighted majority vote: each strategy's vote counts by its
    # base-case TB trade count (so noisy strategies don't dominate).
    vote_weights: Dict[str, float] = defaultdict(float)
    for strat, winner in winners.items():
        vote_weights[winner["mode_label"]] += float(winner["base_trades"])

    total_weight = sum(vote_weights.values())

    # Tie-break preferring simpler modes.
    simpler_order = {
        "current": 0,
        "capped@1.0": 1,
        "capped@1.5": 2,
        "capped@2.0": 3,
        "capped+strategy@1.5": 4,
    }
    ranked = sorted(
        vote_weights.items(),
        key=lambda kv: (-kv[1], simpler_order.get(kv[0], 99)),
    )

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f"  TP MODE EXPERIMENT — SUMMARY ({date_str})")
    lines.append("=" * 70)
    lines.append(f"  Run ID:       {meta['experiment_run_id']}")
    lines.append(f"  Timestamp:    {meta['experiment_timestamp']}")
    lines.append(f"  Tickers:      {meta['n_tickers']}")
    lines.append(f"  Modes swept:  {', '.join(mode_labels)}")
    lines.append(f"  Wall-clock:   {meta['wall_clock_sec']:.0f}s")
    lines.append(f"  Total rows:   {meta['n_rows']}")
    lines.append("=" * 70)
    lines.append("")
    # Show the ranking metric used for the winner selection so
    # readers know how to interpret the deltas.
    if winners:
        any_winner = next(iter(winners.values()))
        ranking_metric = any_winner["ranking_metric"].replace("_median", "")
    else:
        ranking_metric = "tb_edge_ratio"
    lines.append(f"  Ranking metric: {ranking_metric}")
    lines.append("")
    lines.append("  Per-strategy winners:")
    if not winners:
        lines.append(
            "    (no strategy had a viable winner — check filters: "
            "all candidates fail min_trades>=20, min_losses>=5, "
            "or max_dd regression guard)"
        )
    else:
        for strat in sorted(winners.keys()):
            w = winners[strat]
            lines.append(
                f"    {strat:<32} -> {w['mode_label']:<22} "
                f"edge {w['metric_value']:.2f} "
                f"(base {w['base_metric_value']:.2f}, "
                f"Δ {w['metric_delta']:+.2f}; "
                f"losses≈{w['estimated_losing_trades']:.0f})"
            )
    lines.append("")
    lines.append("  Weighted mode votes (weighted by base TB trades):")
    for mode_label, weight in ranked:
        pct = (weight / total_weight * 100) if total_weight else 0.0
        lines.append(f"    {mode_label:<22} {weight:>8.0f}  ({pct:5.1f}%)")
    lines.append("")
    if ranked:
        recommended = ranked[0][0]
        n_wins = sum(
            1 for w in winners.values() if w["mode_label"] == recommended
        )
        lines.append("-" * 70)
        lines.append(f"  RECOMMENDED TP MODE: {recommended}")
        lines.append(
            f"  Per-strategy wins:   {n_wins}/{len(winners)} "
            f"strategies prefer {recommended}"
        )
        # Compute median delta (in the ranking metric) for the
        # recommended mode, so readers know how big the lift is.
        deltas = [
            w["metric_delta"] for w in winners.values()
            if w["mode_label"] == recommended
        ]
        median_delta = _median(deltas) if deltas else 0.0
        lines.append(
            f"  Rationale:           median {ranking_metric} delta "
            f"vs current = {median_delta:+.2f}"
        )
        lines.append("-" * 70)
    lines.append("")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    log.info("Summary TXT written: %s", path)
    print(text)
    return path


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    # On Windows, the default cp1252 stdout can't print Unicode in
    # strategy names or the summary block.  Reconfigure to UTF-8
    # with error replacement so the print never crashes the run.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    base_config = load_config(args.config)

    # Select modes
    if args.modes:
        modes = [m for m in TP_MODES if m[0] in args.modes]
    else:
        modes = list(TP_MODES)

    # Select tickers
    if args.tickers:
        tickers = args.tickers
        log.info("Using --tickers override: %d tickers", len(tickers))
    else:
        log.info("Fetching top %d US stocks by market cap...", args.top_n)
        tickers = fetch_top_us_stocks(
            n=args.top_n,
            cache_dir=base_config.get("data_cache_dir", "cache"),
        )
        if not tickers:
            log.error("Failed to fetch stock universe. Aborting.")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare data once
    enriched = _prepare_enriched_data(base_config, tickers)
    if not enriched:
        log.error("No enriched data available. Aborting.")
        sys.exit(1)

    # Run all modes sequentially, collect rows
    experiment_run_id = str(uuid.uuid4())[:8]
    experiment_timestamp = datetime.now().isoformat(timespec="seconds")
    date_str = datetime.now().strftime("%Y-%m-%d")

    all_rows: List[Dict[str, Any]] = []
    wall_clock_start = time.time()

    for mode_label, tp_mode, cap_mult in modes:
        rows = _run_mode(
            mode_label=mode_label,
            tp_mode=tp_mode,
            cap_multiplier=cap_mult,
            base_config=base_config,
            enriched=enriched,
            experiment_run_id=experiment_run_id,
            experiment_timestamp=experiment_timestamp,
        )
        all_rows.extend(rows)

    wall_clock_sec = time.time() - wall_clock_start

    meta = {
        "experiment_run_id": experiment_run_id,
        "experiment_timestamp": experiment_timestamp,
        "n_tickers": len(enriched),
        "n_rows": len(all_rows),
        "wall_clock_sec": wall_clock_sec,
    }

    if not all_rows:
        log.error("No rows produced. Aborting output.")
        sys.exit(1)

    # Outputs
    _write_raw_csv(all_rows, args.output_dir, date_str)

    mode_labels = [m[0] for m in modes]
    agg = _aggregate_by_strategy_mode(all_rows)
    winners = _pick_winner_per_strategy(agg, mode_labels)
    _write_matrix_md(
        agg, winners, mode_labels, args.output_dir, date_str, meta,
    )
    _write_summary(winners, args.output_dir, date_str, meta, mode_labels)


if __name__ == "__main__":
    main()
