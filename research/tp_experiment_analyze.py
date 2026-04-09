"""Artifact check and re-ranking for TP experiment results.

Reads the raw CSV from a TP experiment run and:

1. **Flags (strategy, mode) cells where tb_profit_factor is untrustworthy**
   due to denominator artifacts. A profit factor of `sum(wins)/abs(sum(losses))`
   explodes when the sample has very few actual losing trades, so a tiny
   denominator can push PF into the thousands even though the strategy's
   edge is ordinary.

2. **Produces side-by-side per-strategy rankings using both
   ``tb_profit_factor`` and ``tb_edge_ratio``.**  Edge ratio is avg MFE /
   avg MAE — it's averaged per-trade and bounded, so it's immune to the
   denominator-shrinkage pathology.

3. **Applies a stricter liquidity filter** that requires a minimum number
   of actual losing trades (default: 5), not just total trades.  This
   rejects rankings based on samples with too few losses to form a
   reliable PF ratio.

Usage
-----
    python research/tp_experiment_analyze.py \\
        research_output/tp_experiment_raw_2026-04-09.csv

    # Custom thresholds
    python research/tp_experiment_analyze.py <csv> \\
        --min-losses 10 --max-pf 20

    # Write markdown report to a file
    python research/tp_experiment_analyze.py <csv> \\
        --output research_output/tp_experiment_analysis.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Config ──────────────────────────────────────────────────────────────

# Default thresholds — override via CLI.
DEFAULT_MIN_LOSSES = 5      # Minimum estimated losing trades per cell
DEFAULT_MAX_PF = 50.0       # PF above this is treated as untrustworthy
DEFAULT_MIN_TRADES = 20     # Minimum total TB trades for a cell to qualify

# Tie-break when two modes have identical metric values — prefer simpler.
_MODE_SIMPLER_ORDER = {
    "current": 0,
    "capped@1.0": 1,
    "capped@1.5": 2,
    "capped@2.0": 3,
    "capped+strategy@1.5": 4,
}


# ── Helpers ─────────────────────────────────────────────────────────────


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _fnum(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _inum(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# ── Data loading ────────────────────────────────────────────────────────


def read_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Load the raw CSV into a list of typed dicts."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "mode_label": r["mode_label"],
                "strategy": r["strategy"],
                "family": r["family"],
                "ticker": r["ticker"],
                "timeframe": r["timeframe"],
                "tb_total_trades": _inum(r["tb_total_trades"]),
                "tb_win_rate": _fnum(r["tb_win_rate"]),
                "tb_sl_rate": _fnum(r["tb_sl_rate"]),
                "tb_timeout_rate": _fnum(r["tb_timeout_rate"]),
                "tb_profit_factor": _fnum(r["tb_profit_factor"]),
                "tb_edge_ratio": _fnum(r["tb_edge_ratio"]),
                "tb_avg_winner_pct": _fnum(r["tb_avg_winner_pct"]),
                "tb_avg_loser_pct": _fnum(r["tb_avg_loser_pct"]),
                "legacy_max_drawdown_pct": _fnum(r["legacy_max_drawdown_pct"]),
                "legacy_sharpe_ratio": _fnum(r["legacy_sharpe_ratio"]),
            })
    return rows


# ── Aggregation ─────────────────────────────────────────────────────────


def aggregate(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Collapse rows to one entry per (strategy, mode_label).

    Only aggregates cells that actually produced TB trades.  Returns
    medians across (ticker × window) cells plus sums for trade counts.
    """
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["tb_total_trades"] <= 0:
            continue
        buckets[(r["strategy"], r["mode_label"])].append(r)

    agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, bucket in buckets.items():
        strategy, mode = key
        family = bucket[0]["family"]
        total_trades = sum(r["tb_total_trades"] for r in bucket)
        # Weighted-by-trade estimate of losing trades — more robust than
        # `median(sl_rate) * total_trades` because cells vary in size.
        weighted_loss_trades = sum(
            r["tb_sl_rate"] * r["tb_total_trades"] for r in bucket
        )
        agg[key] = {
            "strategy": strategy,
            "family": family,
            "mode_label": mode,
            "n_cells": len(bucket),
            "tb_total_trades_sum": total_trades,
            "estimated_losing_trades": weighted_loss_trades,
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
            "tb_avg_winner_pct_median": _median(
                [r["tb_avg_winner_pct"] for r in bucket]
            ),
            "tb_avg_loser_pct_median": _median(
                [r["tb_avg_loser_pct"] for r in bucket]
            ),
            "legacy_max_drawdown_pct_median": _median(
                [r["legacy_max_drawdown_pct"] for r in bucket]
            ),
        }
    return agg


# ── Artifact detection ──────────────────────────────────────────────────


def flag_artifact(
    entry: Dict[str, Any],
    *,
    min_losses: float,
    max_pf: float,
) -> Tuple[bool, str]:
    """Return (is_artifact, reason).

    An entry is flagged as a PF artifact when:
    * fewer than `min_losses` estimated losing trades across the sample
      (denominator too small to trust the ratio), OR
    * profit factor exceeds `max_pf` (implausibly high — almost always
      caused by near-zero denominator)
    """
    est_losses = entry["estimated_losing_trades"]
    pf = entry["tb_profit_factor_median"]
    if est_losses < min_losses:
        return (True, f"only ~{est_losses:.1f} estimated losing trades")
    if pf > max_pf:
        return (True, f"PF={pf:.1f} implausibly high (denominator effect)")
    return (False, "")


# ── Winner selection ────────────────────────────────────────────────────


def pick_winners(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
    mode_labels: List[str],
    *,
    metric: str,
    min_trades: int,
    min_losses: float,
    max_pf: float,
    higher_is_better: bool = True,
    require_artifact_free: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Pick a winning mode per strategy by the given metric.

    Filters applied (all must pass for a mode to qualify):
    * tb_total_trades_sum >= min_trades
    * estimated_losing_trades >= min_losses (only when
      require_artifact_free=True)
    * legacy_max_drawdown_pct >= base_dd - 5pp (no catastrophic DD regression)
    * not flagged as artifact (only when require_artifact_free=True)
    """
    strategies = sorted({key[0] for key in agg.keys()})
    winners: Dict[str, Dict[str, Any]] = {}

    for strat in strategies:
        base_key = (strat, "current")
        if base_key not in agg:
            continue
        base = agg[base_key]
        base_dd = base["legacy_max_drawdown_pct_median"]
        base_metric = base[metric]

        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for mode_label in mode_labels:
            key = (strat, mode_label)
            if key not in agg:
                continue
            entry = agg[key]
            if entry["tb_total_trades_sum"] < min_trades:
                continue
            if require_artifact_free:
                if entry["estimated_losing_trades"] < min_losses:
                    continue
                is_art, _ = flag_artifact(
                    entry, min_losses=min_losses, max_pf=max_pf,
                )
                if is_art:
                    continue
            if entry["legacy_max_drawdown_pct_median"] < base_dd - 5.0:
                continue
            candidates.append((mode_label, entry))

        if not candidates:
            continue

        def _rank(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, int]:
            mode_label, entry = item
            value = entry[metric]
            ordered_value = -value if higher_is_better else value
            return (ordered_value, _MODE_SIMPLER_ORDER.get(mode_label, 99))

        candidates.sort(key=_rank)
        best_label, best_entry = candidates[0]
        winners[strat] = {
            "mode_label": best_label,
            "family": best_entry["family"],
            "metric_value": best_entry[metric],
            "base_metric_value": base_metric,
            "delta": best_entry[metric] - base_metric,
            "tb_total_trades": best_entry["tb_total_trades_sum"],
            "estimated_losing_trades": best_entry["estimated_losing_trades"],
            "base_trades": base["tb_total_trades_sum"],
        }
    return winners


# ── Reporting ───────────────────────────────────────────────────────────


def render_per_strategy_table(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
    mode_labels: List[str],
    pf_winners: Dict[str, Dict[str, Any]],
    edge_winners: Dict[str, Dict[str, Any]],
    *,
    min_losses: float,
    max_pf: float,
) -> List[str]:
    """Per-strategy section with PF & edge rankings side-by-side."""
    lines: List[str] = []
    strategies = sorted({key[0] for key in agg.keys()})

    lines.append("## Per-strategy analysis")
    lines.append("")
    lines.append(
        "Each row is one (strategy, mode) cell.  `est_losses` is the "
        "trade-count-weighted estimate of actual losing trades in the "
        f"sample.  Cells with `est_losses < {min_losses}` or `PF > {max_pf}` "
        "are flagged as denominator artifacts because the PF ratio "
        "becomes unreliable when the loss denominator is tiny."
    )
    lines.append("")

    for strat in strategies:
        base = agg.get((strat, "current"))
        if not base:
            continue
        family = base["family"]
        lines.append(f"### {strat} (_{family}_)")
        lines.append("")
        lines.append(
            "| Mode | Trades | Est Losses | Win% | SL% | PF | Edge | MaxDD% | Flag |"
        )
        lines.append(
            "|------|-------:|-----------:|-----:|----:|---:|-----:|-------:|:-----|"
        )
        for mode_label in mode_labels:
            entry = agg.get((strat, mode_label))
            if not entry:
                lines.append(
                    f"| {mode_label} | — | — | — | — | — | — | — | no data |"
                )
                continue
            is_art, reason = flag_artifact(
                entry, min_losses=min_losses, max_pf=max_pf,
            )
            flag = f"ARTIFACT: {reason}" if is_art else ""
            lines.append(
                f"| {mode_label} "
                f"| {entry['tb_total_trades_sum']} "
                f"| {entry['estimated_losing_trades']:.1f} "
                f"| {entry['tb_win_rate_median']*100:.1f} "
                f"| {entry['tb_sl_rate_median']*100:.1f} "
                f"| {entry['tb_profit_factor_median']:.2f} "
                f"| {entry['tb_edge_ratio_median']:.2f} "
                f"| {entry['legacy_max_drawdown_pct_median']:.1f} "
                f"| {flag} |"
            )

        pf_win = pf_winners.get(strat)
        edge_win = edge_winners.get(strat)
        lines.append("")
        lines.append(
            f"- **Best by PF (artifact-free)**: "
            f"{'`' + pf_win['mode_label'] + '`' if pf_win else '_(none qualified)_'}"
            + (
                f" — PF {pf_win['metric_value']:.2f} vs base "
                f"{pf_win['base_metric_value']:.2f} (Δ {pf_win['delta']:+.2f})"
                if pf_win else ""
            )
        )
        lines.append(
            f"- **Best by edge_ratio**: "
            f"{'`' + edge_win['mode_label'] + '`' if edge_win else '_(none qualified)_'}"
            + (
                f" — edge {edge_win['metric_value']:.2f} vs base "
                f"{edge_win['base_metric_value']:.2f} (Δ {edge_win['delta']:+.2f})"
                if edge_win else ""
            )
        )
        if pf_win and edge_win:
            if pf_win["mode_label"] == edge_win["mode_label"]:
                lines.append(
                    "- **Agreement**: ✅ both metrics agree"
                )
            else:
                lines.append(
                    "- **Agreement**: ⚠️ metrics disagree — UNCERTAIN winner"
                )
        lines.append("")

    return lines


def render_portfolio_summary(
    pf_winners: Dict[str, Dict[str, Any]],
    edge_winners: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Portfolio-level recommendation from each ranking."""
    lines: List[str] = []
    lines.append("## Portfolio-level recommendation")
    lines.append("")

    def _weighted_vote(
        winners: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float, float]]:
        vote: Dict[str, float] = defaultdict(float)
        for strat, w in winners.items():
            vote[w["mode_label"]] += float(w["base_trades"])
        total = sum(vote.values())
        ranked = sorted(
            vote.items(),
            key=lambda kv: (
                -kv[1],
                _MODE_SIMPLER_ORDER.get(kv[0], 99),
            ),
        )
        return [(mode, weight, (weight / total * 100) if total else 0.0)
                for mode, weight in ranked]

    pf_ranked = _weighted_vote(pf_winners)
    edge_ranked = _weighted_vote(edge_winners)

    lines.append("### Ranked by `tb_profit_factor` (with stricter filter)")
    lines.append("")
    lines.append("| Mode | Weight | % |")
    lines.append("|------|-------:|---:|")
    for mode, weight, pct in pf_ranked:
        lines.append(f"| {mode} | {weight:.0f} | {pct:.1f}% |")
    if pf_ranked:
        lines.append("")
        lines.append(f"**PF winner**: `{pf_ranked[0][0]}`")
    lines.append("")

    lines.append("### Ranked by `tb_edge_ratio` (artifact-resistant)")
    lines.append("")
    lines.append("| Mode | Weight | % |")
    lines.append("|------|-------:|---:|")
    for mode, weight, pct in edge_ranked:
        lines.append(f"| {mode} | {weight:.0f} | {pct:.1f}% |")
    if edge_ranked:
        lines.append("")
        lines.append(f"**Edge winner**: `{edge_ranked[0][0]}`")
    lines.append("")

    # Per-family breakdown — separate rankings for trend vs mean_reversion.
    def _per_family_vote(
        winners: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        by_family: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for strat, w in winners.items():
            by_family[w["family"]][w["mode_label"]] += float(w["base_trades"])
        out: Dict[str, List[Tuple[str, float, float]]] = {}
        for family, vote in by_family.items():
            total = sum(vote.values())
            ranked = sorted(
                vote.items(),
                key=lambda kv: (
                    -kv[1],
                    _MODE_SIMPLER_ORDER.get(kv[0], 99),
                ),
            )
            out[family] = [
                (mode, weight, (weight / total * 100) if total else 0.0)
                for mode, weight in ranked
            ]
        return out

    lines.append("### Per-family winners (by edge_ratio)")
    lines.append("")
    edge_family = _per_family_vote(edge_winners)
    for family in sorted(edge_family.keys()):
        lines.append(f"**{family}**")
        lines.append("")
        lines.append("| Mode | Weight | % |")
        lines.append("|------|-------:|---:|")
        for mode, weight, pct in edge_family[family]:
            lines.append(f"| {mode} | {weight:.0f} | {pct:.1f}% |")
        if edge_family[family]:
            lines.append("")
            lines.append(
                f"→ `{edge_family[family][0][0]}` "
                f"preferred for {family}"
            )
        lines.append("")

    return lines


def render_artifact_hall_of_fame(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    min_losses: float,
    max_pf: float,
    top_n: int = 10,
) -> List[str]:
    """Top N most egregious artifact cells — useful for debugging."""
    lines: List[str] = []
    lines.append("## Artifact hall of fame")
    lines.append("")
    lines.append(
        f"Top {top_n} (strategy, mode) cells with the most inflated PF "
        "values.  These are the ones pulling the portfolio-level PF "
        "ranking away from reality."
    )
    lines.append("")
    lines.append(
        "| Strategy | Mode | Trades | Est Losses | PF | Edge | Reason |"
    )
    lines.append(
        "|----------|------|-------:|-----------:|---:|-----:|--------|"
    )
    flagged: List[Tuple[float, Dict[str, Any]]] = []
    for entry in agg.values():
        is_art, reason = flag_artifact(
            entry, min_losses=min_losses, max_pf=max_pf,
        )
        if is_art:
            flagged.append((entry["tb_profit_factor_median"], {**entry, "_reason": reason}))
    flagged.sort(key=lambda x: -x[0])
    for _, entry in flagged[:top_n]:
        lines.append(
            f"| {entry['strategy']} "
            f"| {entry['mode_label']} "
            f"| {entry['tb_total_trades_sum']} "
            f"| {entry['estimated_losing_trades']:.1f} "
            f"| {entry['tb_profit_factor_median']:.2f} "
            f"| {entry['tb_edge_ratio_median']:.2f} "
            f"| {entry['_reason']} |"
        )
    if not flagged:
        lines.append("_(no artifacts detected)_")
    lines.append("")
    lines.append(f"Total artifact cells: **{len(flagged)}**")
    lines.append("")
    return lines


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    # Windows stdout Unicode safety.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path, help="Raw CSV from tp_experiment.py")
    parser.add_argument(
        "--min-losses",
        type=float,
        default=DEFAULT_MIN_LOSSES,
        help=(
            f"Minimum estimated losing trades per (strategy, mode) cell "
            f"(default: {DEFAULT_MIN_LOSSES})"
        ),
    )
    parser.add_argument(
        "--max-pf",
        type=float,
        default=DEFAULT_MAX_PF,
        help=(
            f"PF values above this are flagged as denominator artifacts "
            f"(default: {DEFAULT_MAX_PF})"
        ),
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=DEFAULT_MIN_TRADES,
        help=(
            f"Minimum total TB trades for a mode to qualify "
            f"(default: {DEFAULT_MIN_TRADES})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write the markdown report to this file "
            "(default: print to stdout only)"
        ),
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    rows = read_rows(args.csv)
    agg = aggregate(rows)

    # Discover mode labels that actually appear in the CSV.
    mode_labels_present = []
    seen = set()
    for r in rows:
        if r["mode_label"] not in seen:
            seen.add(r["mode_label"])
            mode_labels_present.append(r["mode_label"])
    # Sort by the simpler_order so the output is consistent.
    mode_labels_present.sort(
        key=lambda m: _MODE_SIMPLER_ORDER.get(m, 99)
    )

    pf_winners = pick_winners(
        agg, mode_labels_present,
        metric="tb_profit_factor_median",
        min_trades=args.min_trades,
        min_losses=args.min_losses,
        max_pf=args.max_pf,
        higher_is_better=True,
        require_artifact_free=True,
    )
    edge_winners = pick_winners(
        agg, mode_labels_present,
        metric="tb_edge_ratio_median",
        min_trades=args.min_trades,
        min_losses=args.min_losses,
        max_pf=args.max_pf,
        higher_is_better=True,
        require_artifact_free=True,
    )

    lines: List[str] = []
    lines.append("# TP Experiment — Artifact Analysis")
    lines.append("")
    lines.append(f"**Source CSV**: `{args.csv}`")
    lines.append(
        f"**Rows**: {len(rows)}  "
        f"**Aggregated cells**: {len(agg)}  "
        f"**Modes**: {', '.join(mode_labels_present)}"
    )
    lines.append("")
    lines.append(
        f"**Thresholds**: min_trades={args.min_trades}, "
        f"min_losses={args.min_losses}, max_pf={args.max_pf}"
    )
    lines.append("")

    lines.extend(render_artifact_hall_of_fame(
        agg, min_losses=args.min_losses, max_pf=args.max_pf,
    ))
    lines.extend(render_portfolio_summary(pf_winners, edge_winners))
    lines.extend(render_per_strategy_table(
        agg, mode_labels_present, pf_winners, edge_winners,
        min_losses=args.min_losses, max_pf=args.max_pf,
    ))

    text = "\n".join(lines)
    print(text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n(wrote markdown report to {args.output})", file=sys.stderr)


if __name__ == "__main__":
    main()
