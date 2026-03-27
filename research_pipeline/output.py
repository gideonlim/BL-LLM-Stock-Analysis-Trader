"""Output writer — persists pipeline results to disk.

Writes to research_output/ by default (configurable).
Each run produces a timestamped JSON file with full provenance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from research_pipeline.config import ResearchConfig
from research_pipeline.models import (
    ExtractedSignal,
    PipelineRun,
    StrategyHypothesis,
    ThemeCluster,
)

log = logging.getLogger(__name__)


def _serialise_datetime(obj: object) -> str:
    """JSON serialiser for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Cannot serialise {type(obj)}")


def write_run_report(
    run: PipelineRun,
    signals: list[ExtractedSignal],
    themes: list[ThemeCluster],
    hypotheses: list[StrategyHypothesis],
    config: ResearchConfig,
) -> Path:
    """Write a full run report as a JSON file.

    Returns the path to the written file.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"research_run_{timestamp}.json"
    path = output_dir / filename

    report = {
        "run": {
            "run_id": run.run_id,
            "started_at": run.started_at.isoformat(),
            "news_ingested": run.news_ingested,
            "signals_extracted": run.signals_extracted,
            "themes_detected": run.themes_detected,
            "hypotheses_generated": run.hypotheses_generated,
            "errors": run.errors,
        },
        "themes": [
            {
                "theme": t.theme,
                "mention_count": t.mention_count,
                "avg_sentiment": t.avg_sentiment,
                "related_tickers": list(t.related_tickers),
                "sample_headlines": list(t.sample_headlines),
                "first_seen": t.first_seen,
                "last_seen": t.last_seen,
            }
            for t in themes
        ],
        "hypotheses": [
            {
                "hypothesis_id": h.hypothesis_id,
                "theme": h.theme,
                "template": h.template,
                "description": h.description,
                "causal_chain": h.causal_chain,
                "target_tickers": list(h.target_tickers),
                "direction": h.direction,
                "holding_days": list(h.holding_days),
                "trigger_description": h.trigger_description,
                "source_headlines": list(h.source_headlines),
                "generated_at": h.generated_at,
                "confidence": h.confidence,
            }
            for h in hypotheses
        ],
        "signals_summary": {
            "total": len(signals),
            "by_event_type": _count_by(
                signals, lambda s: s.event_type
            ),
            "by_sentiment": _count_by(
                signals, lambda s: s.sentiment
            ),
            "unique_tickers": sorted(set(
                t for s in signals for t in s.tickers
            )),
        },
    }

    path.write_text(
        json.dumps(report, indent=2, default=_serialise_datetime),
        encoding="utf-8",
    )

    log.info(f"Run report written to {path}")
    return path


def write_hypotheses_summary(
    hypotheses: list[StrategyHypothesis],
    config: ResearchConfig,
) -> Path | None:
    """Write a human-readable summary of hypotheses.

    This is the file you review to decide what to backtest.
    """
    if not hypotheses:
        return None

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = output_dir / f"hypotheses_{timestamp}.txt"

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("RESEARCH PIPELINE — STRATEGY HYPOTHESES")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)

    for i, h in enumerate(hypotheses, 1):
        lines.append("")
        lines.append(f"--- Hypothesis {i}: {h.hypothesis_id} ---")
        lines.append(f"  Theme:       {h.theme}")
        lines.append(f"  Template:    {h.template}")
        lines.append(f"  Direction:   {h.direction}")
        lines.append(f"  Confidence:  {h.confidence}")
        lines.append(f"  Tickers:     {', '.join(h.target_tickers)}")
        lines.append(f"  Holding:     {h.holding_days} days")
        lines.append(f"  Description: {h.description}")
        lines.append(f"  Causal chain: {h.causal_chain}")
        lines.append(f"  Trigger:     {h.trigger_description}")
        lines.append(f"  Headlines:")
        for hl in h.source_headlines[:3]:
            lines.append(f"    - {hl}")

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"Total: {len(hypotheses)} hypotheses.  "
        "Review and approve before backtesting."
    )
    lines.append("=" * 70)

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Hypotheses summary written to {path}")
    return path


def _count_by(
    items: list[ExtractedSignal],
    key_fn: object,
) -> dict[str, int]:
    """Count items by a key function."""
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts
