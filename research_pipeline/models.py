"""Data models for the research pipeline.

All structured data flows through these frozen dataclasses.
Nothing here imports from trading_bot_bl — the pipeline is
fully independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


# ── Ingestion layer ────────────────────────────────────────────

@dataclass(frozen=True)
class NewsItem:
    """A single news article or headline with provenance."""

    headline: str
    source: str              # "yahoo", "finnhub", "newsapi"
    published_at: datetime
    ticker: str | None = None  # ticker it was fetched for, if any
    url: str = ""
    summary: str = ""


# ── Extraction layer ──────────────────────────────────────────

@dataclass(frozen=True)
class ExtractedSignal:
    """Structured fields extracted from a news item by the LLM."""

    headline: str
    published_at: datetime
    source: str

    # Extracted fields
    sentiment: str = "neutral"   # positive / negative / neutral
    tickers: tuple[str, ...] = ()
    themes: tuple[str, ...] = ()
    event_type: str = ""         # earnings, macro, commodity, regulation, etc.
    urgency: str = "low"         # low / medium / high
    summary: str = ""


# ── Theme layer ───────────────────────────────────────────────

@dataclass(frozen=True)
class ThemeCluster:
    """A group of related extracted signals forming a narrative."""

    theme: str                   # e.g. "oil price spike"
    mention_count: int = 0
    avg_sentiment: float = 0.0   # -1 to +1
    related_tickers: tuple[str, ...] = ()
    sample_headlines: tuple[str, ...] = ()
    first_seen: datetime | None = None
    last_seen: datetime | None = None


# ── Hypothesis layer ──────────────────────────────────────────

@dataclass(frozen=True)
class StrategyHypothesis:
    """A tradable hypothesis generated from a theme cluster."""

    hypothesis_id: str           # slug: "oil-spike-fertilizer-long"
    theme: str
    template: str                # which template generated this
    description: str             # human-readable summary
    causal_chain: str            # "oil spikes → input costs → fertilizer profits"

    # Tradable parameters
    target_tickers: tuple[str, ...] = ()
    direction: str = "long"      # long / short
    holding_days: tuple[int, ...] = (3, 5, 10)
    trigger_description: str = ""

    # Provenance
    source_headlines: tuple[str, ...] = ()
    generated_at: datetime = field(
        default_factory=datetime.now
    )
    confidence: str = "low"      # low / medium / high


# ── Run record ────────────────────────────────────────────────

@dataclass
class PipelineRun:
    """Tracks a single execution of the pipeline."""

    run_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    news_ingested: int = 0
    signals_extracted: int = 0
    themes_detected: int = 0
    hypotheses_generated: int = 0
    errors: list[str] = field(default_factory=list)
