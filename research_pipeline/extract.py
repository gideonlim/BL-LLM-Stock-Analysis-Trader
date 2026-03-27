"""Signal extraction — uses an LLM to parse structured fields from news.

Takes raw NewsItems and extracts sentiment, entities, themes,
event types, and urgency.  Batches headlines to reduce API calls.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from research_pipeline.config import ResearchConfig
from research_pipeline.models import ExtractedSignal, NewsItem

log = logging.getLogger(__name__)

# ── Extraction prompt ──────────────────────────────────────────

EXTRACT_PROMPT = """You are a financial research analyst. Analyze the following batch of news headlines and extract structured signals from each.

For EACH headline, extract:
1. **sentiment**: "positive", "negative", or "neutral" (from a market/investment perspective)
2. **tickers**: any stock tickers, ETFs, or commodities mentioned or strongly implied (e.g., if headline says "oil prices surge", include "USO", "XLE", "CL")
3. **themes**: 1-3 word tags for the macro/sector themes (e.g., "oil price spike", "interest rate hike", "supply chain disruption", "earnings beat", "tech layoffs")
4. **event_type**: one of: "commodity", "macro", "earnings", "regulation", "geopolitical", "sector_rotation", "supply_chain", "labor", "weather", "other"
5. **urgency**: "high" if this is breaking/market-moving, "medium" if notable, "low" if routine
6. **summary**: one sentence explaining the market implication

HEADLINES:
{headlines_block}

Respond with a JSON array. One object per headline, in the same order:
```json
[
  {{
    "headline_index": 0,
    "sentiment": "positive",
    "tickers": ["USO", "XLE"],
    "themes": ["oil price spike", "energy sector"],
    "event_type": "commodity",
    "urgency": "high",
    "summary": "Rising oil prices benefit energy producers."
  }},
  ...
]
```

IMPORTANT:
- Include implied tickers, not just explicitly named ones. If the headline mentions "fertilizer costs rising", include MOS, CF, NTR.
- Themes should be specific enough to cluster. "oil price spike" is good. "economy" is too vague.
- If a headline is generic filler (e.g., "Markets close mixed"), set urgency to "low" and sentiment to "neutral".
- Return ONLY the JSON array, no other text."""


# ── Batch size tuning ──────────────────────────────────────────
# Haiku handles ~20 headlines per call well.  Larger batches risk
# truncation; smaller batches waste API calls.
_BATCH_SIZE = 20


def _build_headlines_block(items: list[NewsItem]) -> str:
    """Format headlines for the prompt."""
    lines: list[str] = []
    for i, item in enumerate(items):
        ts = item.published_at.strftime("%Y-%m-%d %H:%M")
        src = item.source
        ticker_note = f" [{item.ticker}]" if item.ticker else ""
        lines.append(f"[{i}] ({ts}, {src}{ticker_note}) {item.headline}")
    return "\n".join(lines)


def _call_llm(
    prompt: str,
    config: ResearchConfig,
) -> str:
    """Call the Anthropic API and return the raw text response."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        message = client.messages.create(
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    except Exception as e:
        log.error(f"LLM call failed: {e}")
        return ""


def _parse_llm_response(
    raw: str,
    items: list[NewsItem],
) -> list[ExtractedSignal]:
    """Parse the LLM JSON response into ExtractedSignal objects."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [
            ln for ln in lines
            if not ln.strip().startswith("```")
        ]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse LLM response as JSON: {e}")
        return []

    if not isinstance(data, list):
        log.error("LLM response is not a JSON array")
        return []

    signals: list[ExtractedSignal] = []
    for entry in data:
        idx = entry.get("headline_index", -1)
        if idx < 0 or idx >= len(items):
            continue

        item = items[idx]
        signals.append(ExtractedSignal(
            headline=item.headline,
            published_at=item.published_at,
            source=item.source,
            sentiment=entry.get("sentiment", "neutral"),
            tickers=tuple(entry.get("tickers", ())),
            themes=tuple(entry.get("themes", ())),
            event_type=entry.get("event_type", "other"),
            urgency=entry.get("urgency", "low"),
            summary=entry.get("summary", ""),
        ))

    return signals


# ── Public interface ───────────────────────────────────────────

def extract_signals(
    items: list[NewsItem],
    config: ResearchConfig,
) -> list[ExtractedSignal]:
    """Extract structured signals from news items using the LLM.

    Batches items to reduce API calls.  Returns one
    ExtractedSignal per successfully parsed headline.
    """
    if not config.anthropic_api_key:
        log.warning("No ANTHROPIC_API_KEY — skipping extraction")
        return []

    if not items:
        return []

    all_signals: list[ExtractedSignal] = []

    # Process in batches
    for start in range(0, len(items), _BATCH_SIZE):
        batch = items[start : start + _BATCH_SIZE]
        headlines_block = _build_headlines_block(batch)
        prompt = EXTRACT_PROMPT.format(
            headlines_block=headlines_block,
        )

        log.info(
            f"Extracting signals from headlines "
            f"{start + 1}-{start + len(batch)} of {len(items)}"
        )

        raw_response = _call_llm(prompt, config)
        if not raw_response:
            continue

        batch_signals = _parse_llm_response(raw_response, batch)
        all_signals.extend(batch_signals)

        log.info(
            f"  Extracted {len(batch_signals)} signals "
            f"from {len(batch)} headlines"
        )

    log.info(
        f"Extraction complete: {len(all_signals)} signals "
        f"from {len(items)} headlines"
    )

    return all_signals
