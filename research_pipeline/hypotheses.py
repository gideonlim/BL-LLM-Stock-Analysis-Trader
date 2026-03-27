"""Hypothesis generation — converts theme clusters into tradable ideas.

Uses the LLM with structured strategy templates to produce
StrategyHypothesis objects from detected themes.  Templates
constrain the output to economically plausible patterns.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from research_pipeline.config import ResearchConfig
from research_pipeline.models import StrategyHypothesis, ThemeCluster

log = logging.getLogger(__name__)


# ── Strategy templates ─────────────────────────────────────────
# These are the patterns from your research report.  Each
# template tells the LLM what kind of trade to look for.

STRATEGY_TEMPLATES = {
    "upstream_beneficiary": {
        "description": (
            "When an input commodity spikes, producers or "
            "extractors of that commodity benefit from higher "
            "revenue.  Look for companies whose revenue is "
            "directly tied to the commodity price."
        ),
        "example": (
            "Oil price spike → long oil producers (XOM, CVX, "
            "OXY).  Gold surge → long miners (NEM, GOLD, AEM)."
        ),
        "direction": "long",
    },
    "downstream_beneficiary": {
        "description": (
            "When a situation creates stress in one part of a "
            "supply chain, companies further downstream that "
            "can pass costs through or benefit from the "
            "disruption may see gains.  Requires a clear "
            "second-order causal chain."
        ),
        "example": (
            "Oil spike → fertilizer input costs rise → "
            "fertilizer producers (MOS, CF) benefit if they "
            "can pass costs to farmers.  Shipping disruption → "
            "domestic manufacturers benefit from reduced "
            "import competition."
        ),
        "direction": "long",
    },
    "downstream_loser": {
        "description": (
            "When input costs spike, companies that cannot "
            "pass costs through to customers get squeezed.  "
            "Look for margin compression in cost-sensitive "
            "industries."
        ),
        "example": (
            "Oil spike → airlines (DAL, UAL) face higher fuel "
            "costs.  Steel tariffs → auto manufacturers face "
            "margin compression."
        ),
        "direction": "short",
    },
    "substitute_winner": {
        "description": (
            "When one product or sector faces headwinds, "
            "substitutes or competitors may benefit.  Look for "
            "demand shifts between related products."
        ),
        "example": (
            "Natural gas spike → renewable energy (ENPH, SEDG) "
            "becomes relatively cheaper.  Chip export ban → "
            "domestic semiconductor equipment makers benefit."
        ),
        "direction": "long",
    },
    "narrative_momentum": {
        "description": (
            "When a theme is accelerating in media attention "
            "and the associated stocks haven't fully priced it "
            "in, there may be a continuation trade.  Requires "
            "evidence of delayed market reaction."
        ),
        "example": (
            "AI infrastructure buildout narrative accelerating "
            "→ power/utility stocks haven't fully re-rated "
            "yet.  Defence spending narrative → small-cap "
            "defence contractors still cheap."
        ),
        "direction": "long",
    },
    "mean_reversion": {
        "description": (
            "When crowd attention on a theme peaks (very high "
            "mention volume, extreme sentiment), the trade may "
            "be crowded and due for a reversal.  Look for "
            "exhaustion signals."
        ),
        "example": (
            "Meme stock frenzy peaks → implied vol extremely "
            "high → short-term mean reversion likely.  "
            "Panic selling on sector news → oversold bounce "
            "candidate."
        ),
        "direction": "contrarian",
    },
}


# ── LLM hypothesis prompt ─────────────────────────────────────

HYPOTHESIS_PROMPT = """You are a quantitative research analyst generating trading hypotheses from detected market themes.

DETECTED THEME:
- Theme: {theme}
- Mention count: {mention_count} articles in last {days_back} days
- Average sentiment: {avg_sentiment} (-1 = very negative, +1 = very positive)
- Related tickers already mentioned: {related_tickers}
- Sample headlines:
{sample_headlines}

STRATEGY TEMPLATES (use these as frameworks — pick the most applicable ones):
{templates_block}

TASK: Generate 1-3 tradable hypotheses from this theme. For each hypothesis:

1. Pick the most applicable strategy template
2. Identify the causal chain (A → B → C)
3. Name specific target tickers (use real US stock tickers)
4. Explain why this would work over a 3-10 trading day horizon
5. Rate your confidence: "high" if the causal chain is physically/economically obvious, "medium" if plausible but requires assumptions, "low" if speculative

IMPORTANT:
- Only propose hypotheses with a clear, specific causal mechanism. "People are talking about X so X stocks go up" is NOT a valid hypothesis.
- The causal chain must explain WHY the target stocks would move, not just that they're related to the theme.
- Prefer hypotheses where the market may not have fully priced in the second-order effect yet.
- For a {holding_days}-day holding period, focus on effects that take a few days to materialise (not intraday reactions).

Respond with ONLY a JSON array:
```json
[
  {{
    "template": "downstream_beneficiary",
    "description": "One-sentence summary of the trade idea",
    "causal_chain": "A → B → C",
    "target_tickers": ["MOS", "CF"],
    "direction": "long",
    "trigger_description": "When X exceeds Y threshold",
    "confidence": "high",
    "reasoning": "2-3 sentences explaining the economic mechanism"
  }}
]
```"""


def _format_templates_block() -> str:
    """Format strategy templates for the prompt."""
    lines: list[str] = []
    for name, tmpl in STRATEGY_TEMPLATES.items():
        lines.append(f"- **{name}**: {tmpl['description']}")
        lines.append(f"  Example: {tmpl['example']}")
    return "\n".join(lines)


def _format_sample_headlines(
    headlines: tuple[str, ...],
) -> str:
    """Format sample headlines as a numbered list."""
    return "\n".join(
        f"  {i + 1}. {h}" for i, h in enumerate(headlines)
    )


def _call_llm(prompt: str, config: ResearchConfig) -> str:
    """Call the Anthropic API."""
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


def _parse_hypotheses(
    raw: str,
    theme: ThemeCluster,
) -> list[StrategyHypothesis]:
    """Parse the LLM response into StrategyHypothesis objects."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [
            ln for ln in lines
            if not ln.strip().startswith("```")
        ]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse hypothesis JSON: {e}")
        return []

    if not isinstance(data, list):
        log.error("Hypothesis response is not a JSON array")
        return []

    hypotheses: list[StrategyHypothesis] = []
    for i, entry in enumerate(data):
        template = entry.get("template", "unknown")
        tickers = tuple(entry.get("target_tickers", ()))
        description = entry.get("description", "")

        # Build a slug ID
        theme_slug = theme.theme.replace(" ", "-")[:30]
        hyp_id = f"{theme_slug}-{template}-{i}"

        hypotheses.append(StrategyHypothesis(
            hypothesis_id=hyp_id,
            theme=theme.theme,
            template=template,
            description=description,
            causal_chain=entry.get("causal_chain", ""),
            target_tickers=tickers,
            direction=entry.get("direction", "long"),
            holding_days=(3, 5, 10),
            trigger_description=entry.get(
                "trigger_description", ""
            ),
            source_headlines=theme.sample_headlines,
            confidence=entry.get("confidence", "low"),
        ))

    return hypotheses


# ── Public interface ───────────────────────────────────────────

def generate_hypotheses(
    themes: list[ThemeCluster],
    config: ResearchConfig,
) -> list[StrategyHypothesis]:
    """Generate strategy hypotheses from detected themes.

    Each theme cluster is sent to the LLM with strategy
    templates.  Returns a flat list of all generated hypotheses.
    """
    if not config.anthropic_api_key:
        log.warning(
            "No ANTHROPIC_API_KEY — skipping hypothesis generation"
        )
        return []

    if not themes:
        return []

    templates_block = _format_templates_block()
    all_hypotheses: list[StrategyHypothesis] = []

    for theme in themes:
        log.info(
            f"Generating hypotheses for theme: '{theme.theme}' "
            f"({theme.mention_count} mentions)"
        )

        prompt = HYPOTHESIS_PROMPT.format(
            theme=theme.theme,
            mention_count=theme.mention_count,
            days_back=config.news_days_back,
            avg_sentiment=f"{theme.avg_sentiment:+.2f}",
            related_tickers=", ".join(theme.related_tickers),
            sample_headlines=_format_sample_headlines(
                theme.sample_headlines,
            ),
            templates_block=templates_block,
            holding_days="3-10",
        )

        raw = _call_llm(prompt, config)
        if not raw:
            continue

        hyps = _parse_hypotheses(raw, theme)
        all_hypotheses.extend(hyps)

        for h in hyps:
            log.info(
                f"  → [{h.confidence}] {h.template}: "
                f"{h.description} "
                f"({', '.join(h.target_tickers)})"
            )

    log.info(
        f"Hypothesis generation complete: "
        f"{len(all_hypotheses)} hypotheses "
        f"from {len(themes)} themes"
    )

    return all_hypotheses
