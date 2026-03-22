"""
LLM-enhanced view generation for the Black-Litterman model.

Based on the ICLR 2025 paper "Integrating LLM-Generated Views
into Mean-Variance Optimization Using the Black-Litterman Model"
by Young & Bin.

Key insight: prompt the LLM N times with temperature > 0,
then use the mean prediction as the view (Q) and the variance
of predictions as the uncertainty (Ω). Higher variance → less
confident → view has less influence on the posterior.

Supports:
    - Anthropic Claude (preferred)
    - OpenAI GPT models (fallback)
    - Dry-run mode (returns empty views)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from trading_bot_bl.black_litterman import BLView

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SAMPLES = 10       # number of repeated predictions
DEFAULT_TEMPERATURE = 0.7  # needs variation for uncertainty
DEFAULT_MAX_TOKENS = 512
DEFAULT_MAX_TICKERS = 10   # only query LLM for top-N highest confidence signals


@dataclass
class LLMConfig:
    """Configuration for LLM view generation."""

    enabled: bool = False
    provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = DEFAULT_MODEL
    num_samples: int = DEFAULT_SAMPLES
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    llm_weight: float = 0.3  # weight when blending with quant
    max_tickers: int = DEFAULT_MAX_TICKERS  # only LLM the top-N signals

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Build from environment variables."""
        return cls(
            enabled=os.getenv(
                "LLM_VIEWS_ENABLED", "false"
            ).lower() in ("true", "1", "yes"),
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            num_samples=int(
                os.getenv("LLM_NUM_SAMPLES", str(DEFAULT_SAMPLES))
            ),
            temperature=float(
                os.getenv("LLM_TEMPERATURE", str(DEFAULT_TEMPERATURE))
            ),
            llm_weight=float(
                os.getenv("LLM_WEIGHT", "0.3")
            ),
            max_tickers=int(
                os.getenv("LLM_MAX_TICKERS", str(DEFAULT_MAX_TICKERS))
            ),
        )


# ── Prompt Construction ──────────────────────────────────────────

VIEW_PROMPT_TEMPLATE = """You are a quantitative analyst estimating stock returns.

Given the following data for {ticker}:

PRICE & TECHNICALS:
- Current price: ${current_price:.2f}
- 20-day return: {return_20d:+.1%}
- RSI (14): {rsi:.1f}
- Trend: {trend} (SMA50 vs SMA200)
- Volatility regime: {volatility}

BACKTEST SIGNAL:
- Best strategy: {strategy}
- Signal: BUY (confidence {confidence}/6, composite score {composite:.1f})
- Backtest Sharpe ratio: {sharpe:.2f}
- Win rate: {win_rate:.1%}
- Total backtested trades: {total_trades}

STOP LOSS / TAKE PROFIT:
- Stop loss: ${stop_loss:.2f} ({sl_pct:.1%} below entry)
- Take profit: ${take_profit:.2f} ({tp_pct:.1%} above entry)
- Suggested position size: {position_size:.1%} of portfolio

{news_section}

TASK: Estimate the expected return for {ticker} over the next {holding_period} trading days.

Consider:
1. Whether the technical setup supports the signal direction
2. Risk/reward ratio implied by the SL/TP levels
3. Quality of backtest evidence (Sharpe, win rate, trade count)
4. Current market regime and volatility
{news_consideration}

Respond with ONLY a JSON object (no other text):
{{"expected_return_pct": <number between -20 and 40>, "confidence": "<high/medium/low>", "reasoning": "<one sentence>"}}"""


def build_prompt(
    signal,
    news_headlines: list[str] | None = None,
    holding_period: int = 10,
) -> str:
    """
    Build the LLM prompt for a single stock view.

    Args:
        signal: Signal object from the quant bot.
        news_headlines: Optional recent news for context.
        holding_period: Expected holding period in trading days.

    Returns:
        Formatted prompt string.
    """
    # Calculate SL/TP percentages
    price = signal.current_price
    sl_pct = (
        abs(price - signal.stop_loss_price) / price
        if price > 0 else 0
    )
    tp_pct = (
        abs(signal.take_profit_price - price) / price
        if price > 0 else 0
    )

    # Estimate 20-day return from available data
    # (we don't have this directly, so use a placeholder)
    return_20d = 0.0  # will be enriched if data available

    # News section
    news_section = ""
    news_consideration = ""
    if news_headlines:
        headlines_text = "\n".join(
            f"  - {h}" for h in news_headlines[:5]
        )
        news_section = (
            f"RECENT NEWS:\n{headlines_text}\n"
        )
        news_consideration = (
            "5. Any relevant news that could impact "
            "near-term performance"
        )

    return VIEW_PROMPT_TEMPLATE.format(
        ticker=signal.ticker,
        current_price=price,
        return_20d=return_20d,
        rsi=0.0,  # enriched by caller if available
        trend="UNKNOWN",
        volatility="UNKNOWN",
        strategy=signal.strategy,
        confidence=signal.confidence_score,
        composite=signal.composite_score,
        sharpe=signal.sharpe,
        win_rate=signal.win_rate / 100,  # stored as 0-100, format needs 0-1
        total_trades=signal.total_trades,
        stop_loss=signal.stop_loss_price,
        take_profit=signal.take_profit_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        position_size=signal.suggested_position_size_pct / 100,
        news_section=news_section,
        news_consideration=news_consideration,
        holding_period=holding_period,
    )


def build_enriched_prompt(
    signal,
    market_data: dict | None = None,
    news_headlines: list[str] | None = None,
    holding_period: int = 10,
) -> str:
    """
    Build prompt with enriched market data (RSI, trend, vol).

    Args:
        signal: Signal object.
        market_data: Optional dict with keys like 'rsi', 'trend',
            'volatility', 'return_20d'.
        news_headlines: Recent news for the ticker.
        holding_period: Expected holding days.

    Returns:
        Formatted prompt string.
    """
    price = signal.current_price
    sl_pct = (
        abs(price - signal.stop_loss_price) / price
        if price > 0 else 0
    )
    tp_pct = (
        abs(signal.take_profit_price - price) / price
        if price > 0 else 0
    )

    md = market_data or {}

    news_section = ""
    news_consideration = ""
    if news_headlines:
        headlines_text = "\n".join(
            f"  - {h}" for h in news_headlines[:5]
        )
        news_section = f"RECENT NEWS:\n{headlines_text}\n"
        news_consideration = (
            "5. Any relevant news that could impact "
            "near-term performance"
        )

    return VIEW_PROMPT_TEMPLATE.format(
        ticker=signal.ticker,
        current_price=price,
        return_20d=md.get("return_20d", 0.0),
        rsi=md.get("rsi", 0.0),
        trend=md.get("trend", "UNKNOWN"),
        volatility=md.get("volatility", "UNKNOWN"),
        strategy=signal.strategy,
        confidence=signal.confidence_score,
        composite=signal.composite_score,
        sharpe=signal.sharpe,
        win_rate=signal.win_rate / 100,  # stored as 0-100, format needs 0-1
        total_trades=signal.total_trades,
        stop_loss=signal.stop_loss_price,
        take_profit=signal.take_profit_price,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        position_size=signal.suggested_position_size_pct / 100,
        news_section=news_section,
        news_consideration=news_consideration,
        holding_period=holding_period,
    )


# ── LLM API Calls ────────────────────────────────────────────────

def _call_anthropic(
    prompt: str,
    config: LLMConfig,
) -> str | None:
    """Call Anthropic Claude API and return the response text."""
    try:
        import anthropic
    except ImportError:
        log.error(
            "anthropic package not installed. "
            "Run: pip install anthropic"
        )
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error(
            "ANTHROPIC_API_KEY not set. "
            "Cannot generate LLM views."
        )
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        log.warning(f"Anthropic API error: {e}")
        return None


def _call_openai(
    prompt: str,
    config: LLMConfig,
) -> str | None:
    """Call OpenAI API and return the response text."""
    try:
        import openai
    except ImportError:
        log.error(
            "openai package not installed. "
            "Run: pip install openai"
        )
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY not set.")
        return None

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning(f"OpenAI API error: {e}")
        return None


def _call_llm(prompt: str, config: LLMConfig) -> str | None:
    """Route to the configured LLM provider."""
    if config.provider == "anthropic":
        return _call_anthropic(prompt, config)
    elif config.provider == "openai":
        return _call_openai(prompt, config)
    else:
        log.error(f"Unknown LLM provider: {config.provider}")
        return None


def _parse_response(text: str) -> dict | None:
    """
    Parse LLM JSON response, handling common formatting issues.
    """
    if not text:
        return None

    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        data = json.loads(text)
        if "expected_return_pct" in data:
            return data
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

    log.debug(f"Could not parse LLM response: {text[:200]}")
    return None


# ── Repeated Sampling (ICLR 2025 Method) ────────────────────────

def generate_view_for_ticker(
    signal,
    config: LLMConfig,
    market_data: dict | None = None,
    news_headlines: list[str] | None = None,
    holding_period: int = 10,
) -> BLView | None:
    """
    Generate a single BLView for a ticker using repeated
    LLM sampling.

    Prompts the LLM N times and uses:
    - Mean of predictions → expected return (Q)
    - Variance of predictions → uncertainty (Ω)

    This is the key insight from the ICLR 2025 paper:
    higher prediction variance = less confident view.

    Args:
        signal: Signal object for the ticker.
        config: LLM configuration.
        market_data: Enrichment data (RSI, trend, etc.).
        news_headlines: Recent headlines for context.
        holding_period: Expected holding period.

    Returns:
        BLView with expected return and confidence, or None.
    """
    prompt = build_enriched_prompt(
        signal, market_data, news_headlines, holding_period
    )

    predictions: list[float] = []
    confidences: list[str] = []
    reasonings: list[str] = []

    log.info(
        f"  LLM: sampling {config.num_samples}x for "
        f"{signal.ticker}..."
    )

    for i in range(config.num_samples):
        response_text = _call_llm(prompt, config)
        parsed = _parse_response(response_text)

        if parsed and "expected_return_pct" in parsed:
            ret = float(parsed["expected_return_pct"])
            # Sanity check: clip to reasonable range
            ret = np.clip(ret, -30.0, 50.0)
            predictions.append(ret / 100)  # convert to decimal
            confidences.append(
                parsed.get("confidence", "medium")
            )
            reasonings.append(
                parsed.get("reasoning", "")
            )

    if len(predictions) < 3:
        log.warning(
            f"  LLM: only {len(predictions)} valid predictions "
            f"for {signal.ticker} — skipping"
        )
        return None

    # ── Aggregate predictions ─────────────────────────────────
    mean_return = float(np.mean(predictions))
    std_return = float(np.std(predictions))

    # Map variance to confidence (0-1)
    # Low std → high confidence, high std → low confidence
    # Calibrated so std=0.02 (2%) → confidence=0.8
    # and std=0.10 (10%) → confidence=0.2
    if std_return < 0.01:
        confidence = 0.95
    elif std_return < 0.03:
        confidence = 0.8
    elif std_return < 0.06:
        confidence = 0.5
    elif std_return < 0.10:
        confidence = 0.3
    else:
        confidence = 0.15

    # Also factor in LLM's self-reported confidence
    conf_map = {"high": 0.8, "medium": 0.5, "low": 0.2}
    avg_self_conf = np.mean([
        conf_map.get(c.lower(), 0.5) for c in confidences
    ])
    # Blend statistical and self-reported confidence
    final_confidence = 0.7 * confidence + 0.3 * avg_self_conf

    # Most common reasoning
    best_reasoning = max(
        set(reasonings), key=reasonings.count
    ) if reasonings else ""

    log.info(
        f"  LLM: {signal.ticker} → "
        f"return={mean_return:+.1%} "
        f"(std={std_return:.1%}, n={len(predictions)}) "
        f"confidence={final_confidence:.2f}"
    )

    return BLView(
        ticker=signal.ticker,
        expected_return=mean_return,
        confidence=final_confidence,
        source="llm",
        reasoning=best_reasoning,
    )


def generate_all_views(
    signals: list,
    config: LLMConfig,
    market_data_map: dict[str, dict] | None = None,
    news_map: dict[str, list[str]] | None = None,
    holding_period: int = 10,
) -> list[BLView]:
    """
    Generate LLM views for all BUY signals.

    Args:
        signals: List of Signal objects.
        config: LLM configuration.
        market_data_map: Ticker → market data dict.
        news_map: Ticker → list of headlines.
        holding_period: Expected holding period in days.

    Returns:
        List of BLView objects.
    """
    if not config.enabled:
        log.info("  LLM views disabled — skipping")
        return []

    buy_signals = [s for s in signals if s.signal_raw == 1]
    if not buy_signals:
        return []

    # Sort by confidence (highest first) and cap at max_tickers
    # so we only spend API calls on the best signals
    buy_signals.sort(
        key=lambda s: (s.confidence_score, s.composite_score),
        reverse=True,
    )
    if config.max_tickers and len(buy_signals) > config.max_tickers:
        skipped = len(buy_signals) - config.max_tickers
        log.info(
            f"  LLM: {len(buy_signals)} BUY signals, "
            f"capping to top {config.max_tickers} by confidence "
            f"(skipping {skipped} lower-confidence signals)"
        )
        buy_signals = buy_signals[: config.max_tickers]

    market_data_map = market_data_map or {}
    news_map = news_map or {}

    views: list[BLView] = []
    total_cost_estimate = (
        len(buy_signals) * config.num_samples
    )
    log.info(
        f"  LLM: generating views for {len(buy_signals)} tickers "
        f"({total_cost_estimate} API calls)"
    )

    for signal in buy_signals:
        view = generate_view_for_ticker(
            signal=signal,
            config=config,
            market_data=market_data_map.get(signal.ticker),
            news_headlines=news_map.get(signal.ticker),
            holding_period=holding_period,
        )
        if view is not None:
            views.append(view)

    log.info(
        f"  LLM: generated {len(views)} views "
        f"out of {len(buy_signals)} attempted"
    )
    return views
