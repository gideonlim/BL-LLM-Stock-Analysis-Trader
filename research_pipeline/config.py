"""Configuration for the research pipeline.

Reads from the project-level .env (shared with trading_bot_bl)
plus any RESEARCH_* variables specific to this pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResearchConfig:
    """All tuneable knobs for the research pipeline."""

    # ── API keys (shared with trading_bot_bl) ───────────────────
    anthropic_api_key: str = ""
    finnhub_api_key: str = ""

    # ── LLM settings ───────────────────────────────────────────
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 2048

    # ── Ingestion ──────────────────────────────────────────────
    news_days_back: int = 3
    max_headlines_per_ticker: int = 10
    # Broad macro / sector tickers to always scan
    watchlist_tickers: tuple[str, ...] = (
        "SPY", "QQQ", "XLE", "XLF", "XLV", "XLP", "XLU",
        "XLB", "XLI", "XLK", "USO", "GLD", "TLT", "DBA",
    )

    # ── Theme detection ────────────────────────────────────────
    # Minimum number of articles mentioning a theme for it to
    # count as "accelerating"
    min_theme_mentions: int = 3

    # ── Output ─────────────────────────────────────────────────
    output_dir: Path = Path("research_output")
    signals_dir: Path = Path("signals")

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> ResearchConfig:
        """Load config from environment variables.

        Optionally loads a .env file first so this works standalone.
        """
        if dotenv_path and dotenv_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path)
            except ImportError:
                pass

        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
            llm_model=os.getenv(
                "RESEARCH_LLM_MODEL", "claude-haiku-4-5-20251001"
            ),
            llm_temperature=float(
                os.getenv("RESEARCH_LLM_TEMPERATURE", "0.4")
            ),
            llm_max_tokens=int(
                os.getenv("RESEARCH_LLM_MAX_TOKENS", "2048")
            ),
            news_days_back=int(
                os.getenv("RESEARCH_NEWS_DAYS_BACK", "3")
            ),
            max_headlines_per_ticker=int(
                os.getenv("RESEARCH_MAX_HEADLINES", "10")
            ),
            output_dir=Path(
                os.getenv("RESEARCH_OUTPUT_DIR", "research_output")
            ),
            signals_dir=Path(
                os.getenv("SIGNALS_DIR", "signals")
            ),
        )
