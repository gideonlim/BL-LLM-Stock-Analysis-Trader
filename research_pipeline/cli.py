"""CLI entry point for the research pipeline.

Usage:
    python -m research_pipeline scan          # full pipeline
    python -m research_pipeline scan --dry    # ingest only, no LLM
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from research_pipeline.config import ResearchConfig
from research_pipeline.extract import extract_signals
from research_pipeline.hypotheses import generate_hypotheses
from research_pipeline.ingest import ingest_news
from research_pipeline.models import PipelineRun
from research_pipeline.output import (
    write_hypotheses_summary,
    write_run_report,
)
from research_pipeline.themes import detect_themes


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level, format=fmt, stream=sys.stderr,
    )


def cmd_scan(args: argparse.Namespace) -> None:
    """Run the full scan pipeline."""
    # Load config
    dotenv_path = Path(args.env_file) if args.env_file else None
    if dotenv_path is None:
        # Auto-detect .env in common locations
        for candidate in [
            Path(".env"),
            Path("trading_bot_bl/.env"),
        ]:
            if candidate.exists():
                dotenv_path = candidate
                break

    config = ResearchConfig.from_env(dotenv_path)

    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    run = PipelineRun(
        run_id=uuid4().hex[:12],
        started_at=datetime.now(),
    )

    print(f"\n{'=' * 60}")
    print("  RESEARCH PIPELINE — News Scan")
    print(f"  Run ID: {run.run_id}")
    print(f"  Time:   {run.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # ── Step 1: Ingest ─────────────────────────────────────
    print("[1/4] Ingesting news...")
    try:
        news_items = ingest_news(config)
        run.news_ingested = len(news_items)
        print(f"      → {len(news_items)} headlines collected\n")
    except Exception as e:
        run.errors.append(f"Ingestion failed: {e}")
        print(f"      ✗ Ingestion failed: {e}\n")
        news_items = []

    if not news_items:
        print("No news items ingested. Exiting.")
        return

    # ── Step 2: Extract signals ────────────────────────────
    if args.dry:
        print("[2/4] Dry run — skipping LLM extraction")
        print("[3/4] Dry run — skipping theme detection")
        print("[4/4] Dry run — skipping hypothesis generation\n")

        # Still dump raw headlines for inspection
        print("Raw headlines (first 20):")
        for item in news_items[:20]:
            ts = item.published_at.strftime("%m-%d %H:%M")
            src = item.source
            tkr = f" [{item.ticker}]" if item.ticker else ""
            print(f"  ({ts}, {src}{tkr}) {item.headline}")
        return

    print("[2/4] Extracting structured signals via LLM...")
    try:
        signals = extract_signals(news_items, config)
        run.signals_extracted = len(signals)
        print(f"      → {len(signals)} signals extracted\n")
    except Exception as e:
        run.errors.append(f"Extraction failed: {e}")
        print(f"      ✗ Extraction failed: {e}\n")
        signals = []

    # ── Step 3: Detect themes ──────────────────────────────
    print("[3/4] Detecting accelerating themes...")
    try:
        themes = detect_themes(signals, config)
        run.themes_detected = len(themes)
        print(f"      → {len(themes)} themes detected\n")

        for t in themes:
            sent = f"{t.avg_sentiment:+.2f}"
            tickers = ", ".join(t.related_tickers[:8])
            print(
                f"      • {t.theme} "
                f"({t.mention_count} mentions, "
                f"sentiment={sent}) "
                f"→ [{tickers}]"
            )
        print()

    except Exception as e:
        run.errors.append(f"Theme detection failed: {e}")
        print(f"      ✗ Theme detection failed: {e}\n")
        themes = []

    # ── Step 4: Generate hypotheses ────────────────────────
    print("[4/4] Generating strategy hypotheses...")
    try:
        hypotheses = generate_hypotheses(themes, config)
        run.hypotheses_generated = len(hypotheses)
        print(f"      → {len(hypotheses)} hypotheses generated\n")

        for h in hypotheses:
            tickers = ", ".join(h.target_tickers)
            print(
                f"      [{h.confidence}] {h.template}: "
                f"{h.description}"
            )
            print(
                f"             Tickers: {tickers} | "
                f"Direction: {h.direction}"
            )
            print(f"             Chain: {h.causal_chain}")
            print()

    except Exception as e:
        run.errors.append(f"Hypothesis generation failed: {e}")
        print(f"      ✗ Hypothesis generation failed: {e}\n")
        hypotheses = []

    # ── Write outputs ──────────────────────────────────────
    report_path = write_run_report(
        run, signals, themes, hypotheses, config,
    )
    summary_path = write_hypotheses_summary(hypotheses, config)

    print(f"{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Headlines ingested:    {run.news_ingested}")
    print(f"  Signals extracted:     {run.signals_extracted}")
    print(f"  Themes detected:       {run.themes_detected}")
    print(f"  Hypotheses generated:  {run.hypotheses_generated}")
    print(f"  Full report:           {report_path}")
    if summary_path:
        print(f"  Hypotheses summary:    {summary_path}")
    if run.errors:
        print(f"  Errors:                {len(run.errors)}")
        for err in run.errors:
            print(f"    - {err}")
    print(f"{'=' * 60}\n")


def main() -> None:
    """Parse args and dispatch."""
    parser = argparse.ArgumentParser(
        description="Research pipeline for news-driven strategy discovery",
    )
    sub = parser.add_subparsers(dest="command")

    # scan command
    scan_parser = sub.add_parser(
        "scan", help="Run the full news scan pipeline",
    )
    scan_parser.add_argument(
        "--dry", action="store_true",
        help="Ingest only — skip LLM calls",
    )
    scan_parser.add_argument(
        "--env-file", type=str, default=None,
        help="Path to .env file (auto-detected if omitted)",
    )
    scan_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    scan_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    _setup_logging(getattr(args, "verbose", False))

    if args.command == "scan":
        cmd_scan(args)


if __name__ == "__main__":
    main()
