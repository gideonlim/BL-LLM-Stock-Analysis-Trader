"""Day-trader module — 25% sub-portfolio intraday trading.

Sister package to ``quant_analysis_bot/`` (signal generation) and
``trading_bot_bl/`` (swing execution). Shares the broker, journal,
and risk infrastructure with ``trading_bot_bl/`` while running its
own daemon on a cloud VM for sub-minute precision intraday strategies.

Architecture and roadmap: ~/.claude/plans/i-want-to-allocate-bubbly-knuth.md

This is the FOUNDATION + SAFETY layer — package skeleton, NYSE
calendar, order tagging, symbol locks, safe close mechanics, sub-
budget tracking, and recovery reconciliation. Strategies, filters,
data feed, and the executor daemon land in subsequent layers.
"""

__version__ = "0.1.0-foundation"
