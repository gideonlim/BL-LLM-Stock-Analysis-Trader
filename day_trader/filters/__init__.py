"""Filter pipeline for the day-trader.

Per the user's filtering-first philosophy: the entry signal is not
the edge — the filter stack is. Every strategy emits raw signals
that flow through this pipeline; only signals that survive the
full chain get sized into order intents.

Pipeline order matters: cheap O(1) checks first (symbol locks,
regime, cooldowns) so we short-circuit before doing the expensive
ones (liquidity lookups, RVOL math). Order is configurable via
``FilterPipeline(filters=[...])`` but the default in
``build_default_pipeline()`` follows the plan.
"""
