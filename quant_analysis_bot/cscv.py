"""
Combinatorial Symmetric Cross-Validation (CSCV) and
Probability of Backtest Overfitting (PBO).

Based on Bailey, Borwein, López de Prado & Zhu (2017):
"The Probability of Backtest Overfitting"

The idea: split a return matrix (N strategies × T observations) into S
equal-length partitions.  Form all C(S, S/2) ways to assign half the
partitions to "training" (IS) and the other half to "testing" (OOS).
For each combination:
  1. Pick the best strategy on IS (by Sharpe or score).
  2. Measure that strategy's rank / performance on OOS.
  3. If the IS-best strategy has a *below-median* OOS rank, that
     combination is "overfit."

PBO = fraction of combinations where IS-best underperforms OOS.
A PBO > 0.50 means selecting the best IS strategy is *worse* than
random selection — strong evidence of overfitting.

Stochastic Dominance (logits):
  For each combination we also compute the logit of the OOS rank:
    λ = log(rank / (S_strats - rank))
  The distribution of λ tells us how consistently the IS-best
  strategy degrades OOS.  A distribution centered well below 0
  means IS selection transfers to OOS.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant_analysis_bot.backtest import run_backtest, score_single_window
from quant_analysis_bot.config import RISK_PROFILES
from quant_analysis_bot.strategies import ALL_STRATEGIES, Strategy

log = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────


@dataclass
class CSCVResult:
    """Results from a full CSCV / PBO analysis."""

    # Core PBO metric: fraction of combos where IS-best underperforms OOS
    pbo: float = 1.0

    # Number of partition combinations evaluated
    n_combinations: int = 0

    # Number of partitions used
    n_partitions: int = 0

    # Strategy-level OOS rank distribution
    # Key = strategy name, value = list of OOS ranks across combos
    oos_ranks: Dict[str, List[int]] = field(default_factory=dict)

    # Logit distribution: log(rank / (n_strats - rank)) for IS-best
    logits: List[float] = field(default_factory=list)

    # Mean OOS rank of the IS-best strategy (lower = better)
    mean_oos_rank: float = 0.0

    # Rank correlation between IS and OOS (Spearman) averaged across combos
    mean_rank_correlation: float = 0.0

    # Per-strategy summary: {name: {"mean_is_rank", "mean_oos_rank", "pbo"}}
    strategy_summary: Dict[str, dict] = field(default_factory=dict)

    # Minimum data per partition (for diagnostic)
    min_partition_days: int = 0

    # Whether the analysis had enough data to be meaningful
    is_valid: bool = False

    def summary(self) -> str:
        """Human-readable summary."""
        if not self.is_valid:
            return "CSCV: insufficient data for analysis"

        verdict = "LOW" if self.pbo < 0.30 else (
            "MODERATE" if self.pbo < 0.50 else "HIGH"
        )
        lines = [
            f"CSCV Overfitting Analysis ({self.n_combinations} combinations, "
            f"{self.n_partitions} partitions)",
            f"  PBO:                  {self.pbo:.1%}  [{verdict} overfitting risk]",
            f"  Mean OOS rank:        {self.mean_oos_rank:.1f} / {len(ALL_STRATEGIES)}",
            f"  Mean rank correlation: {self.mean_rank_correlation:+.3f}",
            f"  Logit distribution:   mean={_safe_mean(self.logits):+.2f}, "
            f"std={_safe_std(self.logits):.2f}",
        ]
        return "\n".join(lines)


def _safe_mean(arr: list) -> float:
    return float(np.mean(arr)) if arr else 0.0


def _safe_std(arr: list) -> float:
    return float(np.std(arr)) if arr else 0.0


# ── Core CSCV engine ─────────────────────────────────────────────────


def _build_return_matrix(
    df: pd.DataFrame,
    strategies: List[Strategy],
    config: dict,
    ticker: str,
) -> Optional[pd.DataFrame]:
    """
    Build a (T × N_strategies) matrix of daily returns.

    Each column is one strategy's daily return stream from run_backtest.
    Rows are aligned trading days.
    """
    cost_bps = config.get("transaction_cost_bps", 10)
    long_only = config.get("long_only", True)
    use_next_bar = config.get("next_bar_execution", True)

    return_streams = {}
    for strat in strategies:
        try:
            signals = strat.generate_signals(df)
            result, _, returns_arr = run_backtest(
                df, signals, ticker, strat.name, "cscv",
                cost_bps=cost_bps,
                long_only=long_only,
                next_bar_execution=use_next_bar,
            )
            if len(returns_arr) > 0:
                # Align to df index (skip first bar due to return calc)
                idx = df.index[1:len(returns_arr) + 1]
                return_streams[strat.name] = pd.Series(
                    returns_arr[:len(idx)], index=idx
                )
        except Exception as e:
            log.debug(f"CSCV: {strat.name} failed: {e}")

    if len(return_streams) < 3:
        log.warning("CSCV: fewer than 3 strategies produced returns")
        return None

    ret_matrix = pd.DataFrame(return_streams).dropna()
    if len(ret_matrix) < 60:
        log.warning(f"CSCV: only {len(ret_matrix)} usable days (need ≥60)")
        return None

    return ret_matrix


def _partition_matrix(
    ret_matrix: pd.DataFrame, n_partitions: int
) -> List[pd.DataFrame]:
    """Split return matrix into n_partitions roughly equal sub-periods."""
    n_rows = len(ret_matrix)
    partition_size = n_rows // n_partitions
    partitions = []
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size if i < n_partitions - 1 else n_rows
        partitions.append(ret_matrix.iloc[start:end])
    return partitions


def _score_partition_set(
    partitions: List[pd.DataFrame],
    indices: Tuple[int, ...],
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Score each strategy on a subset of partitions.

    Returns {strategy_name: sharpe_ratio} computed on the concatenated
    partitions identified by indices.
    """
    # Concatenate the selected partitions
    combined = pd.concat([partitions[i] for i in indices])

    sqrt_td = np.sqrt(trading_days_per_year)
    scores = {}
    for col in combined.columns:
        returns = combined[col].values
        if len(returns) < 10 or returns.std() == 0:
            scores[col] = 0.0
        else:
            scores[col] = float(
                sqrt_td * returns.mean() / returns.std()
            )
    return scores


def _spearman_rank_corr(ranks_a: List[float], ranks_b: List[float]) -> float:
    """Spearman rank correlation between two rank vectors."""
    n = len(ranks_a)
    if n < 2:
        return 0.0
    a = np.array(ranks_a, dtype=float)
    b = np.array(ranks_b, dtype=float)
    d = a - b
    return float(1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def run_cscv(
    df: pd.DataFrame,
    ticker: str,
    config: dict,
    n_partitions: int = 0,
    strategies: Optional[List[Strategy]] = None,
    max_combinations: int = 500,
    trading_days_per_year: int = 252,
) -> CSCVResult:
    """
    Run full CSCV / PBO analysis on a ticker's data.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched price DataFrame (output of enrich_dataframe).
    ticker : str
        Ticker symbol.
    config : dict
        Standard bot config.
    n_partitions : int
        Number of partitions (S). If 0, auto-select based on data length.
        Must be even.
    strategies : list, optional
        Strategy list to test. Defaults to ALL_STRATEGIES.
    max_combinations : int
        Cap on C(S, S/2) combinations to evaluate (random sample if exceeded).

    Returns
    -------
    CSCVResult with PBO and diagnostic metrics.
    """
    result = CSCVResult()
    strats = strategies or ALL_STRATEGIES

    # Build return matrix
    ret_matrix = _build_return_matrix(df, strats, config, ticker)
    if ret_matrix is None:
        return result

    n_days = len(ret_matrix)
    n_strats = len(ret_matrix.columns)

    # Auto-select partition count: target ~25+ days per partition
    if n_partitions == 0:
        if n_days >= 480:
            n_partitions = 16  # C(16,8) = 12870 → sample
        elif n_days >= 240:
            n_partitions = 10  # C(10,5) = 252
        elif n_days >= 120:
            n_partitions = 8   # C(8,4) = 70
        elif n_days >= 60:
            n_partitions = 6   # C(6,3) = 20
        else:
            log.warning(f"CSCV: only {n_days} days, skipping")
            return result

    # Ensure even
    if n_partitions % 2 != 0:
        n_partitions += 1

    result.n_partitions = n_partitions
    result.min_partition_days = n_days // n_partitions

    if result.min_partition_days < 15:
        log.warning(
            f"CSCV: partition size {result.min_partition_days} days "
            f"too small, skipping"
        )
        return result

    # Partition the data
    partitions = _partition_matrix(ret_matrix, n_partitions)

    # Generate all C(S, S/2) combinations
    half = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), half))

    # Sample if too many
    rng = np.random.default_rng(42)
    if len(all_combos) > max_combinations:
        indices = rng.choice(
            len(all_combos), size=max_combinations, replace=False
        )
        all_combos = [all_combos[i] for i in sorted(indices)]

    result.n_combinations = len(all_combos)

    # Initialize tracking
    overfit_count = 0
    logits = []
    rank_correlations = []
    oos_ranks_by_strat: Dict[str, List[int]] = {
        col: [] for col in ret_matrix.columns
    }
    is_ranks_by_strat: Dict[str, List[int]] = {
        col: [] for col in ret_matrix.columns
    }

    strat_names = list(ret_matrix.columns)

    for combo in all_combos:
        # Training set = combo indices, Test set = complement
        oos_indices = tuple(
            i for i in range(n_partitions) if i not in combo
        )

        # Score strategies on IS and OOS
        is_scores = _score_partition_set(
            partitions, combo, trading_days_per_year
        )
        oos_scores = _score_partition_set(
            partitions, oos_indices, trading_days_per_year
        )

        # Rank strategies (1 = best)
        is_ranked = sorted(
            strat_names, key=lambda s: is_scores.get(s, 0), reverse=True
        )
        oos_ranked = sorted(
            strat_names, key=lambda s: oos_scores.get(s, 0), reverse=True
        )

        is_rank_map = {s: r + 1 for r, s in enumerate(is_ranked)}
        oos_rank_map = {s: r + 1 for r, s in enumerate(oos_ranked)}

        # Track ranks
        for s in strat_names:
            is_ranks_by_strat[s].append(is_rank_map[s])
            oos_ranks_by_strat[s].append(oos_rank_map[s])

        # IS-best strategy
        best_is = is_ranked[0]
        oos_rank_of_best = oos_rank_map[best_is]

        # PBO: IS-best below median OOS rank?
        median_rank = (n_strats + 1) / 2.0
        if oos_rank_of_best > median_rank:
            overfit_count += 1

        # Logit: log(rank / (n - rank))
        # Clamp to avoid log(0)
        r = max(0.5, min(oos_rank_of_best, n_strats - 0.5))
        logit = math.log(r / (n_strats - r))
        logits.append(logit)

        # Spearman rank correlation IS vs OOS
        is_ranks_vec = [is_rank_map[s] for s in strat_names]
        oos_ranks_vec = [oos_rank_map[s] for s in strat_names]
        rho = _spearman_rank_corr(is_ranks_vec, oos_ranks_vec)
        rank_correlations.append(rho)

    # Compute final metrics
    result.pbo = overfit_count / len(all_combos) if all_combos else 1.0
    result.logits = logits
    result.oos_ranks = oos_ranks_by_strat

    # Mean OOS rank of IS-best: recover from logits
    # logit = log(r / (n-r)), so r = n / (1 + exp(-logit))
    if logits:
        ranks_from_logits = [
            n_strats / (1.0 + math.exp(-l)) for l in logits
        ]
        result.mean_oos_rank = float(np.mean(ranks_from_logits))

    result.mean_rank_correlation = float(
        np.mean(rank_correlations)
    ) if rank_correlations else 0.0

    # Per-strategy summary
    for s in strat_names:
        is_mean = float(np.mean(is_ranks_by_strat[s]))
        oos_mean = float(np.mean(oos_ranks_by_strat[s]))
        # Per-strategy PBO: fraction of combos where this strategy
        # was IS-best but OOS below median
        n_was_best = sum(
            1 for ranks in zip(
                is_ranks_by_strat[s], oos_ranks_by_strat[s]
            )
            if ranks[0] == 1
        )
        n_overfit_when_best = sum(
            1 for is_r, oos_r in zip(
                is_ranks_by_strat[s], oos_ranks_by_strat[s]
            )
            if is_r == 1 and oos_r > median_rank
        )
        result.strategy_summary[s] = {
            "mean_is_rank": round(is_mean, 1),
            "mean_oos_rank": round(oos_mean, 1),
            "rank_degradation": round(oos_mean - is_mean, 1),
            "times_is_best": n_was_best,
            "pbo_when_best": (
                round(n_overfit_when_best / n_was_best, 3)
                if n_was_best > 0 else None
            ),
        }

    result.is_valid = True
    return result


# ── Convenience: run CSCV on the longest available window ────────────


def run_cscv_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    config: dict,
) -> CSCVResult:
    """
    Run CSCV using the full available data (longest lookback).

    This is the main entry point for integration with the pipeline.
    Uses the full df rather than individual windows, since CSCV needs
    as much data as possible for meaningful partitioning.
    """
    return run_cscv(df, ticker, config)


# ── Formatted report ─────────────────────────────────────────────────


def format_cscv_report(
    results: Dict[str, CSCVResult],
) -> str:
    """
    Format CSCV results for multiple tickers into a readable report.

    Parameters
    ----------
    results : dict
        {ticker: CSCVResult}
    """
    lines = [
        "=" * 70,
        "  CSCV OVERFITTING ANALYSIS",
        "=" * 70,
        "",
    ]

    valid_results = {
        t: r for t, r in results.items() if r.is_valid
    }

    if not valid_results:
        lines.append("  No tickers had sufficient data for CSCV analysis.")
        lines.append("  Need at least 60 trading days of strategy returns.")
        return "\n".join(lines)

    # Summary table
    lines.append(
        f"  {'Ticker':<8} {'PBO':>6} {'Risk':>10} "
        f"{'OOS Rank':>10} {'Rank Corr':>10} {'Combos':>8}"
    )
    lines.append("  " + "-" * 58)

    for ticker, r in sorted(
        valid_results.items(),
        key=lambda x: x[1].pbo,
        reverse=True,
    ):
        verdict = (
            "LOW" if r.pbo < 0.30
            else "MODERATE" if r.pbo < 0.50
            else "HIGH"
        )
        n_strats = len(r.strategy_summary)
        lines.append(
            f"  {ticker:<8} {r.pbo:>5.1%} {verdict:>10} "
            f"{r.mean_oos_rank:>7.1f}/{n_strats:<2} "
            f"{r.mean_rank_correlation:>+9.3f} {r.n_combinations:>8}"
        )

    # Aggregate statistics
    all_pbos = [r.pbo for r in valid_results.values()]
    lines.extend([
        "",
        f"  Mean PBO across tickers: {np.mean(all_pbos):.1%}",
        f"  Tickers with HIGH overfitting risk (PBO > 50%): "
        f"{sum(1 for p in all_pbos if p > 0.50)}/{len(all_pbos)}",
        "",
    ])

    # Interpretation guide
    lines.extend([
        "  Interpretation:",
        "    PBO < 30%:  Low overfitting risk — IS selection likely transfers to OOS",
        "    PBO 30-50%: Moderate risk — some degradation expected",
        "    PBO > 50%:  High risk — IS-best strategy often underperforms OOS",
        "    Rank Corr:  Positive = IS ranks predict OOS ranks (good)",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)
