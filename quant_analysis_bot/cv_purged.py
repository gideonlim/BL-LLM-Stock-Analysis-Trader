"""Purged K-Fold cross-validation and sample uniqueness weighting.

Prevents information leakage when cross-validating overlapping
triple-barrier labels.  Based on López de Prado (2018), Ch. 7.

Key concepts:
  - **Purging**: remove training samples whose label windows
    overlap with any test sample (per-ticker).
  - **Embargo**: additionally remove a buffer of training
    observations near the test boundary.
  - **Sample uniqueness**: weight each trade by the fraction of
    its label window not shared with concurrent trades.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_sample_weights(events: pd.DataFrame) -> np.ndarray:
    """Average uniqueness of each trade's label window.

    Overlap is measured ONLY within the same ticker — trades on
    different tickers never share a return stream.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain columns ``ticker``, ``entry_ts`` (datetime),
        ``exit_ts`` (datetime).

    Returns
    -------
    np.ndarray
        Weight per trade, normalised so mean = 1.
    """
    if events.empty:
        return np.array([])

    weights = np.ones(len(events))

    for ticker, group in events.groupby("ticker"):
        if len(group) < 2:
            continue

        idx = group.index
        entries = group["entry_ts"].values
        exits = group["exit_ts"].values

        # Build a daily concurrency count for this ticker
        ts_min = pd.Timestamp(entries.min())
        ts_max = pd.Timestamp(exits.max())
        if ts_min == ts_max:
            continue

        ts_range = pd.date_range(ts_min, ts_max, freq="D")
        concurrency = np.zeros(len(ts_range), dtype=np.int32)

        # Map timestamps to integer offsets for fast array ops
        origin = ts_range[0]
        range_days = (ts_range[-1] - origin).days

        for _, row in group.iterrows():
            start_off = max((pd.Timestamp(row["entry_ts"]) - origin).days, 0)
            end_off = min((pd.Timestamp(row["exit_ts"]) - origin).days, range_days)
            concurrency[start_off:end_off + 1] += 1

        # Compute average uniqueness per trade
        for i, (pos, row) in enumerate(group.iterrows()):
            start_off = max((pd.Timestamp(row["entry_ts"]) - origin).days, 0)
            end_off = min((pd.Timestamp(row["exit_ts"]) - origin).days, range_days)
            window = concurrency[start_off:end_off + 1]
            # Avoid division by zero
            safe_window = np.maximum(window, 1)
            weights[pos] = np.mean(1.0 / safe_window)

    # Normalise to mean = 1
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w

    return weights


def purged_kfold_split(
    events: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged K-Fold cross-validation splits.

    For each fold, purges training samples whose label windows
    overlap with any test sample on the same ticker.  Then removes
    an additional ``embargo_pct`` fraction of the nearest training
    observations as a buffer against serial correlation.

    Parameters
    ----------
    events : pd.DataFrame
        Must contain ``ticker``, ``entry_ts``, ``exit_ts``.
        Index should be integer (0..N-1).
    n_splits : int
        Number of CV folds.
    embargo_pct : float
        Fraction of training set to embargo near test boundaries.

    Returns
    -------
    list of (train_indices, test_indices) tuples.
    """
    n = len(events)
    if n < n_splits * 2:
        log.warning(
            "Too few samples (%d) for %d-fold purged CV; "
            "returning single train/test split",
            n, n_splits,
        )
        mid = n // 2
        return [(np.arange(0, mid), np.arange(mid, n))]

    indices = np.arange(n)
    fold_size = n // n_splits
    splits = []

    for k in range(n_splits):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_splits - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Purge: remove training samples overlapping with test
        test_events = events.iloc[test_idx]
        purge_mask = np.ones(len(train_idx), dtype=bool)

        for ticker, test_group in test_events.groupby("ticker"):
            test_min_entry = pd.Timestamp(test_group["entry_ts"].min())
            test_max_exit = pd.Timestamp(test_group["exit_ts"].max())

            for i, ti in enumerate(train_idx):
                row = events.iloc[ti]
                if row["ticker"] != ticker:
                    continue
                train_entry = pd.Timestamp(row["entry_ts"])
                train_exit = pd.Timestamp(row["exit_ts"])

                # Overlap: train label window intersects test label window
                if train_entry <= test_max_exit and train_exit >= test_min_entry:
                    purge_mask[i] = False

        train_idx = train_idx[purge_mask]

        # Embargo: remove additional samples near test boundary
        n_embargo = max(int(len(train_idx) * embargo_pct), 0)
        if n_embargo > 0 and len(train_idx) > n_embargo:
            # Remove training samples just before test start
            embargo_remove = set()
            # Find training indices just before the test window
            pre_test = train_idx[train_idx < test_start]
            if len(pre_test) > 0:
                embargo_remove.update(pre_test[-n_embargo:])
            # Find training indices just after the test window
            post_test = train_idx[train_idx >= test_end]
            if len(post_test) > 0:
                embargo_remove.update(post_test[:n_embargo])

            if embargo_remove:
                train_idx = np.array(
                    [i for i in train_idx if i not in embargo_remove]
                )

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits
