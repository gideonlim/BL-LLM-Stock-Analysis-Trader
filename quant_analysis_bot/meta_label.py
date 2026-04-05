"""Meta-labeling model: learns which primary signals are worth acting on.

The meta-model takes triple-barrier trade outcomes as labels and
technical conditions at entry as features, producing P(success) for
each new signal.  This probability controls position sizing.

Model hierarchy:
  1. LightGBM (preferred, optional dep)
  2. sklearn GradientBoostingClassifier (fallback)
  3. sklearn RandomForestClassifier (last resort)

Versioned persistence with promotion gates and auto-rollback.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_analysis_bot.cv_purged import (
    compute_sample_weights,
    purged_kfold_split,
)
from quant_analysis_bot.triple_barrier import BarrierTrade

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rsi_14", "adx_14", "atr_14_pct", "bb_width",
    "close_vs_sma50", "close_vs_sma200", "vol_20",
    "roc_10", "volume_ratio",
    "sl_pct", "tp_pct", "rr_ratio",
    "strategy_id",
]

META_LABEL_MIN_TRAINING_TRADES = 50
META_LABEL_MIN_CALIBRATION_TRADES = 100
MODELS_DIR = "models"


# ── Feature extraction ───────────────────────────────────────────────


def extract_meta_features(
    df: pd.DataFrame,
    entry_idx: int,
    strategy_id: int,
    sl_pct: float = 0.0,
    tp_pct: float = 0.0,
) -> dict[str, float]:
    """Extract meta-label features at a specific entry bar.

    All features are backward-looking by construction —
    no future information leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched price DataFrame with indicators.
    entry_idx : int
        Positional index (iloc) of the entry bar.
    strategy_id : int
        Integer encoding of the strategy that generated the signal.
    sl_pct, tp_pct : float
        Barrier widths (%) for this trade.

    Returns
    -------
    dict mapping feature name to value.
    """
    row = df.iloc[entry_idx]
    price = float(row.get("Close", 0) or 0)
    atr = float(row.get("ATR_14", 0) or 0)
    sma50 = float(row.get("SMA_50", 0) or 0)
    sma200 = float(row.get("SMA_200", 0) or 0)

    return {
        "rsi_14": float(row.get("RSI_14", 50) or 50),
        "adx_14": float(row.get("ADX_14", 0) or 0),
        "atr_14_pct": (atr / price * 100) if price > 0 else 0.0,
        "bb_width": float(row.get("BB_Width", 0) or 0),
        "close_vs_sma50": (
            (price - sma50) / sma50 * 100 if sma50 > 0 else 0.0
        ),
        "close_vs_sma200": (
            (price - sma200) / sma200 * 100 if sma200 > 0 else 0.0
        ),
        "vol_20": float(row.get("Volatility_20", 0.25) or 0.25),
        "roc_10": float(row.get("ROC_10", 0) or 0),
        "volume_ratio": float(row.get("Vol_Ratio", 1.0) or 1.0),
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "rr_ratio": tp_pct / sl_pct if sl_pct > 0 else 2.0,
        "strategy_id": float(strategy_id),
    }


def build_training_data(
    barrier_trades: list[BarrierTrade],
    df: pd.DataFrame,
    ticker: str,
    strategy_id: int,
    sl_pct_arr: np.ndarray,
    tp_pct_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build feature matrix and labels from barrier trades.

    Parameters
    ----------
    barrier_trades : list[BarrierTrade]
        Trades from ``apply_triple_barrier()``.
    df : pd.DataFrame
        The DataFrame used to generate trades.
    ticker : str
        Ticker symbol for sample weight computation.
    strategy_id : int
        Strategy identifier.
    sl_pct_arr : np.ndarray
        Per-bar SL% array (from ATR computation at entry time).
        Required — using post-trade excursion would leak future info.
    tp_pct_arr : np.ndarray
        Per-bar TP% array (from ATR * RR at entry time).

    Returns
    -------
    X : np.ndarray, shape (n_trades, n_features)
    y : np.ndarray, shape (n_trades,)
        1 = TP hit (success), 0 = SL or vertical loss.
    events : pd.DataFrame
        For sample weight computation (ticker, entry_ts, exit_ts).
    """
    X_rows = []
    y_list = []
    event_rows = []

    dates = df.index

    for trade in barrier_trades:
        idx = trade.entry_idx
        if idx < 0 or idx >= len(df):
            continue

        sl = float(sl_pct_arr[idx])
        tp = float(tp_pct_arr[idx])

        features = extract_meta_features(
            df, idx, strategy_id, sl_pct=sl, tp_pct=tp,
        )
        X_rows.append([features[k] for k in FEATURE_NAMES])

        # Label: 1 = TP hit (success), 0 = everything else
        y_list.append(1 if trade.exit_barrier == "upper" else 0)

        event_rows.append({
            "ticker": ticker,
            "entry_ts": dates[trade.entry_idx],
            "exit_ts": dates[min(trade.exit_idx, len(dates) - 1)],
        })

    X = np.array(X_rows, dtype=np.float64) if X_rows else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(y_list, dtype=np.int32) if y_list else np.empty(0, dtype=np.int32)
    events = pd.DataFrame(event_rows) if event_rows else pd.DataFrame(
        columns=["ticker", "entry_ts", "exit_ts"]
    )

    return X, y, events


# ── Model creation ───────────────────────────────────────────────────


def _create_base_model() -> Any:
    """Create the best available tree model.

    Returns (model, model_name) tuple.
    """
    # Try LightGBM first
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        return model, "lightgbm"
    except ImportError:
        pass

    # Fallback to sklearn GradientBoosting
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        return model, "sklearn_gbc"
    except ImportError:
        pass

    # Last resort: RandomForest
    try:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
        )
        return model, "sklearn_rf"
    except ImportError:
        raise ImportError(
            "No ML backend available. Install lightgbm or scikit-learn."
        )


# ── Training ─────────────────────────────────────────────────────────


@dataclass
class TrainedMetaModel:
    """Wrapper for a trained meta-label model with metadata."""

    model: Any
    model_name: str
    is_calibrated: bool
    n_training_trades: int
    feature_names: list[str]
    train_date: str
    auc_score: float = 0.0
    feature_importance: dict[str, float] | None = None


def train_meta_model(
    X: np.ndarray,
    y: np.ndarray,
    events: pd.DataFrame,
    n_cv_splits: int = 5,
) -> TrainedMetaModel | None:
    """Train the meta-labeling model with purged CV and sample weights.

    Parameters
    ----------
    X : np.ndarray, shape (n, n_features)
    y : np.ndarray, shape (n,)
    events : pd.DataFrame
        For purged CV and sample weights.
    n_cv_splits : int
        Number of CV folds.

    Returns
    -------
    TrainedMetaModel or None if insufficient data.
    """
    n = len(y)

    if n < META_LABEL_MIN_TRAINING_TRADES:
        log.warning(
            "Meta-label: only %d trades (need %d), skipping training",
            n, META_LABEL_MIN_TRAINING_TRADES,
        )
        return None

    # Sample weights
    weights = compute_sample_weights(events)

    # Create base model
    base_model, model_name = _create_base_model()

    is_calibrated = False
    auc = 0.0

    if n >= META_LABEL_MIN_CALIBRATION_TRADES:
        # Train with calibration via PurgedKFold
        try:
            from sklearn.calibration import CalibratedClassifierCV

            cv_splits = purged_kfold_split(
                events, n_splits=n_cv_splits,
            )

            calibrated = CalibratedClassifierCV(
                base_model,
                method="sigmoid",
                cv=cv_splits,
            )
            calibrated.fit(X, y, sample_weight=weights)
            base_model = calibrated
            is_calibrated = True
            log.info(
                "Meta-label: trained calibrated %s on %d trades",
                model_name, n,
            )
        except Exception as e:
            log.warning(
                "Meta-label: calibration failed (%s), "
                "falling back to uncalibrated training",
                e,
            )
            base_model, model_name = _create_base_model()
            base_model.fit(X, y, sample_weight=weights)
    else:
        # Train without calibration (too few samples)
        base_model.fit(X, y, sample_weight=weights)
        log.info(
            "Meta-label: trained uncalibrated %s on %d trades "
            "(< %d for calibration)",
            model_name, n, META_LABEL_MIN_CALIBRATION_TRADES,
        )

    # Compute AUC on training data (for logging / promotion checks)
    try:
        from sklearn.metrics import roc_auc_score
        probs = base_model.predict_proba(X)[:, 1]
        if len(np.unique(y)) > 1:
            auc = round(roc_auc_score(y, probs), 4)
    except Exception:
        auc = 0.0

    # Feature importance (from base estimator if calibrated)
    fi = _extract_feature_importance(base_model, model_name)

    if fi:
        top_5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
        log.info(
            "Meta-label model: %d trades, AUC=%.3f\n  Top features: %s",
            n, auc,
            ", ".join(f"{k} ({v:.2f})" for k, v in top_5),
        )

    return TrainedMetaModel(
        model=base_model,
        model_name=model_name,
        is_calibrated=is_calibrated,
        n_training_trades=n,
        feature_names=list(FEATURE_NAMES),
        train_date=datetime.now().strftime("%Y%m%d"),
        auc_score=auc,
        feature_importance=fi,
    )


def _extract_feature_importance(model: Any, model_name: str) -> dict[str, float] | None:
    """Extract feature importance from model or its base estimator."""
    try:
        # For CalibratedClassifierCV, dig into base estimators
        if hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
        else:
            base = model

        if hasattr(base, "feature_importances_"):
            importances = base.feature_importances_
            return {
                FEATURE_NAMES[i]: round(float(v), 4)
                for i, v in enumerate(importances)
                if i < len(FEATURE_NAMES)
            }
    except Exception:
        pass
    return None


# ── Prediction ───────────────────────────────────────────────────────


def predict_meta_label(
    trained: TrainedMetaModel,
    df: pd.DataFrame,
    entry_idx: int,
    strategy_id: int,
    sl_pct: float = 3.0,
    tp_pct: float = 6.0,
) -> float:
    """Predict P(success) for a single entry.

    Parameters
    ----------
    trained : TrainedMetaModel
        Trained model wrapper.
    df : pd.DataFrame
        Enriched DataFrame.
    entry_idx : int
        Positional index of the entry bar.
    strategy_id : int
        Strategy encoding.
    sl_pct, tp_pct : float
        Barrier widths.

    Returns
    -------
    float
        Probability of success (0.0 to 1.0).
    """
    features = extract_meta_features(
        df, entry_idx, strategy_id, sl_pct=sl_pct, tp_pct=tp_pct,
    )
    X = np.array([[features[k] for k in FEATURE_NAMES]])

    try:
        prob = float(trained.model.predict_proba(X)[0, 1])
        return max(0.0, min(1.0, prob))
    except Exception as e:
        log.warning("Meta-label prediction failed: %s", e)
        return -1.0


# ── Persistence ──────────────────────────────────────────────────────


def _ensure_models_dir(base_dir: str = ".") -> Path:
    """Create and return the models directory path."""
    models_path = Path(base_dir) / MODELS_DIR
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


def save_meta_model(
    trained: TrainedMetaModel,
    base_dir: str = ".",
    ticker: str = "",
) -> str:
    """Save model with versioned filename.

    When ``ticker`` is provided, files are scoped per-ticker
    (e.g. ``meta_label_AAPL_latest.joblib``) to avoid races
    when multiple tickers are analysed in parallel.

    Returns the path to the saved model.
    """
    try:
        import joblib
    except ImportError:
        log.warning("joblib not available, cannot save meta-model")
        return ""

    models_path = _ensure_models_dir(base_dir)
    suffix = f"_{ticker}" if ticker else ""
    version = trained.train_date
    versioned_path = models_path / f"meta_label{suffix}_v{version}.joblib"
    latest_path = models_path / f"meta_label{suffix}_latest.joblib"

    joblib.dump(trained, versioned_path)

    # Update latest (copy, not symlink, for Windows compat)
    joblib.dump(trained, latest_path)

    # Save feature metadata + accuracy tracker
    meta_path = models_path / f"meta_label{suffix}_features.json"
    meta = {
        "feature_names": trained.feature_names,
        "model_name": trained.model_name,
        "is_calibrated": trained.is_calibrated,
        "n_training_trades": trained.n_training_trades,
        "train_date": trained.train_date,
        "auc_score": trained.auc_score,
        "feature_importance": trained.feature_importance,
        "recent_predictions": [],  # accuracy tracker
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Meta-model saved: %s (AUC=%.3f)", versioned_path, trained.auc_score)
    return str(versioned_path)


def load_meta_model(
    base_dir: str = ".",
    ticker: str = "",
) -> TrainedMetaModel | None:
    """Load the latest meta-label model from disk.

    Looks for per-ticker model first (``meta_label_AAPL_latest.joblib``),
    then falls back to the global model (``meta_label_latest.joblib``).
    """
    try:
        import joblib
    except ImportError:
        return None

    models_path = Path(base_dir) / MODELS_DIR

    # Try per-ticker model first, then global
    candidates = []
    if ticker:
        candidates.append(models_path / f"meta_label_{ticker}_latest.joblib")
    candidates.append(models_path / "meta_label_latest.joblib")

    latest_path = None
    for p in candidates:
        if p.exists():
            latest_path = p
            break

    if latest_path is None:
        return None

    try:
        trained = joblib.load(latest_path)
        if isinstance(trained, TrainedMetaModel):
            return trained
        log.warning("Loaded object is not a TrainedMetaModel")
        return None
    except Exception as e:
        log.warning("Failed to load meta-model: %s", e)
        return None


def should_promote(
    new_trained: TrainedMetaModel,
    old_trained: TrainedMetaModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    auc_regression_threshold: float = 0.03,
) -> bool:
    """Check if new model should replace old one.

    Only promotes if AUC doesn't regress by more than threshold.
    """
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y_val)) < 2 or len(y_val) < 10:
            return True  # Not enough validation data to judge

        new_probs = new_trained.model.predict_proba(X_val)[:, 1]
        old_probs = old_trained.model.predict_proba(X_val)[:, 1]

        new_auc = roc_auc_score(y_val, new_probs)
        old_auc = roc_auc_score(y_val, old_probs)

        if new_auc < old_auc - auc_regression_threshold:
            log.warning(
                "Meta-model retrain rejected: AUC %.3f → %.3f "
                "(regression > %.3f)",
                old_auc, new_auc, auc_regression_threshold,
            )
            return False

        log.info(
            "Meta-model promotion approved: AUC %.3f → %.3f",
            old_auc, new_auc,
        )
        return True
    except Exception as e:
        log.warning("Promotion check failed (%s), promoting anyway", e)
        return True


def should_retrain(
    base_dir: str = ".",
    retrain_days: int = 7,
    ticker: str = "",
) -> bool:
    """Check if enough time has passed since last training."""
    suffix = f"_{ticker}" if ticker else ""
    meta_path = Path(base_dir) / MODELS_DIR / f"meta_label{suffix}_features.json"
    if not meta_path.exists():
        return True

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        last_date = datetime.strptime(meta["train_date"], "%Y%m%d")
        return (datetime.now() - last_date) >= timedelta(days=retrain_days)
    except Exception:
        return True


def check_rollback(
    base_dir: str = ".",
    accuracy_threshold: float = 0.45,
    window: int = 50,
) -> bool:
    """Check if running accuracy has dropped below threshold.

    Returns True if rollback is needed.
    """
    meta_path = Path(base_dir) / MODELS_DIR / "meta_label_features.json"
    if not meta_path.exists():
        return False

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        predictions = meta.get("recent_predictions", [])
        if len(predictions) < window:
            return False

        recent = predictions[-window:]
        accuracy = sum(1 for p in recent if p["correct"]) / len(recent)
        if accuracy < accuracy_threshold:
            log.warning(
                "Meta-model rollback triggered: accuracy %.2f < %.2f "
                "over last %d predictions",
                accuracy, accuracy_threshold, window,
            )
            return True
        return False
    except Exception:
        return False


def record_prediction(
    actual_outcome: int,
    predicted_prob: float,
    threshold: float = 0.5,
    base_dir: str = ".",
) -> None:
    """Record a prediction outcome for running accuracy tracking."""
    meta_path = Path(base_dir) / MODELS_DIR / "meta_label_features.json"
    if not meta_path.exists():
        return

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        predictions = meta.get("recent_predictions", [])
        predicted = 1 if predicted_prob >= threshold else 0
        predictions.append({
            "predicted_prob": round(predicted_prob, 4),
            "predicted": predicted,
            "actual": actual_outcome,
            "correct": predicted == actual_outcome,
            "date": datetime.now().strftime("%Y-%m-%d"),
        })

        # Keep last 200 predictions
        meta["recent_predictions"] = predictions[-200:]

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass


# ── Meta-label-adjusted Kelly sizing ─────────────────────────────────


def compute_meta_kelly(
    meta_prob: float,
    base_kelly_f: float,
    profit_factor: float,
    n_training_trades: int,
    is_calibrated: bool,
    min_training_trades: int = META_LABEL_MIN_TRAINING_TRADES,
) -> tuple[float, float]:
    """Compute meta-label-adjusted Kelly fraction with safety bounds.

    Parameters
    ----------
    meta_prob : float
        P(success) from the meta-model (0 to 1).
    base_kelly_f : float
        Static Half-Kelly fraction (from backtest metrics).
    profit_factor : float
        Backtest profit factor.
    n_training_trades : int
        Number of trades the meta-model was trained on.
    is_calibrated : bool
        Whether the model's probabilities are calibrated.
    min_training_trades : int
        Minimum trades before trusting meta-model.

    Returns
    -------
    (final_kelly, size_mult) : tuple[float, float]
        final_kelly is the adjusted Kelly fraction.
        size_mult is the multiplier vs base_kelly (for logging).
    """
    pf = max(profit_factor, 0.01)

    # Gate: not enough training data — use static Kelly
    if n_training_trades < min_training_trades or meta_prob < 0:
        return base_kelly_f, 1.0

    # Clip probability bounds
    if is_calibrated:
        prob_lo, prob_hi = 0.20, 0.85
    else:
        prob_lo, prob_hi = 0.35, 0.65

    clipped_prob = max(prob_lo, min(meta_prob, prob_hi))

    # Meta-Kelly from clipped probability
    meta_kelly_f = max(
        (clipped_prob * pf - (1 - clipped_prob)) / pf, 0.0
    ) * 0.5

    # Blend: trust meta more as training data grows
    blend_weight = min(n_training_trades / 200, 0.7)
    if not is_calibrated:
        blend_weight *= 0.5  # halve trust without calibration

    final_kelly = (1 - blend_weight) * base_kelly_f + blend_weight * meta_kelly_f

    # Hard cap — no floor (zero edge = zero size)
    final_kelly = min(final_kelly, 0.50)

    # Size multiplier for logging
    size_mult = final_kelly / base_kelly_f if base_kelly_f > 0 else 1.0

    return final_kelly, round(size_mult, 4)
