"""Daily signal generation with execution details."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from quant_analysis_bot.config import RISK_PROFILES
from quant_analysis_bot.models import BacktestResult, DailySignal
from quant_analysis_bot.strategies import Strategy
from quant_analysis_bot.tp_logic import (
    apply_tp_cap,
    classify_strategy_family,
    compute_expected_max_move_pct,
    compute_rr_ratio,
)

log = logging.getLogger(__name__)


# ── Earnings context ─────────────────────────────────────────────────


@dataclass(frozen=True)
class EarningsContext:
    """Earnings proximity and surprise data for signal generation.

    Populated by the CLI pipeline from yfinance data.
    When unavailable, a neutral default (no adjustment) is used.
    """

    days_to_earnings: int = -1  # -1 = unknown
    earnings_date: str = ""     # ISO date or ""
    last_surprise_pct: float = float("nan")  # NaN = unknown
    # Trading days since the last earnings event (from PEAD_Days_Since).
    # Used to enforce the 60-day PEAD recency window.
    # 0 = unknown/unavailable (always passes the recency check).
    surprise_days_since: int = 0

    @property
    def is_available(self) -> bool:
        return self.days_to_earnings >= 0


_NEUTRAL_EARNINGS = EarningsContext()


def compute_earnings_confidence_adj(
    ctx: EarningsContext,
    *,
    blackout_pre_days: int = 3,
    surprise_boost_threshold: float = 5.0,
    surprise_penalty_threshold: float = -5.0,
    max_surprise_days: int = 60,
) -> int:
    """Compute a confidence score adjustment from earnings context.

    Rules:
      1. **Pre-earnings penalty**: If earnings are within
         ``blackout_pre_days``, apply -2 to discourage new entries.
         This is a soft version of the hard blackout in risk.py —
         the signal still generates but with reduced confidence,
         making it less likely to survive the min_confidence gate.

      2. **Post-earnings surprise boost**: If the most recent
         earnings surprise was strongly positive (> +5%), apply +1
         to reward PEAD momentum.  Strongly negative (< -5%)
         applies -1.  The boost only applies within 60 trading
         days of the earnings (the academic PEAD window).

      3. **Earnings-day penalty**: On earnings day itself (day 0),
         apply -3 (maximum deterrent).

    Returns an integer adjustment to add to the raw confidence_score.
    """
    adj = 0

    # ── Proximity rules (require a known forward date) ────────
    if ctx.is_available:
        # Rule 3: earnings TODAY — strongest penalty
        if ctx.days_to_earnings == 0:
            return -3

        # Rule 1: approaching earnings — discourage entry
        if 0 < ctx.days_to_earnings <= blackout_pre_days:
            adj -= 2

    # ── Surprise rule (works even without forward date) ───────
    # build_earnings_context can return a valid last_surprise_pct
    # with days_to_earnings=-1 when the yfinance calendar lookup
    # fails but PEAD historical data is available.
    if (
        not math.isnan(ctx.last_surprise_pct)
        and ctx.surprise_days_since <= max_surprise_days
    ):
        if ctx.last_surprise_pct >= surprise_boost_threshold:
            adj += 1
            log.debug(
                f"Earnings surprise boost: "
                f"{ctx.last_surprise_pct:+.1f}% → +1"
            )
        elif ctx.last_surprise_pct <= surprise_penalty_threshold:
            adj -= 1
            log.debug(
                f"Earnings surprise penalty: "
                f"{ctx.last_surprise_pct:+.1f}% → -1"
            )

    return adj


# ── ATR-based stop-loss computation (shared with triple barrier) ───


def compute_atr_stop_loss_pct(
    atr: float,
    vol_20: float,
    price: float,
) -> float:
    """Compute stop-loss percentage from ATR and volatility regime.

    Uses the same volatility-adaptive multiplier as live bracket
    orders, so backtests match real execution.

    Parameters
    ----------
    atr : float
        ATR_14 value.
    vol_20 : float
        Annualised 20-day realised volatility.
    price : float
        Current close price.

    Returns
    -------
    float
        Stop-loss as a percentage (e.g. 5.0 = 5%), clamped [1.5, 12.0].
        Returns 3.0 as default if ATR/price are invalid.
    """
    if atr <= 0 or price <= 0:
        return 3.0  # safe default

    if vol_20 > 0.4:
        atr_mult = 2.0
    elif vol_20 > 0.2:
        atr_mult = 1.5
    else:
        atr_mult = 1.2

    return round(min(max((atr * atr_mult / price) * 100, 1.5), 12.0), 2)


# ── Volatility-targeted sizing ─────────────────────────────────────


def compute_vol_target_size(
    vol_20: float,
    *,
    target_annual_vol: float = 0.15,
    max_positions: int = 8,
) -> float:
    """Per-stock position size (% of portfolio) using vol targeting.

    Sizes each position so its contribution to portfolio volatility
    equals roughly ``target_annual_vol / max_positions``.

    Parameters
    ----------
    vol_20 : float
        Stock's annualised 20-day realised volatility (e.g. 0.30 = 30%).
    target_annual_vol : float
        Target annualised portfolio volatility (default 0.15 = 15%).
    max_positions : int
        Number of positions to budget for (denominator *N*).

    Returns
    -------
    float
        Suggested weight as a percentage of equity (e.g. 6.25),
        or -1.0 if *vol_20* is non-positive / NaN.
    """
    # Coerce to numeric — config values may arrive as strings
    try:
        target_annual_vol = float(target_annual_vol)
        max_positions = int(max_positions)
    except (TypeError, ValueError):
        return -1.0
    if max_positions < 1:
        max_positions = 1
    if (
        vol_20 is None
        or not isinstance(vol_20, (int, float))
        or math.isnan(vol_20)
        or vol_20 <= 0
    ):
        return -1.0

    weight = target_annual_vol / (max_positions * vol_20)
    return round(weight * 100, 2)


def blend_position_sizes(
    kelly_pct: float,
    vol_target_pct: float,
    blend: float = 0.5,
) -> float:
    """Blend Half-Kelly and vol-target sizes.

    Parameters
    ----------
    kelly_pct : float
        Half-Kelly size as % of equity.
    vol_target_pct : float
        Vol-target size as % of equity, or -1.0 if unavailable.
    blend : float
        Weight on vol-target (0 = pure Kelly, 1 = pure vol-target).

    Returns
    -------
    float
        Blended size as % of equity.  If vol-target is unavailable
        (-1.0), returns pure Kelly unchanged.
    """
    # Coerce — config values may arrive as strings
    try:
        blend = float(blend)
    except (TypeError, ValueError):
        blend = 0.5
    blend = max(0.0, min(1.0, blend))
    if vol_target_pct < 0:
        return kelly_pct
    return round((1 - blend) * kelly_pct + blend * vol_target_pct, 2)


def generate_daily_signal(
    df: pd.DataFrame,
    ticker: str,
    strategy: Strategy,
    result: BacktestResult,
    config: dict,
    earnings_ctx: Optional[EarningsContext] = None,
) -> DailySignal:
    """Generate today's actionable signal for a stock.

    Parameters
    ----------
    earnings_ctx : EarningsContext, optional
        Earnings proximity/surprise data.  When provided, the
        confidence score is adjusted (up or down) based on
        earnings proximity and past surprise direction.
    """
    if earnings_ctx is None:
        earnings_ctx = _NEUTRAL_EARNINGS

    risk_config = RISK_PROFILES[config["risk_profile"]]

    # Generate signal on full data
    signals = strategy.generate_signals(df)
    latest_signal_val = (
        signals.iloc[-1] if len(signals) > 0 else 0
    )

    # Map signal
    long_only = config.get("long_only", True)
    if latest_signal_val > 0:
        signal = "BUY"
        signal_val = 1
    elif latest_signal_val < 0:
        signal = "EXIT" if long_only else "SELL/SHORT"
        signal_val = -1
    else:
        signal = "HOLD"
        signal_val = 0

    # ── Confidence assessment ─────────────────────────────────────────
    confidence_score = 0
    if result.sharpe_ratio > 1.0:
        confidence_score += 2
    elif result.sharpe_ratio > 0.5:
        confidence_score += 1
    if result.win_rate > 0.55:
        confidence_score += 1
    if result.profit_factor > 1.5:
        confidence_score += 1
    if result.total_trades >= 15:
        confidence_score += 1
    if (
        hasattr(result, "composite_score")
        and result.composite_score > 50
    ):
        confidence_score += 1

    # ── Earnings confidence adjustment ────────────────────────────────
    earnings_adj = compute_earnings_confidence_adj(earnings_ctx)
    confidence_score = max(0, confidence_score + earnings_adj)

    if confidence_score >= 4:
        confidence = "HIGH"
    elif confidence_score >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Force HOLD if confidence too low under risk profile
    if confidence == "LOW" and signal not in ("HOLD",):
        if result.sharpe_ratio < config["min_sharpe"]:
            signal = "HOLD"

    # ── Market context ────────────────────────────────────────────────
    latest = df.iloc[-1]

    if latest.get("SMA_50", 0) > latest.get("SMA_200", 0):
        trend = "BULLISH"
    elif latest.get("SMA_50", 0) < latest.get("SMA_200", 0):
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    vol_20 = latest.get("Volatility_20", 0)
    if vol_20 > 0.4:
        volatility = "HIGH"
    elif vol_20 > 0.2:
        volatility = "MEDIUM"
    else:
        volatility = "LOW"

    # ── Notes ─────────────────────────────────────────────────────────
    notes_parts: list[str] = []
    rsi_val = latest.get("RSI_14", 50)
    if rsi_val > 70:
        notes_parts.append("RSI overbought")
    elif rsi_val < 30:
        notes_parts.append("RSI oversold")

    if latest.get("Vol_Ratio", 1) > 1.5:
        notes_parts.append("High volume")
    if abs(latest.get("ZScore_20", 0)) > 2:
        notes_parts.append("Price at statistical extreme")

    adx_val = latest.get("ADX_14", 0)
    if adx_val > 30:
        notes_parts.append(f"Strong trend (ADX={adx_val:.0f})")

    # ── Earnings notes ────────────────────────────────────────────────
    if earnings_ctx.is_available:
        if earnings_ctx.days_to_earnings == 0:
            notes_parts.append(
                f"EARNINGS TODAY ({earnings_ctx.earnings_date})"
            )
        elif 0 < earnings_ctx.days_to_earnings <= 5:
            notes_parts.append(
                f"Earnings in {earnings_ctx.days_to_earnings}d "
                f"({earnings_ctx.earnings_date})"
            )
    if not math.isnan(earnings_ctx.last_surprise_pct):
        notes_parts.append(
            f"Last surprise: "
            f"{earnings_ctx.last_surprise_pct:+.1f}%"
        )
    if earnings_adj != 0:
        notes_parts.append(
            f"Earnings conf adj: {earnings_adj:+d}"
        )

    backtest_period = (
        f"{result.backtest_start} to {result.backtest_end} "
        f"({result.trading_days} days)"
    )

    now = datetime.now()
    current_price = round(float(latest["Close"]), 2)

    # ── Stop loss: volatility-adaptive ATR-based ──────────────────────
    atr_val = float(latest.get("ATR_14", 0) or 0)
    if atr_val > 0 and current_price > 0:
        stop_loss_pct = compute_atr_stop_loss_pct(
            atr_val, vol_20, current_price,
        )
    else:
        stop_loss_pct = round(
            min(max(abs(result.max_drawdown_pct) * 0.4, 2.0), 10.0),
            2,
        )

    if signal_val > 0:
        stop_loss_price = round(
            current_price * (1 - stop_loss_pct / 100), 2
        )
    elif signal_val < 0:
        stop_loss_price = round(
            current_price * (1 + stop_loss_pct / 100), 2
        )
    else:
        stop_loss_price = 0.0

    # ── Take profit: dynamic reward/risk ratio ────────────────────────
    # R:R scales with trend strength and confidence; delegates to the
    # shared helper so signals.py (live) and backtest.py (TB path)
    # can never drift.  See quant_analysis_bot/tp_logic.py.
    #
    # tp_mode options (default "current" preserves today's behavior):
    #   "current"         : SL × dynamic RR, no cap
    #   "capped"          : cap TP at tp_cap_multiplier × 1σ expected move
    #   "capped+strategy" : capped + mean-reversion family clamped at 1.5 RR
    tp_mode = config.get("tp_mode", "current")
    cap_multiplier = float(config.get("tp_cap_multiplier", 1.5))
    family = classify_strategy_family(strategy.name)
    rr_ratio = compute_rr_ratio(
        trend=trend,
        adx=float(adx_val),
        confidence_score=confidence_score,
        family=family,
        tp_mode=tp_mode,
    )

    raw_tp_pct = stop_loss_pct * rr_ratio
    if tp_mode in ("capped", "capped+strategy"):
        # Use the strategy's measured holding window from its own
        # backtest result; fall back to config when missing.
        holding = float(result.avg_holding_days) if result.avg_holding_days else 0.0
        if holding <= 0:
            holding = float(config.get("tp_cap_holding_days", 20.0))
        expected_max = compute_expected_max_move_pct(
            vol_20=vol_20,
            holding_days=holding,
        )
        take_profit_pct = round(
            apply_tp_cap(raw_tp_pct, expected_max, cap_multiplier, tp_mode),
            2,
        )
    else:
        take_profit_pct = round(raw_tp_pct, 2)
    if signal_val > 0:
        take_profit_price = round(
            current_price * (1 + take_profit_pct / 100), 2
        )
    elif signal_val < 0:
        take_profit_price = round(
            current_price * (1 - take_profit_pct / 100), 2
        )
    else:
        take_profit_price = 0.0

    # ── Position sizing: half-Kelly blended with vol-target ────────────
    win_rate_f = result.win_rate
    pf = max(result.profit_factor, 0.01)
    loss_rate = 1 - win_rate_f
    kelly_f = max((win_rate_f * pf - loss_rate) / pf, 0.0)
    half_kelly = kelly_f * 0.5
    max_size = {"HIGH": 10.0, "MEDIUM": 10.0, "LOW": 7.0}.get(
        confidence, 7.0
    )
    kelly_pct = round(min(half_kelly * 100, max_size), 2)

    # Vol-target component: inverse-volatility sizing
    vol_target_pct = compute_vol_target_size(
        vol_20,
        target_annual_vol=config.get("vol_target_annual", 0.15),
        max_positions=config.get("vol_target_max_positions", 8),
    )
    # Cap vol-target to same confidence-based ceiling
    if vol_target_pct > 0:
        vol_target_pct = round(min(vol_target_pct, max_size), 2)

    # Blend the two sizing methods
    vol_blend = config.get("vol_sizing_blend", 0.5)
    suggested_position_size_pct = blend_position_sizes(
        kelly_pct, vol_target_pct, blend=vol_blend,
    )
    # Final cap (blending can land between two capped values,
    # but never above max_size by construction)
    suggested_position_size_pct = min(
        suggested_position_size_pct, max_size
    )

    if signal == "HOLD":
        suggested_position_size_pct = 0.0

    # ── Meta-label sizing adjustment ──────────────────────────────────
    meta_label_prob = -1.0
    meta_label_size_mult = 1.0
    if (
        config.get("meta_label_enabled", False)
        and signal_val == 1  # only for BUY signals
    ):
        try:
            from quant_analysis_bot.meta_label import (
                compute_meta_kelly,
                load_meta_model,
                predict_meta_label,
            )

            trained = load_meta_model(
                base_dir=config.get("meta_label_model_dir", "models"),
                ticker=ticker,
            )
            if trained is not None:
                # Map strategy name to integer ID
                from quant_analysis_bot.strategies import ALL_STRATEGIES
                strat_ids = {s.name: i for i, s in enumerate(ALL_STRATEGIES)}
                strat_id = strat_ids.get(strategy.name, 0)

                meta_label_prob = predict_meta_label(
                    trained, df, len(df) - 1, strat_id,
                    sl_pct=stop_loss_pct, tp_pct=take_profit_pct,
                )

                if meta_label_prob >= 0:
                    final_kelly, meta_label_size_mult = compute_meta_kelly(
                        meta_prob=meta_label_prob,
                        base_kelly_f=half_kelly,
                        profit_factor=result.profit_factor,
                        n_training_trades=trained.n_training_trades,
                        is_calibrated=trained.is_calibrated,
                        min_training_trades=config.get(
                            "meta_label_min_training_trades", 50
                        ),
                    )
                    # Apply as multiplier on the already-blended size
                    # (preserves vol-target component)
                    suggested_position_size_pct = round(
                        min(
                            suggested_position_size_pct * meta_label_size_mult,
                            max_size,
                        ),
                        2,
                    )

                    # Gate: low-prob signals downgraded to HOLD
                    min_prob = config.get("meta_label_min_prob", 0.35)
                    if meta_label_prob < min_prob:
                        signal = "HOLD"
                        signal_val = 0
                        suggested_position_size_pct = 0.0
                        notes_parts.append(
                            f"Meta-label gated: P={meta_label_prob:.2f} "
                            f"< {min_prob}"
                        )
                    else:
                        notes_parts.append(
                            f"Meta P={meta_label_prob:.2f}, "
                            f"size×{meta_label_size_mult:.2f}"
                        )
        except ImportError:
            pass
        except Exception as e:
            log.debug("Meta-label sizing skipped: %s", e)

    if signal == "HOLD":
        suggested_position_size_pct = 0.0

    # ── Sizing notes ──────────────────────────────────────────────────
    if vol_target_pct >= 0 and signal != "HOLD":
        notes_parts.append(
            f"Size: Kelly {kelly_pct:.1f}% / "
            f"VolTgt {vol_target_pct:.1f}% / "
            f"Blend {suggested_position_size_pct:.1f}%"
        )

    # ── Signal expiry ─────────────────────────────────────────────────
    hold_days = max(int(result.avg_holding_days), 1)
    signal_expires = (now + timedelta(days=hold_days)).strftime(
        "%Y-%m-%d"
    )

    composite_score_val = (
        getattr(result, "composite_score", 0.0) or 0.0
    )

    return DailySignal(
        generated_at=now.isoformat(timespec="seconds"),
        date=now.strftime("%Y-%m-%d"),
        ticker=ticker,
        signal=signal,
        signal_raw=signal_val,
        strategy=strategy.name,
        confidence=confidence,
        confidence_score=confidence_score,
        composite_score=round(composite_score_val, 2),
        current_price=current_price,
        stop_loss_pct=stop_loss_pct,
        stop_loss_price=stop_loss_price,
        take_profit_pct=take_profit_pct,
        take_profit_price=take_profit_price,
        suggested_position_size_pct=suggested_position_size_pct,
        signal_expires=signal_expires,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        win_rate=round(result.win_rate * 100, 1),
        profit_factor=round(result.profit_factor, 3),
        annual_return_pct=result.annual_return_pct,
        annual_excess_pct=result.annual_excess_pct,
        max_drawdown_pct=result.max_drawdown_pct,
        avg_holding_days=round(result.avg_holding_days, 1),
        total_trades=result.total_trades,
        backtest_period=backtest_period,
        pbo=round(result.pbo, 3) if result.pbo >= 0 else -1.0,
        rsi=round(rsi_val, 1),
        vol_20=round(float(vol_20), 4),
        sma_50=round(float(latest.get("SMA_50", 0) or 0), 2),
        sma_200=round(float(latest.get("SMA_200", 0) or 0), 2),
        trend=trend,
        volatility=volatility,
        notes=(
            "; ".join(notes_parts)
            if notes_parts
            else "No special conditions"
        ),
        vol_target_size_pct=(
            vol_target_pct if vol_target_pct >= 0 else -1.0
        ),
        meta_label_prob=round(meta_label_prob, 4),
        meta_label_size_mult=round(meta_label_size_mult, 4),
        days_to_earnings=earnings_ctx.days_to_earnings,
        earnings_date=earnings_ctx.earnings_date,
        last_surprise_pct=(
            round(earnings_ctx.last_surprise_pct, 2)
            if not math.isnan(earnings_ctx.last_surprise_pct)
            else None
        ),
        earnings_confidence_adj=earnings_adj,
    )
