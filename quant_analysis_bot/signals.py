"""Daily signal generation with execution details."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from quant_analysis_bot.config import RISK_PROFILES
from quant_analysis_bot.models import BacktestResult, DailySignal
from quant_analysis_bot.strategies import Strategy


def generate_daily_signal(
    df: pd.DataFrame,
    ticker: str,
    strategy: Strategy,
    result: BacktestResult,
    config: dict,
) -> DailySignal:
    """Generate today's actionable signal for a stock."""
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

    backtest_period = (
        f"{result.backtest_start} to {result.backtest_end} "
        f"({result.trading_days} days)"
    )

    now = datetime.now()
    current_price = round(float(latest["Close"]), 2)

    # ── Stop loss: volatility-adaptive ATR-based ──────────────────────
    # The ATR multiplier adjusts with the volatility regime so that
    # high-vol stocks get wider stops (avoid whipsaw) and low-vol
    # stocks get tighter stops (capture gains faster).
    atr_val = float(latest.get("ATR_14", 0) or 0)
    if atr_val > 0 and current_price > 0:
        # Volatility regime adjustment
        # vol_20 > 0.4 → HIGH → widen stops (mult 2.0)
        # vol_20 0.2-0.4 → MEDIUM → standard (mult 1.5)
        # vol_20 < 0.2 → LOW → tighten stops (mult 1.2)
        if vol_20 > 0.4:
            atr_mult = 2.0
        elif vol_20 > 0.2:
            atr_mult = 1.5
        else:
            atr_mult = 1.2

        stop_loss_pct = round(
            min(
                max(
                    (atr_val * atr_mult / current_price) * 100,
                    1.5,
                ),
                12.0,
            ),
            2,
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
    # R:R scales with trend strength and confidence:
    #   Strong bullish + HIGH confidence → 3:1 (let winners run)
    #   Neutral or MEDIUM confidence → 2:1 (standard)
    #   Bearish or LOW confidence → 1.5:1 (take profits quicker)
    if trend == "BULLISH" and confidence_score >= 4:
        rr_ratio = 3.0
    elif trend == "BEARISH" or confidence_score <= 1:
        rr_ratio = 1.5
    else:
        rr_ratio = 2.0

    # ADX boost: very strong trend (>30) bumps ratio by 0.5
    if adx_val > 30 and trend == "BULLISH":
        rr_ratio = min(rr_ratio + 0.5, 3.5)

    take_profit_pct = round(stop_loss_pct * rr_ratio, 2)
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

    # ── Position sizing: half-Kelly, capped by confidence ─────────────
    win_rate_f = result.win_rate
    pf = max(result.profit_factor, 0.01)
    loss_rate = 1 - win_rate_f
    kelly_f = max((win_rate_f * pf - loss_rate) / pf, 0.0)
    half_kelly = kelly_f * 0.5
    max_size = {"HIGH": 10.0, "MEDIUM": 10.0, "LOW": 7.0}.get(
        confidence, 7.0
    )
    suggested_position_size_pct = round(
        min(half_kelly * 100, max_size), 2
    )
    if signal == "HOLD":
        suggested_position_size_pct = 0.0

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
    )
